"""Quantized KV cache for mlx-lm's **batch** generation path.

``optiq serve`` used to have to choose: batching, or KV quantization. Not both.
mlx-lm quantizes the cache only on its sequential path, and neither batch cache
supports it -- ``BatchKVCache`` has no ``to_quantized`` at all, and
``BatchRotatingKVCache.to_quantized`` raises ``NotImplementedError``. So a
KV-quant flag had to force the sequential path, giving up cross-request
batching.

:class:`BatchQuantizedKVCache` closes that for full-attention layers. It is
``QuantizedKVCache``'s storage (a ``(packed_uint32, scales, biases)`` triple per
tensor) carrying ``BatchKVCache``'s batching semantics (per-sequence
``left_padding`` and ``offset``, plus ``merge`` / ``extend`` / ``filter`` /
``extract`` / ``prepare`` / ``finalize``). Every batch-shaped operation is a
slice or concatenate along the batch or token axis, and those apply to the three
component arrays independently -- which is what makes this mechanical rather
than a rewrite of the quantization itself.

**Sliding-window layers are deliberately not covered.** Rotating a quantized
buffer with wraparound is a genuinely harder problem (it is what
``RotatingQuantizedKVCache`` solves on the sequential path), and the payoff is
much smaller: a sliding layer's KV is capped at ``window_size`` and does not
grow with context, while a full-attention layer's does. On Gemma-4 at 10k
tokens the 5 full-attention layers hold ~10k entries each and the 25 sliding
layers hold 1024 each, so the growth this eliminates is the growth that matters.
``install_batch_kv_quant`` reports exactly which layers it converted and which
it left alone, rather than implying full coverage.
"""

from __future__ import annotations

from typing import List, Optional

import mlx.core as mx
from mlx.utils import tree_map, tree_reduce
from mlx_lm.models.cache import (
    BatchKVCache,
    QuantizedKVCache,
    create_causal_mask,
)


def _quant_triple(x, group_size: int, bits: int):
    return mx.quantize(x, group_size=group_size, bits=bits)


class BatchQuantizedKVCache(BatchKVCache):
    """A ``BatchKVCache`` whose keys/values are stored affine-quantized.

    ``keys`` and ``values`` are 3-tuples ``(packed, scales, biases)`` rather
    than single arrays, matching ``QuantizedKVCache`` so the same attention
    kernel path (``quantized_scaled_dot_product_attention``) applies.
    """

    step = 256

    def __init__(self, left_padding: List[int], group_size: int = 64, bits: int = 8):
        super().__init__(left_padding)
        self.group_size = group_size
        self.bits = bits

    # ─── storage ────────────────────────────────────────────────────────────

    def _alloc(self, B, n_kv_heads, n_steps, dim, dtype):
        el_per_int = 8 * mx.uint32.size // self.bits
        shape = (B, n_kv_heads, n_steps)
        return (
            mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
            mx.zeros((*shape, dim // self.group_size), dtype=dtype),
            mx.zeros((*shape, dim // self.group_size), dtype=dtype),
        )

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self._idx

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            n_steps = (self.step + num_steps - 1) // self.step * self.step
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values))

                def expand(x):
                    pad = mx.zeros((*x.shape[:2], n_steps, x.shape[-1]), dtype=x.dtype)
                    return mx.concatenate([x, pad], axis=-2)

                self.keys, self.values = tree_map(expand, (self.keys, self.values))
            else:
                self.keys = self._alloc(B, n_kv_heads, n_steps, k_head_dim, keys.dtype)
                self.values = self._alloc(B, n_kv_heads, n_steps, v_head_dim,
                                          values.dtype)

        self.offset = self.offset + num_steps
        self._idx += num_steps

        qk = _quant_triple(keys, self.group_size, self.bits)
        qv = _quant_triple(values, self.group_size, self.bits)
        for i in range(3):
            self.keys[i][..., prev:self._idx, :] = qk[i]
            self.values[i][..., prev:self._idx, :] = qv[i]

        # Evaluate the WHOLE cache state, bookkeeping included. An earlier
        # version evaluated only keys/values and then called mx.clear_cache(),
        # which left `offset` and `left_padding` as unevaluated lazy arrays
        # while the buffer pool was released underneath them. They came back
        # corrupted -- offset read as 1065303936, which is 0x3F800000, the
        # float32 bit pattern for 1.0 -- and since `left_padding` feeds
        # make_mask, the model built a zero-width mask and attention died on a
        # broadcast mismatch several layers later.
        #
        # It was also unnecessary: measured on gemma-4-26B, quantized KV peaks
        # BELOW fp16 on both the sequential path (0.035 vs 0.038 GB) and the
        # batch path (0.061 vs 0.063 GB). There was no co-residency spike here
        # to fix. Keep the eval to bound the lazy graph, drop the clear_cache.
        if num_steps > 1:
            mx.eval(self.keys, self.values, self.offset, self.left_padding)

        return tree_map(lambda x: x[..., :self._idx, :], (self.keys, self.values))

    # ─── batching ───────────────────────────────────────────────────────────

    def finalize(self):
        """Right-padding is rolled away once a prompt is done prefilling.

        ``dynamic_roll`` along the token axis is per-token, so rolling each of
        the three component arrays by the same amount keeps a token's packed
        weights with its own scale and bias.
        """
        if self._right_padding is None:
            return
        from mlx_lm.models.cache import dynamic_roll
        padding = self._right_padding
        self.keys = tree_map(
            lambda x: dynamic_roll(x, padding[:, None], axis=2), self.keys)
        self.values = tree_map(
            lambda x: dynamic_roll(x, padding[:, None], axis=2), self.values)
        self.offset -= padding
        self.left_padding += padding
        self._right_padding = None

    def filter(self, batch_indices):
        if self.keys is not None:
            self.keys = tree_map(lambda x: x[batch_indices], self.keys)
            self.values = tree_map(lambda x: x[batch_indices], self.values)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = tree_map(lambda x: x[..., min_left_pad:, :], self.keys)
                self.values = tree_map(lambda x: x[..., min_left_pad:, :], self.values)
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other):
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        ref = self if self.keys is not None else other
        H = ref.keys[0].shape[1]
        max_size = max(
            (c.keys[0].shape[2] if c.keys is not None else 0) for c in (self, other))

        def pad(c):
            if c.keys is None:
                Bc = c.offset.shape[0]
                empty = tuple(
                    mx.zeros((Bc, H, 0, x.shape[-1]), dtype=x.dtype) for x in ref.keys)
                emptyv = tuple(
                    mx.zeros((Bc, H, 0, x.shape[-1]), dtype=x.dtype) for x in ref.values)
                k, v = empty, emptyv
            else:
                k, v = c.keys, c.values
            left = max_idx - c._idx
            right = max_size - k[0].shape[2] - left
            if right < 0:
                k = tree_map(lambda x: x[..., :right, :], k)
                v = tree_map(lambda x: x[..., :right, :], v)
                right = 0
            if left or right:
                pads = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = tree_map(lambda x: mx.pad(x, pads), k)
                v = tree_map(lambda x: mx.pad(x, pads), v)
            return k, v, c.offset, c.left_padding + left

        a, b = pad(self), pad(other)
        self.keys = tuple(mx.concatenate([x, y]) for x, y in zip(a[0], b[0]))
        self.values = tuple(mx.concatenate([x, y]) for x, y in zip(a[1], b[1]))
        self.offset = mx.concatenate([a[2], b[2]])
        self.left_padding = mx.concatenate([a[3], b[3]])
        self._idx = max_idx

    def extract(self, idx):
        """Pull one sequence out as a *mergeable* sequential quantized cache.

        Returning a plain ``QuantizedKVCache`` here is a round-trip bug, not a
        cosmetic one. A sequence that leaves a batch keeps its cache, and when
        the next request reuses that prefix -- which is exactly what prompt
        caching across agent turns does -- mlx-lm calls ``_merge_caches``, which
        rejects any cache without ``merge``:

            ValueError: <class '...QuantizedKVCache'> does not yet support
            batching with history

        A single-shot request never hits this, so it looks fine right up until
        a multi-turn session.
        """
        cache = MergeableQuantizedKVCache(group_size=self.group_size,
                                          bits=self.bits)
        padding = self.left_padding[idx].item()
        cache.keys = tuple(
            mx.contiguous(x[idx:idx + 1, :, padding:self._idx]) for x in self.keys)
        cache.values = tuple(
            mx.contiguous(x[idx:idx + 1, :, padding:self._idx]) for x in self.values)
        cache.offset = cache.keys[0].shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        """Batch up per-request caches. Mirrors BatchKVCache.merge on triples."""
        lengths = [c.size() for c in caches]
        max_length = max(lengths)
        first = next((c for c in caches if getattr(c, "keys", None) is not None), None)
        group_size = getattr(first, "group_size", 64) if first else 64
        bits = getattr(first, "bits", 8) if first else 8

        if max_length == 0:
            return cls([0] * len(caches), group_size=group_size, bits=bits)

        padding = [max_length - lg for lg in lengths]
        B = len(caches)
        H = first.keys[0].shape[1]

        def zeros_like_component(x):
            return mx.zeros((B, H, max_length, x.shape[-1]), dtype=x.dtype)

        keys = tuple(zeros_like_component(x) for x in first.keys)
        values = tuple(zeros_like_component(x) for x in first.values)
        for i, (p, c) in enumerate(zip(padding, caches)):
            if getattr(c, "keys", None) is None:
                continue
            n = c.size()
            for j in range(3):
                keys[j][i:i + 1, :, p:p + n] = c.keys[j][..., :n, :]
                values[j][i:i + 1, :, p:p + n] = c.values[j][..., :n, :]

        cache = cls(padding, group_size=group_size, bits=bits)
        cache.keys, cache.values = keys, values
        cache.offset = cache.offset + max_length
        cache._idx = max_length
        return cache

    # ─── introspection ──────────────────────────────────────────────────────

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs)

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    def is_trimmable(self):
        return True

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)

    @property
    def state(self):
        k, v = self.keys, self.values
        if k is not None and self._idx < k[0].shape[2]:
            k = tree_map(lambda x: x[..., :self._idx, :], k)
            v = tree_map(lambda x: x[..., :self._idx, :], v)
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v
        self._idx = self.keys[0].shape[2] if self.keys is not None else 0

    @property
    def meta_state(self):
        return tuple(map(str, (self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v):
        self.group_size, self.bits = map(int, v)


def quantize_batch_cache_layer(cache, *, bits: int, group_size: int):
    """Convert a populated ``BatchKVCache`` into a ``BatchQuantizedKVCache``.

    Returns the cache unchanged when it is empty or already quantized, or when
    it is a kind this does not cover (sliding-window), so a caller can treat
    identity as "left alone" rather than needing to pre-classify.
    """
    if isinstance(cache, BatchQuantizedKVCache):
        return cache
    if type(cache) is not BatchKVCache or cache.keys is None:
        return cache

    out = BatchQuantizedKVCache(
        [0], group_size=group_size, bits=bits)
    n = cache._idx
    out.keys = _quant_triple(cache.keys[..., :n, :], group_size, bits)
    out.values = _quant_triple(cache.values[..., :n, :], group_size, bits)
    out.offset = cache.offset
    out.left_padding = cache.left_padding
    out._idx = n
    out._right_padding = cache._right_padding
    return out


class MergeableQuantizedKVCache(QuantizedKVCache):
    """A ``QuantizedKVCache`` that can join a batch.

    mlx-lm decides a model is batchable with
    ``all(hasattr(c, "merge") for c in make_prompt_cache(model))``, and its
    ``QuantizedKVCache`` has no ``merge`` -- which is the structural reason
    quantized KV and batching were mutually exclusive upstream. It also
    inherits ``size()`` from ``_BaseCache``, which returns 0, so a merge would
    silently produce an empty batch even if one were attempted.

    Both gaps are one line each.
    """

    def size(self):
        return self.offset

    @classmethod
    def merge(cls, caches):
        return BatchQuantizedKVCache.merge(caches)

    # Upstream's QuantizedKVCache assumes it is never empty, because upstream
    # only ever builds one via ``to_quantized()`` from a populated cache. We
    # build them empty at cache-creation time, so ``keys`` is None until the
    # first write -- and mlx-lm's server reads ``prompt_cache_nbytes`` over
    # freshly created, not-yet-used caches on every scheduler pass. Inheriting
    # these unguarded killed the generation thread on the first request:
    # AttributeError deep inside a tree_reduce, after which the server accepted
    # requests and answered none, and a Metal OOM surfaced later from __del__
    # running against a dead stream. The OOM was the visible symptom and
    # entirely a red herring.

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)

    @property
    def state(self):
        if self.keys is None:
            return self.keys, self.values
        if self.offset == self.keys[0].shape[2]:
            return self.keys, self.values
        return tree_map(lambda x: x[..., :self.offset, :], (self.keys, self.values))

    @state.setter
    def state(self, v):
        self.keys, self.values = v


def install_batch_kv_quant(bit_map: Optional[dict] = None, *,
                           default: Optional[tuple] = None) -> bool:
    """Let the batch path quantize KV, instead of disabling batching for it.

    ``BatchGenerator`` builds one sequential cache per request
    (``_make_new_cache``) and merges those into batch caches when a batch
    forms. It has no quantization hook of its own and never calls
    ``maybe_quantize_kv_cache``, so rather than converting after the fact this
    swaps in cache classes that quantize as they are written and know how to
    merge.

    ``bit_map`` maps layer index to ``(bits, group_size)`` for the
    mixed-precision path. ``default`` supplies one ``(bits, group_size)`` for
    every eligible layer, which is the uniform ``--kv-bits`` case where the
    layer count is not known until the model loads. Layers this does not cover
    -- sliding-window, and the recurrent state of hybrid models -- are left at
    full precision and counted as skipped, never reported as converted.
    Returns False if the upstream hook point is missing, so the caller can fall
    back to forcing the sequential path rather than assume this worked.
    """
    import logging

    import mlx_lm.server as server_mod
    from mlx_lm.models.cache import KVCache

    if bit_map is None and default is None:
        return False
    BG = getattr(server_mod, "BatchGenerator", None)
    if BG is None or not hasattr(BG, "_make_new_cache"):
        return False
    if getattr(BG, "_optiq_batch_kv_installed", False):
        return True

    original = BG._make_new_cache
    reported: list = []

    def _make_new_cache(self):
        caches = original(self)
        converted, skipped = [], []
        for idx, c in enumerate(caches):
            spec = (bit_map or {}).get(idx, default)
            if spec is None or type(c) is not KVCache:
                skipped.append(idx)
                continue
            bits, group_size = spec
            caches[idx] = MergeableQuantizedKVCache(group_size=group_size, bits=bits)
            converted.append(idx)
        if converted and not reported:
            reported.append(True)
            logging.info(
                "[optiq.serve] batch KV cache quantized: %d/%d configured layers; "
                "%d left at full precision (sliding-window and recurrent layers "
                "are not covered — their KV is bounded and does not grow with "
                "context)", len(converted),
                len(bit_map) if bit_map else len(converted), len(skipped))
        return caches

    BG._make_new_cache = _make_new_cache
    BG._optiq_batch_kv_installed = True
    return True
