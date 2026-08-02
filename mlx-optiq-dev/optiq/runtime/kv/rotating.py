"""Quantized RotatingKVCache.

mlx-lm's ``RotatingKVCache.to_quantized()`` raises::

    raise NotImplementedError("RotatingKVCache Quantization NYI")

This blocks KV-cache quantization on the 15+ model families that use a
sliding-window cache: Gemma 3 / 3n / 4 (text), Cohere Command R 2, OLMo 3,
EXAONE 4 / EXAONE MoE, Ministral 3, Recurrent Gemma, Baichuan M1, AFMoE,
MiMo v2 Flash, Step 3.5, GPT-OSS, and any Llama variant that uses SWA.

This module fills that gap with a plain affine-quantized rotating cache.
Storage matches mlx-lm's :class:`QuantizedKVCache` (a 3-tuple per K and V::

    (packed_uint32, scales, biases)

), but with the rotating-buffer trim / temporal-order logic preserved from
:class:`mlx_lm.models.cache.RotatingKVCache`. ``update_and_fetch`` returns
dequantized fp16/bf16 tensors so the standard SDPA path runs unmodified —
no custom Metal kernel required.

The PolarQuant approach from Google's TurboQuant paper (custom Lloyd-Max
codebook + bits-packed SDPA kernel) gives better compression at 2-3 bits
and is faster on Apple Silicon. It is **not** implemented here. We use
``mx.quantize`` / ``mx.dequantize`` for parity with the rest of OptiQ's
KV stack so per-layer mixed-precision configs land unmodified.
"""

from __future__ import annotations

import os as _os

import mlx.core as mx
from mlx.utils import tree_map, tree_reduce
from mlx_lm.models.cache import RotatingKVCache


def _is_quantized(v):
    return isinstance(v, (tuple, list))


class RotatingQuantizedKVCache(RotatingKVCache):
    """Affine-quantized RotatingKVCache.

    Subclasses :class:`RotatingKVCache` so make_mask / size / is_trimmable /
    trim / state-dict serialization inherit correctly. Overrides the
    storage-shape-sensitive methods (``_trim``, ``_temporal_order``,
    ``_update_concat``, ``_update_in_place``) to work over the 3-tuple
    ``(packed, scales, biases)`` form via ``tree_map``.
    """

    step = 256

    def __init__(
        self,
        max_size: int,
        keep: int = 0,
        group_size: int = 64,
        bits: int = 4,
    ):
        super().__init__(max_size=max_size, keep=keep)
        self.group_size = group_size
        self.bits = bits

    # ---- shape helpers ----

    def _seq_len(self) -> int:
        if self.keys is None:
            return 0
        if _is_quantized(self.keys):
            return self.keys[0].shape[2]
        return self.keys.shape[2]

    def _alloc_pair(self, B: int, n_kv_heads: int, T: int, dim: int, dtype):
        """Allocate empty (packed, scales, biases) tuple of length T."""
        el_per_int = 8 * mx.uint32.size // self.bits
        shape = (B, n_kv_heads, T)
        return (
            mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
            mx.zeros((*shape, dim // self.group_size), dtype=dtype),
            mx.zeros((*shape, dim // self.group_size), dtype=dtype),
        )

    # ---- rotating-buffer mechanics ----

    def _trim(self, trim_size: int, v, append=None):
        if not _is_quantized(v):
            return super()._trim(trim_size, v, append)
        if trim_size > 0:
            parts = [
                tree_map(lambda x: x[..., : self.keep, :], v),
                tree_map(lambda x: x[..., trim_size + self.keep :, :], v),
            ]
        else:
            parts = [v]
        if append is not None:
            parts.append(append)
        if len(parts) == 1:
            return parts[0]
        return tree_map(lambda *arrs: mx.concatenate(arrs, axis=2), *parts)

    def _temporal_order(self, v):
        if not _is_quantized(v):
            return super()._temporal_order(v)
        seq_len = v[0].shape[2]
        if self._idx == seq_len:
            return v
        if self._idx < self.offset:
            return tree_map(
                lambda x: mx.concatenate(
                    [
                        x[..., : self.keep, :],
                        x[..., self._idx :, :],
                        x[..., self.keep : self._idx, :],
                    ],
                    axis=2,
                ),
                v,
            )
        return tree_map(lambda x: x[..., : self._idx, :], v)

    # ---- update_and_fetch entry points ----

    def _update_concat(self, keys, values):
        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        q_values = mx.quantize(values, group_size=self.group_size, bits=self.bits)

        if self.keys is None:
            self.keys = q_keys
            self.values = q_values
        else:
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)
            self._idx = self._seq_len()
            trim_size = self._idx - self.max_size + 1
            self.keys = self._trim(trim_size, self.keys, q_keys)
            self.values = self._trim(trim_size, self.values, q_values)

        self.offset += keys.shape[2]
        self._idx = self._seq_len()
        return self._active_slices()

    def _update_in_place(self, keys, values):
        B, n_kv_heads, S, k_head_dim = keys.shape
        v_head_dim = values.shape[3]
        prev = self.offset

        if self.keys is None or (
            prev >= self._seq_len() and self._seq_len() < self.max_size
        ):
            new_size = min(self.step, self.max_size - prev)
            new_k = self._alloc_pair(B, n_kv_heads, new_size, k_head_dim, keys.dtype)
            new_v = self._alloc_pair(B, n_kv_heads, new_size, v_head_dim, values.dtype)
            if self.keys is not None:
                self.keys = tree_map(
                    lambda a, b: mx.concatenate([a, b], axis=2), self.keys, new_k
                )
                self.values = tree_map(
                    lambda a, b: mx.concatenate([a, b], axis=2), self.values, new_v
                )
            else:
                self.keys = new_k
                self.values = new_v
            self._idx = prev

        trim_size = self._seq_len() - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        if self._idx == self.max_size:
            self._idx = self.keep

        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        q_values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        # Assign tuple components element-wise (matches QuantizedKVCache)
        for i in range(len(self.keys)):
            self.keys[i][..., self._idx : self._idx + S, :] = q_keys[i]
            self.values[i][..., self._idx : self._idx + S, :] = q_values[i]

        self.offset += S
        self._idx += S
        return self._active_slices()

    def _active_slices(self):
        """Return the active window of self.keys / self.values as quantized tuples.

        mlx-lm's ``scaled_dot_product_attention`` dispatches on
        ``hasattr(cache, "bits")`` and expects K/V to be the standard
        ``(packed, scales, biases)`` tuples — same convention as
        :class:`QuantizedKVCache.update_and_fetch`. Return that form so the
        quantized SDPA path runs without dequantizing.

        Also register the returned tuples in the producer registry so the
        Gemma 4 KV-sharing path can recover bits / group_size when the
        downstream layer's ``cache`` is None.
        """
        if self.offset < self.max_size:
            active_k = tree_map(lambda x: x[..., : self.offset, :], self.keys)
            active_v = tree_map(lambda x: x[..., : self.offset, :], self.values)
        else:
            active_k = self.keys
            active_v = self.values
        _register_producer(self, active_k, active_v)
        return active_k, active_v

    # ---- state serialization ----

    @property
    def state(self):
        if self.keys is None:
            return [], []
        seq_len = self._seq_len()
        if self.offset < seq_len:
            return (
                tree_map(lambda x: x[..., : self.offset, :], self.keys),
                tree_map(lambda x: x[..., : self.offset, :], self.values),
            )
        return self.keys, self.values

    @state.setter
    def state(self, v):
        if v is not None and v:
            self.keys, self.values = v
        else:
            self.keys = None
            self.values = None

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.keep,
                    self.max_size,
                    self.offset,
                    self._idx,
                    self.group_size,
                    self.bits,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        (
            self.keep,
            self.max_size,
            self.offset,
            self._idx,
            self.group_size,
            self.bits,
        ) = (int(x) for x in v)

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        return self


def _replay_into_quantized(rkv: RotatingKVCache, group_size: int, bits: int):
    """Quantize the current state of ``rkv`` and produce a new rotating quantized cache."""
    new = RotatingQuantizedKVCache(
        max_size=rkv.max_size,
        keep=rkv.keep,
        group_size=group_size,
        bits=bits,
    )
    if rkv.keys is not None:
        new.keys = mx.quantize(rkv.keys, group_size=group_size, bits=bits)
        new.values = mx.quantize(rkv.values, group_size=group_size, bits=bits)
    new.offset = rkv.offset
    new._idx = rkv._idx
    return new


_patched = False
_sdpa_patched = False


def _patch_sdpa_for_kv_sharing() -> None:
    """Make ``mlx_lm.models.base.scaled_dot_product_attention`` quantization-
    aware when K/V are tuples even though ``cache=None``.

    Gemma 4 (and other KV-sharing models like recent Llama variants) pass
    K/V from one layer's cache into the next layer's attention call with
    ``cache=None`` for the receiver. Upstream's dispatch keys on
    ``hasattr(cache, "bits")`` which is False when ``cache is None``, so it
    routes the call through the fp16 fast SDPA path — which can't accept
    tuple K/V. Fix: also detect tuple K/V and route to the quantized SDPA
    path with the bits / group_size carried on the tuple's origin cache.
    """
    global _sdpa_patched
    if _sdpa_patched:
        return

    from mlx_lm.models import base as base_mod

    _ORIGINAL_SDPA = base_mod.scaled_dot_product_attention

    def _patched_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        # If K/V are tuples we must use the quantized path, even when
        # cache is None (KV-sharing scenario). Read group_size/bits from
        # the cache when present, otherwise infer from the tuple shape.
        if isinstance(keys, (tuple, list)) and not hasattr(cache, "bits"):
            # Gemma-4 shares K/V across layers, so a consumer layer gets the
            # producer's quantized tuple with no cache of its own to read
            # bits/group_size from.
            #
            # The registry (keyed on id(tuple)) is the fast path, but id() is
            # not a durable handle: it changes when the tuple is rebuilt and is
            # recycled after GC. The old fallback then guessed bits=4, which is
            # invisibly correct for a uniform-4-bit cache and silently WRONG for
            # a mixed-precision one -- an 8-bit layer decoded as 4-bit unpacks to
            # the wrong width and blows up as a broadcast mismatch deep in
            # attention. That is the mixed-KV crash on Gemma-4 at long context.
            #
            # bits and group_size are fully determined by the tensors, so derive
            # them instead of guessing. tuple layout: (packed_uint32, scales, biases)
            packed, scales, _ = keys
            shim = _find_producer_cache(keys)
            if shim is not None:
                bits, group_size = shim.bits, shim.group_size
            else:
                bits, group_size = _derive_quant_params(packed, scales,
                                                        queries.shape[-1])
            _kv_debug("quantized", queries, keys, mask, cache,
                      bits=bits, group_size=group_size)
            return base_mod.quantized_scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=scale,
                mask=mask,
                group_size=group_size,
                bits=bits,
            )
        _kv_debug("fallback", queries, keys, mask, cache)
        return _ORIGINAL_SDPA(queries, keys, values, cache, scale, mask, sinks)

    base_mod.scaled_dot_product_attention = _patched_sdpa
    # Re-import in any model module that captured the old function by name.
    for mod_name in list(__import__("sys").modules):
        if mod_name.startswith("mlx_lm.models.") and mod_name != "mlx_lm.models.base":
            mod = __import__("sys").modules.get(mod_name)
            if mod is None:
                continue
            if getattr(mod, "scaled_dot_product_attention", None) is _ORIGINAL_SDPA:
                mod.scaled_dot_product_attention = _patched_sdpa

    _sdpa_patched = True


_producer_registry: "dict[int, object]" = {}


def _register_producer(cache, keys_tuple, values_tuple) -> None:
    """Remember which cache produced these tuples so the shared-KV SDPA
    dispatch can recover the bits / group_size if needed."""
    _producer_registry[id(keys_tuple)] = cache
    _producer_registry[id(values_tuple)] = cache
    # Keep the registry small: drop oldest entries beyond 256.
    if len(_producer_registry) > 256:
        for k in list(_producer_registry.keys())[: -256]:
            del _producer_registry[k]


def _kv_debug(where, queries, keys, mask, cache, bits=None, group_size=None):
    """Internal diagnostic (``OPTIQ_KV_DEBUG=1``): dump cache state on a bad mask.

    Deliberately NOT in the public docs or changelog. It dumps raw cache
    internals (``_idx``, ``max_size``, derived bits) that only mean anything to
    someone working on this file, and documenting it would freeze the name and
    output format into a contract. Undocumented, it stays free to change.

    Kept rather than deleted after the bug it was written for closed: it
    localized four separate KV failures that reading the source had not, and it
    costs one env lookup per attention call when off.

    Fires when the mask's key axis does not match the key length -- the actual
    invariant attention needs. An earlier version of this probe only fired on a
    *zero*-width mask, and so printed nothing for the real failure, which was
    ``mask (1,1,1,20)`` against ``keys (1,16,1,10829)``. Match on
    broadcastability, not on one shape that happened to show up first.

    Both call sites are instrumented: the crash surfaced in the non-quantized
    fallback, which the earlier probe placement covered but its trigger did not.
    """
    if not _os.environ.get("OPTIQ_KV_DEBUG"):
        return
    try:
        if mask is None or not hasattr(mask, "shape"):
            return
        mshape = tuple(mask.shape)
        kshape = (tuple(keys[0].shape) if isinstance(keys, (tuple, list))
                  else tuple(keys.shape))
        n_keys = kshape[2] if len(kshape) > 2 else None
        m_keys = mshape[-1] if mshape else None
        if m_keys in (None, 1) or n_keys in (None, 1) or m_keys == n_keys:
            return                       # broadcasts fine
        print(f"[KVDEBUG] BAD MASK @{where} mask={mshape} q={tuple(queries.shape)} "
              f"k={kshape} mask_keys={m_keys} n_keys={n_keys} "
              f"cache={type(cache).__name__} "
              f"offset={getattr(cache, 'offset', None)} "
              f"_idx={getattr(cache, '_idx', None)} "
              f"max_size={getattr(cache, 'max_size', None)} "
              f"keys_len={getattr(cache, 'keys', None) is not None} "
              f"bits={bits} group_size={group_size} "
              f"shared_kv={isinstance(keys, (tuple, list))}", flush=True)
    except Exception as e:               # never let diagnostics break generation
        print(f"[KVDEBUG] probe failed: {e}", flush=True)


def _find_producer_cache(keys_tuple):
    return _producer_registry.get(id(keys_tuple))


def _derive_quant_params(packed, scales, head_dim: int) -> tuple[int, int]:
    """Recover ``(bits, group_size)`` from a quantized K/V tuple.

    Both are implied by the tensors, so nothing has to be remembered:

      * ``scales`` carries one entry per group along the last axis, so
        ``group_size = head_dim / scales.shape[-1]``.
      * ``packed`` holds ``head_dim`` values at ``bits`` each, packed into
        uint32, so ``bits = packed.shape[-1] * 32 / head_dim``.

    Verified exact for bits ∈ {4, 8} × group_size ∈ {32, 64} × head_dim ∈
    {128, 256}. Falls back to the previous 4/64 assumption only if the shapes
    are degenerate, which should not happen for a real cache.
    """
    try:
        n_groups = scales.shape[-1]
        group_size = head_dim // n_groups if n_groups else 64
        bits = round(packed.shape[-1] * 32 / head_dim) if head_dim else 4
        if bits not in (2, 3, 4, 5, 6, 8) or group_size <= 0:
            return 4, 64
        return bits, group_size
    except (AttributeError, IndexError, ZeroDivisionError):
        return 4, 64


def _patch_quantized_kvcache_register() -> None:
    """Wrap :meth:`QuantizedKVCache.update_and_fetch` so the tuples it
    returns are registered in the producer registry. This is what lets a
    Gemma-4 KV-shared layer recover bits / group_size when its own cache
    is None but the upstream full-attention layer's K/V tuples are passed
    in via ``shared_kv``.

    Idempotent.
    """
    from mlx_lm.models.cache import QuantizedKVCache

    if getattr(QuantizedKVCache.update_and_fetch, "_optiq_wrapped", False):
        return

    _orig = QuantizedKVCache.update_and_fetch

    def _wrapped(self, keys, values):
        k_out, v_out = _orig(self, keys, values)
        _register_producer(self, k_out, v_out)
        return k_out, v_out

    _wrapped._optiq_wrapped = True  # type: ignore[attr-defined]
    QuantizedKVCache.update_and_fetch = _wrapped


def patch_rotating_to_quantized() -> None:
    """Install :class:`RotatingQuantizedKVCache` as the return of
    :meth:`RotatingKVCache.to_quantized`, patch the SDPA dispatch so
    Gemma-4-style KV sharing handles tuple K/V correctly, and wrap
    :meth:`QuantizedKVCache.update_and_fetch` to feed the producer
    registry.

    Idempotent.
    """
    global _patched
    if _patched:
        return

    def _to_quantized(self, group_size: int = 64, bits: int = 4):
        if isinstance(self, RotatingQuantizedKVCache):
            return self
        return _replay_into_quantized(self, group_size, bits)

    RotatingKVCache.to_quantized = _to_quantized
    _patch_sdpa_for_kv_sharing()
    _patch_quantized_kvcache_register()
    _patched = True


def quantize_cache_layer(c, bits: int, group_size: int):
    """Quantize ONE layer's KV cache, picking the right class for its shape.

    A ``RotatingKVCache`` (sliding window) must become a
    ``RotatingQuantizedKVCache`` that carries ``max_size`` / ``keep`` / ``_idx``
    across. A plain ``QuantizedKVCache`` cannot represent a sliding window: its
    ``keys`` is a ring buffer that is NOT in temporal order once it wraps, and
    ``offset`` is total tokens seen, not the number of valid entries. Copy a
    wrapped ring into a plain cache and set ``offset = c.offset`` and you get a
    cache claiming 5000 entries over a 512-slot buffer.

    ``optiq/serve.py`` (the --kv-config path) got this right. ``streaming_kv_quant``
    (the --kv-bits path) built a plain ``QuantizedKVCache`` unconditionally. Both
    patch the SAME symbol -- ``mlx_lm.generate.maybe_quantize_kv_cache`` -- so
    which one you got depended on the flag. On Gemma-4, 4 of every 5 layers are
    sliding-window, so `optiq serve --kv-bits 4` returned correct short answers
    and silently corrupt long-context ones. Upstream mlx-lm REFUSES this
    (``RotatingKVCache.to_quantized`` raises NotImplementedError); the blind copy
    turned a loud refusal into a wrong answer.

    Returns the new cache, or ``c`` unchanged when it is already quantized.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import QuantizedKVCache, RotatingKVCache

    if isinstance(c, (QuantizedKVCache, RotatingQuantizedKVCache)):
        return c

    if isinstance(c, RotatingKVCache):
        new_c = RotatingQuantizedKVCache(
            max_size=c.max_size, keep=c.keep, group_size=group_size, bits=bits,
        )
        new_c._idx = c._idx
    else:
        new_c = QuantizedKVCache(group_size=group_size, bits=bits)
    new_c.offset = c.offset

    new_c.keys = mx.quantize(c.keys, group_size=group_size, bits=bits)
    new_c.values = mx.quantize(c.values, group_size=group_size, bits=bits)
    mx.eval(new_c.keys, new_c.values)
    c.keys = c.values = None
    return new_c


def empty_quantized_like(c, bits: int, group_size: int):
    """An EMPTY quantized cache with the same geometry as ``c``.

    The sensitivity sweep feeds fresh caches rather than converting populated
    ones, but it needs the same rule: a sliding-window layer must get a
    ``RotatingQuantizedKVCache`` carrying ``max_size`` and ``keep``, or the probe
    measures a cache the model cannot actually use.
    """
    from mlx_lm.models.cache import QuantizedKVCache, RotatingKVCache

    if isinstance(c, RotatingKVCache):
        return RotatingQuantizedKVCache(
            max_size=c.max_size, keep=c.keep, group_size=group_size, bits=bits,
        )
    return QuantizedKVCache(group_size=group_size, bits=bits)
