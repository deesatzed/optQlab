"""KV-cache viewer for Gemma-4 speculative decoding.

The Gemma-4 drafter is Q-only and pulls its K/V from two specific
target-model layers: the last ``sliding_attention`` donor and the last
``full_attention`` donor. This module locates those layers and exposes
their K/V tensors in temporal (chronological) order so the drafter's
RoPE positions line up with what was actually cached.

The mlx-lm gemma4 architecture stores K/V only for donor layers
(``has_kv = layer_idx < num_hidden_layers - num_kv_shared_layers``), so
the cache list length is the donor count, not the total layer count.

Two layer-type-specific quirks:

- ``RotatingKVCache`` (used on sliding layers) stores K/V in a ring
  buffer; ``_temporal_order`` reshuffles to chronological order.
- ``KVCache`` (used on full layers) stores in append order; ``.keys`` /
  ``.values`` are already chronological.
"""
from __future__ import annotations

import mlx.core as mx


def find_donor_layers(target) -> tuple[int, int]:
    """Return ``(last_sliding_donor_idx, last_full_donor_idx)`` in cache
    coordinates. Cache indices != logical layer indices because shared
    layers don't get cache entries.
    """
    lm = getattr(target, "language_model", target)
    model = lm.model if hasattr(lm, "model") else lm
    # Walk only donor layers (the ones in the cache list).
    last_sliding = last_full = -1
    cache_idx = 0
    for i, layer in enumerate(model.layers):
        has_kv = getattr(layer.self_attn, "has_kv", True)
        if not has_kv:
            continue
        lt = getattr(layer, "layer_type", "full_attention")
        if lt == "sliding_attention":
            last_sliding = cache_idx
        elif lt == "full_attention":
            last_full = cache_idx
        cache_idx += 1
    if last_sliding < 0 or last_full < 0:
        raise RuntimeError(
            f"target model has no donor layer of one of the required types "
            f"(sliding={last_sliding}, full={last_full})"
        )
    return last_sliding, last_full


def extract_typed_kv(target, caches) -> dict[str, tuple[mx.array, mx.array]]:
    """Pull (K, V) from the target's last sliding-donor and last full-donor
    caches and return them keyed by ``layer_type``.

    Tensors are returned in **chronological** order. For
    ``RotatingKVCache`` this calls ``_temporal_order``; for plain
    ``KVCache`` the order is already chronological.
    """
    last_sliding, last_full = find_donor_layers(target)
    sliding_cache = caches[last_sliding]
    full_cache = caches[last_full]
    return {
        "sliding_attention": _read_cache_temporal(sliding_cache),
        "full_attention": _read_cache_temporal(full_cache),
    }


def _read_cache_temporal(cache) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` from a cache in chronological order.

    Two cases need handling:

    - ``KVCache`` pre-allocates in ``step=256``-sized chunks, so
      ``.keys`` / ``.values`` includes trailing stale slots beyond the
      valid range. Slice to ``cache.offset`` to drop them.
    - ``RotatingKVCache`` uses a ring buffer; ``_temporal_order`` returns
      the buffer in chronological order, possibly with stale tail slots
      if not yet full. We then slice to ``min(offset, max_size)``.
    """
    if cache.keys is None or cache.values is None:
        raise RuntimeError("cache is empty; did you forget to prefill?")
    if hasattr(cache, "_temporal_order"):
        k = cache._temporal_order(cache.keys)
        v = cache._temporal_order(cache.values)
        # Ring buffer: valid length is min(offset, max_size).
        max_size = getattr(cache, "max_size", k.shape[-2])
        valid = min(cache.offset, max_size)
        return k[..., :valid, :], v[..., :valid, :]
    # Plain KVCache: slice to offset to drop the unused allocation tail.
    return cache.keys[..., : cache.offset, :], cache.values[..., : cache.offset, :]
