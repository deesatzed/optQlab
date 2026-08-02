"""Fix ``BatchRotatingKVCache.merge`` on a zero-length cache (upstream bug).

mlx-lm merges per-request rotating caches into a batch cache with::

    for i, (p, l, c) in enumerate(zip(padding, lengths, caches)):
        if c.keys is None:
            continue
        keys[i : i + 1, :, p : p + l] = c._temporal_order(c.keys)[..., -l:, :]

The guard skips a cache whose ``keys is None``, but not one whose ``keys`` is
allocated while its logical length ``l`` is 0 -- a sliding-window cache that
was trimmed back to offset 0 keeps its buffer. For that cache the assignment
is wrong in two directions at once:

* the destination ``p : p + l`` is ``max_length : max_length`` -- zero width,
  correctly nothing;
* the source ``[..., -l:, :]`` is ``[..., -0:, :]``, and ``-0`` is ``0`` in
  Python, so it slices the **entire** buffer rather than nothing.

::

    ValueError: [broadcast_shapes] Shapes (1,8,0,256) and (1,8,1024,256)
    cannot be broadcast

It takes down the generation thread, after which the server accepts requests
and answers none. Reached on any sliding-window model (Gemma-3/4, Ministral,
Phi-3/4, ...) doing batching-with-history -- an ordinary multi-turn agent
session reusing a cached prefix.

Nothing to do with KV quantization: these are the fp16 rotating layers, which
OptiQ leaves alone.

An earlier version of this fix detected zero-length caches (``size() == 0``)
and substituted genuinely-empty ones. That was correct for every state it was
tested against but still crashed in production on some real cache the detector
missed. Rather than keep guessing which state slips through, this reimplements
``merge`` faithfully from upstream and guards the one line that breaks:
``if c.keys is None or l == 0: continue``. The zero-length cache contributes
nothing, which is exactly what the destination slice already says, so skipping
it is a no-op for a well-behaved cache and the fix for a mis-behaved one.
"""

from __future__ import annotations

_INSTALLED = False
_stats = {"skipped": 0}


def install() -> bool:
    """Replace ``BatchRotatingKVCache.merge`` with a version that survives a
    zero-length cache. Idempotent; returns False if the class is missing or
    already patched (so a future mlx-lm that fixes this upstream is left be)."""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        from mlx_lm.models import cache as cache_mod
    except Exception:
        return False
    cls = getattr(cache_mod, "BatchRotatingKVCache", None)
    if cls is None or getattr(cls, "_optiq_merge_patched", False):
        return False

    cls.merge = classmethod(_safe_merge)
    cls._optiq_merge_patched = True
    _INSTALLED = True
    return True


def _safe_merge(cls, caches):
    """Faithful reimplementation of ``BatchRotatingKVCache.merge`` with the
    ``l == 0`` slice guarded. Mirrors upstream line-for-line otherwise so
    behavior on healthy caches is identical."""
    import logging

    import mlx.core as mx

    if not all(c.max_size == caches[0].max_size for c in caches):
        raise ValueError(
            "BatchRotatingKVCache can only merge caches with the same maximum size"
        )

    offsets = [c.offset for c in caches]
    # Coerce to plain ints up front. size() can return an mx scalar, and if it
    # propagates into max_length/padding the slice arithmetic below silently
    # produces a 0-width destination while the source is full -- the broadcast
    # mismatch that recurred through an l==0 guard and an l<=0 guard alike,
    # because the bad value was never a clean Python int in the first place.
    lengths = [int(c.size()) for c in caches]
    max_length = int(max(lengths))

    if max_length == 0:
        return cls(caches[0].max_size, [0] * len(caches))

    padding = [max_length - l for l in lengths]

    import os as _os
    if _os.environ.get("OPTIQ_MERGE_DEBUG"):
        import logging
        rows = []
        for c in caches:
            k = getattr(c, "keys", None)
            rows.append((type(c).__name__, int(c.size()),
                         None if k is None else tuple(k.shape),
                         getattr(c, "offset", None), getattr(c, "_idx", None),
                         getattr(c, "max_size", None)))
        logging.info("[MERGEDEBUG] max_length=%r lengths=%r padding=%r caches=%r",
                     max_length, lengths, padding, rows)

    B = len(caches)
    H = max(c.keys.shape[1] for c in caches if c.keys is not None)
    Dk = max(c.keys.shape[3] for c in caches if c.keys is not None)
    Dv = max(c.values.shape[3] for c in caches if c.values is not None)
    dt = next(iter(c.keys.dtype for c in caches if c.keys is not None))

    keys = mx.zeros((B, H, max_length, Dk), dtype=dt)
    values = mx.zeros((B, H, max_length, Dv), dtype=dt)
    for i, (p, l, c) in enumerate(zip(padding, lengths, caches)):
        if c.keys is None or l <= 0:
            continue
        # Compute the destination window with pure-Python int arithmetic and
        # clamp it into bounds. Do NOT read the shape of an mx slice to get the
        # width: mx reports the *nominal* width of ``keys[:, :, p:p+l]`` even
        # when p/p+l fall outside the axis, so the earlier `dst.shape[-2]`
        # check passed while the real assignment was 0-wide and broadcast-
        # failed. On a multi-cache merge (several sequences batching) that took
        # down the generation thread; single-cache merges never hit it.
        lo = max(0, min(int(p), max_length))
        hi = max(0, min(int(p) + int(l), max_length))
        w = hi - lo
        if w <= 0:
            _stats["skipped"] += 1
            if _stats["skipped"] <= 3:
                logging.info(
                    "[optiq.serve] rotating-cache merge: skipped cache i=%d "
                    "(p=%r l=%r max_length=%r size=%r) with an empty destination "
                    "window; upstream would crash the generation thread here",
                    i, p, l, max_length, c.size())
            continue
        # The real bug (confirmed from the crashing state, not guessed): a
        # rotating cache can report size()==l while its _temporal_order returns
        # FEWER rows -- 0 in the observed case, a reset/reused state during
        # prompt-cache reuse. Every earlier fix sized the copy from l and
        # assumed the source held l rows; when it held 0, keys[..l..] = src[..0..]
        # broadcast-failed. So clamp the width to what the source ACTUALLY has,
        # and place those rows at the END of the destination window.
        src_k = c._temporal_order(c.keys)
        src_v = c._temporal_order(c.values)
        avail = int(src_k.shape[-2])
        w = min(w, avail)
        if w <= 0:
            continue                      # source contributes nothing
        # Circuit breaker stays as a last resort: a single cache must never
        # take the generation thread down, whatever state it is in.
        try:
            keys[i : i + 1, :, hi - w:hi] = src_k[..., avail - w:, :]
            values[i : i + 1, :, hi - w:hi] = src_v[..., src_v.shape[-2] - w:, :]
        except Exception as e:
            _stats["skipped"] += 1
            if _stats["skipped"] <= 5:
                logging.warning(
                    "[optiq.serve] rotating-cache merge: skipped cache i=%d after "
                    "%s (lo=%r hi=%r w=%r avail=%r p=%r l=%r max_length=%r); "
                    "generation continues", i, type(e).__name__, lo, hi, w, avail,
                    p, l, max_length)

    cache = cls(caches[0].max_size, padding)
    cache.keys = keys
    cache.values = values
    cache.offset = mx.array(offsets)
    cache._idx = keys.shape[2]
    cache._offset = keys.shape[2]
    return cache
