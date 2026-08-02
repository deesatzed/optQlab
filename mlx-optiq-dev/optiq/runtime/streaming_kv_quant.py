"""Streaming KV-cache quantization — half of OptiQ's tight-RAM KV-quant fix.

Why this exists
---------------
mlx-lm's stock ``maybe_quantize_kv_cache`` enqueues every layer's
``c.to_quantized(...)`` as a lazy MLX op, then evals them all in a
single batch. At that batch eval point, MLX holds both
(fp16_all_layers + quantized_all_layers) co-resident in memory. On a
24 GB Mac with a 9B model at 32k context this conversion transient is
big enough to OOM the process before the quantized cache ever gets a
chance to save memory.

This module patches mlx-lm to process one layer at a time:
quantize K, mx.eval, drop the fp16 reference, clear the buffer pool,
then the same for V, repeat. The conversion transient drops from
"all layers worth of fp16 K AND V" (~5+ GB on Granite-4.1-8b at 32k)
to roughly one layer's worth (~150 MB).

This is half the OOM fix. The other half is ``fused_quant_sdpa``
(below) which prevents the prefill scores-matrix spike. Both are
default-on in ``optiq serve`` when KV-quant is enabled — pass
``--no-fused-kv`` to opt out (e.g. for bit-exact comparison vs stock).

Usage
-----
    from optiq.runtime.streaming_kv_quant import install
    install()
    # then call mlx_lm.stream_generate / .generate as usual
"""
from __future__ import annotations
import gc

import mlx.core as mx


_INSTALLED = False
_ORIGINAL = None


def streaming_maybe_quantize_kv_cache(
    prompt_cache, quantized_kv_start, kv_group_size, kv_bits
):
    if kv_bits is None:
        return

    # A RotatingKVCache must become a RotatingQuantizedKVCache. This used to build
    # a plain QuantizedKVCache unconditionally, which silently corrupts a sliding
    # window past its first wrap. serve.py's --kv-config path always got this
    # right; this --kv-bits path did not, and both patch the same mlx-lm symbol.
    from ..runtime.kv.rotating import quantize_cache_layer

    for e, c in enumerate(prompt_cache):
        if not hasattr(c, "to_quantized"):
            continue
        if c.offset < quantized_kv_start:
            continue
        if c.keys is None:
            continue

        new_c = quantize_cache_layer(c, bits=kv_bits, group_size=kv_group_size)
        if new_c is c:
            continue
        gc.collect()
        mx.clear_cache()

        prompt_cache[e] = new_c
        del c, new_c
        gc.collect()
        mx.clear_cache()


def install() -> bool:
    """Monkey-patch mlx_lm.generate.maybe_quantize_kv_cache. Idempotent.
    Returns True if the install changed state, False if already installed."""
    global _INSTALLED, _ORIGINAL
    if _INSTALLED:
        return False
    import importlib
    gen_mod = importlib.import_module("mlx_lm.generate")
    _ORIGINAL = gen_mod.maybe_quantize_kv_cache
    gen_mod.maybe_quantize_kv_cache = streaming_maybe_quantize_kv_cache
    _INSTALLED = True
    return True


def uninstall() -> bool:
    """Restore the stock mlx-lm function. Idempotent."""
    global _INSTALLED, _ORIGINAL
    if not _INSTALLED:
        return False
    import importlib
    gen_mod = importlib.import_module("mlx_lm.generate")
    gen_mod.maybe_quantize_kv_cache = _ORIGINAL
    _INSTALLED = False
    _ORIGINAL = None
    return True


def is_installed() -> bool:
    return _INSTALLED
