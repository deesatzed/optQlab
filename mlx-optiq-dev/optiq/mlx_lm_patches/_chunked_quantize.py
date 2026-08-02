"""Chunked replacement for ``mlx.nn.quantize``.

The stock ``mlx.nn.quantize`` builds a lazy graph of ``to_quantized()`` calls
across every quantizable submodule and finalizes them all in one
``model.update_modules`` call. For very large MoE bases (e.g. Gemma-4-26B-A4B
with 30 layers × 128-expert × 250M-param expert tensors) this single Metal
command buffer exceeds the GPU's per-buffer time limit and aborts with
``kIOGPUCommandBufferCallbackErrorTimeout``. ``mx.quantize`` itself runs in
~10 ms on a 484 MB tensor; the failure is the queued pipeline, not the kernel.

This module monkey-patches both ``mlx.nn.quantize`` and the
``mlx_lm.utils.nn.quantize`` reference so that:

  1. Each module's ``to_quantized()`` is followed by an immediate
     ``mx.eval(qm.parameters())`` so the Metal command buffer flushes per-module
     instead of accumulating across the whole model.
  2. The original (now-replaced) module's BF16 weight is dropped right after
     quantization so mmap'd source pages can be released.
  3. ``gc.collect()`` + ``mx.clear_cache()`` are called periodically to keep
     the page cache from growing unbounded across hundreds of expert tensors.

Behavior is identical to ``mlx.nn.quantize`` for small dense models. The
patch is no-op safe (re-import doesn't re-wrap) and gated behind
``register_chunked_quantize()`` — call it once before invoking
``mlx_lm.convert()``.
"""

from __future__ import annotations

import gc

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import Module
from mlx.utils import tree_map_with_path


_INSTALLED = False
_GC_EVERY = 30  # gc + clear_cache cadence (in modules)
_FREEABLE_ATTRS = ("weight", "scales", "biases", "bias")


def _chunked_quantize(
    model,
    group_size: int | None = None,
    bits: int | None = None,
    *,
    mode: str = "affine",
    class_predicate=None,
):
    """Drop-in replacement for ``mlx.nn.quantize`` that flushes per-module."""
    cp = class_predicate or (lambda _, m: hasattr(m, "to_quantized"))
    counter = {"n": 0}

    def _maybe(path, m):
        bp = cp(path, m)
        if not bp:
            return m
        if not hasattr(m, "to_quantized"):
            raise ValueError(f"Unable to quantize module of type {type(m)}")
        if isinstance(bp, dict):
            qm = m.to_quantized(**bp)
        elif isinstance(bp, bool):
            qm = m.to_quantized(group_size=group_size, bits=bits, mode=mode)
        else:
            raise ValueError(
                "class_predicate must return a bool or a dict of params"
            )

        # Force flush of this module's quantize op so the next module's
        # to_quantized() doesn't accumulate into the same Metal command buffer.
        mx.eval(qm.parameters())

        # Release the original module's mmap'd weight references so the page
        # cache doesn't accumulate across hundreds of large expert tensors.
        for attr in _FREEABLE_ATTRS:
            if hasattr(m, attr):
                try:
                    setattr(m, attr, None)
                except Exception:
                    pass

        counter["n"] += 1
        if counter["n"] % _GC_EVERY == 0:
            gc.collect()
            try:
                mx.clear_cache()
            except Exception:
                pass

        return qm

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(_maybe, leaves, is_leaf=Module.is_module)
    model.update_modules(leaves)


def register_chunked_quantize() -> None:
    """Install the chunked quantize patch (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    nn.quantize = _chunked_quantize
    # mlx_lm.utils binds nn.quantize at import time; rebind there too.
    try:
        import mlx_lm.utils as _u
        _u.nn.quantize = _chunked_quantize
    except Exception:
        # mlx_lm not installed in this env — registration is still valid for
        # any direct mlx.nn.quantize call.
        pass
    _INSTALLED = True
