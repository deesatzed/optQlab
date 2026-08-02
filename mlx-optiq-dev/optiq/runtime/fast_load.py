"""Load an already-quantized checkpoint without building it twice.

``mlx_lm``'s loader constructs the model with *randomly initialized float
weights*, calls ``nn.quantize()`` on them, and only then calls ``load_weights()``
to overwrite everything with the real tensors from disk. The middle step is pure
waste: ``QuantizedLinear.__init__`` runs ``mx.random.uniform(shape=(out, in))``
followed by ``mx.quantize(...)``, so a full copy of the model is built and
evaluated before being discarded.

Measured on Qwen3.5-9B-OptiQ-4bit (7.63 GiB on disk)::

    mx.load all shards   peak= 0.00  active= 0.00
    construct            peak= 0.00  active= 0.00
    nn.quantize          peak=10.92  active= 7.13   <-- quantizing noise
    load_weights         peak=10.92  active= 0.00   <-- discarded

On Qwen3.5-122B-A10B-OptiQ-2bit (42.8 GiB) that transient peaks near 53 GiB. So
a 36 GiB Mac swaps ~35 GB to open a model it can comfortably hold, a 24 GiB Mac
is OS-killed inside ``load()``, and the load takes 103 s off a warm SSD. It bites
every caller of ``mlx_lm.load`` — single-Mac serving, SSD expert streaming and
the cluster alike. Sharding does not help: every rank pays it in full.

The fix, in two parts:

* ``mx.load`` is wrapped to remember the mmap-backed, still-unevaluated arrays it
  returns, keyed by parameter path.
* ``nn.quantize`` is replaced with a version that installs those real arrays into
  the quantized modules directly, instead of quantizing noise. The checkpoint's
  own lazy arrays *become* the parameters, so nothing is allocated.

Any parameter it cannot find falls back to a lazy ``mx.zeros`` placeholder, and
any module type it does not recognise falls back to the original ``to_quantized``
— an unusual checkpoint degrades to "cheap", never to "broken".

Usage::

    with fast_quantized_load():
        model, tokenizer = mlx_lm.load(path, lazy=True)
"""

from __future__ import annotations

import contextlib

import mlx.core as mx
import mlx.nn as nn

__all__ = ["fast_quantized_load"]


def _packed_shapes(out_dims: int, in_dims: int, bits: int, group_size: int):
    """Shapes ``mx.quantize`` produces for an ``(out, in)`` affine-quantized
    weight: values packed into uint32 words, one scale/bias per group."""
    return (out_dims, in_dims * bits // 32), (out_dims, in_dims // group_size)


@contextlib.contextmanager
def fast_quantized_load():
    """Load a quantized checkpoint without re-quantizing random weights."""
    from mlx.nn.layers.base import Module
    from mlx.utils import tree_map_with_path

    try:
        from mlx_lm.models import switch_layers as sl
    except Exception:
        sl = None

    seen: dict = {}                       # param path -> lazy array off disk
    orig_load, orig_quantize = mx.load, nn.quantize

    def _load(*a, **kw):
        out = orig_load(*a, **kw)
        if isinstance(out, dict):
            seen.update(out)
        return out

    def _install(mod, path, wshape, sshape, bshape, group_size, bits, mode):
        """Point the module at the checkpoint's own arrays (lazy zeros if absent)."""
        mod.group_size, mod.bits, mod.mode = group_size, bits, mode
        w = seen.get(f"{path}.weight")
        mod.weight = w if w is not None else mx.zeros(wshape, dtype=mx.uint32)
        s = seen.get(f"{path}.scales")
        mod.scales = s if s is not None else mx.zeros(sshape, dtype=mx.float16)
        if mode == "affine":
            b = seen.get(f"{path}.biases")
            mod.biases = b if b is not None else mx.zeros(sshape, dtype=mx.float16)
        else:
            mod.biases = None
        if bshape is not None:
            bi = seen.get(f"{path}.bias")
            mod.bias = bi if bi is not None else mx.zeros(bshape, dtype=mx.float16)
        mod.freeze()
        return mod

    def _quantize(model, group_size=64, bits=4, *, mode="affine",
                  quantize_input=False, class_predicate=None, **_):
        def _maybe(path, m):
            pred = True if class_predicate is None else class_predicate(path, m)
            if pred is False or not hasattr(m, "to_quantized"):
                return m
            kw = dict(pred) if isinstance(pred, dict) else {}
            kw.pop("quantize_input", None)
            gs = kw.get("group_size", group_size)
            bt = kw.get("bits", bits)
            md = kw.get("mode", mode)

            if isinstance(m, nn.Linear):
                out, inp = m.weight.shape
                q = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
                nn.Module.__init__(q)
                w, s = _packed_shapes(out, inp, bt, gs)
                return _install(q, path, w, s, (out,) if "bias" in m else None,
                                gs, bt, md)
            if isinstance(m, nn.Embedding):
                n, dims = m.weight.shape
                q = nn.QuantizedEmbedding.__new__(nn.QuantizedEmbedding)
                nn.Module.__init__(q)
                q.num_embeddings, q.dims = n, dims
                w, s = _packed_shapes(n, dims, bt, gs)
                return _install(q, path, w, s, None, gs, bt, md)
            if sl is not None and isinstance(m, sl.SwitchLinear):
                experts, out, inp = m.weight.shape
                q = sl.QuantizedSwitchLinear.__new__(sl.QuantizedSwitchLinear)
                nn.Module.__init__(q)
                w, s = _packed_shapes(out, inp, bt, gs)
                return _install(q, path, (experts, *w), (experts, *s),
                                (experts, out) if "bias" in m else None, gs, bt, md)

            # Unknown module type: let mlx handle it (slow, but correct).
            full = dict(pred) if isinstance(pred, dict) else {
                "group_size": group_size, "bits": bits, "mode": mode}
            full.pop("quantize_input", None)
            return m.to_quantized(**full)

        leaves = tree_map_with_path(_maybe, model.leaf_modules(),
                                    is_leaf=Module.is_module)
        model.update_modules(leaves)

    mx.load, nn.quantize = _load, _quantize
    try:
        yield
    finally:
        mx.load, nn.quantize = orig_load, orig_quantize
        seen.clear()
