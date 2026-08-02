"""Native C++/Metal GDN tail probes.

These wrappers keep the production path conservative: unsupported shapes fall
back to the stock MLX modules, and callers opt in through an environment flag.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def _stock_tail(x: mx.array, gate: mx.array, weight: mx.array, eps: float, out_proj: Any) -> mx.array:
    normed = mx.fast.rms_norm(x, weight, eps)
    gate_f = gate.astype(mx.float32)
    out = (gate_f * mx.sigmoid(gate_f) * normed.astype(mx.float32)).astype(x.dtype)
    return out_proj(out.reshape(*x.shape[:-2], -1))


def _native_module():
    native_path = Path(__file__).resolve().parents[2] / "native_extensions" / "verify_mlp"
    if str(native_path) not in sys.path:
        sys.path.insert(0, str(native_path))
    try:
        import mtplx_native_mlp  # type: ignore
    except Exception:
        return None
    return mtplx_native_mlp


def is_native_gdn_tail_eligible(
    x: mx.array,
    gate: mx.array,
    weight: mx.array,
    out_proj: Any,
) -> bool:
    if not mx.metal.is_available():
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if gate.dtype != x.dtype or weight.dtype != x.dtype:
        return False
    if tuple(x.shape) != tuple(gate.shape):
        return False
    if len(x.shape) != 4 or len(weight.shape) != 1:
        return False
    if int(x.shape[-1]) != int(weight.shape[0]):
        return False
    if not isinstance(out_proj, nn.QuantizedLinear):
        return False
    if int(getattr(out_proj, "bits", 0) or 0) != 8:
        return False
    if str(getattr(out_proj, "mode", "affine")) != "affine":
        return False
    if int(getattr(out_proj, "group_size", 0) or 0) not in {32, 64, 128}:
        return False
    if "bias" in out_proj:
        return False
    hv = int(x.shape[-2])
    dv = int(x.shape[-1])
    return int(out_proj.weight.shape[1]) * 4 == hv * dv


def native_gdn_norm_gate_out_qmv8(
    x: mx.array,
    gate: mx.array,
    weight: mx.array,
    eps: float,
    out_proj: Any,
    *,
    num_simdgroups: int = 2,
) -> mx.array:
    """Return ``out_proj(Qwen3NextRMSNormGated(x, gate).reshape(...))``.

    Falls back to stock operations unless the exact native probe is available
    and the shapes match the Qwen3.6 GDN verify path.
    """
    if not is_native_gdn_tail_eligible(x, gate, weight, out_proj):
        return _stock_tail(x, gate, weight, eps, out_proj)
    native = _native_module()
    if native is None:
        return _stock_tail(x, gate, weight, eps, out_proj)

    leading = x.shape[:-2]
    m = 1
    for dim in leading:
        m *= int(dim)
    hv = int(x.shape[-2])
    dv = int(x.shape[-1])
    x2 = mx.contiguous(x.reshape(m * hv, dv))
    gate2 = mx.contiguous(gate.reshape(m * hv, dv))
    y = native.gdn_norm_gate_out_qmv8(
        x2,
        gate2,
        weight,
        out_proj.weight,
        out_proj.scales,
        out_proj.biases,
        hv,
        float(eps),
        int(out_proj.group_size),
        int(num_simdgroups),
    )
    return y.reshape(*leading, int(out_proj.weight.shape[0]))
