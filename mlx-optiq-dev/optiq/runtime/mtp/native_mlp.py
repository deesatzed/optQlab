"""Env-gated MLP call probes for Qwen3Next.

This module wires exact MLP boundary variants into the runtime as diagnostic
profiles.  They are deliberately not defaults: prior probes showed the first
native rowwise primitive is slower than stock, but Phase 2 now shows MLP is the
largest target-forward floor.  The direct-HTTP harness therefore needs a clean
way to test MLP/projection boundary changes under the real long-response gates.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn


_PATCHED = False
_STATS = {
    "enabled": False,
    "variant": "stock",
    "variant_calls": 0,
    "native_calls": 0,
    "fallback_calls": 0,
    "min_m": 2,
    "max_m": 6,
    "context_threshold": 0,
    "current_context_tokens": -1,
}


def _env_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _native_extension_path() -> Path:
    return Path(__file__).resolve().parents[1] / "native_extensions" / "verify_mlp"


def _load_native_full_mlp():
    native_path = _native_extension_path()
    if str(native_path) not in sys.path:
        sys.path.insert(0, str(native_path))
    from mtplx_native_mlp import gate_up_swiglu_down_qmv4_rowwise

    return gate_up_swiglu_down_qmv4_rowwise


def _load_native_gate_up_mlp():
    native_path = _native_extension_path()
    if str(native_path) not in sys.path:
        sys.path.insert(0, str(native_path))
    from mtplx_native_mlp import gate_up_swiglu_qmv4_rowwise

    return gate_up_swiglu_qmv4_rowwise


def _normalized_variant() -> str:
    raw = os.environ.get("MTPLX_MLP_CALL_VARIANT", "").strip().lower()
    variant = raw.replace("-", "_")
    if not variant and _env_enabled("MTPLX_NATIVE_MLP_ROWWISE"):
        variant = "native_full_rowwise"
    aliases = {
        "": "stock",
        "0": "stock",
        "false": "stock",
        "off": "stock",
        "none": "stock",
        "stock": "stock",
        "compiled": "compiled_shapeless",
        "compile": "compiled_shapeless",
        "compiled_shapeless": "compiled_shapeless",
        "tiled_gateup": "tiled_gateup",
        "rowwise_sg4": "rowwise_sg4",
        "split_gateup": "split_gateup",
        "native_rowwise": "native_rowwise",
        "native_full_rowwise": "native_full_rowwise",
    }
    if variant not in aliases:
        raise ValueError(
            "Unknown MTPLX_MLP_CALL_VARIANT="
            f"{os.environ.get('MTPLX_MLP_CALL_VARIANT')!r}"
        )
    return aliases[variant]


def _eligible_mlp_call(x: mx.array, mlp: Any, *, min_m: int, max_m: int) -> bool:
    threshold = int(_STATS.get("context_threshold", 0) or 0)
    current_context = int(_STATS.get("current_context_tokens", -1) or -1)
    if threshold > 0 and (current_context < 0 or current_context < threshold):
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if len(x.shape) < 2:
        return False
    m = int(x.shape[-2])
    if m < min_m or m > max_m:
        return False
    batch_count = 1
    for dim in x.shape[:-2]:
        batch_count *= int(dim)
    if batch_count != 1:
        return False

    modules = (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    if not all(isinstance(module, nn.QuantizedLinear) for module in modules):
        return False
    if not all(int(getattr(module, "bits", 0) or 0) == 4 for module in modules):
        return False
    group_sizes = [int(getattr(module, "group_size", 0) or 0) for module in modules]
    if group_sizes[0] not in {32, 64, 128} or len(set(group_sizes)) != 1:
        return False
    if not all(str(getattr(module, "mode", "affine")) == "affine" for module in modules):
        return False
    if any("bias" in module for module in modules):
        return False
    if tuple(mlp.gate_proj.weight.shape) != tuple(mlp.up_proj.weight.shape):
        return False
    if tuple(mlp.gate_proj.scales.shape) != tuple(mlp.up_proj.scales.shape):
        return False
    if tuple(mlp.gate_proj.biases.shape) != tuple(mlp.up_proj.biases.shape):
        return False
    if int(mlp.gate_proj.weight.shape[0]) != int(mlp.down_proj.weight.shape[1]) * 8:
        return False
    return True


def configure_native_mlp(model: Any | None = None) -> dict[str, int | bool | str]:
    """Patch Qwen3NextMLP when an MLP call variant is enabled."""

    global _PATCHED
    variant = _normalized_variant()
    enabled = variant != "stock"
    _STATS["enabled"] = bool(enabled)
    _STATS["variant"] = variant
    _STATS["min_m"] = _env_int("MTPLX_NATIVE_MLP_MIN_M", default=2)
    _STATS["max_m"] = _env_int("MTPLX_NATIVE_MLP_MAX_M", default=6)
    _STATS["context_threshold"] = _env_int(
        "MTPLX_NATIVE_MLP_CONTEXT_THRESHOLD",
        default=0,
    )
    if not enabled:
        return native_mlp_stats()
    if _PATCHED:
        return native_mlp_stats()

    from mlx_lm.models.qwen3_next import Qwen3NextMLP
    from mlx_lm.models.qwen3_next import swiglu

    original_call = Qwen3NextMLP.__call__
    min_m = int(_STATS["min_m"])
    max_m = int(_STATS["max_m"])

    def _fallback(self: Any, x: mx.array):
        _STATS["fallback_calls"] = int(_STATS["fallback_calls"]) + 1
        return original_call(self, x)

    def patched_call(self: Any, x: mx.array):
        if not _eligible_mlp_call(x, self, min_m=min_m, max_m=max_m):
            return _fallback(self, x)
        leading = x.shape[:-2]
        m = int(x.shape[-2])
        k = int(x.shape[-1])

        if variant == "compiled_shapeless":
            compiled = getattr(self, "_mtplx_compiled_mlp_shapeless", None)
            if compiled is None:
                compiled = mx.compile(
                    lambda value: self.down_proj(
                        swiglu(self.gate_proj(value), self.up_proj(value))
                    ),
                    shapeless=True,
                )
                object.__setattr__(self, "_mtplx_compiled_mlp_shapeless", compiled)
            _STATS["variant_calls"] = int(_STATS["variant_calls"]) + 1
            return compiled(x)

        if variant == "tiled_gateup":
            from .kernels.verify_mlp_fused import gate_up_swiglu_qmv4_activation

            act = gate_up_swiglu_qmv4_activation(x, self.gate_proj, self.up_proj)
            _STATS["variant_calls"] = int(_STATS["variant_calls"]) + 1
            return self.down_proj(act)

        if variant == "rowwise_sg4":
            from .kernels.verify_mlp_fused import gate_up_swiglu_qmv4_activation_rowwise

            act = gate_up_swiglu_qmv4_activation_rowwise(
                x,
                self.gate_proj,
                self.up_proj,
                results_per_simdgroup=4,
                num_simdgroups=4,
            )
            _STATS["variant_calls"] = int(_STATS["variant_calls"]) + 1
            return self.down_proj(act)

        if variant == "split_gateup":
            from .kernels.verify_mlp_fused import gate_up_swiglu_qmv4_activation_split

            act = gate_up_swiglu_qmv4_activation_split(
                x,
                self.gate_proj,
                self.up_proj,
            )
            _STATS["variant_calls"] = int(_STATS["variant_calls"]) + 1
            return self.down_proj(act)

        if variant == "native_rowwise":
            native_gate_up_mlp = _load_native_gate_up_mlp()
            act = native_gate_up_mlp(
                x.reshape(m, k),
                self.gate_proj.weight,
                self.gate_proj.scales,
                self.gate_proj.biases,
                self.up_proj.weight,
                self.up_proj.scales,
                self.up_proj.biases,
                int(self.gate_proj.group_size),
                2,
            )
            _STATS["native_calls"] = int(_STATS["native_calls"]) + 1
            _STATS["variant_calls"] = int(_STATS["variant_calls"]) + 1
            return self.down_proj(
                act.reshape(*leading, m, int(self.gate_proj.weight.shape[0]))
            )

        if variant == "native_full_rowwise":
            native_full_mlp = _load_native_full_mlp()
            out = native_full_mlp(
                x.reshape(m, k),
                self.gate_proj.weight,
                self.gate_proj.scales,
                self.gate_proj.biases,
                self.up_proj.weight,
                self.up_proj.scales,
                self.up_proj.biases,
                self.down_proj.weight,
                self.down_proj.scales,
                self.down_proj.biases,
                int(self.gate_proj.group_size),
                2,
            )
            _STATS["native_calls"] = int(_STATS["native_calls"]) + 1
            _STATS["variant_calls"] = int(_STATS["variant_calls"]) + 1
            return out.reshape(*leading, m, int(self.down_proj.weight.shape[0]))

        return _fallback(self, x)

    Qwen3NextMLP.__call__ = patched_call
    _PATCHED = True
    return native_mlp_stats()


def native_mlp_stats() -> dict[str, int | bool | str]:
    return dict(_STATS)


def set_native_mlp_context(context_tokens: int | None) -> None:
    _STATS["current_context_tokens"] = -1 if context_tokens is None else int(context_tokens)
