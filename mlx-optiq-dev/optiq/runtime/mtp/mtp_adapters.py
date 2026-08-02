"""Sidecar adapters for the native MTP path.

The C4 lane needs trainable capacity inside the MTP proposer without changing
the target trunk or overwriting model weights.  This module keeps that contract:
it wraps selected MTP-only ``Linear`` modules with LoRA residuals and saves only
the adapter tensors plus metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np


SCHEMA_VERSION = 1
ADAPTER_KIND = "c4_mtp_lora_adapter"

DEFAULT_C4_LORA_TARGETS = (
    "fc",
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.k_proj",
    "layers.0.self_attn.v_proj",
    "layers.0.self_attn.o_proj",
    "layers.0.mlp.gate_proj",
    "layers.0.mlp.up_proj",
    "layers.0.mlp.down_proj",
)


@dataclass(frozen=True)
class AdapterState:
    metadata: dict[str, Any]
    tensors: dict[str, np.ndarray]


class LoRALinear(nn.Module):
    """LoRA residual around an existing MLX linear module."""

    def __init__(
        self,
        base: nn.Module,
        *,
        rank: int,
        alpha: float | None = None,
        init_scale: float = 0.01,
        depth_scales: Iterable[float] | None = None,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        weight = getattr(base, "weight", None)
        if weight is None or len(weight.shape) != 2:
            raise TypeError("LoRALinear requires a base module with 2D weight")
        output_dims, input_dims = self._dense_dims(base)
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = float(self.alpha / self.rank)
        self.lora_a = (float(init_scale) * mx.random.normal((self.rank, input_dims))).astype(mx.float32)
        self.lora_b = mx.zeros((output_dims, self.rank), dtype=mx.float32)
        self.depth_scales = (
            mx.array(np.asarray(list(depth_scales), dtype=np.float32))
            if depth_scales is not None
            else None
        )
        self.active_depth: int | None = None

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base(x)
        x32 = x.astype(mx.float32)
        delta = (x32 @ mx.transpose(self.lora_a)) @ mx.transpose(self.lora_b)
        scale = self.scaling
        if self.depth_scales is not None and self.active_depth is not None:
            if int(self.active_depth) <= 0:
                scale = 0.0
            else:
                index = min(max(int(self.active_depth) - 1, 0), int(self.depth_scales.shape[0]) - 1)
                scale = scale * self.depth_scales[index]
        return base_out + (delta * scale).astype(base_out.dtype)

    def adapter_tensors(self) -> dict[str, np.ndarray]:
        mx.eval(self.lora_a, self.lora_b)
        return {
            "lora_a": np.asarray(self.lora_a, dtype=np.float32),
            "lora_b": np.asarray(self.lora_b, dtype=np.float32),
            **(
                {"depth_scales": np.asarray(self.depth_scales, dtype=np.float32)}
                if self.depth_scales is not None
                else {}
            ),
        }

    def load_adapter_tensors(self, tensors: dict[str, np.ndarray]) -> None:
        a = np.asarray(tensors["lora_a"], dtype=np.float32)
        b = np.asarray(tensors["lora_b"], dtype=np.float32)
        if tuple(a.shape) != tuple(self.lora_a.shape):
            raise ValueError(f"lora_a shape mismatch: expected {self.lora_a.shape}, got {a.shape}")
        if tuple(b.shape) != tuple(self.lora_b.shape):
            raise ValueError(f"lora_b shape mismatch: expected {self.lora_b.shape}, got {b.shape}")
        self.lora_a = mx.array(a)
        self.lora_b = mx.array(b)
        if "depth_scales" in tensors:
            self.depth_scales = mx.array(np.asarray(tensors["depth_scales"], dtype=np.float32))
            mx.eval(self.lora_a, self.lora_b, self.depth_scales)
        else:
            self.depth_scales = None
            mx.eval(self.lora_a, self.lora_b)

    @staticmethod
    def _dense_dims(base: nn.Module) -> tuple[int, int]:
        weight = getattr(base, "weight")
        output_dims = int(weight.shape[0])
        input_dims = int(weight.shape[1])
        if isinstance(base, nn.QuantizedLinear):
            bits = int(getattr(base, "bits", 0) or 0)
            if bits <= 0:
                raise ValueError("QuantizedLinear adapter target has invalid bit width")
            input_dims = (input_dims * 32) // bits
        return output_dims, input_dims


def _text_model(model: Any) -> Any:
    return getattr(model, "language_model", model)


def _mtp_root(model: Any) -> Any:
    text_model = _text_model(model)
    root = getattr(text_model, "mtp", None)
    if root is None:
        raise RuntimeError("model has no injected MTP module")
    return root


def _get_child(obj: Any, part: str) -> Any:
    if isinstance(obj, (list, tuple)):
        return obj[int(part)]
    return getattr(obj, part)


def _set_child(obj: Any, part: str, value: Any) -> None:
    if isinstance(obj, list):
        obj[int(part)] = value
    else:
        setattr(obj, part, value)


def _resolve_parent(root: Any, path: str) -> tuple[Any, str]:
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise ValueError("empty adapter target path")
    parent = root
    for part in parts[:-1]:
        parent = _get_child(parent, part)
    return parent, parts[-1]


def _get_target(root: Any, path: str) -> Any:
    parent, leaf = _resolve_parent(root, path)
    return _get_child(parent, leaf)


def _set_target(root: Any, path: str, value: Any) -> None:
    parent, leaf = _resolve_parent(root, path)
    _set_child(parent, leaf, value)


def _normalize_targets(targets: str | Iterable[str] | None) -> list[str]:
    if targets is None:
        return list(DEFAULT_C4_LORA_TARGETS)
    if isinstance(targets, str):
        values = [part.strip() for part in targets.split(",") if part.strip()]
    else:
        values = [str(part).strip() for part in targets if str(part).strip()]
    if not values:
        raise ValueError("at least one adapter target is required")
    return values


def install_mtp_lora_adapters(
    model: Any,
    *,
    rank: int = 16,
    alpha: float | None = None,
    targets: str | Iterable[str] | None = None,
    init_scale: float = 0.01,
    depth_scales: Iterable[float] | None = None,
    trainable: bool = True,
    strict: bool = True,
) -> list[str]:
    """Wrap selected MTP-only linears with LoRA modules.

    Returns the installed target list.  The target trunk is left untouched; the
    only replaced modules live below ``text_model.mtp``.
    """

    if rank < 1:
        raise ValueError("rank must be >= 1")
    root = _mtp_root(model)
    installed: list[str] = []
    for target in _normalize_targets(targets):
        try:
            module = _get_target(root, target)
        except (AttributeError, IndexError, ValueError) as exc:
            if strict:
                raise ValueError(f"MTP adapter target not found: {target}") from exc
            continue
        if isinstance(module, LoRALinear):
            installed.append(target)
            continue
        wrapped = LoRALinear(
            module,
            rank=rank,
            alpha=alpha,
            init_scale=init_scale,
            depth_scales=depth_scales,
        )
        _set_target(root, target, wrapped)
        installed.append(target)

    previous = list(getattr(root, "_mtplx_lora_targets", []))
    root._mtplx_lora_targets = list(dict.fromkeys([*previous, *installed]))
    if trainable:
        freeze_for_mtp_adapter_training(model)
    return installed


def freeze_for_mtp_adapter_training(model: Any) -> None:
    """Freeze everything except LoRA tensors under the MTP module."""

    model.freeze()

    def _unfreeze_lora(_: str, module: Any) -> None:
        if isinstance(module, LoRALinear):
            module.unfreeze(recurse=False, keys=["lora_a", "lora_b"], strict=True)
            module.base.freeze()

    model.apply_to_modules(_unfreeze_lora)


def iter_mtp_lora_modules(model: Any) -> list[tuple[str, LoRALinear]]:
    root = _mtp_root(model)
    modules: list[tuple[str, LoRALinear]] = []
    targets = list(getattr(root, "_mtplx_lora_targets", DEFAULT_C4_LORA_TARGETS))
    for target in targets:
        try:
            module = _get_target(root, target)
        except Exception:
            continue
        if isinstance(module, LoRALinear):
            modules.append((target, module))
    return modules


def collect_mtp_lora_state(model: Any) -> AdapterState:
    modules = iter_mtp_lora_modules(model)
    if not modules:
        raise RuntimeError("no MTP LoRA adapters are installed")
    tensors: dict[str, np.ndarray] = {}
    target_meta: list[dict[str, Any]] = []
    for target, module in modules:
        local = module.adapter_tensors()
        tensors[f"{target}.lora_a"] = local["lora_a"]
        tensors[f"{target}.lora_b"] = local["lora_b"]
        target_meta.append(
            {
                "target": target,
                "rank": module.rank,
                "alpha": module.alpha,
                "scaling": module.scaling,
                "lora_a_shape": list(local["lora_a"].shape),
                "lora_b_shape": list(local["lora_b"].shape),
                "depth_scales": (
                    local["depth_scales"].astype(float).tolist()
                    if "depth_scales" in local
                    else None
                ),
            }
        )
        if "depth_scales" in local:
            tensors[f"{target}.depth_scales"] = local["depth_scales"]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": ADAPTER_KIND,
        "targets": target_meta,
    }
    return AdapterState(metadata=metadata, tensors=tensors)


def save_mtp_lora_adapter(
    path: Path | str,
    model: Any,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    state = collect_mtp_lora_state(model)
    merged_metadata = dict(state.metadata)
    if metadata:
        merged_metadata.update(metadata)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        metadata_json=np.array(json.dumps(merged_metadata, sort_keys=True)),
        **state.tensors,
    )
    return out


def load_mtp_lora_adapter(path: Path | str) -> AdapterState:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"unsupported MTP adapter schema: {metadata.get('schema_version')}")
        if metadata.get("kind") != ADAPTER_KIND:
            raise ValueError(f"unsupported MTP adapter kind: {metadata.get('kind')}")
        tensors = {
            key: np.asarray(data[key], dtype=np.float32)
            for key in data.files
            if key != "metadata_json"
        }
    return AdapterState(metadata=metadata, tensors=tensors)


def install_saved_mtp_lora_adapter(
    model: Any,
    path: Path | str,
    *,
    trainable: bool = False,
) -> dict[str, Any]:
    state = load_mtp_lora_adapter(path)
    target_entries = list(state.metadata.get("targets", []))
    targets = [str(entry["target"]) for entry in target_entries]
    rank_by_target = {str(entry["target"]): int(entry["rank"]) for entry in target_entries}
    alpha_by_target = {str(entry["target"]): float(entry["alpha"]) for entry in target_entries}
    depth_scales_by_target = {
        str(entry["target"]): entry.get("depth_scales")
        for entry in target_entries
    }
    if not targets:
        raise ValueError("adapter artifact contains no targets")

    root = _mtp_root(model)
    for target in targets:
        install_mtp_lora_adapters(
            model,
            rank=rank_by_target[target],
            alpha=alpha_by_target[target],
            targets=[target],
            depth_scales=depth_scales_by_target[target],
            trainable=False,
            strict=True,
        )
        module = _get_target(root, target)
        if not isinstance(module, LoRALinear):
            raise TypeError(f"target did not install as LoRALinear: {target}")
        module.load_adapter_tensors(
            {
                "lora_a": state.tensors[f"{target}.lora_a"],
                "lora_b": state.tensors[f"{target}.lora_b"],
                **(
                    {"depth_scales": state.tensors[f"{target}.depth_scales"]}
                    if f"{target}.depth_scales" in state.tensors
                    else {}
                ),
            }
        )
    if trainable:
        freeze_for_mtp_adapter_training(model)
    else:
        model.freeze()
    return state.metadata


def merge_installed_mtp_lora_adapters(model: Any) -> dict[str, Any]:
    """Bake installed MTP LoRA modules into their base linears in-memory.

    This is an inference probe helper.  Quantized base linears are dequantized,
    the LoRA delta is added, and the result is requantized with the base
    module's original quantization settings.  Dense base linears remain dense.

    Depth-gated LoRA adapters cannot be represented exactly as one static
    merged weight.  For those modules we merge the un-gated LoRA delta and
    report the source ``depth_scales`` so callers can treat the result as a
    diagnostic, not a faithful replacement for depth-gated inference.
    """
    import mlx.nn as nn

    modules = iter_mtp_lora_modules(model)
    if not modules:
        return {"merged": 0, "targets": []}

    root = _mtp_root(model)
    merged: list[dict[str, Any]] = []
    for target, module in modules:
        base = module.base
        dense_dims = LoRALinear._dense_dims(base)
        output_dims, input_dims = dense_dims
        if isinstance(base, nn.QuantizedLinear):
            base_weight = mx.dequantize(
                base.weight,
                base.scales,
                base.biases,
                group_size=base.group_size,
                bits=base.bits,
                mode=base.mode,
            ).astype(mx.float32)
            quantized = True
            base_bits = int(base.bits)
            base_group_size = int(base.group_size)
            base_mode = str(base.mode)
        else:
            weight = getattr(base, "weight", None)
            if weight is None or len(weight.shape) != 2:
                raise TypeError(f"cannot merge LoRA target without 2D base weight: {target}")
            base_weight = weight.astype(mx.float32)
            quantized = False
            base_bits = None
            base_group_size = None
            base_mode = None

        delta = (mx.matmul(module.lora_b, module.lora_a) * float(module.scaling)).astype(mx.float32)
        if tuple(delta.shape) != (output_dims, input_dims):
            raise ValueError(f"LoRA delta shape mismatch for {target}: {delta.shape} vs {(output_dims, input_dims)}")
        merged_weight = (base_weight + delta).astype(mx.bfloat16)
        mx.eval(merged_weight)

        linear = nn.Linear(input_dims, output_dims, bias=("bias" in base))
        linear.weight = merged_weight
        if "bias" in base:
            linear.bias = base.bias

        if quantized:
            replacement = nn.QuantizedLinear.from_linear(
                linear,
                group_size=base_group_size,
                bits=base_bits,
                mode=base_mode,
            )
            mx.eval(replacement.weight, replacement.scales, replacement.biases)
        else:
            replacement = linear
            mx.eval(replacement.weight)
            if "bias" in replacement:
                mx.eval(replacement.bias)

        _set_target(root, target, replacement)
        merged.append(
            {
                "target": target,
                "quantized": quantized,
                "bits": base_bits,
                "group_size": base_group_size,
                "mode": base_mode,
                "weight_shape": [int(item) for item in merged_weight.shape],
                "depth_scales": (
                    np.asarray(module.depth_scales, dtype=np.float32).astype(float).tolist()
                    if module.depth_scales is not None
                    else None
                ),
            }
        )

    root._mtplx_lora_targets = []
    model.freeze()
    return {"merged": len(merged), "targets": merged}


def set_mtp_adapter_depth(model: Any, depth: int | None) -> None:
    for _target, module in iter_mtp_lora_modules(model):
        module.active_depth = None if depth is None else int(depth)


@contextmanager
def mtp_adapter_depth(model: Any, depth: int | None):
    modules = iter_mtp_lora_modules(model)
    previous = [(module, module.active_depth) for _target, module in modules]
    try:
        for _target, module in modules:
            module.active_depth = None if depth is None else int(depth)
        yield
    finally:
        for module, old_depth in previous:
            module.active_depth = old_depth
