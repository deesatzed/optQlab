"""Shape-ranked QuantizedLinear probe for small-M VerifyCore work."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from optiq.runtime.mtp.runtime import load


_KIND_SUFFIXES = (
    ("mlp.gate_proj", "mlp_gate"),
    ("mlp.up_proj", "mlp_up"),
    ("mlp.down_proj", "mlp_down"),
    ("self_attn.q_proj", "attn_q"),
    ("self_attn.k_proj", "attn_k"),
    ("self_attn.v_proj", "attn_v"),
    ("self_attn.o_proj", "attn_o"),
    ("linear_attn.in_proj_qkv", "gdn_qkv"),
    ("linear_attn.in_proj_z", "gdn_z"),
    ("linear_attn.in_proj_b", "gdn_b"),
    ("linear_attn.in_proj_a", "gdn_a"),
    ("linear_attn.out_proj", "gdn_o"),
    ("lm_head", "lm_head"),
)


@dataclass
class QuantizedLinearGroup:
    key: tuple[Any, ...]
    kind: str
    sample_path: str
    module: nn.QuantizedLinear
    bits: int
    group_size: int
    mode: str
    k: int
    n: int
    has_bias: bool
    paths: list[str] = field(default_factory=list)


def _path_kind(path: str) -> str:
    for suffix, kind in _KIND_SUFFIXES:
        if path.endswith(suffix):
            return kind
    if ".mtp." in f".{path}." or path.startswith("mtp."):
        return "mtp_other"
    return "other"


def _include_allowed(kind: str, include: set[str]) -> bool:
    if not include or "all" in include:
        return True
    if kind in include:
        return True
    family = kind.split("_", 1)[0]
    return family in include


def _input_dim(module: nn.QuantizedLinear) -> int:
    bits = int(getattr(module, "bits", 0) or 0)
    if bits <= 0:
        return 0
    return int(module.weight.shape[1]) * (32 // bits)


def _output_dim(module: nn.QuantizedLinear) -> int:
    return int(module.weight.shape[0])


def collect_quantized_linear_groups(
    model: nn.Module,
    *,
    include: str = "all",
) -> list[QuantizedLinearGroup]:
    include_set = {part.strip().lower() for part in include.split(",") if part.strip()}
    groups: dict[tuple[Any, ...], QuantizedLinearGroup] = {}
    for raw_path, module in model.named_modules():
        if not isinstance(module, nn.QuantizedLinear):
            continue
        path = str(raw_path)
        kind = _path_kind(path)
        if not _include_allowed(kind, include_set):
            continue
        bits = int(getattr(module, "bits", 0) or 0)
        group_size = int(getattr(module, "group_size", 0) or 0)
        mode = str(getattr(module, "mode", "affine"))
        k = _input_dim(module)
        n = _output_dim(module)
        has_bias = "bias" in module
        key = (kind, bits, group_size, mode, k, n, has_bias)
        group = groups.get(key)
        if group is None:
            group = QuantizedLinearGroup(
                key=key,
                kind=kind,
                sample_path=path,
                module=module,
                bits=bits,
                group_size=group_size,
                mode=mode,
                k=k,
                n=n,
                has_bias=has_bias,
                paths=[],
            )
            groups[key] = group
        group.paths.append(path)
    return sorted(groups.values(), key=lambda g: (g.kind, g.k, g.n, g.sample_path))


def _dtype(name: str) -> mx.Dtype:
    normalized = name.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return mx.bfloat16
    if normalized in {"fp16", "float16"}:
        return mx.float16
    if normalized in {"fp32", "float32"}:
        return mx.float32
    raise ValueError(f"unsupported dtype: {name}")


def _dflash_m16_eligible(group: QuantizedLinearGroup) -> bool:
    return (
        group.bits == 4
        and group.group_size in {32, 64, 128}
        and group.mode == "affine"
        and group.k % 32 == 0
        and group.n % 32 == 0
    )


def _time_module(
    module: nn.QuantizedLinear,
    *,
    m: int,
    k: int,
    dtype: mx.Dtype,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    x = mx.random.normal((1, m, k), dtype=dtype)
    mx.eval(x)
    for _ in range(warmup):
        y = module(x)
        mx.eval(y)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        y = module(x)
        mx.eval(y)
        samples.append(time.perf_counter() - started)
    return {
        "m": m,
        "samples_s": samples,
        "min_s": min(samples) if samples else 0.0,
        "mean_s": statistics.mean(samples) if samples else 0.0,
        "median_s": statistics.median(samples) if samples else 0.0,
    }


def _time_dense_mirror(
    module: nn.QuantizedLinear,
    *,
    x: mx.array,
    dtype: mx.Dtype,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    dense_weight = mx.dequantize(
        module.weight,
        module.scales,
        module.biases,
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
        dtype=dtype,
    )
    mx.eval(dense_weight)
    dequant_s = time.perf_counter() - started

    def dense_call():
        y = mx.matmul(x, dense_weight.T)
        if "bias" in module:
            y = y + module["bias"]
        return y

    for _ in range(warmup):
        y = dense_call()
        mx.eval(y)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        y = dense_call()
        mx.eval(y)
        samples.append(time.perf_counter() - started)

    quant_y = module(x)
    dense_y = dense_call()
    diff = mx.max(mx.abs(quant_y.astype(mx.float32) - dense_y.astype(mx.float32)))
    mx.eval(diff)
    return {
        "dequant_once_s": dequant_s,
        "samples_s": samples,
        "min_s": min(samples) if samples else 0.0,
        "mean_s": statistics.mean(samples) if samples else 0.0,
        "median_s": statistics.median(samples) if samples else 0.0,
        "max_abs_diff": float(diff.item()),
    }


def _parse_m_values(values: str | list[int]) -> list[int]:
    if isinstance(values, str):
        parsed = [int(part.strip()) for part in values.split(",") if part.strip()]
    else:
        parsed = [int(v) for v in values]
    return [v for v in parsed if v > 0]


def run_verify_qmm_probe(
    model_path: Path | str,
    *,
    m_values: str | list[int] = "1,3,4,5,16",
    repeats: int = 5,
    warmup: int = 2,
    include: str = "mlp,gdn,attn,lm_head,mtp",
    dtype: str = "bf16",
    mtp: bool = True,
    max_groups: int | None = None,
    seed: int = 0,
    dense_mirror: bool = False,
) -> dict[str, Any]:
    mx.random.seed(seed)
    rt = load(model_path, mtp=mtp)
    groups = collect_quantized_linear_groups(rt.model, include=include)
    if max_groups is not None:
        groups = groups[:max_groups]
    dtype_value = _dtype(dtype)
    m_list = _parse_m_values(m_values)

    rows: list[dict[str, Any]] = []
    for group in groups:
        m_results = []
        for m in m_list:
            x = mx.random.normal((1, m, group.k), dtype=dtype_value)
            mx.eval(x)
            timing = _time_module(
                group.module,
                m=m,
                k=group.k,
                dtype=dtype_value,
                repeats=repeats,
                warmup=warmup,
            )
            timing["estimated_full_forward_mean_s"] = timing["mean_s"] * len(group.paths)
            timing["estimated_full_forward_min_s"] = timing["min_s"] * len(group.paths)
            if dense_mirror:
                dense_timing = _time_dense_mirror(
                    group.module,
                    x=x,
                    dtype=dtype_value,
                    repeats=repeats,
                    warmup=warmup,
                )
                timing["dense_mirror"] = dense_timing
                timing["dense_mirror_estimated_full_forward_mean_s"] = (
                    dense_timing["mean_s"] * len(group.paths)
                )
                timing["dense_mirror_vs_quant_ratio"] = (
                    timing["mean_s"] / dense_timing["mean_s"]
                    if dense_timing["mean_s"]
                    else None
                )
                timing["dense_mirror_profitable"] = dense_timing["mean_s"] < timing["mean_s"]
            m_results.append(timing)
        rows.append(
            {
                "kind": group.kind,
                "sample_path": group.sample_path,
                "module_count": len(group.paths),
                "sample_paths": group.paths[:8],
                "bits": group.bits,
                "group_size": group.group_size,
                "mode": group.mode,
                "k": group.k,
                "n": group.n,
                "has_bias": group.has_bias,
                "weight_shape": list(group.module.weight.shape),
                "dflash_m16_eligible": _dflash_m16_eligible(group),
                "small_m_verify_qmm_candidate": _dflash_m16_eligible(group),
                "m_results": m_results,
            }
        )

    ranked_by_m: dict[str, list[dict[str, Any]]] = {}
    for m in m_list:
        ranked = []
        for row in rows:
            timing = next((item for item in row["m_results"] if item["m"] == m), None)
            if timing is None:
                continue
            ranked.append(
                {
                    "kind": row["kind"],
                    "sample_path": row["sample_path"],
                    "module_count": row["module_count"],
                    "bits": row["bits"],
                    "group_size": row["group_size"],
                    "mode": row["mode"],
                    "k": row["k"],
                    "n": row["n"],
                    "mean_s": timing["mean_s"],
                    "min_s": timing["min_s"],
                    "estimated_full_forward_mean_s": timing["estimated_full_forward_mean_s"],
                    "estimated_full_forward_min_s": timing["estimated_full_forward_min_s"],
                    "dflash_m16_eligible": row["dflash_m16_eligible"],
                    "dense_mirror_vs_quant_ratio": timing.get("dense_mirror_vs_quant_ratio"),
                    "dense_mirror_profitable": timing.get("dense_mirror_profitable"),
                }
            )
        ranked_by_m[str(m)] = sorted(
            ranked,
            key=lambda item: item["estimated_full_forward_mean_s"],
            reverse=True,
        )

    return {
        "model_path": str(model_path),
        "mtp": mtp,
        "include": include,
        "dtype": str(dtype_value),
        "m_values": m_list,
        "repeats": repeats,
        "warmup": warmup,
        "dense_mirror": dense_mirror,
        "groups": rows,
        "ranked_by_m": ranked_by_m,
        "note": (
            "Times are synchronized isolated QuantizedLinear calls grouped by repeated shape. "
            "Use estimated_full_forward_* as a kernel-target ranking, not a headline runtime."
        ),
    }


def write_verify_qmm_probe(path: Path | str, result: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
