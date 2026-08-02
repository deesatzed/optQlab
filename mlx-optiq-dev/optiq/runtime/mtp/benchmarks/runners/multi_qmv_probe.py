"""Probe the experimental M=3 multi-vector qmv VerifyCore primitive."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from optiq.runtime.mtp.benchmarks.runners.verify_qmm_probe import collect_quantized_linear_groups
from optiq.runtime.mtp.runtime import load
from optiq.runtime.mtp.verify_qmv import (
    is_multi3_qmv4_eligible,
    multi3_dual_qmv4_matmul,
    multi3_parallel_dual_qmv4_matmul,
    multi3_qmv4_matmul,
    multi3_swiglu_down_qmv4_matmul,
)


def _dtype(name: str) -> mx.Dtype:
    normalized = name.lower().strip()
    if normalized in {"bf16", "bfloat16"}:
        return mx.bfloat16
    if normalized in {"fp16", "float16"}:
        return mx.float16
    raise ValueError(f"unsupported dtype for multi-qmv probe: {name}")


def _time_call(fn, *, repeats: int, warmup: int) -> tuple[list[float], Any]:
    result = None
    for _ in range(warmup):
        result = fn()
        mx.eval(result)
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        mx.eval(result)
        samples.append(time.perf_counter() - started)
    return samples, result


def run_multi_qmv_probe(
    model_path: Path | str,
    *,
    include: str = "mlp",
    repeats: int = 10,
    warmup: int = 3,
    dtype: str = "bf16",
    seed: int = 0,
    mtp: bool = True,
) -> dict[str, Any]:
    mx.random.seed(seed)
    rt = load(model_path, mtp=mtp)
    dtype_value = _dtype(dtype)
    rows: list[dict[str, Any]] = []
    groups = collect_quantized_linear_groups(rt.model, include=include)
    mlp_gate_group = next((group for group in groups if group.kind == "mlp_gate"), None)
    mlp_up_group = next((group for group in groups if group.kind == "mlp_up"), None)

    for group in groups:
        module = group.module
        x = mx.random.normal((1, 3, group.k), dtype=dtype_value)
        mx.eval(x)

        stock_samples, stock = _time_call(lambda: module(x), repeats=repeats, warmup=warmup)
        eligible = is_multi3_qmv4_eligible(module)
        custom_samples: list[float] = []
        custom = None
        max_abs_diff = None
        if eligible:
            custom_samples, custom = _time_call(
                lambda: multi3_qmv4_matmul(x, module),
                repeats=repeats,
                warmup=warmup,
            )
            diff = mx.max(mx.abs(stock.astype(mx.float32) - custom.astype(mx.float32)))
            mx.eval(diff)
            max_abs_diff = float(diff.item())

        stock_mean = statistics.mean(stock_samples)
        custom_mean = statistics.mean(custom_samples) if custom_samples else None
        rows.append(
            {
                "kind": group.kind,
                "sample_path": group.sample_path,
                "count": len(group.paths),
                "k": group.k,
                "n": group.n,
                "bits": group.bits,
                "group_size": group.group_size,
                "mode": group.mode,
                "eligible": eligible,
                "stock_samples_s": stock_samples,
                "stock_mean_s": stock_mean,
                "stock_estimated_full_forward_mean_s": stock_mean * len(group.paths),
                "multi3_qmv_samples_s": custom_samples,
                "multi3_qmv_mean_s": custom_mean,
                "multi3_qmv_estimated_full_forward_mean_s": (
                    custom_mean * len(group.paths) if custom_mean is not None else None
                ),
                "speedup": stock_mean / custom_mean if custom_mean else None,
                "max_abs_diff": max_abs_diff,
                "kernel_tolerance_pass": (
                    max_abs_diff is not None and max_abs_diff <= 8e-3
                ),
                "exact_gate_pass": (
                    max_abs_diff is not None and max_abs_diff <= 1e-3
                ),
            }
        )

        if group.kind == "mlp_down" and eligible:
            gate = mx.random.normal((1, 3, group.k), dtype=dtype_value)
            up = mx.random.normal((1, 3, group.k), dtype=dtype_value)
            mx.eval(gate, up)
            materialized_samples, materialized = _time_call(
                lambda: module(nn.silu(gate) * up),
                repeats=repeats,
                warmup=warmup,
            )
            fused_samples, fused = _time_call(
                lambda: multi3_swiglu_down_qmv4_matmul(gate, up, module),
                repeats=repeats,
                warmup=warmup,
            )
            diff = mx.max(
                mx.abs(materialized.astype(mx.float32) - fused.astype(mx.float32))
            )
            mx.eval(diff)
            max_abs_diff = float(diff.item())
            materialized_mean = statistics.mean(materialized_samples)
            fused_mean = statistics.mean(fused_samples)
            rows.append(
                {
                    "kind": "mlp_swiglu_down",
                    "sample_path": group.sample_path,
                    "count": len(group.paths),
                    "k": group.k,
                    "n": group.n,
                    "bits": group.bits,
                    "group_size": group.group_size,
                    "mode": group.mode,
                    "eligible": True,
                    "stock_samples_s": materialized_samples,
                    "stock_mean_s": materialized_mean,
                    "stock_estimated_full_forward_mean_s": (
                        materialized_mean * len(group.paths)
                    ),
                    "multi3_qmv_samples_s": fused_samples,
                    "multi3_qmv_mean_s": fused_mean,
                    "multi3_qmv_estimated_full_forward_mean_s": (
                        fused_mean * len(group.paths)
                    ),
                    "speedup": materialized_mean / fused_mean,
                    "max_abs_diff": max_abs_diff,
                    "kernel_tolerance_pass": max_abs_diff <= 8e-3,
                    "exact_gate_pass": max_abs_diff <= 1e-3,
                    "comparison": "module(nn.silu(gate) * up)",
                }
            )

    if (
        mlp_gate_group is not None
        and mlp_up_group is not None
        and is_multi3_qmv4_eligible(mlp_gate_group.module)
        and is_multi3_qmv4_eligible(mlp_up_group.module)
        and mlp_gate_group.k == mlp_up_group.k
        and mlp_gate_group.n == mlp_up_group.n
    ):
        x = mx.random.normal((1, 3, mlp_gate_group.k), dtype=dtype_value)
        mx.eval(x)
        separate_samples, separate = _time_call(
            lambda: (mlp_gate_group.module(x), mlp_up_group.module(x)),
            repeats=repeats,
            warmup=warmup,
        )
        dual_samples, dual = _time_call(
            lambda: multi3_dual_qmv4_matmul(
                x, mlp_gate_group.module, mlp_up_group.module
            ),
            repeats=repeats,
            warmup=warmup,
        )
        sep_gate, sep_up = separate
        dual_gate, dual_up = dual
        gate_diff = mx.max(mx.abs(sep_gate.astype(mx.float32) - dual_gate.astype(mx.float32)))
        up_diff = mx.max(mx.abs(sep_up.astype(mx.float32) - dual_up.astype(mx.float32)))
        mx.eval(gate_diff, up_diff)
        max_abs_diff = max(float(gate_diff.item()), float(up_diff.item()))
        separate_mean = statistics.mean(separate_samples)
        dual_mean = statistics.mean(dual_samples)
        pair_count = min(len(mlp_gate_group.paths), len(mlp_up_group.paths))
        rows.append(
            {
                "kind": "mlp_gate_up_dual",
                "sample_path": f"{mlp_gate_group.sample_path} + {mlp_up_group.sample_path}",
                "count": pair_count,
                "k": mlp_gate_group.k,
                "n": mlp_gate_group.n,
                "bits": mlp_gate_group.bits,
                "group_size": mlp_gate_group.group_size,
                "mode": mlp_gate_group.mode,
                "eligible": True,
                "stock_samples_s": separate_samples,
                "stock_mean_s": separate_mean,
                "stock_estimated_full_forward_mean_s": separate_mean * pair_count,
                "multi3_qmv_samples_s": dual_samples,
                "multi3_qmv_mean_s": dual_mean,
                "multi3_qmv_estimated_full_forward_mean_s": dual_mean * pair_count,
                "speedup": separate_mean / dual_mean,
                "max_abs_diff": max_abs_diff,
                "kernel_tolerance_pass": max_abs_diff <= 8e-3,
                "exact_gate_pass": max_abs_diff <= 1e-3,
                "comparison": "gate(x), up(x)",
            }
        )
        parallel_dual_samples, parallel_dual = _time_call(
            lambda: multi3_parallel_dual_qmv4_matmul(
                x, mlp_gate_group.module, mlp_up_group.module
            ),
            repeats=repeats,
            warmup=warmup,
        )
        parallel_gate, parallel_up = parallel_dual
        gate_diff = mx.max(mx.abs(sep_gate.astype(mx.float32) - parallel_gate.astype(mx.float32)))
        up_diff = mx.max(mx.abs(sep_up.astype(mx.float32) - parallel_up.astype(mx.float32)))
        mx.eval(gate_diff, up_diff)
        max_abs_diff = max(float(gate_diff.item()), float(up_diff.item()))
        parallel_dual_mean = statistics.mean(parallel_dual_samples)
        rows.append(
            {
                "kind": "mlp_gate_up_parallel_dual",
                "sample_path": f"{mlp_gate_group.sample_path} + {mlp_up_group.sample_path}",
                "count": pair_count,
                "k": mlp_gate_group.k,
                "n": mlp_gate_group.n,
                "bits": mlp_gate_group.bits,
                "group_size": mlp_gate_group.group_size,
                "mode": mlp_gate_group.mode,
                "eligible": True,
                "stock_samples_s": separate_samples,
                "stock_mean_s": separate_mean,
                "stock_estimated_full_forward_mean_s": separate_mean * pair_count,
                "multi3_qmv_samples_s": parallel_dual_samples,
                "multi3_qmv_mean_s": parallel_dual_mean,
                "multi3_qmv_estimated_full_forward_mean_s": (
                    parallel_dual_mean * pair_count
                ),
                "speedup": separate_mean / parallel_dual_mean,
                "max_abs_diff": max_abs_diff,
                "kernel_tolerance_pass": max_abs_diff <= 8e-3,
                "exact_gate_pass": max_abs_diff <= 1e-3,
                "comparison": "gate(x), up(x)",
            }
        )

    rows.sort(key=lambda row: (row["speedup"] or 0.0), reverse=True)
    return {
        "model": str(model_path),
        "dtype": dtype,
        "m": 3,
        "include": include,
        "repeats": repeats,
        "warmup": warmup,
        "rows": rows,
        "summary": {
            "eligible_groups": sum(1 for row in rows if row["eligible"]),
            "kernel_tolerance_failures": sum(
                1 for row in rows if row["eligible"] and not row["kernel_tolerance_pass"]
            ),
            "exact_gate_failures": sum(
                1 for row in rows if row["eligible"] and not row["exact_gate_pass"]
            ),
        },
    }


def write_multi_qmv_probe(path: Path | str, result: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
