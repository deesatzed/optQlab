"""vLLM/Nsight reference parsing for the public benchmark CLI."""

from __future__ import annotations

import json
from typing import Any


def _parse_nsys_summary_rows(text: str) -> list[dict[str, Any]]:
    """Parse common ``nsys stats`` summary rows.

    Nsight emits a human table, not a stable machine schema. Keep the parser
    conservative: only accept rows whose first three columns are time percent,
    total time, and count/instances/calls.
    """

    rows: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=8)
        if len(parts) < 8:
            continue
        try:
            time_percent = float(parts[0])
            total_time_ns = int(parts[1])
            instances = int(parts[2])
            avg_ns = float(parts[3])
        except ValueError:
            continue
        rows.append(
            {
                "name": parts[8] if len(parts) > 8 else "",
                "time_percent": time_percent,
                "total_time_ns": total_time_ns,
                "count": instances,
                "avg_ns": avg_ns,
            }
        )
    return rows


def parse_cuda_kernel_summary(text: str) -> dict[str, Any]:
    """Parse ``nsys stats --report cuda_gpu_kern_sum`` text."""

    kernels = _parse_nsys_summary_rows(text)
    for row in kernels:
        row["instances"] = row["count"]
    total_instances = sum(int(row["instances"]) for row in kernels)
    return {
        "kernel_types": len(kernels),
        "total_kernel_instances": total_instances,
        "top_kernels": kernels[:15],
    }


def parse_cuda_api_summary(text: str) -> dict[str, Any]:
    """Parse ``nsys stats --report cuda_api_sum`` text.

    This is closer to host-side submission pressure than kernel instances, but
    it still includes syncs, event calls, and allocation APIs. Keep launch-like
    calls broken out instead of pretending every CUDA API call is a GPU launch.
    """

    calls = _parse_nsys_summary_rows(text)
    total_api_calls = sum(int(row["count"]) for row in calls)
    launch_like_names = ("cudaGraphLaunch", "cudaLaunchKernel")
    launch_like_calls = sum(
        int(row["count"])
        for row in calls
        if any(name in str(row.get("name", "")) for name in launch_like_names)
    )
    return {
        "api_types": len(calls),
        "total_api_calls": total_api_calls,
        "launch_like_api_calls": launch_like_calls,
        "top_api_calls": calls[:15],
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _generated_tokens_from_row(row: dict[str, Any]) -> int | None:
    for key in (
        "generated_tokens",
        "completion_tokens",
        "output_tokens",
        "num_generated_tokens",
        "tokens_generated",
    ):
        value = _number(row.get(key))
        if value is not None:
            return int(value)
    usage = row.get("usage")
    if isinstance(usage, dict):
        for key in ("completion_tokens", "output_tokens"):
            value = _number(usage.get(key))
            if value is not None:
                return int(value)
    return None


def parse_bench_json(text: str | None) -> dict[str, Any]:
    if not text:
        return {"available": False}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"available": False, "error": str(exc)}

    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]]
    if isinstance(data, list):
        rows = [row for row in data if isinstance(row, dict)]
    elif isinstance(data, dict):
        if isinstance(data.get("summary"), dict):
            summary = data["summary"]
        if isinstance(data.get("rows"), list):
            rows = [row for row in data["rows"] if isinstance(row, dict)]
        elif isinstance(data.get("results"), list):
            rows = [row for row in data["results"] if isinstance(row, dict)]
        else:
            rows = [data]
    else:
        rows = []

    generated = sum(token for row in rows if (token := _generated_tokens_from_row(row)) is not None)
    decode_tok_s_values = [
        float(value)
        for row in rows
        for key in ("decode_tok_s", "tok_s", "tokens_per_second", "output_tok_s")
        if (value := _number(row.get(key))) is not None
    ]
    end_to_end_values = [
        float(value)
        for row in rows
        for key in ("end_to_end_tok_s", "overall_tok_s")
        if (value := _number(row.get(key))) is not None
    ]
    summary_decode = _number(summary.get("mean_decode_tok_s"))
    summary_end_to_end = _number(summary.get("mean_end_to_end_tok_s"))
    return {
        "available": True,
        "rows": len(rows),
        "generated_tokens": generated or None,
        "decode_tok_s_values": decode_tok_s_values,
        "end_to_end_tok_s_values": end_to_end_values,
        "mean_decode_tok_s": (
            summary_decode
            if summary_decode is not None
            else (sum(decode_tok_s_values) / len(decode_tok_s_values) if decode_tok_s_values else None)
        ),
        "mean_end_to_end_tok_s": (
            summary_end_to_end
            if summary_end_to_end is not None
            else (sum(end_to_end_values) / len(end_to_end_values) if end_to_end_values else None)
        ),
    }


def summarize_vllm_reference(
    *,
    cuda_kernel_summary_text: str | None,
    cuda_api_summary_text: str | None = None,
    bench_json_text: str | None = None,
) -> dict[str, Any]:
    kernel_summary = parse_cuda_kernel_summary(cuda_kernel_summary_text or "")
    api_summary = parse_cuda_api_summary(cuda_api_summary_text or "")
    bench = parse_bench_json(bench_json_text)
    generated_tokens = bench.get("generated_tokens")
    total_instances = kernel_summary.get("total_kernel_instances") or 0
    launch_like_api_calls = api_summary.get("launch_like_api_calls") or 0
    launches_per_token = (
        float(total_instances) / float(generated_tokens)
        if generated_tokens and total_instances
        else None
    )
    api_launches_per_token = (
        float(launch_like_api_calls) / float(generated_tokens)
        if generated_tokens and launch_like_api_calls
        else None
    )
    command_buffer_basis = (
        api_launches_per_token if api_launches_per_token is not None else launches_per_token
    )
    return {
        "cuda_kernel_summary_available": bool(cuda_kernel_summary_text),
        "cuda_api_summary_available": bool(cuda_api_summary_text),
        "kernel_summary": kernel_summary,
        "api_summary": api_summary,
        "bench": bench,
        "kernel_launches_per_generated_token": launches_per_token,
        "launch_like_cuda_api_calls_per_generated_token": api_launches_per_token,
        "promotion_target": {
            "mtplx_initial_max_multiplier": 8.0,
            "mtplx_kernel_instances_per_token_target": (
                launches_per_token * 8.0 if launches_per_token is not None else None
            ),
            "mtplx_command_buffers_per_token_target": (
                command_buffer_basis * 8.0
                if command_buffer_basis is not None
                else None
            ),
        },
    }
