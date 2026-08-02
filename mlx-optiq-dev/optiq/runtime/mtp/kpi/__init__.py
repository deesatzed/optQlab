"""Canonical KPI helpers for MTPLX CLI runs."""

from .runtime_kpis import (
    EXIT_EXACTNESS,
    EXIT_QUALITY,
    EXIT_STRICT_GATE,
    EXIT_TELEMETRY,
    EXIT_UNSUPPORTED_MODEL,
    EXACT_PAGED_ATTENTION_ENV,
    NATIVE_MTP_FAST_PATH_ENV,
    build_benchmark_envelope,
    default_output_path,
    exact_paged_attention_env,
    prompt_suite_path,
    public_bench_runtime_profile_env,
    run_exactness_smoke,
    summarize_decode_trace,
    write_json,
)
from .reference_vllm import summarize_vllm_reference

__all__ = [
    "EXIT_EXACTNESS",
    "EXIT_QUALITY",
    "EXIT_STRICT_GATE",
    "EXIT_TELEMETRY",
    "EXIT_UNSUPPORTED_MODEL",
    "EXACT_PAGED_ATTENTION_ENV",
    "NATIVE_MTP_FAST_PATH_ENV",
    "build_benchmark_envelope",
    "default_output_path",
    "exact_paged_attention_env",
    "prompt_suite_path",
    "public_bench_runtime_profile_env",
    "run_exactness_smoke",
    "summarize_decode_trace",
    "summarize_vllm_reference",
    "write_json",
]
