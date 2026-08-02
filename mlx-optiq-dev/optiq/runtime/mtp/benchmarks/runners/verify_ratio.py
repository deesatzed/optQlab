"""Decode verify-ratio microbenchmark."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from optiq.runtime.mtp.runtime import load


def _candidate_tokens(tokenizer: Any, min_len: int) -> list[int]:
    text = (
        "    result = a + b\n"
        "    return result\n"
        "\n"
        "def subtract(a, b):\n"
        "    return a - b\n"
    )
    ids = tokenizer.encode(text)
    while len(ids) < min_len:
        ids.extend(ids)
    return ids[:min_len]


def run_verify_ratio(
    model_path: Path | str,
    prompt: str,
    *,
    max_k: int = 8,
    repeats: int = 3,
) -> dict[str, Any]:
    """Measure cached forward(S tokens) / cached forward(1 token)."""
    rt = load(model_path, mtp=False)
    prompt_ids = rt.tokenizer.encode(prompt)
    candidates = _candidate_tokens(rt.tokenizer, max_k + 1)
    rows = []

    for length in range(1, max_k + 2):
        timings: list[float] = []
        for repeat in range(repeats + 1):
            cache = rt.make_cache()
            prefill = rt.forward_ar(mx.array([prompt_ids]), cache=cache, return_hidden=False)
            mx.eval(prefill)
            decode = mx.array([candidates[:length]])
            started = time.perf_counter()
            out = rt.forward_ar(decode, cache=cache, return_hidden=False)
            mx.eval(out)
            elapsed = time.perf_counter() - started
            if repeat > 0:
                timings.append(elapsed)
        rows.append(
            {
                "tokens": length,
                "mean_s": statistics.mean(timings),
                "min_s": min(timings),
                "max_s": max(timings),
                "repeats": repeats,
            }
        )

    baseline = rows[0]["mean_s"]
    for row in rows:
        row["ratio_vs_1"] = row["mean_s"] / baseline if baseline else None

    return {
        "model_path": str(model_path),
        "prompt_tokens": len(prompt_ids),
        "max_k": max_k,
        "repeats": repeats,
        "rows": rows,
        "verify_ratio_k5": rows[5]["ratio_vs_1"] if len(rows) > 5 else None,
    }
