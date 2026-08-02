"""Benchmark harness entrypoints."""

from __future__ import annotations

import time
from pathlib import Path

from optiq.runtime.mtp.benchmarks.schema import (
    BenchmarkConfig,
    BenchmarkRecord,
    empty_record,
    load_prompt_suite,
    write_jsonl,
)


def run_manifest_only(
    prompt_suite: Path | str,
    config: BenchmarkConfig,
    output_jsonl: Path | str,
) -> list[BenchmarkRecord]:
    """Record prompt/sampler metadata without inference.

    This is a real harness smoke test, not a speed benchmark. It proves the
    benchmark output shape before model loading enters the loop.
    """
    records = []
    started = time.perf_counter()
    for case in load_prompt_suite(prompt_suite):
        record = empty_record(case, config)
        record.elapsed_s = time.perf_counter() - started
        record.error = "manifest_only_no_inference"
        records.append(record)
    write_jsonl(output_jsonl, records)
    return records
