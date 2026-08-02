"""Temperature MTP-1 smoke runner."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from optiq.runtime.mtp.benchmarks.schema import encode_prompt_case, load_prompt_suite
from optiq.runtime.mtp.benchmarks.validators.basic import (
    validate_json_text,
    validate_no_degenerate_loop,
)
from optiq.runtime.mtp.generation import generate_ar, generate_mtp1
from optiq.runtime.mtp.mtp_patch import MTPContract
from optiq.runtime.mtp.runtime import load
from optiq.runtime.mtp.sampling import SamplerConfig


def run_mtp1_sampler_smoke(
    model_path: Path | str,
    prompt_suite: Path | str,
    *,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    draft_temperature: float | None = None,
    draft_top_p: float | None = None,
    draft_top_k: int | None = None,
    max_tokens: int = 96,
    seed: int = 0,
    limit: int | None = None,
    enable_thinking: bool | None = None,
    compare_ar: bool = False,
    verify_strategy: str = "batched",
    verify_core: str = "stock",
    draft_margin_threshold: float | None = None,
    mtp_quant_bits: int | None = None,
    mtp_quant_group_size: int = 64,
    mtp_quant_mode: str = "affine",
) -> dict[str, Any]:
    rt = load(
        model_path,
        mtp=True,
        contract=MTPContract(
            mtp_quant_bits=mtp_quant_bits,
            mtp_quant_group_size=mtp_quant_group_size,
            mtp_quant_mode=mtp_quant_mode,
        ),
    )
    prompts = load_prompt_suite(prompt_suite)
    if limit is not None:
        prompts = prompts[:limit]
    sampler = SamplerConfig(temperature=temperature, top_p=top_p, top_k=top_k)
    draft_sampler = SamplerConfig(
        temperature=temperature if draft_temperature is None else draft_temperature,
        top_p=top_p if draft_top_p is None else draft_top_p,
        top_k=top_k if draft_top_k is None else draft_top_k,
    )

    rows = []
    for index, case in enumerate(prompts):
        ids = encode_prompt_case(
            rt.tokenizer,
            case,
            chat_template=True,
            enable_thinking=enable_thinking,
        )
        ar = None
        if compare_ar:
            ar = generate_ar(
                rt,
                ids,
                max_tokens=min(max_tokens, case.max_tokens),
                sampler=sampler,
                seed=seed + index,
            )
        out = generate_mtp1(
            rt,
            ids,
            max_tokens=min(max_tokens, case.max_tokens),
            sampler=sampler,
            seed=seed + index,
            draft_sampler=draft_sampler,
            verify_strategy=verify_strategy,
            verify_core=verify_core,
            draft_margin_threshold=draft_margin_threshold,
        )
        validations = [asdict(validate_no_degenerate_loop(out.text))]
        if case.category == "json_tool":
            validations.append(asdict(validate_json_text(out.text.strip())))
        rows.append(
            {
                "prompt_id": case.id,
                "category": case.category,
                "prompt_sha256": case.prompt_sha256,
                "generated_tokens": out.stats.generated_tokens,
                "tok_s": out.stats.tok_s,
                "ar_tok_s": ar.stats.tok_s if ar is not None else None,
                "speedup_vs_ar": (
                    out.stats.tok_s / ar.stats.tok_s
                    if ar is not None and ar.stats.tok_s
                    else None
                ),
                "accepted_drafts": out.stats.accepted_drafts,
                "rejected_drafts": out.stats.rejected_drafts,
                "drafted_tokens": out.stats.drafted_tokens,
                "skipped_drafts": out.stats.skipped_drafts,
                "acceptance_rate": (
                    out.stats.accepted_drafts / out.stats.drafted_tokens
                    if out.stats.drafted_tokens
                    else None
                ),
                "verify_time_s": out.stats.verify_time_s,
                "draft_time_s": out.stats.draft_time_s,
                "target_forward_time_s": out.stats.target_forward_time_s,
                "snapshot_time_s": out.stats.snapshot_time_s,
                "accept_time_s": out.stats.accept_time_s,
                "rollback_time_s": out.stats.rollback_time_s,
                "repair_time_s": out.stats.repair_time_s,
                "commit_time_s": out.stats.commit_time_s,
                "capture_commit_time_s": out.stats.capture_commit_time_s,
                "bonus_time_s": out.stats.bonus_time_s,
                "bonus_tokens": out.stats.bonus_tokens,
                "correction_tokens": out.stats.correction_tokens,
                "deferred_correction_repairs": out.stats.deferred_correction_repairs,
                "verify_calls": out.stats.verify_calls,
                "graphbank": out.stats.graphbank,
                "peak_memory_bytes": out.stats.peak_memory_bytes,
                "validations": validations,
                "text": out.text,
                "events": out.stats.events,
            }
        )

    drafted = sum(r["drafted_tokens"] for r in rows)
    accepted = sum(r["accepted_drafts"] for r in rows)
    rejected = sum(r["rejected_drafts"] for r in rows)
    skipped = sum(r["skipped_drafts"] for r in rows)
    validations = [v for r in rows for v in r["validations"]]
    return {
        "model_path": str(model_path),
        "prompt_suite": str(prompt_suite),
        "sampler": asdict(sampler),
        "draft_sampler": asdict(draft_sampler),
        "max_tokens": max_tokens,
        "seed": seed,
        "enable_thinking": enable_thinking,
        "compare_ar": compare_ar,
        "verify_strategy": verify_strategy,
        "verify_core": verify_core,
        "draft_margin_threshold": draft_margin_threshold,
        "mtp_quant_bits": rt.contract.mtp_quant_bits,
        "mtp_quant_group_size": rt.contract.mtp_quant_group_size,
        "mtp_quant_mode": rt.contract.mtp_quant_mode,
        "mtp_quant_policy": rt.contract.mtp_quant_policy,
        "rows": rows,
        "summary": {
            "prompts": len(rows),
            "generated_tokens": sum(r["generated_tokens"] for r in rows),
            "accepted_drafts": accepted,
            "rejected_drafts": rejected,
            "drafted_tokens": drafted,
            "skipped_drafts": skipped,
            "acceptance_rate": accepted / drafted if drafted else None,
            "mean_tok_s": statistics.mean([r["tok_s"] for r in rows]) if rows else 0.0,
            "mean_ar_tok_s": (
                statistics.mean([r["ar_tok_s"] for r in rows if r["ar_tok_s"] is not None])
                if compare_ar and rows
                else None
            ),
            "mean_speedup_vs_ar": (
                statistics.mean(
                    [r["speedup_vs_ar"] for r in rows if r["speedup_vs_ar"] is not None]
                )
                if compare_ar and rows
                else None
            ),
            "verify_time_s": sum(r["verify_time_s"] for r in rows),
            "draft_time_s": sum(r["draft_time_s"] for r in rows),
            "target_forward_time_s": sum(r["target_forward_time_s"] for r in rows),
            "snapshot_time_s": sum(r["snapshot_time_s"] for r in rows),
            "accept_time_s": sum(r["accept_time_s"] for r in rows),
            "rollback_time_s": sum(r["rollback_time_s"] for r in rows),
            "repair_time_s": sum(r["repair_time_s"] for r in rows),
            "commit_time_s": sum(r["commit_time_s"] for r in rows),
            "capture_commit_time_s": sum(r["capture_commit_time_s"] for r in rows),
            "bonus_time_s": sum(r["bonus_time_s"] for r in rows),
            "bonus_tokens": sum(r["bonus_tokens"] for r in rows),
            "correction_tokens": sum(r["correction_tokens"] for r in rows),
            "deferred_correction_repairs": sum(r["deferred_correction_repairs"] for r in rows),
            "verify_calls": sum(r["verify_calls"] for r in rows),
            "validations_passed": sum(1 for v in validations if v["passed"]),
            "validations_total": len(validations),
            "peak_memory_bytes": max([r["peak_memory_bytes"] for r in rows] or [0]),
        },
    }


def write_sampler_smoke(path: Path | str, result: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
