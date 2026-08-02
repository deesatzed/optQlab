"""Adaptive-depth native MTP runner."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from optiq.runtime.mtp.benchmarks.runners.mtp_depth_sweep import _rate_by_depth, _sum_lists
from optiq.runtime.mtp.benchmarks.schema import encode_prompt_case, load_prompt_suite
from optiq.runtime.mtp.benchmarks.validators.basic import (
    validate_json_text,
    validate_no_degenerate_loop,
)
from optiq.runtime.mtp.adaptive import AdaptiveDepthPolicy, ExpectedValueDepthPolicy
from optiq.runtime.mtp.generation import generate_ar, generate_mtpk
from optiq.runtime.mtp.mtp_patch import MTPContract
from optiq.runtime.mtp.runtime import load
from optiq.runtime.mtp.sampling import SamplerConfig


def run_mtp_adaptive(
    model_path: Path | str,
    prompt_suite: Path | str,
    *,
    max_depth: int = 5,
    min_depth: int = 1,
    start_depth: int = 1,
    increase_after: int = 4,
    decrease_after: int = 1,
    policy_kind: str = "streak",
    ev_base_depth: int = 2,
    ev_accept_priors: tuple[float, ...] = (0.92, 0.64, 0.32),
    ev_draft_cost_s: float = 0.0048,
    ev_extra_verify_cost_s: float = 0.0060,
    ev_baseline_tok_s: float = 40.0,
    ev_safety_margin: float = 0.10,
    ev_margin_center: float = 1.0,
    ev_margin_scale: float = 2.0,
    ev_confidence_weight: float = 0.35,
    ev_min_extra_accept_probability: float = 0.18,
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
    mtp_hidden_variant: str = "post_norm",
    mtp_cache_policy: str = "persistent",
    mtp_history_policy: str = "cycle",
    verify_strategy: str = "batched",
    verify_core: str = "stock",
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
    prompt_cases = load_prompt_suite(prompt_suite)
    if limit is not None:
        prompt_cases = prompt_cases[:limit]
    sampler = SamplerConfig(temperature=temperature, top_p=top_p, top_k=top_k)
    draft_sampler = SamplerConfig(
        temperature=temperature if draft_temperature is None else draft_temperature,
        top_p=top_p if draft_top_p is None else draft_top_p,
        top_k=top_k if draft_top_k is None else draft_top_k,
    )

    encoded = [
        (
            case,
            encode_prompt_case(
                rt.tokenizer,
                case,
                chat_template=True,
                enable_thinking=enable_thinking,
            ),
        )
        for case in prompt_cases
    ]

    ar_rows: list[dict[str, Any]] = []
    if compare_ar:
        for index, (case, ids) in enumerate(encoded):
            ar = generate_ar(
                rt,
                ids,
                max_tokens=min(max_tokens, case.max_tokens),
                sampler=sampler,
                seed=seed + index,
            )
            ar_rows.append(
                {
                    "prompt_id": case.id,
                    "category": case.category,
                    "generated_tokens": ar.stats.generated_tokens,
                    "tok_s": ar.stats.tok_s,
                    "tokens": ar.tokens,
                    "text": ar.text,
                }
            )

    rows = []
    for index, (case, ids) in enumerate(encoded):
        if policy_kind == "expected_value":
            policy = ExpectedValueDepthPolicy(
                max_depth=max_depth,
                min_depth=min_depth,
                base_depth=ev_base_depth,
                accept_priors=ev_accept_priors,
                draft_cost_s=ev_draft_cost_s,
                extra_verify_cost_s=ev_extra_verify_cost_s,
                baseline_tok_s=ev_baseline_tok_s,
                safety_margin=ev_safety_margin,
                margin_center=ev_margin_center,
                margin_scale=ev_margin_scale,
                confidence_weight=ev_confidence_weight,
                min_extra_accept_probability=ev_min_extra_accept_probability,
            )
        elif policy_kind == "streak":
            policy = AdaptiveDepthPolicy(
                max_depth=max_depth,
                min_depth=min_depth,
                start_depth=start_depth,
                increase_after=increase_after,
                decrease_after=decrease_after,
            )
        else:
            raise ValueError("policy_kind must be 'streak' or 'expected_value'")
        out = generate_mtpk(
            rt,
            ids,
            max_tokens=min(max_tokens, case.max_tokens),
            sampler=sampler,
            speculative_depth=max_depth,
            seed=seed + index,
            mtp_hidden_variant=mtp_hidden_variant,
            mtp_cache_policy=mtp_cache_policy,
            mtp_history_policy=mtp_history_policy,
            draft_sampler=draft_sampler,
            verify_strategy=verify_strategy,
            verify_core=verify_core,
            adaptive_policy=policy,
        )
        validations = [asdict(validate_no_degenerate_loop(out.text))]
        if case.category == "json_tool":
            validations.append(asdict(validate_json_text(out.text.strip())))
        ar_row = ar_rows[index] if compare_ar else None
        rows.append(
            {
                "prompt_id": case.id,
                "category": case.category,
                "prompt_sha256": case.prompt_sha256,
                "generated_tokens": out.stats.generated_tokens,
                "tok_s": out.stats.tok_s,
                "ar_tok_s": ar_row["tok_s"] if ar_row is not None else None,
                "exact_match": (
                    out.tokens == ar_row["tokens"]
                    if ar_row is not None and temperature <= 0
                    else None
                ),
                "tokens": out.tokens,
                "speedup_vs_ar": (
                    out.stats.tok_s / ar_row["tok_s"]
                    if ar_row is not None and ar_row["tok_s"]
                    else None
                ),
                "accepted_drafts": out.stats.accepted_drafts,
                "rejected_drafts": out.stats.rejected_drafts,
                "drafted_tokens": out.stats.drafted_tokens,
                "accepted_by_depth": out.stats.accepted_by_depth,
                "drafted_by_depth": out.stats.drafted_by_depth,
                "acceptance_by_depth": _rate_by_depth(
                    out.stats.accepted_by_depth,
                    out.stats.drafted_by_depth,
                ),
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
                "verify_calls": out.stats.verify_calls,
                "peak_memory_bytes": out.stats.peak_memory_bytes,
                "validations": validations,
                "text": out.text,
                "events": out.stats.events,
            }
        )

    validations = [v for row in rows for v in row["validations"]]
    accepted_by_depth = _sum_lists([row["accepted_by_depth"] for row in rows], max_depth)
    drafted_by_depth = _sum_lists([row["drafted_by_depth"] for row in rows], max_depth)
    return {
        "model_path": str(model_path),
        "prompt_suite": str(prompt_suite),
        "sampler": asdict(sampler),
        "draft_sampler": asdict(draft_sampler),
        "max_tokens": max_tokens,
        "seed": seed,
        "enable_thinking": enable_thinking,
        "compare_ar": compare_ar,
        "policy_kind": policy_kind,
        "mtp_hidden_variant": mtp_hidden_variant,
        "mtp_cache_policy": mtp_cache_policy,
        "mtp_history_policy": mtp_history_policy,
        "verify_strategy": verify_strategy,
        "verify_core": verify_core,
        "mtp_quant_bits": mtp_quant_bits,
        "mtp_quant_group_size": mtp_quant_group_size,
        "mtp_quant_mode": mtp_quant_mode,
        "policy": {
            "kind": policy_kind,
            "max_depth": max_depth,
            "min_depth": min_depth,
            "start_depth": start_depth,
            "increase_after": increase_after,
            "decrease_after": decrease_after,
            "ev_base_depth": ev_base_depth,
            "ev_accept_priors": list(ev_accept_priors),
            "ev_draft_cost_s": ev_draft_cost_s,
            "ev_extra_verify_cost_s": ev_extra_verify_cost_s,
            "ev_baseline_tok_s": ev_baseline_tok_s,
            "ev_safety_margin": ev_safety_margin,
            "ev_margin_center": ev_margin_center,
            "ev_margin_scale": ev_margin_scale,
            "ev_confidence_weight": ev_confidence_weight,
            "ev_min_extra_accept_probability": ev_min_extra_accept_probability,
        },
        "ar_rows": ar_rows,
        "rows": rows,
        "summary": {
            "prompts": len(rows),
            "generated_tokens": sum(row["generated_tokens"] for row in rows),
            "accepted_drafts": sum(row["accepted_drafts"] for row in rows),
            "rejected_drafts": sum(row["rejected_drafts"] for row in rows),
            "drafted_tokens": sum(row["drafted_tokens"] for row in rows),
            "accepted_by_depth": accepted_by_depth,
            "drafted_by_depth": drafted_by_depth,
            "acceptance_by_depth": _rate_by_depth(accepted_by_depth, drafted_by_depth),
            "mean_tok_s": statistics.mean([row["tok_s"] for row in rows]) if rows else 0.0,
            "mean_ar_tok_s": (
                statistics.mean([row["ar_tok_s"] for row in rows if row["ar_tok_s"] is not None])
                if compare_ar and rows
                else None
            ),
            "mean_speedup_vs_ar": (
                statistics.mean([row["speedup_vs_ar"] for row in rows if row["speedup_vs_ar"] is not None])
                if compare_ar and rows
                else None
            ),
            "verify_time_s": sum(row["verify_time_s"] for row in rows),
            "draft_time_s": sum(row["draft_time_s"] for row in rows),
            "target_forward_time_s": sum(row["target_forward_time_s"] for row in rows),
            "snapshot_time_s": sum(row["snapshot_time_s"] for row in rows),
            "accept_time_s": sum(row["accept_time_s"] for row in rows),
            "rollback_time_s": sum(row["rollback_time_s"] for row in rows),
            "repair_time_s": sum(row["repair_time_s"] for row in rows),
            "commit_time_s": sum(row["commit_time_s"] for row in rows),
            "capture_commit_time_s": sum(row["capture_commit_time_s"] for row in rows),
            "bonus_time_s": sum(row["bonus_time_s"] for row in rows),
            "bonus_tokens": sum(row["bonus_tokens"] for row in rows),
            "correction_tokens": sum(row["correction_tokens"] for row in rows),
            "verify_calls": sum(row["verify_calls"] for row in rows),
            "validations_passed": sum(1 for v in validations if v["passed"]),
            "validations_total": len(validations),
            "exact_matches": (
                sum(1 for row in rows if row["exact_match"])
                if compare_ar and temperature <= 0
                else None
            ),
            "exact_total": len(rows) if compare_ar and temperature <= 0 else None,
            "peak_memory_bytes": max([row["peak_memory_bytes"] for row in rows] or [0]),
        },
    }


def write_adaptive(path: Path | str, result: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
