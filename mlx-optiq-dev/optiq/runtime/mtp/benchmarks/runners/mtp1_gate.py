"""MTP-1 correctness and smoke benchmark gates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from optiq.runtime.mtp.benchmarks.schema import encode_prompt_case, load_prompt_suite
from optiq.runtime.mtp.generation import generate_ar, generate_mtp1
from optiq.runtime.mtp.mtp_patch import MTPContract
from optiq.runtime.mtp.runtime import load
from optiq.runtime.mtp.sampling import SamplerConfig


@dataclass
class MTP1GateRow:
    prompt_id: str
    category: str
    prompt_sha256: str
    max_tokens: int
    exact_match: bool
    ar_tokens: list[int]
    mtp_tokens: list[int]
    ar_tok_s: float
    mtp_tok_s: float
    accepted_drafts: int
    rejected_drafts: int
    drafted_tokens: int
    skipped_drafts: int
    bonus_tokens: int
    correction_tokens: int
    verify_calls: int
    graphbank: dict[str, Any]
    mtp_text_preview: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_mtp1_greedy_gate(
    model_path: Path | str,
    prompt_suite: Path | str,
    *,
    max_tokens: int = 32,
    seed: int = 0,
    limit: int | None = None,
    expand_to: int | None = None,
    enable_thinking: bool | None = None,
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
    if expand_to is not None and prompts:
        prompts = _expand_prompts(prompts, expand_to)
    if limit is not None:
        prompts = prompts[:limit]
    sampler = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
    rows: list[MTP1GateRow] = []

    for case in prompts:
        ids = encode_prompt_case(
            rt.tokenizer,
            case,
            chat_template=True,
            enable_thinking=enable_thinking,
        )
        token_budget = min(max_tokens, case.max_tokens)
        ar = generate_ar(rt, ids, max_tokens=token_budget, sampler=sampler, seed=seed)
        mtp = generate_mtp1(
            rt,
            ids,
            max_tokens=token_budget,
            sampler=sampler,
            seed=seed,
            verify_strategy=verify_strategy,
            verify_core=verify_core,
            draft_margin_threshold=draft_margin_threshold,
        )
        rows.append(
            MTP1GateRow(
                prompt_id=case.id,
                category=case.category,
                prompt_sha256=case.prompt_sha256,
                max_tokens=token_budget,
                exact_match=ar.tokens == mtp.tokens,
                ar_tokens=ar.tokens,
                mtp_tokens=mtp.tokens,
                ar_tok_s=ar.stats.tok_s,
                mtp_tok_s=mtp.stats.tok_s,
                accepted_drafts=mtp.stats.accepted_drafts,
                rejected_drafts=mtp.stats.rejected_drafts,
                drafted_tokens=mtp.stats.drafted_tokens,
                skipped_drafts=mtp.stats.skipped_drafts,
                bonus_tokens=mtp.stats.bonus_tokens,
                correction_tokens=mtp.stats.correction_tokens,
                verify_calls=mtp.stats.verify_calls,
                graphbank=mtp.stats.graphbank,
                mtp_text_preview=mtp.text[:240],
            )
        )

    matches = sum(1 for r in rows if r.exact_match)
    return {
        "model_path": str(model_path),
        "prompt_suite": str(prompt_suite),
        "sampler": asdict(sampler),
        "max_tokens": max_tokens,
        "seed": seed,
        "expand_to": expand_to,
        "enable_thinking": enable_thinking,
        "verify_strategy": verify_strategy,
        "verify_core": verify_core,
        "draft_margin_threshold": draft_margin_threshold,
        "mtp_quant_bits": rt.contract.mtp_quant_bits,
        "mtp_quant_group_size": rt.contract.mtp_quant_group_size,
        "mtp_quant_mode": rt.contract.mtp_quant_mode,
        "mtp_quant_policy": rt.contract.mtp_quant_policy,
        "matches": matches,
        "total": len(rows),
        "passed": matches == len(rows),
        "rows": [r.to_dict() for r in rows],
    }


def write_gate_result(path: Path | str, result: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))


def _expand_prompts(prompts, total: int):
    constraints = [
        "Prefer compact output.",
        "Preserve exact JSON when JSON is requested.",
        "Use production-quality Python when code is requested.",
        "Avoid explanatory prose unless explicitly asked.",
        "Keep identifiers readable.",
        "Use deterministic structure.",
        "Include edge-case handling.",
        "Prefer simple control flow.",
        "Preserve the requested output shape.",
        "Make the result useful for a coding assistant.",
    ]
    expanded = []
    idx = 0
    while len(expanded) < total:
        case = prompts[idx % len(prompts)]
        variant = idx // len(prompts)
        suffix = constraints[idx % len(constraints)]
        expanded.append(
            replace(
                case,
                id=f"{case.id}__v{variant:02d}",
                prompt=f"{case.prompt}\n\nVariant constraint: {suffix}",
                messages=None if case.messages is None else case.messages,
            )
        )
        idx += 1
    return expanded
