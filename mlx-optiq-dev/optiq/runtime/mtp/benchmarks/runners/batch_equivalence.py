"""Batched-vs-sequential target forward equivalence gate."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from optiq.runtime.mtp.benchmarks.schema import encode_prompt_case, load_prompt_suite
from optiq.runtime.mtp.runtime import load


@dataclass
class BatchEquivalenceRow:
    prompt_id: str
    category: str
    prompt_sha256: str
    suffix_tokens: list[int]
    max_logit_abs_diff: float
    max_hidden_abs_diff: float
    argmax_matches: list[bool]
    batched_argmax: list[int]
    sequential_argmax: list[int]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_batch_equivalence(
    model_path: Path | str,
    prompt_suite: Path | str,
    *,
    suffix_len: int = 2,
    limit: int | None = None,
    expand_to: int | None = None,
    enable_thinking: bool | None = None,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Compare target forward over a suffix as one batch vs one token at a time.

    Speculative verify relies on `forward([primary, draft...])` being equivalent
    to sequential AR decode from the same prefix cache. A quant artifact that
    fails this gate cannot safely use batched speculative verification.
    """

    if suffix_len < 1:
        raise ValueError("suffix_len must be >= 1")

    rt = load(model_path, mtp=True)
    prompts = load_prompt_suite(prompt_suite)
    if expand_to is not None and prompts:
        prompts = _expand_prompts(prompts, expand_to)
    if limit is not None:
        prompts = prompts[:limit]

    rows: list[BatchEquivalenceRow] = []
    for case in prompts:
        prompt_ids = encode_prompt_case(
            rt.tokenizer,
            case,
            chat_template=True,
            enable_thinking=enable_thinking,
        )
        suffix_tokens = _greedy_suffix(rt, prompt_ids, suffix_len)
        batched_logits, batched_hidden = _forward_suffix_batched(rt, prompt_ids, suffix_tokens)
        seq_logits, seq_hidden = _forward_suffix_sequential(rt, prompt_ids, suffix_tokens)

        logit_diff = mx.max(mx.abs(batched_logits.astype(mx.float32) - seq_logits.astype(mx.float32)))
        hidden_diff = mx.max(mx.abs(batched_hidden.astype(mx.float32) - seq_hidden.astype(mx.float32)))
        batched_argmax = mx.argmax(batched_logits.astype(mx.float32), axis=-1)
        seq_argmax = mx.argmax(seq_logits.astype(mx.float32), axis=-1)
        mx.eval(logit_diff, hidden_diff, batched_argmax, seq_argmax)

        batched_ids = np.asarray(batched_argmax, dtype=np.int64).reshape(-1).tolist()
        seq_ids = np.asarray(seq_argmax, dtype=np.int64).reshape(-1).tolist()
        argmax_matches = [int(a) == int(b) for a, b in zip(batched_ids, seq_ids)]
        max_logit_abs_diff = float(logit_diff.item())
        max_hidden_abs_diff = float(hidden_diff.item())
        passed = bool(all(argmax_matches) and max_logit_abs_diff <= tolerance)

        rows.append(
            BatchEquivalenceRow(
                prompt_id=case.id,
                category=case.category,
                prompt_sha256=case.prompt_sha256,
                suffix_tokens=[int(x) for x in suffix_tokens],
                max_logit_abs_diff=max_logit_abs_diff,
                max_hidden_abs_diff=max_hidden_abs_diff,
                argmax_matches=argmax_matches,
                batched_argmax=[int(x) for x in batched_ids],
                sequential_argmax=[int(x) for x in seq_ids],
                passed=passed,
            )
        )

    passed_rows = sum(1 for row in rows if row.passed)
    return {
        "model_path": str(model_path),
        "prompt_suite": str(prompt_suite),
        "suffix_len": suffix_len,
        "tolerance": tolerance,
        "enable_thinking": enable_thinking,
        "passed": passed_rows == len(rows),
        "matches": passed_rows,
        "total": len(rows),
        "rows": [row.to_dict() for row in rows],
    }


def write_batch_equivalence(path: Path | str, result: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))


def _prefill(rt, prompt_ids: list[int]):
    cache = rt.make_cache()
    logits, hidden = rt.forward_ar(
        mx.array([prompt_ids]),
        cache=cache,
        return_hidden=True,
    )
    mx.eval(logits, hidden)
    return cache, logits[:, -1, :], hidden[:, -1:, :]


def _greedy_suffix(rt, prompt_ids: list[int], suffix_len: int) -> list[int]:
    cache, logits, _hidden = _prefill(rt, prompt_ids)
    suffix: list[int] = []
    for _ in range(suffix_len):
        token = int(mx.argmax(logits[0], axis=-1).item())
        suffix.append(token)
        logits_next, hidden_next = rt.forward_ar(
            mx.array([[token]]),
            cache=cache,
            return_hidden=True,
        )
        mx.eval(logits_next, hidden_next)
        logits = logits_next[:, -1, :]
    return suffix


def _forward_suffix_batched(rt, prompt_ids: list[int], suffix_tokens: list[int]):
    cache, _logits, _hidden = _prefill(rt, prompt_ids)
    logits, hidden = rt.forward_ar(
        mx.array([suffix_tokens]),
        cache=cache,
        return_hidden=True,
    )
    mx.eval(logits, hidden)
    return logits, hidden


def _forward_suffix_sequential(rt, prompt_ids: list[int], suffix_tokens: list[int]):
    cache, _logits, _hidden = _prefill(rt, prompt_ids)
    logits_parts = []
    hidden_parts = []
    for token in suffix_tokens:
        logits, hidden = rt.forward_ar(
            mx.array([[int(token)]]),
            cache=cache,
            return_hidden=True,
        )
        mx.eval(logits, hidden)
        logits_parts.append(logits)
        hidden_parts.append(hidden)
    return mx.concatenate(logits_parts, axis=1), mx.concatenate(hidden_parts, axis=1)


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
