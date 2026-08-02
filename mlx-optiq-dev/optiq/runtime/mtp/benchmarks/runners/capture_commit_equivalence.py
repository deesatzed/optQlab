"""Captured-prefix commit equivalence gate."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from optiq.runtime.mtp.benchmarks.schema import encode_prompt_case, load_prompt_suite
from optiq.runtime.mtp.gdn_capture import commit_captured_prefix
from optiq.runtime.mtp.graphbank import SpecDecodeGraphBank
from optiq.runtime.mtp.runtime import load


@dataclass
class CaptureCommitRow:
    prompt_id: str
    category: str
    prompt_sha256: str
    suffix_tokens: list[int]
    keep_tokens: int
    committed: bool
    max_cached_logit_abs_diff: float
    max_cached_hidden_abs_diff: float
    max_next_logit_abs_diff: float
    max_next_hidden_abs_diff: float
    cached_argmax_match: bool
    next_argmax_match: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_capture_commit_equivalence(
    model_path: Path | str,
    prompt_suite: Path | str,
    *,
    suffix_len: int = 6,
    min_keep_tokens: int = 1,
    limit: int | None = None,
    expand_to: int | None = None,
    enable_thinking: bool | None = None,
    tolerance: float = 1e-3,
    verify_backend: str = "direct",
    verify_core: str = "stock",
) -> dict[str, Any]:
    if suffix_len < 1:
        raise ValueError("suffix_len must be >= 1")
    if min_keep_tokens < 1 or min_keep_tokens > suffix_len:
        raise ValueError("min_keep_tokens must be in [1, suffix_len]")
    if verify_backend not in {"direct", "graphbank"}:
        raise ValueError("verify_backend must be 'direct' or 'graphbank'")

    rt = load(model_path, mtp=True)
    graphbank = SpecDecodeGraphBank(rt, capture_backend=verify_core) if verify_backend == "graphbank" else None
    prompts = load_prompt_suite(prompt_suite)
    if expand_to is not None and prompts:
        prompts = _expand_prompts(prompts, expand_to)
    if limit is not None:
        prompts = prompts[:limit]

    rows: list[CaptureCommitRow] = []
    for case in prompts:
        prompt_ids = encode_prompt_case(
            rt.tokenizer,
            case,
            chat_template=True,
            enable_thinking=enable_thinking,
        )
        suffix_tokens = _greedy_suffix(rt, prompt_ids, suffix_len + 1)
        verify_tokens = suffix_tokens[:suffix_len]
        next_after_window = suffix_tokens[suffix_len]

        for keep_tokens in range(min_keep_tokens, suffix_len + 1):
            committed_cache, _logits, _hidden = _prefill(rt, prompt_ids)
            if graphbank is not None:
                verify_logits_i, verify_hidden_i, captures_i = graphbank.forward_ar_capture(
                    mx.array([verify_tokens]),
                    cache=committed_cache,
                    return_hidden=True,
                )
            else:
                verify_logits_i, verify_hidden_i, captures_i = rt.forward_ar_capture(
                    mx.array([verify_tokens]),
                    cache=committed_cache,
                    return_hidden=True,
                    capture_backend=verify_core,
                )
            mx.eval(verify_logits_i, verify_hidden_i, captures_i)
            committed = commit_captured_prefix(
                committed_cache,
                captures_i,
                keep_tokens=keep_tokens,
                verified_tokens=suffix_len,
            )

            seq_cache, seq_logits, seq_hidden = _forward_prefix_sequential(
                rt,
                prompt_ids,
                verify_tokens[:keep_tokens],
            )
            cached_logits = verify_logits_i[:, keep_tokens - 1 : keep_tokens, :]
            cached_hidden = verify_hidden_i[:, keep_tokens - 1 : keep_tokens, :]

            next_token = (
                verify_tokens[keep_tokens]
                if keep_tokens < suffix_len
                else next_after_window
            )
            commit_next_logits, commit_next_hidden = rt.forward_ar(
                mx.array([[next_token]]),
                cache=committed_cache,
                return_hidden=True,
            )
            seq_next_logits, seq_next_hidden = rt.forward_ar(
                mx.array([[next_token]]),
                cache=seq_cache,
                return_hidden=True,
            )
            mx.eval(commit_next_logits, commit_next_hidden, seq_next_logits, seq_next_hidden)

            cached_logit_diff = _max_abs_diff(cached_logits, seq_logits)
            cached_hidden_diff = _max_abs_diff(cached_hidden, seq_hidden)
            next_logit_diff = _max_abs_diff(commit_next_logits, seq_next_logits)
            next_hidden_diff = _max_abs_diff(commit_next_hidden, seq_next_hidden)
            cached_argmax_match = _argmax_match(cached_logits, seq_logits)
            next_argmax_match = _argmax_match(commit_next_logits, seq_next_logits)
            passed = bool(
                committed
                and cached_argmax_match
                and next_argmax_match
                and cached_logit_diff <= tolerance
                and next_logit_diff <= tolerance
            )

            rows.append(
                CaptureCommitRow(
                    prompt_id=case.id,
                    category=case.category,
                    prompt_sha256=case.prompt_sha256,
                    suffix_tokens=[int(x) for x in verify_tokens],
                    keep_tokens=keep_tokens,
                    committed=bool(committed),
                    max_cached_logit_abs_diff=cached_logit_diff,
                    max_cached_hidden_abs_diff=cached_hidden_diff,
                    max_next_logit_abs_diff=next_logit_diff,
                    max_next_hidden_abs_diff=next_hidden_diff,
                    cached_argmax_match=cached_argmax_match,
                    next_argmax_match=next_argmax_match,
                    passed=passed,
                )
            )

    passed_rows = sum(1 for row in rows if row.passed)
    return {
        "model_path": str(model_path),
        "prompt_suite": str(prompt_suite),
        "suffix_len": suffix_len,
        "min_keep_tokens": min_keep_tokens,
        "tolerance": tolerance,
        "enable_thinking": enable_thinking,
        "verify_backend": verify_backend,
        "verify_core": verify_core,
        "graphbank": graphbank.to_dict() if graphbank is not None else {},
        "passed": passed_rows == len(rows),
        "matches": passed_rows,
        "total": len(rows),
        "rows": [row.to_dict() for row in rows],
    }


def write_capture_commit_equivalence(path: Path | str, result: dict[str, Any]) -> None:
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
    return cache, logits[:, -1:, :], hidden[:, -1:, :]


def _greedy_suffix(rt, prompt_ids: list[int], suffix_len: int) -> list[int]:
    cache, logits, _hidden = _prefill(rt, prompt_ids)
    suffix: list[int] = []
    for _ in range(suffix_len):
        token = int(mx.argmax(logits[:, -1, :][0], axis=-1).item())
        suffix.append(token)
        logits, hidden = rt.forward_ar(
            mx.array([[token]]),
            cache=cache,
            return_hidden=True,
        )
        mx.eval(logits, hidden)
    return suffix


def _forward_prefix_sequential(rt, prompt_ids: list[int], prefix_tokens: list[int]):
    cache, logits, hidden = _prefill(rt, prompt_ids)
    for token in prefix_tokens:
        logits, hidden = rt.forward_ar(
            mx.array([[int(token)]]),
            cache=cache,
            return_hidden=True,
        )
        mx.eval(logits, hidden)
    return cache, logits[:, -1:, :], hidden[:, -1:, :]


def _max_abs_diff(left: mx.array, right: mx.array) -> float:
    diff = mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32)))
    mx.eval(diff)
    return float(diff.item())


def _argmax_match(left: mx.array, right: mx.array) -> bool:
    left_id = mx.argmax(left.astype(mx.float32), axis=-1)
    right_id = mx.argmax(right.astype(mx.float32), axis=-1)
    mx.eval(left_id, right_id)
    return bool(np.asarray(left_id).reshape(-1).tolist() == np.asarray(right_id).reshape(-1).tolist())


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
