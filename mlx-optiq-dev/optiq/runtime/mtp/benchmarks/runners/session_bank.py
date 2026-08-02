"""SessionBank warm-prefix correctness and TTFT benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from optiq.runtime.mtp.benchmarks.schema import encode_prompt_case, load_prompt_suite, now_run_id
from optiq.runtime.mtp.mtp_patch import MTPContract
from optiq.runtime.mtp.runtime import load
from optiq.runtime.mtp.session_bank import (
    SessionBank,
    max_abs_diff,
    prefill_target,
    prefill_target_with_session_bank,
)


def _encode_suffix(tokenizer: Any, suffix: str) -> list[int]:
    if not suffix:
        return []
    try:
        return list(tokenizer.encode(suffix, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(suffix))


def _argmax_token(logits) -> int:
    token = mx.argmax(logits[0], axis=-1)
    mx.eval(token)
    return int(token.item())


def _continue_suffix(rt, cache, suffix_ids: list[int]):
    if not suffix_ids:
        raise ValueError("suffix_ids must not be empty")
    elapsed = 0.0
    if len(suffix_ids) > 1:
        import time

        started = time.perf_counter()
        prefill = rt.forward_ar(
            mx.array([suffix_ids[:-1]]),
            cache=cache,
            return_hidden=False,
        )
        mx.eval(prefill)
        elapsed += time.perf_counter() - started
    import time

    started = time.perf_counter()
    logits, hidden_seq = rt.forward_ar(
        mx.array([[suffix_ids[-1]]]),
        cache=cache,
        return_hidden=True,
    )
    mx.eval(logits, hidden_seq)
    elapsed += time.perf_counter() - started
    return logits[:, -1, :], hidden_seq[:, -1:, :], elapsed


def run_session_bank_benchmark(
    model_path: Path | str,
    prompt_suite: Path | str,
    *,
    prompt_index: int = 0,
    suffix_text: str = "\n\n# Follow-up request:\nRefactor this into a cleaner implementation.\n",
    max_prompt_tokens: int = 512,
    chat_template: bool = True,
    enable_thinking: bool | None = None,
    max_entries: int = 4,
    tolerance: float = 1e-3,
    restore_mode: str = "clone",
) -> dict[str, Any]:
    if restore_mode not in {"clone", "reference"}:
        raise ValueError("restore_mode must be 'clone' or 'reference'")
    rt = load(model_path, mtp=True, contract=MTPContract())
    cases = load_prompt_suite(prompt_suite)
    if not cases:
        raise ValueError("prompt suite is empty")
    case = cases[int(prompt_index) % len(cases)]
    base_ids = encode_prompt_case(
        rt.tokenizer,
        case,
        chat_template=chat_template,
        enable_thinking=enable_thinking,
    )
    if max_prompt_tokens > 0:
        base_ids = base_ids[-int(max_prompt_tokens) :]
    suffix_ids = _encode_suffix(rt.tokenizer, suffix_text)
    followup_ids = base_ids + suffix_ids
    if len(followup_ids) == len(base_ids):
        raise ValueError("suffix_text produced no suffix tokens")

    bank = SessionBank(max_entries=max_entries)

    base_cache, base_logits, base_hidden, base_prefill_s = prefill_target(
        rt,
        base_ids,
        return_hidden=True,
    )
    entry = bank.put(
        runtime=rt,
        token_ids=base_ids,
        cache=base_cache,
        logits=base_logits,
        hidden=base_hidden,
        hidden_variant=rt.contract.hidden_variant,
        keep_live_ref=restore_mode == "reference",
    )

    cold_cache, cold_logits, cold_hidden, cold_followup_prefill_s = prefill_target(
        rt,
        followup_ids,
        return_hidden=True,
    )
    if restore_mode == "reference":
        warm_cache, warm_logits, warm_hidden, warm_followup_prefill_s, warm_info = (
            prefill_target_with_session_bank(
                rt,
                followup_ids,
                bank,
                return_hidden=True,
                restore_mode="reference",
            )
        )
        (
            _clone_cache,
            continuation_logits,
            continuation_hidden,
            continuation_prefill_s,
            continuation_info,
        ) = prefill_target_with_session_bank(
            rt,
            followup_ids,
            bank,
            return_hidden=True,
            restore_mode="clone",
        )
    else:
        continuation_logits, continuation_hidden, continuation_prefill_s = _continue_suffix(
            rt,
            base_cache,
            suffix_ids,
        )
        continuation_info = {"baseline": "direct_continuation"}
        warm_cache, warm_logits, warm_hidden, warm_followup_prefill_s, warm_info = (
            prefill_target_with_session_bank(
                rt,
                followup_ids,
                bank,
                return_hidden=True,
                restore_mode="clone",
            )
        )

    logits_diff = max_abs_diff(continuation_logits, warm_logits)
    hidden_diff = max_abs_diff(continuation_hidden, warm_hidden)
    cold_logits_diff = max_abs_diff(cold_logits, warm_logits)
    cold_hidden_diff = max_abs_diff(cold_hidden, warm_hidden)
    cold_vs_continuation_logits_diff = max_abs_diff(cold_logits, continuation_logits)
    cold_vs_continuation_hidden_diff = max_abs_diff(cold_hidden, continuation_hidden)
    cold_argmax = _argmax_token(cold_logits)
    continuation_argmax = _argmax_token(continuation_logits)
    warm_argmax = _argmax_token(warm_logits)
    exact = (
        logits_diff is not None
        and hidden_diff is not None
        and logits_diff <= tolerance
        and hidden_diff <= tolerance
        and continuation_argmax == warm_argmax
    )
    speedup = (
        cold_followup_prefill_s / warm_followup_prefill_s
        if warm_followup_prefill_s > 0
        else None
    )

    # Keep references live until after exactness checks so lazy work cannot be
    # optimized away before timings are synchronized.
    mx.eval(cold_logits, warm_logits)
    _ = (cold_cache, warm_cache)

    return {
        "run_id": now_run_id("session-bank"),
        "model_path": str(model_path),
        "prompt_suite": str(prompt_suite),
        "prompt_id": case.id,
        "category": case.category,
        "chat_template": chat_template,
        "enable_thinking": enable_thinking,
        "base_tokens": len(base_ids),
        "suffix_tokens": len(suffix_ids),
        "followup_tokens": len(followup_ids),
        "base_prefill_s": base_prefill_s,
        "continuation_followup_prefill_s": continuation_prefill_s,
        "cold_followup_prefill_s": cold_followup_prefill_s,
        "warm_followup_prefill_s": warm_followup_prefill_s,
        "warm_speedup_vs_cold": speedup,
        "warm_speedup_vs_continuation": (
            continuation_prefill_s / warm_followup_prefill_s
            if warm_followup_prefill_s > 0
            else None
        ),
        "warm_info": warm_info,
        "continuation_info": continuation_info,
        "restore_mode": restore_mode,
        "exact": exact,
        "tolerance": tolerance,
        "logits_max_abs_diff": logits_diff,
        "hidden_max_abs_diff": hidden_diff,
        "cold_stateless_vs_warm_logits_max_abs_diff": cold_logits_diff,
        "cold_stateless_vs_warm_hidden_max_abs_diff": cold_hidden_diff,
        "cold_stateless_vs_continuation_logits_max_abs_diff": cold_vs_continuation_logits_diff,
        "cold_stateless_vs_continuation_hidden_max_abs_diff": cold_vs_continuation_hidden_diff,
        "cold_argmax": cold_argmax,
        "continuation_argmax": continuation_argmax,
        "warm_argmax": warm_argmax,
        "entry": {
            "prefix_len": entry.prefix_len,
            "token_hash": entry.token_hash,
            "nbytes": entry.nbytes,
        },
        "bank": bank.to_dict(),
    }


def write_session_bank_report(path: Path | str, result: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
