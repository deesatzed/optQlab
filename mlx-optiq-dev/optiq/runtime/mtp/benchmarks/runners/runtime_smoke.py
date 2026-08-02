"""Runtime smoke gates for AR hidden-state and MTP forward execution."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from optiq.runtime.mtp.runtime import load
from optiq.runtime.mtp.mtp_patch import validate_mtp_support


def run_runtime_smoke(model_path: Path | str, prompt: str) -> dict[str, Any]:
    started = time.perf_counter()
    rt = load(model_path, mtp=True)
    load_s = time.perf_counter() - started

    tokens = rt.tokenizer.encode(prompt)
    cache = rt.make_cache()
    started = time.perf_counter()
    logits, hidden = rt.forward_ar(mx.array([tokens]), cache=cache, return_hidden=True)
    mx.eval(logits, hidden)
    ar_s = time.perf_counter() - started

    next_id = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    started = time.perf_counter()
    draft_logits = rt.draft_mtp(
        hidden[:, -1:, :],
        mx.array([[next_id]]),
        mtp_cache=rt.make_mtp_cache(),
    )
    mx.eval(draft_logits)
    mtp_s = time.perf_counter() - started

    return {
        "model_path": str(model_path),
        "prompt_tokens": len(tokens),
        "load_s": load_s,
        "mtp_enabled": rt.mtp_enabled,
        "mtp_valid": validate_mtp_support(rt.model),
        "ar_logits_shape": list(logits.shape),
        "hidden_shape": list(hidden.shape),
        "mtp_logits_shape": list(draft_logits.shape),
        "greedy_next_id": next_id,
        "ar_forward_s": ar_s,
        "mtp_forward_s": mtp_s,
        "active_memory": int(mx.get_active_memory()),
        "peak_memory": int(mx.get_peak_memory()),
    }
