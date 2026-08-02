"""Lightweight attention-phase telemetry context."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

VALID_ATTENTION_PHASES = {
    "prefill",
    "decode_verify",
    "ar_decode",
    "postcommit",
    "unknown",
}

_ATTENTION_PHASE: ContextVar[str] = ContextVar(
    "mtplx_attention_phase",
    default="unknown",
)


def normalize_attention_phase(phase: str | None) -> str:
    value = (phase or "unknown").strip().lower()
    return value if value in VALID_ATTENTION_PHASES else "unknown"


def current_attention_phase() -> str:
    return normalize_attention_phase(_ATTENTION_PHASE.get())


@contextmanager
def attention_phase(phase: str | None) -> Iterator[None]:
    token = _ATTENTION_PHASE.set(normalize_attention_phase(phase))
    try:
        yield
    finally:
        _ATTENTION_PHASE.reset(token)
