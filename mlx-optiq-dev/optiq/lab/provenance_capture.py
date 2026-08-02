"""Partial provenance capture for chat stream end (Phase 0).

Builds a provenance envelope from *known* server-side fields only.
Never invents tok/s, peak memory, or other unmeasured metrics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from . import chat_store, spine


# Canonical envelope keys from design §A5 (Phase 0).
_ENVELOPE_KEYS = (
    "build_id",
    "quant_profile",
    "adapter_stack",
    "kv_bits",
    "sampler",
    "context_used",
    "context_window",
    "tools_enabled",
    "tools_called",
    "retrieved_chunk_ids",
    "healed",
    "retry_hits",
    "thinking_used",
    "thinking_budget",
    "tok_per_sec",
    "peak_mem_gb",
    "server_model_label",
    "captured_at",
)


def _iso_now() -> str:
    """UTC ISO-8601 timestamp with Z suffix (second precision)."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_partial_provenance(
    *,
    model: str | None,
    sampler: dict | None,
    tools_called: list | None = None,
    healed: bool | None = None,
    retry_hits: int | None = None,
    context_window: int | None = None,
    context_used: int | None = None,
    build_id: str | None = None,
    captured_at: str | None = None,
    tools_enabled: list | None = None,
    quant_profile: str | None = None,
    adapter_stack: list | None = None,
    kv_bits: int | None = None,
    retrieved_chunk_ids: list | None = None,
    thinking_used: int | None = None,
    thinking_budget: int | None = None,
    tok_per_sec: float | None = None,
    peak_mem_gb: float | None = None,
) -> dict:
    """Return envelope with only known fields set; others null.

    Always set ``captured_at`` (ISO-8601 if not provided).
    Never invent ``tok_per_sec`` or ``peak_mem_gb`` — they stay null unless
    an explicit measured value is passed.
    """
    envelope: dict[str, Any] = {k: None for k in _ENVELOPE_KEYS}

    envelope["build_id"] = build_id
    envelope["quant_profile"] = quant_profile
    envelope["adapter_stack"] = adapter_stack
    envelope["kv_bits"] = kv_bits
    envelope["sampler"] = sampler
    envelope["context_used"] = context_used
    envelope["context_window"] = context_window
    envelope["tools_enabled"] = tools_enabled
    envelope["tools_called"] = tools_called
    envelope["retrieved_chunk_ids"] = retrieved_chunk_ids
    envelope["healed"] = healed
    envelope["retry_hits"] = retry_hits
    envelope["thinking_used"] = thinking_used
    envelope["thinking_budget"] = thinking_budget
    # Metrics: only set when an explicit measured value is provided.
    envelope["tok_per_sec"] = tok_per_sec
    envelope["peak_mem_gb"] = peak_mem_gb
    envelope["server_model_label"] = model
    envelope["captured_at"] = captured_at if captured_at else _iso_now()

    return envelope


def apply_stream_provenance(
    chat_id: str | None,
    messages: list[dict],
    provenance: dict,
) -> str | None:
    """If chat_id is None, return None.

    Else ensure conversation is saved with messages; attach provenance to the
    last assistant message via ``spine.set_message_provenance``.
    Return that message_id or None when no assistant message exists.
    """
    if chat_id is None:
        return None

    if not isinstance(messages, list):
        messages = []

    existing = chat_store.load_chat_record(chat_id)
    title = "Untitled chat"
    model = ""
    if isinstance(existing, dict):
        title = existing.get("title") or title
        model = existing.get("model") or model
    if not model and isinstance(provenance, dict):
        model = provenance.get("server_model_label") or ""

    chat_store.save_chat_record({
        "id": chat_id,
        "title": title,
        "model": model or "",
        "messages": messages,
    })

    conv = spine.get_conversation(chat_id)
    if conv is None:
        return None

    message_id: str | None = None
    for msg in reversed(conv.get("messages") or []):
        if msg.get("role") == "assistant":
            message_id = msg.get("id")
            break

    if not message_id:
        return None

    spine.set_message_provenance(message_id, provenance)
    return message_id
