"""Draft proposal sampler metadata helpers."""

from __future__ import annotations

from typing import Any


def normalize_draft_sampler_spec(
    value: Any,
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a validated draft sampler override from profile/contract data."""
    if value is None:
        return fallback
    if not isinstance(value, dict):
        raise ValueError("draft sampler spec must be an object")
    temperature = float(value.get("temperature", 0.6))
    top_p = float(value.get("top_p", 0.95))
    top_k = int(value.get("top_k", 20))
    if temperature < 0:
        raise ValueError("draft sampler temperature must be non-negative")
    if top_p <= 0:
        raise ValueError("draft sampler top_p must be positive")
    if top_k < 0:
        raise ValueError("draft sampler top_k must be non-negative")
    return {"temperature": temperature, "top_p": top_p, "top_k": top_k}


def draft_sampler_spec_from_runtime_contract(
    contract_data: Any,
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Resolve a model-specific draft sampler recommendation."""
    if not isinstance(contract_data, dict):
        return fallback
    return normalize_draft_sampler_spec(
        contract_data.get("recommended_draft_sampler"),
        fallback=fallback,
    )
