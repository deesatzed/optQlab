"""Backend-agnostic speculative sampling primitives."""

from __future__ import annotations

from optiq.runtime.mtp.sampling import (
    SpeculativeDecision,
    acceptance_probability,
    residual_distribution,
    sample_from_distribution,
    speculative_output_marginal,
    verify_one_token,
)

__all__ = [
    "SpeculativeDecision",
    "acceptance_probability",
    "residual_distribution",
    "sample_from_distribution",
    "speculative_output_marginal",
    "verify_one_token",
]
