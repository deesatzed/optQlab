"""Adaptive-depth policies for native MTP experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class AdaptiveDepthPolicy:
    max_depth: int
    min_depth: int = 1
    start_depth: int = 1
    increase_after: int = 4
    decrease_after: int = 1

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.min_depth < 1:
            raise ValueError("min_depth must be >= 1")
        if self.min_depth > self.max_depth:
            raise ValueError("min_depth must be <= max_depth")
        self.current_depth = min(max(self.start_depth, self.min_depth), self.max_depth)
        self._full_accept_streak = 0
        self._early_reject_streak = 0

    def observe(self, *, attempted_depth: int, accepted_depths: int) -> dict[str, int | str]:
        """Update depth from one cycle outcome and return a loggable decision."""
        attempted_depth = max(1, min(int(attempted_depth), self.max_depth))
        accepted_depths = max(0, min(int(accepted_depths), attempted_depth))
        previous_depth = self.current_depth
        action = "hold"

        if accepted_depths == attempted_depth:
            self._full_accept_streak += 1
            self._early_reject_streak = 0
            if self._full_accept_streak >= self.increase_after and self.current_depth < self.max_depth:
                self.current_depth += 1
                self._full_accept_streak = 0
                action = "increase"
        else:
            self._full_accept_streak = 0
            rejected_at = accepted_depths + 1
            if rejected_at <= max(1, previous_depth // 2):
                self._early_reject_streak += 1
            else:
                self._early_reject_streak = 0

            if self._early_reject_streak >= self.decrease_after and self.current_depth > self.min_depth:
                self.current_depth -= 1
                self._early_reject_streak = 0
                action = "decrease"

        return {
            "previous_depth": previous_depth,
            "attempted_depth": attempted_depth,
            "accepted_depths": accepted_depths,
            "next_depth": self.current_depth,
            "action": action,
        }


@dataclass
class ExpectedValueDepthPolicy:
    """Cost-aware D2/D3 controller.

    The policy starts each cycle at ``max_depth`` so the generation loop can
    draft sequentially, then it may stop after ``base_depth`` if the next
    depth does not clear a measured expected-value gate.
    """

    max_depth: int
    base_depth: int = 2
    min_depth: int = 1
    accept_priors: tuple[float, ...] = (0.92, 0.64, 0.32)
    ewma_alpha: float = 0.12
    draft_cost_s: float = 0.0048
    extra_verify_cost_s: float = 0.0060
    baseline_tok_s: float = 40.0
    safety_margin: float = 0.10
    margin_center: float = 1.0
    margin_scale: float = 2.0
    confidence_weight: float = 0.35
    min_extra_accept_probability: float = 0.18

    wants_draft_metrics: bool = True

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.min_depth < 1:
            raise ValueError("min_depth must be >= 1")
        if self.min_depth > self.max_depth:
            raise ValueError("min_depth must be <= max_depth")
        if self.base_depth < self.min_depth:
            raise ValueError("base_depth must be >= min_depth")
        if self.base_depth > self.max_depth:
            raise ValueError("base_depth must be <= max_depth")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if self.draft_cost_s < 0.0 or self.extra_verify_cost_s < 0.0:
            raise ValueError("cost estimates must be non-negative")
        if self.baseline_tok_s <= 0.0:
            raise ValueError("baseline_tok_s must be > 0")
        self.current_depth = self.max_depth
        priors = list(self.accept_priors)
        if len(priors) < self.max_depth:
            priors.extend([priors[-1] if priors else 0.5] * (self.max_depth - len(priors)))
        self._accept_ewma = [_clamp(float(value), 0.0, 1.0) for value in priors[: self.max_depth]]
        self._last_continue_decision: dict[str, int | float | bool | str] | None = None

    def should_continue_after_draft(
        self,
        *,
        drafted_depth: int,
        max_depth: int,
        draft_metrics: dict,
    ) -> dict[str, int | float | bool | str]:
        """Return whether generation should draft the next depth."""
        drafted_depth = int(drafted_depth)
        max_depth = int(max_depth)
        if drafted_depth < self.base_depth or drafted_depth >= max_depth:
            decision = {
                "continue": True,
                "action": "continue",
                "reason": "below_base_or_at_max",
                "drafted_depth": drafted_depth,
            }
            self._last_continue_decision = decision
            return decision

        next_depth = drafted_depth + 1
        if next_depth > self.max_depth:
            decision = {
                "continue": False,
                "action": "stop",
                "reason": "beyond_max_depth",
                "drafted_depth": drafted_depth,
            }
            self._last_continue_decision = decision
            return decision

        prefix_probability = 1.0
        for index in range(drafted_depth):
            prefix_probability *= self._accept_ewma[index]
        next_probability = self._accept_ewma[next_depth - 1]
        confidence_factor = self._confidence_factor(draft_metrics)
        expected_extra_accept = _clamp(
            prefix_probability * next_probability * confidence_factor,
            0.0,
            0.999,
        )
        extra_cost_s = self.draft_cost_s + self.extra_verify_cost_s
        required_extra_accept = max(
            self.min_extra_accept_probability,
            extra_cost_s * self.baseline_tok_s * (1.0 + self.safety_margin),
        )
        should_continue = expected_extra_accept >= required_extra_accept
        decision = {
            "continue": bool(should_continue),
            "action": "continue" if should_continue else "stop",
            "reason": "ev_pass" if should_continue else "ev_fail",
            "drafted_depth": drafted_depth,
            "next_depth": next_depth,
            "prefix_accept_estimate": float(prefix_probability),
            "next_accept_estimate": float(next_probability),
            "confidence_factor": float(confidence_factor),
            "expected_extra_accept": float(expected_extra_accept),
            "required_extra_accept": float(required_extra_accept),
            "extra_cost_s": float(extra_cost_s),
            "baseline_tok_s": float(self.baseline_tok_s),
        }
        self._last_continue_decision = decision
        return decision

    def observe(self, *, attempted_depth: int, accepted_depths: int) -> dict[str, int | float | bool | str | list[float] | dict | None]:
        """Update EWMAs from one cycle outcome and return a loggable decision."""
        attempted_depth = max(1, min(int(attempted_depth), self.max_depth))
        accepted_depths = max(0, min(int(accepted_depths), attempted_depth))
        for index in range(attempted_depth):
            accepted = 1.0 if accepted_depths > index else 0.0
            self._accept_ewma[index] = (
                (1.0 - self.ewma_alpha) * self._accept_ewma[index]
                + self.ewma_alpha * accepted
            )
        return {
            "kind": "expected_value",
            "attempted_depth": attempted_depth,
            "accepted_depths": accepted_depths,
            "next_depth": self.current_depth,
            "action": "update_ewma",
            "accept_ewma": [float(v) for v in self._accept_ewma],
            "last_continue_decision": self._last_continue_decision,
        }

    def _confidence_factor(self, draft_metrics: dict) -> float:
        margin = draft_metrics.get("top2_margin")
        if margin is None:
            return 1.0
        scaled = (float(margin) - self.margin_center) / max(self.margin_scale, 1e-6)
        margin_term = math.tanh(scaled)
        top1_prob = draft_metrics.get("top1_prob_topk")
        prob_term = 0.0
        if top1_prob is not None:
            prob_term = 2.0 * _clamp(float(top1_prob), 0.0, 1.0) - 1.0
        raw = 1.0 + self.confidence_weight * (0.75 * margin_term + 0.25 * prob_term)
        return _clamp(raw, 0.25, 1.75)
