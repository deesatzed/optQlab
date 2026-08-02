"""Output quality verification for quantized models."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VerificationResult:
    metric_name: str
    value: float
    threshold: float
    passed: bool
    details: str = ""


def verify_llm_perplexity(
    perplexity_original: float,
    perplexity_quantized: float,
    max_degradation: float = 0.5,
) -> VerificationResult:
    """Verify that quantized LLM perplexity hasn't degraded too much."""
    degradation = perplexity_quantized - perplexity_original
    passed = degradation <= max_degradation
    return VerificationResult(
        metric_name="Perplexity Degradation",
        value=degradation,
        threshold=max_degradation,
        passed=passed,
        details=f"Original: {perplexity_original:.3f}, Quantized: {perplexity_quantized:.3f}, "
                f"Delta: {degradation:+.3f} (max allowed: {max_degradation})",
    )


def verify_llm_kl_divergence(
    kl_divergence: float,
    max_kl: float = 0.1,
) -> VerificationResult:
    """Verify KL divergence between original and quantized model outputs."""
    passed = kl_divergence <= max_kl
    return VerificationResult(
        metric_name="KL Divergence",
        value=kl_divergence,
        threshold=max_kl,
        passed=passed,
        details=f"KL(original || quantized) = {kl_divergence:.6f} (max allowed: {max_kl})",
    )



def print_verification_report(results: list[VerificationResult]):
    """Print a formatted verification report."""
    print("\n  Verification Report")
    print("  " + "=" * 70)
    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        icon = "+" if r.passed else "!"
        print(f"  [{icon}] {status}: {r.metric_name}")
        print(f"      {r.details}")
        if not r.passed:
            all_passed = False

    print("  " + "=" * 70)
    if all_passed:
        print("  All checks passed!")
    else:
        print("  Some checks failed. Consider adjusting target BPW or thresholds.")
