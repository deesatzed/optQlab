"""Aggregate capability score for cross-quant ranking.

A single scalar that's the simple unweighted mean of the six
benchmarks (MMLU + GSM8K + IFEval + BFCL + HumanEval + HashHop). Disk
size is reported separately as a transparent context number. No penalty
math hidden in the score.

Why simple average rather than a weighted formula:

  * Weighted formulas (like Unsloth's ``MMLU − 25 × disk_GB``) embed
    a particular quality/disk tradeoff that may not match the user's.
  * Disk-penalized scores can hide real capability wins. A model that's
    +1.5 pp on capability and +1 GB on disk is "worse" by such formulas
    even though most users would prefer the bigger artifact.
  * A simple mean across the 6 benchmarks is honest, transparent, and
    leaves the disk-vs-quality tradeoff to the reader.

Disk is still tracked in the result so users can compare quants at the
same size class side by side.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CapabilityScore:
    """Simple unweighted average of benchmark scores.

    ``score`` is the mean of whichever of MMLU/GSM8K/IFEval/BFCL/HumanEval/HashHop
    were provided (None values are skipped). ``disk_gb`` is reported
    separately and does NOT affect the score.
    """

    score: float                   # 0–100, simple mean of benchmark percents
    mmlu_pct: float | None
    gsm8k_pct: float | None
    ifeval_pct: float | None
    bfcl_pct: float | None
    humaneval_pct: float | None
    hashhop_pct: float | None
    disk_gb: float
    components: dict[str, float]   # per-benchmark contribution to the mean

    def __str__(self) -> str:
        parts = [f"Capability_Score = {self.score:.2f}  (disk: {self.disk_gb:.1f} GB)"]
        for name, val in self.components.items():
            parts.append(f"  {name:14s} {val:.2f}")
        return "\n".join(parts)


# Backwards-compatibility alias for old code that imports QualityScore.
QualityScore = CapabilityScore


def _disk_gb(model_path: str) -> float:
    """Sum the safetensors shard sizes (GB) for ``model_path``.

    Accepts:
      * A local directory — sum its ``*.safetensors`` files.
      * An HF repo id (``org/name``) — query ``HfApi.model_info`` for
        per-shard sizes. Falls back to summing the on-disk HF cache
        shards if the API is unreachable.
    """
    p = Path(model_path)
    if p.is_dir():
        total = sum(f.stat().st_size for f in p.glob("*.safetensors"))
        return total / 1024**3

    if "/" not in model_path:
        return 0.0
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_path, files_metadata=True)
        total = sum(
            (s.size or 0) for s in (info.siblings or [])
            if s.rfilename and s.rfilename.endswith(".safetensors")
        )
        if total > 0:
            return total / 1024**3
    except Exception:
        pass

    try:
        org, name = model_path.split("/", 1)
        cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{name}" / "snapshots"
        if cache.exists():
            for snap in cache.iterdir():
                if snap.is_dir():
                    total = sum(
                        f.stat().st_size for f in snap.glob("*.safetensors")
                        if f.exists()
                    )
                    if total > 0:
                        return total / 1024**3
                    break
    except Exception:
        pass

    return 0.0


def compute_quality_score(
    model_path: str,
    mmlu_pct: float | None = None,
    gsm8k_pct: float | None = None,
    ifeval_pct: float | None = None,
    bfcl_pct: float | None = None,
    humaneval_pct: float | None = None,
    hashhop_pct: float | None = None,
    disk_gb: float | None = None,
) -> CapabilityScore:
    """Compute the simple unweighted mean of provided benchmark scores.

    All inputs are percentages in 0–100. ``disk_gb`` is auto-computed
    from the model path if not provided. Disk is reported but does NOT
    enter the score formula — a model's score is purely its average
    capability across the benchmarks.
    """
    if disk_gb is None:
        disk_gb = _disk_gb(model_path)

    components: dict[str, float] = {}
    if mmlu_pct is not None:
        components["MMLU"] = float(mmlu_pct)
    if gsm8k_pct is not None:
        components["GSM8K"] = float(gsm8k_pct)
    if ifeval_pct is not None:
        components["IFEval"] = float(ifeval_pct)
    if bfcl_pct is not None:
        components["BFCL"] = float(bfcl_pct)
    if humaneval_pct is not None:
        components["HumanEval"] = float(humaneval_pct)
    if hashhop_pct is not None:
        components["HashHop"] = float(hashhop_pct)

    if not components:
        score = 0.0
    else:
        score = sum(components.values()) / len(components)

    return CapabilityScore(
        score=score,
        mmlu_pct=mmlu_pct,
        gsm8k_pct=gsm8k_pct,
        ifeval_pct=ifeval_pct,
        bfcl_pct=bfcl_pct,
        humaneval_pct=humaneval_pct,
        hashhop_pct=hashhop_pct,
        disk_gb=disk_gb,
        components=components,
    )
