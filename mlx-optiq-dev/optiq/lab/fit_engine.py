"""Fit Engine — predict whether a model load fits this Mac.

Uses real free unified memory (psutil), on-disk weight size when a path is
given, and documented priors for KV + activation headroom. Verdicts:

  comfortable | degraded | will_not_fit | hard_fail

Never invents Capability Scores. Estimates that are not measured are
labeled in the returned dict (``estimate_notes``).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import ensure_lab_dirs

# Ratio of predicted total vs free RAM (after reserved overhead).
RATIO_COMFORTABLE = 0.72
RATIO_DEGRADED = 0.88

# Apple-specific priors (from redesign / empirical notes — not live Metal counters).
# High context + low KV precision risks MTLResource bind ceiling before byte OOM.
MTL_CTX_THRESHOLD = 65536
MTL_KV_BITS_THRESHOLD = 6

# Default framework / OS reserved when no calibration exists.
DEFAULT_RESERVED_GB = 2.0

# Compressed-memory cliff: when free RAM is below this fraction of total,
# throughput often degrades even if the load "fits" by arithmetic.
COMPRESSED_FREE_FRAC = 0.28


@dataclass
class FitResult:
    verdict: str
    title: str
    detail: str
    blocks_load: bool
    weights_gb: float
    kv_gb: float
    act_gb: float
    reserved_gb: float
    total_gb: float
    free_gb: float
    total_ram_gb: float
    free_ram_gb: float
    ratio: float
    estimate_notes: list[str]
    what_if: dict[str, Any]
    calibrated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _memory_snapshot() -> tuple[float, float]:
    """Return (total_gb, available_gb) from the OS. Raises if unavailable."""
    try:
        import psutil
    except ImportError as e:
        raise RuntimeError(
            "psutil is required for Fit Engine memory reads; pip install psutil"
        ) from e
    vm = psutil.virtual_memory()
    return vm.total / (1024**3), vm.available / (1024**3)


def weights_gb_for_path(path: str | Path | None, explicit: float | None = None) -> tuple[float, str]:
    """Resolve weights size in GB. Prefer explicit, then sum safetensors on disk."""
    if explicit is not None and explicit > 0:
        return float(explicit), "explicit_weights_gb"
    if not path:
        return 0.0, "missing_path"
    p = Path(path)
    if not p.exists():
        return 0.0, "path_not_found"
    total = 0
    if p.is_file():
        total = p.stat().st_size
    else:
        for f in p.rglob("*.safetensors"):
            try:
                total += f.stat().st_size
            except OSError:
                continue
        if total == 0:
            for f in p.rglob("*"):
                if f.is_file() and f.suffix in {".npz", ".bin", ".gguf"}:
                    try:
                        total += f.stat().st_size
                    except OSError:
                        continue
    if total <= 0:
        return 0.0, "no_weight_files"
    return total / (1024**3), "disk_safetensors"


def estimate_kv_gb(ctx: int, kv_bits: int, *, n_layers: int = 48, n_kv_heads: int = 8, head_dim: int = 128) -> float:
    """KV cache size estimate in GB for a single sequence.

    bytes ≈ 2 (K+V) * layers * kv_heads * head_dim * ctx * (kv_bits/8)
    """
    ctx = max(1, int(ctx))
    kv_bits = max(1, int(kv_bits))
    bytes_ = 2 * n_layers * n_kv_heads * head_dim * ctx * (kv_bits / 8.0)
    return bytes_ / (1024**3)


def estimate_activation_gb(ctx: int) -> float:
    """Activation / workspace headroom prior (GB)."""
    ctx = max(1, int(ctx))
    return 1.2 + (ctx / 1024.0) * 0.05


def calibration_path() -> Path:
    return ensure_lab_dirs().root / "fit_calibration.json"


def load_calibration() -> dict[str, Any]:
    path = calibration_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_calibration(data: dict[str, Any]) -> Path:
    path = calibration_path()
    payload = dict(data)
    payload["updated_at"] = time.time()
    path.write_text(json.dumps(payload, indent=2))
    return path


def run_calibration_snapshot() -> dict[str, Any]:
    """One-time (or re-run) snapshot of machine memory for Fit priors.

    Does not load a model — records total/available and a reserved floor.
    """
    total_gb, free_gb = _memory_snapshot()
    used_gb = max(0.0, total_gb - free_gb)
    # Reserve at least DEFAULT or 8% of total for OS/UI.
    reserved = max(DEFAULT_RESERVED_GB, round(total_gb * 0.08, 2))
    data = {
        "total_ram_gb": round(total_gb, 3),
        "available_gb_at_calibration": round(free_gb, 3),
        "used_gb_at_calibration": round(used_gb, 3),
        "reserved_gb": reserved,
        "source": "psutil.virtual_memory",
    }
    save_calibration(data)
    return data


def predict(
    *,
    path: str | None = None,
    weights_gb: float | None = None,
    ctx: int = 32768,
    kv_bits: int = 8,
    n_layers: int = 48,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    free_ram_gb: float | None = None,
    total_ram_gb: float | None = None,
) -> FitResult:
    """Predict fit for a candidate load. Pure arithmetic + OS memory."""
    notes: list[str] = []
    cal = load_calibration()
    calibrated = bool(cal.get("reserved_gb"))

    if total_ram_gb is None or free_ram_gb is None:
        t, f = _memory_snapshot()
        total_ram_gb = float(total_ram_gb if total_ram_gb is not None else t)
        free_ram_gb = float(free_ram_gb if free_ram_gb is not None else f)
    else:
        total_ram_gb = float(total_ram_gb)
        free_ram_gb = float(free_ram_gb)

    w_gb, w_src = weights_gb_for_path(path, weights_gb)
    notes.append(f"weights_source={w_src}")
    if w_gb <= 0:
        notes.append("weights_gb_unknown_treat_as_zero")

    kv_gb = round(estimate_kv_gb(ctx, kv_bits, n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim), 3)
    act_gb = round(estimate_activation_gb(ctx), 3)
    notes.append("kv_and_activation_are_estimates_not_measured_peaks")

    reserved = float(cal.get("reserved_gb") or DEFAULT_RESERVED_GB)
    if not calibrated:
        notes.append("not_calibrated_using_default_reserved_gb")

    total_needed = round(w_gb + kv_gb + act_gb, 3)
    # Free budget after reserved floor (do not spend OS reserve).
    usable_free = max(0.0, free_ram_gb - reserved)
    ratio = (total_needed / usable_free) if usable_free > 0 else 999.0

    # Hard-fail prior: long context + low KV bits (MTLResource bind risk).
    if ctx >= MTL_CTX_THRESHOLD and kv_bits <= MTL_KV_BITS_THRESHOLD:
        verdict = "hard_fail"
        title = "Will hard-fail — resource ceiling risk"
        detail = (
            f"Context {ctx} with {kv_bits}-bit KV is past the MTLResource prior "
            f"(ctx≥{MTL_CTX_THRESHOLD}, kv≤{MTL_KV_BITS_THRESHOLD}). "
            "Raise KV bits or lower context before loading."
        )
        blocks = True
    elif usable_free <= 0 or ratio >= 1.0:
        verdict = "will_not_fit"
        title = "Will not fit"
        detail = (
            f"Needs ~{total_needed:.1f} GB (weights {w_gb:.1f} + KV {kv_gb:.1f} + act {act_gb:.1f}) "
            f"but only ~{usable_free:.1f} GB usable free after {reserved:.1f} GB reserved."
        )
        blocks = True
    elif ratio >= RATIO_DEGRADED:
        verdict = "will_not_fit"
        title = "Will not fit (tight)"
        detail = (
            f"{total_needed:.1f} GB needed vs {usable_free:.1f} GB usable free "
            f"(ratio {ratio:.2f}). Reduce context, KV bits, or free memory."
        )
        blocks = True
    elif ratio >= RATIO_COMFORTABLE or (free_ram_gb / max(total_ram_gb, 1e-6)) < COMPRESSED_FREE_FRAC:
        verdict = "degraded"
        title = "Fits with degraded throughput"
        detail = (
            f"{total_needed:.1f} GB of {usable_free:.1f} GB usable free — near the "
            "compressed-memory / headroom cliff; expect slower decode."
        )
        blocks = False
        if (free_ram_gb / max(total_ram_gb, 1e-6)) < COMPRESSED_FREE_FRAC:
            notes.append("system_already_near_compressed_memory_threshold")
    else:
        verdict = "comfortable"
        title = "Fits comfortably"
        detail = (
            f"{total_needed:.1f} GB of {usable_free:.1f} GB usable free — full throughput expected."
        )
        blocks = False

    # What-if: kv 4-bit and 64k context
    kv4 = estimate_kv_gb(ctx, 4, n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim)
    total_kv4 = w_gb + kv4 + act_gb
    kv64 = estimate_kv_gb(65536, kv_bits, n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim)
    what_if = {
        "kv_4bit_total_gb": round(total_kv4, 3),
        "kv_4bit_delta_gb": round(total_kv4 - total_needed, 3),
        "ctx_64k_kv_gb": round(kv64, 3),
        "ctx_64k_hard_fail_risk": kv_bits <= MTL_KV_BITS_THRESHOLD,
    }

    return FitResult(
        verdict=verdict,
        title=title,
        detail=detail,
        blocks_load=blocks,
        weights_gb=round(w_gb, 3),
        kv_gb=kv_gb,
        act_gb=act_gb,
        reserved_gb=reserved,
        total_gb=total_needed,
        free_gb=round(max(0.0, usable_free - total_needed), 3),
        total_ram_gb=round(total_ram_gb, 3),
        free_ram_gb=round(free_ram_gb, 3),
        ratio=round(ratio, 4),
        estimate_notes=notes,
        what_if=what_if,
        calibrated=calibrated,
    )
