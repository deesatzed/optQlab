"""Derive per-layer LoRA rank from OptiQ's sensitivity measurements.

OptiQ quantization records each layer's chosen bit-width in
``optiq_metadata.json`` under ``optimization.per_layer[<layer_key>].bits``.
Higher bits = OptiQ identified the layer as sensitive. Sensitive layers
benefit from higher LoRA rank during adaptation; robust layers can use
lower rank with negligible quality loss.

Two scaling strategies:

  ``by_bits``
      ``rank_for(layer) = base_rank * (bits / 4)``
      A 4-bit layer gets the base rank (e.g. 8), an 8-bit layer gets 2×,
      a 3-bit layer gets 0.75× (rounded up to at least 2).

  ``by_kl``
      ``rank_for(layer) = base_rank * clip(kl / median_kl, 0.5, 2.0)``
      Uses the raw KL sensitivity values. More granular than ``by_bits``
      but depends on the KL values being recorded in metadata.

  ``constant``
      All adapted layers use ``base_rank``. Equivalent to standard mlx-lm
      LoRA — provided as a baseline for A/B testing sensitivity-aware.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .config import OptiqLoraConfig, RankScaling


MIN_RANK = 2  # MLX requires rank >= 2 for its LoRA implementation.


def _load_meta(model_dir: str | Path) -> dict:
    """Load optiq_metadata.json. Handles models on disk OR HF repo ids."""
    model_dir = str(model_dir)
    local = Path(model_dir) / "optiq_metadata.json"
    if local.exists():
        return json.loads(local.read_text())
    # HF repo id — fetch the metadata file only
    if "/" in model_dir and not model_dir.startswith(("./", "/", "~")):
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(model_dir, "optiq_metadata.json")
            return json.loads(Path(p).read_text())
        except Exception:
            return {}
    return {}


def _extract_per_layer(meta: dict) -> dict:
    """``per_layer`` has lived at different paths across OptiQ versions.

    Accepts all known shapes:

      * v0.0.5+: top-level ``per_layer``
      * earlier: ``optimization.per_layer``
    """
    if isinstance(meta.get("per_layer"), dict):
        return meta["per_layer"]
    opt = meta.get("optimization") or {}
    if isinstance(opt.get("per_layer"), dict):
        return opt["per_layer"]
    return {}


def read_per_layer_bits(model_dir: str | Path) -> dict[str, int]:
    """Return ``{layer_key: bits}`` read from ``optiq_metadata.json``.

    Layer keys match the form OptiQ wrote (e.g.
    ``model.layers.0.self_attn.q_proj`` or the VLM-wrapped
    ``language_model.model.layers.0.self_attn.q_proj``). Empty dict if
    no metadata file exists.
    """
    per_layer = _extract_per_layer(_load_meta(model_dir))
    out: dict[str, int] = {}
    for layer_key, cfg in per_layer.items():
        if isinstance(cfg, dict) and "bits" in cfg:
            out[layer_key] = int(cfg["bits"])
    return out


def read_per_layer_kl(model_dir: str | Path) -> dict[str, float]:
    """Return ``{layer_key: kl_divergence}`` if sensitivity scores were
    recorded alongside the bit assignments. Empty dict if not recorded
    (older OptiQ versions only stored bits).
    """
    per_layer = _extract_per_layer(_load_meta(model_dir))
    out: dict[str, float] = {}
    for layer_key, cfg in per_layer.items():
        if isinstance(cfg, dict) and "kl" in cfg:
            out[layer_key] = float(cfg["kl"])
    return out


def rank_for_layer(
    layer_key: str,
    config: OptiqLoraConfig,
    bits_map: dict[str, int],
    kl_map: dict[str, float] | None = None,
) -> int:
    """Compute LoRA rank for ``layer_key`` given OptiQ sensitivity data.

    Unknown layers (not present in metadata) fall back to ``config.rank``.
    Returned rank is always >= ``MIN_RANK``.
    """
    base = config.rank

    if config.rank_scaling == "constant":
        return max(MIN_RANK, base)

    if config.rank_scaling == "by_bits":
        bits = bits_map.get(layer_key)
        if bits is None:
            return max(MIN_RANK, base)
        # 4-bit → base, 8-bit → 2× base, 3-bit → 0.75× base, etc.
        scaled = base * (bits / 4.0)
        return max(MIN_RANK, int(math.ceil(scaled)))

    if config.rank_scaling == "by_kl":
        if not kl_map:
            # Fall back to by_bits if KL wasn't recorded
            return rank_for_layer(layer_key, _clone_with(config, "by_bits"),
                                  bits_map, None)
        kl = kl_map.get(layer_key)
        if kl is None:
            return max(MIN_RANK, base)
        vals = sorted(kl_map.values())
        median = vals[len(vals) // 2] if vals else 1.0
        factor = max(0.5, min(2.0, (kl / median) if median else 1.0))
        return max(MIN_RANK, int(math.ceil(base * factor)))

    raise ValueError(f"unknown rank_scaling: {config.rank_scaling!r}")


def _clone_with(config: OptiqLoraConfig, rank_scaling: RankScaling) -> OptiqLoraConfig:
    # Cheap clone for fallback path without importing dataclasses.replace
    c = OptiqLoraConfig(**config.__dict__)
    c.rank_scaling = rank_scaling
    return c


def summarize_rank_distribution(
    config: OptiqLoraConfig,
    bits_map: dict[str, int],
    kl_map: dict[str, float] | None = None,
    target_module_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
) -> dict:
    """Summarize how many layers get each rank. Useful for logging before
    training kicks off."""
    # Callers may pass a list (from a JSON-loaded config). ``str.endswith``
    # only accepts ``str`` or ``tuple`` — coerce to be permissive.
    if isinstance(target_module_suffixes, list):
        target_module_suffixes = tuple(target_module_suffixes)

    counts: dict[int, int] = {}
    matched_layers: list[tuple[str, int]] = []
    for layer_key in bits_map:
        if not layer_key.endswith(target_module_suffixes):
            continue
        r = rank_for_layer(layer_key, config, bits_map, kl_map)
        counts[r] = counts.get(r, 0) + 1
        matched_layers.append((layer_key, r))
    return {
        "rank_counts": dict(sorted(counts.items())),
        "total_adapted": len(matched_layers),
        "examples": matched_layers[:6],
    }
