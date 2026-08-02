"""Merge multiple LoRA adapters into a single composite adapter.

The merge is **mathematically exact**, not a low-rank approximation:
for two adapters with ranks ``r1`` and ``r2`` on the same layer, the
output adapter on that layer has rank ``r1 + r2``, and its forward
pass reproduces the sum of the original two LoRA residuals exactly.

Use case: the standard SFT -> DPO continuation pipeline produces two
adapters that get *stacked* at serve time (via ``optiq serve --adapter
sft --adapter dpo``, request body ``"adapter": "sft+dpo"``). For
shipping the result as a single drop-in artifact, this module folds
the stack into one adapter file that any LoRA-aware runtime can load.

Math
----

For one adapted layer with input ``x``:

  - adapter A residual: ``s_a * (x @ A_a @ B_a)``  shapes A_a:(in,r_a),  B_a:(r_a,out)
  - adapter B residual: ``s_b * (x @ A_b @ B_b)``  shapes A_b:(in,r_b),  B_b:(r_b,out)

Stack-sum we want at inference:

  ``s_a * (x @ A_a @ B_a) + s_b * (x @ A_b @ B_b)``

Equivalent single LoRA with rank r_a + r_b, scale=1.0:

  ``A_m = concat([A_a, A_b], axis=rank_axis_of_A)         shape (in, r_a + r_b)``
  ``B_m = concat([s_a * B_a, s_b * B_b], axis=rank_axis_of_B)  shape (r_a + r_b, out)``
  ``(x @ A_m) @ B_m  ==  s_a*(x @ A_a @ B_a) + s_b*(x @ A_b @ B_b)``

Layers that appear in only one source adapter are included verbatim
(with the original scale baked into ``B``). Layers that appear in
neither source adapter are not present in the output (the trainer
treats them as no-op LoRA).

The output adapter writes ``scale=1.0`` in its ``adapter_config.json``
because the per-source scales are folded into the B matrices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import mlx.core as mx


def _resolve_safetensors(adapter_path: str | Path) -> Path:
    """Accept a PEFT directory, an explicit safetensors file, or a
    directory containing ``best/adapters.safetensors``."""
    p = Path(adapter_path)
    if p.is_file():
        return p
    if p.is_dir():
        if (p / "best" / "adapters.safetensors").exists():
            return p / "best" / "adapters.safetensors"
        if (p / "adapters.safetensors").exists():
            return p / "adapters.safetensors"
    raise FileNotFoundError(
        f"{adapter_path}: expected adapters.safetensors or "
        f"best/adapters.safetensors"
    )


def _resolve_scale(adapter_path: str | Path) -> float:
    """Pull the LoRA scale (= alpha / r) from the sibling
    ``adapter_config.json``. Defaults to 1.0 if not found."""
    p = Path(adapter_path)
    candidates = []
    if p.is_file():
        candidates.append(p.parent / "adapter_config.json")
        candidates.append(p.parent.parent / "adapter_config.json")
    elif p.is_dir():
        candidates.append(p / "adapter_config.json")
        candidates.append(p / "best" / "adapter_config.json")
    for c in candidates:
        if c.exists():
            try:
                cfg = json.loads(c.read_text())
            except Exception:
                continue
            # OptiQ / mlx-lm convention: lora_parameters.scale
            if isinstance(cfg.get("lora_parameters"), dict):
                v = cfg["lora_parameters"].get("scale")
                if v is not None:
                    return float(v)
            # PEFT convention: alpha / r
            if "lora_alpha" in cfg and cfg.get("r"):
                try:
                    return float(cfg["lora_alpha"]) / float(cfg["r"])
                except Exception:
                    pass
            if "scale" in cfg:
                return float(cfg["scale"])
    return 1.0


def _rank_axis(arr: mx.array) -> int:
    """The rank dim is the smallest axis on a (typically rank-2) LoRA
    matrix. For ``LoRASwitchLinear`` outputs that carry a leading
    ``num_experts`` dim, the rank still ends up as the smallest axis
    in practice for any rank ``<= num_experts``."""
    return int(min(range(arr.ndim), key=lambda i: arr.shape[i]))


def merge_adapters(
    adapter_paths: Iterable[str | Path],
    output_dir: str | Path,
    scales: list[float] | None = None,
) -> dict:
    """Rank-concat merge of N LoRA adapters into a single adapter dir.

    Args:
        adapter_paths: ordered list of source adapter paths (PEFT
            directories or explicit ``adapters.safetensors`` files).
        output_dir: where to write the merged ``adapters.safetensors``
            and ``adapter_config.json``.
        scales: optional per-source scale override. If ``None``, each
            source's scale is recovered from its sibling
            ``adapter_config.json`` (falls back to 1.0). The merged
            adapter always reports ``scale=1.0`` because per-source
            scales are baked into ``lora_b``.

    Returns:
        ``{layers_merged: int, layers_only_in_one: int,
           total_keys_in: int, total_keys_out: int,
           source_paths: [...], source_scales: [...]}`` for logging.
    """
    paths = [Path(p) for p in adapter_paths]
    if len(paths) < 2:
        raise ValueError(
            "merge_adapters needs at least two source adapters; "
            "got {len(paths)}"
        )

    sf_paths = [_resolve_safetensors(p) for p in paths]
    weights_per_source = [mx.load(str(sp)) for sp in sf_paths]
    if scales is None:
        scales = [_resolve_scale(p) for p in paths]
    else:
        if len(scales) != len(paths):
            raise ValueError(
                f"scales length {len(scales)} != adapter count "
                f"{len(paths)}"
            )

    # Build the universe of modpaths (LoRA wrapper module names) seen
    # across any source.
    def _mods_in(weights: dict) -> set[str]:
        mods = set()
        for k in weights:
            if k.endswith(".lora_a"):
                mods.add(k[: -len(".lora_a")])
        return mods

    all_mods = set().union(*(_mods_in(w) for w in weights_per_source))

    merged: dict[str, mx.array] = {}
    stats = {
        "layers_merged": 0,
        "layers_only_in_one": 0,
        "total_keys_in": sum(len(w) for w in weights_per_source),
        "total_keys_out": 0,
        "source_paths": [str(p) for p in sf_paths],
        "source_scales": list(scales),
    }

    for modpath in sorted(all_mods):
        a_key = f"{modpath}.lora_a"
        b_key = f"{modpath}.lora_b"

        # Collect contributions from every source that has this layer.
        a_parts: list[mx.array] = []
        b_parts: list[mx.array] = []
        for weights, scale in zip(weights_per_source, scales):
            if a_key in weights and b_key in weights:
                a = weights[a_key]
                b = weights[b_key]
                # Bake the source scale into B so the merged adapter
                # can carry scale=1.0.
                if scale != 1.0:
                    b = (scale * b).astype(b.dtype)
                a_parts.append(a)
                b_parts.append(b)

        if not a_parts:
            continue  # no source had this layer (shouldn't happen)

        if len(a_parts) == 1:
            stats["layers_only_in_one"] += 1
            merged[a_key] = a_parts[0]
            merged[b_key] = b_parts[0]
        else:
            # Concatenate along the rank axis of each matrix. Rank
            # axis of A is the SMALLEST dim (typically axis=1 for
            # shape (in, r)); rank axis of B is also the smallest dim
            # (typically axis=0 for shape (r, out)).
            ax_a = _rank_axis(a_parts[0])
            ax_b = _rank_axis(b_parts[0])
            # Sanity-check shapes: every non-rank dim must match
            # across sources for the concatenation to be valid.
            for ap, bp in zip(a_parts[1:], b_parts[1:]):
                if (a_parts[0].shape[1 - ax_a] !=
                        ap.shape[1 - ax_a]):
                    raise ValueError(
                        f"layer {modpath}: incompatible non-rank "
                        f"A-dim "
                        f"({a_parts[0].shape} vs {ap.shape})"
                    )
                if (b_parts[0].shape[1 - ax_b] !=
                        bp.shape[1 - ax_b]):
                    raise ValueError(
                        f"layer {modpath}: incompatible non-rank "
                        f"B-dim "
                        f"({b_parts[0].shape} vs {bp.shape})"
                    )
            merged[a_key] = mx.concatenate(a_parts, axis=ax_a)
            merged[b_key] = mx.concatenate(b_parts, axis=ax_b)
            stats["layers_merged"] += 1

    stats["total_keys_out"] = len(merged)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), merged)

    # Build adapter_config.json. Use the first source's config as a
    # template, then override scale=1.0 and add an audit trail.
    first_cfg_path = (
        Path(paths[0]) / "adapter_config.json"
        if Path(paths[0]).is_dir()
        else Path(paths[0]).parent / "adapter_config.json"
    )
    base_cfg: dict = {}
    if first_cfg_path.exists():
        try:
            base_cfg = json.loads(first_cfg_path.read_text())
        except Exception:
            base_cfg = {}
    if "lora_parameters" not in base_cfg:
        base_cfg["lora_parameters"] = {}
    base_cfg["lora_parameters"]["scale"] = 1.0
    # The rank field in the config is informational; merged ranks
    # vary per layer. Set to the max source rank for clarity.
    src_ranks: list[int] = []
    for sp in paths:
        sc_path = (
            Path(sp) / "adapter_config.json"
            if Path(sp).is_dir()
            else Path(sp).parent / "adapter_config.json"
        )
        if sc_path.exists():
            try:
                c = json.loads(sc_path.read_text())
                lp = c.get("lora_parameters") or {}
                if lp.get("rank"):
                    src_ranks.append(int(lp["rank"]))
            except Exception:
                pass
    if src_ranks:
        base_cfg["lora_parameters"]["rank"] = sum(src_ranks)
    base_cfg["optiq_merge"] = {
        "sources": [str(p) for p in paths],
        "scales": list(scales),
        "stats": {
            "layers_merged": stats["layers_merged"],
            "layers_only_in_one": stats["layers_only_in_one"],
            "total_keys_out": stats["total_keys_out"],
        },
    }
    (out_dir / "adapter_config.json").write_text(
        json.dumps(base_cfg, indent=2) + "\n"
    )

    return stats
