"""Apply sensitivity-aware LoRA to an mlx-lm model.

mlx-lm's built-in ``linear_to_lora_layers`` applies a uniform rank across
every layer. This module walks each transformer block individually,
computes that block's target LoRA rank from OptiQ's metadata, and adapts
each matching linear module with that rank.

We reuse mlx-lm's ``LoRALinear`` / ``DoRALinear`` classes so the resulting
adapter is load-compatible with ``mlx_lm.generate`` and with PEFT tooling.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .config import OptiqLoraConfig
from .sensitivity_rank import (
    read_per_layer_bits,
    read_per_layer_kl,
    rank_for_layer,
)


logger = logging.getLogger(__name__)


def _match_target_suffix(name: str, suffixes: Tuple[str, ...]) -> bool:
    # The mlx-lm naming scheme for adapted layers is like
    # "self_attn.q_proj" — compare against the module-name tail.
    tail = name.split(".")[-1]
    return tail in suffixes


def apply_sensitivity_aware_lora(
    model: nn.Module,
    model_dir: str,
    config: OptiqLoraConfig,
) -> Dict[str, int]:
    """Replace selected linear modules with LoRA-wrapped versions.

    Args:
        model: Loaded mlx-lm model (must have ``.layers`` — the standard
            shape for all causal-LM variants).
        model_dir: Path to the model directory containing
            ``optiq_metadata.json``.
        config: OptiQ LoRA configuration.

    Returns:
        ``{layer_key: applied_rank}`` map of layers that were actually
        wrapped. Useful for logging / saving alongside the adapter.
    """
    # Lazy-import the mlx-lm LoRA layer classes. Kept lazy because at
    # import time the module-level references in mlx-lm touch MLX arrays.
    # mlx-lm 0.31.x exposes LoRALinear / LoRASwitchLinear / LoRAEmbedding.
    # SwitchLinear / QuantizedSwitchLinear are the dense / quantized MoE
    # expert-pool modules (one tensor per (block, projection) holding all
    # experts fused along axis 0). LoRASwitchLinear adds a per-expert
    # LoRA delta along that same axis — i.e. its lora_a / lora_b carry a
    # leading ``num_experts`` dim so each expert gets its own (A, B)
    # slice. This matches mlx-lm / current-Unsloth recipe and, on a
    # 128-expert pool, multiplies LoRA params per projection by
    # ``num_experts``. ESFT (Expert-Selective Fine-Tuning, adapt only
    # the top-K most active experts) is a future enhancement layered on
    # top.
    # DoRA variants are not in 0.31.x as first-class classes — mlx-lm gates
    # DoRA via a flag in linear_to_lora_layers. For our sensitivity-aware
    # apply path we only support LoRA today; DoRA support is tracked as a
    # future enhancement.
    from mlx_lm.tuner.lora import (  # type: ignore
        LoRALinear,
        LoRASwitchLinear,
        SwitchLinear,
        QuantizedSwitchLinear,
    )
    if config.use_dora:
        raise NotImplementedError(
            "DoRA isn't supported by optiq.lora.apply in mlx-lm 0.31.x; "
            "drop --use-dora to train a LoRA adapter instead."
        )

    bits_map = read_per_layer_bits(model_dir)
    kl_map = read_per_layer_kl(model_dir) or None

    # Qwen3_5 / Gemma-4 VLM wrappers expose layers via ``.language_model.model.layers``;
    # pure-LLM variants expose ``.model.layers``. Walk both.
    blocks = _find_transformer_blocks(model)

    applied: Dict[str, int] = {}
    n_skipped_experts = 0

    # Adapt only the last N blocks (matching mlx-lm's convention); -1 = all
    if config.num_layers == -1:
        target_blocks = blocks
    else:
        target_blocks = blocks[-max(config.num_layers, 0):]

    for block_idx_in_model, (layer_idx, block) in enumerate(target_blocks):
        adapted_modules: List[Tuple[str, nn.Module]] = []
        for name, module in block.named_modules():
            if not _match_target_suffix(name, config.target_modules):
                continue

            # Pick the matching LoRA wrapper for the base module type.
            # Plain Linear / QuantizedLinear -> LoRALinear (dense path).
            # SwitchLinear / QuantizedSwitchLinear -> LoRASwitchLinear,
            # which adds a per-expert LoRA delta (one (A, B) pair per
            # expert) across the fused experts in that block × projection.
            if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
                wrapper_cls = LoRALinear
            elif isinstance(module, (SwitchLinear, QuantizedSwitchLinear)):
                # A MoE expert pool. LoRASwitchLinear gives every expert its
                # own (A, B), so this multiplies the adapter by the expert
                # count -- 128x on a 256-expert model, because gate/up/down
                # are three of the seven default target_modules. Opt-in only.
                if not config.adapt_experts:
                    n_skipped_experts += 1
                    continue
                wrapper_cls = LoRASwitchLinear
            else:
                continue

            # Per-layer rank. The OptiQ metadata key follows the model
            # weight-key convention — we probe a few prefix shapes to be
            # resilient across VLM-wrapped vs pure-LLM layouts.
            r = _resolve_layer_rank(
                layer_idx, name, config, bits_map, kl_map
            )

            adapted = wrapper_cls.from_base(
                module,
                r=r,
                scale=config.scale,
                dropout=config.dropout,
            )
            adapted_modules.append((name, adapted))
            applied[f"layer_{layer_idx}.{name}"] = r

        if adapted_modules:
            block.update_modules(tree_unflatten(adapted_modules))

    if n_skipped_experts:
        logger.info(
            "Skipped %d MoE expert pool(s): each expert would get its own "
            "LoRA, multiplying the adapter by the expert count. Attention "
            "projections are adapted as usual. Pass adapt_experts=True "
            "(--adapt-experts) to adapt the experts anyway.",
            n_skipped_experts,
        )

    n_params = sum(
        v.size for _, v in tree_flatten(model.trainable_parameters())
    )
    logger.info(
        "LoRA: %d module(s) adapted, %.1fM trainable parameters",
        len(applied), n_params / 1e6,
    )

    return applied


def _find_transformer_blocks(model: nn.Module) -> List[Tuple[int, nn.Module]]:
    """Return ``[(layer_idx, block), ...]`` covering the text transformer.

    Handles both the pure-LLM shape (``model.model.layers``) and the
    VLM-wrapped shape (``model.language_model.model.layers``).
    """
    # Pure LLM shape
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return list(enumerate(inner.layers))

    # VLM-wrapped shape
    lm = getattr(model, "language_model", None)
    if lm is not None:
        inner = getattr(lm, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return list(enumerate(inner.layers))

    # Hybrid backbone shape (NVIDIA Nemotron-H: Mamba2 + attention blocks
    # registered under ``backbone.layers``). The attention blocks carry the
    # usual q/k/v/o + MLP projections; the Mamba blocks have none of the
    # target modules, so the apply loop simply skips them.
    bb = getattr(model, "backbone", None)
    if bb is not None and hasattr(bb, "layers"):
        return list(enumerate(bb.layers))

    # Some architectures expose the block list directly on the top model
    # (e.g. a ``self.layers`` attribute rather than a nested submodule).
    top_layers = getattr(model, "layers", None)
    if isinstance(top_layers, (list, tuple)) and top_layers:
        return list(enumerate(top_layers))

    raise RuntimeError(
        "Could not locate transformer blocks on model. Expected one of "
        "``model.model.layers`` (pure LLM), "
        "``model.language_model.model.layers`` (VLM-wrapped), "
        "``model.backbone.layers`` (Nemotron-H hybrid), or a top-level "
        "``model.layers`` list. "
        f"Got model class {type(model).__name__}."
    )


def _resolve_layer_rank(
    layer_idx: int,
    module_suffix: str,
    config: OptiqLoraConfig,
    bits_map: Dict[str, int],
    kl_map: Dict[str, float] | None,
) -> int:
    """Look up rank for this specific (layer_idx, module) pair.

    The metadata key written by ``optiq convert`` is the on-disk weight
    name with ``.weight`` stripped. Different mlx-lm model classes pack
    layers under different prefixes; we probe every shape OptiQ has been
    observed to emit.
    """
    candidates = [
        # Pure LLM (Qwen3 base, Gemma-4 text-only)
        f"model.layers.{layer_idx}.{module_suffix}",
        # VLM-wrapped (Gemma-4 multimodal, Qwen3.5/3.6) — two orderings
        # exist in the wild; older OptiQ runs wrote one, newer ones the
        # other depending on the underlying mlx-lm version's sanitizer.
        f"model.language_model.layers.{layer_idx}.{module_suffix}",
        f"language_model.model.layers.{layer_idx}.{module_suffix}",
        # Hybrid backbone (NVIDIA Nemotron-H): weights live under backbone.layers
        f"backbone.layers.{layer_idx}.{module_suffix}",
        # Bare form (rare; some early OptiQ versions)
        f"layers.{layer_idx}.{module_suffix}",
    ]
    for key in candidates:
        if key in bits_map:
            return rank_for_layer(key, config, bits_map, kl_map)
    # Fallback: use base rank if nothing matched
    return rank_for_layer("<unknown>", config, bits_map, kl_map)


# ---------------------------------------------------------------------------
# Stacked LoRA: mounted SFT (frozen) + DPO delta (trainable)
# ---------------------------------------------------------------------------
#
# The textbook SFT -> DPO continuation recipe is: train SFT first, then
# train DPO with the reference distribution = SFT (not = base). Setting
# the LoRA scale to 0 on a single LoRA initialized from SFT weights
# (the simpler `--init-from-adapter` path) approximates this but anchors
# the KL term against the base, not against SFT. The stacked path below
# is the principled fix: keep the SFT LoRA active and frozen, then add a
# *second* LoRA on top that's trainable. Toggling the DPO scale to 0
# for the reference forward yields base + SFT — the correct DPO
# reference distribution.


class StackedLoRALinear(nn.Module):
    """Base + frozen SFT-LoRA delta + trainable DPO-LoRA delta.

    Forward pass::

        y = base(x)
            + sft_scale * (x @ sft_a @ sft_b)            # frozen
            + dpo_scale * ((dropout(x) @ lora_a) @ lora_b)  # trainable

    The trainable attributes are deliberately named ``lora_a`` /
    ``lora_b`` / ``scale`` so the existing DPO trainer helpers
    (``_iter_lora_layers``, ``_set_lora_scale``,
    ``model.trainable_parameters()``) pick them up unchanged. The frozen
    slot uses ``sft_a`` / ``sft_b`` / ``sft_scale``; they're MX-arrays
    so MLX's parameter machinery sees them, then the apply helper calls
    ``module.freeze(keys=["sft_a", "sft_b"])`` to mark them
    non-trainable.

    Use ``from_base_and_sft(...)`` to construct; do not call the raw
    constructor with externally-supplied tensors directly.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        sft_r: int,
        dpo_r: int,
        sft_scale: float,
        dpo_scale: float,
        dropout: float,
    ):
        super().__init__()
        import math as _math
        # frozen slot — populated by from_base_and_sft after construction
        import mlx.core as mx
        self.sft_a = mx.zeros((input_dims, sft_r))
        self.sft_b = mx.zeros((sft_r, output_dims))
        self.sft_scale = float(sft_scale)
        # trainable slot — standard LoRA init: A uniform, B zero
        bound = 1.0 / _math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-bound, high=bound, shape=(input_dims, dpo_r))
        self.lora_b = mx.zeros((dpo_r, output_dims))
        self.scale = float(dpo_scale)
        self.dropout = nn.Dropout(p=dropout)
        # ``self.linear`` set externally to point at the base
        # (QuantizedLinear / Linear); kept out of the init signature so
        # we can reuse the original base module without re-instantiating.
        self.linear = None  # type: ignore[assignment]

    @staticmethod
    def from_base_and_sft(
        base_module: nn.Module,
        sft_a,
        sft_b,
        sft_scale: float,
        dpo_r: int,
        dpo_scale: float = 1.0,
        dropout: float = 0.0,
    ) -> "StackedLoRALinear":
        """Construct a stacked-LoRA wrapper around ``base_module``.

        ``sft_a`` / ``sft_b`` are the LoRA matrices loaded from the SFT
        adapter (shapes ``(in, sft_r)`` and ``(sft_r, out)`` respectively).
        ``dpo_r`` is the rank of the new trainable delta (typically the
        same as ``sft_r`` but doesn't have to be).
        """
        # Derive in/out dims from the base module the same way mlx-lm's
        # LoRALinear.from_base does.
        output_dims, input_dims = base_module.weight.shape
        if isinstance(base_module, nn.QuantizedLinear):
            input_dims = input_dims * 32 // base_module.bits
        sft_r = sft_a.shape[-1]
        m = StackedLoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            sft_r=sft_r,
            dpo_r=dpo_r,
            sft_scale=sft_scale,
            dpo_scale=dpo_scale,
            dropout=dropout,
        )
        m.linear = base_module
        m.sft_a = sft_a
        m.sft_b = sft_b
        return m

    def __call__(self, x):
        import mlx.core as mx  # noqa: F401 — keep symmetric with LoRALinear
        y = self.linear(x)
        # Frozen SFT contribution
        z_sft = (x @ self.sft_a) @ self.sft_b
        z_sft = (self.sft_scale * z_sft).astype(x.dtype)
        # Trainable DPO contribution
        z_dpo = (self.dropout(x) @ self.lora_a) @ self.lora_b
        z_dpo = (self.scale * z_dpo).astype(x.dtype)
        return y + z_sft + z_dpo


def apply_stacked_lora_for_dpo(
    model: nn.Module,
    model_dir: str,
    config: OptiqLoraConfig,
    sft_adapter_path: str,
) -> Tuple[Dict[str, int], dict]:
    """Wrap the model with ``StackedLoRALinear`` for DPO continuation.

    For every layer that gets adapted, mounts an SFT-LoRA (frozen,
    loaded from ``sft_adapter_path``) and a fresh trainable DPO-LoRA at
    the rank ``config`` would produce on that layer. Layers that the
    SFT adapter doesn't cover fall back to plain ``LoRALinear``
    (zero-init), so a DPO-from-mounted-SFT run still adapts every layer
    the rest of OptiQ would.

    Returns ``(applied_ranks, stats)`` where applied_ranks maps
    ``layer_<i>.<modname>`` to the DPO rank and stats summarizes how
    many layers got stacked vs plain LoRA.
    """
    import mlx.core as mx
    from pathlib import Path as _Path

    from mlx_lm.tuner.lora import (  # type: ignore
        LoRALinear,
        LoRASwitchLinear,
        SwitchLinear,
        QuantizedSwitchLinear,
    )

    # Resolve SFT adapter file (same logic as _load_init_adapter_weights).
    p = _Path(sft_adapter_path)
    if p.is_dir():
        if (p / "best" / "adapters.safetensors").exists():
            sf_path = p / "best" / "adapters.safetensors"
        elif (p / "adapters.safetensors").exists():
            sf_path = p / "adapters.safetensors"
        else:
            raise FileNotFoundError(
                f"--mount-adapter {sft_adapter_path}: expected "
                f"adapters.safetensors or best/adapters.safetensors"
            )
    elif p.is_file():
        sf_path = p
    else:
        raise FileNotFoundError(
            f"--mount-adapter {sft_adapter_path}: path does not exist"
        )

    # Try to recover the SFT scale from the sibling adapter_config.json
    # if present; otherwise default to mlx-lm's stock value (1.0).
    import json as _json
    sft_scale = 1.0
    cfg_candidates = [sf_path.parent / "adapter_config.json",
                      sf_path.parent.parent / "adapter_config.json"]
    for c in cfg_candidates:
        if c.exists():
            try:
                cfg_json = _json.loads(c.read_text())
                # PEFT uses ``lora_alpha`` and ``r``; scale = alpha / r.
                # mlx-lm's optiq sidecar carries ``scale`` directly.
                if "scale" in cfg_json:
                    sft_scale = float(cfg_json["scale"])
                elif "lora_alpha" in cfg_json and "r" in cfg_json \
                        and cfg_json["r"]:
                    sft_scale = float(cfg_json["lora_alpha"]) \
                        / float(cfg_json["r"])
                break
            except Exception:
                pass

    sft_weights: dict = mx.load(str(sf_path))

    # The DPO rank schedule comes from OptiQ's sensitivity metadata
    # (the same path apply_sensitivity_aware_lora walks).
    bits_map = read_per_layer_bits(model_dir)
    kl_map = read_per_layer_kl(model_dir) or None

    blocks = _find_transformer_blocks(model)
    target_blocks = (blocks if config.num_layers == -1
                     else blocks[-max(config.num_layers, 0):])

    applied: Dict[str, int] = {}
    stats = {"stacked": 0, "plain_lora": 0, "skipped_unsupported": 0,
             "sft_scale": sft_scale}

    for _, (layer_idx, block) in enumerate(target_blocks):
        wrapped: List[Tuple[str, nn.Module]] = []
        for name, module in block.named_modules():
            if not _match_target_suffix(name, config.target_modules):
                continue
            dpo_r = _resolve_layer_rank(
                layer_idx, name, config, bits_map, kl_map)

            # ``named_modules`` returns block-local names; the SFT
            # adapter's keys are model-global. Build the global key
            # the same way _find_transformer_blocks discovered the
            # block's home prefix.
            global_prefixes = [
                f"model.layers.{layer_idx}.",
                f"model.language_model.layers.{layer_idx}.",
                f"language_model.model.layers.{layer_idx}.",
                f"layers.{layer_idx}.",
            ]
            sft_a = sft_b = None
            for prefix in global_prefixes:
                a_key = f"{prefix}{name}.lora_a"
                b_key = f"{prefix}{name}.lora_b"
                if a_key in sft_weights and b_key in sft_weights:
                    sft_a, sft_b = sft_weights[a_key], sft_weights[b_key]
                    break

            if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
                if sft_a is not None and sft_b is not None:
                    new_mod = StackedLoRALinear.from_base_and_sft(
                        module,
                        sft_a=sft_a, sft_b=sft_b, sft_scale=sft_scale,
                        dpo_r=dpo_r, dpo_scale=config.scale,
                        dropout=config.dropout,
                    )
                    stats["stacked"] += 1
                else:
                    # SFT didn't adapt this projection; just stack a
                    # plain LoRA so DPO can still adapt it. Falls back
                    # to a zero-init delta on top of base.
                    new_mod = LoRALinear.from_base(
                        module, r=dpo_r, scale=config.scale,
                        dropout=config.dropout,
                    )
                    stats["plain_lora"] += 1
                wrapped.append((name, new_mod))
                applied[f"layer_{layer_idx}.{name}"] = dpo_r
            elif isinstance(module, (SwitchLinear, QuantizedSwitchLinear)):
                # MoE: defer to plain LoRA for now (per-expert stacking
                # is a bigger refactor). Logs but doesn't error.
                wrapped.append((name, LoRASwitchLinear.from_base(
                    module, r=dpo_r, scale=config.scale,
                    dropout=config.dropout,
                )))
                applied[f"layer_{layer_idx}.{name}"] = dpo_r
                stats["plain_lora"] += 1
                if sft_a is not None:
                    stats["skipped_unsupported"] += 1
        if wrapped:
            block.update_modules(tree_unflatten(wrapped))

    # Freeze the SFT slot in every StackedLoRALinear so only the DPO
    # delta gets gradients. We freeze the entire model first (mirroring
    # mlx-lm's lora.py:224 pattern) and then unfreeze the trainable
    # LoRA params by name.
    model.freeze(recurse=True)
    model.unfreeze(keys=["lora_a", "lora_b"], recurse=True)
    # ``LoRASwitchLinear`` carries dropout layer too; nothing else needs
    # to be unfrozen.

    return applied, stats
