"""Config dataclass for OptiQ sensitivity-aware LoRA training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal


RankScaling = Literal["constant", "by_bits", "by_kl"]
TrainMethod = Literal["sft", "dpo"]


# Unsloth-aligned target modules (all 7 trainable linears per transformer
# block). The OptiQ default matches Unsloth's recipe so the only on-top
# differentiator is the per-layer rank overlay from rank_scaling="by_bits".
UNSLOTH_TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


# Preset bundles for ``--preset`` on the CLI. Each entry sets (rank, scale).
# Convention follows Unsloth's docs: alpha = rank (scale = 1.0) is the
# recommended default; alpha = 2*rank is the more aggressive variant for
# small ranks. With ``rank_scaling="by_bits"`` (the default), this is the
# BASE rank — sensitive (higher-bit) layers get bumped proportionally.
RANK_PRESETS: dict[str, tuple[int, float]] = {
    "small":   (8,   2.0),   # r=8,   alpha=16  — aggressive at low rank
    "default": (8,   1.0),   # r=8,   alpha=8   — Unsloth's small-model recipe + our overlay
    "medium":  (16,  1.0),   # r=16,  alpha=16  — Unsloth's go-to
    "large":   (32,  1.0),   # r=32,  alpha=32
    "xl":      (64,  1.0),   # r=64,  alpha=64
    "xxl":     (128, 1.0),   # r=128, alpha=128
}


@dataclass
class OptiqLoraConfig:
    """Per-run configuration for ``optiq lora train``.

    Attributes:
        rank: Base LoRA rank. For ``rank_scaling="constant"`` every adapted
            layer uses exactly this rank. For ``"by_bits"`` or ``"by_kl"``
            the rank is multiplied by a per-layer factor, with this value
            treated as the nominal rank for a 4-bit / median-sensitivity
            layer.
        scale: LoRA alpha / rank scaling factor. ``alpha = rank * scale``.
            Follows mlx-lm's convention.
        dropout: Dropout on the LoRA pathway during training.
        rank_scaling: Strategy for deriving per-layer rank from OptiQ's
            sensitivity measurements. See ``sensitivity_rank.py``.
        target_modules: Which linear layers inside each transformer block
            get adapted. Default follows mlx-lm's default for Qwen/Gemma
            (``q_proj`` and ``v_proj``). Pass a custom list to widen.
        adapt_experts: Adapt MoE expert pools (SwitchLinear). Off by default:
            each expert gets its own LoRA, so this multiplies the adapter
            size by the expert count. Attention projections are unaffected.
        num_layers: Number of transformer blocks (from the last) to adapt.
            Default ``16`` matches mlx-lm's stock behaviour and is the
            safe choice on consumer Apple Silicon. ``-1`` adapts all
            blocks — **do not set this on a 9 B+ model on a 36 GB Mac**:
            full-depth backward exceeds the GPU's 499 000 MTLResource
            cap and can hard-crash the machine (validated on Qwen3.5-9B,
            2026-04-24). Reserve ``-1`` for small models (≤ 2 B) or
            machines with higher GPU resource limits (M3 Ultra, CUDA).
        use_dora: If True, use DoRA instead of LoRA.
    """

    rank: int = 8
    scale: float = 1.0
    dropout: float = 0.0
    rank_scaling: RankScaling = "by_bits"

    # Training objective. "sft" = standard supervised fine-tuning with
    # cross-entropy loss on the response tokens (default). "dpo" = Direct
    # Preference Optimization on {prompt, chosen, rejected} triples; uses
    # the same adapted model with adapter scale temporarily zeroed for
    # the reference forward pass, so no second model load.
    method: TrainMethod = "sft"

    # DPO beta (KL constraint strength). Only used when method="dpo".
    # Standard DPO default (Rafailov et al. 2023).
    dpo_beta: float = 0.1

    # DPO learning rate. **Much** smaller than the SFT default (~10-40×
    # lower) because DPO is a preference-tuning step, not a from-scratch
    # supervised fit. The Rafailov 2023 paper and the HF TRL DPOTrainer
    # both use 5e-7 to 5e-5; 2e-4 (SFT-grade) on DPO collapses the policy
    # within ~100 iters by inflating per-step weight updates faster than
    # beta can constrain them, regardless of beta. ``resolve_learning_rate``
    # picks this value when ``method == "dpo"`` and the user didn't
    # override ``learning_rate`` from the SFT default.
    dpo_learning_rate: float = 5e-5

    # Linear warmup over the first N iters before holding (or decaying
    # via ``dpo_lr_schedule``). Default 10 % of ``iters``, with a 10-iter
    # floor, computed in ``resolve_warmup_iters``. Critical for DPO:
    # without warmup the first 1-3 steps move the policy hard enough
    # that the rewards-margin signal blows out and never recovers,
    # producing the loss=0 / rewards-drifting-negative pathology.
    dpo_warmup_iters: int | None = None

    # DPO LR schedule shape after warmup. "constant" holds the peak;
    # "cosine" anneals to 10 % of the peak over the remaining steps.
    # Default cosine matches every modern alignment recipe (DPO, IPO,
    # ORPO, KTO).
    dpo_lr_schedule: Literal["constant", "cosine"] = "cosine"

    # DPO loss variant. "sigmoid" = standard DPO (Rafailov 2023). "ipo" =
    # Identity-PO (Azar 2023): a squared loss that regresses the log-ratio
    # margin toward a bounded target 1/(2*beta) instead of pushing it to
    # infinity -- the fix for small-data margin explosion / collapse.
    dpo_loss: Literal["sigmoid", "ipo"] = "sigmoid"
    # cDPO label smoothing (Mitchell 2023). Only used with dpo_loss="sigmoid".
    # Treats preference labels as flipped with probability epsilon, which puts
    # a hard FLOOR under the loss so it cannot collapse to 0 by memorizing a
    # small set of trivially-separable pairs. 0.0 = plain DPO; 0.1-0.3 for
    # small / noisy preference data. Matches HF TRL DPOConfig.label_smoothing.
    dpo_label_smoothing: float = 0.0

    # Fused (chunked) DPO logp. Runs the transformer body once, then applies
    # the vocab head + log_softmax + label-gather in token chunks so the full
    # [B,L,V] logit tensor is never materialized (it is otherwise built 4x per
    # step: policy+reference x chosen+rejected). Lets DPO reach the same long
    # contexts as fused-CE SFT. Opt-in (not a hardcoded threshold) because the
    # context at which the plain path OOMs is VRAM-dependent -- ~2-4k on a 24GB
    # Mac, higher on bigger machines. Also settable via OPTIQ_FUSED_DPO=1.
    fused_dpo: bool = False

    # Fused cut-cross-entropy for SFT: same idea as fused_dpo, applied to the
    # LM-head loss. None = decide from the estimated logit-tensor size (see
    # trainer._decide_fused_ce); True/False force it. Also OPTIQ_FUSED_CE=0|1.
    fused_ce: bool | None = None
    target_modules: tuple[str, ...] = UNSLOTH_TARGET_MODULES
    # MoE expert pools are fused: one SwitchLinear per (block, projection)
    # holding every expert. mlx-lm's LoRASwitchLinear gives each expert its
    # own (A, B) pair, so adapting gate/up/down on a MoE multiplies the
    # adapter by num_experts. On Qwen3.5-122B (48 layers, 256 experts) that
    # is a 1.2 B-parameter adapter at r=8 versus 9.4 M for attention alone.
    # Nobody wants that by accident, so the expert pools are opt-in even
    # though gate/up/down are in the default target_modules (where, on a
    # dense model, they mean the ordinary MLP).
    adapt_experts: bool = False
    # num_layers=-1 (all transformer blocks) matches Unsloth/PEFT default
    # and lets the by_bits overlay see every layer's bit assignment, not
    # just the last 16. mlx-lm's original 16-block default was a hedge
    # against a since-fixed @mx.compile shape blowup; with that patched
    # (optiq/lora/trainer.py:_patch_out_mx_compile_in_mlx_lm_trainer)
    # all-layers is safe up through ~4B at batch=1, seq=512 on a 24 GB
    # Mac. For 9B+ models on 24 GB, drop back to a smaller num_layers
    # (or smaller seq / batch) since the GPU's MTLResource cap (499 000)
    # can still bite on full-depth backward at large model sizes.
    num_layers: int = -1
    use_dora: bool = False

    # Mask prompt tokens from the loss. When True, cross-entropy is
    # computed only over the assistant's response tokens; the prompt
    # (system + user turns + few-shot demos) is masked out. This is
    # what Unsloth, PEFT, llama-factory and every other production
    # SFT library does by default — without it, the LoRA learns to
    # memorize prompt boilerplate (chat-template tokens, few-shot
    # demos) and degrades the base model's task ability instead of
    # improving it. Requires the data to be in ``{"messages": [...]}``
    # (chat) or ``{"prompt": ..., "completion": ...}`` format so the
    # dataset class knows where the response begins.
    mask_prompt: bool = True

    # NEFTune (Noisy Embedding instruction Fine-Tuning, Jain et al. 2023).
    # During the SFT forward pass, adds uniform noise scaled by
    # ``alpha / sqrt(seq_len * embed_dim)`` to the token embeddings — and
    # nothing at inference — a cheap regularizer that improves instruction-
    # following on small datasets. ``0.0`` (default) disables it, matching
    # TRL / Unsloth (whose ``neftune_noise_alpha`` defaults to None); the
    # paper suggests 5-15 when enabled. SFT only, mirroring TRL, which applies
    # NEFTune in the SFTTrainer and not the DPOTrainer.
    neftune_noise_alpha: float = 0.0

    # Multi-turn / agentic SFT. mlx-lm's ChatDataset masks everything before the
    # LAST message (single-turn prompt->response), so for multi-turn agentic
    # trajectories ({"messages": ...} with many assistant turns interleaved with
    # tool/user turns) it trains ONLY the final turn and masks the real actions.
    # "auto" (default) detects multi-turn data and trains on EVERY assistant turn
    # (per-token mask, see optiq/lora/multiturn.py); True forces it on; False
    # keeps mlx-lm's single-turn behavior. SFT only.
    train_on_all_turns: bool | str = "auto"

    # Training hyperparameters (forwarded to mlx-lm's TrainingArgs)
    batch_size: int = 1
    # iters / learning_rate default to None = "resolve by method + data size".
    # iters:   None -> num_epochs (or the method default: 3 SFT, 1 DPO) x
    #          ceil(n_examples / batch_size). Set an int to force absolute iters.
    # lr:      None -> 2e-4 for SFT, dpo_learning_rate (5e-5) for DPO. Set a
    #          float to force it. Resolving here (not just in the CLI) means a
    #          programmatic OptiqLoraConfig(method="dpo") gets the DPO LR too,
    #          instead of silently inheriting the SFT 2e-4 and collapsing.
    iters: int | None = None
    num_epochs: float | None = None
    learning_rate: float | None = None
    # max_seq_length=512 covers ~99.8% of typical SFT data (GSM8K
    # measured: max 515 tokens) and is the highest value compatible
    # with num_layers=-1 + batch=1 on a 24 GB Mac for a 4B model.
    # Bump to 1024+ for longer-context tasks if you also reduce
    # num_layers or have more RAM.
    max_seq_length: int = 512
    grad_accumulation_steps: int = 1
    grad_checkpoint: bool = True
    val_batches: int = 25
    steps_per_report: int = 10
    steps_per_eval: int = 200
    steps_per_save: int = 100

    # Where to write the adapter (PEFT-compatible layout)
    adapter_path: str = "adapters"

    # Path to an existing adapter directory to mount as a *frozen* LoRA
    # alongside a *trainable* LoRA on the same layers. The textbook
    # SFT -> DPO continuation recipe: the policy forward sums the
    # frozen-SFT delta + the trainable-DPO delta on top of the base,
    # and the DPO reference forward zeroes only the trainable scale, so
    # KL is anchored against base + SFT (which is the SFT model) -
    # exactly the standard alignment-pipeline definition of the
    # reference. The trained adapter that gets saved contains only the
    # DPO delta, so it composes cleanly with the SFT adapter at serving
    # time via OptiQ's multi-LoRA registry.
    mount_adapter: str | None = None

    # Clear MLX's buffer reuse pool when it exceeds this many bytes,
    # checked between training steps. Mirrors mlx-lm's own
    # ``_clear_cache(threshold)`` pattern. ``0`` disables (mlx-lm's
    # native default). On 24 GB Macs ~2.4 GB is a reasonable cap;
    # we default to 10 % of total RAM so it scales across hardware.
    clear_cache_threshold: int = 0

    # --- Early stopping (optional) --------------------------------------
    # Stop training once the validation loss hasn't improved by more than
    # ``early_stopping_min_delta`` for ``early_stopping_patience`` consecutive
    # evaluations (evals fire every ``steps_per_eval``). ``0`` (default)
    # disables it entirely — training runs the full ``iters``. Requires a
    # validation set (``valid.jsonl``); with no val data there is nothing to
    # monitor and the setting is a no-op (with a warning). On stop, the best
    # adapter (already snapshotted under ``<adapter>/best`` by
    # BestAdapterCallback) is promoted to the top-level ``adapters.safetensors``
    # so the returned adapter is the best one, not the last (overfit) step.
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0

    # --- Experiment logging (optional) ----------------------------------
    # Log train/val metrics to an experiment tracker. ``None``/""(default)
    # disables it. ``"wandb"`` logs to Weights & Biases (needs ``pip install
    # wandb`` + a ``WANDB_API_KEY`` or ``wandb login``); ``"swanlab"`` is also
    # supported. Comma-separated to enable several. Reuses mlx-lm's own
    # reporting callbacks, chained under OptiQ's best-adapter / progress hooks
    # so the CLI log and Lab live-chart keep working alongside the tracker.
    report_to: str | None = None
    # W&B project name when ``report_to`` includes wandb.
    wandb_project: str = "optiq-lora"

    def to_mlx_lora_config(self) -> dict:
        """Subset of the config consumed by mlx-lm's ``linear_to_lora_layers``."""
        return {
            "rank": self.rank,
            "scale": self.scale,
            "dropout": self.dropout,
            "keys": None,  # resolved per-layer by apply_sensitivity_aware_lora
        }

    # Method-aware defaults, applied by the resolvers below so BOTH the CLI
    # and direct OptiqLoraConfig(...) construction land on sensible values.
    _DEFAULT_SFT_LR: ClassVar[float] = 2e-4      # Unsloth-aligned SFT LR
    # Default epochs per method. SFT gets 3 (small on-device datasets); DPO
    # gets 1 (a preference *nudge* on the SFT policy -- more epochs invite the
    # collapse pathology). Overridable via ``num_epochs`` or explicit ``iters``.
    _DEFAULT_EPOCHS: ClassVar[dict] = {"sft": 3.0, "dpo": 1.0, "vision": 3.0}

    def effective_learning_rate(self) -> float:
        """Peak LR the trainer should use. An explicit ``learning_rate`` wins;
        otherwise SFT -> 2e-4, DPO -> ``dpo_learning_rate`` (5e-5)."""
        if self.learning_rate is not None:
            return float(self.learning_rate)
        if self.method == "dpo":
            return float(self.dpo_learning_rate)
        return self._DEFAULT_SFT_LR

    def effective_iters(self, n_examples: int) -> int:
        """Absolute training iterations. An explicit ``iters`` wins; otherwise
        ``num_epochs`` (or the method default: 3 SFT / 1 DPO) x steps-per-epoch,
        where steps-per-epoch = ceil(n_examples / batch_size)."""
        if self.iters is not None:
            return int(self.iters)
        epochs = (self.num_epochs if self.num_epochs is not None
                  else self._DEFAULT_EPOCHS.get(self.method, 3.0))
        bs = max(1, int(self.batch_size))
        steps_per_epoch = max(1, -(-int(n_examples) // bs))   # ceil div
        return max(1, int(round(float(epochs) * steps_per_epoch)))

    def resolve_warmup_iters(self, iters: int | None = None) -> int:
        """Warmup iterations for the chosen method. SFT: none (mlx-lm default).
        DPO: ~10 % of the (resolved) ``iters``, floored at 2 -- without warmup
        the first preference steps wreck the reward-margin signal, but a fixed
        floor of 10 was half of a 20-step small-data run and starved it of
        learning. An explicit ``dpo_warmup_iters`` is honored as-is. Pass the
        resolved ``iters`` (from ``effective_iters``); falls back to
        ``self.iters`` for back-compat."""
        if self.method != "dpo":
            return 0
        if self.dpo_warmup_iters is not None:
            return max(0, int(self.dpo_warmup_iters))
        base = iters if iters is not None else (self.iters or 100)
        return min(max(2, int(base) // 10), 100)
