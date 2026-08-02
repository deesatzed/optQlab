"""OptiQ LoRA — sensitivity-aware adapter fine-tuning for OptiQ-quantized models.

Standard LoRA uses uniform rank across all adapted layers. OptiQ's quantization
pass already measured each layer's KL-divergence sensitivity to decide its
weight bit-width. Layers OptiQ kept at higher precision (e.g. 8-bit because
they were sensitive to quantization) are also more expressive and benefit
from higher-rank LoRA; layers it aggressively quantized (e.g. 4-bit because
they were robust) can train with lower rank at the same quality.

This module:

  1. Reads ``optiq_metadata.json`` to recover per-layer bit assignments.
  2. Derives a per-layer LoRA rank from those bits (``by_bits`` scaling by
     default).
  3. Applies LoRA with the variable rank per layer to an mlx-lm loaded
     OptiQ model.
  4. Hands the prepared model to mlx-lm's trainer, which writes a
     PEFT-compatible adapter file on completion. The adapter can be
     loaded by ``mlx_lm.generate`` with ``--adapter-path``, or by OptiQ's
     serving hot-swap layer (Phase 4).

Entry points:
    optiq lora train <model_dir> --data <train_data> [--rank 16] [...]
    optiq lora info <adapter_dir>
"""

from .apply import apply_sensitivity_aware_lora
from .config import OptiqLoraConfig
from .sensitivity_rank import (
    rank_for_layer,
    read_per_layer_bits,
    summarize_rank_distribution,
)

__all__ = [
    "OptiqLoraConfig",
    "apply_sensitivity_aware_lora",
    "rank_for_layer",
    "read_per_layer_bits",
    "summarize_rank_distribution",
]
