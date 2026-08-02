"""Gemma-4 unified vision config (vendored from mlx-vlm, BSD-3)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UnifiedVisionConfig:
    model_type: str = "gemma4_unified_vision"
    patch_size: int = 16
    pooling_kernel_size: int = 3
    model_patch_size: int = 48
    mm_embed_dim: int = 3840
    mm_posemb_size: int = 1120
    num_soft_tokens: int = 280
    rms_norm_eps: float = 1e-6
    output_proj_dims: int = 3840

    @classmethod
    def from_dict(cls, d: dict) -> "UnifiedVisionConfig":
        import dataclasses

        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})
