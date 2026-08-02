"""Qwen3.5 vision config (vendored from mlx-vlm, BSD-3). deepstack disabled."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QwenVisionConfig:
    model_type: str = "qwen3_5"
    depth: int = 32
    hidden_size: int = 1280
    intermediate_size: int = 3420
    out_hidden_size: int = 1536
    num_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    in_channels: int = 3
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    num_position_embeddings: int = 2304
    window_size: int = 112
    fullatt_block_indexes: list = field(default_factory=lambda: [7, 15, 23, 31])
    deepstack_visual_indexes: list = field(default_factory=list)  # disabled for 3.5

    @classmethod
    def from_dict(cls, d: dict) -> "QwenVisionConfig":
        import dataclasses

        fields = {f.name for f in dataclasses.fields(cls)}
        cfg = cls(**{k: v for k, v in d.items() if k in fields})
        cfg.deepstack_visual_indexes = []  # qwen3.5 forces deepstack off
        return cfg


# the vendored vision.py imports the name ``VisionConfig``
VisionConfig = QwenVisionConfig
