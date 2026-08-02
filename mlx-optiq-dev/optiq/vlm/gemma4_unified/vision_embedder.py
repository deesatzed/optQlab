"""Gemma-4 unified VisionEmbedder (vendored from mlx-vlm, BSD-3).

Encoder-free: image patches -> LayerNorm -> dense -> LayerNorm -> add 2D
position embedding -> LayerNorm. No transformer; the shared language backbone
does the rest.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import UnifiedVisionConfig


class VisionEmbedder(nn.Module):
    def __init__(self, config: UnifiedVisionConfig):
        super().__init__()
        patch_dim = config.model_patch_size * config.model_patch_size * 3
        self.patch_dim = patch_dim
        self.patch_ln1 = nn.LayerNorm(patch_dim)
        self.patch_dense = nn.Linear(patch_dim, config.mm_embed_dim)
        self.patch_ln2 = nn.LayerNorm(config.mm_embed_dim)
        self.pos_embedding = mx.zeros((config.mm_posemb_size, 2, config.mm_embed_dim))
        self.pos_norm = nn.LayerNorm(config.mm_embed_dim)

    def __call__(
        self,
        pixel_values: mx.array,
        image_position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        if pixel_values.ndim == 4 and pixel_values.shape[-1] == self.patch_dim:
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0], -1, self.patch_dim
            )
        hidden_states = self.patch_ln1(pixel_values)
        hidden_states = self.patch_dense(hidden_states)
        hidden_states = self.patch_ln2(hidden_states)

        if image_position_ids is not None:
            clamped = mx.maximum(image_position_ids, 0).astype(mx.int32)
            valid = (image_position_ids != -1).astype(hidden_states.dtype)
            x_pos = self.pos_embedding[clamped[..., 0], 0]
            y_pos = self.pos_embedding[clamped[..., 1], 1]
            hidden_states = hidden_states + (
                x_pos * mx.expand_dims(valid[..., 0], -1)
                + y_pos * mx.expand_dims(valid[..., 1], -1)
            )

        return self.pos_norm(hidden_states)
