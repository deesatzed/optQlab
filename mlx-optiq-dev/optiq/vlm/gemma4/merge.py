"""Gemma-4 multimodal merge helpers (vendored from mlx-vlm, BSD-3).

These project encoder features into the language hidden space and scatter them
into the text-token embedding sequence at the image/audio placeholder positions.
The merged sequence is then handed to mlx-lm's language model via its
``input_embeddings`` hook.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class RMSNormNoScale(nn.Module):
    """RMSNorm without a learnable scale (with_scale=False)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


def masked_scatter(input_tensor: mx.array, mask: mx.array, source: mx.array) -> mx.array:
    """Place ``source`` values into ``input_tensor`` wherever ``mask`` is True,
    in row-major order (matches PyTorch ``masked_scatter``)."""
    mask_flat = mask.flatten().astype(mx.int32)
    indices = mx.cumsum(mask_flat) - 1
    aligned = source.flatten()[indices % source.size]
    return mx.where(mask_flat, aligned, input_tensor.flatten()).reshape(input_tensor.shape)


class MultimodalEmbedder(nn.Module):
    """Projects vision/audio soft tokens into the language model space."""

    def __init__(self, embedding_dim: int, text_hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.embedding_projection = nn.Linear(embedding_dim, text_hidden_size, bias=False)
        self.embedding_pre_projection_norm = RMSNormNoScale(embedding_dim, eps=eps)

    def __call__(self, inputs_embeds: mx.array) -> mx.array:
        normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(normed)
