"""Pixtral vision tower (vendored from mlx-vlm, BSD-3).

Kept behaviourally identical to ``mlx_vlm.models.pixtral.vision`` -- the tests in
``tests/test_mistral3_vision.py`` diff this module's output against mlx-vlm's on
real Devstral weights -- with two OptiQ-local changes:

  * attention goes through :func:`optiq.vlm.base.ensure_fused_sdpa` like every
    other vendored tower here (head_dim is 64 for Pixtral, so this is the plain
    fused kernel; the shim only matters if a future variant picks an odd size).
  * no ``sanitize()``. OptiQ fixes the ``patch_conv`` layout once, at sidecar
    build time (``optiq.vlm.sidecar._sanitize_conv``), so the published artifact
    is correct for every loader instead of only for the front-end that remembers
    to call sanitize on the way in.

Pixtral is *native resolution*: images of different sizes are packed into one
sequence and kept apart by a block-diagonal attention mask, so nothing here may
assume a fixed square input.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig
from ..base import ensure_fused_sdpa


def position_ids_in_meshgrid(patch_embeds_list, max_width: int) -> mx.array:
    """Row-major 2D patch positions, flattened and concatenated over images."""
    positions = []
    for patch in patch_embeds_list:
        height, width = patch.shape[0], patch.shape[1]
        h_grid, v_grid = mx.meshgrid(mx.arange(height), mx.arange(width), indexing="ij")
        ids = h_grid.reshape(-1, 1) * max_width + v_grid.reshape(-1, 1)
        positions.append(ids.flatten())
    return mx.concatenate(positions)


def generate_block_attention_mask(patch_counts, tensor: mx.array) -> mx.array:
    """Block-diagonal mask: every image attends to itself only."""
    seq_len = tensor.shape[1]
    d_min = -1e9  # MLX has no finfo; matches mlx-vlm's constant exactly.

    causal_mask = mx.full((seq_len, seq_len), vals=d_min)

    block_end_idx = mx.cumsum(mx.array(patch_counts))
    block_start_idx = mx.cumsum(
        mx.concatenate([mx.array([0]), mx.array(patch_counts[:-1])])
    )
    for start, end in zip(block_start_idx, block_end_idx):
        start, end = int(start), int(end)
        causal_mask[start:end, start:end] = 0

    causal_mask = mx.broadcast_to(
        causal_mask[None, None, :, :], (tensor.shape[0], 1, seq_len, seq_len)
    )
    return causal_mask.astype(tensor.dtype)


def rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return mx.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Attention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        if (dims % num_heads) != 0:
            raise ValueError(
                f"vision hidden size must divide the head count ({dims} % {num_heads})"
            )
        self.embed_dim = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.scale = self.head_dim ** -0.5

        self.k_proj = nn.Linear(dims, dims, bias=False)
        self.v_proj = nn.Linear(dims, dims, bias=False)
        self.q_proj = nn.Linear(dims, dims, bias=False)
        self.o_proj = nn.Linear(dims, dims, bias=False)

    def __call__(self, queries, keys, values, position_embeddings, mask=None):
        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        h = self.num_heads
        queries = queries.reshape(B, L, h, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, h, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, h, -1).transpose(0, 2, 1, 3)

        cos, sin = position_embeddings
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin, unsqueeze_dim=0)

        output = ensure_fused_sdpa(queries, keys, values, self.scale, mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dim, hidden = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dim = config.hidden_size
        self.attention = Attention(dim, config.num_attention_heads)
        self.attention_norm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
        self.feed_forward = MLP(config)
        self.ffn_norm = nn.RMSNorm(dim, eps=config.rms_norm_eps)

    def __call__(self, x, position_embeddings, mask=None):
        y = self.attention_norm(x)
        x = x + self.attention(y, y, y, position_embeddings, mask)
        return x + self.feed_forward(self.ffn_norm(x))


class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]


class PixtralRotaryEmbedding:
    """2D RoPE over the patch grid; the table covers the largest legal image."""

    def __init__(self, config: VisionConfig):
        self.dim = config.head_dim
        self.base = config.rope_theta
        side = config.image_size // config.patch_size
        freqs = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim)
        )
        h = mx.arange(side)
        w = mx.arange(side)
        freqs_h = mx.outer(h, freqs[::2]).astype(mx.float32)
        freqs_w = mx.outer(w, freqs[1::2]).astype(mx.float32)
        inv_freq = mx.concatenate(
            [
                mx.tile(freqs_h[:, None, :], (1, side, 1)),
                mx.tile(freqs_w[None, :, :], (side, 1, 1)),
            ],
            axis=-1,
        ).reshape(-1, self.dim // 2)
        self.inv_freq = mx.concatenate((inv_freq, inv_freq), axis=-1)

    def __call__(self, x: mx.array, position_ids: mx.array):
        emb = self.inv_freq[position_ids]
        return mx.cos(emb).astype(x.dtype), mx.sin(emb).astype(x.dtype)


class PixtralVisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.patch_conv = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.ln_pre = nn.RMSNorm(config.hidden_size)
        self.transformer = Encoder(config)
        self.patch_positional_embedding = PixtralRotaryEmbedding(config)

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
        image_sizes=None,
    ):
        """``x`` is NHWC (already transposed by the caller), possibly zero-padded
        to the largest image in the batch; ``image_sizes`` gives each image's real
        ``(height, width)`` in pixels so the padding can be cropped away."""
        if x.dtype != self.patch_conv.weight.dtype:
            x = x.astype(self.patch_conv.weight.dtype)

        if image_sizes is None:
            image_sizes = [(x.shape[1], x.shape[2])] * x.shape[0]
        else:
            image_sizes = [
                (int(s[0]), int(s[1]))
                for s in (
                    v.tolist() if hasattr(v, "tolist") else v for v in image_sizes
                )
            ]
        if len(image_sizes) != x.shape[0]:
            raise ValueError(
                f"image_sizes length ({len(image_sizes)}) must match batch size "
                f"({x.shape[0]})"
            )

        patch_embeds = self.patch_conv(x)
        patch_embeds_list = [
            emb[: size[0] // self.patch_size, : size[1] // self.patch_size]
            for emb, size in zip(patch_embeds, image_sizes)
        ]
        patch_embeds = mx.concatenate(
            [p.reshape(-1, p.shape[-1]) for p in patch_embeds_list], axis=0
        )[None, ...]
        patch_embeds = self.ln_pre(patch_embeds)

        position_ids = position_ids_in_meshgrid(
            patch_embeds_list,
            max_width=self.config.image_size // self.config.patch_size,
        )
        position_embedding = self.patch_positional_embedding(patch_embeds, position_ids)
        mask = generate_block_attention_mask(
            [p.shape[0] * p.shape[1] for p in patch_embeds_list], patch_embeds
        )

        encoder_states = (patch_embeds,) if output_hidden_states else None
        for layer in self.transformer.layers:
            patch_embeds = layer(patch_embeds, position_embedding, mask)
            if output_hidden_states:
                encoder_states = encoder_states + (patch_embeds,)

        return patch_embeds, encoder_states


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        if self.model_type not in ("pixtral", "clip_vision_model"):
            raise ValueError(f"Unsupported vision model type: {self.model_type}")
        self.vision_model = PixtralVisionModel(config)

    def __call__(self, x, output_hidden_states=None, image_sizes=None):
        return self.vision_model(
            x, output_hidden_states=output_hidden_states, image_sizes=image_sizes
        )
