"""Mistral-3 multimodal projector (vendored from mlx-vlm, BSD-3).

RMSNorm -> learned 2x2 patch merge -> Linear -> GELU -> Linear, taking the
Pixtral tower's last hidden state into the language hidden size.

The merge is where the two implementations could most easily disagree, so it is
worth being precise about what ``unfold`` does. mlx-vlm reproduces PyTorch's
``nn.functional.unfold`` with a Python double loop over blocks, which is
correct but costs one graph node per block (a 55x55 patch grid is 756 blocks x
4 gathers). The same tensor comes out of a single reshape/transpose, which is
what :func:`unfold_2x2` does; ``test_mistral3_vision.py`` diffs it against
mlx-vlm's loop to keep that claim honest.

Column ordering inside a block is ``c * k*k + di * k + dj`` (channel-major, then
row, then column within the kernel), and blocks are emitted row-major over the
merged grid. Get either wrong and the model still runs -- it just sees scrambled
patches.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def unfold_2x2(grid: mx.array, k: int) -> mx.array:
    """im2col for a non-overlapping ``k x k`` kernel (kernel == stride).

    Args:
        grid: ``[C, H, W]`` with ``H`` and ``W`` divisible by ``k``.

    Returns:
        ``[C * k * k, (H // k) * (W // k)]`` -- PyTorch ``unfold`` layout with the
        batch axis dropped.
    """
    c, h, w = grid.shape
    if h % k or w % k:
        raise ValueError(f"grid {h}x{w} is not divisible by the merge size {k}")
    x = grid.reshape(c, h // k, k, w // k, k)      # [C, h', di, w', dj]
    x = x.transpose(0, 2, 4, 1, 3)                 # [C, di, dj, h', w']
    return x.reshape(c * k * k, (h // k) * (w // k))


class Mistral3PatchMerger(nn.Module):
    """Learned merge of ``spatial_merge_size ** 2`` neighbouring patches."""

    def __init__(self, hidden_size: int, spatial_merge_size: int, patch_size: int):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.merging_layer = nn.Linear(
            hidden_size * spatial_merge_size ** 2, hidden_size, bias=False
        )

    def __call__(self, image_features: mx.array, image_sizes) -> mx.array:
        grids = [
            (int(s[0]) // self.patch_size, int(s[1]) // self.patch_size)
            for s in (
                v.tolist() if hasattr(v, "tolist") else v for v in image_sizes
            )
        ]

        if image_features.ndim == 3 and image_features.shape[0] == 1:
            image_features = image_features.squeeze(0)
        if image_features.ndim != 2:
            raise ValueError(
                f"expected image_features [tokens, dim], got {image_features.shape}"
            )
        d = image_features.shape[-1]

        offsets, cursor = [], 0
        for h, w in grids[:-1]:
            cursor += h * w
            offsets.append(cursor)
        chunks = mx.split(image_features, offsets, axis=0) if offsets else [image_features]

        k = self.spatial_merge_size
        merged = []
        for (h, w), tokens in zip(grids, chunks):
            grid = tokens.reshape(h, w, d).transpose(2, 0, 1)   # [d, h, w]
            merged.append(unfold_2x2(grid, k).T)                # [h/k * w/k, d*k*k]
        return self.merging_layer(mx.concatenate(merged, axis=0))


class Mistral3MultiModalProjector(nn.Module):
    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        *,
        spatial_merge_size: int = 2,
        patch_size: int = 14,
        rms_norm_eps: float = 1e-5,
        bias: bool = False,
        num_feature_layers: int = 1,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(vision_hidden_size, eps=rms_norm_eps)
        self.patch_merger = Mistral3PatchMerger(
            vision_hidden_size, spatial_merge_size, patch_size
        )
        self.linear_1 = nn.Linear(
            vision_hidden_size * num_feature_layers, text_hidden_size, bias=bias
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(text_hidden_size, text_hidden_size, bias=bias)

    def __call__(self, x: mx.array, image_sizes) -> mx.array:
        x = self.norm(x)
        x = self.patch_merger(x, image_sizes)
        return self.linear_2(self.gelu(self.linear_1(x)))
