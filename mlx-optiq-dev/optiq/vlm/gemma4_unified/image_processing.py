"""Gemma-4 unified image preprocessing (vendored from mlx-vlm, BSD-3).

Same aspect-ratio-preserving resize + rescale as the SigLIP path, then patchify
into ``model_patch_size`` x ``model_patch_size`` patch vectors with 2D grid
positions, padded to the soft-token budget. Pillow + numpy only.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..gemma4.image_processing import _aspect_ratio_preserving_resize

PATCH_SIZE = 16
POOLING_KERNEL_SIZE = 3
MODEL_PATCH_SIZE = 48  # = patch_size * pooling_kernel_size
NUM_SOFT_TOKENS = 280
RESCALE_FACTOR = 1.0 / 255.0


def _convert_image_to_model_patches(image_chw: np.ndarray, model_patch_size: int):
    """``image_chw``: float32 [C, H, W] in [0,1]. Returns (patches, positions)
    where patches is [n, model_patch_size**2 * C] and positions is [n, 2]."""
    channels, height, width = image_chw.shape
    pH = height // model_patch_size
    pW = width // model_patch_size
    patches = image_chw.reshape(
        channels, pH, model_patch_size, pW, model_patch_size
    )
    patches = patches.transpose(1, 3, 2, 4, 0)  # [pH, pW, p, p, C]
    patches = patches.reshape(pH * pW, model_patch_size * model_patch_size * channels)
    grid = np.meshgrid(
        np.arange(pW, dtype=np.int64), np.arange(pH, dtype=np.int64)
    )
    positions = np.stack(grid, axis=-1).reshape(-1, 2)  # (x, y) per patch
    return patches.astype(np.float32), positions


def _pad_patches(patches, positions, target_length):
    cur = patches.shape[0]
    if cur >= target_length:
        return patches[:target_length], positions[:target_length]
    patches = np.pad(patches, ((0, target_length - cur), (0, 0)))
    positions = np.pad(
        positions, ((0, target_length - cur), (0, 0)), constant_values=-1
    )
    return patches, positions


def preprocess_images(
    images: Image.Image | list[Image.Image],
    *,
    patch_size: int = PATCH_SIZE,
    pooling_kernel_size: int = POOLING_KERNEL_SIZE,
    model_patch_size: int = MODEL_PATCH_SIZE,
    num_soft_tokens: int = NUM_SOFT_TOKENS,
):
    """Return ``(pixel_values, image_position_ids, soft_tokens)``:
    pixel_values [N, num_soft_tokens, patch_dim], image_position_ids
    [N, num_soft_tokens, 2], soft_tokens list of real (unpadded) counts."""
    if isinstance(images, Image.Image):
        images = [images]
    max_patches = num_soft_tokens * pooling_kernel_size**2

    pvs, pos_ids, soft = [], [], []
    for img in images:
        img = img.convert("RGB")
        img = _aspect_ratio_preserving_resize(
            img, patch_size, max_patches, pooling_kernel_size
        )
        arr = np.asarray(img, dtype=np.float32) * RESCALE_FACTOR  # [H, W, 3]
        arr = np.transpose(arr, (2, 0, 1))  # [3, H, W]
        patches, positions = _convert_image_to_model_patches(arr, model_patch_size)
        soft.append(patches.shape[0])
        patches, positions = _pad_patches(patches, positions, num_soft_tokens)
        pvs.append(patches)
        pos_ids.append(positions)

    return np.stack(pvs), np.stack(pos_ids), soft
