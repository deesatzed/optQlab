"""Pixtral / Mistral-3 image preprocessing (from transformers' PixtralImageProcessor,
Apache-2.0). Pillow + numpy only -- no torchvision, no transformers at runtime.

Pixtral is native-resolution: the image is scaled down (never up) so its longest
edge fits ``longest_edge``, then rounded UP to whole patches. Mistral-3 asks the
processor for ``patch_size * spatial_merge_size`` (28, not 14), because the
projector merges 2x2 patches -- so the resized dimensions are multiples of 28 and
the merged token grid is exactly ``(H // 28, W // 28)``.

Normalization is the CLIP mean/std, unlike Gemma-4 (rescale only) and Qwen
(symmetric 0.5), so the three families deliberately do not share this module.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
LONGEST_EDGE = 1540
RESCALE_FACTOR = 1.0 / 255.0
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def resize_output_size(
    height: int, width: int, longest_edge: int, patch_size: int
) -> tuple[int, int]:
    """Target ``(height, width)`` after the Pixtral resize.

    Scale down by the largest edge ratio (floor, so the result is never *larger*
    than ``longest_edge``), then round each side up to a whole patch. Images
    already inside the budget are not upscaled -- only patch-rounded.
    """
    ratio = max(height / longest_edge, width / longest_edge)
    if ratio > 1:
        height = int(math.floor(height / ratio))
        width = int(math.floor(width / ratio))
    n_h = (height - 1) // patch_size + 1
    n_w = (width - 1) // patch_size + 1
    return n_h * patch_size, n_w * patch_size


def preprocess_images(
    images: Image.Image | list[Image.Image],
    *,
    patch_size: int = PATCH_SIZE,
    spatial_merge_size: int = SPATIAL_MERGE_SIZE,
    longest_edge: int = LONGEST_EDGE,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Return ``(pixel_values_per_image, image_sizes)``.

    Each entry is float32 ``[3, H, W]``, CLIP-normalized. Sizes differ per image
    (native resolution), so the result is a list; the caller pads into a batch and
    passes ``image_sizes`` so the tower can crop the padding back off.
    """
    if isinstance(images, Image.Image):
        images = [images]
    effective_patch = patch_size * spatial_merge_size

    mean = np.array(IMAGE_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGE_STD, dtype=np.float32).reshape(3, 1, 1)

    processed: list[np.ndarray] = []
    sizes: list[tuple[int, int]] = []
    for img in images:
        img = img.convert("RGB")
        h, w = resize_output_size(
            img.height, img.width, longest_edge, effective_patch
        )
        if (h, w) != (img.height, img.width):
            img = img.resize((w, h), resample=Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) * RESCALE_FACTOR   # [H, W, 3]
        arr = np.transpose(arr, (2, 0, 1))                          # [3, H, W]
        processed.append((arr - mean) / std)
        sizes.append((h, w))
    return processed, sizes


def pad_to_batch(pixel_values: list[np.ndarray]) -> np.ndarray:
    """Stack per-image ``[3, H, W]`` arrays into ``[N, 3, Hmax, Wmax]``.

    Zero-padding is safe: ``patch_conv`` has stride == kernel, so padded patches
    are whole patches, and the tower drops them using ``image_sizes`` before the
    transformer ever sees them.
    """
    max_h = max(a.shape[1] for a in pixel_values)
    max_w = max(a.shape[2] for a in pixel_values)
    out = np.zeros((len(pixel_values), 3, max_h, max_w), dtype=np.float32)
    for i, a in enumerate(pixel_values):
        out[i, :, : a.shape[1], : a.shape[2]] = a
    return out
