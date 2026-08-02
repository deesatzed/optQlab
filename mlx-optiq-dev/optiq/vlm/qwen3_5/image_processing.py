"""Qwen3.5 image preprocessing (vendored from transformers Qwen2VLImageProcessor
logic, Apache-2.0). Smart-resize to a patch grid, normalize to [-1, 1], and
patchify into Qwen's merge-grouped flattened patches + grid_thw. Pillow + numpy
only (no torchvision / transformers at runtime)."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

PATCH_SIZE = 16
MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
MIN_PIXELS = 65536      # size.shortest_edge
MAX_PIXELS = 16777216   # size.longest_edge


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Round h,w to multiples of ``factor`` and scale so the pixel count is in
    [min_pixels, max_pixels]. Matches transformers' Qwen smart_resize."""
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _patchify(img_chw: np.ndarray, patch=PATCH_SIZE, merge=MERGE_SIZE,
              temporal=TEMPORAL_PATCH_SIZE):
    """``img_chw``: float32 [C, H, W]. Returns (flatten_patches, grid_thw)."""
    c, h, w = img_chw.shape
    gh, gw = h // patch, w // patch
    p = img_chw.reshape(1, c, gh // merge, merge, patch, gw // merge, merge, patch)
    # -> [batch, gh/m, gw/m, m, m, C, patch, patch]
    p = p.transpose(0, 2, 5, 3, 6, 1, 4, 7)
    p = np.expand_dims(p, 6)  # temporal slot
    p = np.broadcast_to(
        p, (1, gh // merge, gw // merge, merge, merge, c, temporal, patch, patch)
    )
    flat = p.reshape(gh * gw, c * temporal * patch * patch)
    return flat.astype(np.float32), np.array([[1, gh, gw]], dtype=np.int64)


def preprocess_images(images: Image.Image | list[Image.Image]):
    """Return ``(pixel_values, grid_thw)``: pixel_values is the concatenation of
    every image's flattened patches [sum(t*h*w), C*temporal*patch*patch]; grid_thw
    is [N, 3] (t, h, w in patch units)."""
    if isinstance(images, Image.Image):
        images = [images]
    factor = PATCH_SIZE * MERGE_SIZE
    all_patches, all_grids = [], []
    for img in images:
        img = img.convert("RGB")
        h_bar, w_bar = smart_resize(img.height, img.width, factor,
                                    MIN_PIXELS, MAX_PIXELS)
        img = img.resize((w_bar, h_bar), resample=Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]
        arr = (arr - IMAGE_MEAN) / IMAGE_STD  # -> [-1, 1]
        arr = np.transpose(arr, (2, 0, 1))  # [3, H, W]
        patches, grid = _patchify(arr)
        all_patches.append(patches)
        all_grids.append(grid)
    return np.concatenate(all_patches, axis=0), np.concatenate(all_grids, axis=0)
