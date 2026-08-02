"""OptiQ ops — custom kernels / fused primitives for MLX.

``flash_attention_tiled`` is the production path: stock fused forward plus a
FlashAttention-2 tiled backward that never materializes the [B, Hq, T, T] score
tensor. ``flash_attention_metal`` is the superseded hand-written Metal kernel,
kept for reference and its tests; it is correct but 14-137x slower than stock.
"""

from .flash_attention import flash_attention
from .flash_attention_metal import flash_attention_metal
from .flash_attention_tiled import flash_attention_tiled
from .attention_patch import enable_flash_attention_training

__all__ = [
    "flash_attention",
    "flash_attention_metal",
    "flash_attention_tiled",
    "enable_flash_attention_training",
]
