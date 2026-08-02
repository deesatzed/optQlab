"""OptiQ KV cache extensions.

Bridges gaps in mlx-lm's cache layer. Currently:

- ``RotatingQuantizedKVCache`` (rotating.py): affine-quantized variant of
  mlx-lm's ``RotatingKVCache``. Used by sliding-window-attention models
  (Gemma 3/4, Cohere R2, OLMo 3, Phi-3/4, EXAONE 4, Ministral 3, etc.).
  Replaces mlx-lm's ``RotatingKVCache.to_quantized`` which raises
  ``NotImplementedError("RotatingKVCache Quantization NYI")``.

Call :func:`patch_rotating_to_quantized` once at process start to install
the implementation. ``optiq.serve`` does this automatically when
``--kv-bits`` or ``--kv-config`` is set.
"""

from .rotating import (
    RotatingQuantizedKVCache,
    patch_rotating_to_quantized,
)

__all__ = [
    "RotatingQuantizedKVCache",
    "patch_rotating_to_quantized",
]
