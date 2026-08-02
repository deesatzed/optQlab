"""Shared low-level helpers for OptiQ vision/audio front-ends.

Kept arch-agnostic so every vendored encoder (gemma4, qwen3_5, …) can reuse the
same attention shim instead of each re-deriving the fused-SDPA padding dance.
"""

from __future__ import annotations

import mlx.core as mx

# head_dim sizes MLX's fused scaled_dot_product_attention supports natively.
_FUSED_HEAD_DIMS = (64, 80, 128)


def ensure_fused_sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    mask=None,
) -> mx.array:
    """Scaled-dot-product attention that always takes MLX's fused kernel.

    The fused kernel only accepts head_dim in :data:`_FUSED_HEAD_DIMS`. Vision
    encoders often use other sizes (e.g. 72); the non-fused fallback then emits
    NaNs on fully-masked padding rows. We zero-pad head_dim up to the next
    supported size, run the fused kernel, and slice the real dims back out.

    Args:
        q, k, v: ``[B, H, L, D]`` tensors (already head-transposed).
        scale: attention scale applied to the logits.
        mask: optional additive/boolean mask passed straight through.
    """
    d = q.shape[-1]
    if d in _FUSED_HEAD_DIMS:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    target = next((s for s in _FUSED_HEAD_DIMS if s >= d), None)
    if target is None:
        # Larger than any fused size — fall back to the plain path.
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    pad = target - d
    pad_spec = [(0, 0), (0, 0), (0, 0), (0, pad)]
    # Scale q by sqrt(d/target) so the padded zeros don't change the logits:
    # fused applies `scale` over `target` dims, but only the real `d` contribute.
    qp = mx.pad(q, pad_spec)
    kp = mx.pad(k, pad_spec)
    vp = mx.pad(v, pad_spec)
    out = mx.fast.scaled_dot_product_attention(qp, kp, vp, scale=scale, mask=mask)
    return out[..., :d]
