"""Fused quantized SDPA — second half of OptiQ's tight-RAM KV-quant fix.

Why this exists
---------------
mlx-lm's stock ``quantized_scaled_dot_product_attention`` computes one
big ``Q @ K^T`` scores matrix per call. With GQA expansion that's shape
``(B, n_kv_heads, n_repeats, L_q, N)``. At prefill chunk-size 2048 and
N=16k on Granite-4.1-8b that's about 4.3 GB of fp16, plus the softmax
output is another 4.3 GB co-resident at the softmax step. Combined
with the conversion spike (see ``streaming_kv_quant``), u4 KV peaks
*higher* than fp16 KV on tight-RAM Macs — defeating the entire purpose
of KV-cache quantization. On a 24 GB Mac at 32k context, stock u4
peaks at 16.35 GB vs fp16's 11.51 GB.

This module installs a FlashAttention-2 replacement that never
materializes the full scores matrix. Per-tile scores shape is bounded
by ``n_chunk`` (default 512), reducing the prefill transient ~10x.
Combined with the streaming conversion patch, u4 KV at 32k drops to
**7.60 GB peak** — 34% below fp16, and unlocks long contexts that
otherwise OOM.

How it works
------------
The FlashAttention-2 algorithm tiles the K/V N axis. For each tile we
compute partial scores via ``mx.quantized_matmul`` (Apple's tuned
Metal kernel), apply an online softmax update, accumulate into the
output via another ``mx.quantized_matmul``. The combine logic
(max/sum/exp) is expressed as MLX ops which each lower to Metal — no
Python in the GPU hot path, but the dispatch is from Python.

When the installed mlx ships PR #3120's split-K for small-M quantized
matmul (in mlx HEAD as of 2026-03-21, awaiting next stable release),
this path runs at ±2% of fp16 KV cache speed on Qwen3.5-9B and similar
hybrid models. The speed improvement comes from upstream
``mx.quantized_matmul`` itself; our contribution is the FlashAttention
orchestration that makes long-context KV-quant actually fit in memory.

Supported configuration
-----------------------
- affine quantization, bits ∈ {4, 8}, group_size ∈ {32, 64, 128}
- any head_dim (the tiling uses mx.quantized_matmul which supports it)
- GQA (n_repeats = H_q / H_kv >= 1)
- causal mask or no mask

Anything outside this configuration falls back to the upstream unfused
path so behavior is safe and backwards-compatible.
"""
from __future__ import annotations
from typing import Optional

import mlx.core as mx
from mlx.utils import tree_map


_INSTALLED = False
_ORIGINAL = None
_N_CHUNK = 512


def _supported(queries, q_keys, group_size, bits):
    if bits not in (4, 8):
        return False
    if group_size not in (32, 64, 128):
        return False
    if queries.dtype not in (mx.float16, mx.bfloat16):
        return False
    return True


def _prefill_flashattn_n_tiled(
    queries, q_keys, q_values, scale, mask, group_size, bits,
):
    """L_q > 1 path — FlashAttention-2 tiling over the N axis using
    ``mx.quantized_matmul`` as the inner kernel."""
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads
    N = q_keys[0].shape[-2]

    queries = queries * scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    mask_arr = None
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            q_indices = mx.arange(N - L, N)
            k_indices = mx.arange(N)
            mask_arr = q_indices[:, None] >= k_indices[None]
        else:
            mask_arr = mask

    o_acc = None
    row_max = None
    row_sum = None

    for n_start in range(0, N, _N_CHUNK):
        n_end = min(n_start + _N_CHUNK, N)
        k_packed = q_keys[0][..., n_start:n_end, :]
        k_scales = q_keys[1][..., n_start:n_end, :]
        k_biases = q_keys[2][..., n_start:n_end, :]
        v_packed = q_values[0][..., n_start:n_end, :]
        v_scales = q_values[1][..., n_start:n_end, :]
        v_biases = q_values[2][..., n_start:n_end, :]

        scores = mx.quantized_matmul(
            queries, k_packed, k_scales, k_biases,
            transpose=True, group_size=group_size, bits=bits,
        )

        if mask_arr is not None:
            if mask_arr.ndim == 2:
                mask_chunk = mask_arr[:, n_start:n_end]
            else:
                mask_chunk = mask_arr[..., :, n_start:n_end]
            if mask_chunk.dtype == mx.bool_:
                scores = mx.where(mask_chunk, scores,
                                  mx.finfo(scores.dtype).min)
            else:
                scores = scores + mask_chunk

        chunk_max = mx.max(scores, axis=-1, keepdims=True)
        if o_acc is None:
            row_max = chunk_max
            exps = mx.exp(scores - row_max)
            row_sum = mx.sum(exps, axis=-1, keepdims=True)
            o_acc = mx.quantized_matmul(
                exps, v_packed, v_scales, v_biases,
                transpose=False, group_size=group_size, bits=bits,
            )
        else:
            new_max = mx.maximum(row_max, chunk_max)
            factor = mx.exp(row_max - new_max)
            exps = mx.exp(scores - new_max)
            new_sum = factor * row_sum + mx.sum(exps, axis=-1, keepdims=True)
            delta_out = mx.quantized_matmul(
                exps, v_packed, v_scales, v_biases,
                transpose=False, group_size=group_size, bits=bits,
            )
            o_acc = o_acc * factor + delta_out
            row_max = new_max
            row_sum = new_sum

    out = o_acc / row_sum

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))
    return out


def fused_quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys,
    q_values,
    scale: float,
    mask,
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    """Drop-in for mlx-lm's quantized_scaled_dot_product_attention."""
    if not _supported(queries, q_keys, group_size, bits):
        return _ORIGINAL(queries, q_keys, q_values, scale=scale, mask=mask,
                         group_size=group_size, bits=bits)

    if mask is not None and not (isinstance(mask, str) and mask == "causal"):
        return _ORIGINAL(queries, q_keys, q_values, scale=scale, mask=mask,
                         group_size=group_size, bits=bits)

    return _prefill_flashattn_n_tiled(
        queries, q_keys, q_values, scale, mask, group_size, bits,
    )


def install(n_chunk: int = 512) -> bool:
    """Monkey-patch mlx_lm.models.base.quantized_scaled_dot_product_attention.

    Idempotent. Returns True if the install actually changed state.
    """
    global _INSTALLED, _ORIGINAL, _N_CHUNK
    if _INSTALLED:
        return False
    _N_CHUNK = n_chunk
    import importlib
    base_mod = importlib.import_module("mlx_lm.models.base")
    _ORIGINAL = base_mod.quantized_scaled_dot_product_attention
    base_mod.quantized_scaled_dot_product_attention = fused_quantized_scaled_dot_product_attention
    _INSTALLED = True
    return True


def uninstall() -> bool:
    global _INSTALLED, _ORIGINAL
    if not _INSTALLED:
        return False
    import importlib
    base_mod = importlib.import_module("mlx_lm.models.base")
    base_mod.quantized_scaled_dot_product_attention = _ORIGINAL
    _INSTALLED = False
    _ORIGINAL = None
    return True


def is_installed() -> bool:
    return _INSTALLED
