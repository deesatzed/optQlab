"""FlashAttention-2 as a single Metal kernel for MLX.

Core idea: the whole forward pass runs inside **one** Metal dispatch.
Threadgroup memory holds the current Q tile, current K/V tile, and the
running (m, l, O) softmax statistics. We never materialize the full
``seq × seq`` attention matrix in global memory — at any moment the
largest live tensor is a single ``BLOCK_Q × BLOCK_KV`` score block in
threadgroup memory. That drops attention forward memory from
O(seq² × n_heads) down to O(seq × head_dim × n_heads).

We keep the implementation focused: fp16 inputs, causal mask support,
head_dim = 128 (the common case for modern LLMs; Qwen3.5 and Gemma-4
both use 128). Other head dims are handled by the Python fallback.

Design choices:
  * grid = (n_q_tiles, n_heads_q, batch).
  * threadgroup = (BLOCK_Q,) — one thread per Q row in the tile.
  * Each thread owns one Q row's (m_i, l_i, O_i) and accumulates them
    in registers across KV tiles.
  * BLOCK_Q = BLOCK_KV = 32 — small enough to fit Q tile + K tile +
    V tile + S tile in threadgroup memory (32 rows × 128 cols × fp16 =
    8 KB each; 4 buffers = 32 KB, under the 32 KB typical cap).
  * GQA/MQA: we expand K/V to Hq heads before the kernel. MLX will
    allocate, but the per-head forward cost is the same.

Current status (v0): forward-only. Backward still uses the Python
tiled path from ``flash_attention.py`` — good for correctness but
O(seq²) memory. A backward Metal kernel lands in v0.1 once forward is
validated in real training.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import mlx.core as mx


# Tile sizes tuned for the 32 KB threadgroup-memory cap on M-series GPUs.
# Forward carries three fp16 tiles (Q, K, V), so the envelope is
# 3 · BQ · D · 2 bytes ≤ 32 000. Backward dKV carries four (Q, dO, K, V),
# so the envelope tightens to 4 · BQ · D · 2 bytes ≤ 32 000. Pick per-D.
SUPPORTED_HEAD_DIMS = (64, 96, 128, 256, 512)


def _forward_tiles(head_dim: int) -> tuple[int, int]:
    """Return ``(BLOCK_Q, BLOCK_KV)`` for the forward kernel at this D."""
    if head_dim <= 64:
        return 64, 64
    if head_dim <= 128:
        return 32, 32
    if head_dim <= 256:
        return 16, 16
    if head_dim <= 512:
        return 8, 8
    raise ValueError(f"unsupported head_dim={head_dim}")


# Legacy module-level constants — kept for callers that imported them at D=128.
BLOCK_Q, BLOCK_KV = _forward_tiles(128)
SUPPORTED_HEAD_DIM = 128  # deprecated: use SUPPORTED_HEAD_DIMS


_KERNEL_SOURCE = r"""
// thread_position_in_threadgroup.x: which Q row within the tile (0..BQ-1)
// threadgroup_position_in_grid: (q_tile_idx, q_head, batch)
// Inputs:
//   q: (B, Hq, Tq, D)  fp16
//   k: (B, Hkv, Tkv, D) fp16   (Hkv may be < Hq — GQA/MQA)
//   v: (B, Hkv, Tkv, D) fp16
//   shape: [B, Hq, Hkv, Tq, Tkv, D]  uint32
//   scale: float
//   causal: uint32 (0 or 1)
// Outputs:
//   o: (B, Hq, Tq, D) fp16
//   l: (B, Hq, Tq)    float32 (log-sum-exp saved for backward)

const uint q_tile_idx = threadgroup_position_in_grid.x;
const uint head       = threadgroup_position_in_grid.y;   // 0..Hq-1
const uint batch      = threadgroup_position_in_grid.z;
const uint tid        = thread_position_in_threadgroup.x;

const uint B   = shape[0];
const uint Hq  = shape[1];
const uint Hkv = shape[2];
const uint Tq  = shape[3];
const uint Tkv = shape[4];
const uint D   = shape[5];

const float fscale = scale[0];
const uint  is_causal = causal[0];

// GQA/MQA: Hq Q-heads share each of the Hkv K/V heads.
const uint reps    = Hq / Hkv;
const uint kv_head = head / reps;

const uint q_row_global = q_tile_idx * BQ + tid;
const bool q_row_valid  = q_row_global < Tq;

// Threadgroup tiles.
threadgroup half q_tile[BQ * D_MAX];
threadgroup half k_tile[BKV * D_MAX];
threadgroup half v_tile[BKV * D_MAX];

// Per-thread (per Q row) statistics.
float m_i = -INFINITY;
float l_i = 0.0f;
float o_i[D_MAX];
for (uint d = 0; d < D; ++d) { o_i[d] = 0.0f; }

// Load Q tile into threadgroup memory (each thread loads one row).
if (q_row_valid) {
    const uint q_base = ((batch * Hq + head) * Tq + q_row_global) * D;
    for (uint d = 0; d < D; ++d) {
        q_tile[tid * D + d] = q[q_base + d];
    }
} else {
    for (uint d = 0; d < D; ++d) {
        q_tile[tid * D + d] = 0.0h;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Iterate KV tiles.
const uint n_kv_tiles = (Tkv + BKV - 1u) / BKV;
for (uint kv_tile = 0u; kv_tile < n_kv_tiles; ++kv_tile) {
    const uint kv_base_row = kv_tile * BKV;

    // Load K and V tiles cooperatively — index into the Hkv-sized K/V.
    const uint rows_this_tile = min((uint)BKV, Tkv - kv_base_row);
    if (tid < rows_this_tile) {
        const uint kv_global_row = kv_base_row + tid;
        const uint kv_global = ((batch * Hkv + kv_head) * Tkv + kv_global_row) * D;
        for (uint d = 0; d < D; ++d) {
            k_tile[tid * D + d] = k[kv_global + d];
            v_tile[tid * D + d] = v[kv_global + d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Causal skip: if this entire kv tile is in the future of every
    // q row in our q_tile, nothing to do.
    if (is_causal && kv_base_row > q_row_global) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        continue;
    }

    // Score vector s_j for this Q row against every K row in tile.
    float s[BKV];
    float row_max = -INFINITY;
    for (uint j = 0u; j < rows_this_tile; ++j) {
        float dot = 0.0f;
        for (uint d = 0u; d < D; ++d) {
            dot += (float)q_tile[tid * D + d] * (float)k_tile[j * D + d];
        }
        dot *= fscale;
        // Causal mask inside tile
        if (is_causal && (kv_base_row + j) > q_row_global) {
            dot = -INFINITY;
        }
        s[j] = dot;
        row_max = max(row_max, dot);
    }

    // Online softmax update
    const float m_new = max(m_i, row_max);
    const float rescale = exp(m_i - m_new);
    float row_sum = 0.0f;
    for (uint j = 0u; j < rows_this_tile; ++j) {
        s[j] = exp(s[j] - m_new);
        row_sum += s[j];
    }
    l_i = rescale * l_i + row_sum;

    // Rescale accumulated O, then add new contribution ∑_j p_j · V_j.
    for (uint d = 0u; d < D; ++d) {
        float new_contrib = 0.0f;
        for (uint j = 0u; j < rows_this_tile; ++j) {
            new_contrib += s[j] * (float)v_tile[j * D + d];
        }
        o_i[d] = rescale * o_i[d] + new_contrib;
    }
    m_i = m_new;

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Normalize and write out.
if (q_row_valid) {
    const uint out_base = ((batch * Hq + head) * Tq + q_row_global) * D;
    const float inv_l = 1.0f / l_i;
    for (uint d = 0u; d < D; ++d) {
        o[out_base + d] = (half)(o_i[d] * inv_l);
    }
    const uint lse_base = (batch * Hq + head) * Tq + q_row_global;
    l_out[lse_base] = m_i + log(l_i);
}
"""


@lru_cache(maxsize=8)
def _compiled_kernel(head_dim: int):
    bq, bkv = _forward_tiles(head_dim)
    header = f"""
    #define BQ  {bq}
    #define BKV {bkv}
    #define D_MAX {head_dim}
    """
    return mx.fast.metal_kernel(
        name=f"optiq_flash_attention_fwd_d{head_dim}",
        input_names=["q", "k", "v", "shape", "scale", "causal"],
        output_names=["o", "l_out"],
        source=_KERNEL_SOURCE,
        header=header,
        ensure_row_contiguous=True,
    )


def flash_attention_metal_forward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: Optional[float] = None,
    causal: bool = True,
) -> tuple[mx.array, mx.array]:
    """Single-dispatch Metal flash-attention forward.

    Args:
        q: (B, Hq,  Tq,  D)  fp16
        k: (B, Hkv, Tkv, D)  fp16 — expanded to Hq if Hq != Hkv
        v: (B, Hkv, Tkv, D)  fp16 — expanded similarly
        scale: 1/sqrt(D) default
        causal: apply lower-triangular mask

    Returns:
        (O, L) where O is output (B,Hq,Tq,D) fp16 and L is log-sum-exp
        (B,Hq,Tq) float32 (saved for backward).
    """
    if q.dtype != mx.float16 or k.dtype != mx.float16 or v.dtype != mx.float16:
        raise ValueError("flash_attention_metal_forward expects fp16 inputs")

    B, Hq, Tq, D = q.shape
    _, Hkv, Tkv, _ = k.shape
    if D not in SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"head_dim={D} unsupported; Metal kernel supports {SUPPORTED_HEAD_DIMS}"
        )
    if Hq % Hkv != 0:
        raise ValueError(f"Hq ({Hq}) must be a multiple of Hkv ({Hkv})")
    if scale is None:
        scale = D ** -0.5

    # Native GQA/MQA: the kernel reads K/V at head = q_head // (Hq/Hkv).
    # No mx.repeat — saves (Hq/Hkv - 1) * Hkv * Tkv * D * 2 bytes per call.
    shape = mx.array([B, Hq, Hkv, Tq, Tkv, D], dtype=mx.uint32)
    scale_arr = mx.array([scale], dtype=mx.float32)
    causal_arr = mx.array([1 if causal else 0], dtype=mx.uint32)

    bq, _bkv = _forward_tiles(D)
    n_q_tiles = (Tq + bq - 1) // bq

    kernel = _compiled_kernel(D)
    outs = kernel(
        inputs=[q, k, v, shape, scale_arr, causal_arr],
        grid=(n_q_tiles * bq, Hq, B),
        threadgroup=(bq, 1, 1),
        output_shapes=[(B, Hq, Tq, D), (B, Hq, Tq)],
        output_dtypes=[mx.float16, mx.float32],
    )
    return outs[0], outs[1]


# ---------------------------------------------------------------------------
# Autograd-wired public API
# ---------------------------------------------------------------------------
def flash_attention_metal(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: Optional[float] = None,
    causal: bool = True,
) -> mx.array:
    """Drop-in attention op with Metal forward + Metal backward under
    ``mx.custom_function``. Both directions are O(seq × head_dim) memory.

    Shapes match ``mx.fast.scaled_dot_product_attention``:
      q: (B, Hq,  Tq,  D)
      k: (B, Hkv, Tkv, D)
      v: (B, Hkv, Tkv, D)

    GQA/MQA is handled natively by the kernel — K/V stay at Hkv heads
    and the kernel indexes them by ``q_head // (Hq/Hkv)``. Gradients
    dK, dV are produced at Hkv shape directly (no expand + reduce).

    The forward's log-sum-exp ``L`` is emitted as a second output of the
    custom_function so the vjp can read it from the stored graph instead
    of recomputing the forward.
    """
    from .flash_attention_backward_metal import flash_attention_metal_backward

    if scale is None:
        scale = q.shape[-1] ** -0.5

    @mx.custom_function
    def _fa(q_, k_, v_):
        # Returns (O, L). L is not used downstream by the caller, but by
        # returning it from the primitive we let MLX keep it alive for the
        # backward pass. Without this the vjp has to recompute the entire
        # forward to rebuild L, doubling peak memory during training.
        return flash_attention_metal_forward(q_, k_, v_, scale=scale, causal=causal)

    @_fa.vjp
    def _fa_vjp(primals, cotangents, outputs):
        q_, k_, v_ = primals
        # outputs is (O, L); cotangents is (dO, dL). dL will be zeros
        # because the caller discards L, so we only need dO.
        O, L = outputs
        dO = cotangents[0]
        dQ, dK, dV = flash_attention_metal_backward(
            q_, k_, v_, O, L, dO, scale=scale, causal=causal
        )
        return dQ, dK, dV

    O, _L = _fa(q, k, v)
    return O
