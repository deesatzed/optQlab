"""FlashAttention-2 backward as a single Metal kernel for MLX.

Companion to ``flash_attention_metal.py``'s forward kernel. Given saved
``O`` and ``L`` (log-sum-exp) from the forward, this computes dQ, dK, dV
with the same O(seq × head_dim) memory footprint and no seq×seq
materialization.

Strategy (matches Dao 2023 FA-2 §4.2):

  For each (Q tile, KV tile) pair:
    1. Recompute S_ij = Q_i @ K_j^T · scale   (tile-sized)
    2. Recompute P_ij = exp(S_ij − L_i)       (using saved LSE)
    3. dV_j += P_ij^T @ dO_i                  (accumulate)
    4. dP_ij = dO_i @ V_j^T
    5. D_i   = rowsum(dO_i · O_i)             (precompute once per Q tile)
    6. dS_ij = P_ij * (dP_ij − D_i)
    7. dQ_i += dS_ij @ K_j · scale            (accumulate)
    8. dK_j += dS_ij^T @ Q_i · scale          (accumulate)

Tricky parts compared to forward:

  * dQ is accumulated PER Q-row, so we parallelize over Q tiles and keep
    dQ rows in registers (same pattern as forward).
  * dK and dV are accumulated PER KV-row, across ALL Q tiles. The
    simplest approach: one outer dispatch per KV tile, inner loop over
    Q tiles, write dK and dV back once at the end. We use ``atomic_fetch_add``
    semantics via MLX's ``atomic_outputs`` since multiple Q-tile passes
    write the same dK/dV cells.

  * Causal masking: both the forward and backward skip score cells where
    q_idx < k_idx.

Current scope matches forward: fp16 I/O, head_dim=128, MHA only after
GQA/MQA expansion. Non-standard head dims fall back to Python.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import mlx.core as mx

from .flash_attention_metal import SUPPORTED_HEAD_DIMS


def _backward_tiles(head_dim: int) -> tuple[int, int]:
    """Return ``(BQ_BWD, BKV_BWD)`` for the backward kernels at this D.

    Envelope: dKV carries four fp16 tiles (Q, dO, K, V), so
    ``8 · BQ · D + 8 · BQ ≤ 32 000`` bytes. Pick BQ conservatively.
    BKV can match BQ — no advantage to widening it for the dQ kernel.
    """
    if head_dim <= 64:
        return 32, 32
    if head_dim <= 128:
        return 16, 32  # BKV=32 to keep forward-like throughput
    if head_dim <= 256:
        return 8, 16
    if head_dim <= 512:
        return 4, 8
    raise ValueError(f"unsupported head_dim={head_dim}")


# Legacy alias for callers using head_dim=128 directly.
BLOCK_Q_BWD, BLOCK_KV_BWD = _backward_tiles(128)


# --------------------------------------------------------------------------
# Pass 1: precompute D_i = rowsum(dO_i ⊙ O_i) for every Q row.
# --------------------------------------------------------------------------
_D_KERNEL_SOURCE = r"""
// Each thread computes D for one (batch, head, q_row) triple.
// Inputs:
//   o:  (B, Hq, Tq, D) fp16
//   do_: (B, Hq, Tq, D) fp16
// Output:
//   d_out: (B, Hq, Tq) float32

const uint idx = thread_position_in_grid.x;
const uint B   = shape[0];
const uint Hq  = shape[1];
const uint Tq  = shape[2];
const uint D   = shape[3];
const uint total = B * Hq * Tq;
if (idx >= total) return;

const uint base = idx * D;
float acc = 0.0f;
for (uint d = 0u; d < D; ++d) {
    acc += (float)o[base + d] * (float)do_[base + d];
}
d_out[idx] = acc;
"""


@lru_cache(maxsize=1)
def _compiled_d_kernel():
    return mx.fast.metal_kernel(
        name="optiq_flash_attention_D",
        input_names=["o", "do_", "shape"],
        output_names=["d_out"],
        source=_D_KERNEL_SOURCE,
        ensure_row_contiguous=True,
    )


def _compute_D(O: mx.array, dO: mx.array) -> mx.array:
    B, Hq, Tq, D = O.shape
    shape = mx.array([B, Hq, Tq, D], dtype=mx.uint32)
    kernel = _compiled_d_kernel()
    outs = kernel(
        inputs=[O, dO, shape],
        grid=(B * Hq * Tq, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, Hq, Tq)],
        output_dtypes=[mx.float32],
    )
    return outs[0]


# --------------------------------------------------------------------------
# Pass 2: dK, dV — one dispatch per KV tile, inner loop over Q tiles.
# This is the simpler of the two backward kernels because dK/dV only
# depend on Q, K, V, dO, O, L (no dependency on dQ).
# --------------------------------------------------------------------------
_DKV_KERNEL_SOURCE = r"""
// threadgroup_position_in_grid: (kv_tile_idx, kv_head, batch)
// Dispatched per-Hkv: the outer Q-head loop is folded INTO the kernel,
// so this pass produces dK, dV at Hkv shape directly (no expand/reduce).
//
// Inputs:
//   q: (B, Hq, Tq, D)    fp16
//   k: (B, Hkv, Tkv, D)  fp16
//   v: (B, Hkv, Tkv, D)  fp16
//   do_: (B, Hq, Tq, D)  fp16
//   l_in: (B, Hq, Tq)    float32
//   d_vec: (B, Hq, Tq)   float32
//   shape: [B, Hq, Hkv, Tq, Tkv, D]  uint32
// Outputs:
//   dk_out: (B, Hkv, Tkv, D) fp16
//   dv_out: (B, Hkv, Tkv, D) fp16

const uint kv_tile_idx = threadgroup_position_in_grid.x;
const uint kv_head     = threadgroup_position_in_grid.y;   // 0..Hkv-1
const uint batch       = threadgroup_position_in_grid.z;
const uint tid         = thread_position_in_threadgroup.x;

const uint B   = shape[0];
const uint Hq  = shape[1];
const uint Hkv = shape[2];
const uint Tq  = shape[3];
const uint Tkv = shape[4];
const uint D   = shape[5];
const uint reps = Hq / Hkv;

const float fscale = scale[0];
const uint  is_causal = causal[0];

const uint kv_row_global = kv_tile_idx * BKV + tid;
const bool kv_row_valid  = kv_row_global < Tkv;

// Shared tiles
threadgroup half q_tile[BQ * D_MAX];
threadgroup half do_tile[BQ * D_MAX];
threadgroup half k_tile[BKV * D_MAX];
threadgroup half v_tile[BKV * D_MAX];
threadgroup float l_tile[BQ];
threadgroup float D_tile[BQ];

// This thread's (dK_j, dV_j) row accumulators — across ALL q_heads that
// map to this kv_head.
float dK_j[D_MAX];
float dV_j[D_MAX];
for (uint d = 0u; d < D; ++d) {
    dK_j[d] = 0.0f;
    dV_j[d] = 0.0f;
}

// Load K_j and V_j (shared across q_heads) into threadgroup memory.
if (kv_row_valid) {
    const uint kv_base = ((batch * Hkv + kv_head) * Tkv + kv_row_global) * D;
    for (uint d = 0u; d < D; ++d) {
        k_tile[tid * D + d] = k[kv_base + d];
        v_tile[tid * D + d] = v[kv_base + d];
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

const uint n_q_tiles = (Tq + BQ - 1u) / BQ;
// Outer loop over the Q-heads that share this KV head (GQA group).
for (uint q_h_rel = 0u; q_h_rel < reps; ++q_h_rel) {
    const uint q_head = kv_head * reps + q_h_rel;

    for (uint qi = 0u; qi < n_q_tiles; ++qi) {
        const uint q_base_row = qi * BQ;

        // Causal skip — if the entire Q tile is strictly before this kv_row,
        // there's no contribution (attention is zero there).
        if (is_causal && (q_base_row + BQ) <= kv_tile_idx * BKV) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        // Cooperative load of Q tile, dO tile, and L/D rows for THIS q_head.
        const uint rows_this_q = min((uint)BQ, Tq - q_base_row);
        if (tid < rows_this_q) {
            const uint q_global_row = q_base_row + tid;
            const uint q_base = ((batch * Hq + q_head) * Tq + q_global_row) * D;
            for (uint d = 0u; d < D; ++d) {
                q_tile[tid * D + d] = q[q_base + d];
                do_tile[tid * D + d] = do_[q_base + d];
            }
            const uint lse_idx = (batch * Hq + q_head) * Tq + q_global_row;
            l_tile[tid] = l_in[lse_idx];
            D_tile[tid] = d_vec[lse_idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // For each Q row in the tile, compute score s_i_j with THIS thread's K_j row,
        // then P, then accumulate dV_j and dK_j from all Q rows.
        if (kv_row_valid) {
            for (uint i = 0u; i < rows_this_q; ++i) {
                // s = Q_i · K_j * scale
                float s = 0.0f;
                for (uint d = 0u; d < D; ++d) {
                    s += (float)q_tile[i * D + d] * (float)k_tile[tid * D + d];
                }
                s *= fscale;
                const uint q_row_global = q_base_row + i;
                if (is_causal && kv_row_global > q_row_global) {
                    s = -INFINITY;
                }
                const float p = exp(s - l_tile[i]);

                // dP_ij = dO_i · V_j
                float dp = 0.0f;
                for (uint d = 0u; d < D; ++d) {
                    dp += (float)do_tile[i * D + d] * (float)v_tile[tid * D + d];
                }
                const float ds = p * (dp - D_tile[i]);

                for (uint d = 0u; d < D; ++d) {
                    dV_j[d] += p * (float)do_tile[i * D + d];
                }
                for (uint d = 0u; d < D; ++d) {
                    dK_j[d] += ds * (float)q_tile[i * D + d] * fscale;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Write out dK_j, dV_j for this KV row (Hkv-sized output).
if (kv_row_valid) {
    const uint out_base = ((batch * Hkv + kv_head) * Tkv + kv_row_global) * D;
    for (uint d = 0u; d < D; ++d) {
        dk_out[out_base + d] = (half)dK_j[d];
        dv_out[out_base + d] = (half)dV_j[d];
    }
}
"""


@lru_cache(maxsize=8)
def _compiled_dkv_kernel(head_dim: int):
    bq, bkv = _backward_tiles(head_dim)
    header = f"""
    #define BQ  {bq}
    #define BKV {bkv}
    #define D_MAX {head_dim}
    """
    return mx.fast.metal_kernel(
        name=f"optiq_flash_attention_dKV_d{head_dim}",
        input_names=["q", "k", "v", "do_", "l_in", "d_vec", "shape", "scale", "causal"],
        output_names=["dk_out", "dv_out"],
        source=_DKV_KERNEL_SOURCE,
        header=header,
        ensure_row_contiguous=True,
    )


# --------------------------------------------------------------------------
# Pass 3: dQ — one dispatch per Q tile, inner loop over KV tiles.
# --------------------------------------------------------------------------
_DQ_KERNEL_SOURCE = r"""
// threadgroup_position_in_grid: (q_tile_idx, q_head, batch)
// K/V are Hkv-sized; we compute kv_head = q_head / (Hq/Hkv).

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

const uint reps    = Hq / Hkv;
const uint kv_head = head / reps;

const float fscale = scale[0];
const uint  is_causal = causal[0];

const uint q_row_global = q_tile_idx * BQ + tid;
const bool q_row_valid  = q_row_global < Tq;

threadgroup half k_tile[BKV * D_MAX];
threadgroup half v_tile[BKV * D_MAX];

// This thread's Q row, dO row, L, D in registers.
half q_i[D_MAX];
half do_i[D_MAX];
float l_i = 0.0f;
float D_i = 0.0f;
float dq_i[D_MAX];
for (uint d = 0u; d < D; ++d) { dq_i[d] = 0.0f; }

if (q_row_valid) {
    const uint q_base = ((batch * Hq + head) * Tq + q_row_global) * D;
    for (uint d = 0u; d < D; ++d) {
        q_i[d] = q[q_base + d];
        do_i[d] = do_[q_base + d];
    }
    const uint lse_idx = (batch * Hq + head) * Tq + q_row_global;
    l_i = l_in[lse_idx];
    D_i = d_vec[lse_idx];
}

const uint n_kv_tiles = (Tkv + BKV - 1u) / BKV;
for (uint kj = 0u; kj < n_kv_tiles; ++kj) {
    const uint kv_base_row = kj * BKV;

    // Load K, V tiles cooperatively from the Hkv-sized input.
    const uint rows_this_tile = min((uint)BKV, Tkv - kv_base_row);
    for (uint r = tid; r < rows_this_tile; r += BQ) {
        const uint kv_global_row = kv_base_row + r;
        const uint kv_base = ((batch * Hkv + kv_head) * Tkv + kv_global_row) * D;
        for (uint d = 0u; d < D; ++d) {
            k_tile[r * D + d] = k[kv_base + d];
            v_tile[r * D + d] = v[kv_base + d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (is_causal && kv_base_row > q_row_global) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        continue;
    }

    if (q_row_valid) {
        // s_j, p_j per K row in tile.
        for (uint j = 0u; j < rows_this_tile; ++j) {
            float s = 0.0f;
            for (uint d = 0u; d < D; ++d) {
                s += (float)q_i[d] * (float)k_tile[j * D + d];
            }
            s *= fscale;
            if (is_causal && (kv_base_row + j) > q_row_global) {
                s = -INFINITY;
            }
            float p = exp(s - l_i);
            // dp = dO_i · V_j
            float dp = 0.0f;
            for (uint d = 0u; d < D; ++d) {
                dp += (float)do_i[d] * (float)v_tile[j * D + d];
            }
            float ds = p * (dp - D_i);
            // dQ_i += ds * K_j * scale
            for (uint d = 0u; d < D; ++d) {
                dq_i[d] += ds * (float)k_tile[j * D + d] * fscale;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (q_row_valid) {
    const uint out_base = ((batch * Hq + head) * Tq + q_row_global) * D;
    for (uint d = 0u; d < D; ++d) {
        dq_out[out_base + d] = (half)dq_i[d];
    }
}
"""


@lru_cache(maxsize=8)
def _compiled_dq_kernel(head_dim: int):
    bq, bkv = _backward_tiles(head_dim)
    header = f"""
    #define BQ  {bq}
    #define BKV {bkv}
    #define D_MAX {head_dim}
    """
    return mx.fast.metal_kernel(
        name=f"optiq_flash_attention_dQ_d{head_dim}",
        input_names=["q", "k", "v", "do_", "l_in", "d_vec", "shape", "scale", "causal"],
        output_names=["dq_out"],
        source=_DQ_KERNEL_SOURCE,
        header=header,
        ensure_row_contiguous=True,
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def flash_attention_metal_backward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    O: mx.array,
    L: mx.array,
    dO: mx.array,
    scale: Optional[float] = None,
    causal: bool = True,
) -> tuple[mx.array, mx.array, mx.array]:
    """Three Metal dispatches (D-vec, dKV, dQ). Returns (dQ, dK, dV).

    GQA/MQA is handled natively: pass K, V at their original Hkv shape and
    dK, dV come back at Hkv shape too (dKV kernel accumulates contributions
    from all Q-heads that share a KV head inside the kernel).
    """
    if q.dtype != mx.float16:
        raise ValueError("fp16 inputs required")
    B, Hq, Tq, D = q.shape
    _, Hkv, Tkv, _ = k.shape
    if D not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim={D} unsupported; backward supports {SUPPORTED_HEAD_DIMS}")
    if Hq % Hkv != 0:
        raise ValueError(f"Hq ({Hq}) must be a multiple of Hkv ({Hkv})")
    if scale is None:
        scale = D ** -0.5

    bq, bkv = _backward_tiles(D)

    # Step 1: D_i = rowsum(dO ⊙ O)  (Hq-shaped; O is Hq heads)
    D_vec = _compute_D(O, dO)

    # Step 2: dK, dV — dispatch per KV head. Kernel folds the per-Q-head
    # loop inside so no expand is needed. Outputs are native Hkv shape.
    shape = mx.array([B, Hq, Hkv, Tq, Tkv, D], dtype=mx.uint32)
    scale_arr = mx.array([scale], dtype=mx.float32)
    causal_arr = mx.array([1 if causal else 0], dtype=mx.uint32)

    n_kv_tiles = (Tkv + bkv - 1) // bkv
    dkv_kernel = _compiled_dkv_kernel(D)
    dkv_outs = dkv_kernel(
        inputs=[q, k, v, dO, L, D_vec, shape, scale_arr, causal_arr],
        grid=(n_kv_tiles * bkv, Hkv, B),
        threadgroup=(bkv, 1, 1),
        output_shapes=[(B, Hkv, Tkv, D), (B, Hkv, Tkv, D)],
        output_dtypes=[mx.float16, mx.float16],
    )
    dK, dV = dkv_outs

    # Step 3: dQ — dispatch per Q head. Kernel reads K/V at kv_head = q_head / reps.
    n_q_tiles = (Tq + bq - 1) // bq
    dq_kernel = _compiled_dq_kernel(D)
    dq_outs = dq_kernel(
        inputs=[q, k, v, dO, L, D_vec, shape, scale_arr, causal_arr],
        grid=(n_q_tiles * bq, Hq, B),
        threadgroup=(bq, 1, 1),
        output_shapes=[(B, Hq, Tq, D)],
        output_dtypes=[mx.float16],
    )
    dQ = dq_outs[0]

    return dQ, dK, dV
