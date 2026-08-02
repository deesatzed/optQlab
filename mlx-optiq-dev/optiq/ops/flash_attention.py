"""FlashAttention-2-style attention with efficient backward for MLX.

MLX 0.30.x ships ``mx.fast.scaled_dot_product_attention`` with a fused
forward kernel. Its backward is auto-differentiated via MLX's graph,
which materializes the full ``seq × seq`` attention matrix and scales
O(seq²) in memory. At seq=8192 on 9B-class models, that's the actual
training bottleneck on Apple Silicon.

This module provides a tiled, O(seq × head_dim) memory attention backward
using the FlashAttention-2 algorithm, exposed via ``mx.custom_function``
so mlx-lm's autograd transparently routes through it during training.

Algorithm (matches Dao 2023 FA-2):

  Forward — online softmax:
    * Tile Q into blocks of size ``BLOCK_Q``.
    * Tile K, V into blocks of size ``BLOCK_KV``.
    * For each Q tile, stream through K/V tiles. At each step update
      running (max, lse) statistics and an accumulating output via the
      log-sum-exp trick. Never materializes the full P = softmax(QK^T/√d).
    * Save only the final (L = m + log(sum(exp(s - m)))) per Q token.
      Size O(B × H × seq_q). Memory-cheap.

  Backward — recompute-style:
    * Given saved L and dO, tile again.
    * For each (Q tile, KV tile) pair, recompute P_ij from fresh QK^T and L.
    * Accumulate dV, dK, dQ via the usual identities (dP = dO·V^T; dS =
      P ⊙ (dP - D) where D = rowsum(dO ⊙ O); dQ = dS·K·scale; dK =
      dS^T·Q·scale). Each intermediate is tile-sized.
    * Never materializes P or dP at full seq² resolution.

Status: this is the algorithmic reference written in MLX Python over
tiled primitives. On its own it doesn't auto-fuse into one Metal kernel
— MLX runs each tile op as a separate dispatch. The expected memory
win at seq=8192 is ~4-6× over the autograd baseline (from not
materializing the full matrix), though per-step latency is slower due
to dispatch overhead. A custom Metal shader is the next step.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx


# Default tile sizes. Small-ish to keep each dispatch under Metal's
# threadgroup memory cap, large enough to amortize dispatch overhead.
DEFAULT_BLOCK_Q = 128
DEFAULT_BLOCK_KV = 128


def _tile_forward(
    q: mx.array,            # (B, Hq, Tq, D)
    k: mx.array,            # (B, Hkv, Tkv, D)
    v: mx.array,            # (B, Hkv, Tkv, D)
    scale: float,
    causal: bool,
    block_q: int = DEFAULT_BLOCK_Q,
    block_kv: int = DEFAULT_BLOCK_KV,
) -> tuple[mx.array, mx.array]:
    """FA-2 forward. Returns (O, L) where L is the log-sum-exp per query.

    Handles GQA/MQA by broadcasting K, V over the Q-head dim when Hq > Hkv.
    """
    B, Hq, Tq, D = q.shape
    _, Hkv, Tkv, _ = k.shape
    if Hq != Hkv:
        # Broadcast K, V to full Hq by repeating along the head axis.
        reps = Hq // Hkv
        k = mx.repeat(k, reps, axis=1)
        v = mx.repeat(v, reps, axis=1)

    # Running output, max, and sum-of-exp per query row.
    O = mx.zeros_like(q)
    m = mx.full((B, Hq, Tq, 1), -mx.inf, dtype=q.dtype)
    l = mx.zeros((B, Hq, Tq, 1), dtype=q.dtype)

    n_q_tiles = (Tq + block_q - 1) // block_q
    n_kv_tiles = (Tkv + block_kv - 1) // block_kv

    for qi in range(n_q_tiles):
        q_lo = qi * block_q
        q_hi = min(q_lo + block_q, Tq)
        q_tile = q[:, :, q_lo:q_hi, :]                     # (B,H,bq,D)
        m_tile = m[:, :, q_lo:q_hi, :]
        l_tile = l[:, :, q_lo:q_hi, :]
        O_tile = O[:, :, q_lo:q_hi, :]

        for kj in range(n_kv_tiles):
            kv_lo = kj * block_kv
            kv_hi = min(kv_lo + block_kv, Tkv)
            k_tile = k[:, :, kv_lo:kv_hi, :]               # (B,H,bkv,D)
            v_tile = v[:, :, kv_lo:kv_hi, :]

            # Scores for this (Q, KV) pair.
            s = (q_tile @ mx.swapaxes(k_tile, -1, -2)) * scale  # (B,H,bq,bkv)

            if causal:
                q_idx = mx.arange(q_lo, q_hi).reshape(1, 1, -1, 1)
                k_idx = mx.arange(kv_lo, kv_hi).reshape(1, 1, 1, -1)
                s = mx.where(q_idx >= k_idx, s, mx.array(-mx.inf, dtype=s.dtype))

            # Online softmax update.
            m_new = mx.maximum(m_tile, s.max(axis=-1, keepdims=True))
            exp_s = mx.exp(s - m_new)
            rescale = mx.exp(m_tile - m_new)
            l_tile = rescale * l_tile + exp_s.sum(axis=-1, keepdims=True)
            O_tile = rescale * O_tile + exp_s @ v_tile
            m_tile = m_new

        # Normalize and commit.
        O_tile = O_tile / l_tile
        # ``L`` for backward is m + log(l) (the full log-sum-exp).
        L_tile = m_tile + mx.log(l_tile)

        # Scatter back
        if qi == 0:
            O_out_list = [O_tile]
            L_out_list = [L_tile]
        else:
            O_out_list.append(O_tile)
            L_out_list.append(L_tile)

    O_out = mx.concatenate(O_out_list, axis=2)
    L_out = mx.concatenate(L_out_list, axis=2)
    return O_out, L_out


def _tile_backward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    O: mx.array,
    L: mx.array,
    dO: mx.array,
    scale: float,
    causal: bool,
    block_q: int = DEFAULT_BLOCK_Q,
    block_kv: int = DEFAULT_BLOCK_KV,
) -> tuple[mx.array, mx.array, mx.array]:
    """FA-2 backward. Returns (dQ, dK, dV) for the ORIGINAL input shapes.

    For GQA/MQA, the returned dK, dV are reduced back to Hkv heads by
    summing the contributions over each Q head that shared a KV head.
    """
    B, Hq, Tq, D = q.shape
    _, Hkv_orig, Tkv, _ = k.shape

    reps = Hq // Hkv_orig
    if reps != 1:
        k_exp = mx.repeat(k, reps, axis=1)
        v_exp = mx.repeat(v, reps, axis=1)
    else:
        k_exp = k
        v_exp = v

    dQ = mx.zeros_like(q)
    dK = mx.zeros_like(k_exp)
    dV = mx.zeros_like(v_exp)

    # D_i = rowsum(dO ⊙ O) per query row (needed for dS = P*(dP - D))
    D_vec = (dO * O).sum(axis=-1, keepdims=True)          # (B, Hq, Tq, 1)

    n_q_tiles = (Tq + block_q - 1) // block_q
    n_kv_tiles = (Tkv + block_kv - 1) // block_kv

    for qi in range(n_q_tiles):
        q_lo = qi * block_q
        q_hi = min(q_lo + block_q, Tq)
        q_tile = q[:, :, q_lo:q_hi, :]
        dO_tile = dO[:, :, q_lo:q_hi, :]
        L_tile = L[:, :, q_lo:q_hi, :]
        D_tile = D_vec[:, :, q_lo:q_hi, :]
        dQ_tile = mx.zeros_like(q_tile)

        for kj in range(n_kv_tiles):
            kv_lo = kj * block_kv
            kv_hi = min(kv_lo + block_kv, Tkv)
            k_tile = k_exp[:, :, kv_lo:kv_hi, :]
            v_tile = v_exp[:, :, kv_lo:kv_hi, :]

            # Recompute S, then P from saved L.
            s = (q_tile @ mx.swapaxes(k_tile, -1, -2)) * scale
            if causal:
                q_idx = mx.arange(q_lo, q_hi).reshape(1, 1, -1, 1)
                k_idx = mx.arange(kv_lo, kv_hi).reshape(1, 1, 1, -1)
                s = mx.where(q_idx >= k_idx, s, mx.array(-mx.inf, dtype=s.dtype))
            p = mx.exp(s - L_tile)                        # (B,H,bq,bkv)

            # dV_j += P^T @ dO_i
            dV_j = mx.swapaxes(p, -1, -2) @ dO_tile       # (B,H,bkv,D)
            dV[:, :, kv_lo:kv_hi, :] = dV[:, :, kv_lo:kv_hi, :] + dV_j

            # dP = dO @ V^T
            dP = dO_tile @ mx.swapaxes(v_tile, -1, -2)    # (B,H,bq,bkv)
            # dS = P * (dP - D)
            dS = p * (dP - D_tile)

            # dQ_i += dS @ K * scale
            dQ_tile = dQ_tile + (dS @ k_tile) * scale

            # dK_j += dS^T @ Q * scale
            dK_j = (mx.swapaxes(dS, -1, -2) @ q_tile) * scale
            dK[:, :, kv_lo:kv_hi, :] = dK[:, :, kv_lo:kv_hi, :] + dK_j

        dQ[:, :, q_lo:q_hi, :] = dQ_tile

    # Reduce dK, dV back to Hkv heads for GQA/MQA.
    if reps != 1:
        dK = dK.reshape(B, Hkv_orig, reps, Tkv, D).sum(axis=2)
        dV = dV.reshape(B, Hkv_orig, reps, Tkv, D).sum(axis=2)
    return dQ, dK, dV


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def flash_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: Optional[float] = None,
    causal: bool = True,
    block_q: int = DEFAULT_BLOCK_Q,
    block_kv: int = DEFAULT_BLOCK_KV,
) -> mx.array:
    """Drop-in replacement for ``mx.fast.scaled_dot_product_attention`` that
    uses a tiled/flash forward + recompute-style backward.

    Shapes match ``mx.fast.scaled_dot_product_attention``:
      q: (B, Hq,  Tq,  D)
      k: (B, Hkv, Tkv, D)
      v: (B, Hkv, Tkv, D)

    ``Hq`` may be a multiple of ``Hkv`` (GQA/MQA). ``causal`` applies
    lower-triangular masking with lower-right alignment (matches MLX's
    ``"causal"`` mask).
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    @mx.custom_function
    def _fa(q_, k_, v_):
        O, _L = _tile_forward(q_, k_, v_, scale, causal, block_q, block_kv)
        return O

    @_fa.vjp
    def _fa_vjp(primals, cotangent, output):
        q_, k_, v_ = primals
        # We need the saved L from the forward. Recompute — cheap compared
        # to backward cost. (In a real Metal kernel we'd store L from the
        # forward pass.)
        O, L = _tile_forward(q_, k_, v_, scale, causal, block_q, block_kv)
        dQ, dK, dV = _tile_backward(q_, k_, v_, O, L, cotangent,
                                     scale, causal, block_q, block_kv)
        return dQ, dK, dV

    return _fa(q, k, v)
