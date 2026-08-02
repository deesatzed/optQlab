"""FlashAttention-2 for MLX: fused forward, tiled backward, no O(T²) tensor.

Why this exists
---------------
``mx.fast.scaled_dot_product_attention`` has an excellent fused forward. Its
*backward*, however, is ordinary autograd over that graph, and it materializes
the ``[B, Hq, T, T]`` score tensor. Gradient checkpointing keeps one layer's
copy live at a time but cannot shrink it, so a high-head-count model at long
context runs out of memory: 32 heads at 16k is 48 GiB of scores.

The previous answer was a hand-written Metal kernel (``flash_attention_metal``).
It bounds memory correctly but is a v0 — one thread per query row, scalar loops
over ``head_dim``, tiles shrinking 64→32→16→8 as ``head_dim`` grows — and runs
14–137× slower than stock, worst at the ``head_dim=256`` that Qwen3.5 uses.

This module takes the third path. The forward is stock (fused, fast, cheap).
The backward is the standard FlashAttention-2 recomputation: walk the query
axis in blocks, rebuild that block's scores, and consume them immediately. Peak
extra memory is ``O(B · Hq · BLOCK · T)`` rather than ``O(B · Hq · T²)`` — the
same asymptotic win as the Metal kernel — while every matmul goes through MLX's
GEMM, which is already simdgroup-backed and tuned. We do not out-engineer
Apple's matmul; we stop bypassing it.

Math (per query block, with S = scale · Q Kᵀ and P = softmax(S)):

    D  = rowsum(dO ⊙ O)          # == rowsum(dP ⊙ P), the softmax Jacobian term
    dV += Pᵀ dO
    dP  = dO Vᵀ
    dS  = P ⊙ (dP − D)
    dQ  = scale · dS K
    dK += scale · dSᵀ Q

GQA/MQA: K and V are broadcast to ``Hq`` heads for the block matmuls, and the
resulting dK/dV are summed back down to ``Hkv`` at the end.
"""

from __future__ import annotations

import os
from typing import Optional

import mlx.core as mx

__all__ = ["flash_attention_tiled", "DEFAULT_BLOCK"]

# Query-block size. Peak extra memory is B·Hq·BLOCK·T·4 bytes for the fp32
# softmax. Measured on Hq=32, T=8192 (peak GiB / step ms): 512 -> 17.34/1456,
# 256 -> 10.77/1397, 128 -> 5.86/1452, 64 -> 5.19/1445. Time is flat, so take
# the memory; below 128 the peak is dominated by the fused forward itself and
# there is nothing left to win.
DEFAULT_BLOCK = 128


def _block_size() -> int:
    env = os.environ.get("OPTIQ_FLASH_BLOCK")
    if env:
        try:
            return max(32, int(env))
        except ValueError:
            pass
    return DEFAULT_BLOCK


def _expand_kv(x: mx.array, n_rep: int) -> mx.array:
    """(B, Hkv, T, D) -> (B, Hkv*n_rep, T, D), repeating each kv head n_rep times."""
    if n_rep == 1:
        return x
    B, Hkv, T, D = x.shape
    return mx.broadcast_to(x[:, :, None], (B, Hkv, n_rep, T, D)).reshape(B, Hkv * n_rep, T, D)


def _reduce_kv(x: mx.array, Hkv: int, n_rep: int) -> mx.array:
    """(B, Hkv*n_rep, T, D) -> (B, Hkv, T, D), summing each query head's group."""
    if n_rep == 1:
        return x
    B, _, T, D = x.shape
    return x.reshape(B, Hkv, n_rep, T, D).sum(axis=2)


def _backward(q, k, v, o, do, scale: float, causal: bool):
    B, Hq, Tq, D = q.shape
    Hkv, Tk = k.shape[1], k.shape[2]
    n_rep = Hq // Hkv
    wdt = q.dtype

    ke = _expand_kv(k, n_rep)
    ve = _expand_kv(v, n_rep)
    kT = ke.swapaxes(-1, -2)          # (B, Hq, D, Tk)
    vT = ve.swapaxes(-1, -2)          # (B, Hq, D, Tk)

    # rowsum(dP ⊙ P) == rowsum(dO ⊙ O); one O(T·D) reduction instead of O(T²).
    d_row = (do.astype(mx.float32) * o.astype(mx.float32)).sum(-1, keepdims=True)

    block = _block_size()
    k_idx = mx.arange(Tk)

    # Accumulate dK/dV already reduced to Hkv heads: holding them expanded to Hq
    # in fp32 costs n_rep x more for nothing.
    dq_blocks = []
    dk = mx.zeros(k.shape, dtype=mx.float32)
    dv = mx.zeros(v.shape, dtype=mx.float32)

    for start in range(0, Tq, block):
        end = min(start + block, Tq)
        qb = q[:, :, start:end, :]                       # (B, Hq, bq, D)
        dob = do[:, :, start:end, :]

        s = mx.matmul(qb, kT).astype(mx.float32) * scale  # (B, Hq, bq, Tk)
        if causal:
            # Query t attends to keys <= t. Tq may differ from Tk (prefix cache);
            # align the query positions to the *end* of the key axis, as mlx does.
            q_pos = mx.arange(start, end) + (Tk - Tq)
            s = mx.where(k_idx[None, :] <= q_pos[:, None], s, mx.array(-mx.inf, mx.float32))

        p = mx.softmax(s, axis=-1)                        # (B, Hq, bq, Tk) fp32
        del s

        p_w = p.astype(wdt)
        dv = dv + _reduce_kv(mx.matmul(p_w.swapaxes(-1, -2), dob), Hkv, n_rep).astype(mx.float32)

        dp = mx.matmul(dob, vT).astype(mx.float32)        # (B, Hq, bq, Tk)
        ds = (p * (dp - d_row[:, :, start:end, :])).astype(wdt)
        del p, dp

        dq_blocks.append(mx.matmul(ds, ke) * scale)
        dkb = mx.matmul(ds.swapaxes(-1, -2), qb) * scale
        dk = dk + _reduce_kv(dkb, Hkv, n_rep).astype(mx.float32)

    dq = mx.concatenate(dq_blocks, axis=2)
    return dq, dk.astype(wdt), dv.astype(wdt)


def flash_attention_tiled(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: Optional[float] = None,
    causal: bool = True,
) -> mx.array:
    """Drop-in attention with a fused forward and an O(BLOCK × T) backward.

    Shapes match ``mx.fast.scaled_dot_product_attention``::

        q: (B, Hq,  Tq, D)
        k: (B, Hkv, Tk, D)
        v: (B, Hkv, Tk, D)
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    mask = "causal" if causal else None

    @mx.custom_function
    def _fa(q_, k_, v_):
        return mx.fast.scaled_dot_product_attention(q_, k_, v_, scale=scale, mask=mask)

    @_fa.vjp
    def _fa_vjp(primals, cotangents, output):
        q_, k_, v_ = primals
        do = cotangents if isinstance(cotangents, mx.array) else cotangents[0]
        o = output if isinstance(output, mx.array) else output[0]
        return _backward(q_, k_, v_, o, do, scale, causal)

    return _fa(q, k, v)
