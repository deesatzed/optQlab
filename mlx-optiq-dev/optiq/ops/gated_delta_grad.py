"""Memory-efficient, differentiable gated-delta-rule for LoRA training.

mlx-lm's ``gated_delta`` ships a fast FORWARD-ONLY Metal scan kernel (no vjp)
for inference, and a pure-ops reference (``gated_delta_ops``) used during
training. The ops path is differentiable but its autograd graph stores every
time-step state — O(T) memory — so on the 24 GatedDeltaNet layers of a
qwen3_next 4B it OOMs a 24 GB Mac at seq > ~1k.

This module provides ``gated_delta_diff`` — a drop-in for ``gated_delta_ops``
that wraps the fast forward kernel + a hand-written **Metal backward kernel**
in ``mx.custom_function``. The backward uses the gated-delta-rule adjoint
recurrence with **chunked recompute** (store only √T boundary states, recompute
each chunk's states; numerically stable, O(√T) memory). Gradients verified to
float32 precision against autodiff of ``gated_delta_ops``.

Routing is scoped to qwen3_next via ``enable_gated_delta_training()`` — every
other architecture keeps stock MLX, exactly as intended.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Tuple

import mlx.core as mx

from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops


# ── backward Metal kernel ────────────────────────────────────────────────────
# One thread per (n=B*Hv, dv). Phase A recomputes the forward storing only
# chunk-boundary states (every C steps). Phase B walks chunks in reverse: reload
# the boundary state, recompute that chunk's states into a small reused scratch,
# then run the adjoint recurrence, using atomic adds for the cross-Dv reductions
# (dq, dk, dg, dbeta) and a direct write for dv. GQA is handled by the caller
# (q,k repeated to Hv on the way in; dq,dk reduced to Hk on the way out).
# 32 threads per (n, dv) — one simdgroup over the Dk axis (NPT=Dk/32 slots each).
# Dk-axis reductions (u, ddelta, dg) use simd_sum; Dv-axis reductions (dq, dk via
# per-slot atomics; dg, dbeta, dv via the dk_idx==0 thread) use atomics. Chunked
# recompute: store only √T boundary states, recompute each chunk into a small
# reused scratch. All 32 threads share dv (from grid.y) so simd_sum never diverges.
_BW_SOURCE = r"""
    uint dk_idx=thread_position_in_threadgroup.x; uint dv=thread_position_in_grid.y; uint n=thread_position_in_grid.z;
    if (dv>=Dv) return;
    uint b=n/H; uint h=n%H; uint idx=n*Dv+dv; uint sb=dk_idx*NPT;
    float st[NPT]; for (uint i=0;i<NPT;i++) st[i]=s0[(n*Dv+dv)*Dk+sb+i];
    for (uint i=0;i<NPT;i++) bnd[(idx*NCH+0)*Dk+sb+i]=st[i];
    for (uint t=0;t<T;t++){
        uint bqk=((b*T+t)*H+h)*Dk; uint bv=((b*T+t)*H+h)*Dv;
        float gt=g[(b*T+t)*H+h], bt=beta[(b*T+t)*H+h]; float up=0.0f;
        for (uint i=0;i<NPT;i++){ st[i]*=gt; up+=st[i]*k[bqk+sb+i]; }
        float u=simd_sum(up); float delta=bt*(v[bv+dv]-u);
        for (uint i=0;i<NPT;i++) st[i]+=delta*k[bqk+sb+i];
        if ((t+1)%C==0 && (t+1)/C<NCH) for (uint i=0;i<NPT;i++) bnd[(idx*NCH+(t+1)/C)*Dk+sb+i]=st[i];
    }
    float dS[NPT]; for (uint i=0;i<NPT;i++) dS[i]=0.0f;
    for (int c=(int)NCH-1;c>=0;c--){
        uint a=(uint)c*C; uint e=min((uint)T,a+C);
        for (uint i=0;i<NPT;i++) ch[(idx*(C+1)+0)*Dk+sb+i]=bnd[(idx*NCH+(uint)c)*Dk+sb+i];
        for (uint t=a;t<e;t++){
            uint bqk=((b*T+t)*H+h)*Dk; uint bv=((b*T+t)*H+h)*Dv; uint j=t-a;
            float gt=g[(b*T+t)*H+h], bt=beta[(b*T+t)*H+h];
            for (uint i=0;i<NPT;i++) ch[(idx*(C+1)+j+1)*Dk+sb+i]=ch[(idx*(C+1)+j)*Dk+sb+i]*gt;
            float up=0.0f; for (uint i=0;i<NPT;i++) up+=ch[(idx*(C+1)+j+1)*Dk+sb+i]*k[bqk+sb+i];
            float u=simd_sum(up); float delta=bt*(v[bv+dv]-u);
            for (uint i=0;i<NPT;i++) ch[(idx*(C+1)+j+1)*Dk+sb+i]+=delta*k[bqk+sb+i];
        }
        for (int tt=(int)e-1;tt>=(int)a;tt--){
            uint t=(uint)tt; uint j=t-a; uint bqk=((b*T+t)*H+h)*Dk; uint bv=((b*T+t)*H+h)*Dv; uint bs=(b*T+t)*H+h;
            float gt=g[bs], bt=beta[bs], dyt=dy[bv+dv];
            float up=0.0f; for (uint i=0;i<NPT;i++) up+=ch[(idx*(C+1)+j)*Dk+sb+i]*gt*k[bqk+sb+i];
            float u=simd_sum(up); float delta=bt*(v[bv+dv]-u);
            float ddp=0.0f; for (uint i=0;i<NPT;i++){ float Z=dS[i]+dyt*q[bqk+sb+i]; ddp+=Z*k[bqk+sb+i]; }
            float ddelta=simd_sum(ddp); float du=-bt*ddelta;
            if (dk_idx==0){ dv_out[bv+dv]=bt*ddelta; atomic_fetch_add_explicit((device atomic_float*)&dbeta[bs], ddelta*(v[bv+dv]-u), memory_order_relaxed); }
            float dgp=0.0f;
            for (uint i=0;i<NPT;i++){
                float Stm1=ch[(idx*(C+1)+j)*Dk+sb+i]; float St=ch[(idx*(C+1)+j+1)*Dk+sb+i];
                float Z=dS[i]+dyt*q[bqk+sb+i]; float P=Stm1*gt; float dP=Z+du*k[bqk+sb+i];
                atomic_fetch_add_explicit((device atomic_float*)&dq[bqk+sb+i], St*dyt, memory_order_relaxed);
                atomic_fetch_add_explicit((device atomic_float*)&dk[bqk+sb+i], P*du+Z*delta, memory_order_relaxed);
                dgp+=dP*Stm1; dS[i]=gt*dP;
            }
            float dg_t=simd_sum(dgp);
            if (dk_idx==0) atomic_fetch_add_explicit((device atomic_float*)&dg[bs], dg_t, memory_order_relaxed);
        }
    }
"""

_bw_kernel = mx.fast.metal_kernel(
    name="gated_delta_bw_chunked",
    input_names=["q", "k", "v", "g", "beta", "dy", "s0"],
    output_names=["dq", "dk", "dv_out", "dg", "dbeta", "bnd", "ch"],
    source=_BW_SOURCE,
)


def _choose_chunk(T: int) -> int:
    """~√T, clamped, so scratch stays small without too many chunks."""
    import math
    return max(8, min(128, int(round(math.sqrt(max(1, T))))))


def _backward(qH, kH, v, g, beta, dy, s0fp, B, T, Hv, Dk, Dv):
    C = _choose_chunk(T)
    NCH = (T + C - 1) // C
    NPT = Dk // 32                       # state slots per thread (32 threads / Dk axis)
    dq, dk, dvv, dg, dbeta, _, _ = _bw_kernel(
        inputs=[qH, kH, v, g, beta, dy, s0fp],
        template=[("H", Hv), ("Dk", Dk), ("Dv", Dv), ("T", T),
                  ("C", C), ("NCH", NCH), ("NPT", NPT)],
        grid=(32, Dv, B * Hv), threadgroup=(32, 1, 1),
        output_shapes=[qH.shape, kH.shape, v.shape, g.shape, beta.shape,
                       (B * Hv * Dv, NCH * Dk), (B * Hv * Dv, (C + 1) * Dk)],
        output_dtypes=[mx.float32] * 7, init_value=0.0)
    return dq, dk, dvv, dg, dbeta


def _supported(q, k, v, g, beta, state, mask) -> bool:
    """Only the common training-prefill case the kernel covers; otherwise the
    caller falls back to stock gated_delta_ops (correct, just slower)."""
    if mask is not None:
        return False
    if g.ndim != 3:           # vectorized (per-Dk) gating not yet in the kernel
        return False
    if q.ndim != 4:
        return False
    if q.shape[-1] % 32 != 0:  # simd kernel splits Dk across 32 threads
        return False
    return True


def gated_delta_diff(q, k, v, g, beta, state=None, mask=None):
    """Drop-in for ``gated_delta_ops`` — same (q,k,v,g,beta,state,mask) -> (y,state),
    but with a fast O(√T)-memory Metal backward. Falls back to the ops path for
    shapes the kernel doesn't cover (vectorized gating, masks, non-GPU)."""
    if mx.default_device() != mx.gpu or not mx.metal.is_available() \
            or not _supported(q, k, v, g, beta, state, mask):
        return gated_delta_ops(q, k, v, g, beta, state, mask)

    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    R = Hv // Hk
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    s0fp = state.astype(mx.float32)
    in_dtype = q.dtype

    @mx.custom_function
    def _gd(q, k, v, g, beta):
        return gated_delta_kernel(q, k, v, g, beta, state, mask)   # (y, final_state)

    @_gd.vjp
    def _gd_vjp(primals, cotangents, outputs):
        q_, k_, v_, g_, beta_ = primals
        dy = cotangents[0].astype(mx.float32)               # cotangent for y
        qH = mx.repeat(q_.astype(mx.float32), R, -2) if R > 1 else q_.astype(mx.float32)
        kH = mx.repeat(k_.astype(mx.float32), R, -2) if R > 1 else k_.astype(mx.float32)
        dq, dk, dvv, dg, dbeta = _backward(
            qH, kH, v_.astype(mx.float32), g_.astype(mx.float32),
            beta_.astype(mx.float32), dy, s0fp, B, T, Hv, Dk, Dv)
        if R > 1:                                           # GQA: reduce Hv -> Hk
            dq = dq.reshape(B, T, Hk, R, Dk).sum(3)
            dk = dk.reshape(B, T, Hk, R, Dk).sum(3)
        return (dq.astype(in_dtype), dk.astype(in_dtype), dvv.astype(in_dtype),
                dg.astype(g.dtype), dbeta.astype(beta.dtype))

    return _gd(q, k, v, g, beta)


# ── routing (scoped to qwen3_next) ───────────────────────────────────────────
_install_depth = 0


def _install() -> None:
    import mlx_lm.models.gated_delta as _gd_mod
    if getattr(_gd_mod, "_optiq_gd_installed", False):
        _gd_mod._optiq_gd_depth += 1
        return
    _gd_mod._optiq_gd_orig_ops = _gd_mod.gated_delta_ops
    _gd_mod._optiq_gd_orig_kernel = _gd_mod.gated_delta_kernel
    _gd_mod.gated_delta_ops = gated_delta_diff
    # `gated_delta_update` dispatches to the raw Metal kernel whenever
    # `use_kernel` is set, and that kernel has no vjp. qwen3_next passes
    # `use_kernel=not self.training`, so patching gated_delta_ops was enough
    # for it; qwen3_5 never passes the flag, so it always took the kernel
    # branch and every backward died with
    #   [Primitive::vjp] Not implemented for CustomKernel.
    # Route both entry points at the differentiable implementation. The
    # inference kernel is restored on exit, so decode is unaffected.
    _gd_mod.gated_delta_kernel = gated_delta_diff
    _gd_mod._optiq_gd_installed = True
    _gd_mod._optiq_gd_depth = 1


def _uninstall() -> None:
    import mlx_lm.models.gated_delta as _gd_mod
    if not getattr(_gd_mod, "_optiq_gd_installed", False):
        return
    _gd_mod._optiq_gd_depth -= 1
    if _gd_mod._optiq_gd_depth > 0:
        return
    _gd_mod.gated_delta_ops = _gd_mod._optiq_gd_orig_ops
    _gd_mod.gated_delta_kernel = _gd_mod._optiq_gd_orig_kernel
    del _gd_mod._optiq_gd_orig_ops
    del _gd_mod._optiq_gd_orig_kernel
    _gd_mod._optiq_gd_installed = False


@contextmanager
def enable_gated_delta_training():
    """Route qwen3_next GatedDeltaNet training through the fast Metal backward."""
    _install()
    try:
        yield
    finally:
        _uninstall()
