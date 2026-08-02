"""Fused SwiGLU MLP for MLX — forward + analytic backward under a single
``mx.custom_function`` so the gate/up/silu_gate/h intermediates never enter
the autograd graph.

Standard LLaMA/Qwen MLP:

    h = silu(gate_proj(x)) * up_proj(x)
    y = down_proj(h)

Autograd through the generic MLP saves ``gate``, ``up``, ``h`` (and
transitively ``silu(gate)``) for backward. At Qwen3.5-9B ``intermediate_size
= 5632`` and T=8192, that's ``8192 × 5632 × 2 ≈ 92 MB`` per tensor per
layer — ~280 MB of MLP transients alive per active block.

With ``grad_checkpoint`` already wrapping each ``DecoderLayer``, MLX
recomputes the whole layer during backward, so these transients live only
for one layer at a time — but that one layer's worth is still a real slice
of peak memory.

This module puts the MLP behind ``mx.custom_function`` with an analytic
vjp. MLX's autograd tape only sees ``(x, gate_w, up_w, down_w) -> y``, so
the per-layer MLP transients never enter the saved-for-backward set. The
vjp recomputes ``gate``, ``up``, ``silu`` on demand and emits
``dx, dgate_w, dup_w, ddown_w`` using the closed-form derivative:

    d/dg silu(g)   = sigmoid(g) * (1 + g*(1 - sigmoid(g)))
    dh/dg          = silu'(g) * up
    dh/du          = silu(g)

This is the same math every LLM backend implements — we keep it in MLX
Python so any head_dim / dtype MLX supports just works, at the cost of a
single extra matmul pair during recompute.
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx
import mlx.nn as nn


def _silu_and_deriv(g: mx.array) -> tuple[mx.array, mx.array]:
    """Return ``(silu(g), silu'(g))`` with a single sigmoid evaluation.

    silu(g)  = g * sigmoid(g)
    silu'(g) = sigmoid(g) + g * sigmoid(g) * (1 - sigmoid(g))
             = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
    """
    sig = mx.sigmoid(g)
    silu = g * sig
    # Factor for cheap reuse of sigmoid + silu.
    silu_prime = sig * (1.0 + g * (1.0 - sig))
    return silu, silu_prime


def fused_swiglu_mlp(
    x: mx.array,            # (B, T, D)
    gate_w: mx.array,       # (H, D)
    up_w: mx.array,         # (H, D)
    down_w: mx.array,       # (D, H)
) -> mx.array:
    """Fused SwiGLU MLP: ``down(silu(gate(x)) * up(x))`` with analytic vjp.

    Shapes follow the mlx-lm Linear convention (weight is out × in).
    """
    @mx.custom_function
    def _fwd(x_, gw_, uw_, dw_):
        gate = x_ @ gw_.T                             # (B, T, H)
        up   = x_ @ uw_.T                             # (B, T, H)
        silu = gate * mx.sigmoid(gate)                # (B, T, H)
        h = silu * up                                 # (B, T, H)
        y = h @ dw_.T                                 # (B, T, D)
        return y

    @_fwd.vjp
    def _vjp(primals, dy, _output):
        x_, gw_, uw_, dw_ = primals
        # Recompute forward intermediates — these never live past this vjp.
        gate = x_ @ gw_.T
        up   = x_ @ uw_.T
        silu, silu_prime = _silu_and_deriv(gate)
        h = silu * up

        # dh = dy @ down_w       (upstream through the down projection)
        dh = dy @ dw_                                 # (B, T, H)

        # ddown_w = dy.T @ h  (sum over (B, T) via reshape/matmul)
        # Flatten batch+seq so the weight grad is a 2D matmul.
        B, T, D = x_.shape
        H = gw_.shape[0]
        dy_flat = dy.reshape(-1, D)                   # (BT, D)
        h_flat = h.reshape(-1, H)                     # (BT, H)
        ddown_w = dy_flat.T @ h_flat                   # (D, H)

        # du = dh * silu;   dg = dh * up * silu'(gate)
        du = dh * silu
        dg = dh * up * silu_prime

        # dgate_w = dg^T @ x ;  dup_w = du^T @ x     (both (H, D))
        x_flat = x_.reshape(-1, D)                    # (BT, D)
        dg_flat = dg.reshape(-1, H)
        du_flat = du.reshape(-1, H)
        dgate_w = dg_flat.T @ x_flat                  # (H, D)
        dup_w   = du_flat.T @ x_flat                  # (H, D)

        # dx = dg @ gate_w + du @ up_w                 # (B, T, D)
        dx = dg @ gw_ + du @ uw_

        return dx, dgate_w, dup_w, ddown_w

    return _fwd(x, gate_w, up_w, down_w)


class FusedSwiGLU(nn.Module):
    """Drop-in replacement for mlx-lm's Qwen3NextMLP / LLaMA MLP.

    Wraps three ``nn.Linear`` (or quantized) projections and dispatches through
    the fused forward+backward. When the projections are QuantizedLinear the
    fused path falls back to the stock module (`fused_swiglu_mlp` assumes
    plain weight tensors). We detect this at init time.
    """

    def __init__(self, dim: int, hidden_dim: int, use_bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=use_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        # Fused path requires plain Linear with no bias.
        gate = self.gate_proj
        up = self.up_proj
        down = self.down_proj
        plain = (
            isinstance(gate, nn.Linear) and not getattr(gate, "bias", None)
            and isinstance(up, nn.Linear) and not getattr(up, "bias", None)
            and isinstance(down, nn.Linear) and not getattr(down, "bias", None)
        )
        if plain:
            return fused_swiglu_mlp(x, gate.weight, up.weight, down.weight)
        # Fallback: standard SwiGLU expressed through the submodules so
        # quantized / biased variants still work.
        return down(gate(x) * mx.sigmoid(gate(x)) * up(x))
