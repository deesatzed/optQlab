"""Laguna (poolside) MLX architecture, vendored for OptiQ.

model_type == "laguna" (LagunaForCausalLM). mlx-lm has no upstream class and
the only public "mlx" quants ship the *PyTorch* modeling_laguna.py, which stock
mlx-lm cannot run — so OptiQ vendors the arch here and aliases it into
``mlx_lm.models.laguna`` on ``import optiq`` (see _register.py).

Laguna is a combination of well-known pieces (validated against poolside's
reference modeling_laguna.py and the safetensors index):

* MoE: DeepSeek-V3-style sigmoid router with an auxiliary-loss-free selection
  bias (``e_score_correction_bias``, added for *selection* only; the returned
  weights come from the unbiased sigmoid scores), norm_topk_prob, a shared
  expert, and fused SwiGLU experts. Layer 0 is a dense MLP (``mlp_only_layers``).
  The checkpoint stores experts un-fused per-expert; ``sanitize`` stacks them
  into mlx-lm's ``SwitchGLU``.
* Attention: GQA + per-head QK-RMSNorm (applied before RoPE) + GLM-style partial
  rotary + a softplus output gate applied before ``o_proj``. ``gating="per-head"``
  emits one gate per head (broadcast across head_dim); ``"per-element"``/True
  emits one per channel.
* Hybrid attention: ``layer_types`` alternates ``full_attention`` /
  ``sliding_attention`` (window 512), each with its own RoPE — full layers use
  YaRN with partial_rotary_factor 0.5, sliding layers use plain RoPE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "laguna"
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 40
    num_attention_heads: int = 48
    num_attention_heads_per_layer: Optional[List[int]] = None
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 100352
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    # MoE
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = field(default_factory=lambda: [0])
    moe_router_logit_softcapping: float = 0.0
    moe_routed_scaling_factor: float = 1.0
    # Attention / gating
    gating: Any = "per-head"
    sliding_window: int = 512
    layer_types: Optional[List[str]] = None
    swa_attention_sink_enabled: bool = False
    # RoPE (nested: {"full_attention": {...}, "sliding_attention": {...}})
    rope_parameters: Optional[Dict[str, Any]] = None
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.num_attention_heads_per_layer is None:
            self.num_attention_heads_per_layer = [
                self.num_attention_heads
            ] * self.num_hidden_layers
        if self.layer_types is None:
            # Fall back to "every 4th layer is full" if not specified.
            self.layer_types = [
                "full_attention" if i % 4 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {"rope_type": "default", "rope_theta": 500000.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            }


def _softplus(x: mx.array) -> mx.array:
    # log(1 + exp(x)), numerically stable, matches F.softplus.
    return mx.logaddexp(x, mx.zeros_like(x))


def _build_rope(head_dim: int, rp: Dict[str, Any], max_pos: int):
    """Build a RoPE for one attention flavour from a Laguna rope_parameters entry.

    Handles partial rotary (``partial_rotary_factor``) and YaRN scaling.
    """
    rp = dict(rp or {})
    partial = float(rp.get("partial_rotary_factor", 1.0))
    dims = int(head_dim * partial)
    base = float(rp.get("rope_theta", 10000.0))
    rope_type = rp.get("rope_type", "default")
    if rope_type in (None, "default", "linear"):
        scaling = {"rope_type": "linear", "factor": rp["factor"]} if rope_type == "linear" else None
        return initialize_rope(dims, base, False, scaling, max_pos), dims
    # YaRN (and any other advanced type initialize_rope supports): pass the
    # entry straight through — it carries factor / original_max_position_embeddings
    # / beta_fast / beta_slow, and initialize_rope reads rope_type from it.
    scaling = {k: v for k, v in rp.items() if k not in ("rope_theta", "partial_rotary_factor")}
    return initialize_rope(dims, base, False, scaling, max_pos), dims


class LagunaMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LagunaRouter(nn.Module):
    """Sigmoid router with aux-loss-free selection bias (DeepSeek-V3 noaux_tc)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.softcap = float(args.moe_router_logit_softcapping or 0.0)
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.num_experts,))

    def __call__(self, x):
        logits = x @ self.weight.T
        if self.softcap > 0.0:
            logits = mx.tanh(logits / self.softcap) * self.softcap
        scores = mx.sigmoid(logits.astype(mx.float32))
        sel = scores + self.e_score_correction_bias.astype(mx.float32)
        inds = mx.argpartition(-sel, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        return inds, weights.astype(x.dtype)


class LagunaMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = LagunaRouter(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_expert = LagunaMLP(args, args.shared_expert_intermediate_size)
        self.routed_scaling = float(args.moe_routed_scaling_factor)

    def __call__(self, x):
        inds, weights = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(y.dtype)
        if self.routed_scaling != 1.0:
            y = y * self.routed_scaling
        return y + self.shared_expert(x)


class LagunaAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.head_dim = hd = args.head_dim
        # Laguna varies attention heads per layer (full=48, sliding=64).
        self.n_heads = nh = args.num_attention_heads_per_layer[layer_idx]
        self.n_kv = nkv = args.num_key_value_heads
        self.scale = hd**-0.5

        self.q_proj = nn.Linear(dim, nh * hd, bias=False)
        self.k_proj = nn.Linear(dim, nkv * hd, bias=False)
        self.v_proj = nn.Linear(dim, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, dim, bias=False)

        self.q_norm = nn.RMSNorm(hd, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(hd, eps=args.rms_norm_eps)

        self.gate_per_head = args.gating == "per-head"
        self.gating = bool(args.gating)
        if self.gating:
            g_out = nh if self.gate_per_head else nh * hd
            self.g_proj = nn.Linear(dim, g_out, bias=False)

        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"
        key = "sliding_attention" if self.is_sliding else "full_attention"
        self.rope, self._rope_dims = _build_rope(
            hd, args.rope_parameters.get(key, {}), args.max_position_embeddings
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.n_kv, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.n_kv, self.head_dim)

        # QK-norm per head, before RoPE, then move heads to axis 1.
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)

        if self.gating:
            gate = _softplus(self.g_proj(x).astype(mx.float32)).astype(out.dtype)
            if self.gate_per_head:
                out = (
                    out.reshape(B, L, self.n_heads, self.head_dim) * gate[..., None]
                ).reshape(B, L, -1)
            else:
                out = out * gate

        return self.o_proj(out)


class LagunaDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = LagunaAttention(args, layer_idx)
        is_moe = (
            layer_idx not in args.mlp_only_layers
            and args.num_experts > 0
            and (layer_idx + 1) % args.decoder_sparse_step == 0
        )
        self.mlp = LagunaMoE(args) if is_moe else LagunaMLP(args, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class LagunaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            LagunaDecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._full = [t == "full_attention" for t in args.layer_types]

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        # Anchor each mask on a cache of the matching type — Laguna's layer 0 is
        # full_attention, so the sliding mask must come from a sliding layer's
        # RotatingKVCache, not cache[0] (unlike gemma3, whose layer 0 is sliding).
        first_full = next((i for i, f in enumerate(self._full) if f), 0)
        first_slide = next((i for i, f in enumerate(self._full) if not f), first_full)
        full_mask = create_attention_mask(h, cache[first_full])
        sliding_mask = create_attention_mask(
            h, cache[first_slide], window_size=self.args.sliding_window
        )

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            mask = full_mask if self._full[i] else sliding_mask
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LagunaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        # Stack per-expert projections into SwitchGLU 3D tensors, and move the
        # router selection bias from mlp.experts.* onto the gate.
        for l in range(self.args.num_hidden_layers):
            p = f"model.layers.{l}"
            bias = weights.pop(f"{p}.mlp.experts.e_score_correction_bias", None)
            if bias is not None:
                weights[f"{p}.mlp.gate.e_score_correction_bias"] = bias
            if f"{p}.mlp.experts.0.gate_proj.weight" in weights:
                for n in ("gate_proj", "up_proj", "down_proj"):
                    stacked = [
                        weights.pop(f"{p}.mlp.experts.{e}.{n}.weight")
                        for e in range(self.args.num_experts)
                    ]
                    weights[f"{p}.mlp.switch_mlp.{n}.weight"] = mx.stack(stacked)
        return weights

    @property
    def quant_predicate(self):
        # Keep the router weight (mlp.gate.weight) out of low-bit quantization,
        # matching qwen3_moe/deepseek_v3 practice — it's tiny and sensitive.
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for t in self.args.layer_types:
            if t == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches
