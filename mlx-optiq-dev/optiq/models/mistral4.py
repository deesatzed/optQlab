"""Vendored ``mistral4`` decoder for mlx-lm.

Mistral-Small-4-119B (``model_type: mistral4`` inside a ``mistral3`` multimodal
wrapper) is, architecturally, **DeepSeek-V3**: MLA attention (``q_a_proj`` /
``q_b_proj`` / ``kv_a_proj_with_mqa`` / ``kv_b_proj``, split nope/rope head dims,
YaRN rope) plus a DeepSeek-style MoE (a ``gate`` router, ``n_shared_experts``
shared experts, and ``n_routed_experts`` fused into ``switch_mlp``). mlx-lm ships
``deepseek_v3.py`` but has **no ``mistral4`` class**, so ``mistral3.py`` falls back
to ``llama`` and silently drops every expert tensor at load. Converting that way
produces a broken shell (239 GB bf16 collapses to 4 GB with 0 expert tensors).

This module reuses deepseek_v3's attention / MoE / model, changing only what
mistral4 does differently:

* **Router**: mistral4 scores experts with ``softmax`` (deepseek_v3 uses
  ``sigmoid`` + a ``noaux_tc`` correction bias). mistral4's gate has **no**
  ``e_score_correction_bias``. With ``n_group == topk_group == 1`` (this model)
  the grouped selection is the identity, so routing is softmax -> top-k ->
  (optionally) renormalize, matching ``transformers`` ``Mistral4`` exactly.
* **Config**: rope lives under ``rope_parameters`` (not ``rope_scaling``); it is
  YaRN with ``rope_interleave: true`` (so ``traditional=True`` rope). The extra
  ``llama_4_scaling_beta`` only reshapes the long-context YaRN ramp beyond
  ``original_max_position_embeddings`` (8192); short-context decode is unaffected.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU
from mlx_lm.models.activations import swiglu


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "mistral4"
    vocab_size: int = 131072
    hidden_size: int = 4096
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    n_shared_experts: Optional[int] = 1
    n_routed_experts: Optional[int] = 128
    routed_scaling_factor: float = 1.0
    kv_lora_rank: int = 256
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 64
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 4
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_parameters: Dict = field(default_factory=dict)
    rope_scaling: Dict = None
    attention_bias: bool = False

    def __post_init__(self):
        # mistral4 keeps rope config under `rope_parameters`; deepseek-style rope
        # helpers read `rope_theta` + a yarn `scaling_config`. Map it across.
        rp = self.rope_parameters or {}
        if rp:
            self.rope_theta = rp.get("rope_theta", self.rope_theta)
            if rp.get("rope_type", rp.get("type")) in ("yarn",):
                self.rope_scaling = {
                    "rope_type": "yarn",
                    "factor": rp.get("factor", 1.0),
                    "beta_fast": rp.get("beta_fast", 32.0),
                    "beta_slow": rp.get("beta_slow", 1.0),
                    "mscale": rp.get("mscale", 1.0),
                    "mscale_all_dim": rp.get("mscale_all_dim", 0.0),
                    "original_max_position_embeddings": rp.get(
                        "original_max_position_embeddings", 8192
                    ),
                }


class Mistral4Attention(nn.Module):
    """DeepSeek-V3 MLA attention (mistral4 shares the exact tensor layout)."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.scale = self.q_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, self.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.embed_q = MultiLinear(self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads)
        self.unembed_out = MultiLinear(self.kv_lora_rank, self.v_head_dim, self.num_heads)
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias
        )

        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            if mscale_all_dim:
                scaling_factor = self.config.rope_scaling["factor"]
                if scaling_factor > 1:
                    s = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                    self.scale = self.scale * s * s

        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=self.rope_theta,
            traditional=True,
            max_position_embeddings=self.max_position_embeddings,
            scaling_config=self.config.rope_scaling,
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)
        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
            )
        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)
        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Mistral4MLP(nn.Module):
    def __init__(self, config: ModelArgs, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size if hidden_size is None else hidden_size,
            config.intermediate_size if intermediate_size is None else intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size if hidden_size is None else hidden_size,
            config.intermediate_size if intermediate_size is None else intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size if intermediate_size is None else intermediate_size,
            config.hidden_size if hidden_size is None else hidden_size,
            bias=False,
        )

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


@mx.compile
def softmax_expert_select(gates, top_k, n_group, topk_group, routed_scaling_factor, norm_topk_prob):
    # transformers Mistral4: softmax over experts, (grouped) top-k, renormalize.
    scores = mx.softmax(gates.astype(mx.float32), axis=-1)
    orig = scores
    if n_group > 1:
        s = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(s, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        s = mx.put_along_axis(s, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2)
        scores = mx.flatten(s, -2, -1)
    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(orig, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return inds, weights


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))

    def __call__(self, x):
        return softmax_expert_select(
            x @ self.weight.T,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class Mistral4MoE(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.switch_mlp = SwitchGLU(
            config.hidden_size, config.moe_intermediate_size, config.n_routed_experts
        )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            self.shared_experts = Mistral4MLP(
                config,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            )

    def __call__(self, x):
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class Mistral4DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Mistral4Attention(config)
        self.mlp = (
            Mistral4MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else Mistral4MLP(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Mistral4Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Mistral4DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, cache=None, input_embeddings=None):
        h = self.embed_tokens(x) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0], return_array=True)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = Mistral4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        return self.lm_head(out)

    def sanitize(self, weights):
        # mistral4 experts already ship fused as `switch_mlp` (no per-expert
        # stacking needed). The only remap is deepseek's MLA absorption:
        # kv_b_proj -> embed_q (nope side) + unembed_out (v side).
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}.self_attn"
            if f"{prefix}.kv_b_proj.weight" not in weights:
                continue
            quantized = f"{prefix}.kv_b_proj.scales" in weights
            v = weights.pop(f"{prefix}.kv_b_proj.weight")
            head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim
            if quantized:
                dims = self.args.kv_lora_rank
                scales = weights.pop(f"{prefix}.kv_b_proj.scales")
                biases = weights.pop(f"{prefix}.kv_b_proj.biases")
                bits = (v.shape[-1] * 32) // dims
                group_size = dims // scales.shape[-1]
                v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)
            num_heads = self.args.num_attention_heads
            v = v.reshape(num_heads, head_dim, -1)
            wk = mx.contiguous(v[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2))
            wv = mx.contiguous(v[:, self.args.qk_nope_head_dim :, :])
            if quantized:
                wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
                wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
                weights[f"{prefix}.embed_q.scales"] = wk_s
                weights[f"{prefix}.unembed_out.scales"] = wv_s
                weights[f"{prefix}.embed_q.biases"] = wk_b
                weights[f"{prefix}.unembed_out.biases"] = wv_b
            weights[f"{prefix}.embed_q.weight"] = wk
            weights[f"{prefix}.unembed_out.weight"] = wv
        return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

    @property
    def layers(self):
        return self.model.layers


def install():
    """Teach mlx-lm to load mistral4 text towers.

    The base repo's ``mistral3`` wrapper dispatches ``text_config.model_type`` to
    ``ministral3`` or (fallback) ``llama`` — neither has the MLA + 128-expert MoE,
    so every expert tensor is dropped at load. This patches that dispatch to use
    this module for ``mistral4``, and registers the module so ``mlx_lm`` can find
    it. Idempotent."""
    import sys
    import mlx.nn as nn
    import mlx_lm.models.mistral3 as m3
    from mlx_lm.models import llama, ministral3

    sys.modules.setdefault("mlx_lm.models.mistral4", sys.modules[__name__])
    if getattr(m3.Model, "_optiq_mistral4", False):
        return

    def __init__(self, args):
        nn.Module.__init__(self)
        self.args = args
        self.model_type = args.model_type
        t = args.text_config.get("model_type")
        if t == "mistral4":
            self.language_model = Model(ModelArgs.from_dict(args.text_config))
        elif t == "ministral3":
            self.language_model = ministral3.Model(ministral3.ModelArgs.from_dict(args.text_config))
        else:
            self.language_model = llama.Model(llama.ModelArgs.from_dict(args.text_config))

    m3.Model.__init__ = __init__
    m3.Model._optiq_mistral4 = True
