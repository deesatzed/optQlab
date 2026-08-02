"""MLX implementation of Dhara-AR (``model_type: dhara_ar``).

A port of codelion/dhara-250m's custom ``DharaARForCausalLM`` — a LLaMA-3-style
autoregressive transformer with **Canon layers** (causal depthwise 1-D
convolutions at four positions A/B/C/D, from "Physics of Language Models 4.1"),
QK-norm applied *after* RoPE, logit soft-capping, and tied embeddings. Dhara has
no upstream mlx-lm class; OptiQ ships this one and aliases it into
``mlx_lm.models.dhara_ar`` via ``optiq.mlx_lm_patches._register`` so
``mlx_lm.load`` (and the whole OptiQ pipeline) can load it.

The "tri-mode" decoding (block-diffusion / self-speculation) lives in the
generation logic, not the forward; this is the autoregressive core, which is all
OptiQ's sensitivity / convert / eval / LoRA / serve paths need.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache


class DharaCache(KVCache):
    """Attention KV cache + per-Canon-layer conv state.

    The Canon layers are causal depthwise convs, so correct incremental decoding
    needs each conv's previous ``kernel-1`` input activations carried across
    steps (a single-token step otherwise left-pads with zeros and loses the
    local context). Subclassing ``KVCache`` lets this object be used anywhere a
    standard cache is (attention, ``create_attention_mask``) while also holding
    the conv states in ``self.conv``.
    """

    def __init__(self):
        super().__init__()
        self.conv = {}


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "dhara_ar"
    hidden_size: int = 768
    num_hidden_layers: int = 32
    intermediate_size: int = 2176
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    vocab_size: int = 49155
    rms_norm_eps: float = 1e-6
    rope_theta: float = 8000000.0
    tie_word_embeddings: bool = True
    use_qk_norm: bool = True
    use_logit_softcap: bool = True
    logit_softcap: float = 30.0
    canon_set: str = "ABCD"
    canon_kernel: int = 4
    canon_residual: bool = True
    canon_activation: bool = False
    canon_bias: bool = False
    mask_token_id: int = 49152
    head_dim: int | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


def _rotate_half(x):
    d = x.shape[-1] // 2
    return mx.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def build_block_causal_mask(seq_len: int, block_len: int, dtype=mx.float32) -> mx.array:
    """Additive (1,1,S,S) attention bias: causal ACROSS blocks, bidirectional
    WITHIN a block. ``block_len==1`` -> standard causal; ``>=S`` -> full
    bidirectional. Used by block-diffusion + the self-speculation draft."""
    idx = mx.arange(seq_len)
    blk = (idx // block_len)
    allowed = blk[None, :] <= blk[:, None]  # (S, S): col-block <= row-block
    bias = mx.where(allowed, mx.array(0.0, dtype), mx.array(-mx.inf, dtype))
    return bias[None, None]


class DharaRoPE(nn.Module):
    """RoPE using the model's *own* ``inv_freq`` buffer (rotate-half style).

    Dhara ships per-layer ``rotary_emb.inv_freq`` and they don't match the plain
    ``θ^(-2i/d)`` schedule a stock RoPE recomputes — so we load and use them
    directly instead of ``nn.RoPE``.
    """

    def __init__(self, dim: int, base: float):
        super().__init__()
        self.inv_freq = base ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)

    def __call__(self, x, offset=0):
        if not isinstance(offset, int):  # BatchKVCache.offset is an mx.array
            offset = int(offset.reshape(-1)[0]) if hasattr(offset, "reshape") else int(offset)
        L = x.shape[2]
        pos = mx.arange(offset, offset + L, dtype=mx.float32)
        freqs = pos[:, None] * self.inv_freq[None, :].astype(mx.float32)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb).astype(x.dtype)[None, None]
        sin = mx.sin(emb).astype(x.dtype)[None, None]
        return x * cos + _rotate_half(x) * sin


class CanonLayer(nn.Module):
    """Causal depthwise 1-D conv for local mixing (left-pad kernel-1, residual)."""

    def __init__(self, dim: int, args: ModelArgs):
        super().__init__()
        self.kernel = args.canon_kernel
        self.residual = args.canon_residual
        self.activation = args.canon_activation
        self.conv = nn.Conv1d(
            dim, dim, kernel_size=self.kernel, padding=0, groups=dim,
            bias=args.canon_bias,
        )

    def __call__(self, x, cache=None, name=None):  # x: (B, L, C)
        B, L, C = x.shape
        prev = None
        if cache is not None and name is not None:
            # mlx_lm.server uses its own cache type (BatchKVCache) that lacks the
            # Canon conv state — attach one on demand so any cache works.
            if not hasattr(cache, "conv"):
                cache.conv = {}
            prev = cache.conv.get(name)
        if prev is None:
            prev = mx.zeros((B, self.kernel - 1, C), dtype=x.dtype)
        xp = mx.concatenate([prev, x], axis=1)
        if cache is not None and name is not None:
            cache.conv[name] = xp[:, -(self.kernel - 1):, :]
            # Self-speculation records the full conv input window so it can roll
            # the conv state back to an arbitrary accepted position (xp[:, m:m+k-1])
            # without a separate commit forward. Off by default (zero overhead).
            if getattr(cache, "record_conv", False):
                if not hasattr(cache, "conv_full"):
                    cache.conv_full = {}
                cache.conv_full[name] = xp
        out = self.conv(xp)  # padding=0 over (kernel-1 + L) -> length L
        if self.activation:
            out = nn.silu(out)
        return x + out if self.residual else out


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv * self.head_dim

        self.q_proj = nn.Linear(args.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, args.hidden_size, bias=False)

        if "B" in args.canon_set:
            self.canon_b_q = CanonLayer(q_dim, args)
            self.canon_b_k = CanonLayer(kv_dim, args)
            self.canon_b_v = CanonLayer(kv_dim, args)

        if args.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        # Named ``rotary_emb`` so the checkpoint's per-layer inv_freq buffer loads.
        self.rotary_emb = DharaRoPE(self.head_dim, args.rope_theta)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if "canon_b_q" in self:
            q = self.canon_b_q(q, cache, "canon_b_q")
            k = self.canon_b_k(k, cache, "canon_b_k")
            v = self.canon_b_v(v, cache, "canon_b_v")

        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache.offset if cache else 0
        q = self.rotary_emb(q, offset=offset)
        k = self.rotary_emb(k, offset=offset)
        # QK-norm is applied AFTER RoPE in Dhara (per-head over head_dim).
        if "q_norm" in self:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        if "D" in args.canon_set:
            self.canon_d = CanonLayer(args.intermediate_size, args)

    def __call__(self, x, cache=None):
        inter = nn.silu(self.gate_proj(x)) * self.up_proj(x)
        if "canon_d" in self:
            inter = self.canon_d(inter, cache, "canon_d")
        return self.down_proj(inter)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        if "A" in args.canon_set:
            self.canon_a = CanonLayer(args.hidden_size, args)
        if "C" in args.canon_set:
            self.canon_c = CanonLayer(args.hidden_size, args)

    def __call__(self, x, mask=None, cache=None):
        h = self.input_layernorm(x)
        if "canon_a" in self:
            h = self.canon_a(h, cache, "canon_a")
        x = x + self.self_attn(h, mask, cache)
        h = self.post_attention_layernorm(x)
        if "canon_c" in self:
            h = self.canon_c(h, cache, "canon_c")
        return x + self.mlp(h, cache)


class DharaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None, input_embeddings=None, mask=None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:  # default causal; a ``trimode_bias`` block mask overrides
            mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DharaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None, mask=None):
        h = self.model(inputs, cache, input_embeddings, mask=mask)
        logits = (
            self.model.embed_tokens.as_linear(h)
            if self.args.tie_word_embeddings else self.lm_head(h)
        )
        if self.args.use_logit_softcap and self.args.logit_softcap > 0:
            cap = self.args.logit_softcap
            logits = cap * mx.tanh(logits / cap)
        return logits

    def sanitize(self, weights):
        out = {}
        for k, w in weights.items():
            # keep rotary_emb.inv_freq — DharaRoPE uses the model's own buffer
            if k.endswith("lm_head.weight") and self.args.tie_word_embeddings:
                continue  # tied — drop the (duplicate) head
            if ".conv.weight" in k and w.ndim == 3 and w.shape[1] == 1:
                # PyTorch depthwise Conv1d (C, 1, kernel) -> MLX (C, kernel, 1).
                w = w.transpose(0, 2, 1)
            out[k] = w
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [DharaCache() for _ in self.model.layers]
