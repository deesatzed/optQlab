"""Vendored MLX decoder for LLaDA2-MoE (`model_type: "llada2_moe"`).

LLaDA2.2-flash (inclusionAI) is a ~103B **diffusion** MoE LM: 256 routed experts
(top-8) + 1 shared expert, 32 layers, and a *bidirectional* attention (a diffusion
model attends to the whole masked canvas, so `is_causal=False`). mlx-lm has no
`llada2_moe` class, so — like `mistral4` and DiffusionGemma — OptiQ vendors one.

The MoE is DeepSeek-V3-style (sigmoid grouped routing with an expert bias,
`n_group`/`topk_group`, `routed_scaling_factor`, a shared expert), so the routing
and fused-expert machinery are reused from `mlx_lm.models.deepseek_v3`
(`SwitchGLU`, `group_expert_select`). What is LLaDA2-specific and written here:

  * a **fused `query_key_value`** projection split into GQA q/k/v,
  * per-head **QK-RMSNorm** (`query_layernorm` / `key_layernorm`),
  * **partial RoPE** (`rotary_dim` = head_dim · `partial_rotary_factor`),
  * **non-causal** attention,
  * the checkpoint naming (`attention.*`, `model.word_embeddings`,
    `mlp.gate.expert_bias`, `first_k_dense_replace` dense layers).

For a static 2-bit quant only the module *structure* has to match the checkpoint
(mlx-lm.convert loads + quantizes weights, it does not run a forward), so the
diffusion decode loop lives elsewhere; this file makes the weights loadable and
quantizable, and the forward is correct for later inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.deepseek_v3 import SwitchGLU, group_expert_select


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "llada2_moe"
    hidden_size: int = 4096
    intermediate_size: int = 9216
    moe_intermediate_size: int = 1024
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    partial_rotary_factor: float = 0.5
    rope_theta: float = 3000000.0
    rope_scaling: Optional[dict] = None
    rms_norm_eps: float = 1e-6
    vocab_size: int = 157184
    num_experts: int = 256
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    first_k_dense_replace: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    n_group: int = 8
    topk_group: int = 4
    use_qk_norm: bool = True
    use_qkv_bias: bool = False
    use_bias: bool = False
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False


class LLaDA2MLP(nn.Module):
    """Standard gated MLP — used for the dense (first_k_dense_replace) layers and
    for the shared expert."""

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LLaDA2Gate(nn.Module):
    """Router: sigmoid scores + expert bias + DeepSeek-V3 grouped top-k select."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.n_routed_experts = args.num_experts
        self.routed_scaling_factor = args.routed_scaling_factor
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.expert_bias = mx.zeros((args.num_experts,))

    def __call__(self, x):
        return group_expert_select(
            x @ self.weight.T,
            self.expert_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class LLaDA2MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.gate = LLaDA2Gate(args)
        self.shared_experts = LLaDA2MLP(
            args.hidden_size, args.moe_intermediate_size * args.num_shared_experts,
            bias=args.use_bias,
        )

    def __call__(self, x):
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        return y + self.shared_experts(x)


class LLaDA2Attention(nn.Module):
    """GQA with a fused QKV projection, per-head QK-RMSNorm, partial RoPE, and
    bidirectional (non-causal) attention."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        rope_dim = int(self.head_dim * args.partial_rotary_factor)

        qkv_out = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.query_key_value = nn.Linear(args.hidden_size, qkv_out, bias=args.use_qkv_bias)
        self.dense = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=args.use_bias)

        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(rope_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        qkv = self.query_key_value(x)
        q, k, v = mx.split(
            qkv,
            [self.n_heads * self.head_dim,
             (self.n_heads + self.n_kv_heads) * self.head_dim],
            axis=-1,
        )
        q = q.reshape(B, L, self.n_heads, self.head_dim)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.use_qk_norm:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(out)


class LLaDA2DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attention = LLaDA2Attention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if layer_idx < args.first_k_dense_replace:
            self.mlp = LLaDA2MLP(args.hidden_size, args.intermediate_size, bias=args.use_bias)
        else:
            self.mlp = LLaDA2MoE(args)

    def __call__(self, x, mask=None, cache=None):
        h = x + self.attention(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class LLaDA2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [LLaDA2DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None, input_embeddings=None, mask=None):
        h = input_embeddings if input_embeddings is not None else self.word_embeddings(inputs)
        # LLaDA2 is a diffusion LM (bidirectional). Attention is full by default; the
        # block-diffusion decoder passes an explicit block-causal `mask` (lower-triangular
        # over blocks, bidirectional within a block). A causal mask is used only when an
        # autoregressive cache is supplied (so mlx-lm's generate still works for AR probes).
        if mask is None and cache is not None and cache[0] is not None:
            mask = create_attention_mask(h, cache)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LLaDA2Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None, mask=None):
        h = self.model(inputs, cache=cache, input_embeddings=input_embeddings, mask=mask)
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        # Stack the 256 per-expert projections into a fused switch_mlp tensor, the
        # same way mlx-lm's deepseek_v3 does. rotary_emb.inv_freq is precomputed and
        # not a parameter here.
        n_experts = self.args.num_experts
        out = {}
        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k:
                continue
            out[k] = v
        weights = out
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}.mlp"
            for m in ("gate_proj", "up_proj", "down_proj"):
                for suffix in ("weight", "scales", "biases"):
                    first = f"{prefix}.experts.0.{m}.{suffix}"
                    if first in weights:
                        stacked = [
                            weights.pop(f"{prefix}.experts.{e}.{m}.{suffix}")
                            for e in range(n_experts)
                        ]
                        weights[f"{prefix}.switch_mlp.{m}.{suffix}"] = mx.stack(stacked)
        return weights


def _block_causal_mask(total_len: int, block_len: int, dtype=mx.bfloat16):
    """Additive attention mask for block-diffusion decode: bidirectional within a
    block, causal across blocks (block b attends to blocks 0..b). 0 where allowed,
    -inf where not. Shape (total_len, total_len)."""
    idx = mx.arange(total_len)
    blk = idx // block_len                                   # block index per position
    allowed = blk[None, :] <= blk[:, None]                   # query row i, key col j
    return mx.where(allowed, mx.array(0.0, dtype), mx.array(-mx.inf, dtype))


def _num_transfer_tokens(block_length: int, steps: int):
    """How many masked positions to force-unmask at each step (sums to block_length).
    Fewer steps -> more forced unmaskings per step. Mirrors the reference schedule."""
    if steps == 0:
        return []
    base, rem = divmod(block_length, steps)
    return [base + (1 if i < rem else 0) for i in range(steps)]


def diffusion_generate(model, prompt_ids, *, gen_length=512, block_length=32,
                       steps=32, threshold=0.5, temperature=0.0,
                       mask_id=156895, eos_id=156892, eos_early_stop=True,
                       progress=None):
    """Block-wise masked-diffusion decode (M2T) for LLaDA2-MoE, ported from the
    reference `generate`. Greedy by default. Semi-autoregressive: the canvas is a
    grid of `block_length` blocks; each block starts fully masked and is refined
    over `steps` forward passes, unmasking the highest-confidence positions (with a
    per-step floor from the transfer schedule) until it is fully committed, then the
    next block begins. Block-causal attention keeps frozen blocks from attending to
    still-masked future positions.

    `prompt_ids`: python list[int]. Returns list[int] of generated ids (post-prompt,
    up to and including the first eos)."""
    prompt = list(prompt_ids)
    P = len(prompt)
    num_blocks = (P + gen_length + block_length - 1) // block_length
    total = num_blocks * block_length

    x = [mask_id] * total
    x[:P] = prompt
    amask = _block_causal_mask(total, block_length)
    prefill_blocks = P // block_length

    for nb in range(prefill_blocks, num_blocks):
        b0, b1 = nb * block_length, (nb + 1) * block_length
        win_mask = amask[:b1, :b1][None, None]              # (1,1,b1,b1)
        # positions in this block that are prompt/context (never rewritten)
        frozen = [i for i in range(b0, b1) if x[i] != mask_id]
        n_init_mask = (b1 - b0) - len(frozen)
        if n_init_mask == 0:
            continue
        schedule = _num_transfer_tokens(n_init_mask, steps)

        for step in range(steps):
            masked_pos = [i for i in range(b0, b1) if x[i] == mask_id]
            if not masked_pos:
                break
            xin = mx.array([x[:b1]], dtype=mx.int32)
            logits = model(xin, mask=win_mask)[0]           # (b1, vocab)
            blk = logits[b0:b1]                             # (block_length, vocab)
            probs = mx.softmax(blk.astype(mx.float32), axis=-1)
            greedy = mx.argmax(blk, axis=-1)
            conf = mx.take_along_axis(probs, greedy[:, None], axis=-1)[:, 0]
            greedy = greedy.tolist(); conf = conf.tolist()

            # candidates = currently-masked positions, ranked by confidence
            cands = sorted(masked_pos, key=lambda i: conf[i - b0], reverse=True)
            floor = schedule[step] if step < len(schedule) else len(cands)
            n_take = 0
            for rank, i in enumerate(cands):
                take = conf[i - b0] > threshold or rank < floor
                if take:
                    x[i] = greedy[i - b0]; n_take += 1
            if n_take == 0 and cands:                       # guarantee progress
                x[cands[0]] = greedy[cands[0] - b0]
            if progress:
                progress(nb, num_blocks, step, sum(1 for i in range(P, b1) if x[i] != mask_id))

        # force-commit any residual masks in the block
        for i in range(b0, b1):
            if x[i] == mask_id:
                xin = mx.array([x[:b1]], dtype=mx.int32)
                g = mx.argmax(model(xin, mask=win_mask)[0][i], axis=-1).item()
                x[i] = g

        if eos_early_stop and eos_id in x[P:b1]:
            break

    gen = x[P:P + gen_length]
    if eos_id in gen:
        gen = gen[:gen.index(eos_id) + 1]
    # strip any residual mask tokens defensively
    return [t for t in gen if t != mask_id]


def install():
    """Register this module as ``mlx_lm.models.llada2_moe`` so mlx-lm's
    load_model / convert can build LLaDA2-MoE checkpoints. Idempotent."""
    import sys
    sys.modules.setdefault("mlx_lm.models.llada2_moe", sys.modules[__name__])
