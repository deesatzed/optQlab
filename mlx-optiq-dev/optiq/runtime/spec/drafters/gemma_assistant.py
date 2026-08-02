"""Gemma-4 ``-assistant`` drafter for speculative decoding.

This is a port of Ollama's `x/models/gemma4/assistant.go` from PR #15980
(MIT-licensed, https://github.com/ollama/ollama/pull/15980). The
algorithm and tensor naming follow theirs; the MLX implementation here
is fresh.

Architecture (per Google's Gemma-4 MTP paper / model card):

  - 4 transformer layers, Q-only attention (no ``k_proj`` / ``v_proj``).
    At inference the drafter pulls K/V from the **target** model's last
    full-attention layer and last sliding-attention layer.
  - 3 of the 4 layers are ``sliding_attention`` (head_dim=256), the
    last is ``full_attention`` (head_dim=512). Both layer types are
    represented in the target so the K/V shapes line up.
  - ``pre_projection: (2*backbone, draft_hidden)`` projects the concat
    of (target token embedding, target last hidden state) down to the
    drafter's 256-d hidden space.
  - ``post_projection: (draft_hidden, backbone)`` projects back up so
    the next iteration can chain on the target's hidden-state space.
  - Optional centroid-clustering output head (only on the small E2B /
    E4B drafters). Decodes to top-K centroids, then expands to the
    full vocab via a precomputed token ordering. Skipped on the larger
    26B-A4B / 31B drafters which fall back to a plain LM head.

The drafter never writes its own K/V cache. ``forward`` is called once
per draft step with the typed K/V tensors from the target.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GemmaAssistantConfig:
    """Subset of ``config.json`` we actually use."""
    backbone_hidden_size: int                  # target model hidden dim
    hidden_size: int                           # drafter hidden dim (typically 256)
    intermediate_size: int                     # MLP inner dim
    num_hidden_layers: int                     # always 4 on shipped Gemma-4 drafters
    num_attention_heads: int                   # always 4 on E4B
    num_key_value_heads: int                   # target's KV head count (for shape alignment)
    head_dim: int                              # sliding-attn head dim (256 on E4B)
    global_head_dim: int                       # full-attn head dim (512 on E4B)
    rms_norm_eps: float
    layer_types: list[str]                     # ["sliding_attention", ..., "full_attention"]
    sliding_window: int
    max_position_embeddings: int
    vocab_size: int
    # Centroid head (optional)
    use_ordered_embeddings: bool = False
    num_centroids: int = 0
    centroid_intermediate_top_k: int = 0
    vocab_per_centroid: int = 0
    # RoPE
    rope_theta_full: float = 1_000_000.0
    rope_theta_sliding: float = 10_000.0
    full_partial_rotary_factor: float = 1.0

    @classmethod
    def from_dict(cls, cfg: dict) -> "GemmaAssistantConfig":
        tc = cfg.get("text_config", cfg)
        bh = cfg.get("backbone_hidden_size") or tc.get("backbone_hidden_size") or 2560

        # RoPE
        rope_params = tc.get("rope_parameters") or {}
        full_p = rope_params.get("full_attention") or {}
        slide_p = rope_params.get("sliding_attention") or {}

        # Centroid head config (lives at the top level next to text_config)
        num_centroids = int(cfg.get("num_centroids") or 0)
        top_k = int(cfg.get("centroid_intermediate_top_k") or 0)
        use_ord = bool(num_centroids and top_k)
        vocab_size = int(tc.get("vocab_size") or 262144)
        vpc = vocab_size // num_centroids if num_centroids else 0

        layer_types = list(tc.get("layer_types") or [])
        n_layers = int(tc.get("num_hidden_layers") or len(layer_types))

        return cls(
            backbone_hidden_size=int(bh),
            hidden_size=int(tc.get("hidden_size", 256)),
            intermediate_size=int(tc.get("intermediate_size", 2048)),
            num_hidden_layers=n_layers,
            num_attention_heads=int(tc.get("num_attention_heads", 4)),
            num_key_value_heads=int(tc.get("num_key_value_heads", 2)),
            head_dim=int(tc.get("head_dim", 256)),
            global_head_dim=int(tc.get("global_head_dim") or tc.get("head_dim", 256)),
            rms_norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
            layer_types=layer_types,
            sliding_window=int(tc.get("sliding_window", 512)),
            max_position_embeddings=int(tc.get("max_position_embeddings", 131072)),
            vocab_size=vocab_size,
            use_ordered_embeddings=use_ord,
            num_centroids=num_centroids,
            centroid_intermediate_top_k=top_k,
            vocab_per_centroid=vpc,
            rope_theta_full=float(full_p.get("rope_theta", 1_000_000.0)),
            rope_theta_sliding=float(slide_p.get("rope_theta", 10_000.0)),
            full_partial_rotary_factor=float(full_p.get("partial_rotary_factor", 1.0)),
        )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """Standard ``weight * normed_x`` RMSNorm.

    Gemma 1/2/3 used ``(1 + weight) * normed_x`` because the saved weights
    were stored as deltas around zero. **Gemma 4 uses raw scales** —
    saved weights range up to ~64 — so the (1+w) shift would clobber any
    small-magnitude scale (e.g. w=0.05 with (1+w)=1.05 is a 21x error
    versus the intended 0.05x).
    """
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        out = mx.fast.rms_norm(x.astype(mx.float32), self.weight, self.eps)
        return out.astype(x.dtype)


class _AssistantAttention(nn.Module):
    """Q-only attention. K/V come from the target at call time."""
    def __init__(self, cfg: GemmaAssistantConfig, layer_idx: int):
        super().__init__()
        is_full = cfg.layer_types[layer_idx] == "full_attention"
        self.is_full = is_full
        self.head_dim = cfg.global_head_dim if is_full else cfg.head_dim
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = _RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

        # RoPE: must MATCH the target's per-layer RoPE because we're attending
        # against K's the target produced. Full-attention layers use the
        # `proportional` RoPE variant (Gemma-4 specific); sliding layers use
        # plain RoPE. Delegate to mlx-lm's initialize_rope so the variant
        # dispatch matches the target.
        from mlx_lm.models.rope_utils import initialize_rope
        if is_full:
            self.rope = initialize_rope(
                self.head_dim,
                traditional=False,
                base=cfg.rope_theta_full,
                scaling_config={
                    "rope_type": "proportional",
                    "partial_rotary_factor": cfg.full_partial_rotary_factor,
                    "factor": 1.0,
                },
            )
        else:
            self.rope = initialize_rope(
                self.head_dim, traditional=False, base=cfg.rope_theta_sliding,
            )

    def __call__(
        self,
        x: mx.array,                    # (B, L, hidden_size=256)
        *,
        keys: mx.array,                 # (B, n_kv_heads, K, head_dim) from target
        values: mx.array,               # (B, n_kv_heads, K, head_dim) from target
        position: int,                  # absolute RoPE position to anchor Q at
        sliding_mask: mx.array | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        q = self.q_norm(q)
        # RoPE — Q is at absolute position ``position`` (target's last token).
        q = q.transpose(0, 2, 1, 3)  # (B, n_heads, L, head_dim)
        q = self.rope(q, offset=position)

        # SDPA. Repeat K/V heads if GQA mismatch.
        kv_groups = self.n_heads // self.n_kv_heads
        if kv_groups > 1:
            keys = mx.repeat(keys, kv_groups, axis=1)
            values = mx.repeat(values, kv_groups, axis=1)

        out = mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=self.scale, mask=sliding_mask,
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * self.head_dim)
        return self.o_proj(out)


class _AssistantMLP(nn.Module):
    def __init__(self, cfg: GemmaAssistantConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class _AssistantBlock(nn.Module):
    """One transformer block: norm -> Q-only attn -> norm -> norm -> MLP -> norm."""
    def __init__(self, cfg: GemmaAssistantConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = cfg.layer_types[layer_idx]
        self.self_attn = _AssistantAttention(cfg, layer_idx)
        self.mlp = _AssistantMLP(cfg)
        self.input_layernorm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.pre_feedforward_layernorm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_feedforward_layernorm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.layer_scalar = mx.zeros((1,))

    def __call__(
        self,
        h: mx.array,
        *,
        shared_kv: dict[str, tuple[mx.array, mx.array]],
        position: int,
        sliding_window: int = 512,
    ) -> mx.array:
        k, v = shared_kv[self.layer_type]
        # Build a sliding-window mask over the K positions for sliding-
        # attention layers. Q is at ``position``; can attend to K's in
        # [max(0, position - sliding_window + 1), position]. Older K's
        # get -inf. Full-attention layers have no mask (attend to all).
        mask = None
        if self.layer_type == "sliding_attention":
            k_len = k.shape[-2]
            # Cache covers absolute positions [position - k_len + 1 .. position].
            first_cached = position - k_len + 1
            window_start = max(0, position - sliding_window + 1)
            allowed_from_idx = max(0, window_start - first_cached)
            if allowed_from_idx > 0:
                mask = mx.zeros((1, 1, 1, k_len), dtype=h.dtype)
                mask = mx.where(
                    mx.arange(k_len) < allowed_from_idx,
                    mx.array(-1e9, dtype=h.dtype),
                    mask,
                )
        residual = h
        h = self.input_layernorm(h)
        attn = self.self_attn(
            h, keys=k, values=v, position=position, sliding_mask=mask,
        )
        attn = self.post_attention_layernorm(attn)
        h = residual + attn
        residual = h
        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        h = residual + h
        # Per-layer scalar gain. Gemma-4 stores this as a raw scalar (init
        # to 1.0 at training start, drifts per layer). mlx-lm's gemma4_text
        # applies it as ``h * layer_scalar`` (gemma4_text.py:386-387).
        # Some trained drafter layers learn very small values (0.03–0.36)
        # which heavily attenuate; the network compensates elsewhere.
        h = h * self.layer_scalar
        return h


# ---------------------------------------------------------------------------
# Full drafter model
# ---------------------------------------------------------------------------


class GemmaAssistantDrafter(nn.Module):
    """The full Gemma-4 ``-assistant`` drafter.

    Use ``GemmaAssistantDrafter.from_pretrained(repo_or_path)`` to load.
    Call ``forward(last_token_emb, target_hidden, shared_kv, position)``
    to produce one draft token plus its post-projected hidden (which
    feeds the next draft step).
    """

    def __init__(self, cfg: GemmaAssistantConfig):
        super().__init__()
        self.cfg = cfg
        # Drafter has its own token embeddings (NOT tied to target's). The
        # embed_tokens.weight is shape (vocab, drafter_hidden=256), so it's
        # natively in drafter space; no need for the target's tokenizer to
        # produce drafter-space embeddings.
        self.model_embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [_AssistantBlock(cfg, i) for i in range(cfg.num_hidden_layers)]
        self.model_norm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        # Projections that bridge target hidden space <-> drafter hidden space.
        # pre_projection: in=(target token emb || target last hidden) -> drafter hidden
        # Conventionally 2*backbone=5120 -> 256 on E4B.
        self.pre_projection = nn.Linear(
            cfg.backbone_hidden_size + cfg.hidden_size,
            cfg.hidden_size, bias=False,
        )
        # post_projection: drafter hidden 256 -> backbone hidden 2560 so the
        # output can re-enter the chain as the next "target hidden".
        self.post_projection = nn.Linear(
            cfg.hidden_size, cfg.backbone_hidden_size, bias=False,
        )

        # Output head — centroid clustering OR plain LM head.
        if cfg.use_ordered_embeddings:
            # Two-stage decode: project hidden -> centroid logits, take top-K
            # centroids, expand to those centroids' vocab slice via gather.
            self.centroid_weight = mx.zeros((cfg.num_centroids, cfg.hidden_size))
            # Token ordering: maps centroid_id * vocab_per_centroid + offset -> token_id
            self.token_ordering = mx.zeros((cfg.vocab_size,), dtype=mx.int64)
        else:
            # Plain LM head over full vocab.
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        *,
        last_token_emb: mx.array,       # (1, 1, backbone_hidden_size) - the TARGET's
                                        # embed_tokens lookup of the most recent token
        target_hidden: mx.array,        # (B, L, backbone_hidden_size) typically L=1
        shared_kv: dict[str, tuple[mx.array, mx.array]],
        position: int,                  # target's last position
    ) -> tuple[mx.array, mx.array]:
        """One drafter forward.

        Parameters
        ----------
        last_token_emb : (1, 1, backbone_hidden_size)
            The TARGET model's ``embed_tokens`` lookup of the most recent
            emitted token. Must be in the target's backbone hidden space
            (2560-d on E4B) so the concat with ``target_hidden`` lines up
            with the trained ``pre_projection (5120 -> 256)`` weight.

        Returns
        -------
        next_logits : (vocab_size,)
        next_target_hidden : (1, 1, backbone_hidden_size)
            The post-projected drafter output, ready to feed back as
            ``target_hidden`` for the next draft step.
        """
        # 1. Build the drafter input: concat (target_token_emb, target_hidden).
        x = mx.concatenate([last_token_emb, target_hidden], axis=-1)  # (1, 1, 2*backbone)
        h = self.pre_projection(x)                                    # (1, 1, hidden=256)

        # 2. Run the 4 transformer blocks. Each block constructs its own
        #    sliding-window mask if needed.
        for blk in self.layers:
            h = blk(h, shared_kv=shared_kv, position=position,
                    sliding_window=self.cfg.sliding_window)

        # 3. Final norm.
        h = self.model_norm(h)

        # 4. Output head.
        if self.cfg.use_ordered_embeddings:
            logits = self._centroid_decode(h)
        else:
            logits = self.lm_head(h)

        # 5. post_projection: drafter hidden -> backbone hidden for the next step.
        next_target_hidden = self.post_projection(h)  # (1, 1, 2560)

        return logits.squeeze(0).squeeze(0), next_target_hidden

    # ------------------------------------------------------------------

    def _centroid_decode(self, h: mx.array) -> mx.array:
        """Centroid clustering output head.

        h:           (1, 1, hidden)
        centroids:   (num_centroids, hidden)        — score every centroid
        select top-K centroids by score, gather their VocabPerCentroid token IDs
        from token_ordering, materialise sparse vocab-shape logits.
        """
        cfg = self.cfg
        # Centroid scores: (1, 1, num_centroids)
        centroid_scores = h @ self.centroid_weight.T

        # Top-K centroids: indices in (1, 1, K)
        top_k = min(cfg.centroid_intermediate_top_k, cfg.num_centroids)
        topk_idx = mx.argpartition(-centroid_scores, kth=top_k - 1, axis=-1)[..., :top_k]
        # Gather candidate token IDs: ordering is laid out as
        # token_ordering[c * VocabPerCentroid + offset] -> actual token id.
        # For each chosen centroid we take VocabPerCentroid token ids.
        ordering = self.token_ordering.reshape(cfg.num_centroids, cfg.vocab_per_centroid)
        # ordering[topk_idx] -> (1, 1, K, VocabPerCentroid)
        candidate_ids = ordering[topk_idx]                       # (1, 1, K, vpc)
        candidate_emb = self.model_embed_tokens.weight[candidate_ids]  # (1, 1, K, vpc, hidden)

        # Score each candidate token via dot product with h (broadcast).
        # h: (1, 1, hidden); candidate_emb: (1, 1, K, vpc, hidden)
        h_b = h[..., None, None, :]                              # (1, 1, 1, 1, hidden)
        scores = (h_b * candidate_emb).sum(axis=-1)              # (1, 1, K, vpc)
        scores = scores.reshape(1, 1, top_k * cfg.vocab_per_centroid)
        flat_ids = candidate_ids.reshape(1, 1, top_k * cfg.vocab_per_centroid)

        # Materialise sparse vocab-shape logits initialised to -inf-ish.
        # Use mx.put_along_axis (available in MLX 0.31+) which is a native
        # scatter and keeps dtype safe (avoids the numpy bfloat16 dance).
        logits = mx.full((1, 1, cfg.vocab_size), -1e30, dtype=h.dtype)
        logits = mx.put_along_axis(
            logits, flat_ids.astype(mx.int32),
            scores.astype(logits.dtype), axis=-1,
        )
        return logits

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, repo_or_path: str) -> "GemmaAssistantDrafter":
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(repo_or_path) if not Path(repo_or_path).is_dir()
                    else repo_or_path)
        cfg_dict = json.loads((path / "config.json").read_text())
        cfg = GemmaAssistantConfig.from_dict(cfg_dict)
        model = cls(cfg)
        weights = mx.load(str(next(path.glob("*.safetensors"))))
        weights = _rename_weights(weights)
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
        return model


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------


def _rename_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Map HF tensor names to the names our nn.Module hierarchy expects.

    HF naming uses ``model.layers.N.subname`` plus the special
    ``masked_embedding.*`` block for the centroid head. Our module uses
    ``model_embed_tokens``, ``layers.N.``, etc.
    """
    out: dict[str, mx.array] = {}
    for k, v in weights.items():
        nk = k
        # Token embedding
        if k == "model.embed_tokens.weight":
            nk = "model_embed_tokens.weight"
        elif k.startswith("model.norm."):
            nk = "model_norm" + k[len("model.norm"):]
        elif k.startswith("model.layers."):
            nk = "layers." + k[len("model.layers."):]
        elif k == "masked_embedding.centroids.weight":
            nk = "centroid_weight"
        elif k == "masked_embedding.token_ordering":
            nk = "token_ordering"
        # pre_projection / post_projection / lm_head stay as-is
        out[nk] = v
    return out
