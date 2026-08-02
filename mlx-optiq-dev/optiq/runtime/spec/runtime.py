"""Generic speculative-decoding loop for OptiQ.

Supports γ>=1 greedy spec decoding for Gemma-4 -assistant drafters.
For γ>1 the drafter is run γ times autoregressively (each draft chains
the previous draft's post_projection output and embedding back in),
then the target verifies all γ drafts in one batched forward and we
accept the longest matching prefix; cache rolls back on partial accept.

Reference: Ollama's mtp.go (MIT). The recurrent-hidden / typed-K/V
discipline follows their implementation; the MLX code is fresh.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Literal

import mlx.core as mx

from .drafters.gemma_assistant import GemmaAssistantDrafter
from .kv_view import extract_typed_kv


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class SpecConfig:
    """Configuration for one speculative-decoding session."""
    gamma: int = 1                            # draft tokens per outer step
    max_tokens: int = 256                     # generation cap
    eos_token_id: int | None = None           # stop at this id (model-specific)
    accept_temp: float = 0.0                  # 0 = greedy verify (lossless)


@dataclass
class SpecEvent:
    """One event yielded by ``spec_generate``."""
    kind: Literal["token", "done"]
    token_id: int = -1
    text: str = ""
    from_draft: bool = False                  # True iff this token was drafted+accepted


@dataclass
class SpecStats:
    """Aggregate stats for one generation, attached to the final event."""
    n_emitted: int = 0
    n_drafted: int = 0
    n_accepted: int = 0
    n_target_calls: int = 0
    elapsed_s: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.n_accepted / self.n_drafted if self.n_drafted else 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.n_emitted / self.elapsed_s if self.elapsed_s else 0.0


# ---------------------------------------------------------------------------
# Target forward helpers
# ---------------------------------------------------------------------------


def _target_inner(target, ids: mx.array, cache):
    """Run the target's inner forward and return the post-final-norm
    hidden state (BEFORE the lm_head projection).

    Gemma-4 layout: ``target.language_model.model(...)`` returns ``norm(h)``.
    """
    lm = target.language_model if hasattr(target, "language_model") else target
    inner = lm.model
    return inner(ids, cache=cache)


def _project_to_logits(target, hidden: mx.array) -> mx.array:
    """Apply lm_head (or tied embedding) to a hidden state.

    Mirrors the projection step in ``Gemma4ForConditionalGeneration.__call__``
    so we can split "hidden" out separately for the drafter while still
    getting the same logits as the standard forward.
    """
    lm = target.language_model if hasattr(target, "language_model") else target
    if getattr(lm, "tie_word_embeddings", True):
        return lm.model.embed_tokens.as_linear(hidden)
    return lm.lm_head(hidden)


def _target_step(target, token_ids: mx.array, cache) -> tuple[mx.array, mx.array]:
    """Advance the target by ``token_ids`` (shape (1, n_new)) using its KV
    cache. Returns ``(hidden, logits)`` where shapes are ``(1, n_new, H)``
    and ``(1, n_new, V)``.
    """
    hidden = _target_inner(target, token_ids, cache)
    logits = _project_to_logits(target, hidden)
    return hidden, logits


def _target_embed(target, token_id: int) -> mx.array:
    """Look up the TARGET's ``embed_tokens`` for ``token_id``, SCALED by
    ``embed_scale = sqrt(hidden_size)``, returned as
    ``(1, 1, backbone_hidden_size)``.

    Why scaled: Gemma's training-time convention applies the scale at the
    input-embedding step. Ollama's ``TokenEmbeddings`` does the same
    (gemma4.go:1094-1097), and the drafter's ``pre_projection`` was
    trained against the scaled form. Without it acceptance is 0%.
    """
    lm = target.language_model if hasattr(target, "language_model") else target
    emb = lm.model.embed_tokens
    scale = lm.model.embed_scale
    return emb(mx.array([[token_id]])) * scale


def _greedy(logits: mx.array) -> int:
    """argmax over the last token's vocab dimension."""
    return int(mx.argmax(logits[0, -1, :]))


# ---------------------------------------------------------------------------
# γ=1 speculative loop
# ---------------------------------------------------------------------------


def spec_generate(
    target,
    drafter: GemmaAssistantDrafter,
    tokenizer,
    prompt: str,
    cfg: SpecConfig | None = None,
) -> Iterator[SpecEvent]:
    """Generate tokens using the target + drafter, yielding ``SpecEvent``s.

    Supports any ``gamma >= 1``. Always greedy verify. EOS handling honors
    ``cfg.eos_token_id``; if None we fall through to the model's own EOS.

    For ``gamma > 1`` the drafter runs autoregressively γ times against the
    same target K/V snapshot, the target verifies all γ drafts in one
    forward, and we accept the longest matching prefix. On partial accept
    the target cache rolls back by (γ - k_accept) entries.
    """
    from mlx_lm.models.cache import make_prompt_cache

    cfg = cfg or SpecConfig()
    if cfg.gamma < 1:
        raise ValueError(f"gamma must be >= 1; got {cfg.gamma}")
    if cfg.accept_temp != 0.0:
        raise NotImplementedError("only greedy verify (accept_temp=0.0) is supported")

    eos_id = cfg.eos_token_id
    if eos_id is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)

    gamma = cfg.gamma

    # 1. Prefill the target with the prompt.
    input_ids = mx.array([tokenizer.encode(prompt)])
    caches = make_prompt_cache(target)
    t0 = time.time()
    hidden, logits = _target_step(target, input_ids, caches)
    next_token = _greedy(logits)
    stats = SpecStats(n_target_calls=1)
    last_target_hidden = hidden[:, -1:, :]

    yield _emit_token(next_token, tokenizer, from_draft=False)
    stats.n_emitted += 1

    if next_token == eos_id:
        stats.elapsed_s = time.time() - t0
        yield SpecEvent(kind="done", text=str(stats))
        return

    # 2. Outer loop. Each iteration drafts γ tokens, then verifies, then
    #    emits between 1 and γ+1 tokens.
    while stats.n_emitted < cfg.max_tokens:
        # 2a. Draft γ tokens against a frozen snapshot of the target K/V.
        position = caches[0].offset - 1
        shared_kv = extract_typed_kv(target, caches)

        drafts: list[int] = []
        draft_token_for_input = next_token
        draft_hidden_for_input = last_target_hidden
        for k in range(gamma):
            last_emb = _target_embed(target, draft_token_for_input)
            draft_logits, next_drafter_hidden = drafter.forward(
                last_token_emb=last_emb,
                target_hidden=draft_hidden_for_input,
                shared_kv=shared_kv,
                position=position + k,
            )
            d_k = int(mx.argmax(draft_logits))
            drafts.append(d_k)
            stats.n_drafted += 1
            # Chain into the next draft step. The drafter's post_projection
            # output replaces target_hidden; the draft token's target-side
            # embedding replaces last_token_emb. Both stay in target hidden
            # space so the trained pre_projection still applies.
            draft_token_for_input = d_k
            draft_hidden_for_input = next_drafter_hidden

        # 2b. Target verify: feed [next_token, d_1, ..., d_γ] in one forward
        #     (γ+1 new positions). Logits[..., k, :] predicts what should
        #     follow the token at verify_ids[..., k]. So:
        #       - logits[0] should predict d_1 (or correct it)
        #       - logits[γ-1] should predict d_γ (or correct it)
        #       - logits[γ] predicts the BONUS token after d_γ (only useful
        #         if all γ drafts are accepted)
        verify_ids = mx.array([[next_token] + drafts])
        v_hidden, v_logits = _target_step(target, verify_ids, caches)
        stats.n_target_calls += 1

        # 2c. Accept the longest matching prefix.
        gt_tokens = [_greedy(v_logits[:, k:k+1, :]) for k in range(gamma + 1)]
        k_accept = 0
        while k_accept < gamma and drafts[k_accept] == gt_tokens[k_accept]:
            k_accept += 1
        stats.n_accepted += k_accept

        # 2d. Emit accepted drafts.
        stop = False
        for k in range(k_accept):
            yield _emit_token(drafts[k], tokenizer, from_draft=True)
            stats.n_emitted += 1
            if drafts[k] == eos_id or stats.n_emitted >= cfg.max_tokens:
                stop = True
                break

        if stop:
            break

        # 2e. Emit the correction (or bonus, if all γ accepted).
        emit_token = gt_tokens[k_accept]
        yield _emit_token(emit_token, tokenizer, from_draft=False)
        stats.n_emitted += 1

        # 2f. Roll back the cache by the rejected count. If k_accept == γ
        #     (all accepted), the bonus token's K/V is NOT in cache (verify
        #     only added γ+1 positions: next_token + γ drafts; bonus comes
        #     after that), so no trim.
        if k_accept < gamma:
            _trim_cache(caches, gamma - k_accept)

        if emit_token == eos_id or stats.n_emitted >= cfg.max_tokens:
            break

        # 2g. Set up state for the next outer iteration. The new "next_token"
        #     is the correction/bonus we just emitted (whose K/V is NOT in
        #     cache yet). last_target_hidden is the target hidden at position
        #     k_accept (the position right BEFORE the new next_token).
        next_token = emit_token
        last_target_hidden = v_hidden[:, k_accept:k_accept + 1, :]

    stats.elapsed_s = time.time() - t0
    yield SpecEvent(kind="done", text=str(stats))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emit_token(tok_id: int, tokenizer, *, from_draft: bool) -> SpecEvent:
    try:
        text = tokenizer.decode([tok_id])
    except Exception:
        text = ""
    return SpecEvent(
        kind="token", token_id=tok_id, text=text, from_draft=from_draft,
    )


def _trim_cache(caches, n: int) -> None:
    """Roll back every cache by ``n`` tokens. Used on partial-accept."""
    if n <= 0:
        return
    for c in caches:
        if hasattr(c, "trim") and c.is_trimmable():
            c.trim(n)
        else:
            # RotatingKVCache becomes non-trimmable once the ring buffer
            # has wrapped. For short contexts this is not hit; raise so we
            # notice during testing rather than silently producing wrong
            # output.
            raise RuntimeError(
                f"cache {type(c).__name__} is not trimmable; "
                "spec decoding requires trimmable caches"
            )
