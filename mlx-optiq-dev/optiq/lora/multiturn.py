"""Multi-turn / agentic SFT: train on EVERY assistant turn, not just the last.

mlx-lm's ``ChatDataset`` masks everything before the final message — correct for
single-turn (prompt -> one response), but for multi-turn agentic trajectories
({"messages": [...]} with many assistant turns interleaved with tool/user turns)
it trains ONLY the last turn and masks the real actions (read_file, write_file,
run_tests). This module builds a PER-TOKEN assistant mask so the loss trains on
every assistant turn's tokens while masking system/user/tool.

Same goal as Unsloth's ``train_on_responses_only``, but template-agnostic: we
locate each assistant turn's token span via prefix tokenization
(apply_chat_template on growing message prefixes) rather than hardcoded marker
strings, so it works for any chat template that renders prefixes consistently.

Used by the LoRA trainer when the data contains multi-turn assistant
conversations (auto-detected) or when ``config.train_on_all_turns`` is set.
Plugged into mlx-lm's ``train`` via its ``loss=`` and ``iterate_batches=`` hooks.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn


def assistant_mask(messages, tokenizer):
    """Return (tokens, mask) for one conversation. ``mask[i]==1`` iff token ``i``
    belongs to an assistant turn's rendered span (content + closing tokens) —
    i.e. the tokens we want the model to learn to produce."""
    full = tokenizer.apply_chat_template(messages, tokenize=True)
    mask = [0] * len(full)
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        # `pre` = prefix + this turn's assistant HEADER. With add_generation_prompt
        # some templates (Qwen3.5) append a `<think>` token that is NOT in the
        # actual (no-think) turn, so `pre` isn't a literal prefix of `full`. Use
        # the COMMON-PREFIX length of (pre, full) = exactly where this turn's
        # content begins in `full`.
        pre = tokenizer.apply_chat_template(
            messages[:i], tokenize=True, add_generation_prompt=True)
        inc = tokenizer.apply_chat_template(messages[: i + 1], tokenize=True)
        a = 0
        for x, y in zip(pre, full):
            if x == y:
                a += 1
            else:
                break
        b = min(len(inc), len(full))   # `inc` (no gen prompt) is a true prefix of full
        for j in range(a, b):
            mask[j] = 1
    return full, mask


def count_multi_turn(data, sample=200):
    """True if the dataset has examples with >1 assistant message (agentic)."""
    n = multi = 0
    for d in data:
        msgs = d.get("messages") if isinstance(d, dict) else None
        if not msgs:
            continue
        n += 1
        if sum(1 for m in msgs if m.get("role") == "assistant") > 1:
            multi += 1
        if n >= sample:
            break
    return multi, n


class MultiTurnDataset:
    """Tokenizes {"messages": ...} into (tokens, assistant_mask) pairs."""

    def __init__(self, data, tokenizer, max_seq_length=4096):
        self.items = []
        skipped = 0
        for d in data:
            msgs = d.get("messages") if isinstance(d, dict) else None
            if not msgs:
                continue
            toks, mask = assistant_mask(msgs, tokenizer)
            if max_seq_length and len(toks) > max_seq_length:
                toks, mask = toks[:max_seq_length], mask[:max_seq_length]
            if sum(mask) == 0:           # nothing to train on -> drop
                skipped += 1
                continue
            self.items.append((toks, mask))
        self.skipped = skipped

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def iterate_batches(dataset=None, batch_size=1, max_seq_length=4096,
                    loop=True, comm_group=None, **_):
    """Yield ``(tokens, mask)`` batches, both right-padded to the batch max len.
    Matches mlx-lm's ``train`` iterate_batches call signature."""
    ds = dataset.items if hasattr(dataset, "items") else dataset
    idx = np.arange(len(ds))
    while True:
        np.random.shuffle(idx)
        for s in range(0, len(idx) - batch_size + 1, batch_size):
            chunk = [ds[j] for j in idx[s: s + batch_size]]
            ml = min(max(len(t) for t, _ in chunk), max_seq_length)
            bt = np.zeros((batch_size, ml), np.int32)
            bm = np.zeros((batch_size, ml), np.int32)
            for k, (t, m) in enumerate(chunk):
                t, m = t[:ml], m[:ml]
                bt[k, : len(t)] = t
                bm[k, : len(m)] = m
            yield mx.array(bt), mx.array(bm)
        if not loop:
            break


# --- model-agnostic body/head accessors -------------------------------------
# Works for any mlx-lm model following the standard layout, including the
# multimodal-wrapped variants (qwen3_5 etc.) where the text tower lives under
# ``model.language_model``. The fused CE only touches the final hidden->vocab
# projection + softmax CE, so it is architecture-independent: same code path for
# all Qwen models AND Llama/Gemma/Mistral/etc.


def _text_container(model):
    """The module holding the transformer body (``.model``) and optionally an
    untied ``.lm_head`` — either ``model`` itself or ``model.language_model``."""
    lm = getattr(model, "language_model", None)
    return lm if lm is not None else model


def _body(model):
    """Module whose ``__call__(inputs)`` returns hidden states ``[B,T,H]`` (the
    transformer trunk, incl. final norm), i.e. logits BEFORE the vocab head."""
    return _text_container(model).model


def _head_layer(container):
    """The vocab head layer: explicit ``lm_head`` or the tied embedding."""
    lm_head = getattr(container, "lm_head", None)
    return lm_head if lm_head is not None else container.model.embed_tokens


def _make_head_matmul(container):
    """Return ``apply(x, transpose)`` for the head projection, using the
    QUANTIZED matmul directly (no full dequant of the [vocab, H] weight — saves
    ~1.3GB on a 250k-vocab head). ``transpose=True`` => logits = x @ Wᵀ (forward);
    ``transpose=False`` => grad_h = x @ W (backward). Falls back to dense matmul
    for non-quantized heads. Model-agnostic."""
    layer = _head_layer(container)
    if hasattr(layer, "scales"):                       # quantized head
        w, sc, bi, gs, bits = (layer.weight, layer.scales, layer.biases,
                               layer.group_size, layer.bits)

        def apply(x, transpose):
            return mx.quantized_matmul(x, w, sc, bi, transpose=transpose,
                                       group_size=gs, bits=bits)
        return apply
    W = layer.weight                                   # dense [vocab, H]

    def apply(x, transpose):
        return x @ W.T if transpose else x @ W
    return apply


# Chunk size (tokens) for the fused CE. Peak extra memory is ~one
# [LOGIT_CHUNK, vocab] logits tile, independent of sequence length -> trains far
# past the ~4k wall where the full [B,T,vocab] logits OOM on a 24GB M4.
LOGIT_CHUNK = 256


def _fused_ce_loss(model, batch, mask):
    """Memory-bounded masked next-token CE via an explicit cut-cross-entropy
    custom VJP. The body runs once for hidden states; the vocab head + CE are
    computed in token chunks so the full [B,T,vocab] logit tensor is NEVER
    materialized — in EITHER the forward or the backward. The custom_function
    presents a clean grad-wrt-hidden to the outer graph, composing with the
    gated-delta training kernel's custom backward.

    Memory-minimal: uses the QUANTIZED head matmul directly (no dequant of the
    [vocab, H] weight) and subtracts the target one-hot via put_along_axis (no
    separate [chunk, vocab] one-hot tile). Peak ≈ one [LOGIT_CHUNK, vocab] tile.

    Math: logits = h @ Wᵀ; loss = Σ_i mask_i·CE(logits_i, t_i) / N.
    grad wrt hidden: dL/dh_i = mask_i/N · (softmax(logits_i) − onehot(t_i)) @ W."""
    container = _text_container(model)
    head = _make_head_matmul(container)
    h = _body(model)(batch)[:, :-1, :]              # [B, T-1, H] — small (H, not vocab)
    targets = batch[:, 1:]
    tmask = mask[:, 1:]
    B, Tm1, H = h.shape
    h = h.reshape(B * Tm1, H)
    targets = targets.reshape(B * Tm1)
    tmask = tmask.reshape(B * Tm1).astype(mx.float32)
    ntoks = tmask.sum()
    denom = mx.maximum(ntoks, mx.array(1.0))
    wdtype = h.dtype
    N = B * Tm1

    @mx.custom_function
    def ce_from_hidden(hh):
        total = mx.array(0.0, dtype=mx.float32)
        for s in range(0, N, LOGIT_CHUNK):
            e = min(s + LOGIT_CHUNK, N)
            logits = head(hh[s:e], True)             # [c, vocab] — transient
            ce = nn.losses.cross_entropy(logits, targets[s:e], reduction="none")
            total = total + (ce * tmask[s:e]).sum()
        return total / denom

    @ce_from_hidden.vjp
    def ce_from_hidden_vjp(primals, cotan, output):
        hh = primals                                 # [N, H]
        scale = cotan / denom                        # scalar
        parts = []
        for s in range(0, N, LOGIT_CHUNK):
            e = min(s + LOGIT_CHUNK, N)
            logits = head(hh[s:e], True).astype(mx.float32)      # [c, vocab]
            p = mx.softmax(logits, axis=-1)                      # [c, vocab]
            # grad_logits = (softmax - onehot)*mask*scale, in place of a separate
            # one-hot tile: subtract the per-row weight at the target column.
            tw = tmask[s:e] * scale                              # [c]
            gl = p * tw[:, None]                                 # [c, vocab]
            tgt = targets[s:e][:, None]                          # [c, 1]
            at = mx.take_along_axis(gl, tgt, axis=1) - tw[:, None]
            gl = mx.put_along_axis(gl, tgt, at, axis=1)
            parts.append(head(gl.astype(wdtype), False))         # [c, H]
        return (mx.concatenate(parts, axis=0),)

    return ce_from_hidden(h), ntoks


def loss(model, batch, mask, fused=None):
    """Masked next-token CE over assistant tokens only. ``mask`` aligns with
    inputs; we shift it to targets (predict token t from t-1).

    ``fused=True`` uses the cut-cross-entropy custom VJP (``_fused_ce_loss``),
    which bounds the CE-logit peak memory to one [LOGIT_CHUNK, vocab] tile in
    both the forward and the backward. It is gradient-equivalent to the default
    path (verified) and composes with the gated-delta training kernel, since the
    custom_function presents a clean grad-wrt-hidden to the outer graph. (The
    earlier mx.checkpoint-based version stalled above ~4k against that kernel;
    the explicit VJP that replaced it does not.)

    ``fused=False`` is the full-logits path: it materializes [B,T,vocab], which
    is what OOMs on long contexts with a large vocab.

    ``train_lora`` binds ``fused`` explicitly from ``trainer._decide_fused_ce``.
    For direct callers that leave it None it falls back to ``OPTIQ_FUSED_CE``."""
    if fused is None:
        import os
        fused = os.environ.get("OPTIQ_FUSED_CE", "0") == "1"
    if fused:
        return _fused_ce_loss(model, batch, mask)
    # Original path: materializes the full [B,T,vocab] logits (OOMs >~4k).
    logits = model(batch)[:, :-1, :]
    targets = batch[:, 1:]
    tmask = mask[:, 1:]
    ce = nn.losses.cross_entropy(logits, targets) * tmask
    ntoks = tmask.sum()
    return ce.sum() / mx.maximum(ntoks, mx.array(1)), ntoks
