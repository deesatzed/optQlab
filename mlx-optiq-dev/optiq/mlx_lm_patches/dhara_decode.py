"""Block-diffusion + self-speculation decode for Dhara-AR (the tri-mode model).

The base ``dhara_ar`` MLX model is autoregressive. Its other two decode modes —
ported here from codelion/dhara-chat — drive the *same* weights with a block
attention bias (``build_block_causal_mask``: causal across blocks, bidirectional
within), which lets the model fill a block of ``[MASK]`` tokens in parallel:

* **block-diffusion** — append a block of MASK, iteratively commit the high-
  confidence predictions until the block is filled (DiffusionGemma-style).
  Prefix-cached: each forward processes only the block on top of a cached
  KV + Canon-conv prefix (O(block) per step), so it stays coherent at long
  context where greedy AR collapses into repetition.
* **self-speculation** — draft a block with the block forward, verify it with
  the causal (AR) forward, accept the matching prefix + 1 (AR-quality). This is
  what OptiQ's ``--mtp`` speculative path drives; it runs full-sequence forwards.

Note on speed: dhara is overhead-bound (a 32-token forward costs ~the same as a
1-token forward), so AR's one-token-per-forward is not memory-bound — the modes
trade throughput for the parallel/iterative structure, not weight loading.
"""

from __future__ import annotations

import re

import mlx.core as mx
import numpy as np

from .dhara_ar import build_block_causal_mask

REP_DEFAULT = 1.3


def _mask_dtype(model):
    return model.model.embed_tokens.weight.dtype


def _im_end_id(tokenizer) -> int:
    try:
        tid = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if tid:
            return tid[-1]
    except Exception:
        pass
    return int(getattr(tokenizer, "eos_token_id", 0) or 0)


def _rep_pen(logits, seen_ids, penalty, vocab):
    """Repetition penalty on a (n, V) logit block over already-seen token ids."""
    if penalty == 1.0 or len(seen_ids) == 0:
        return logits
    vm = np.zeros(vocab, dtype=bool)
    vm[np.unique(np.asarray(seen_ids))] = True
    vmask = mx.array(vm)[None, :]
    adj = mx.where(logits > 0, logits / penalty, logits * penalty)
    return mx.where(vmask, adj, logits)


def diffusion_generate(model, tokenizer, prompt_ids, **kw):
    """Block-diffusion decode -> cleaned text."""
    im_end = _im_end_id(tokenizer)
    return _decode_clean(tokenizer, diffusion_ids(model, tokenizer, prompt_ids, **kw), im_end)


def selfspec_generate(model, tokenizer, prompt_ids, **kw):
    """Self-speculative decode -> cleaned text."""
    im_end = _im_end_id(tokenizer)
    return _decode_clean(tokenizer, selfspec_ids(model, tokenizer, prompt_ids, **kw), im_end)


def _global_block_mask(S0, n_new, block_len, dt):
    """Additive mask for ``n_new`` appended query positions [S0, S0+n_new).

    Blocks are defined by GLOBAL position (``pos // block_len``, aligned from 0) —
    exactly as ``build_block_causal_mask`` and the training recipe (block 32) define
    them. A query may attend to a key only if the key's block <= the query's block:
    bidirectional WITHIN a block, causal ACROSS blocks.

    This matters: when the cached prefix length is not a multiple of ``block_len``,
    the appended tokens STRADDLE two global blocks, and the earlier ones must not see
    the later ones. Masking the append with all-zeros (fully bidirectional) presents
    the model with an attention pattern it never saw in training — which collapses
    block-diffusion into repetition ("water, water, water...") and reduces
    self-speculation drafts to noise (near-zero accept -> slower than plain AR).
    """
    L = S0 + n_new
    qb = (mx.arange(S0, L) // block_len)[:, None]      # query block ids (global)
    kb = (mx.arange(L) // block_len)[None, :]          # key   block ids (global)
    m = mx.where(kb <= qb, mx.array(0.0, dt), mx.array(-1e9, dt))
    return m[None, None].astype(dt)


def _snap_cache(cache):
    """Snapshot the per-layer (offset, conv-state) of a committed prefix."""
    return [(c.offset, dict(c.conv)) for c in cache]


def _restore_cache(cache, snap):
    """Restore a snapshot. ``KVCache.update_and_fetch`` writes from ``offset``,
    so resetting offset overwrites/trims — the block forward re-uses the slot."""
    for c, (o, cv) in zip(cache, snap):
        c.offset = o
        c.conv = dict(cv)


def diffusion_ids(model, tokenizer, prompt_ids, *, block_len=32,
                  threshold=0.5, max_new=128, rep=REP_DEFAULT):
    """Prefix-cached block-diffusion. Returns generated token ids (np array).

    The prompt (and every committed block) lives in a KV + Canon-conv cache, so
    each refinement forward processes only the ``block_len`` block on top of the
    cached prefix — O(block) per step, not O(sequence). Within a block we
    snapshot the committed-prefix cache, then restore it before each forward
    (KVCache overwrites from ``offset``) and commit the conf>=threshold
    predictions until the block fills. ~1.8x faster than the uncached path and
    coherent at long context, where greedy AR collapses into repetition.
    """
    MASK = model.args.mask_token_id
    V = model.args.vocab_size
    im_end = _im_end_id(tokenizer)
    dt = _mask_dtype(model)
    cache = model.make_cache()
    pe = np.asarray(prompt_ids, dtype=np.int32)
    # block-causal prefill of the prompt (diffusion-mode attention over the prefix)
    model(mx.array(pe)[None], cache=cache,
          mask=build_block_causal_mask(pe.shape[0], block_len, dt))
    seen = list(map(int, pe))
    out = []
    while len(out) < max_new:
        snap = _snap_cache(cache)
        S0 = snap[0][0]
        block = np.full((block_len,), MASK, np.int32)
        # block-causal over GLOBAL positions (matches build_block_causal_mask/training).
        # NOT all-zeros: the appended block straddles two global blocks whenever S0 is
        # not a multiple of block_len, and the earlier half must not see the later half.
        fullmask = _global_block_mask(S0, block_len, block_len, dt)
        for _ in range(block_len):
            mp = np.where(block == MASK)[0]
            if mp.size == 0:
                break
            _restore_cache(cache, snap)
            logits = model(mx.array(block)[None], cache=cache, mask=fullmask)[0]
            lgm = _rep_pen(logits[mx.array(mp)], np.asarray(seen), rep, V)
            probs = mx.softmax(lgm.astype(mx.float32), axis=-1)
            conf = np.asarray(probs.max(axis=-1))
            pred = np.asarray(probs.argmax(axis=-1))
            take = conf >= threshold
            if take.sum() == 0:
                take[conf.argmax()] = True
            block[mp[take]] = pred[take]
        # commit the filled block permanently (cache advances to S0 + block_len)
        _restore_cache(cache, snap)
        model(mx.array(block)[None], cache=cache, mask=fullmask)
        bl = block.tolist()
        seen.extend(bl)
        out.extend(bl)
        if im_end in bl:
            break
    return np.asarray(out[:max_new], np.int32)


def selfspec_ids(model, tokenizer, prompt_ids, *, k=8, block_len=32,
                 max_new=128, rep=REP_DEFAULT):
    """Prefix-cached, two-forward self-speculation. Returns token ids (np array).

    Each round runs exactly **two** forwards: (1) a single block forward that
    drafts ``k`` tokens (seeded at position 0 with the known AR next-token), and
    (2) a causal AR forward that verifies them. We accept the matching prefix;
    the AR-correction token at the first divergence becomes the *next* round's
    draft seed (so it is verified next round rather than committed with a third
    forward). The verify records the full Canon-conv window (model ``record_conv``
    hook), so the committed conv state is rolled back to the accepted position
    (``xp[:, m:m+k-1]``) and the KV offset trimmed to ``n+m`` — no commit forward.

    The AR-verify decides every emitted token, so the output is **identical to
    autoregressive greedy decode**, while ~3-4 tokens commit per round: ~1.4x
    faster than token-by-token AR at the same accuracy on dhara-250m (M3 Max).
    The speedup is largest decoding greedily (the fine-tuned-deployment case); a
    repetition penalty lowers accept length toward AR parity. Single-shot drafting
    is optimal (extra diffusion draft steps cost more forwards than they save).
    ``block_len`` is unused (kept for call compatibility). This is OptiQ's
    ``--mtp`` path for dhara.
    """
    MASK = model.args.mask_token_id
    V = model.args.vocab_size
    im_end = _im_end_id(tokenizer)
    dt = _mask_dtype(model)
    K1 = model.args.canon_kernel - 1
    cache = model.make_cache()
    pe = np.asarray(prompt_ids, dtype=np.int32)
    pl = model(mx.array(pe)[None], cache=cache)[0]            # causal prefill
    seen = list(map(int, pe))
    anchor = int(_rep_pen(pl[-1:], np.asarray(seen), rep, V).argmax(axis=-1)[0])
    out = []
    while len(out) < max_new:
        snap = _snap_cache(cache)
        S0 = snap[0][0]
        block = np.full((k,), MASK, np.int32)
        block[0] = anchor                                    # pos 0 = known AR token
        # same global block-causal grid as training; an all-zeros mask here makes the
        # draft noise -> near-zero accept length -> two forwards/token for nothing.
        fullmask = _global_block_mask(S0, k, block_len, dt)
        _restore_cache(cache, snap)                          # 1) draft k-1 in one forward
        dl = model(mx.array(block)[None], cache=cache, mask=fullmask)[0]
        mp = np.where(block == MASK)[0]
        block[mp] = np.asarray(_rep_pen(dl[mx.array(mp)], np.asarray(seen), rep, V).argmax(axis=-1))
        draft = block
        _restore_cache(cache, snap)                          # 2) verify causally (record conv window)
        for c in cache:
            c.record_conv = True
            c.conv_full = {}
        al = model(mx.array(draft)[None], cache=cache)[0]
        for c in cache:
            c.record_conv = False
        if rep == 1.0:
            ar_pred = np.asarray(al.argmax(axis=-1))
        else:
            # repetition penalty over the true per-position context (seen + draft[:j+1])
            al_np = np.asarray(al.astype(mx.float32))
            ctx = list(seen)
            ar_pred = np.empty(k, np.int32)
            for j in range(k):
                ctx.append(int(draft[j]))
                ar_pred[j] = int(_rep_pen(mx.array(al_np[j:j + 1]), np.asarray(ctx), rep, V).argmax(axis=-1)[0])
        m = 1
        while m < k and draft[m] == ar_pred[m - 1]:
            m += 1
        for c in cache:                                      # commit without a 3rd forward
            c.offset = S0 + m                                #   trim KV to the accepted length
            for name, xpf in c.conv_full.items():            #   roll each conv state to position m
                c.conv[name] = xpf[:, m:m + K1, :]
            c.conv_full = {}
        emitted = list(draft[:m])
        seen.extend(emitted)
        anchor = int(ar_pred[m - 1])                         # AR correction -> next round's draft seed
        out.extend(emitted)
        if im_end in emitted or anchor == im_end:
            break
    return np.asarray(out[:max_new], np.int32)


def _decode_clean(tokenizer, ids, im_end):
    ids = list(ids)
    if im_end in ids:
        ids = ids[:ids.index(im_end)]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    # trim a trailing incomplete sentence (matches the space's _clip)
    t = text.rstrip()
    ends = [m.start() for m in re.finditer(r"[.!?](?=\s|$)", t)]
    if ends and ends[-1] >= 16:
        t = t[:ends[-1] + 1]
    return t.rstrip()
