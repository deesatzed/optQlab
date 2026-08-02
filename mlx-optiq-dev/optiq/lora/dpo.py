"""DPO (Direct Preference Optimization) trainer for OptiQ LoRA adapters.

Loads a quantized OptiQ model, applies LoRA, and trains the adapter on
preference triples ``{prompt, chosen, rejected}`` using the DPO loss.

The reference distribution comes from the *same* model with the LoRA
contribution disabled (``scale=0`` on every LoRA layer). No second model
copy in memory.

Loss (per pair):

    π_logp_c = log P_policy(chosen | prompt)
    π_logp_r = log P_policy(rejected | prompt)
    ref_logp_c = log P_ref(chosen | prompt)    # adapter scale=0
    ref_logp_r = log P_ref(rejected | prompt)  # adapter scale=0
    L = -log σ( β · ((π_logp_c - ref_logp_c) - (π_logp_r - ref_logp_r)) )

Standard DPO from Rafailov et al. 2023.

Data format (``train.jsonl`` / ``valid.jsonl``):

    {"prompt": "...", "chosen": "...", "rejected": "..."}

If ``prompt`` is a list of messages instead of a string, it is rendered
through the tokenizer's chat template.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .config import OptiqLoraConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_lora_layers(model):
    """Yield every LoRA/DoRA-wrapped module in the model.

    We detect by attribute presence rather than isinstance because the
    layer classes live inside mlx-lm and could shift between releases.
    Any module with both a ``scale`` attribute and a LoRA-style weight
    pair is treated as a LoRA layer.
    """
    seen: set[int] = set()
    for _, module in model.named_modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        if hasattr(module, "scale") and (
            hasattr(module, "lora_a") or hasattr(module, "lora_b")
            or hasattr(module, "lora_a_proj") or hasattr(module, "lora_b_proj")
        ):
            yield module


def _set_lora_scale(model, scale: float) -> None:
    """Set ``scale`` on every LoRA layer in the model."""
    for layer in _iter_lora_layers(model):
        layer.scale = scale


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _render_prompt(prompt, tokenizer) -> str:
    """Accept either a string or a list of chat messages."""
    if isinstance(prompt, list):
        try:
            return tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=False,
            )
    return prompt


def _tokenize_pair(prompt: str, response: str, tokenizer,
                   max_length: int) -> tuple[list[int], int]:
    """Tokenize ``prompt + response`` and return (input_ids, prompt_len).

    ``prompt_len`` is the count of tokens belonging to the prompt; the
    response starts at index ``prompt_len`` in ``input_ids``.

    We tokenize the prompt and the full concatenation independently so we
    can locate the boundary unambiguously even for tokenizers that merge
    across the seam. If the concatenated form does not start with the
    prompt tokens, we fall back to tokenizing prompt+response and the
    prompt separately and using ``len(prompt_ids)`` as the boundary —
    which is what most production DPO trainers do anyway.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + response, add_special_tokens=False)
    # Boundary: assume the concatenation respects the prompt's token
    # boundary (true for most tokenizers when no special tokens collide).
    if full_ids[: len(prompt_ids)] == prompt_ids:
        prompt_len = len(prompt_ids)
    else:
        # Fall back: prompt_len = len(prompt_ids); response is whatever's left
        prompt_len = len(prompt_ids)
        # Re-construct full_ids as prompt_ids + response_ids to avoid mismatch
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        full_ids = prompt_ids + response_ids

    # Truncate from the LEFT of the prompt if too long, preserving response.
    if len(full_ids) > max_length:
        excess = len(full_ids) - max_length
        if excess >= prompt_len:
            # Response itself is too long; truncate the response too.
            full_ids = full_ids[-max_length:]
            prompt_len = 0
        else:
            full_ids = full_ids[excess:]
            prompt_len -= excess
    return full_ids, prompt_len


def _make_batch(triples: list[dict], tokenizer, max_length: int):
    """Build one DPO batch from a list of ``{prompt, chosen, rejected}`` dicts.

    Returns a dict with mx.arrays:
      chosen_ids   (B, L_c)
      rejected_ids (B, L_r)
      chosen_resp_mask   (B, L_c)  1 at response positions, 0 elsewhere
      rejected_resp_mask (B, L_r)  1 at response positions, 0 elsewhere

    Both branches are padded to the longest sequence within their kind.
    Mask is 0 at pad positions too.
    """
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        pad_id = 0

    chosen_seqs = []
    rejected_seqs = []
    chosen_resp_masks = []
    rejected_resp_masks = []

    for t in triples:
        if isinstance(t["chosen"], list):
            # MULTI-TURN / agentic DPO: chosen & rejected are full {messages}
            # trajectories (sharing the task prefix). Score over EVERY assistant
            # turn via the same per-token mask used for multi-turn SFT, instead
            # of a single prompt->completion boundary.
            from .multiturn import assistant_mask
            chosen_ids, cm = assistant_mask(t["chosen"], tokenizer)
            rejected_ids, rm = assistant_mask(t["rejected"], tokenizer)
            chosen_ids, cm = chosen_ids[:max_length], cm[:max_length]
            rejected_ids, rm = rejected_ids[:max_length], rm[:max_length]
        else:
            prompt_str = _render_prompt(t["prompt"], tokenizer)
            chosen_ids, prompt_len_c = _tokenize_pair(
                prompt_str, t["chosen"], tokenizer, max_length)
            rejected_ids, prompt_len_r = _tokenize_pair(
                prompt_str, t["rejected"], tokenizer, max_length)
            cm = [0] * prompt_len_c + [1] * (len(chosen_ids) - prompt_len_c)
            rm = [0] * prompt_len_r + [1] * (len(rejected_ids) - prompt_len_r)
        chosen_seqs.append(chosen_ids)
        rejected_seqs.append(rejected_ids)
        chosen_resp_masks.append(cm)
        rejected_resp_masks.append(rm)

    L_c = max(len(s) for s in chosen_seqs)
    L_r = max(len(s) for s in rejected_seqs)

    def pad(seqs, masks, L, pad_id):
        ids_padded = np.full((len(seqs), L), pad_id, dtype=np.int32)
        masks_padded = np.zeros((len(seqs), L), dtype=np.int32)
        for i, (s, m) in enumerate(zip(seqs, masks)):
            ids_padded[i, : len(s)] = s
            masks_padded[i, : len(m)] = m
        return mx.array(ids_padded), mx.array(masks_padded)

    chosen_ids, chosen_resp_mask = pad(
        chosen_seqs, chosen_resp_masks, L_c, pad_id)
    rejected_ids, rejected_resp_mask = pad(
        rejected_seqs, rejected_resp_masks, L_r, pad_id)
    return {
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
        "chosen_resp_mask": chosen_resp_mask,
        "rejected_resp_mask": rejected_resp_mask,
    }


def _seq_logp(logits, input_ids, resp_mask):
    """Per-sequence sum of log-probs at response positions.

    ``logits`` has shape (B, L, V) and predicts the NEXT token at each
    position. The token at position ``t`` predicted by ``logits[:, t, :]``
    is the one at ``input_ids[:, t+1]``. So we score input_ids[1:] using
    logits[:-1], masked by resp_mask[1:].
    """
    log_probs = nn.log_softmax(logits[:, :-1, :], axis=-1)  # (B, L-1, V)
    labels = input_ids[:, 1:]                                # (B, L-1)
    mask = resp_mask[:, 1:].astype(log_probs.dtype)          # (B, L-1)
    # Gather log-prob of each label
    label_lp = mx.take_along_axis(log_probs, labels[..., None], axis=-1)
    label_lp = label_lp.squeeze(-1)                          # (B, L-1)
    return (label_lp * mask).sum(axis=-1)                    # (B,)


# Chunk size (sequence positions per head-matmul tile) for the fused DPO logp.
# Peak extra memory ~ one [LOGP_CHUNK, vocab] tile, independent of sequence
# length -> lets DPO train past the ~2-4k wall where the full [B,L,V] logits
# (materialized 4x per step: policy+ref x chosen+rejected) OOM/deadlock on a
# 24GB Mac. Same idea as the SFT fused CE (multiturn.LOGIT_CHUNK).
LOGP_CHUNK = 256


def _fused_logp_one(head, h, labels, mask):
    """Chunked response-logp for ONE sequence via an explicit custom VJP so the
    full [T, vocab] logits are never materialized -- in the forward OR backward.

    ``h`` [T, H], ``labels`` [T], ``mask`` [T] -> scalar sum of response logps.
    grad wrt hidden: d(logp)/dh_t = mask_t*(onehot(label_t) - softmax(logits_t)) @ W
    (the sign-flip of the fused-CE grad; cotangent is the scalar dL/dlogp)."""
    T = h.shape[0]
    mask = mask.astype(mx.float32)
    wdtype = h.dtype

    @mx.custom_function
    def logp(hh):
        total = mx.array(0.0, dtype=mx.float32)
        for s in range(0, T, LOGP_CHUNK):
            e = min(s + LOGP_CHUNK, T)
            logits = head(hh[s:e], True).astype(mx.float32)          # [c, V] transient
            lsm = nn.log_softmax(logits, axis=-1)
            lp = mx.take_along_axis(lsm, labels[s:e][:, None], axis=1).squeeze(1)
            total = total + (lp * mask[s:e]).sum()
        return total

    @logp.vjp
    def logp_vjp(primals, cotan, output):
        hh = primals                                                # [T, H]
        parts = []
        for s in range(0, T, LOGP_CHUNK):
            e = min(s + LOGP_CHUNK, T)
            logits = head(hh[s:e], True).astype(mx.float32)          # [c, V]
            p = mx.softmax(logits, axis=-1)                         # [c, V]
            w = cotan * mask[s:e]                                    # [c]
            gl = -p * w[:, None]                                     # = -softmax * w
            tgt = labels[s:e][:, None]
            at = mx.take_along_axis(gl, tgt, axis=1) + w[:, None]    # + onehot * w
            gl = mx.put_along_axis(gl, tgt, at, axis=1)
            parts.append(head(gl.astype(wdtype), False))            # [c, H]
        return (mx.concatenate(parts, axis=0),)

    return logp(h)


def _fused_seq_logp(model, input_ids, resp_mask):
    """Memory-bounded equivalent of ``_seq_logp(model(input_ids), ...)`` that
    runs the transformer body ONCE for hidden states, then applies the vocab
    head + log_softmax + label-gather in token chunks -> the full [B,L,V] logit
    tensor is never materialized. Reuses the fused-CE head/chunking accessors."""
    from .multiturn import _text_container, _body, _make_head_matmul
    container = _text_container(model)
    head = _make_head_matmul(container)
    h = _body(model)(input_ids)[:, :-1, :]          # [B, L-1, H] -- H, not V
    labels = input_ids[:, 1:]                        # [B, L-1]
    mask = resp_mask[:, 1:]                          # [B, L-1]
    B = h.shape[0]
    return mx.stack([
        _fused_logp_one(head, h[b], labels[b], mask[b]) for b in range(B)
    ])                                               # [B]


def _logp(model, input_ids, resp_mask, fused):
    """Dispatch: chunked fused logp (``--fused-dpo``) vs the plain full-logits
    path. Both are gradient-equivalent; the fused path bounds peak memory so
    DPO reaches the same long contexts as fused-CE SFT."""
    if fused:
        return _fused_seq_logp(model, input_ids, resp_mask)
    return _seq_logp(model(input_ids), input_ids, resp_mask)


def _compute_reference_logps(model, batch, fused=False) -> tuple:
    """Run the two reference forwards (adapter scale=0) and reduce them
    to detached scalars BEFORE the policy forwards run.

    Critical for memory: MLX is lazy-evaluated, and if we computed the
    reference forwards as part of the same value_and_grad-traced graph
    as the policy forwards, four full forward computations would be
    held in scope simultaneously (4x peak memory). By materializing the
    reference logp values here with ``mx.eval`` and then re-attaching
    them via ``mx.stop_gradient``, the reference activations get freed
    before the policy forwards allocate theirs.

    Caller is responsible for setting the adapter scale to 0.0 before
    invoking and restoring the original scale after.
    """
    ref_logp_c = _logp(
        model, batch["chosen_ids"], batch["chosen_resp_mask"], fused)
    ref_logp_r = _logp(
        model, batch["rejected_ids"], batch["rejected_resp_mask"], fused)
    # Force materialization so the intermediate (B, L, V) logits get
    # garbage-collected before the policy forwards allocate. Without this
    # MLX may defer evaluation and keep the activations alive.
    mx.eval(ref_logp_c, ref_logp_r)
    return mx.stop_gradient(ref_logp_c), mx.stop_gradient(ref_logp_r)


def _dpo_loss_from_logratios(logr_chosen, logr_rejected, beta: float,
                             loss_type: str = "sigmoid",
                             label_smoothing: float = 0.0):
    """The DPO objective, shared by the training loss and the metrics pass.

    ``h`` = (log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log
    ref(rejected)) is the difference of log-ratios.

    - "sigmoid" (Rafailov 2023): -log sigmoid(beta*h). With ``label_smoothing``
      eps > 0 this is cDPO (Mitchell 2023): -( (1-eps)*log sigmoid(beta*h)
      + eps*log sigmoid(-beta*h) ), which FLOORS the loss at
      H(eps) so it cannot collapse to 0 by memorizing separable pairs.
    - "ipo" (Azar 2023): (h - 1/(2*beta))^2 -- regress the margin toward a
      BOUNDED target instead of pushing it to infinity. label_smoothing is
      ignored (IPO is already bounded)."""
    h = logr_chosen - logr_rejected
    if loss_type == "ipo":
        tau = 1.0 / (2.0 * beta)
        return ((h - tau) ** 2).mean()
    adv = beta * h
    if label_smoothing > 0.0:
        return -((1.0 - label_smoothing) * nn.log_sigmoid(adv)
                 + label_smoothing * nn.log_sigmoid(-adv)).mean()
    return -nn.log_sigmoid(adv).mean()


def _policy_dpo_loss(model, batch, beta: float,
                     ref_logp_c, ref_logp_r, fused=False,
                     loss_type="sigmoid", label_smoothing=0.0):
    """Compute the DPO loss using pre-computed reference log-probs.

    Run inside ``nn.value_and_grad``. Only the policy forwards happen
    here; the reference values are detached scalars from
    ``_compute_reference_logps``. Memory-wise: 2 forward passes worth
    of activations held for backward, instead of 4. With ``fused`` the
    policy logp is the chunked custom-VJP path (no [B,L,V] logits).
    ``loss_type`` / ``label_smoothing`` select sigmoid / cDPO / IPO.
    """
    pi_logp_c = _logp(model, batch["chosen_ids"],
                      batch["chosen_resp_mask"], fused)
    pi_logp_r = _logp(model, batch["rejected_ids"],
                      batch["rejected_resp_mask"], fused)

    logr_chosen = pi_logp_c - ref_logp_c
    logr_rejected = pi_logp_r - ref_logp_r
    return _dpo_loss_from_logratios(logr_chosen, logr_rejected, beta,
                                    loss_type, label_smoothing)


def _dpo_metrics(model, batch, beta: float, original_scale: float,
                 fused=False, loss_type="sigmoid", label_smoothing=0.0) -> dict:
    """Compute DPO diagnostic metrics with NO gradients held.

    Used for progress logging + validation passes. Walks the same code
    as the training loss but without ``value_and_grad`` overhead, and
    materializes everything eagerly so memory drops between calls.
    """
    # Reference pass
    _set_lora_scale(model, 0.0)
    ref_logp_c, ref_logp_r = _compute_reference_logps(model, batch, fused)
    _set_lora_scale(model, original_scale)
    # Policy pass (no grad tape needed for metrics)
    pi_logp_c = _logp(model, batch["chosen_ids"],
                      batch["chosen_resp_mask"], fused)
    pi_logp_r = _logp(model, batch["rejected_ids"],
                      batch["rejected_resp_mask"], fused)
    mx.eval(pi_logp_c, pi_logp_r)
    logr_chosen = pi_logp_c - ref_logp_c
    logr_rejected = pi_logp_r - ref_logp_r
    advantage = beta * (logr_chosen - logr_rejected)
    loss = _dpo_loss_from_logratios(logr_chosen, logr_rejected, beta,
                                    loss_type, label_smoothing)
    mx.eval(loss)
    return {
        "loss": loss.item() if hasattr(loss, "item") else float(loss),
        "chosen_reward": (beta * logr_chosen).mean().item(),
        "rejected_reward": (beta * logr_rejected).mean().item(),
        "margin": (beta * (logr_chosen - logr_rejected)).mean().item(),
        "accuracy": (advantage > 0).astype(mx.float32).mean().item(),
    }


# ---------------------------------------------------------------------------
# Trainer entry point
# ---------------------------------------------------------------------------

def train_dpo(
    model_dir: str,
    data_dir: str,
    config: OptiqLoraConfig,
    progress_callback: Callable | None = None,
) -> dict:
    """Run a DPO LoRA fine-tune on an OptiQ model.

    **Data requirement.** Both ``chosen`` and ``rejected`` must be plausible
    completions of the same ``prompt`` under the *base* model's distribution.
    DPO learns from the *relative* log-prob margin between the two; if one
    completion is structurally out-of-distribution for the prompt (e.g.
    ``chosen`` is a paraphrase of a different text than what ``prompt``
    introduces), both reward terms drift in lockstep and the margin signal
    saturates near zero, producing the characteristic "loss=0, both rewards
    drifting to -hundreds" pathology. The trainer prints a one-shot warning
    if the first validation pass shows this signature.

    Args:
        model_dir: Path to an OptiQ-quantized model directory.
        data_dir: Path containing ``train.jsonl`` (and optionally
            ``valid.jsonl``) with one ``{prompt, chosen, rejected}`` per line.
        config: OptiQ LoRA config. Must have ``method='dpo'`` (or the
            CLI sets that before calling). ``dpo_beta`` controls the KL
            penalty strength (default 0.1). ``dpo_learning_rate``
            (default 5e-5), ``dpo_warmup_iters`` (default 10% of iters)
            and ``dpo_lr_schedule`` (default "cosine") control the LR
            curve. LR and iters are resolved here via
            ``config.effective_learning_rate()`` / ``effective_iters()`` so
            a bare ``OptiqLoraConfig(method='dpo')`` gets the DPO LR (5e-5)
            and 1-epoch default, not the SFT ones.
        progress_callback: Optional callable ``cb(step, metrics_dict)``
            invoked every ``steps_per_report`` steps.

    Returns ``{adapter_path, applied_ranks, num_iters}``.
    """
    from mlx_lm.utils import load
    from mlx_lm.tuner.utils import print_trainable_parameters

    from .apply import apply_sensitivity_aware_lora
    from .sensitivity_rank import (
        summarize_rank_distribution, read_per_layer_bits, read_per_layer_kl,
    )
    from .trainer import (
        _patch_out_mx_compile_in_mlx_lm_trainer,
        _write_peft_config,
    )

    _patch_out_mx_compile_in_mlx_lm_trainer()

    print(f"[optiq-dpo] loading model from {model_dir}")
    model, tokenizer = load(
        model_dir, tokenizer_config={"trust_remote_code": True})

    # Apply sensitivity-aware LoRA (same as SFT)
    bits = read_per_layer_bits(model_dir)
    kl = read_per_layer_kl(model_dir) or None
    summary = summarize_rank_distribution(
        config, bits, kl, config.target_modules)
    print(f"[optiq-dpo] rank_scaling={config.rank_scaling}, "
          f"distribution {summary['rank_counts']} "
          f"(total adapted linear targets: {summary['total_adapted']})")
    # Two paths:
    #   - ``mount_adapter`` set (textbook SFT -> DPO continuation): use
    #     ``apply_stacked_lora_for_dpo`` which mounts a *frozen* SFT
    #     LoRA + a *trainable* DPO LoRA on the same layers. Reference
    #     forward sets the trainable scale to 0 -> KL is anchored
    #     against base + SFT (= the SFT model), the standard DPO
    #     reference. The saved adapter is the DPO delta only; it
    #     composes with the SFT adapter at serving time.
    #   - ``mount_adapter`` unset: fresh zero-init LoRA, DPO from
    #     scratch on top of base. Empirically this rarely outperforms
    #     SFT on small models because the preference signal alone
    #     isn't strong enough to reach the SFT distribution; prefer
    #     the mount path when you have an SFT checkpoint.
    mount_path = getattr(config, "mount_adapter", None)
    if mount_path:
        from .apply import apply_stacked_lora_for_dpo
        print(f"[optiq-dpo] mount_adapter: stacking frozen SFT LoRA "
              f"from {mount_path} alongside a trainable DPO LoRA")
        applied_ranks, mount_stats = apply_stacked_lora_for_dpo(
            model, model_dir, config, mount_path)
        print(f"[optiq-dpo] mount_adapter: "
              f"{mount_stats['stacked']} stacked, "
              f"{mount_stats['plain_lora']} plain-LoRA (no SFT cover), "
              f"sft_scale={mount_stats['sft_scale']}")
        if mount_stats.get("skipped_unsupported"):
            print(f"[optiq-dpo] mount_adapter: "
                  f"{mount_stats['skipped_unsupported']} MoE expert "
                  f"projections fell back to plain LoRA (stacked path "
                  f"not yet implemented for MoE)")
    else:
        applied_ranks = apply_sensitivity_aware_lora(
            model, model_dir, config)

    print_trainable_parameters(model)

    # Gradient checkpointing on the transformer block. Mirrors the SFT
    # trainer's behavior. Without this the DPO trainer's 2 policy
    # forward passes hold every layer's activations for backward, which
    # combined with the 2 reference forwards used to peak at 75+ GB on
    # a 1B model at seq=2048. With checkpointing + the reordered
    # _compute_reference_logps + _policy_dpo_loss flow, peak drops to
    # ~SFT-at-same-seq levels (3-5 GB for 1B at seq=2048).
    if config.grad_checkpoint:
        try:
            from mlx_lm.tuner.trainer import grad_checkpoint as _gc
            # Find the first transformer block to patch its class. Same
            # discovery layout as mlx-lm's stock SFT trainer.
            inner = getattr(model, "model", None)
            if inner is not None and hasattr(inner, "layers") and inner.layers:
                _gc(inner.layers[0])
                print(f"[optiq-dpo] gradient checkpointing: enabled on "
                      f"{type(inner.layers[0]).__name__}")
            else:
                lm = getattr(model, "language_model", None)
                inner = getattr(lm, "model", None) if lm is not None else None
                if inner is not None and hasattr(inner, "layers") and inner.layers:
                    _gc(inner.layers[0])
                    print(f"[optiq-dpo] gradient checkpointing: enabled on "
                          f"{type(inner.layers[0]).__name__}")
                else:
                    print("[optiq-dpo] gradient checkpointing: skipped "
                          "(could not locate transformer blocks)")
        except Exception as exc:
            print(f"[optiq-dpo] gradient checkpointing: skipped ({exc})")

    # Load DPO data
    train_path = Path(data_dir) / "train.jsonl"
    valid_path = Path(data_dir) / "valid.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"DPO trainer expects {train_path}. The file should contain one "
            f'{{"prompt": ..., "chosen": ..., "rejected": ...}} per line.')
    train_data = _load_jsonl(train_path)
    valid_data = _load_jsonl(valid_path) if valid_path.exists() else []
    print(f"[optiq-dpo] train={len(train_data)} pairs"
          + (f", valid={len(valid_data)} pairs" if valid_data else ""))

    # Resolve iters: epoch-based (default 1 epoch for DPO) unless the user
    # forced an absolute ``iters``. Keeps a bare OptiqLoraConfig(method="dpo")
    # from over-training the policy into collapse.
    n_iters = config.effective_iters(len(train_data))
    if config.iters is None:
        _ep = config.num_epochs if config.num_epochs is not None else config._DEFAULT_EPOCHS.get("dpo", 1.0)
        print(f"[optiq-dpo] iters resolved to {n_iters} "
              f"({_ep:g} epoch(s) x {len(train_data)} pairs / batch {config.batch_size})")

    # Adapter output dir
    adapter_dir = Path(config.adapter_path)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "optiq_lora_config.json").write_text(
        json.dumps({
            **{k: v for k, v in config.__dict__.items()
               if not k.startswith("_")},
            "applied_ranks": applied_ranks,
            "source_model": model_dir,
        }, indent=2, default=str) + "\n"
    )
    _write_peft_config(adapter_dir, config, applied_ranks, model_dir)

    # Training setup. The LR is a callable schedule rather than a flat
    # float so we can do linear warmup → cosine decay. Without warmup
    # the first 1-3 preference-loss steps move the policy hard enough to
    # wreck the rewards-margin signal for the rest of training (rewards
    # drift unboundedly negative, loss saturates at 0). Cosine decay
    # after warmup matches every modern alignment recipe.
    peak_lr = config.effective_learning_rate()
    if peak_lr > 1e-4:
        # DPO with an SFT-grade LR (>= 2e-4 is the mlx-lm SFT default)
        # collapses the policy within ~100 iters by inflating per-step
        # weight updates faster than beta can constrain them. Surface
        # this loudly so a user who passed --learning-rate by hand sees
        # it before burning a 1+ hour run.
        print(f"[optiq-dpo] WARNING: learning_rate={peak_lr:.0e} is "
              f"unusually high for DPO. Standard recipes use 5e-7 to "
              f"5e-5 (see Rafailov 2023 / TRL DPOTrainer). At this LR "
              f"the policy will likely collapse within ~100 iters. "
              f"Consider --learning-rate 5e-5 or omit the flag to take "
              f"the OptiQ DPO default.")
    warmup_iters = config.resolve_warmup_iters(n_iters)
    total_iters = n_iters
    schedule_shape = getattr(config, "dpo_lr_schedule", "cosine")

    if warmup_iters > 0 or schedule_shape == "cosine":
        # MLX optimizers accept a callable schedule that is invoked with
        # the AdamW step counter (an ``mx.array``) and must return an
        # ``mx.array``. We compose primitive ops only so the result
        # stays a scalar mx.array and threads through the optimizer's
        # ``.astype(gradient.dtype)`` path. Linear warmup 0 -> peak over
        # ``warmup_iters`` steps, then either hold or cosine-decay to
        # 10% of peak over the remaining ``iters - warmup_iters`` steps.
        peak_lr_arr = mx.array(peak_lr, dtype=mx.float32)
        warmup_arr = mx.array(max(warmup_iters, 1), dtype=mx.float32)
        remaining_arr = mx.array(
            max(total_iters - warmup_iters, 1), dtype=mx.float32)
        min_lr_arr = mx.array(0.1 * peak_lr, dtype=mx.float32)
        pi_arr = mx.array(3.141592653589793, dtype=mx.float32)
        warmup_iters_int = warmup_iters
        is_cosine = schedule_shape == "cosine"

        def lr_schedule(step):
            step_f = step.astype(mx.float32)
            # Warmup ramp: (step + 1) / warmup_iters * peak
            warm = peak_lr_arr * (step_f + 1.0) / warmup_arr
            warm = mx.minimum(warm, peak_lr_arr)
            if is_cosine:
                # Cosine from peak -> min_lr after warmup.
                progress = mx.minimum(
                    (step_f - warmup_arr) / remaining_arr,
                    mx.array(1.0, dtype=mx.float32),
                )
                progress = mx.maximum(progress, mx.array(0.0, dtype=mx.float32))
                cos_lr = min_lr_arr + 0.5 * (peak_lr_arr - min_lr_arr) * (
                    1.0 + mx.cos(pi_arr * progress)
                )
            else:
                cos_lr = peak_lr_arr
            # Pick warmup vs post-warmup branch via a boolean mask.
            in_warmup = step_f < warmup_arr
            return mx.where(in_warmup, warm, cos_lr)
        optimizer = optim.AdamW(learning_rate=lr_schedule)
        print(f"[optiq-dpo] lr schedule: peak={peak_lr:.0e}, "
              f"warmup={warmup_iters_int} iters, decay={schedule_shape}")
    else:
        optimizer = optim.AdamW(learning_rate=peak_lr)
        print(f"[optiq-dpo] lr schedule: constant {peak_lr:.0e}, no warmup")

    beta = getattr(config, "dpo_beta", 0.1)
    original_scale = float(config.scale)

    # DPO loss variant: sigmoid (Rafailov) / cDPO (sigmoid + label smoothing,
    # Mitchell) / ipo (Azar). cDPO + IPO both prevent the loss->0 collapse that
    # plain sigmoid DPO hits on small, trivially-separable preference sets.
    _loss_type = getattr(config, "dpo_loss", "sigmoid")
    _label_smoothing = float(getattr(config, "dpo_label_smoothing", 0.0))
    if _loss_type == "ipo":
        print(f"[optiq-dpo] loss: IPO (bounded margin target 1/(2*beta)="
              f"{1.0/(2.0*beta):.2f})")
    elif _label_smoothing > 0.0:
        print(f"[optiq-dpo] loss: cDPO (sigmoid, label_smoothing="
              f"{_label_smoothing:g}) -> loss floored, cannot collapse to 0")
    else:
        print("[optiq-dpo] loss: sigmoid (standard DPO)")

    # Fused (chunked) logp: the context at which the plain full-[B,L,V]-logits
    # path OOMs/deadlocks is VRAM-dependent (~2-4k on a 24GB Mac; higher on
    # bigger machines), so this is an opt-in flag rather than a hardcoded
    # threshold. config.fused_dpo (from --fused-dpo) OR OPTIQ_FUSED_DPO=1.
    _fused = bool(getattr(config, "fused_dpo", False)) or \
        os.environ.get("OPTIQ_FUSED_DPO", "0") == "1"
    if _fused:
        print("[optiq-dpo] fused logp: ENABLED (chunked head, no full [B,L,V] "
              "logits) -> long-context DPO within VRAM")

    # Closure-captured reference log-probs. Each training step computes
    # them outside the value_and_grad call (so the reference forwards'
    # activations get freed before the policy forwards run) and then
    # `loss_fn` reads them as detached constants.
    _ref_logps: dict = {"c": None, "r": None}

    def loss_fn(model, batch):
        return _policy_dpo_loss(
            model, batch, beta, _ref_logps["c"], _ref_logps["r"], _fused,
            _loss_type, _label_smoothing)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Trainable params (LoRA only — set by apply_sensitivity_aware_lora)
    model.train()

    # Sampler — sequential epochs over train_data
    rng = np.random.default_rng(0)
    train_indices = np.arange(len(train_data))

    def _batch_at(step: int):
        # Reshuffle each epoch
        epoch_size = max(len(train_data) // config.batch_size, 1)
        if step % epoch_size == 0:
            rng.shuffle(train_indices)
        offset = (step % epoch_size) * config.batch_size
        idx = train_indices[offset : offset + config.batch_size]
        triples = [train_data[i] for i in idx]
        return _make_batch(triples, tokenizer, config.max_seq_length)

    # Training loop
    t0 = time.time()
    best_val = math.inf
    _collapse_warned = False

    # Optional experiment logging (wandb / swanlab), same registry the SFT
    # trainer uses. DPO's metrics dict is mapped into mlx-lm's info shape.
    _report_cb = None
    if getattr(config, "report_to", None):
        from mlx_lm.tuner.callbacks import SUPPORT_CALLBACK
        _rc = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
        for _name in [x.strip().lower() for x in config.report_to.split(",") if x.strip()]:
            if _name not in SUPPORT_CALLBACK:
                raise ValueError(
                    f"--report-to '{_name}' is not supported; choose from "
                    f"{', '.join(sorted(SUPPORT_CALLBACK))}")
            _report_cb = SUPPORT_CALLBACK[_name](
                project_name=getattr(config, "wandb_project", "optiq-lora"),
                log_dir=str(adapter_dir), config=_rc, wrapped_callback=_report_cb)
            print(f"[optiq-dpo] experiment logging -> {_name}")

    # Optional early stopping on validation loss. Needs a val set to monitor.
    _es_patience = int(getattr(config, "early_stopping_patience", 0) or 0)
    if _es_patience > 0 and not valid_data:
        print("[optiq-dpo] WARNING: early stopping requested but no valid.jsonl; "
              "disabled.")
        _es_patience = 0
    _es_min_delta = float(getattr(config, "early_stopping_min_delta", 0.0) or 0.0)
    # Reuse the unit-tested EarlyStoppingCallback for the stop decision (its
    # patience/min_delta logic is identical for SFT and DPO); it raises
    # EarlyStop, which we catch to break DPO's own loop.
    _es = None
    if _es_patience > 0:
        from .callbacks import EarlyStoppingCallback
        _es = EarlyStoppingCallback(_es_patience, min_delta=_es_min_delta)
    early_stopped = False

    print(f"[optiq-dpo] starting training: iters={n_iters} "
          f"batch_size={config.batch_size} beta={beta} lr={peak_lr:.0e}")
    # Route gated-delta (qwen3_next linear-attn) + full-attention through OptiQ's
    # Metal kernels for the DPO forward/backward passes. Without them the stock
    # O(T) gated-delta autograd, multiplied by DPO's 4 passes (policy+ref ×
    # chosen+rejected) and the stacked SFT+DPO LoRA, blows the Metal 499k
    # buffer-count cap. Same kernels the SFT trainer uses; scoped to compatible
    # shapes (other arches fall through to stock mlx-lm).
    import contextlib as _ctx
    from optiq.ops import enable_flash_attention_training as _eflash
    from optiq.ops.gated_delta_grad import enable_gated_delta_training as _egd
    _kernels = _ctx.ExitStack()
    _kernels.enter_context(_eflash())
    _kernels.enter_context(_egd())
    for step in range(1, n_iters + 1):
        batch = _batch_at(step - 1)

        # 1) Reference forwards FIRST, outside value_and_grad. Scale=0
        # turns off the adapter contribution so we get the base model
        # distribution. mx.eval inside _compute_reference_logps forces
        # the (B,L,V) logits + activations to be released before the
        # policy forwards allocate. Critical for memory.
        _set_lora_scale(model, 0.0)
        _ref_logps["c"], _ref_logps["r"] = _compute_reference_logps(model, batch, _fused)
        _set_lora_scale(model, original_scale)

        # 2) Policy forwards INSIDE value_and_grad. Only 2 forward
        # graphs held in scope (chosen, rejected with adapter active).
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % config.steps_per_report == 0:
            # Recompute metrics for logging (cheap; uses same batch)
            m = _dpo_metrics(model, batch, beta, original_scale, _fused, _loss_type, _label_smoothing)
            elapsed = time.time() - t0
            print(f"[optiq-dpo] step {step}/{n_iters}  "
                  f"loss={m['loss']:.4f}  acc={m['accuracy']:.2f}  "
                  f"margin={m['margin']:+.3f}  "
                  f"chosen_r={m['chosen_reward']:+.3f}  "
                  f"rejected_r={m['rejected_reward']:+.3f}  "
                  f"elapsed={elapsed:.0f}s")
            if progress_callback is not None:
                progress_callback(step, m)
            if _report_cb is not None:
                _report_cb.on_train_loss_report({
                    "iteration": step, "train_loss": float(m["loss"]),
                    "accuracy": float(m["accuracy"]), "margin": float(m["margin"]),
                    "learning_rate": float(optimizer.learning_rate)})

        if valid_data and step % config.steps_per_eval == 0:
            val_metrics_list = []
            for vstart in range(0, len(valid_data), config.batch_size):
                vbatch = _make_batch(
                    valid_data[vstart : vstart + config.batch_size],
                    tokenizer, config.max_seq_length)
                vm = _dpo_metrics(model, vbatch, beta, original_scale, _fused, _loss_type, _label_smoothing)
                val_metrics_list.append(vm)
            val_loss = np.mean([vm["loss"] for vm in val_metrics_list])
            val_acc = np.mean([vm["accuracy"] for vm in val_metrics_list])
            val_margin = float(np.mean([vm["margin"] for vm in val_metrics_list]))
            val_cr = float(np.mean([vm["chosen_reward"] for vm in val_metrics_list]))
            val_rr = float(np.mean([vm["rejected_reward"] for vm in val_metrics_list]))
            print(f"[optiq-dpo] [val] step {step}  loss={val_loss:.4f}  "
                  f"accuracy={val_acc:.2f}  margin={val_margin:+.3f}  "
                  f"chosen_r={val_cr:+.3f}  rejected_r={val_rr:+.3f}")

            # Loss=0 collapse detector. After the first eval, if loss is
            # near zero AND both reward sides are drifting hard negative
            # in lockstep (i.e. the margin signal has saturated), the
            # data pairing is almost certainly wrong: chosen and rejected
            # are not plausible completions of the same prompt under the
            # base distribution. Warn once and point at the docstring.
            if (not _collapse_warned
                    and val_loss < 1e-3
                    and (val_cr < -50.0 or val_rr < -50.0)
                    and abs(val_margin) < 5.0):
                print(
                    "[optiq-dpo] WARNING: val loss ~0 with both rewards "
                    "drifting deeply negative and a near-zero margin. "
                    "This is the classic DPO data-pairing pathology: "
                    "chosen and rejected are unlikely to both be valid "
                    "completions of the same prompt under the base "
                    "model's distribution, so DPO cannot extract a "
                    "preference signal. See the train_dpo docstring "
                    "'Data requirement' section."
                )
                _collapse_warned = True
            if _report_cb is not None:
                _report_cb.on_val_loss_report({
                    "iteration": step, "val_loss": float(val_loss),
                    "val_accuracy": float(val_acc), "val_margin": float(val_margin)})
            # Best-val snapshot uses strict < (any improvement saves); early
            # stopping uses the meaningful-improvement delta via the callback.
            if val_loss < best_val:
                best_val = val_loss
                _save_adapter(model, adapter_dir / "best" / "adapters.safetensors")
                _write_peft_config(adapter_dir / "best", config,
                                   applied_ranks, model_dir)
            if _es is not None:
                from .callbacks import EarlyStop
                try:
                    _es.on_val_loss_report(
                        {"iteration": step, "val_loss": float(val_loss)})
                except EarlyStop:
                    # Report the true minimum (loop's strict-< best_val), not
                    # the callback's min-delta patience anchor.
                    print(f"[optiq-dpo] early stopping at step {step}: no val "
                          f"improvement over {_es_patience} evals "
                          f"(best val_loss={best_val:.4f}).")
                    early_stopped = True
                    break

        if step % config.steps_per_save == 0:
            _save_adapter(model, adapter_dir / "adapters.safetensors")

    _kernels.close()

    # Final adapter: on early stop the best-val snapshot beats the last step,
    # so promote it (load-best-at-end); otherwise save the final model.
    from .callbacks import promote_best_adapter
    if early_stopped and promote_best_adapter(adapter_dir):
        print(f"[optiq-dpo] promoted best adapter (val_loss={best_val:.4f}) -> "
              f"{adapter_dir / 'adapters.safetensors'}")
    else:
        _save_adapter(model, adapter_dir / "adapters.safetensors")
    _write_peft_config(adapter_dir, config, applied_ranks, model_dir)
    print(f"[optiq-dpo] done. adapter at {adapter_dir}")

    return {
        "adapter_path": str(adapter_dir),
        "applied_ranks": applied_ranks,
        "num_iters": n_iters,
        "early_stopped": early_stopped,
    }


def _save_adapter(model, path: Path) -> None:
    """Save only the LoRA-trainable parameters to ``path``."""
    from mlx.utils import tree_flatten
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(path), trainable)
