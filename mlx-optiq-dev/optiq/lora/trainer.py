"""Training loop wrapper that reuses mlx-lm's trainer.

Responsibilities:

  1. Load the OptiQ model (text-generation or VLM-wrapped shape; both supported).
  2. Apply sensitivity-aware LoRA per layer via
     ``apply_sensitivity_aware_lora``.
  3. Hand the adapted model to mlx-lm's ``trainer.train`` with PEFT-style
     adapter saving enabled.
  4. Write ``adapter_config.json`` and ``adapter_model.safetensors`` in the
     PEFT convention so the adapter is portable to Hugging Face and
     loadable via ``mlx_lm.generate --adapter-path`` and OptiQ's future
     hot-swap serving.

Side effects on mlx-lm internals:

We disable the ``@mx.compile`` decorator on mlx-lm's training step function
at import time. On Apple Silicon with Qwen3.5-9B it causes a ~7 GB Metal
command-buffer blowup per unique shape and OOMs training within the first
backward pass. Documented in our project memory; safe to bypass with a
small throughput cost.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .config import OptiqLoraConfig


def _patch_out_mx_compile_in_mlx_lm_trainer() -> None:
    """Idempotently remove the ``@mx.compile`` decorator from mlx-lm's
    trainer step function at runtime.

    We can't just unwrap the decorator because mx.compile wraps at
    module-load time. Instead we rebind the ``train`` function to a
    version that skips the decorator. This is a known workaround — see
    project memory note on the Conjure training runs.
    """
    try:
        from mlx_lm.tuner import trainer as _t
    except ImportError:
        return

    # Sentinel so we don't double-patch.
    if getattr(_t, "_optiq_compile_disabled", False):
        return
    _t._optiq_compile_disabled = True

    # mx.compile is used only on the inner ``step`` function. We can't
    # trivially reach into a closure, but we CAN make ``mx.compile`` a
    # no-op during train() invocations by monkey-patching the whole
    # ``train`` function. To stay minimally invasive we shim mx.compile
    # to be identity only while train() is on the stack.
    _original_train = _t.train
    _original_compile = mx.compile

    def _identity_compile(*args, **kwargs):
        # mx.compile can be called as @mx.compile or mx.compile(fn, ...).
        # Support both forms.
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(fn):
            return fn
        return _decorator

    def _patched_train(*args, **kwargs):
        mx.compile = _identity_compile  # type: ignore[assignment]
        try:
            return _original_train(*args, **kwargs)
        finally:
            mx.compile = _original_compile  # type: ignore[assignment]

    _t.train = _patched_train


# Below this many bytes of logits the full-logits path is not worth avoiding.
# Override with OPTIQ_FUSED_CE_BUDGET_MB.
_FUSED_CE_BUDGET_MB = 512


def _logit_bytes(config: OptiqLoraConfig, vocab: int) -> int:
    """Bytes the full [B, T, vocab] logit tensor occupies in bf16."""
    return config.batch_size * config.max_seq_length * vocab * 2


def _decide_fused_ce(config: OptiqLoraConfig, model) -> bool:
    """Whether SFT should use the fused cut-cross-entropy loss.

    Pure: the answer is passed explicitly to ``multiturn.loss`` rather than
    published on the environment, so repeated calls in one process (the Lab runs
    several jobs) cannot read back an earlier job's decision.

    What decides whether the fused path pays is the size of the logit tensor,
    ``batch x seq x vocab x itemsize``, not the sequence length alone. A
    248k-vocab model at 4k spends 1.89 GiB on logits; a 32k-vocab model at 8k
    spends 0.49 GiB. Gating on sequence length alone gets both of those wrong.

    The fused path is gradient-exact and, measured on Qwen3.5-4B-OptiQ-4bit at
    seq 4096 (M4, batch 1, grad-checkpoint), costs no wall-clock time while
    saving 2.16 GB of peak memory: 14.06 GB -> 11.90 GB, both at 0.017 it/s.

    Precedence: explicit config.fused_ce, then OPTIQ_FUSED_CE, then the estimate.
    """
    from .multiturn import _text_container

    if config.fused_ce is not None:
        return config.fused_ce
    env = os.environ.get("OPTIQ_FUSED_CE")
    if env is not None:
        return env == "1"

    vocab = getattr(getattr(_text_container(model), "args", None), "vocab_size", 0)
    if not vocab:
        return config.max_seq_length > 4096      # no vocab: fall back to seq

    budget = int(os.environ.get("OPTIQ_FUSED_CE_BUDGET_MB", _FUSED_CE_BUDGET_MB))
    nbytes = _logit_bytes(config, vocab)
    on = nbytes > budget * 1024 * 1024
    if on:
        print(f"[optiq-lora] logits are {nbytes / 2**30:.2f} GiB "
              f"(batch {config.batch_size} x seq {config.max_seq_length} x vocab "
              f"{vocab}) -> enabling fused cut-cross-entropy")
    return on


def train_lora(
    model_dir: str,
    data_dir: str,
    config: OptiqLoraConfig,
    progress_callback: Callable | None = None,
) -> dict:
    """Run a sensitivity-aware LoRA fine-tune on an OptiQ model.

    Args:
        model_dir: Path to an OptiQ-quantized model directory.
        data_dir: Path to a directory with ``train.jsonl`` and
            ``valid.jsonl`` in mlx-lm's format.
        config: OptiQ LoRA configuration.
        progress_callback: Optional callback passed through to mlx-lm's
            training loop.

    Returns:
        ``{adapter_path: str, applied_ranks: dict, num_iters: int}``.
    """
    from mlx_lm.utils import load
    from mlx_lm.tuner.datasets import (
        CacheDataset,
        load_dataset as _mlx_load_dataset,
    )
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import (
        build_schedule,
        print_trainable_parameters,
    )
    from mlx_lm.tuner.callbacks import TrainingCallback

    from .apply import apply_sensitivity_aware_lora
    from .sensitivity_rank import summarize_rank_distribution, read_per_layer_bits, read_per_layer_kl

    _patch_out_mx_compile_in_mlx_lm_trainer()

    print(f"[optiq-lora] loading model from {model_dir}")
    model, tokenizer = load(model_dir, tokenizer_config={"trust_remote_code": True})

    # Sensitivity-aware LoRA application
    bits = read_per_layer_bits(model_dir)
    kl = read_per_layer_kl(model_dir) or None
    summary = summarize_rank_distribution(config, bits, kl, config.target_modules)
    print(f"[optiq-lora] rank_scaling={config.rank_scaling}, "
          f"distribution {summary['rank_counts']} "
          f"(total adapted linear targets: {summary['total_adapted']})")
    applied_ranks = apply_sensitivity_aware_lora(model, model_dir, config)
    print_trainable_parameters(model)

    # Load dataset in mlx-lm's expected shape.
    #
    # CRITICAL: ``mask_prompt`` controls whether the loss is computed
    # over the full sequence (prompt + response) or only over the
    # assistant's response. We default to True (mask the prompt out
    # of the loss) because computing CE over the prompt teaches the
    # model to memorize chat-template boilerplate and few-shot demos
    # rather than the actual task; empirically observed to degrade
    # GSM8K accuracy by 35+ points vs the base model when False.
    # This matches Unsloth / PEFT / llama-factory's default behavior.
    #
    # mask_prompt only takes effect when the data is in a format that
    # exposes a prompt/response boundary:
    #   - ChatDataset:        {"messages": [...]} (chat-templated)
    #   - CompletionsDataset: {"prompt": ..., "completion": ...}
    # TextDataset (data with bare {"text": ...} field) cannot mask
    # because there is no boundary to mark. Users who need masking
    # must use one of the above shapes.
    print(f"[optiq-lora] loading dataset from {data_dir}")
    import argparse
    args_ns = argparse.Namespace(data=data_dir, chat_template="default",
                                  text_field="text", train=True, test=False,
                                  test_batches=0, hf_dataset=None,
                                  mask_prompt=config.mask_prompt)
    def _load(ns):
        try:
            return _mlx_load_dataset(ns, tokenizer)
        except TypeError:
            # Older/newer signature; try minimal positional form
            return _mlx_load_dataset(argparse.Namespace(data=ns.data), tokenizer)

    try:
        train_set, valid_set, _test_set = _load(args_ns)
    except ValueError as e:
        # Newer mlx-lm raises (instead of warning) when mask_prompt=True but the
        # data is a bare-'text' dataset (no prompt/response boundary to mask).
        # Our chat/completion data ({"messages": ...} / {"prompt","completion"})
        # is unaffected; this only triggers on bare {"text": ...}. Fall back to
        # full-sequence loss instead of crashing.
        if config.mask_prompt and "masking not supported for text" in str(e).lower():
            print(
                f"[optiq-lora] WARNING: mask_prompt=True but data is a bare-'text' "
                f"dataset (no prompt/response boundary). Falling back to "
                f"mask_prompt=False (loss over the full sequence). Use the chat "
                f'shape ({{"messages": [...]}}) or {{"prompt","completion"}} to mask.'
            )
            args_ns.mask_prompt = False
            train_set, valid_set, _test_set = _load(args_ns)
        else:
            raise

    # Multi-turn / agentic SFT detection. mlx-lm's ChatDataset masks everything
    # before the LAST message (single-turn prompt->response), so for multi-turn
    # agentic trajectories ({"messages": ...} with many assistant turns) it would
    # train ONLY the final turn and mask the real actions (write_file, etc.). When
    # the data is multi-turn we instead build a PER-TOKEN assistant mask and train
    # on EVERY assistant turn (see optiq/lora/multiturn.py).
    fused_ce = _decide_fused_ce(config, model)

    use_multiturn = False
    mt_train = mt_valid = None
    if config.method == "sft" and getattr(config, "train_on_all_turns", "auto") is not False:
        import json as _json
        from .multiturn import MultiTurnDataset, count_multi_turn
        def _read_jsonl(name):
            p = Path(data_dir) / name
            return [_json.loads(l) for l in open(p)] if p.is_file() else []
        raw_train = _read_jsonl("train.jsonl")
        multi, tot = count_multi_turn(raw_train)
        forced = getattr(config, "train_on_all_turns", "auto") is True
        if tot and (forced or multi > 0):
            mt_train = MultiTurnDataset(raw_train, tokenizer, config.max_seq_length)
            mt_valid = MultiTurnDataset(_read_jsonl("valid.jsonl"), tokenizer, config.max_seq_length)
            use_multiturn = len(mt_train) > 0
            print(f"[optiq-lora] multi-turn agentic SFT: {multi}/{tot} sampled examples "
                  f"are multi-turn -> training on ALL assistant turns "
                  f"({len(mt_train)} train / {len(mt_valid)} valid examples, "
                  f"{mt_train.skipped} dropped with no assistant tokens)")

    # Adapter output directory
    adapter_dir = Path(config.adapter_path)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the full OptiQ lora config for reproducibility
    (adapter_dir / "optiq_lora_config.json").write_text(
        json.dumps({
            **{k: v for k, v in config.__dict__.items()
               if not k.startswith("_")},
            "applied_ranks": applied_ranks,
            "source_model": model_dir,
        }, indent=2, default=str) + "\n"
    )

    # Write PEFT/mlx-lm-compatible adapter_config.json NOW (before training)
    # so mid-training best-adapter snapshots can mirror it. We re-write at
    # the end after training via ``_write_peft_config`` so any config that
    # got refined during the run wins.
    _write_peft_config(adapter_dir, config, applied_ranks, model_dir)

    # Auto-derive threshold if user left it at default 0 — same rule as
    # the server-side cleanup hook (10 % of total RAM, floor 1 GB).
    cct = config.clear_cache_threshold
    if cct == 0:
        from optiq.lab.mlx_cleanup import default_threshold_bytes
        cct = default_threshold_bytes()
        print(f"[optiq-lora] clear_cache_threshold auto-set to "
              f"{cct / 1024**3:.2f} GB (10% of total RAM)")

    # Resolve iters (epoch-based unless the user forced an absolute count) and
    # the method-aware LR, so a bare OptiqLoraConfig() lands on good defaults.
    try:
        n_examples = len(mt_train) if use_multiturn else len(train_set)
    except TypeError:
        n_examples = 0
    n_iters = config.effective_iters(n_examples) if n_examples > 0 else (config.iters or 1000)
    eff_lr = config.effective_learning_rate()
    if config.iters is None and n_examples > 0:
        _ep = config.num_epochs if config.num_epochs is not None else config._DEFAULT_EPOCHS.get(config.method, 3.0)
        print(f"[optiq-lora] iters resolved to {n_iters} "
              f"({_ep:g} epochs x {n_examples} examples / batch {config.batch_size})")

    training_args = TrainingArgs(
        batch_size=config.batch_size,
        iters=n_iters,
        val_batches=config.val_batches,
        steps_per_report=config.steps_per_report,
        steps_per_eval=config.steps_per_eval,
        steps_per_save=config.steps_per_save,
        max_seq_length=config.max_seq_length,
        adapter_file=str(adapter_dir / "adapters.safetensors"),
        grad_checkpoint=config.grad_checkpoint,
        grad_accumulation_steps=config.grad_accumulation_steps,
        clear_cache_threshold=cct,
    )

    optimizer = optim.AdamW(learning_rate=eff_lr)

    model.train()

    # NEFTune (optional, SFT only): add uniform noise to token embeddings on
    # the training forward pass. Gated on the embedding's `training` flag, so
    # mlx-lm's model.eval() around validation keeps val loss on clean
    # embeddings. Restored after training via the returned callable.
    neftune_restore = None
    _neft_alpha = float(getattr(config, "neftune_noise_alpha", 0.0) or 0.0)
    if _neft_alpha > 0:
        from .neftune import enable_neftune
        neftune_restore = enable_neftune(model, _neft_alpha)
        if neftune_restore is not None:
            print(f"[optiq-lora] NEFTune enabled: noise_alpha={_neft_alpha} "
                  f"(training forward only)")
        else:
            print("[optiq-lora] WARNING: NEFTune requested but the token "
                  "embedding could not be located; skipped.")

    # The inner callback prints/logs as mlx-lm normally does; we wrap it
    # with ``BestAdapterCallback`` so every new val-loss minimum copies a
    # snapshot of the adapter into ``<adapter_dir>/best/``. That way mlx-lm
    # still overwrites ``<adapter_dir>/adapters.safetensors`` on its normal
    # save cadence (= "last adapter"), but we never lose the best mid-run
    # to a later overfit or noisy step.
    from .callbacks import (
        BestAdapterCallback,
        EarlyStop,
        EarlyStoppingCallback,
        promote_best_adapter,
    )

    inner = TrainingCallback() if progress_callback is None else progress_callback

    # Optional experiment logging (wandb / swanlab). Reuse mlx-lm's own
    # reporting callbacks, but chain them UNDER our progress callback so the
    # CLI log and the Lab live-chart keep receiving every event too. mlx-lm's
    # get_reporting_callbacks wraps an innermost None, which would drop our
    # `inner`; instead we chain the registry callbacks onto `inner` directly.
    if getattr(config, "report_to", None):
        from mlx_lm.tuner.callbacks import SUPPORT_CALLBACK
        report_cfg = {k: v for k, v in config.__dict__.items()
                      if not k.startswith("_")}
        for _name in [x.strip().lower() for x in config.report_to.split(",") if x.strip()]:
            if _name not in SUPPORT_CALLBACK:
                raise ValueError(
                    f"--report-to '{_name}' is not supported; choose from "
                    f"{', '.join(sorted(SUPPORT_CALLBACK))}")
            inner = SUPPORT_CALLBACK[_name](
                project_name=getattr(config, "wandb_project", "optiq-lora"),
                log_dir=str(adapter_dir),
                config=report_cfg,
                wrapped_callback=inner,
            )
            print(f"[optiq-lora] experiment logging -> {_name} "
                  f"(project={getattr(config, 'wandb_project', 'optiq-lora')})")

    # Optional early stopping on validation loss. Requires a val set to
    # monitor; if there's none, warn and leave it off (the callback would
    # never receive a val report to act on).
    patience = int(getattr(config, "early_stopping_patience", 0) or 0)
    has_val = (mt_valid is not None and len(mt_valid) > 0) if use_multiturn \
        else (valid_set is not None)
    if patience > 0 and not has_val:
        print("[optiq-lora] WARNING: --early-stopping-patience set but there is "
              "no validation set (valid.jsonl); early stopping disabled.")
        patience = 0
    if patience > 0:
        inner = EarlyStoppingCallback(
            patience,
            min_delta=float(getattr(config, "early_stopping_min_delta", 0.0) or 0.0),
            inner=inner,
        )
        print(f"[optiq-lora] early stopping enabled: patience={patience} evals, "
              f"min_delta={float(getattr(config, 'early_stopping_min_delta', 0.0) or 0.0)} "
              f"(evals every {config.steps_per_eval} steps)")

    # Exit-watchdog plumbing: mlx-lm's train() does a final top-level
    # `mx.save_safetensors` after the last eval that can deadlock in Metal on
    # long 8k+ runs (frozen process, never exits — had to be pkilled). By then
    # the best/ adapter is already fully saved by BestAdapterCallback, so no
    # work is at risk. The callback sets `_final_eval` once the final eval
    # completes; the watchdog below gives the final save a short grace, then
    # (if it hasn't returned) finalizes best/ -> top-level and force-exits.
    import threading as _threading
    _final_eval = _threading.Event()
    _train_returned = _threading.Event()
    callback = BestAdapterCallback(
        model=model,
        adapter_path=adapter_dir,
        inner=inner,
        final_event=_final_eval,
        total_iters=n_iters,
    )

    def _exit_watchdog():
        import os as _os, shutil as _shutil
        # Wait for the final eval (best/ guaranteed on disk); if it never fires
        # (training died some other way) this thread just idles as a daemon.
        if not _final_eval.wait(timeout=None):
            return
        # mlx-lm's final top-level save normally completes in well under a
        # minute; give it a generous grace, then treat it as deadlocked.
        if _train_returned.wait(timeout=180):
            return  # train() returned cleanly — no hang, nothing to do.
        best_w = adapter_dir / "best" / "adapters.safetensors"
        top_w = adapter_dir / "adapters.safetensors"
        try:
            if best_w.exists() and (not top_w.exists()
                                    or top_w.stat().st_size == 0):
                _shutil.copy2(best_w, top_w)
        except Exception:
            pass
        print("[optiq-lora] final top-level adapter save appears deadlocked "
              "(known Metal hang on long 8k+ runs); the best adapter is fully "
              "saved under best/ — exiting cleanly.", flush=True)
        _os._exit(0)

    _threading.Thread(target=_exit_watchdog, daemon=True).start()

    # mlx-lm's training loop expects CacheDataset wrappers so it can reuse
    # its iterate_batches fast path (itemlen / cached tokenization). For the
    # multi-turn path we pass our own (tokens, mask) datasets + loss/batcher.
    if use_multiturn:
        train_wrapped, valid_wrapped = mt_train, mt_valid
    else:
        train_wrapped = CacheDataset(train_set)
        valid_wrapped = CacheDataset(valid_set) if valid_set is not None else None

    # Route attention through our Metal flash kernel during training when
    # the input shape is compatible (fp16, head_dim=128, no KV cache). For
    # every other shape the patch falls through to stock mlx-lm. This is
    # what unlocks long-context LoRA on Apple Silicon — without it, autograd
    # through the fused SDPA materializes the [B, Hq, T, T] score tensor.
    # It routes by memory budget: stock SDPA is 14-137x faster and is used
    # whenever its backward fits, so the kernel only fires on the shapes that
    # actually need it (high head count at long context).
    from optiq.ops import enable_flash_attention_training
    from optiq.ops.gated_delta_grad import enable_gated_delta_training

    # enable_flash_attention_training: memory-efficient differentiable SDPA for
    #   the full-attention layers, taken only when stock SDPA's score tensor
    #   would not fit the budget. enable_gated_delta_training: fast O(√T)-memory
    #   Metal backward for the GatedDeltaNet (linear-attention) layers of
    #   qwen3_next AND qwen3_5, replacing an autograd path that has no vjp at
    #   all. Both scoped to compatible shapes; every other arch falls through
    #   to stock MLX.
    train_kwargs = dict(
        model=model, optimizer=optimizer, train_dataset=train_wrapped,
        val_dataset=valid_wrapped, args=training_args, training_callback=callback)
    if use_multiturn:
        from . import multiturn as _mt
        # mlx-lm calls loss(model, *batch); bind the decision rather than letting
        # multiturn.loss re-read it from the environment.
        train_kwargs["loss"] = functools.partial(_mt.loss, fused=fused_ce)
        train_kwargs["iterate_batches"] = _mt.iterate_batches

    early_stopped = False
    try:
        with enable_flash_attention_training(), enable_gated_delta_training():
            train(**train_kwargs)
    except EarlyStop as es:
        early_stopped = True
        # Report the true best (from BestAdapterCallback), not the callback's
        # min-delta patience anchor which can read higher.
        print(f"[optiq-lora] early stopping at iter {es.iteration}: no val "
              f"improvement over the last {es.patience} evals "
              f"(best val_loss={callback.best_val:.4f}).", flush=True)
        # load-best-at-end: the trailing evals that tripped patience are worse
        # than the best, so promote the best snapshot to the returned adapter.
        if promote_best_adapter(adapter_dir):
            print(f"[optiq-lora] promoted best adapter "
                  f"(val_loss={callback.best_val:.4f}) -> "
                  f"{adapter_dir / 'adapters.safetensors'}", flush=True)
    finally:
        _train_returned.set()  # train() done (or stopped); stand the watchdog down.
        if neftune_restore is not None:
            neftune_restore()  # remove the embedding noise wrapper

    # Also emit a PEFT-style ``adapter_config.json`` next to the weights
    # so the adapter is loadable by PEFT / huggingface_hub consumers.
    _write_peft_config(adapter_dir, config, applied_ranks, model_dir)

    return {
        "adapter_path": str(adapter_dir),
        "applied_ranks": applied_ranks,
        "num_iters": n_iters,
        "early_stopped": early_stopped,
    }


def _write_peft_config(adapter_dir: Path, config: OptiqLoraConfig,
                      applied_ranks: dict, base_model: str) -> None:
    """Write adapter_config.json that's loadable by BOTH mlx-lm and PEFT.

    mlx-lm's ``load_adapters`` reads the config via ``SimpleNamespace`` and
    requires:
      * ``fine_tune_type``  — "lora" | "dora" | "full"
      * ``num_layers``      — how many blocks to adapt
      * ``lora_parameters`` — dict with rank / scale / dropout / keys

    HuggingFace PEFT reads:
      * ``base_model_name_or_path``, ``peft_type``, ``r``, ``lora_alpha``,
        ``lora_dropout``, ``target_modules``, ``task_type``, etc.

    We write keys that satisfy both. OptiQ-specific data lives under
    ``optiq`` for tooling like ``optiq lora info``.
    """
    cfg = {
        # ---------- mlx-lm-required keys ----------
        "fine_tune_type": "dora" if config.use_dora else "lora",
        "num_layers": config.num_layers,
        "lora_parameters": {
            "rank": config.rank,
            "scale": config.scale,
            "dropout": float(config.dropout),
            "keys": None,
        },
        # ---------- PEFT-compatible keys ----------
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": False,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "lora_alpha": int(config.rank * config.scale),
        "lora_dropout": float(config.dropout),
        "modules_to_save": None,
        "peft_type": "LORA" if not config.use_dora else "DORA",
        "r": config.rank,
        "revision": None,
        "target_modules": list(config.target_modules),
        "task_type": "CAUSAL_LM",
        # ---------- OptiQ extensions ----------
        "optiq": {
            "rank_scaling": config.rank_scaling,
            "applied_ranks": applied_ranks,
        },
    }
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(cfg, indent=2) + "\n"
    )
