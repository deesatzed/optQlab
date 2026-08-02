"""OptiQ CLI: convert, benchmark, and evaluate models."""

import os
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version

import click

try:
    _OPTIQ_VERSION = _pkg_version("mlx-optiq")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    _OPTIQ_VERSION = "0.0.0+unknown"


@click.group()
@click.version_option(version=_OPTIQ_VERSION, prog_name="mlx-optiq")
def cli():
    """mlx-optiq: optimizing compiler for MLX with data-driven mixed-precision quantization."""
    pass


class _DefaultGroup(click.Group):
    """A group whose bare / non-subcommand invocation falls through to a default
    subcommand, so `optiq code [PATH] [flags]` launches while `optiq code export`
    still dispatches as a subcommand."""

    def __init__(self, *a, **k):
        self.default_cmd = k.pop("default", None)
        self.default_if_no_args = k.pop("default_if_no_args", False)
        super().__init__(*a, **k)

    def parse_args(self, ctx, args):
        if self.default_cmd:
            if not args and self.default_if_no_args:
                # Bare `optiq code` -> the default command (interactive launch).
                args = [self.default_cmd]
            elif args and args[0] not in self.commands and args[0] not in ("-h", "--help"):
                # First token is neither a subcommand nor a request for the group
                # help, so it belongs to the default command: a flag (`optiq code
                # -p GOAL`, `-c`, `--model X`) or a PATH (`optiq code ./repo`).
                # Prepend the command name so Click parses the flags in its
                # context. This is what makes `optiq code -p` work like
                # `claude -p`, rather than dying on "No such option".
                args = [self.default_cmd, *args]
        return super().parse_args(ctx, args)

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            if self.default_cmd is None:
                raise
            return super().resolve_command(ctx, [self.default_cmd, *args])


@cli.group(cls=_DefaultGroup, name="code", default="launch", default_if_no_args=True)
def code():
    """OptiQ Code — terminal coding agent for your locally-served model.

    Zero-config: it uses whatever model OptiQ is serving. `optiq code` (or
    `optiq code PATH`) launches in a repo; `optiq code export` writes a session
    transcript. Run `optiq code launch --help` for the launch options.
    """
    pass


@code.command(name="launch")
@click.argument("path", required=False, default=".")
@click.option("-c", "--continue", "continue_", is_flag=True,
              help="Resume the most recent session in this repo.")
@click.option("-r", "--resume", default=None, is_flag=False, flag_value="",
              help="Resume a session by id (or most recent if given no id).")
@click.option("-p", "--print", "print_goal", default=None,
              help="Headless: run non-interactively toward GOAL, print result, set exit code.")
@click.option("--model", default=None,
              help="(advanced) Override which served model to use.")
@click.option("--no-color", is_flag=True, help="Disable ANSI color in the banner/output.")
@click.option("-y", "--yes", is_flag=True,
              help="Auto-approve every tool call (no per-edit approval prompt).")
def code_launch(path, continue_, resume, print_goal, model, no_color, yes):
    """Launch OptiQ Code in PATH (default: current directory)."""
    from optiq.code.cli import run_code
    # -r with an id -> that id; -r alone ("") or -c -> most recent (None).
    do_resume = continue_ or (resume is not None)
    resume_id = resume if (resume not in (None, "")) else None
    raise SystemExit(run_code(
        path=path, do_resume=do_resume, resume_id=resume_id,
        print_goal=print_goal, model=model, no_color=no_color, auto_approve=yes,
    ))


@code.command(name="config")
@click.argument("path", required=False, default=".")
def code_config(path):
    """Show every setting, its value, and where that value came from."""
    from optiq.code.cli import run_config
    raise SystemExit(run_config(path=path))


@code.command(name="export")
@click.option("--session", default=None, help="Session id to export (default: most recent).")
@click.option("-o", "--output", default=None, help="Output file (default: stdout).")
def code_export(session, output):
    """Export a session as HF Session-Traces JSONL."""
    from optiq.code.cli import run_export
    raise SystemExit(run_export(session=session, output=output))


@cli.group()
def lora():
    """Sensitivity-aware LoRA fine-tuning on OptiQ-quantized models.

    Standard LoRA uses uniform rank across every adapted layer. OptiQ
    knows which layers were sensitive (higher bits) vs robust (lower
    bits) from quantization-time KL measurements, and this subcommand
    uses that signal to assign per-layer rank.
    """
    pass


@lora.command("train")
@click.argument("model_dir")
@click.option("--data", "data_dir", required=True,
              help="Directory with train.jsonl and valid.jsonl (mlx-lm format)")
@click.option("--output", "-o", "adapter_path", default="optiq_lora_adapters",
              help="Where to save the adapter (default: ./optiq_lora_adapters)")
@click.option("--preset", default=None,
              type=click.Choice(["small", "default", "medium", "large", "xl", "xxl"]),
              help="Convenience bundle that sets --rank and --scale together. "
                   "small=(r=8,a=16), default=(r=8,a=8), medium=(r=16,a=16), "
                   "large=(r=32,a=32), xl=(r=64,a=64), xxl=(r=128,a=128). "
                   "Explicit --rank/--scale still override individual fields.")
@click.option("--rank", default=8, type=int,
              help="Base LoRA rank (default: 8). With --rank-scaling=by_bits "
                   "(the default), sensitive layers bump above this base.")
@click.option("--scale", default=1.0, type=float,
              help="LoRA alpha scaling (alpha = rank * scale). Default: 1.0 "
                   "(alpha = rank, matching Unsloth's recommendation).")
@click.option("--dropout", default=0.0, type=float, help="LoRA dropout")
@click.option("--rank-scaling", default="by_bits",
              type=click.Choice(["constant", "by_bits", "by_kl"]),
              help=("How to scale rank per layer. by_bits: layers OptiQ "
                    "kept at higher bits get proportionally higher rank. "
                    "by_kl: scale by raw KL sensitivity. constant: uniform "
                    "(pure Unsloth-style baseline)."))
@click.option("--num-layers", default=-1, type=int,
              help="Number of transformer blocks (from the last) to adapt. "
                   "-1 = all layers (default; matches Unsloth/PEFT and lets "
                   "the by_bits overlay see every layer's bit assignment). "
                   "Drop to a smaller N on 9B+ models if you hit the 499k "
                   "MTLResource cap on full-depth backward.")
@click.option("--target-modules",
              default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
              help="Comma-separated list of module name suffixes to adapt. "
                   "Default matches Unsloth: all 7 trainable linears per block.")
@click.option("--adapt-experts", is_flag=True,
              help="Also adapt MoE expert pools. Each expert gets its own "
                   "LoRA, so the adapter grows by the expert count "
                   "(256 experts -> ~128x). Off by default; attention "
                   "projections are always adapted.")
@click.option("--use-dora", is_flag=True, help="Use DoRA instead of LoRA")
@click.option("--method", default="sft", type=click.Choice(["sft", "dpo"]),
              help="Training objective. 'sft' = standard supervised "
                   "fine-tuning (cross-entropy on response tokens). "
                   "'dpo' = Direct Preference Optimization on "
                   "{prompt, chosen, rejected} triples; uses the same "
                   "adapted model with adapter scale temporarily zeroed "
                   "for the reference forward pass.")
@click.option("--dpo-beta", default=0.1, type=float,
              help="DPO KL constraint strength. Only used with --method=dpo. "
                   "Standard default 0.1 (Rafailov et al. 2023). Higher = "
                   "stronger pull toward reference; lower = more aggressive "
                   "preference learning.")
@click.option("--mask-prompt/--no-mask-prompt", default=True,
              help="Mask prompt tokens from the cross-entropy loss so only "
                   "the assistant's response contributes. Default: True. "
                   "Only used with --method=sft. Requires data in "
                   "{'messages': [...]} or {'prompt': ..., 'completion': ...} "
                   "format; bare {'text': ...} cannot mask. Without masking, "
                   "the model memorizes prompt boilerplate and degrades "
                   "base-model task quality.")
@click.option("--all-turns", type=click.Choice(["auto", "on", "off"]), default="auto",
              help="Multi-turn / agentic SFT: train on EVERY assistant turn, not "
                   "just the last. mlx-lm's ChatDataset masks all-but-last (single-"
                   "turn), which for multi-turn {'messages':[...]} trajectories "
                   "trains only the final turn and masks the real actions. 'auto' "
                   "(default) detects multi-turn data and switches on; 'on' forces; "
                   "'off' keeps single-turn masking. SFT only.")
@click.option("--batch-size", default=1, type=int)
@click.option("--iters", default=None, type=int,
              help="Absolute training iterations. Omit to train by epoch: "
                   "--num-epochs x ceil(examples/batch), defaulting to 3 "
                   "epochs for SFT and 1 for DPO.")
@click.option("--num-epochs", "num_epochs", default=None, type=float,
              help="Epochs over the dataset (overrides the per-method "
                   "default). Ignored if --iters is given.")
@click.option("--learning-rate", "--lr", default=None, type=float,
              help=("AdamW learning rate. Method-aware default if omitted: "
                    "2e-4 for --method=sft (matches Unsloth), 5e-5 for "
                    "--method=dpo (matches HF TRL DPOTrainer / Rafailov 2023). "
                    "Pass explicitly to override either default. Using the "
                    "SFT default for a DPO run is the most common reason DPO "
                    "training collapses to loss=0 with rewards drifting "
                    "unboundedly negative — 2e-4 is 4-40× too aggressive "
                    "for preference tuning."))
@click.option("--dpo-warmup-iters", default=None, type=int,
              help=("Linear-warmup iterations for DPO. Only used with "
                    "--method=dpo. Omit for the method-aware default: "
                    "max(10, iters // 10). Without warmup the first 1-3 "
                    "preference-loss steps move the policy hard enough to "
                    "wreck the rewards-margin signal for the rest of "
                    "training, regardless of --dpo-beta."))
@click.option("--dpo-lr-schedule", default="cosine",
              type=click.Choice(["constant", "cosine"]),
              help=("LR schedule shape after warmup for --method=dpo. "
                    "'cosine' anneals to 10 %% of the peak over remaining "
                    "steps (matches every modern alignment recipe). "
                    "'constant' holds peak — use only if you're explicitly "
                    "iterating on the warmup length."))
@click.option("--mount-adapter", "mount_adapter", default=None, type=str,
              help=("SFT -> DPO continuation: mount the supplied "
                    "adapter as a *frozen* LoRA alongside a fresh "
                    "*trainable* LoRA on the same layers. The DPO "
                    "reference forward zeroes only the trainable "
                    "scale, so KL is anchored against base + SFT "
                    "(= the SFT model), the standard DPO reference. "
                    "The saved adapter contains only the DPO delta "
                    "and composes with the SFT adapter at serving "
                    "time. Only --method=dpo."))
@click.option("--dpo-loss", "dpo_loss", default="sigmoid",
              type=click.Choice(["sigmoid", "ipo"]),
              help=("DPO loss variant. 'sigmoid' = standard DPO. 'ipo' "
                    "(Azar 2023) regresses the margin toward a bounded target "
                    "1/(2*beta) instead of pushing it to infinity — use for "
                    "small preference sets that otherwise collapse. Only "
                    "--method=dpo."))
@click.option("--dpo-label-smoothing", "dpo_label_smoothing", default=0.0,
              type=float,
              help=("cDPO label smoothing epsilon (Mitchell 2023). Floors the "
                    "sigmoid loss so it can't collapse to 0 by memorizing "
                    "trivially-separable pairs. 0.0 = plain DPO; try 0.1–0.3 "
                    "for small/noisy data. Ignored with --dpo-loss ipo."))
@click.option("--fused-dpo", "fused_dpo", is_flag=True, default=False,
              help=("Chunked DPO logp: run the body once, then apply the "
                    "vocab head + log_softmax in token chunks so the full "
                    "[B,L,V] logits are never materialized (built 4x/step "
                    "otherwise). Lets DPO reach long contexts (8k+) that "
                    "otherwise OOM. Enable when you hit the memory wall — the "
                    "break point is VRAM-dependent (~2-4k on a 24GB Mac). "
                    "Only --method=dpo; the SFT equivalent is --fused-ce."))
@click.option("--fused-ce/--no-fused-ce", "fused_ce", default=None,
              help=("Fused cut-cross-entropy for SFT: compute the LM-head loss "
                    "in token chunks so the full [B,T,vocab] logit tensor is "
                    "never materialized. Gradient-exact. Default: auto, enabled "
                    "when the logit tensor (batch x seq x vocab x 2 bytes) "
                    "exceeds OPTIQ_FUSED_CE_BUDGET_MB (512). Measured on "
                    "Qwen3.5-4B at seq 4096: 14.06 -> 11.90 GB peak, same "
                    "it/s. Only --method=sft."))
@click.option("--max-seq-length", default=512, type=int,
              help="Max sequence length (default 512). Covers ~99.8% of "
                   "typical SFT data (GSM8K measured: max 515 tokens). "
                   "Bump to 1024+ for longer-context tasks; you may need "
                   "to drop --num-layers or --batch-size accordingly.")
@click.option("--grad-accumulation-steps", default=1, type=int)
@click.option("--grad-checkpoint/--no-grad-checkpoint", default=True)
@click.option("--val-batches", default=25, type=int)
@click.option("--steps-per-report", default=10, type=int)
@click.option("--steps-per-eval", default=200, type=int)
@click.option("--steps-per-save", default=100, type=int)
@click.option("--early-stopping-patience", "early_stopping_patience", default=0,
              type=int,
              help="Stop once val loss hasn't improved for N consecutive "
                   "evaluations (evals fire every --steps-per-eval). 0 = off "
                   "(default). Requires a validation set (valid.jsonl). On stop "
                   "the best-val adapter is promoted to the returned adapter.")
@click.option("--early-stopping-min-delta", "early_stopping_min_delta",
              default=0.0, type=float,
              help="Minimum val-loss improvement that counts as progress for "
                   "--early-stopping-patience. Default 0.0.")
@click.option("--neftune-noise-alpha", "neftune_noise_alpha", default=0.0,
              type=float,
              help="NEFTune (Jain et al. 2023): add uniform noise scaled by "
                   "alpha/sqrt(seq*dim) to token embeddings during the SFT "
                   "forward pass only (nothing at inference). 0 = off (default, "
                   "matching TRL/Unsloth); the paper suggests 5-15. A cheap "
                   "regularizer that improves instruction-following on small "
                   "datasets. SFT only.")
@click.option("--report-to", "report_to", default=None,
              help="Log training metrics to an experiment tracker: 'wandb' "
                   "(needs `pip install wandb` + WANDB_API_KEY) or 'swanlab'. "
                   "Comma-separated for several. Off by default. Runs alongside "
                   "the normal CLI log / Lab live chart.")
@click.option("--wandb-project", "wandb_project", default="optiq-lora",
              help="W&B project name when --report-to includes wandb.")
@click.option("--vision", is_flag=True, default=False,
              help="Image+text LoRA on a VLM quant (one that ships an "
                   "optiq_vision sidecar). --data is an image+text jsonl "
                   "({image, prompt, completion} or {messages}). The vision "
                   "tower stays frozen; LoRA trains the language tower. "
                   "Auto-enabled when the model dir has a vision sidecar.")
def lora_train(model_dir, data_dir, adapter_path, preset, rank, scale, dropout,
               rank_scaling, num_layers, target_modules, adapt_experts, use_dora, method,
               dpo_beta, dpo_warmup_iters, dpo_lr_schedule, mount_adapter,
               dpo_loss, dpo_label_smoothing, fused_dpo, fused_ce,
               mask_prompt, all_turns,
               batch_size, iters, num_epochs, learning_rate, max_seq_length,
               grad_accumulation_steps, grad_checkpoint, val_batches,
               steps_per_report, steps_per_eval, steps_per_save,
               early_stopping_patience, early_stopping_min_delta,
               neftune_noise_alpha, report_to, wandb_project, vision):
    """Train a LoRA adapter on an OptiQ-quantized model.

    Examples:

        # Supervised fine-tuning (default)
        optiq lora train ./optiq_output/Qwen3.5-9B-OptiQ-4bit \\
            --data ./my_training_data \\
            --preset medium --iters 2000

        # DPO from preference pairs (data is {prompt, chosen, rejected})
        optiq lora train mlx-community/Qwen3.5-0.8B-OptiQ-4bit \\
            --data ./dpo_pairs --method dpo --dpo-beta 0.1 \\
            --preset small --iters 200
    """
    from .lora.config import OptiqLoraConfig, RANK_PRESETS

    # Image+text LoRA on a VLM quant. Auto-enable when the model ships an
    # optiq_vision sidecar, so users don't have to remember --vision.
    #
    # This used to read `os.path.isdir(p) and _sidecar_exists(p, ...)`. It is the
    # SAME gate that silently disabled vision in the engine, `optiq serve` and the
    # Lab, and it survived the fix because the fix visited three call sites and
    # this was the fourth. A Hub repo id is not a directory, so
    # `optiq lora train mlx-community/Qwen3.5-9B-OptiQ-4bit` fell through to the
    # TEXT path -- with the text LR (2e-4, which mode-collapses a vision LoRA)
    # and no images -- while the same model in a local dir trained correctly.
    def _is_vlm_dir(p: str) -> bool:
        from .sidecar_layout import exists as _sidecar_exists, local_model_dir
        d = local_model_dir(p)
        return d is not None and _sidecar_exists(d, "optiq_vision.safetensors")

    if vision or _is_vlm_dir(model_dir):
        # 5e-5 (not the 2e-4 text-SFT default): vision LoRA on short targets
        # mode-collapses at 2e-4. Gradient clipping (in train_vlm_lora) helps too.
        eff_lr = learning_rate if learning_rate is not None else 5e-5
        # Qwen3.5/3.6 hybrid (gated-delta) family collapses at the AR default
        # scale of 20; ~8 is stable. See optiq.vlm.lora.train_vlm_lora.
        eff_scale = scale if scale != 1.0 else 8.0
        click.echo(f"[optiq.lora] vision (image+text) LoRA: frozen vision "
                   f"tower, training language-tower LoRA")
        from .vlm.lora import train_vlm_lora
        out = train_vlm_lora(
            model_dir, data_dir, adapter_path,
            rank=rank, scale=eff_scale, dropout=dropout,
            num_layers=num_layers, iters=iters, learning_rate=eff_lr,
            max_target=max_seq_length, report_every=steps_per_report,
            grad_checkpoint=grad_checkpoint,
        )
        click.echo(f"\nDone. Adapter saved to: {out}")
        return

    if preset is not None:
        preset_r, preset_s = RANK_PRESETS[preset]
        if rank == 8:
            rank = preset_r
        if scale == 1.0:
            scale = preset_s

    # ``--learning-rate`` / ``--iters`` default to None -> the config resolves
    # method-aware defaults (SFT 2e-4 / DPO 5e-5; SFT 3 epochs / DPO 1) for
    # BOTH the CLI and direct construction. An explicit flag always wins.
    config = OptiqLoraConfig(
        rank=rank,
        scale=scale,
        dropout=dropout,
        rank_scaling=rank_scaling,
        target_modules=tuple(s.strip() for s in target_modules.split(",") if s.strip()),
        adapt_experts=adapt_experts,
        num_layers=num_layers,
        use_dora=use_dora,
        method=method,
        dpo_beta=dpo_beta,
        dpo_warmup_iters=dpo_warmup_iters,
        dpo_lr_schedule=dpo_lr_schedule,
        mount_adapter=mount_adapter,
        dpo_loss=dpo_loss,
        dpo_label_smoothing=dpo_label_smoothing,
        fused_dpo=fused_dpo,
        fused_ce=fused_ce,
        mask_prompt=mask_prompt,
        train_on_all_turns={"auto": "auto", "on": True, "off": False}[all_turns],
        batch_size=batch_size,
        iters=iters,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        grad_accumulation_steps=grad_accumulation_steps,
        grad_checkpoint=grad_checkpoint,
        val_batches=val_batches,
        steps_per_report=steps_per_report,
        steps_per_eval=steps_per_eval,
        steps_per_save=steps_per_save,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        neftune_noise_alpha=neftune_noise_alpha,
        report_to=report_to,
        wandb_project=wandb_project,
        adapter_path=adapter_path,
    )

    if method == "dpo" and learning_rate is None:
        click.echo(f"[optiq.lora] --method=dpo: using DPO learning rate "
                   f"{config.effective_learning_rate():.0e} "
                   f"(override with --learning-rate)")
    if method == "dpo":
        from .lora.dpo import train_dpo
        result = train_dpo(model_dir=model_dir, data_dir=data_dir, config=config)
    else:
        from .lora.trainer import train_lora
        result = train_lora(model_dir=model_dir, data_dir=data_dir, config=config)
    click.echo(f"\nDone. Adapter saved to: {result['adapter_path']}")


@lora.command("info")
@click.argument("adapter_dir")
def lora_info(adapter_dir):
    """Summarize an OptiQ LoRA adapter (rank distribution, base model).

    Examples:

        optiq lora info ./my_adapter
    """
    import json
    from pathlib import Path

    adir = Path(adapter_dir)
    if not adir.exists():
        click.echo(f"adapter dir not found: {adir}", err=True)
        sys.exit(1)

    # Prefer OptiQ sidecar; fall back to PEFT adapter_config.
    optiq_path = adir / "optiq_lora_config.json"
    peft_path = adir / "adapter_config.json"

    if optiq_path.exists():
        cfg = json.loads(optiq_path.read_text())
        click.echo(f"OptiQ LoRA adapter")
        click.echo(f"  base: {cfg.get('source_model')}")
        click.echo(f"  rank: {cfg.get('rank')} (scaling: {cfg.get('rank_scaling')})")
        click.echo(f"  scale: {cfg.get('scale')}  dropout: {cfg.get('dropout')}")
        applied = cfg.get("applied_ranks") or {}
        if applied:
            from collections import Counter
            counts = Counter(applied.values())
            click.echo(f"  rank distribution: "
                       f"{dict(sorted(counts.items()))} "
                       f"across {len(applied)} adapted modules")
    elif peft_path.exists():
        cfg = json.loads(peft_path.read_text())
        click.echo(f"PEFT LoRA adapter (no OptiQ sidecar)")
        click.echo(f"  base: {cfg.get('base_model_name_or_path')}")
        click.echo(f"  rank: {cfg.get('r')}  alpha: {cfg.get('lora_alpha')}")
        click.echo(f"  target: {cfg.get('target_modules')}")
    else:
        click.echo(f"no adapter config found in {adir}", err=True)
        sys.exit(1)


@lora.command("merge")
@click.argument("adapters", nargs=-1, required=True)
@click.option("--output", "-o", required=True,
              help="Where to write the merged adapter directory.")
@click.option("--scales", default=None,
              help=("Optional comma-separated per-source scale "
                    "overrides (e.g. '1.0,0.8'). If omitted, each "
                    "source's scale is read from its sibling "
                    "adapter_config.json (defaults to 1.0). The "
                    "merged adapter writes scale=1.0 because the "
                    "per-source scales are folded into the B "
                    "matrices."))
@click.option("--force", is_flag=True,
              help="Overwrite output dir if it exists.")
def lora_merge(adapters, output, scales, force):
    """Rank-concat merge of N LoRA adapters into a single adapter.

    The merge is mathematically exact (not a low-rank approximation):
    for sources with ranks r1, r2, ... on a layer, the merged adapter
    on that layer has rank r1+r2+..., and its forward pass reproduces
    the sum of the original LoRA residuals.

    Typical use: after training SFT then DPO via
    `optiq lora train --method dpo --mount-adapter ./sft_adapter`,
    merge the two adapters into one drop-in artifact for shipping.

    Examples:

        optiq lora merge ./adapters/humanizer-sft \\
                         ./adapters/humanizer-dpo \\
                         -o ./adapters/humanizer-merged
    """
    from pathlib import Path

    from .lora.merge import merge_adapters

    out = Path(output)
    if out.exists() and not force:
        click.echo(
            f"output dir exists: {out}. Pass --force to overwrite.",
            err=True,
        )
        sys.exit(1)
    if out.exists() and force:
        import shutil
        shutil.rmtree(out)

    scale_overrides = None
    if scales:
        try:
            scale_overrides = [float(s.strip())
                               for s in scales.split(",") if s.strip()]
        except ValueError:
            click.echo(
                f"--scales: expected comma-separated floats, "
                f"got {scales!r}", err=True)
            sys.exit(1)

    stats = merge_adapters(
        adapter_paths=list(adapters),
        output_dir=out,
        scales=scale_overrides,
    )

    click.echo(f"merged {len(adapters)} adapters → {out}")
    click.echo(
        f"  layers merged (rank-concat): "
        f"{stats['layers_merged']}"
    )
    click.echo(
        f"  layers in only one source: "
        f"{stats['layers_only_in_one']}"
    )
    click.echo(f"  total keys: in={stats['total_keys_in']} → "
               f"out={stats['total_keys_out']}")
    click.echo(f"  scales folded: {stats['source_scales']}")


@lora.command("export")
@click.argument("model")
@click.option("--adapter", "adapters", multiple=True, required=True,
              help=("Adapter directory or safetensors file to bundle. "
                    "Pass --adapter repeatedly to include multiple; "
                    "they will be rank-concat merged into a single "
                    "top-level adapter, with the originals preserved "
                    "under source_adapters/<name>/ for reference."))
@click.option("--output", "-o", required=True,
              help="Output directory for the self-contained model.")
@click.option("--force", is_flag=True,
              help="Overwrite output dir if it exists.")
def lora_export(model, adapters, output, force):
    """Bundle a base model + one or more LoRA adapters into a single
    deployable directory ready for ``optiq serve --model <dir>`` or
    ``mlx_lm.generate --model <dir>``.

    Layout produced::

        <output>/
        ├── *.safetensors, config.json, tokenizer*    base files
        ├── adapters.safetensors                       merged adapter
        ├── adapter_config.json
        ├── source_adapters/<name>/...                 originals
        └── optiq_export.json                          audit trail

    Examples:

        optiq lora export mlx-community/MiniCPM5-1B-OptiQ-4bit \\
            --adapter ./adapters/humanizer-sft \\
            --adapter ./adapters/humanizer-dpo \\
            -o ./humanizer-1B-OptiQ-4bit
    """
    import json
    import shutil
    from pathlib import Path

    from .lora.merge import merge_adapters

    out = Path(output)
    if out.exists() and not force:
        click.echo(
            f"output dir exists: {out}. Pass --force to overwrite.",
            err=True,
        )
        sys.exit(1)
    if out.exists() and force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Materialize the base model directory locally (HF id -> cache
    # path, or just use the local path as-is).
    click.echo(f"[optiq.lora.export] resolving base model: {model}")
    base_dir = Path(model)
    if not base_dir.exists():
        try:
            from huggingface_hub import snapshot_download
            base_dir = Path(snapshot_download(repo_id=model))
        except Exception as exc:
            click.echo(
                f"could not resolve {model} (not a local path, and HF "
                f"snapshot_download failed: {exc})", err=True)
            sys.exit(1)

    # 2) Copy base model files into the output.
    click.echo(f"[optiq.lora.export] copying base files from {base_dir}")
    skip_names = {"adapter_config.json", "adapters.safetensors"}
    for src in sorted(base_dir.iterdir()):
        if src.name in skip_names or src.name.startswith("."):
            continue
        if src.is_file():
            shutil.copy2(src, out / src.name)
        elif src.is_dir() and src.name != "source_adapters":
            shutil.copytree(src, out / src.name)

    # 3) Merge adapters into the top-level adapter slot.
    adapter_list = list(adapters)
    if len(adapter_list) == 1:
        # Single adapter — just copy it.
        click.echo(f"[optiq.lora.export] single adapter; copying as-is")
        adp = Path(adapter_list[0])
        if adp.is_dir():
            # PEFT-style directory
            sf = (adp / "best" / "adapters.safetensors"
                  if (adp / "best" / "adapters.safetensors").exists()
                  else adp / "adapters.safetensors")
            cfg = adp / "adapter_config.json"
        else:
            sf = adp
            cfg = adp.parent / "adapter_config.json"
        shutil.copy2(sf, out / "adapters.safetensors")
        if cfg.exists():
            shutil.copy2(cfg, out / "adapter_config.json")
    else:
        click.echo(
            f"[optiq.lora.export] rank-concat merging "
            f"{len(adapter_list)} adapters"
        )
        merge_stats = merge_adapters(
            adapter_paths=adapter_list, output_dir=out,
        )
        click.echo(
            f"  merged {merge_stats['layers_merged']} layers; "
            f"{merge_stats['layers_only_in_one']} layers only in one "
            f"source"
        )

    # 4) Preserve original adapters as source_adapters/<name>/.
    src_root = out / "source_adapters"
    src_root.mkdir(exist_ok=True)
    for adp_in in adapter_list:
        adp = Path(adp_in)
        name = adp.name
        dest = src_root / name
        if dest.exists():
            shutil.rmtree(dest)
        if adp.is_dir():
            shutil.copytree(adp, dest)
        else:
            dest.mkdir(parents=True)
            shutil.copy2(adp, dest / adp.name)

    # 5) Audit-trail.
    (out / "optiq_export.json").write_text(json.dumps({
        "base_model": str(model),
        "base_model_local": str(base_dir),
        "adapters": [str(p) for p in adapter_list],
        "merged": len(adapter_list) > 1,
    }, indent=2) + "\n")

    click.echo(f"[optiq.lora.export] done → {out}")
    click.echo(
        f"  use:  optiq serve --model {out} "
        f"# adapter loads via top-level adapters.safetensors"
    )


@cli.command()
@click.argument("model")
@click.option("--target", default="mlx", help="Target backend (mlx)")
@click.option("--model-type", default="auto", type=click.Choice(["auto", "llm"]),
              help="Model type (auto-detected if not specified)")
@click.option("--target-bpw", default=5.0, type=float, help="Target bits per weight (default: 5.0)")
@click.option("--candidate-bits", default="4,8", help="Candidate bit-widths (comma-separated, e.g. 3,4,8)")
@click.option("--group-size", default=64, type=int, help="Quantization group size")
@click.option("--output", "-o", default=None, help="Output directory")
@click.option("--n-calibration", default=24, type=int,
              help=("Number of calibration samples drawn from the mix per "
                    "sensitivity probe. Default 24 → ~4 samples from each of "
                    "the 6 domains in optiq.jsonl. Higher = more stable KL "
                    "ranking, linearly slower convert."))
@click.option("--skip-baselines", is_flag=True, help="Skip baseline conversions")
@click.option("--reference", default="auto",
              type=click.Choice(["auto", "bf16", "uniform_4bit"]),
              help=("Sensitivity reference mode. 'auto' (default) tries bf16 "
                    "first; if the bf16 base won't fit in available RAM, "
                    "falls back to a uniform-4-bit baseline reference and "
                    "streams bf16 weights from disk for layer-level probes. "
                    "'bf16' forces the bf16 path (errors on too-big models). "
                    "'uniform_4bit' forces the baseline-streaming path "
                    "(useful when you want a deterministic mode regardless "
                    "of available RAM)."))
@click.option("--calibration-mix", default="optiq",
              help=("Calibration mix to use. 'optiq' (default) is the "
                    "hand-curated 6-domain mix shipped with the package "
                    "(prose · thinking traces · code · agent loops · tool "
                    "calling · constraint-bearing instructions). Or pass a "
                    "path to a JSONL file in the same "
                    "schema as optiq/calibration/data/optiq.jsonl for a "
                    "custom domain-specialised mix."))
@click.option("--n-floor-per-block", default=2, type=int,
              help=("Block-aware floor: minimum number of components per "
                    "transformer block kept at >= 2nd-lowest candidate bit. "
                    "Prevents per-layer sensitivity from concentrating all "
                    "the lowest-bit assignments into the middle of the "
                    "network (the compounding-error failure mode). "
                    "Set to 0 to disable. Default 2."))
@click.option("--method", default="optiq",
              type=click.Choice(["optiq", "static"]),
              help=("Bit-allocation method. 'optiq' (default): exact "
                    "calibration-driven KL sensitivity — the gold-standard "
                    "signal, but n_layers x n_bits x n_samples forward passes, "
                    "slow and memory-heavy on large models. 'static': "
                    "rule-based structural priority (embed/head, first/last "
                    "block, attention, MoE router > dense MLP > experts) at the "
                    "requested candidate bits + target BPW — no calibration, no "
                    "forward passes, runs in seconds. Matches optiq on typical "
                    "dense models; the fast path for large bases."))
def convert(model, target, model_type, target_bpw, candidate_bits,
            group_size, output, n_calibration, skip_baselines,
            reference, calibration_mix,
            n_floor_per_block, method):
    """Convert a model to optimized MLX format.

    Uses per-layer KL-divergence sensitivity analysis (calibration-driven)
    with multi-tier bit allocation to minimize size while preserving quality.
    The pipeline auto-routes between bf16 and uniform-4-bit-baseline references
    based on available RAM — large models that won't fit bf16 still get a
    calibration-driven signal.

    For multi-modal base models (Qwen3.5-VLM, Gemma-4 multimodal), mlx-lm's
    convert step quantizes only the language sub-module weights (vision /
    audio weights are dropped by mlx-lm during conversion, since the resulting
    artifact is for text inference). The original config.json shape is kept
    verbatim — mlx-lm loads the model via its VLM-arch class, which delegates
    to the language path for text-only inference.

    Examples:

        optiq convert Qwen/Qwen3-0.6B-base --target-bpw 4.0

        optiq convert Qwen/Qwen3-0.6B-base --target-bpw 3.5 --candidate-bits 2,3,4,8
    """
    bits_list = [int(b.strip()) for b in candidate_bits.split(",")]

    if output is None:
        safe_name = model.replace("/", "_").replace(".", "_")
        output = f"optiq_output/{safe_name}"

    click.echo(f"OptiQ: Converting {model} (type={model_type}) → {target}")
    click.echo(f"  Method: {method}")
    click.echo(f"  Target BPW: {target_bpw}, candidate bits: {bits_list}, group_size: {group_size}")
    if method == "optiq":
        click.echo(f"  Reference: {reference}")
        click.echo(f"  Calibration mix: {calibration_mix}")

    # Diffusion LLMs (DiffusionGemma) take a different pipeline: mlx-lm cannot
    # load them at all, and their meaningful forward is a denoising step over a
    # masked canvas, not a causal pass — calibrating one on causal forwards
    # would let the optimizer crush the layers only denoising depends on.
    from .models.diffusion import is_diffusion_model

    if is_diffusion_model(model):
        from .models.diffusion import run_diffusion_pipeline

        click.echo("  Diffusion LLM detected → masked-canvas calibration "
                   "over the denoising schedule")
        results = run_diffusion_pipeline(
            model_name=model,
            output_dir=output,
            target_bpw=target_bpw,
            n_calibration=n_calibration,
            candidate_bits=bits_list,
            group_size=group_size,
            skip_baselines=skip_baselines,
        )
        click.echo(f"\nOptiQ: done → {results['optiq_path']} "
                   f"({results['achieved_bpw']:.3f} bpw)")
        return

    from .models.llm import run_llm_pipeline

    results = run_llm_pipeline(
        model_name=model,
        output_dir=output,
        target_bpw=target_bpw,
        n_calibration=n_calibration,
        candidate_bits=bits_list,
        group_size=group_size,
        skip_baselines=skip_baselines,
        reference=reference,
        calibration_mix=calibration_mix,
        n_floor_per_block=n_floor_per_block,
        method=method,
    )
    click.echo(f"\nDone! OptiQ model saved to: {results.get('optiq_path', output)}")


@cli.command()
@click.argument("model_path")
@click.option("--baseline", default=None, help="Baseline model path for comparison")
@click.option("--n-samples", default=50, type=int, help="Number of evaluation samples")
def benchmark(model_path, baseline, n_samples):
    """Benchmark a converted model.

    Examples:

        optiq benchmark ./optiq_model --baseline ./uniform_4bit
    """
    from .utils.benchmark import (
        BenchmarkResult,
        measure_llm_perplexity,
        measure_llm_throughput,
        get_model_size,
        measure_model_bpw,
        print_comparison_table,
    )

    results = []

    click.echo(f"Benchmarking LLM: {model_path}")
    size = get_model_size(model_path)
    bpw = measure_model_bpw(model_path)

    click.echo("  Measuring perplexity...")
    ppl = measure_llm_perplexity(model_path, n_samples=n_samples)

    click.echo("  Measuring throughput...")
    tps = measure_llm_throughput(model_path)

    results.append(BenchmarkResult(
        name=os.path.basename(model_path),
        model_size_mb=size,
        bpw=bpw,
        perplexity=ppl,
        tokens_per_sec=tps,
    ))

    if baseline:
        click.echo(f"\nBenchmarking baseline: {baseline}")
        b_size = get_model_size(baseline)
        b_bpw = measure_model_bpw(baseline)

        click.echo("  Measuring perplexity...")
        b_ppl = measure_llm_perplexity(baseline, n_samples=n_samples)

        click.echo("  Measuring throughput...")
        b_tps = measure_llm_throughput(baseline)

        results.append(BenchmarkResult(
            name=os.path.basename(baseline),
            model_size_mb=b_size,
            bpw=b_bpw,
            perplexity=b_ppl,
            tokens_per_sec=b_tps,
        ))

    print_comparison_table(results)


_TIER1_TASKS = ("smoketest", "kl", "gsm8k-50")
_TIER2_TASKS = ("mmlu", "gsm8k", "ifeval", "bfcl", "humaneval", "hashhop")
_ALL_TASK = "all"
_ALL_CHOICES = (_ALL_TASK,) + _TIER1_TASKS + _TIER2_TASKS


@cli.command(name="eval")
@click.argument("model_path")
@click.option("--task", default="gsm8k", type=click.Choice(_ALL_CHOICES),
              help=("Evaluation task. 'smoketest' runs the fast triage "
                    "(KL + GSM8K-50, ~5 min on 27B). 'all' runs the full "
                    "benchmark suite (MMLU + GSM8K-1k + IFEval + BFCL + "
                    "HumanEval + HashHop, ~1.5 h on 27B). Or pick a single "
                    "benchmark."))
@click.option("--baseline", default=None,
              help="Baseline model path for side-by-side comparison. Works with "
                   "--task gsm8k (per-question diff) and --task all (full "
                   "Capability Score comparison table).")
@click.option("--n-samples", default=None, type=int,
              help="Number of evaluation samples (overrides per-task default)")
@click.option("--reference-model", default=None,
              help="Explicit reference model for the KL eval. Defaults to "
                   "auto-pick: bf16 base if it fits in RAM, else uniform-4-bit "
                   "quant of the same base.")
@click.option("--reference-mode", default="auto",
              type=click.Choice(["auto", "bf16", "uniform_4bit"]),
              help="KL reference selection strategy when --reference-model is unset.")
@click.option("--score", "show_score", is_flag=True,
              help="With --task all: also compute the Capability_Score (mean of the 6 benchmarks).")
@click.option("--reasoning", "reasoning", is_flag=True, default=False,
              help="Score a thinking model fairly: let it emit its <think> "
                   "block (don't suppress), give generation tasks a large token "
                   "budget so the trace completes, and score MMLU generatively "
                   "(parse the letter from the answer) instead of first-letter "
                   "logit argmax, which collapses to chance for reasoning models.")
@click.option("--reasoning-max-tokens", "reasoning_max_tokens", default=4096, type=int,
              help="Per-question generation budget in --reasoning mode (default "
                   "4096). It's a cap, not a fixed length — greedy decoding stops "
                   "at EOS, so short traces cost nothing extra. The old default "
                   "3072 was calibrated on VibeThinker's MMLU traces (p95 ~3050), "
                   "but TOOL-CALLING reasoning runs longer: on VibeThinker-BFCL a "
                   "3072 cap truncated the trace before the call landed, and the "
                   "OptiQ quant (which reasons longer) scored 43 where 6144 gave "
                   "57.5. 4096 is the safer floor; raise it for long reasoners on "
                   "tool-use or code tasks.")
@click.option("--repetition-penalty", "repetition_penalty", default=None, type=float,
              help="Decode the benchmarks with this repetition penalty. Default "
                   "is per-architecture (optiq.eval.decode): 1.0 for everything "
                   "except dhara, which loops into '1 + 1 + 1 + ...' under bare "
                   "greedy and needs 1.3, the penalty its reference Space uses. "
                   "Benchmarking a model outside the decode config it is meant to "
                   "run with scores the loop, not the model.")
@click.option("--kv-bits", default=None, type=int,
              help="Enable uniform KV-cache quantization for benchmarks that "
                   "generate (GSM8K / IFEval / BFCL / HumanEval). 4 or 8.")
@click.option("--kv-group-size", default=64, type=int,
              help="KV quantization group size (default: 64). Only used with --kv-bits.")
@click.option("--kv-config", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Path to OptiQ kv_config.json for per-layer mixed-precision KV. "
                   "Overrides --kv-bits. Generate via `optiq kv-cache <model>`.")
@click.option("--output-json", default=None, type=click.Path(dir_okay=False),
              help="Write the structured eval results to this JSON path. With "
                   "--task all, contains every benchmark metric + Capability_Score; "
                   "with single-task modes, contains that task's record.")
@click.option("--skip-kl", "skip_kl", is_flag=True, default=False,
              help="With --task all: skip the KL-divergence step. KL loads the "
                   "candidate AND a reference model at once, so a large quant on "
                   "a memory-tight Mac gets OOM-killed (taking the whole suite "
                   "with it). The Capability Score is the 6 benchmarks; KL is a "
                   "separate diagnostic.")
def eval_cmd(model_path, task, baseline, n_samples, reference_model,
             reference_mode, show_score, reasoning, reasoning_max_tokens,
             repetition_penalty, kv_bits, kv_group_size, kv_config, output_json,
             skip_kl):
    """Evaluate a model on standard benchmarks.

    \b
    Smoketest (~5 min on 27B):
        optiq eval ./optiq_model --task smoketest

    \b
    Full benchmark suite (~1.5 h on 27B):
        optiq eval ./optiq_model --task all
        optiq eval ./optiq_model --task all --score

    \b
    Single benchmarks:
        optiq eval ./optiq_model --task mmlu        # 1000-sample MMLU
        optiq eval ./optiq_model --task gsm8k       # 1000-sample GSM8K
        optiq eval ./optiq_model --task kl          # KL vs bf16/uniform-4
        optiq eval ./optiq_model --task humaneval   # 164 problems, sandboxed
        optiq eval ./optiq_model --task bfcl        # 200 simple tool calls
        optiq eval ./optiq_model --task ifeval      # full 540-prompt IFEval
    """
    # mlx-lm cannot load `diffusion_gemma` at all, and every eval module goes
    # through `from mlx_lm import load, generate`. Route it before ANY task runs,
    # not just --task all — otherwise `optiq eval` dies on a diffusion checkpoint
    # and its Capability Score can only come from an unshipped script.
    _install_diffusion_eval_if_needed(model_path)

    # Per-task default sample counts (None = full set or task-specific default)
    defaults = {
        "smoketest": None,
        "kl": 64,
        "gsm8k-50": 50,
        "mmlu": 1000,
        "gsm8k": 1000,
        "ifeval": None,         # full 540-prompt set
        "bfcl": 200,
        "humaneval": None,      # full 164 problems
        "hashhop": 25,          # 25 instances per hop × 4 hop levels = 100
        "all": None,
        # Legacy
    }
    if n_samples is None:
        n_samples = defaults.get(task)

    # KV cache quantization, if requested. Patches mlx_lm.generate so that
    # any downstream eval that uses generation picks it up. KL / MMLU only
    # use single forward passes — KV quant has no effect on them.
    if kv_config is not None:
        from .serve import _load_kv_config, install_mixed_kv, summarize_kv_config
        configs = _load_kv_config(kv_config)
        click.echo(f"[optiq.eval] mixed-precision KV from {kv_config}")
        click.echo(f"[optiq.eval]   {summarize_kv_config(configs)}")
        install_mixed_kv(kv_configs=configs, quantized_kv_start=0)
        # install_mixed_kv only patches mlx_lm.server.stream_generate;
        # eval modules call mlx_lm.generate.stream_generate directly, so
        # mirror the kwarg-injection patch onto that path too.
        _install_eval_kv_kwargs(max(c.bits for c in configs), 64)
    elif kv_bits is not None:
        click.echo(f"[optiq.eval] uniform {kv_bits}-bit KV (group_size={kv_group_size})")
        _install_eval_kv_kwargs(kv_bits, kv_group_size)

    # Routing
    _opts = _task_opts(reasoning, reasoning_max_tokens, repetition_penalty)

    if task == "smoketest":
        _run_smoketest(model_path, reference_model, reference_mode)
        return
    if task == "all":
        _run_all(model_path, n_samples, reference_model, reference_mode, show_score,
                 output_json=output_json, baseline=baseline,
                 reasoning=reasoning, reasoning_max_tokens=reasoning_max_tokens,
                 repetition_penalty=repetition_penalty, skip_kl=skip_kl)
        return
    if task == "kl":
        _run_kl(model_path, n_samples or 64, reference_model, reference_mode)
        return
    if task == "gsm8k-50":
        _run_gsm8k(model_path, baseline, n_samples or 50, opts=_opts, output_json=output_json)
        return
    if task == "gsm8k":
        _run_gsm8k(model_path, baseline, n_samples or 1000, opts=_opts, output_json=output_json)
        return
    if task == "mmlu":
        _run_mmlu(model_path, n_samples or 1000, opts=_opts, output_json=output_json)
        return
    if task == "ifeval":
        _run_ifeval(model_path, n_samples, opts=_opts, output_json=output_json)
        return
    if task == "bfcl":
        _run_bfcl(model_path, n_samples or 200, opts=_opts, output_json=output_json)
        return
    if task == "humaneval":
        _run_humaneval(model_path, n_samples, opts=_opts, output_json=output_json)
        return
    if task == "hashhop":
        _run_hashhop(model_path, n_samples or 25, opts=_opts, output_json=output_json)
        return


# ─── per-task runners ──────────────────────────────────────────────────

def _dump_task_json(output_json, model_path, task, res):
    """`--output-json` on a single task wrote nothing, silently, though its help
    says it "contains that task's record"."""
    if not output_json or res is None:
        return
    import json as _json

    def _val(r):
        for a in ("accuracy", "strict_acc", "pass_at_1", "overall_acc"):
            if hasattr(r, a):
                return getattr(r, a) * 100
        return float(r) if isinstance(r, (int, float)) else None

    rec = {"model_path": model_path, "task": task, "score_pct": _val(res)}
    with open(output_json, "w") as f:
        _json.dump(rec, f, indent=2, default=str)
    click.echo(f"[output] wrote {output_json}")


def _task_opts(reasoning, reasoning_max_tokens, repetition_penalty):
    """The decode options every single-task path used to drop on the floor.

    Only `--task all` forwarded these. The other eight branches did not even
    accept them, so:

      * `--task mmlu --reasoning` silently gave the first-letter logit argmax that
        the flag's own help says "collapses to chance for reasoning models" -- a
        user benchmarking a thinking quant read ~25% and concluded it was broken;
      * `--task gsm8k --output-json out.json` wrote no file, though the help says
        it "contains that task's record";
      * `--repetition-penalty` was ignored.
    """
    o = {}
    if reasoning:
        o["reasoning"] = True
        o["max_tokens"] = reasoning_max_tokens
    if repetition_penalty is not None:
        o["repetition_penalty"] = repetition_penalty
    return o



def _install_eval_kv_kwargs(kv_bits: int, kv_group_size: int) -> None:
    """Route eval's KV-quant through the same patch serve uses.

    `optiq eval --kv-bits 4` installed no maybe_quantize_kv_cache patch at all, so
    a sliding-window model hit mlx-lm's NotImplementedError at the first question
    past the window.
    """
    from .runtime.kv import patch_rotating_to_quantized
    patch_rotating_to_quantized()

    """Patch ``mlx_lm.generate.stream_generate`` so the eval modules' calls
    to ``mlx_lm.generate(...)`` pick up KV-quant kwargs.

    The serve-side ``install_quantized_kv`` only patches ``mlx_lm.server``;
    eval modules call ``mlx_lm.generate`` directly, so we need a parallel
    patch on that path.
    """
    import sys
    import mlx_lm.generate  # noqa: F401  ensure submodule loaded
    gen_mod = sys.modules["mlx_lm.generate"]  # __init__ shadows the attr
    original = gen_mod.stream_generate

    def patched(*args, **kwargs):
        kwargs.setdefault("kv_bits", kv_bits)
        kwargs.setdefault("kv_group_size", kv_group_size)
        kwargs.setdefault("quantized_kv_start", 0)
        yield from original(*args, **kwargs)

    gen_mod.stream_generate = patched


def _run_kl(model_path, n_prompts, reference_model, reference_mode):
    from .eval.kl import evaluate_kl
    click.echo(f"\nKL divergence eval: {model_path}")
    res = evaluate_kl(
        model_path,
        reference_path=reference_model,
        reference_mode=reference_mode,
        n_prompts=n_prompts,
    )
    click.echo(f"\n{res}")


def _run_gsm8k(model_path, baseline, n_samples, opts=None, output_json=None):
    from .eval.gsm8k import evaluate_gsm8k, print_gsm8k_report
    models = [(model_path, os.path.basename(model_path.rstrip("/")))]
    if baseline:
        models.append((baseline, os.path.basename(baseline.rstrip("/"))))
    all_results = []
    for path, name in models:
        click.echo(f"\nEvaluating {name} on GSM8K ({n_samples} questions)...")
        result = evaluate_gsm8k(path, n_samples=n_samples, **(opts or {}))
        print_gsm8k_report(result)
        all_results.append((name, result))
    if len(all_results) > 1:
        click.echo(f"\n  GSM8K Comparison")
        click.echo(f"  {'=' * 50}")
        click.echo(f"  {'Model':<30} {'Accuracy':>10}")
        click.echo(f"  {'-' * 50}")
        for name, r in all_results:
            click.echo(f"  {name:<30} {r.accuracy:>9.1%}")
        click.echo(f"  {'=' * 50}")
    # Lab WP-3 / --output-json: write primary model result
    if all_results:
        _dump_task_json(output_json, model_path, "gsm8k", all_results[0][1])


def _mmlu_fn(model_path):
    """The MMLU scorer this model needs.

    A diffusion forward predicts a CANVAS, not a next token, so the stock
    first-letter logit argmax reads " A"/" B"/" C"/" D" out of a denoising canvas
    and returns a plausible, meaningless number. `_run_all_for_model` substituted
    the canvas scorer; `--task mmlu` did not, so the same checkpoint printed two
    different MMLU numbers depending on the flag, with no error either way.
    """
    if _install_diffusion_eval_if_needed(model_path):
        from .vlm.diffusion_gemma.eval_adapter import evaluate_mmlu_diffusion
        return evaluate_mmlu_diffusion
    from .eval.mmlu import evaluate_mmlu
    return evaluate_mmlu


def _run_mmlu(model_path, n_samples, opts=None, output_json=None):
    evaluate_mmlu = _mmlu_fn(model_path)
    click.echo(f"\n5-shot MMLU eval ({n_samples} samples): {model_path}")
    res = evaluate_mmlu(model_path, n_samples=n_samples, **(opts or {}))
    click.echo(f"\n{res}")
    _dump_task_json(output_json, model_path, "mmlu", res)


def _run_ifeval(model_path, n_samples, opts=None, output_json=None):
    from .eval.ifeval import evaluate_ifeval
    label = "full set" if n_samples is None else f"{n_samples} prompts"
    click.echo(f"\nIFEval ({label}): {model_path}")
    res = evaluate_ifeval(model_path, n_samples=n_samples, **(opts or {}))
    click.echo(f"\n{res}")


def _run_bfcl(model_path, n_samples, opts=None, output_json=None):
    from .eval.bfcl import evaluate_bfcl
    click.echo(f"\nBFCL-V3 simple ({n_samples} questions): {model_path}")
    res = evaluate_bfcl(model_path, n_samples=n_samples, **(opts or {}))
    click.echo(f"\n{res}")
    _dump_task_json(output_json, model_path, "bfcl", res)


def _run_humaneval(model_path, n_samples, opts=None, output_json=None):
    from .eval.humaneval import evaluate_humaneval
    label = "full 164 problems" if n_samples is None else f"{n_samples} problems"
    click.echo(f"\nHumanEval pass@1 ({label}): {model_path}")
    res = evaluate_humaneval(model_path, n_samples=n_samples, **(opts or {}))
    click.echo(f"\n{res}")


def _run_hashhop(model_path, n_per_hops, opts=None, output_json=None):
    from .eval.hashhop import evaluate_hashhop
    click.echo(f"\nHashHop ({n_per_hops} per hop × 4 hop levels): {model_path}")
    res = evaluate_hashhop(model_path, n_per_hops=n_per_hops, **{('max_new_tokens' if k=='max_tokens' else k): v for k, v in (opts or {}).items()})
    click.echo(f"\n{res}")


def _run_smoketest(model_path, reference_model, reference_mode):
    """Smoketest: KL + GSM8K-50, ~5 min on a 27B."""
    from .eval.kl import evaluate_kl
    from .eval.gsm8k import evaluate_gsm8k, print_gsm8k_report
    click.echo(f"\nSmoketest: {model_path}\n" + "═" * 60)

    click.echo("\n── KL divergence vs reference ──")
    # A model can BE the reference — notably the bf16 base, which is exactly what
    # you pass as --baseline when asking "does the quant still behave like the
    # original". It has no reference of its own and KL against itself is 0, so skip
    # it rather than abort the whole suite on the last model of the run.
    kl = None
    try:
        kl = evaluate_kl(
            model_path,
            reference_path=reference_model,
            reference_mode=reference_mode,
            n_prompts=64, seq_len=256,
        )
        click.echo(f"  {kl}")
    except RuntimeError as e:
        if "reference" not in str(e).lower():
            raise
        click.echo("  skipped — this model is the reference (KL against itself is 0)")

    click.echo("\n── GSM8K-50 (math reasoning, 50 samples) ──")
    gs = evaluate_gsm8k(model_path, n_samples=50)
    print_gsm8k_report(gs)

    click.echo("\n" + "═" * 60)
    _kl_txt = f"KL mean = {kl.mean_kl:.4f}" if kl is not None else "KL = n/a (no reference)"
    click.echo(f"Smoketest complete. {_kl_txt}, "
               f"GSM8K-50 = {gs.accuracy * 100:.1f}%")


def _install_diffusion_eval_if_needed(model_path) -> bool:
    """Route a diffusion checkpoint through the vendored decode for the eval suite.

    Every eval module does `from mlx_lm import load, generate`, and mlx-lm cannot
    load `diffusion_gemma` at all ("Model type diffusion_gemma not supported"). The
    adapter monkeypatches both to the diffusion API. MMLU needs more than that: it
    is logit-based, and a diffusion forward predicts a *canvas*, not a next token —
    so it is scored off the canvas's answer slot instead.

    Without this, `optiq eval --task all` on DiffusionGemma simply dies, and its
    Capability Score could only ever be produced by an unshipped script.
    """
    try:
        from .models.diffusion import is_diffusion_model
        if not is_diffusion_model(model_path):
            return False
        from .vlm.diffusion_gemma.eval_adapter import install_diffusion_eval_adapter
        install_diffusion_eval_adapter()
        click.echo("[optiq.eval] DiffusionGemma → vendored masked-diffusion decode "
                   "(MMLU scored off the canvas answer slot)")
        return True
    except Exception:
        return False


def _run_all_for_model(model_path, reference_model, reference_mode, show_score,
                       reasoning=False, reasoning_max_tokens=4096,  # see --reasoning-max-tokens help
                       repetition_penalty=None, skip_kl=False):
    """Run the 6-metric suite + KL on a single model. Returns
    (record_dict, quality_score_or_None). Used by ``_run_all`` for both
    the candidate path and (optionally) the baseline path.

    ``reasoning=True`` scores a thinking model fairly (generative MMLU + large
    token budget so <think> traces complete; see ``optiq eval --reasoning``)."""
    # Reasoning kwargs: the per-question budget key is max_tokens everywhere
    # except hashhop (max_new_tokens). Empty in normal mode (keep task defaults).
    rk = dict(reasoning=True, max_tokens=reasoning_max_tokens) if reasoning else {}
    rk_hh = dict(reasoning=True, max_new_tokens=reasoning_max_tokens) if reasoning else {}

    # A model that needs a repetition penalty to stay coherent (dhara) gets one
    # from optiq.eval.decode by architecture; --repetition-penalty overrides it.
    # Left at None, every model that was fine before decodes exactly as before.
    if repetition_penalty is not None:
        rk["repetition_penalty"] = repetition_penalty
        rk_hh["repetition_penalty"] = repetition_penalty
    from .eval.kl import evaluate_kl
    from .eval.mmlu import evaluate_mmlu

    evaluate_mmlu = _mmlu_fn(model_path)      # canvas scorer for a diffusion model
    from .eval.gsm8k import evaluate_gsm8k, print_gsm8k_report
    from .eval.ifeval import evaluate_ifeval
    from .eval.bfcl import evaluate_bfcl
    from .eval.humaneval import evaluate_humaneval
    from .eval.hashhop import evaluate_hashhop

    click.echo(f"\nBenchmark suite: {model_path}\n" + "═" * 60)

    click.echo("\n── KL divergence vs reference ──")
    # A model can BE the reference — notably the bf16 base, which is exactly what
    # you pass as --baseline when asking "does the quant still behave like the
    # original". It has no reference of its own and KL against itself is 0, so skip
    # it rather than abort the whole suite on the last model of the run.
    kl = None
    if skip_kl:
        # KL loads BOTH the candidate and a reference model at once, so on a
        # memory-tight machine a large candidate (e.g. a 20 GB mixed quant + an
        # 11 GB uniform-4 reference) is OOM-killed by the OS — an uncatchable
        # SIGKILL that takes the whole suite with it. The Capability Score is the
        # 6 benchmarks; KL is a separate diagnostic, so --skip-kl drops it.
        click.echo("  skipped (--skip-kl) — KL needs two models resident at once")
    else:
        try:
            kl = evaluate_kl(
                model_path,
                reference_path=reference_model,
                reference_mode=reference_mode,
                n_prompts=64, seq_len=256,
            )
            click.echo(f"  {kl}")
        except RuntimeError as e:
            if "reference" not in str(e).lower():
                raise
            click.echo("  skipped — this model is the reference (KL against itself is 0)")

    # A benchmark is fed untrusted model output, so it can raise on garbage a
    # model produced. That must not discard the benchmarks that already ran: the
    # suite takes hours, and losing MMLU + GSM8K + IFEval because BFCL choked on
    # a malformed tool call is the same class of bug as the summary print that
    # used to crash after every benchmark had been paid for.
    failed: list[str] = []

    def _bench(name, fn):
        try:
            return fn()
        except Exception as e:                                   # noqa: BLE001
            failed.append(name)
            click.echo(f"  FAILED: {type(e).__name__}: {e}")
            click.echo(f"  (continuing — the other benchmarks are not lost)")
            return None

    label = "MMLU (reasoning, generative)" if reasoning else "5-shot MMLU"
    click.echo(f"\n── {label} (1000 samples) ──")
    mmlu = _bench("MMLU", lambda: evaluate_mmlu(model_path, n_samples=1000, **rk))
    if mmlu is not None:
        click.echo(f"  {mmlu}")

    click.echo("\n── GSM8K (1000 samples) ──")
    gs = _bench("GSM8K", lambda: evaluate_gsm8k(model_path, n_samples=1000, **rk))
    if gs is not None:
        print_gsm8k_report(gs)

    click.echo("\n── IFEval (full set) ──")
    ife = _bench("IFEval", lambda: evaluate_ifeval(model_path, n_samples=None, **rk))
    if ife is not None:
        click.echo(f"  {ife}")

    click.echo("\n── BFCL-V3 simple (200 questions) ──")
    bfcl = _bench("BFCL", lambda: evaluate_bfcl(model_path, n_samples=200, **rk))
    if bfcl is not None:
        click.echo(f"  {bfcl}")

    click.echo("\n── HumanEval pass@1 (164 problems, sandboxed) ──")
    he = _bench("HumanEval", lambda: evaluate_humaneval(model_path, n_samples=None, **rk))
    if he is not None:
        click.echo(f"  {he}")

    click.echo("\n── HashHop (25 instances × 4 hop levels, ~12k ctx) ──")
    hh = _bench("HashHop", lambda: evaluate_hashhop(model_path, n_per_hops=25, **rk_hh))
    if hh is not None:
        click.echo(f"  {hh}")

    def _pct(res, attr):
        """Percentage from a benchmark result, whatever shape it came back in.

        Most evaluate_* return a dataclass with a 0..1 ratio. The diffusion MMLU
        adapter returns a bare float that is ALREADY a percentage, because a
        diffusion forward has no next-token logits to build an MMLUResult from.
        Calling .accuracy on that float raises, and it raises HERE -- in the
        summary, after all six benchmarks have finished. On a 26B that is five
        hours of completed work destroyed by the last line of the run.
        """
        if res is None:
            return None
        if isinstance(res, (int, float)):
            return float(res)
        return getattr(res, attr) * 100

    def _fmt(v):
        return f"{v:.1f}%" if v is not None else "FAILED"

    mmlu_pct = _pct(mmlu, "accuracy")
    gsm8k_pct = _pct(gs, "accuracy")
    ifeval_pct = _pct(ife, "strict_acc")
    bfcl_pct = _pct(bfcl, "accuracy")
    humaneval_pct = _pct(he, "pass_at_1")
    hashhop_pct = _pct(hh, "overall_acc")

    click.echo("\n" + "═" * 60)
    click.echo("Benchmark summary")
    click.echo(f"  KL mean         {kl.mean_kl:.4f}" if kl is not None
               else "  KL mean         n/a (no reference)")
    click.echo(f"  MMLU            {_fmt(mmlu_pct)}")
    click.echo(f"  GSM8K           {_fmt(gsm8k_pct)}")
    click.echo(f"  IFEval (strict) {_fmt(ifeval_pct)}")
    click.echo(f"  BFCL            {_fmt(bfcl_pct)}")
    click.echo(f"  HumanEval       {_fmt(humaneval_pct)}")
    click.echo(f"  HashHop         {_fmt(hashhop_pct)}")
    if failed:
        click.echo(f"\n  {len(failed)} benchmark(s) failed: {', '.join(failed)}")
        click.echo("  The Capability Score is a mean of all six, so it is not "
                   "comparable to a complete run.")

    qs = None
    if show_score:
        from .eval.score import compute_quality_score
        qs = compute_quality_score(
            model_path,
            mmlu_pct=mmlu_pct,
            gsm8k_pct=gsm8k_pct,
            ifeval_pct=ifeval_pct,
            bfcl_pct=bfcl_pct,
            humaneval_pct=humaneval_pct,
            hashhop_pct=hashhop_pct,
        )
        click.echo("\n" + "─" * 60)
        click.echo(qs)

    record = {
        "model_path": model_path,
        "kl_mean": kl.mean_kl if kl is not None else None,
        "kl_p95": kl.p95_kl if kl is not None else None,
        "kl_reference_mode": kl.reference_mode if kl is not None else None,
        "kl_reference_path": kl.reference_path if kl is not None else None,
        "mmlu_pct": mmlu_pct,
        "gsm8k_pct": gsm8k_pct,
        "ifeval_strict_pct": ifeval_pct,
        "ifeval_loose_pct": _pct(ife, "loose_acc"),
        "bfcl_pct": bfcl_pct,
        "humaneval_pct": humaneval_pct,
        "hashhop_pct": hashhop_pct,
        "hashhop_by_hops": hh.by_hops if hh is not None else None,
        "hashhop_context_tokens": hh.context_tokens_observed if hh is not None else None,
        "failed_benchmarks": failed,
    }
    if qs is not None:
        record["quality_score"] = qs.score
        record["disk_gb"] = qs.disk_gb
    return record, qs


def _format_capability_comparison(cand_rec, base_rec, cand_qs, base_qs):
    """Two-row Capability Score comparison table. Each metric prints both
    values with the delta in the third column (positive = candidate better)."""
    rows = [
        ("MMLU",      "mmlu_pct"),
        ("GSM8K",     "gsm8k_pct"),
        ("IFEval",    "ifeval_strict_pct"),
        ("BFCL",      "bfcl_pct"),
        ("HumanEval", "humaneval_pct"),
        ("HashHop",   "hashhop_pct"),
    ]
    out = []
    out.append("\n" + "═" * 70)
    out.append("Capability Score comparison  (candidate vs baseline)")
    out.append("─" * 70)
    out.append(f"  {'metric':<14}{'candidate':>14}{'baseline':>14}{'Δ (cand-base)':>18}")

    def _cell(v):  # a benchmark that _bench caught is None, not a float
        return f"{v:.1f}%" if v is not None else "FAILED"

    for label, key in rows:
        c = cand_rec.get(key)
        b = base_rec.get(key)
        if c is not None and b is not None:
            d = c - b
            delta = f"{'+' if d >= 0 else ''}{d:.1f} pp"
        else:
            delta = "n/a"       # one side failed; a delta would be meaningless
        out.append(f"  {label:<14}{_cell(c):>14}{_cell(b):>14}{delta:>18}")
    out.append("─" * 70)
    if cand_qs is not None and base_qs is not None:
        cs, bs = cand_qs.score, base_qs.score
        ds = cs - bs
        sign = "+" if ds >= 0 else ""
        out.append(f"  {'Capability':<14}{cs:>13.2f} {bs:>13.2f} {sign + f'{ds:.2f}':>18}")
        # The two means are only comparable over the SAME benchmarks; if a
        # benchmark failed on one side, say so instead of implying a clean delta.
        cand_have = {k for _, k in rows if cand_rec.get(k) is not None}
        base_have = {k for _, k in rows if base_rec.get(k) is not None}
        if cand_have != base_have:
            out.append("  (Δ not comparable: the two runs cover different benchmarks)")
        if cand_qs.disk_gb is not None and base_qs.disk_gb is not None:
            cd, bd = cand_qs.disk_gb, base_qs.disk_gb
            dd = cd - bd
            sign = "+" if dd >= 0 else ""
            out.append(f"  {'Disk (GB)':<14}{cd:>13.2f} {bd:>13.2f} {sign + f'{dd:.2f}':>18}")
    out.append("═" * 70)
    return "\n".join(out)


def _run_all(model_path, n_samples, reference_model, reference_mode, show_score,
             output_json: str | None = None, baseline: str | None = None,
             reasoning: bool = False, reasoning_max_tokens: int = 4096,  # 3072 truncated tool-call traces
             repetition_penalty: float | None = None, skip_kl: bool = False):
    """Full benchmark suite — MMLU + GSM8K-1k + IFEval + BFCL + HumanEval + HashHop.

    When ``baseline`` is set, the same suite runs on the baseline path
    too and a two-row comparison table is printed at the end. JSON output
    contains both records under ``candidate`` and ``baseline`` keys.
    """
    if n_samples is not None:
        # `--task all` uses fixed per-benchmark sizes (MMLU/GSM8K 1000, BFCL 200,
        # HashHop 25/hop, IFEval/HumanEval full). Say so instead of silently
        # ignoring the flag.
        click.echo(f"  note: --n-samples {n_samples} is ignored for --task all; "
                   "each benchmark uses its fixed size. Run a single --task to override.")
    cand_rec, cand_qs = _run_all_for_model(
        model_path, reference_model, reference_mode, show_score,
        reasoning=reasoning, reasoning_max_tokens=reasoning_max_tokens,
        repetition_penalty=repetition_penalty, skip_kl=skip_kl,
    )

    base_rec = None
    base_qs = None
    if baseline:
        click.echo("\n" + "▓" * 60)
        click.echo(f"Now running same suite on baseline: {baseline}")
        click.echo("▓" * 60)
        base_rec, base_qs = _run_all_for_model(
            baseline, reference_model, reference_mode, show_score,
            reasoning=reasoning, reasoning_max_tokens=reasoning_max_tokens,
            repetition_penalty=repetition_penalty,
        )
        click.echo(_format_capability_comparison(cand_rec, base_rec, cand_qs, base_qs))

    if output_json:
        import json
        if base_rec is None:
            record = cand_rec
        else:
            record = {"candidate": cand_rec, "baseline": base_rec}
        with open(output_json, "w") as f:
            json.dump(record, f, indent=2)
        click.echo(f"\n[output] wrote {output_json}")


@cli.command()
@click.argument("model_path")
@click.option("--calibrate", is_flag=True, help="Calibrate by measuring actual throughput")
def latency(model_path, calibrate):
    """Predict decode throughput for a quantized model.

    Uses the Apple Silicon latency model to estimate throughput without
    running inference. With --calibrate, measures actual throughput once
    and adjusts predictions for remaining models.

    Examples:

        optiq latency ./optiq_model

        optiq latency ./optiq_model --calibrate
    """
    import json
    from .core.latency import detect_hardware, CalibratedLatencyModel
    from .utils.benchmark import get_model_size

    hw = detect_hardware()
    click.echo(f"Hardware: {hw.name} ({hw.memory_bandwidth_gbps:.0f} GB/s)")

    size_mb = get_model_size(model_path)
    size_bytes = size_mb * 1024 * 1024

    # Read config to get layer count
    config_path = os.path.join(model_path, "config.json")
    n_layers = 197
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        n_layers = config.get("num_hidden_layers", 28) * 7 + 1

    # Raw prediction (linear layers only)
    linear_us = (size_bytes / hw.memory_bandwidth_bps) * 1e6 * 1.15 + 5.0 * n_layers
    raw_tps = 1e6 / linear_us

    click.echo(f"Model: {model_path}")
    click.echo(f"  Size: {size_mb:.1f} MB")
    click.echo(f"  Linear-layer latency: {linear_us:.0f} us")

    if calibrate:
        import time
        import mlx.core as mx
        from mlx_lm import load, generate
        import numpy as np

        click.echo("  Calibrating (measuring actual throughput)...")
        model, tokenizer = load(model_path)
        prompt = "The quick brown fox jumps over the lazy dog."

        for _ in range(8):
            generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)

        times = []
        for _ in range(15):
            mx.synchronize()
            t0 = time.perf_counter()
            generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
            mx.synchronize()
            times.append(50 / (time.perf_counter() - t0))

        actual_tps = float(np.median(times))
        del model

        cal = CalibratedLatencyModel(hw=hw)
        cal.calibrate(linear_us, actual_tps)

        click.echo(f"  Actual throughput: {actual_tps:.1f} tok/s")
        click.echo(f"  Overhead: {cal.overhead_us:.0f} us ({cal.overhead_us/(1e6/actual_tps)*100:.0f}%)")
        click.echo(f"  Calibrated prediction: {actual_tps:.1f} tok/s")
    else:
        click.echo(f"  Raw prediction: {raw_tps:.0f} tok/s (linear layers only)")
        click.echo(f"  Note: actual throughput will be lower due to attention/norm/framework overhead.")
        click.echo(f"  Use --calibrate for accurate prediction.")


@cli.command(name="kv-cache")
@click.argument("model_path")
@click.option("--target-bits", default=5.0, type=float, help="Target average KV cache bits (default: 5.0)")
@click.option("--candidate-bits", default="4,8", help="Candidate bit-widths (comma-separated)")
@click.option("--n-samples", default=5, type=int, help="Number of calibration samples")
@click.option("--seq-len", default=512, type=int, help="Calibration sequence length")
@click.option("--group-size", default=64, type=int, help="Quantization group size")
@click.option("--output", "-o", default=None, help="Output directory for results")
def kv_cache_cmd(model_path, target_bits, candidate_bits, n_samples, seq_len,
                 group_size, output):
    """Analyze and optimize KV cache quantization.

    Measures per-layer KV cache sensitivity and generates a mixed-precision
    configuration that keeps sensitive layers at higher precision.

    Examples:

        optiq kv-cache Qwen/Qwen3-0.6B-base --target-bits 5.0

        optiq kv-cache ./my_model --target-bits 6.0 --n-samples 10
    """
    import json
    from .core.kv_cache import (
        measure_kv_sensitivity, optimize_kv_cache, KVCacheConfig,
    )

    bits_list = [int(b.strip()) for b in candidate_bits.split(",")]

    if output is None:
        output = "optiq_output/kv_cache"
    os.makedirs(output, exist_ok=True)

    click.echo(f"OptiQ KV Cache Analysis: {model_path}")
    click.echo(f"  Target bits: {target_bits}, candidates: {bits_list}")

    # Step 1: Sensitivity analysis
    click.echo("\nStep 1: Measuring per-layer KV cache sensitivity...")
    results = measure_kv_sensitivity(
        model_path,
        candidate_bits=bits_list,
        n_samples=n_samples,
        seq_len=seq_len,
        group_size=group_size,
    )

    # Save sensitivity results
    sens_path = os.path.join(output, "kv_sensitivity.json")
    data = [
        {
            "layer_idx": r.layer_idx,
            "layer_name": r.layer_name,
            "sensitivities": {str(k): v for k, v in r.sensitivities.items()},
            "n_kv_heads": r.n_kv_heads,
            "head_dim": r.head_dim,
        }
        for r in results
    ]
    with open(sens_path, "w") as f:
        json.dump(data, f, indent=2)
    click.echo(f"  Saved sensitivity data to {sens_path}")

    # Step 2: Optimize
    click.echo(f"\nStep 2: Optimizing KV cache allocation (target: {target_bits} bits)...")
    configs = optimize_kv_cache(
        results,
        target_kv_bits=target_bits,
        candidate_bits=bits_list,
        group_size=group_size,
    )

    # Save config
    config_path = os.path.join(output, "kv_config.json")
    config_data = [{"layer_idx": c.layer_idx, "bits": c.bits, "group_size": c.group_size}
                   for c in configs]
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    click.echo(f"  Saved KV config to {config_path}")

    # Summary
    min_bits = min(bits_list)
    max_bits = max(bits_list)
    n_high = sum(1 for c in configs if c.bits == max_bits)
    n_low = sum(1 for c in configs if c.bits == min_bits)
    achieved = sum(c.bits for c in configs) / len(configs)
    click.echo(f"\n  Result: {n_low} layers @ {min_bits}-bit, {n_high} layers @ {max_bits}-bit")
    click.echo(f"  Achieved: {achieved:.2f} average bits")


def _resolve_model_dir(model_arg):
    """Local directory holding a model's weights, or None.

    Never downloads: a model that is not already cached returns None, and every
    caller treats that as "size unknown" and falls back to a safe default.
    """
    if not model_arg:
        return None
    if os.path.isdir(model_arg):
        return model_arg
    try:
        from mlx_lm.utils import hf_repo_to_path
        return str(hf_repo_to_path(model_arg))
    except Exception:
        return None


def _report_context_window(model_arg, cap):
    """Publish the effective context window at GET /v1/optiq/context.

    A client that must stay inside the window can't derive it: the native
    context from config.json is wrong whenever --max-context engaged, which is
    precisely the tight-RAM case where staying inside it matters. Best-effort —
    a client that gets nothing falls back to its own default.
    """
    import json as _json

    from .runtime import context_report
    native = None
    _dir = _resolve_model_dir(model_arg)
    if _dir:
        # Narrow: a missing/!unreadable config.json is expected and fine, but a
        # NameError here must not be swallowed. A broad except is how the first
        # version of this reported native=null for a model that plainly has it.
        try:
            with open(os.path.join(_dir, "config.json")) as f:
                cfg = _json.load(f)
        except (OSError, ValueError):
            cfg = None
        if isinstance(cfg, dict):
            tc = cfg.get("text_config")     # multimodal nests the LM geometry
            tc = tc if isinstance(tc, dict) else cfg
            native = tc.get("max_position_embeddings")
    context_report.set_native(native)
    context_report.set_cap(cap)
    context_report.install()


@cli.command(
    name="serve",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.option("--kv-bits", default=None, type=int,
              help="Uniform bits for KV cache quantization (4 or 8). Omit for fp16.")
@click.option("--kv-group-size", default=64, type=int, help="KV quantization group size (default: 64)")
@click.option("--quantized-kv-start", default=0, type=int,
              help="Token offset at which quantization kicks in (default: 0)")
@click.option("--kv-config", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Path to OptiQ kv_config.json for per-layer mixed-precision KV. "
                   "Overrides --kv-bits. Generate via `optiq kv-cache <model>`.")
@click.option("--adapter", "adapter", multiple=True, default=(),
              help=("LoRA adapter(s) to mount at startup. Accepts either a "
                    "HuggingFace repo id (e.g. '<owner>/<adapter-name>'), "
                    "auto-downloaded on first use, or a local directory "
                    "containing adapter files. Pass --adapter repeatedly to "
                    "mount multiple adapters on the same base; requests "
                    "pick one per call via the 'adapters' field in the body "
                    "(adapter name = the directory's basename). With one "
                    "--adapter the behavior is the classic single-adapter "
                    "boot; with N>1 the OptiQ mounted-LoRA path is used so "
                    "switching between adapters is instant (one base in RAM, "
                    "N adapter sidecars, ContextVar-gated forward pass)."))
@click.option("--anthropic/--no-anthropic", default=True,
              help=("Expose an Anthropic-compatible /v1/messages endpoint "
                    "alongside the OpenAI /v1/chat/completions one. Lets "
                    "tools that speak the Anthropic API (e.g. Claude Code "
                    "via ANTHROPIC_BASE_URL or OpenClaw) talk to the local "
                    "model. Default: on."))
@click.option("--responses/--no-responses", default=True,
              help=("Expose an OpenAI Responses /v1/responses endpoint "
                    "alongside /v1/chat/completions. Required for Codex "
                    "(which uses Responses exclusively as of 2026); also "
                    "used by Cursor, Continue, Cline, and the openai SDK "
                    "Responses path. Default: on."))
@click.option("--context-scale", "context_scale", default=1.0, type=float,
              help=("Multiply reported usage token counts by this factor so a "
                    "context-aware agent (e.g. Claude Code) auto-compacts at the "
                    "right time for a smaller-context model. Factor = (window "
                    "your client assumes) / (your model's context); e.g. ~6.25 "
                    "for a 32k model behind a client that assumes 200k. Only the "
                    "reported usage is scaled; generation is untouched. "
                    "Default: 1.0 (off)."))
@click.option("--max-concurrent", "max_concurrent", default=8, type=int,
              help=("Cap how many requests decode in parallel (excess queue). "
                    "Each in-flight request holds its own KV cache, so mlx-lm's "
                    "datacenter default of 32 can OOM-crash on unified memory "
                    "under a burst. Lower it for big models / long contexts; "
                    "raise it if you have RAM to spare. Sets mlx-lm's "
                    "--decode-concurrency (and --prompt-concurrency to ~1/4). "
                    "Default: 8. An explicit --decode-concurrency wins."))
@click.option("--auth/--no-auth", default=True,
              help=("Require Bearer tokens starting with 'sk-optiq-' on "
                    "all POST requests. Missing Authorization header is "
                    "allowed (local-dev tolerance). Default: on."))
@click.option("--mtp", "mtp", is_flag=True, default=False,
              help=("Enable MTP (multi-token prediction) speculative "
                    "decoding via OptiqEngine. Requires a model with an "
                    "MTP head — produced by v0.1.0+ ``optiq convert`` "
                    "(which preserves mtp.* tensors). Default: off."))
@click.option("--mtp-depth", default=2, type=int,
              help="MTP speculation depth (number of draft tokens per "
                   "verify cycle). Default: 2 — empirical sweet spot on "
                   "Qwen3.5/3.6 (acceptance ~70%% at depth 2, drops at "
                   "depth 3+). Only used when --mtp is set.")
@click.option("--drafter", "drafter", default=None,
              help=("HF id (or local path) of a separate drafter model "
                    "for speculative decoding — the Gemma-4 family's "
                    "counterpart to --mtp (e.g. "
                    "``mlx-community/gemma-4-E2B-it-assistant-bf16`` as "
                    "drafter for gemma-4-26B/31B targets). Routes "
                    "generation through optiq.runtime.spec with γ=1 "
                    "greedy. Mutually exclusive with --mtp. Default: off."))
@click.option("--no-fused-kv", "no_fused_kv", is_flag=True, default=False,
              help=("Disable OptiQ's tight-RAM KV-quant path (streaming "
                    "per-layer cache conversion + fused quantized SDPA "
                    "with FlashAttention-2 N-tiling). The patches are "
                    "on by default whenever --kv-bits or --kv-config is "
                    "set, because stock mlx-lm's KV-quant path holds "
                    "both the fp16 and quantized cache co-resident at "
                    "conversion AND materializes a full (B, H_q, L_q, N) "
                    "scores matrix during prefill, leading to OOMs on "
                    "tight-RAM Macs. With our patches: 24 GB Mac, "
                    "granite-4.1-8b-4bit at 32k peak goes from 16.35 GB "
                    "(stock u4, would OOM) to 7.60 GB. Pass --no-fused-kv "
                    "only if you need bit-exact upstream mlx-lm behavior "
                    "for comparison."))
@click.option("--stream-experts/--no-stream-experts", "stream_experts",
              default=None,
              help=("Stream MoE expert weights from SSD instead of holding "
                    "the fused expert tensors resident, so large MoE quants "
                    "that would OOM (e.g. Qwen3.6-35B-A3B at 21 GB on a 24 GB "
                    "Mac) run at a few GB. Default: auto — streams only when "
                    "the model is too big to fit resident. Decode is slower "
                    "(experts read per token, ~3 tok/s on M4); prefill is "
                    "bucketed to avoid graph recompiles."))
@click.option("--stream-experts-cache", "stream_experts_cache", default=0,
              type=int, help=("LRU expert-weight cache size per projection "
                              "for expert streaming (default 0 = off; MoE "
                              "decode locality is poor so caching rarely "
                              "helps and adds overhead)."))
@click.option("--models-dir", "models_dir", default=None,
              type=click.Path(exists=True, file_okay=False),
              help=("Directory of locally-built OptiQ quants to advertise in "
                    "/v1/models (in addition to the HuggingFace cache and the "
                    "served model). Implies --allow-model-switch, so a client "
                    "can switch to one by passing its path as the request "
                    "'model'."))
@click.option("--allow-model-switch/--single-model", "allow_model_switch", default=False,
              help=("Let a request's 'model' field hot-swap the running server "
                    "to a different cached model (stock mlx-lm behavior). Default "
                    "is single-model: the 'model' field is a label and every "
                    "request serves --model, so a client sending the basename, a "
                    "friendly alias, or a wrong default (e.g. Claude Code's "
                    "'claude-…') is served the one model instead of 404ing. "
                    "Auto-enabled by --models-dir."))
@click.option("--idle-timeout", "idle_timeout", default=0, type=int,
              help=("Unload the model and free its RAM after this many seconds "
                    "with no requests; it reloads lazily on the next call. On "
                    "unified memory a served model holds RAM the machine can't "
                    "reuse while idle. Set this LONGER than your longest single "
                    "generation so a normal long decode is never interrupted. "
                    "Default: 0 (off — model stays resident)."))
@click.option("--max-context", "max_context", default="auto",
              help=("Cap the KV-cache context window so a long prompt can't OOM-"
                    "crash the server. 'auto' (default) engages a memory-safe cap "
                    "ONLY when the model's full native context wouldn't fit RAM — "
                    "otherwise a no-op. Pass an integer for a hard token cap, or "
                    "'off' to disable. Uses mlx-lm's RotatingKVCache (batching / "
                    "prompt-reuse / KV-quant all keep working); sliding-window "
                    "models manage their own KV and are untouched."))
@click.pass_context
def serve_cmd(ctx, kv_bits, kv_group_size, quantized_kv_start, kv_config,
              adapter, anthropic, responses, auth, context_scale, max_concurrent,
              mtp, mtp_depth, drafter,
              no_fused_kv, stream_experts, stream_experts_cache, models_dir,
              allow_model_switch, idle_timeout, max_context):
    """Run an OpenAI-compatible HTTP server with OptiQ-aware KV cache.

    Three KV modes:
      • --kv-config   per-layer mixed precision (OptiQ sensitivity-guided)
      • --kv-bits     uniform quantization (4 or 8 bits)
      • neither       fp16 KV (stock behavior)

    Optional LoRA:
      • --adapter     serve with a LoRA adapter applied (HF repo id or local path)

    All extra arguments are forwarded to mlx_lm.server (--model, --host,
    --port, --max-tokens, --temp, --top-p, --top-k, etc.).

    Examples:

        optiq kv-cache mlx-community/Qwen3.5-9B-OptiQ-4bit --target-bits 5.0
        optiq serve --kv-config optiq_output/kv_cache/kv_config.json \\
                    --model mlx-community/Qwen3.5-9B-OptiQ-4bit \\
                    --host 127.0.0.1 --port 8080

        # serve with a community LoRA adapter
        optiq serve --model mlx-community/Qwen3.5-9B-OptiQ-4bit \\
                    --adapter ./my_adapter
    """
    from .serve import (
        _load_kv_config, install_mixed_kv, install_quantized_kv, summarize_kv_config
    )

    # Resilient downloads: retry, then fall back Xet→HTTPS if the high-perf
    # transfer stalls (corporate proxy / flaky link). Installed first, before
    # anything can fetch config or weights.
    from .runtime.download import install as install_download_fallback
    install_download_fallback()

    # Tool-call arguments arrive as a JSON string on the OpenAI wire but must be
    # a mapping by the time the chat template runs (Gemma-4's canonical template
    # raises otherwise). Universally correct, so installed unconditionally.
    from .serve import install_tool_argument_normalization
    install_tool_argument_normalization()

    # Mistral/Devstral end a tool call with EOS rather than a marker, which
    # mlx-lm's state machine reports as a stop — dropping the call. Universally
    # correct (scoped to that shape), so installed unconditionally.
    from .serve import install_eos_terminated_tool_calls
    install_eos_terminated_tool_calls()

    # Upstream's BatchRotatingKVCache.merge crashes on a zero-length cache
    # during batching-with-history, killing the generation thread: the server
    # then accepts requests and answers none. Hits any sliding-window model
    # (Gemma-3/4, Ministral, Phi-3/4) on an ordinary multi-turn session, with
    # or without KV quantization, so it is installed unconditionally.
    from .runtime.kv.batch_rotating_merge import install as install_rot_merge
    if install_rot_merge():
        click.echo("[optiq.serve] rotating-cache merge fix installed "
                   "(upstream crashes batching-with-history on a zero-length cache)")
    else:
        click.echo("[optiq.serve] rotating-cache merge fix NOT installed "
                   "(upstream class missing or already patched) — multi-turn "
                   "on a sliding-window model may kill the generation thread")

    # Tight-RAM KV-quant path: on by default whenever KV-quant is enabled,
    # because stock mlx-lm causes a conversion spike + prefill scores-matrix
    # spike that OOMs on tight RAM. Users can pass --no-fused-kv to opt out.
    kv_quant_enabled = (kv_bits is not None) or (kv_config is not None)
    if kv_quant_enabled and not no_fused_kv:
        from .runtime.streaming_kv_quant import install as install_streaming_kv
        from .runtime.fused_quant_sdpa import install as install_fused_sdpa
        install_streaming_kv()
        installed_fused = install_fused_sdpa()
        click.echo("[optiq.serve] streaming per-layer KV-cache conversion enabled "
                   "(prevents fp16+quantized cache co-residency OOM)")
        if installed_fused:
            click.echo("[optiq.serve] fused quantized SDPA enabled "
                       "(FlashAttention-2 N-tiling, never materializes "
                       "the full scores matrix)")
    elif kv_quant_enabled and no_fused_kv:
        click.echo("[optiq.serve] --no-fused-kv: using stock mlx-lm KV-quant "
                   "(may OOM on tight RAM)")

    if kv_config is not None:
        configs = _load_kv_config(kv_config)
        click.echo(f"[optiq.serve] mixed-precision KV from {kv_config}")
        click.echo(f"[optiq.serve]   {summarize_kv_config(configs)}")
        install_mixed_kv(kv_configs=configs, quantized_kv_start=quantized_kv_start)
    elif kv_bits is not None:
        click.echo(
            f"[optiq.serve] uniform QuantizedKVCache (bits={kv_bits}, "
            f"group_size={kv_group_size}, start={quantized_kv_start})"
        )
        install_quantized_kv(
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )
    else:
        click.echo("[optiq.serve] fp16 KV (no quantization)")

    # Resolve --adapter (HF repo id or local path) to a local directory.
    # One adapter → forward as --adapter-path to mlx_lm.server's classic
    # single-adapter boot. Two or more → use OptiQ's mounted-LoRA path so
    # all N stay resident, switchable per-request via the 'adapters' field
    # in the body.
    argv_extra = list(ctx.args)
    from pathlib import Path as _Path
    if adapter:
        from .adapters.resolver import resolve_adapter_source
        from .adapters.registry import _inspect_adapter
        adapter_specs: list[tuple[str, str]] = []
        for raw_id in adapter:
            try:
                adapter_dir = resolve_adapter_source(raw_id)
            except Exception as exc:
                click.echo(f"[optiq.serve] failed to resolve adapter "
                           f"{raw_id!r}: {exc}", err=True)
                sys.exit(1)
            info = _inspect_adapter(raw_id, adapter_dir)
            extras = []
            if info.rank:
                extras.append(f"rank {info.rank}")
            if info.rank_distribution:
                extras.append(f"rank_dist {info.rank_distribution}")
            if info.optiq_sidecar:
                extras.append("OptiQ-trained")
            suffix = f" ({', '.join(extras)})" if extras else ""
            # Use the resolved directory's basename as the adapter name so
            # requests can pick it: {"adapters": "humanizer-sft-..."}.
            name = _Path(str(adapter_dir)).name
            adapter_specs.append((name, str(adapter_dir)))
            click.echo(f"[optiq.serve] adapter: {raw_id} → {adapter_dir}{suffix}  "
                       f"(name={name})")

        # Always route through OptiQ's mounted-LoRA path. The legacy
        # ``--adapter-path`` route mlx-lm provides is a no-op on OptiQ's
        # mixed-precision quantized base in some configs, so we no longer
        # use it. ``install_multi_adapter`` handles both the single- and
        # multi-adapter cases uniformly; in single-adapter mode the
        # patched handler default-activates the only registered adapter
        # when the request body omits ``"adapters"``.
        from .serve import install_multi_adapter
        install_multi_adapter(adapter_specs)
        if len(adapter_specs) == 1:
            click.echo(f"[optiq.serve] mounted-LoRA mode: 1 adapter "
                       f"({adapter_specs[0][0]}) auto-activated for every "
                       f"request.")
        else:
            click.echo(f"[optiq.serve] mounted-LoRA mode: "
                       f"{len(adapter_specs)} adapters, "
                       f"switch per-request via the 'adapters' field in "
                       f"the body. GET /v1/adapters to list registered "
                       f"names.")

    # Install Anthropic /v1/messages endpoint before the server starts.
    if anthropic:
        from .anthropic_server import install_anthropic_endpoint
        install_anthropic_endpoint()
        click.echo("[optiq.serve] Anthropic /v1/messages endpoint enabled "
                   "(set ANTHROPIC_BASE_URL=http://host:port to use from Claude Code / OpenClaw)")

    # Install OpenAI Responses /v1/responses endpoint (for Codex etc.)
    if responses:
        from .responses_server import install_responses_endpoint
        install_responses_endpoint()
        click.echo("[optiq.serve] OpenAI /v1/responses endpoint enabled "
                   "(point Codex / Cursor / Continue / Cline at http://host:port/v1)")

    # Global no-think: disable chain-of-thought on ALL chat endpoints. Useful
    # for agentic harnesses on local models (reasoning dominates per-turn
    # latency) and for fair, apples-to-apples harness comparisons where
    # thinking must be held constant. Opt in with OPTIQ_NO_THINK=1.
    if os.environ.get("OPTIQ_NO_THINK") == "1":
        from .no_think import install_no_think
        install_no_think()
        click.echo("[optiq.serve] global no-think enabled "
                   "(enable_thinking=False on all chat/messages/responses requests)")

    # Install auth LAST so it wraps the dispatch + endpoint handlers.
    if auth:
        from .auth import install_auth
        install_auth()
        click.echo("[optiq.serve] auth: requires Bearer sk-optiq-* tokens "
                   "(missing header allowed for local-dev)")

    # Parse --model once, up front: several memory-sizing decisions below
    # need the model's on-disk size, and re-deriving it per call site is how
    # one of them ended up referencing a variable defined 50 lines later.
    model_arg = None
    for i, arg in enumerate(argv_extra):
        if arg == "--model" and i + 1 < len(argv_extra):
            model_arg = argv_extra[i + 1]; break
        if arg.startswith("--model="):
            model_arg = arg.split("=", 1)[1]; break

    # Per-request thinking control via a model-id suffix (always on): a request
    # for `<model>:no-think` generates with enable_thinking=False, `:think`
    # forces it on. Lets `optiq code launch --model <id>:no-think` work with no
    # global env var, and advertises the variants in /v1/models so they are
    # discoverable. A request with no suffix is untouched.
    from .no_think import install_thinking_variants
    install_thinking_variants(served_model=model_arg)
    click.echo("[optiq.serve] thinking variants: request `<model>:no-think` / "
               "`:think` to control chain-of-thought per session")

    # Install threshold-based MLX cache cleanup so long-running servers
    # don't bloat the buffer reuse pool until Metal panics. Same default
    # as `optiq lab` (10 % of total RAM, floor 1 GB).
    from .lab.mlx_cleanup import (
        default_threshold_bytes, default_prompt_cache_bytes,
        inject_prompt_cache_bytes, install_server_cleanup,
    )
    cleanup_threshold = default_threshold_bytes()
    install_server_cleanup(cleanup_threshold)
    click.echo(f"[optiq.serve] mlx cleanup threshold: "
               f"{cleanup_threshold / 1024**3:.2f} GB "
               f"(clears MLX reuse pool when it grows past this)")

    # Cap mlx-lm's LRUPromptCache so long-context unique prompts don't
    # OOM the device. mlx-lm's default holds 10 caches with no byte cap.
    # Budget from what is left AFTER the weights, not a slice of total RAM:
    # a model that nearly fills the machine leaves little for the cache.
    pc_bytes = default_prompt_cache_bytes(_resolve_model_dir(model_arg))
    argv_before = list(argv_extra)
    argv_extra = inject_prompt_cache_bytes(argv_extra, pc_bytes)
    if argv_extra != argv_before:
        click.echo(f"[optiq.serve] mlx prompt-cache cap: "
                   f"{pc_bytes / 1024**3:.2f} GB "
                   f"(--prompt-cache-bytes; override with the same flag)")

    # Cap decode/prefill concurrency. Each in-flight request holds its own KV
    # cache, so mlx-lm's datacenter defaults (32 decode / 8 prefill) can OOM the
    # device under a burst on unified memory. Inject a Mac-safe cap unless the
    # user set the underlying flags directly.
    if max_concurrent and max_concurrent > 0:
        has_decode = any(a == "--decode-concurrency" or a.startswith("--decode-concurrency=")
                         for a in argv_extra)
        has_prompt = any(a == "--prompt-concurrency" or a.startswith("--prompt-concurrency=")
                         for a in argv_extra)
        if not has_decode:
            argv_extra += ["--decode-concurrency", str(int(max_concurrent))]
        if not has_prompt:
            argv_extra += ["--prompt-concurrency", str(max(1, int(max_concurrent) // 4))]
        if not (has_decode and has_prompt):
            click.echo(f"[optiq.serve] concurrency cap: {int(max_concurrent)} decode "
                       f"/ {max(1, int(max_concurrent)//4)} prefill in parallel "
                       f"(--max-concurrent; excess requests queue)")

    # Model-field policy. Single-model by default: the request's `model` field
    # is a label and every request serves --model (fixes the basename / alias /
    # wrong-default 404 from stock mlx-lm's load-on-mismatch). --allow-model-switch
    # (or --models-dir, which advertises switchable quants) opts back into
    # per-request hot-swap between cached models.
    switch_enabled = bool(allow_model_switch) or bool(models_dir)
    from .serve import install_model_field_policy
    install_model_field_policy(allow_switch=switch_enabled)
    if not switch_enabled:
        click.echo("[optiq.serve] single-model: the request 'model' field is a label "
                   "(any value serves --model; --allow-model-switch to hot-swap)")

    # Apply the model's recommended sampler from generation_config.json
    # unless the user already passed equivalent flags. Critical for MTP
    # acceptance — rejection sampling against a truncated distribution
    # (top_p / top_k) recovers ~30 ppt of literature acceptance vs the
    # untruncated temperature-only path.
    from .runtime.gen_config import read_recommended_sampling, merge_into_argv
    if model_arg:
        recommended = read_recommended_sampling(model_arg)
        if recommended:
            argv_extra = merge_into_argv(
                argv_extra, recommended,
                prefix_log="[optiq.serve] applying model-recommended sampler:",
            )

    # Diffusion LMs (DiffusionGemma, LLaDA2-MoE): route mlx_lm.server's load +
    # generation through OptiQ's vendored masked-diffusion decoder.
    if model_arg:
        try:
            from .vlm.diffusion_gemma.loader import load_config as _diff_cfg
            _mt = _diff_cfg(model_arg).get("model_type")
            if _mt in ("diffusion_gemma", "llada2_moe"):
                from .serve import install_diffusion_serving
                install_diffusion_serving()
                _label = {"diffusion_gemma": "DiffusionGemma",
                          "llada2_moe": "LLaDA2-MoE"}[_mt]
                click.echo(f"[optiq.serve] {_label}: vendored masked-diffusion decode")
        except Exception:
            pass

    # Install speculative decoding (opt-in). Two mutually exclusive paths:
    #   • --mtp       Qwen-style bundled MTP head via OptiqEngine
    #   • --drafter   Gemma-style separate drafter via optiq.runtime.spec
    if mtp and drafter:
        click.echo("[optiq.serve] --mtp and --drafter are mutually exclusive",
                   err=True)
        sys.exit(1)

    if mtp or drafter:
        model_path = None
        for i, arg in enumerate(argv_extra):
            if arg == "--model" and i + 1 < len(argv_extra):
                model_path = argv_extra[i + 1]
                break
            if arg.startswith("--model="):
                model_path = arg.split("=", 1)[1]
                break
        flag = "--mtp" if mtp else "--drafter"
        if not model_path:
            click.echo(f"[optiq.serve] {flag} requires --model <path>", err=True)
            sys.exit(1)

        if mtp:
            # Dhara-AR has no MTP head — it self-speculates (drafts a block via
            # its block-diffusion forward, verifies with its AR forward). Route
            # --mtp to that self-speculation path for dhara_ar models.
            _is_dhara = False
            try:
                from .vlm.diffusion_gemma.loader import load_config as _lc
                _is_dhara = _lc(model_path).get("model_type") == "dhara_ar"
            except Exception:
                pass
            if _is_dhara:
                from .serve import install_dhara_serving
                click.echo("[optiq.serve] Dhara self-speculation enabled "
                           f"(--mtp → selfspec, model={model_path})")
                install_dhara_serving("selfspec")
            else:
                from .serve import install_mtp_speculation
                click.echo(f"[optiq.serve] MTP speculation enabled "
                           f"(depth={mtp_depth}, model={model_path})")
                install_mtp_speculation(model_path=model_path, depth=mtp_depth)
        else:
            from .serve import install_assistant_drafter
            click.echo(f"[optiq.serve] assistant drafter enabled "
                       f"(drafter={drafter}, target={model_path})")
            install_assistant_drafter(target_model_path=model_path,
                                       drafter_id=drafter)

    # Enable multimodal (image+text) serving when the model ships an
    # ``optiq_vision`` sidecar. Installed LAST so it wraps whatever
    # stream_generate the speculation paths set up; transparent for text-only
    # requests (delegates) and text-only models (the image path never fires).
    if model_arg:
        try:
            from .sidecar_layout import local_model_dir
            from .vlm import has_vision_sidecar
            from .serve import install_vision_serving

            # model_arg is usually a Hub repo id, which is NOT a directory. This
            # used to be `os.path.isdir(model_arg) and has_vision_sidecar(...)`,
            # so `optiq serve --model mlx-community/...` never enabled vision.
            _vis_dir = local_model_dir(model_arg)
            _vis_local = _vis_dir is not None and has_vision_sidecar(_vis_dir)
            install_vision_serving(model_arg)
            if _vis_local:
                click.echo("[optiq.serve] vision serving enabled "
                           "(image+text via optiq_vision sidecar)")
        except Exception as _exc:  # never block text serving on this
            click.echo(f"[optiq.serve] vision serving not installed: {_exc}",
                       err=True)

    # SSD expert streaming for large MoE quants. The patched loader decides
    # per-model: "auto" streams only a MoE too big to fit resident; "on" always
    # streams a streamable MoE. Installed regardless of --model so per-request
    # model loads are covered too.
    _stream_mode = ("auto" if stream_experts is None
                    else ("on" if stream_experts else "off"))
    if _stream_mode != "off":
        from .serve import install_streaming_experts
        _suffix = (" (streams a MoE only if it won't fit resident)"
                   if _stream_mode == "auto" else "")
        click.echo(f"[optiq.serve] SSD expert streaming: {_stream_mode}{_suffix}")
        install_streaming_experts(mode=_stream_mode,
                                  cache_experts=stream_experts_cache,
                                  model_path=model_arg)

    # Resolve host/port from the forwarded args for the ready banner.
    serve_host, serve_port = "127.0.0.1", "8080"
    for i, arg in enumerate(argv_extra):
        if arg == "--host" and i + 1 < len(argv_extra):
            serve_host = argv_extra[i + 1]
        elif arg.startswith("--host="):
            serve_host = arg.split("=", 1)[1]
        elif arg == "--port" and i + 1 < len(argv_extra):
            serve_port = argv_extra[i + 1]
        elif arg.startswith("--port="):
            serve_port = arg.split("=", 1)[1]

    # mlx-lm prints "Starting httpd at ... port ..." via logging.info, but a
    # root logging handler already exists by now (huggingface_hub / transformers
    # configure one at import), so mlx-lm's own logging.basicConfig(level=INFO)
    # is a no-op and that line is swallowed. The last thing users would
    # otherwise see is mlx-lm's scary "not recommended for production" warning,
    # which reads like the process died. Force the root level to INFO so the
    # "Starting httpd" confirmation actually appears, and print our own
    # unmissable ready banner so it is obvious the server is up and what to
    # expect on the first request (lazy model load → pause + memory spike).
    import logging as _logging
    if _logging.getLogger().level > _logging.INFO or _logging.getLogger().level == 0:
        _logging.getLogger().setLevel(_logging.INFO)

    click.echo("[optiq.serve] " + "─" * 56)
    click.echo(f"[optiq.serve] server is starting at http://{serve_host}:{serve_port}")
    click.echo("[optiq.serve] the model loads on the FIRST request, not now. "
               "Expect a one-time pause and memory spike when the first call arrives.")
    click.echo("[optiq.serve] this process stays in the foreground and keeps "
               "serving; the mlx-lm warning just below is harmless. Ctrl-C to stop.")
    click.echo("[optiq.serve] " + "─" * 56)

    # Structured / JSON-constrained output: OpenAI `response_format`
    # (json_object / json_schema) + vLLM-style guided_json/regex/choice.
    # Always on; activates only when a request carries a spec. Torch-free
    # (lm-format-enforcer), so it doesn't pull PyTorch into the MLX runtime.
    import mlx_lm.server as _srv_mod

    # Per-request sampling diversity: mlx-lm's compiled categorical sampler
    # freezes the RNG on the generation worker thread, so temperature/seed are
    # ignored and output is effectively greedy. Swap in a thread-safe sampler.
    from .runtime.sampler_rng import install as install_sampler_rng
    if install_sampler_rng():
        click.echo("[optiq.serve] sampling RNG fix applied "
                   "(per-request temperature/seed now vary on the worker thread)")

    from .runtime.structured import install as install_structured
    install_structured(_srv_mod)
    click.echo("[optiq.serve] structured output ready "
               "(response_format / guided_json / guided_regex / guided_choice)")

    # Tool-call healing: recover malformed tool calls (Hermes tags, fenced/bare
    # JSON, trailing commas, function-call form) that leak into content into
    # proper OpenAI tool_calls. Applies to non-streaming completions; requests
    # without tools are untouched.
    from .runtime.tool_healing import install as install_tool_healing
    install_tool_healing(_srv_mod)
    click.echo("[optiq.serve] tool-call healing ready (malformed tool calls "
               "repaired server-side)")

    # Model switching: mlx-lm already hot-swaps to whatever model a request asks
    # for and lists HF-cache models in /v1/models. --models-dir additionally
    # advertises locally-built quants so a client can discover + switch to them.
    if models_dir:
        from .runtime.model_listing import install as install_model_listing
        install_model_listing(_srv_mod, models_dir)
        click.echo(f"[optiq.serve] /v1/models also advertises local quants in {models_dir}")

    # Model variants: expose thinking on/off as API-visible model ids. A client
    # sends "<model>:no-think" and gets the variant without needing the
    # non-standard chat_template_kwargs field. Installed last so its /v1/models
    # wrapper composes over the model_listing / mlx-lm handler.
    from .runtime.variants import install as install_variants
    install_variants(_srv_mod)
    click.echo("[optiq.serve] model variants ready "
               "(append :think / :no-think to the model id)")

    # Context scaling: report usage token counts x factor so a context-aware
    # agent (Claude Code) auto-compacts at the right time for a smaller model.
    if context_scale and float(context_scale) != 1.0:
        from .runtime.context_scale import install as install_context_scale
        install_context_scale(_srv_mod, context_scale)
        click.echo(f"[optiq.serve] context scaling on (usage x{float(context_scale):g})")

    # Idle auto-unload: free the model's RAM after N seconds with no requests;
    # reloads lazily on the next call. Off by default (0).
    if idle_timeout and int(idle_timeout) > 0:
        from .runtime.idle_unload import install as install_idle_unload
        if install_idle_unload(float(idle_timeout)):
            click.echo(f"[optiq.serve] idle auto-unload: frees the model after "
                       f"{int(idle_timeout)}s idle (reloads on next request)")

    # Memory-aware context cap: keep a long prompt from OOM-crashing the server.
    # 'auto' engages only when the model's full native context wouldn't fit RAM.
    _installed_cap = None
    _mc = str(max_context).strip().lower()
    if _mc not in ("off", "none", "0", ""):
        from .runtime.context_cap import estimate_kv_token_cap
        from .runtime.context_cap import install as install_context_cap
        cap = None
        if _mc == "auto":
            # Estimate only from a resolved local dir (config + weights present).
            _dir = None
            if model_arg and os.path.isdir(model_arg):
                _dir = model_arg
            elif model_arg:
                try:
                    from mlx_lm.utils import hf_repo_to_path
                    _dir = str(hf_repo_to_path(model_arg))
                except Exception:
                    _dir = None  # not cached locally — skip auto-cap pre-download
            if _dir:
                # The cache is only bf16 when neither --kv-config nor --kv-bits
                # is set. Telling the estimator otherwise shrinks the window by
                # ~3x for no reason.
                _bits = None
                if kv_config is not None:
                    try:
                        from .serve import _load_kv_config
                        _entries = _load_kv_config(kv_config)
                        _bs = [int(e["bits"]) for e in _entries]
                        _bits = sum(_bs) / len(_bs) if _bs else None
                    except Exception:
                        _bits = None
                elif kv_bits is not None:
                    _bits = float(kv_bits)
                cap = estimate_kv_token_cap(
                    _dir, kv_bits=_bits, kv_group_size=int(kv_group_size or 64),
                )
        else:
            try:
                cap = int(_mc)
            except ValueError:
                cap = None
        if cap and install_context_cap(int(cap)):
            _how = "memory-safe auto" if _mc == "auto" else "explicit"
            click.echo(f"[optiq.serve] context cap: {int(cap)} tokens ({_how}); "
                       f"longer prompts rotate the KV window instead of OOM-crashing")
            _installed_cap = int(cap)

    # Tell clients (optiq code) what window they actually have to stay inside.
    # The native context alone is the wrong answer whenever a cap engaged.
    _report_context_window(model_arg, _installed_cap)

    sys.argv = ["mlx_lm.server"] + argv_extra
    from mlx_lm.server import main as mlx_main
    mlx_main()


@cli.command(name="lab")
@click.option("--lab-port", default=7860, type=int,
              help="Port for the Lab web UI (default: 7860).")
@click.option("--api-port", default=8080, type=int,
              help="Port for the model-serving API (default: 8080).")
@click.option("--host", default="127.0.0.1",
              help="Bind address for both the Lab UI and the API. "
                   "Localhost-only by default — change with care.")
@click.option("--model", default=None,
              help="Model to load into the API server. If omitted, the "
                   "Lab boots with no model; load one from the Models tab.")
@click.option("--mtp/--no-mtp", default=False,
              help="Enable MTP speculative decoding on the API server. "
                   "Requires a model with an MTP head.")
@click.option("--mtp-depth", default=2, type=int,
              help="MTP speculation depth (default: 2). Only used with --mtp.")
@click.option("--drafter", default=None,
              help="HF id (or local path) of a separate drafter model for "
                   "speculative decoding — Gemma-4 family counterpart to "
                   "--mtp. Mutually exclusive with --mtp.")
@click.option("--no-browser", is_flag=True,
              help="Don't auto-open the Lab URL in the system browser.")
@click.option("--clear-cache-threshold-bytes", "clear_cache_threshold_bytes",
              default=None, type=int,
              help="Bytes: clear MLX buffer reuse pool when it exceeds this. "
                   "Default: 10%% of total RAM, floor 1 GB. Mirrors mlx-lm "
                   "trainer's _clear_cache(threshold) pattern.")
def lab_cmd(lab_port, api_port, host, model, mtp, mtp_depth, drafter,
            no_browser, clear_cache_threshold_bytes):
    """Launch OptiQ Lab: local web UI for quantize / fine-tune / dataset / chat.

    Boots two services in one process:

      • The Lab UI at http://<host>:<lab-port>/  (open this in a browser)
      • The model-serving API at http://<host>:<api-port>/v1  (paste into
        Claude Code, Codex, etc.)

    The Lab UI sidebar shows the API URL with copy buttons for each
    supported integration's config. First launch shows a password-setup
    screen; subsequent launches go to login.

    Examples:

        # Just open the Lab — pick a model from the UI later
        optiq lab

        # Boot with a specific model + MTP speculation (Qwen path)
        optiq lab --model mlx-community/Qwen3.5-9B-OptiQ-4bit --mtp

        # Boot with a Gemma target + assistant drafter (Gemma path)
        optiq lab --model mlx-community/gemma-4-31B-it-OptiQ-4bit \\
                  --drafter mlx-community/gemma-4-E2B-it-assistant-bf16

        # Bind to all interfaces (e.g. remote dev box; auth is enforced)
        optiq lab --host 0.0.0.0
    """
    import atexit
    import threading
    import webbrowser

    # The Lab web UI is the only part of OptiQ that needs the optional ``lab``
    # extra (flask, argon2-cffi, pyjwt, ...). Serve / convert / eval / lora /
    # kv-cache all run on the core install, so don't make a missing extra look
    # like a broken package — point the user straight at the fix.
    try:
        from .lab import create_app
        from .lab.api_supervisor import ApiSupervisor
        from .lab.config import ensure_lab_dirs
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        click.echo(
            f"[optiq.lab] The OptiQ Lab web UI needs the optional 'lab' extra, "
            f"which is not installed (missing: {missing}).\n"
            f"\n"
            f"    pip install 'mlx-optiq[lab]'\n"
            f"\n"
            f"Only the Lab UI requires it - 'optiq serve', 'convert', 'eval', "
            f"'lora', and 'kv-cache' all work on the core install.",
            err=True,
        )
        sys.exit(1)

    if mtp and drafter:
        click.echo("[optiq.lab] --mtp and --drafter are mutually exclusive",
                   err=True)
        sys.exit(1)

    api_url = f"http://{host}:{api_port}"
    lab_url = f"http://{host}:{lab_port}"

    # 1. Build the API supervisor (manages mlx_lm.server as a child
    # subprocess so users can hot-swap models from the UI).
    paths = ensure_lab_dirs()
    supervisor = ApiSupervisor(
        host=host, port=api_port, log_dir=paths.cache_dir,
        clear_cache_threshold_bytes=clear_cache_threshold_bytes,
    )

    if model:
        click.echo(f"[optiq.lab] starting API server: {api_url}  (model: {model})")
        if mtp:
            click.echo(f"[optiq.lab] MTP speculation: depth={mtp_depth}")
        elif drafter:
            click.echo(f"[optiq.lab] assistant drafter: {drafter}")
        supervisor.start(model=model, mtp=mtp, mtp_depth=mtp_depth,
                         drafter_id=drafter)
    else:
        click.echo("[optiq.lab] no --model given. Load one from the Lab UI "
                   "(Settings → Server).")

    # Ensure the API child is reaped on Lab exit.
    atexit.register(supervisor.stop)

    # 2. Build + run the Lab UI in the main thread
    app = create_app(
        api_url=api_url,  # always — the Lab can be useful even with no model
        mtp_enabled=bool(mtp and model),
        mtp_depth=mtp_depth if mtp else 0,
        drafter_id=drafter if (drafter and model) else None,
        loaded_model=model,
        supervisor=supervisor,
    )
    click.echo(f"[optiq.lab] Lab UI:  {lab_url}")
    click.echo("[optiq.lab] Press Ctrl+C to stop.")

    if not no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(lab_url)).start()

    # use_reloader=False so the API thread doesn't get re-forked
    app.run(host=host, port=lab_port, debug=False, use_reloader=False, threaded=True)


@cli.group()
def cluster():
    """Zero-config multi-Mac inference over Thunderbolt / network.

    `pip install mlx-optiq` on two Macs, `optiq cluster up` on each, and
    `optiq cluster peers` from either sees the others — no IPs, no hostfiles,
    no SSH-key dance. Discovery is macOS Bonjour; Thunderbolt links are
    auto-detected and preferred.
    """



def _resolve_cluster_launch_paths(cwd, python_path):
    """Return (cwd, python) for mlx.launch, both relative to $HOME where possible.

    ``mlx.launch`` executes ``cd <cwd> && <cwd>/<python>`` in each peer's shell,
    which starts in that peer's home. Usernames differ between Macs, so an
    absolute path from this machine is useless there; a home-relative one works
    as long as every peer installed mlx-optiq at the same relative location.
    """
    import os
    import sys
    from pathlib import Path

    cwd = cwd or os.environ.get("OPTIQ_CLUSTER_CWD")
    venv = Path(sys.prefix).resolve()
    if cwd is None:
        home = Path.home().resolve()
        try:
            cwd = str(venv.parent.relative_to(home))
        except ValueError:
            # Not under $HOME (system python, /opt, ...). Fall back to absolute;
            # it only works if the peers share the layout, which we cannot know.
            cwd = str(venv.parent)
    if python_path is None:
        python_path = f"{venv.name}/bin/python3"
    return cwd, python_path


@cluster.command("peers")
@click.option("--timeout", default=4.0, type=float,
              help="Seconds to browse for peers (default 4).")
@click.option("--all", "show_all", is_flag=True,
              help="Include this node itself in the listing.")
def cluster_peers(timeout, show_all):
    """Discover OptiQ nodes advertising on the local links.

    Run `optiq cluster up` on the other Mac(s) first so they advertise.
    """
    from .cluster import discover
    from .cluster.net import local_interfaces

    click.echo("[optiq.cluster] browsing for peers (Bonjour)…")
    peers = discover(timeout=timeout, include_self=True)
    others = [p for p in peers if not p.is_self]
    shown = peers if show_all else others

    # Show this node's links so the user sees what's available.
    tb = [i for i in local_interfaces() if i.is_thunderbolt]
    tb_up = [i for i in tb if i.inet]
    click.echo(f"[optiq.cluster] this node · Thunderbolt links: {len(tb)} "
               f"({'up: ' + ', '.join(i.device for i in tb_up) if tb_up else 'none carrying an IP — is the cable connected?'})")

    if not shown:
        click.echo("[optiq.cluster] no peers found. Start `optiq cluster up` on "
                   "the other Mac(s), and make sure they're on the same "
                   "Thunderbolt/LAN.")
        return

    click.echo(f"\n{'NODE':<26} {'CHIP':<10} {'RAM':>5}  {'LINK':<12} ADDRESS")
    click.echo("-" * 74)
    for p in sorted(shown, key=lambda x: (x.is_self, x.link)):
        icon = "⚡" if p.link == "Thunderbolt" else ("· " if p.is_self else "  ")
        name = (p.hostname.rstrip(".") or p.instance)[:24]
        if p.is_self:
            name += " (self)"
        best = p.preferred_address or next(
            (a for a in p.addresses
             if not a.startswith(("fe80", "::1", "127."))), "")
        click.echo(f"{icon}{name:<24} {p.chip:<10} {p.ram_gb + 'GB':>5}  "
                   f"{p.link:<12} {best}")
    total = sum(int(p.ram_gb) for p in peers if p.ram_gb.isdigit())
    if total:
        click.echo(f"\n[optiq.cluster] {len(peers)} node(s), ~{total} GB combined RAM.")


@cluster.command("serve")
@click.option("--model", "model", required=True,
              help="OptiQ model to shard across the cluster (HF repo or local dir).")
@click.option("--port", default=8100, type=int, help="OpenAI endpoint port (rank 0).")
@click.option("--host", default="127.0.0.1", help="Bind host for the endpoint.")
@click.option("--link", type=click.Choice(["measured", "any"]), default="measured",
              help="'measured' (default): refuse peers slower than the fast-link "
                   "RTT threshold — pipeline inference over a slow link is slower "
                   "than one Mac, so fail loudly instead of silently degrading. "
                   "'any': run over whatever link is available (you accept the "
                   "latency).")
@click.option("--ring-port", default=3720, type=int,
              help="TCP port for the MLX ring backend (bump if bind error 49).")
@click.option("--python", "python_path", default=None,
              help="Per-host python, relative to --cwd. Must have mlx-optiq. "
                   "Defaults to this machine's interpreter.")
@click.option("--cwd", "cwd", default=None,
              help="Per-host working dir, relative to $HOME, that --python is "
                   "resolved against. Defaults to the directory holding this "
                   "machine's virtualenv. Every peer must have mlx-optiq at the "
                   "same path relative to its own home (usernames may differ). "
                   "Override with OPTIQ_CLUSTER_CWD.")
@click.option("--dry-run", is_flag=True,
              help="Print the plan + launch command without launching.")
def cluster_serve(model, port, host, link, ring_port, python_path, cwd, dry_run):
    """Serve an OptiQ model sharded across the discovered cluster (one command).

    Discovers peers over Bonjour, checks the link is fast enough, builds the
    ring hostfile, and launches pipeline-parallel serving with an OpenAI
    endpoint on this node. Point any OpenAI client (or the Lab chat) at it.
    """
    import json as _json
    import os
    import subprocess
    from .cluster import discover
    from .cluster.discovery import build_ring_hostfile
    from .cluster.net import (local_interfaces, measure_rtt_ms, FAST_LINK_RTT_MS)

    click.echo("[optiq.cluster] discovering peers…")
    peers = [p for p in discover(timeout=4.0, include_self=True) if not p.is_self]
    if not peers:
        raise click.ClickException(
            "No peers found. Run `optiq cluster up` on the other Mac(s), and "
            "make sure they're on the same Thunderbolt/LAN.")

    # Measure the real link to each peer and gate on it.
    click.echo(f"\n{'PEER':<26} {'CHIP':<9} {'RAM':>5} {'LINK':<12} {'RTT':>8}")
    click.echo("-" * 66)
    rejected = []
    for p in peers:
        addr = p.preferred_address or ""
        p._rtt = measure_rtt_ms(addr) if addr else float("inf")
        icon = "⚡" if p.link == "Thunderbolt" else "  "
        rtt = f"{p._rtt:.2f}ms" if p._rtt != float("inf") else "unreachable"
        click.echo(f"{icon}{p.hostname.rstrip('.')[:24]:<24} {p.chip:<9} "
                   f"{p.ram_gb + 'GB':>5} {p.link:<12} {rtt:>8}")
        if p._rtt > FAST_LINK_RTT_MS:
            rejected.append(p)

    if rejected and link != "any":
        names = ", ".join(p.hostname.rstrip(".") for p in rejected)
        raise click.ClickException(
            f"\n✗ {names} exceed the fast-link threshold ({FAST_LINK_RTT_MS} ms).\n"
            f"  Pipeline inference over a slow link is SLOWER than one Mac "
            f"(per-token round-trip dominates), so refusing by default.\n"
            f"  Fix: connect Thunderbolt (see the flakiness notes / re-plug the "
            f"cable), or pass --link any to run anyway and accept the latency.")

    # Self's fast-link (Thunderbolt bridge0) IP for the ring.
    tb = [i for i in local_interfaces() if i.is_thunderbolt and i.inet]
    self_ip = tb[0].inet if tb else next(
        (i.inet for i in local_interfaces() if i.inet and i.device == "en0"), "127.0.0.1")
    hostfile = build_ring_hostfile(peers, self_ip)

    # Runner staged at a common /tmp path (home dirs differ across Macs).
    # Resolve the per-host launch paths. mlx.launch runs `cd <cwd>` on every
    # peer's shell (whose cwd is its own $HOME) and then executes <cwd>/<python>,
    # so both must be expressed relative to $HOME and must exist on each Mac.
    # Default them from this machine's interpreter rather than any fixed layout.
    cwd, python_path = _resolve_cluster_launch_paths(cwd, python_path)

    runner = "/tmp/optiq_cluster_serve_run.py"
    # mlx.launch does not forward the environment to remote ranks, so bake the
    # run-reserve into the runner: every rank must agree on it or they compute
    # different shard bounds.
    #
    # No cache limit here. Unlike the SSD-streaming path (I/O-bound, so a small
    # reuse pool costs little), the ring holds its layers resident and decodes
    # from RAM: starving MLX's buffer-reuse pool makes it reallocate activations
    # every op and throughput collapses. Set OPTIQ_CLUSTER_CACHE_MB if a node is
    # so tight that the pool must be bounded.
    headroom = os.environ.get("OPTIQ_CLUSTER_HEADROOM_GB")
    cache_mb = os.environ.get("OPTIQ_CLUSTER_CACHE_MB")
    with open(runner, "w") as f:
        f.write("import os\n")
        if headroom:
            f.write(f"os.environ['OPTIQ_CLUSTER_HEADROOM_GB'] = {headroom!r}\n")
        if cache_mb:
            f.write("import mlx.core as mx\n"
                    f"mx.set_cache_limit({int(cache_mb)} * 1024 * 1024)\n")
        f.write("from optiq.cluster.serve import cluster_serve\n"
                f"cluster_serve({model!r}, host={host!r}, port={port})\n")
    hostfile_path = "/tmp/optiq_cluster_hosts.json"
    with open(hostfile_path, "w") as f:
        _json.dump(hostfile, f, indent=2)

    launch = [f"{python_path.rsplit('/',1)[0]}/mlx.launch", "--backend", "ring",
              "--hostfile", hostfile_path, "--python", python_path,
              "--cwd", cwd, "-p", str(ring_port), runner]
    click.echo(f"\n[optiq.cluster] ring hostfile:\n{_json.dumps(hostfile, indent=2)}")
    click.echo(f"[optiq.cluster] endpoint will be http://{host}:{port}/v1/chat/completions")

    if dry_run:
        click.echo("\n[dry-run] would scp the runner to each peer and launch:\n  "
                   + " ".join(launch))
        return

    # Push runner to each peer, warm ARP, launch.
    for hentry in hostfile[1:]:
        ssh = hentry["ssh"]
        subprocess.run(["scp", "-q", runner, f"{ssh}:{runner}"], check=False)
        subprocess.run(["ssh", ssh, f"ping -c 2 {self_ip} >/dev/null 2>&1"], check=False)
        subprocess.run(["ping", "-c", "2", hentry["ips"][0]],
                       stdout=subprocess.DEVNULL, check=False)
    click.echo("[optiq.cluster] launching cluster serve (Ctrl+C to stop)…\n")
    os.execvp(launch[0], launch)


@cluster.command("up")
@click.option("--port", default=None, type=int,
              help="Port to advertise (default: OptiQ's 51737).")
def cluster_up(port):
    """Advertise this node so other Macs can discover it. Runs until Ctrl+C.

    (Prototype: this advertises the node for discovery. Distributed serving
    that binds discovered peers into a pipeline is the next step.)
    """
    import time as _time
    from .cluster import advertise, node_txt, DEFAULT_PORT

    txt = node_txt()
    port = port or DEFAULT_PORT
    proc = advertise(port=port, txt=txt)
    click.echo(f"[optiq.cluster] advertising this node: chip={txt.get('chip')} "
               f"ram={txt.get('ram')}GB on port {port}")
    click.echo("[optiq.cluster] discoverable from other Macs via "
               "`optiq cluster peers`. Ctrl+C to stop.")
    # A bare SIGTERM (what `pkill` sends) would skip the `finally` and orphan the
    # dns-sd child, leaving a stale mDNS record that shows this Mac twice in the
    # topology and double-counts its RAM. Turn it into a clean shutdown.
    import signal as _signal

    def _bye(signum, frame):
        raise KeyboardInterrupt

    for _sig in (_signal.SIGTERM, _signal.SIGHUP):
        try:
            _signal.signal(_sig, _bye)
        except Exception:
            pass

    try:
        while True:
            _time.sleep(3600)
    except KeyboardInterrupt:
        click.echo("\n[optiq.cluster] stopping advertisement.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    cli()
