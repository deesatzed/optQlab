"""OptiQ mixed-precision pipeline for diffusion LLMs (DiffusionGemma).

The sibling of ``models/llm.py`` for masked/block-diffusion models. It chains the
real OptiQ components — no ad-hoc re-implementation:

  1. Resolve the bf16 source.
  2. Build a uniform-4-bit baseline          (``vlm.diffusion_gemma.convert``)
  3. Load it as the running model            (``vlm.diffusion_gemma.loader``)
  4. Masked-canvas calibration over the
     denoising schedule                      (``vlm.diffusion_gemma.calibration``)
  5. Streaming per-layer KL sensitivity      (``core.sensitivity``, uniform_4bit
                                              reference — bf16 is streamed off
                                              disk one layer at a time)
  6. Greedy-knapsack bit allocation          (``core.optimizer``)
  7. Final mixed-precision convert           (``vlm.diffusion_gemma.convert``)

Why this is not the LLM pipeline: mlx-lm cannot load ``diffusion_gemma`` (there
is no such module in ``mlx_lm.models``), and a diffusion model's meaningful
forward is a *denoising step* over a masked canvas, not a causal pass over
tokens. Calibrating a diffusion model on causal forwards means the optimizer
never observes the path the model actually runs, so it crushes the layers only
denoising depends on. (The same failure was observed on dhara and is what its
tri-mode calibration exists to prevent.)

Note the diffusion stack here is fully vendored — this imports no ``mlx_vlm``.
"""

from __future__ import annotations

import json
import os
import time

from ..core.optimizer import (
    compute_bpw,
    make_quant_predicate,
    optimize_mixed_precision,
)
from ..core.sensitivity import analyze_sensitivity_exact, print_sensitivity_report

# NOTE: nothing from ``optiq.vlm`` is imported at module scope. Importing that
# package eagerly pulls in the three vision frontends, each of which imports
# Pillow at module level — so a plain `optiq convert Qwen/...` on a TEXT model
# would die with `No module named 'PIL'` unless the user had installed the vision
# extra. `optiq convert` imports `is_diffusion_model` from here purely to decide
# how to route, and that decision must not cost a dependency the user does not
# need. The heavy imports live inside `run_diffusion_pipeline`.

DEFAULT_DENOISE_STEPS = 3
DIFFUSION_MODEL_TYPES = ("diffusion_gemma",)


def _peek_model_type(model_path: str) -> str | None:
    """Read ``model_type`` from config.json alone — no weights, no vlm imports."""
    if os.path.isdir(model_path):
        cfg_path = os.path.join(model_path, "config.json")
        if not os.path.isfile(cfg_path):
            return None
        with open(cfg_path) as f:
            return json.load(f).get("model_type")

    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(model_path, "config.json")) as f:
        return json.load(f).get("model_type")


def is_diffusion_model(model_path: str) -> bool:
    """True when ``model_path``'s config declares a diffusion architecture.

    Cheap: reads config.json only (no weights), so ``optiq convert`` can route
    before it downloads anything big — and without importing the vision stack.
    """
    try:
        return _peek_model_type(model_path) in DIFFUSION_MODEL_TYPES
    except Exception:
        return False


def run_diffusion_pipeline(
    model_name: str,
    output_dir: str,
    target_bpw: float = 4.4,
    n_calibration: int = 12,
    candidate_bits: list[int] | None = None,
    group_size: int = 64,
    skip_baselines: bool = False,
    n_denoise_steps: int = DEFAULT_DENOISE_STEPS,
    calibration_prompts: list[str] | None = None,
) -> dict:
    """Run the full OptiQ pipeline on a diffusion LLM.

    Returns a dict of output paths + the bit allocation, mirroring
    ``run_llm_pipeline``.
    """
    # Imported here, not at module scope: see the note at the top of this file —
    # routing a text-only convert must not require the vision extra.
    from ..vlm.diffusion_gemma.calibration import make_diffusion_calibration
    from ..vlm.diffusion_gemma.convert import convert_diffusion_gemma, is_vision_layer
    from ..vlm.diffusion_gemma.loader import _resolve_dir, load_diffusion_gemma

    candidate_bits = sorted(candidate_bits or [4, 8])
    os.makedirs(output_dir, exist_ok=True)

    uniform_path = os.path.join(output_dir, "uniform_4bit")
    optiq_path = os.path.join(output_dir, "optiq_mixed")
    checkpoint_path = os.path.join(output_dir, "sensitivity_checkpoint.json")

    print(f"\n[1/6] Resolving bf16 source: {model_name}")
    bf16_dir = _resolve_dir(model_name)
    print(f"  bf16: {bf16_dir}")

    # The uniform-4-bit build is both the running model for calibration AND the
    # reference the sensitivity sweep probes against, so it is never a throwaway
    # even with --skip-baselines.
    if os.path.exists(os.path.join(uniform_path, "config.json")):
        print(f"\n[2/6] uniform-4-bit baseline exists → {uniform_path}")
    else:
        print(f"\n[2/6] Building uniform-4-bit baseline → {uniform_path}")
        convert_diffusion_gemma(bf16_dir, uniform_path, bits=4, group_size=group_size)

    print("\n[3/6] Loading the baseline as the running model")
    from mlx_lm.utils import load_tokenizer
    from pathlib import Path

    model, _config = load_diffusion_gemma(uniform_path)
    tokenizer = load_tokenizer(Path(uniform_path))

    print(f"\n[4/6] Masked-canvas calibration "
          f"({n_denoise_steps} denoising steps per prompt)")
    calibration_fn = make_diffusion_calibration(
        model, tokenizer,
        prompts=calibration_prompts,
        n_denoise_steps=n_denoise_steps,
    )

    print(f"\n[5/6] Per-layer KL sensitivity (bf16 streamed from disk)")
    t0 = time.time()
    results = analyze_sensitivity_exact(
        model, calibration_fn,
        candidate_bits=candidate_bits,
        group_size=group_size,
        n_calibration=n_calibration,
        checkpoint_path=checkpoint_path,
        bf16_source_dir=bf16_dir,
    )
    print(f"  {len(results)} layers in {time.time() - t0:.0f}s")

    # The towers stay bf16 (OptiQ's policy for every VLM family), so they must not
    # enter the knapsack: the calibration is text-only, they score KL == 0, and the
    # optimizer would hand them the floor bit-width on a signal it never measured
    # while their params skewed the bpw budget the language tower is spending.
    lang_results = [r for r in results if not is_vision_layer(r.layer_name)]
    n_vision = len(results) - len(lang_results)
    if n_vision:
        print(f"  {n_vision} vision/audio tower layers held at bf16 "
              f"(excluded from the bit budget)")
    print_sensitivity_report(lang_results, top_n=15)

    print(f"\n[6/6] Knapsack @ {target_bpw} bpw → {optiq_path}")
    opt = optimize_mixed_precision(
        lang_results, target_bpw=target_bpw, candidate_bits=candidate_bits
    )
    achieved = compute_bpw(opt.configs)
    hist: dict[int, int] = {}
    for c in opt.configs:
        hist[c.bits] = hist.get(c.bits, 0) + 1
    print(f"  achieved {achieved:.3f} bpw; bit distribution {dict(sorted(hist.items()))}")

    metadata = {
        "method": "optiq",
        "model_type": "diffusion",
        "target_bpw": target_bpw,
        "achieved_bpw": achieved,
        "candidate_bits": candidate_bits,
        "n_high_bits": opt.n_high_bits,
        "n_low_bits": opt.n_low_bits,
        "calibration": {
            "kind": "masked-canvas",
            "n_denoise_steps": n_denoise_steps,
            "n_samples": n_calibration,
        },
        "vision_towers": "bf16",
        "n_vision_layers_bf16": n_vision,
        "per_layer": {
            c.layer_name: {"bits": c.bits, "group_size": c.group_size}
            for c in opt.configs
        },
    }

    convert_diffusion_gemma(
        bf16_dir, optiq_path,
        group_size=group_size,
        quant_predicate=make_quant_predicate(opt),
    )
    with open(os.path.join(optiq_path, "optiq_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    out = {
        "optiq_path": optiq_path,
        "target_bpw": target_bpw,
        "achieved_bpw": achieved,
        "candidate_bits": candidate_bits,
        "per_layer": metadata["per_layer"],
        "n_layers": len(results),
    }
    if not skip_baselines:
        out["uniform_path"] = uniform_path

    print(f"\nDONE. OptiQ diffusion quant → {optiq_path}")
    return out
