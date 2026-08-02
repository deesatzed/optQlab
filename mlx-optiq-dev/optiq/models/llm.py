"""LLM-specific pipeline: sensitivity analysis, optimization, conversion.

Single sensitivity method — calibration-driven exact analysis. The
``reference`` parameter selects the physical execution mode:

  * ``"bf16"`` — load the bf16 base model into RAM and probe per-layer KL
    by swapping each weight between bf16 and a simulate-quantized copy.
    Highest-fidelity signal. Requires the bf16 model to fit in available RAM.

  * ``"uniform_4bit"`` — first build a uniform-4-bit MLX baseline from the
    bf16 source, load that as the running model (~25% of bf16 size), and
    stream bf16 weights off disk one layer at a time to swap in for
    sensitivity probes. Lets 27 B+ models still get a calibration-driven
    signal on a 36 GB Mac.

  * ``"auto"`` (default) — pre-check available RAM. If bf16 fits, use the
    bf16 reference. Otherwise log a warning and fall back to uniform_4bit.

Whichever reference is used, the FINAL OptiQ output is the same mixed-
precision artifact — built freshly by ``mlx_lm.convert`` with our per-layer
bit predicate (e.g. ~4.5 BPW = mix of 4-bit and 8-bit per layer).
"""

import os
import json
import shutil
import gc
from typing import Literal, Optional


def mx_clear_cache_quietly():
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass

from ..core.sensitivity import (
    analyze_sensitivity_exact,
    analyze_sensitivity_static,
    print_sensitivity_report,
    SensitivityResult,
)
from ..core.optimizer import (
    optimize_mixed_precision,
    make_quant_predicate,
    OptimizationResult,
)
from ..backends.mlx_backend import (
    convert_llm_to_mlx,
    convert_llm_uniform,
    convert_llm_static_mixed,
)
from ..calibration.datasets import load_llm_calibration


ReferenceMode = Literal["auto", "bf16", "uniform_4bit"]

# bf16 fits if the on-disk safetensors total is ≤ this fraction of available
# RAM (some headroom for the forward-pass activation footprint and OS reserve).
_BF16_RAM_FIT_FRACTION = 0.70


def _estimate_bf16_size_bytes(local_dir: str) -> int:
    import glob
    return sum(
        os.path.getsize(p)
        for p in glob.glob(os.path.join(local_dir, "*.safetensors"))
    )


def _available_ram_bytes() -> int:
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        return 28 * 1024 * 1024 * 1024  # safe fallback for a 36 GB Mac


def _resolve_reference_mode(
    requested: ReferenceMode, local_dir: str
) -> ReferenceMode:
    """Return the actual reference mode to use.

    ``"auto"`` resolves to ``"bf16"`` if the bf16 weights are likely to fit,
    else ``"uniform_4bit"``. ``"bf16"`` and ``"uniform_4bit"`` pass through
    unchanged (and we let downstream OOM if bf16 was forced on a too-big model).
    """
    if requested != "auto":
        return requested

    bf16_bytes = _estimate_bf16_size_bytes(local_dir)
    avail_bytes = _available_ram_bytes()
    if bf16_bytes <= _BF16_RAM_FIT_FRACTION * avail_bytes:
        print(
            f"  [auto] bf16 ≈ {bf16_bytes / 1024**3:.2f} GB fits in available "
            f"RAM ({avail_bytes / 1024**3:.1f} GB) — using bf16 reference"
        )
        return "bf16"
    print(
        f"  [auto] bf16 ≈ {bf16_bytes / 1024**3:.2f} GB > {_BF16_RAM_FIT_FRACTION:.0%} "
        f"of available RAM ({avail_bytes / 1024**3:.1f} GB) — falling back to "
        f"uniform_4bit reference (slightly weaker calibration signal)"
    )
    return "uniform_4bit"


def run_llm_pipeline(
    model_name: str,
    output_dir: str,
    target_bpw: float = 5.0,
    n_calibration: int = 8,
    candidate_bits: list[int] | None = None,
    group_size: int = 64,
    skip_baselines: bool = False,
    reference: ReferenceMode = "auto",
    calibration_mix: str = "optiq",
    n_floor_per_block: int = 2,
    # Pre-N-tier compat shims; ignored once candidate_bits is set
    low_bits: int = 4,
    high_bits: int = 8,
    method: str = "optiq",
) -> dict:
    """Run the full OptiQ LLM pipeline (MLX-native).

    Steps:
      1. Resolve model snapshot (bf16 weights).
      2. Decide reference mode (bf16 vs uniform-4-bit) per ``reference``.
      3. Build the running model + WikiText-2 calibration.
      4. Per-layer KL sensitivity.
      5. Greedy knapsack → per-layer bit allocation.
      6. Convert to MLX via ``mlx_lm.convert`` with our quant_predicate.

    Returns a dict with output paths + intermediate results.
    """
    if candidate_bits is None:
        candidate_bits = [4, 8]

    os.makedirs(output_dir, exist_ok=True)
    results: dict = {}

    # Step 1: resolve snapshot
    print(f"\n[1/4] Resolving {model_name}…")
    from huggingface_hub import snapshot_download
    if os.path.isdir(model_name):
        local_dir = model_name
    else:
        local_dir = snapshot_download(
            model_name,
            allow_patterns=["*.safetensors", "*.json", "*.txt",
                            "*.model", "tokenizer*", "*.jinja", "README*"],
        )
    print(f"  snapshot at {local_dir}")

    # Step 2: pick reference mode
    mode = _resolve_reference_mode(reference, local_dir)

    # Step 3: build running model + calibration
    #
    # We use load_model+load_tokenizer directly with strict=False instead of
    # the higher-level ``mlx_lm.utils.load`` because some bf16 sources
    # publish weight tensors the current mlx-lm class doesn't have weight
    # slots for. Concrete case: Google's google/gemma-4-e2b-it bf16
    # checkpoint ships per-layer k_proj/v_proj/k_norm for layers 15-34,
    # but current mlx-lm gemma4 treats those as KV-shared layers that read
    # K/V from donor layers and therefore has no weight slots for them.
    # strict=True (the load helper's default) raises ``ValueError: Received
    # N parameters not in model`` and aborts the convert. strict=False
    # drops the extras silently, which is correct because the new
    # architecture doesn't use them anyway. The resulting quant has only
    # the tensors the new class expects and loads cleanly downstream.
    from mlx_lm.utils import load_model, load_tokenizer
    from pathlib import Path

    # Teach mlx-lm to build any custom OptiQ decoder this base needs before any
    # load_model / mlx_lm.convert runs in this process. For mistral4 the base's
    # ``mistral3`` wrapper otherwise falls back to ``llama`` and drops all 128
    # experts. Idempotent + arch-gated → a no-op for every other family.
    from optiq.models import mistral4, llada2
    mistral4.install()
    llada2.install()

    def _load_bf16_or_quant(d: str, lazy: bool = False):
        m, _cfg = load_model(Path(d), lazy=lazy, strict=False)
        t = load_tokenizer(Path(d),
                           tokenizer_config_extra={"trust_remote_code": True})
        return m, t

    baseline_dir_for_cleanup: Optional[str] = None
    bf16_source_dir: Optional[str] = None
    if method == "static":
        # ``static`` (structural rules) reads only layer names + shapes, never
        # weight values — so load LAZILY (mmap, not materialized). This is what
        # lets it allocate a 100 B+ base on a 36 GB Mac; lazy=False would try to
        # resident-load the whole bf16 model and OOM.
        running_model, tokenizer = _load_bf16_or_quant(local_dir, lazy=True)
        calibration_fn = None
    elif mode == "bf16":
        running_model, tokenizer = _load_bf16_or_quant(local_dir)
    else:
        baseline_dir = os.path.join(output_dir, "_uniform_4bit_baseline")
        # Reuse a pre-placed uniform-4-bit baseline if one is already present
        # (e.g. a published mlx-community 4-bit dropped in by hand, or a prior
        # run). Building it streams/quantizes the full bf16 base, which is the
        # memory-heaviest step and can swap-stall on large models, so skipping
        # it when a valid baseline already exists is both faster and safer.
        _has_baseline = os.path.exists(
            os.path.join(baseline_dir, "model.safetensors.index.json")
        ) or os.path.exists(os.path.join(baseline_dir, "model.safetensors"))
        if _has_baseline:
            print(f"  reusing existing uniform-4-bit baseline at {baseline_dir} (skip build)")
        else:
            if os.path.exists(baseline_dir):
                shutil.rmtree(baseline_dir)
            print(f"  building uniform-4-bit baseline at {baseline_dir}…")
            convert_llm_uniform(
                model_name, baseline_dir, bits=4, group_size=group_size,
            )
        running_model, tokenizer = _load_bf16_or_quant(baseline_dir)
        bf16_source_dir = local_dir
        baseline_dir_for_cleanup = baseline_dir

    if method == "optiq":
        calibration_fn = load_llm_calibration(
            tokenizer,
            n_samples=max(n_calibration * 2, 16),
            # NOTE: dhara is TRI-MODE — see the _dhara_trimode_calibration wrap below.
            # A causal-only calibration silently destroys its block-diffusion path.
            # 512 token-per-sample is the memory-safe sweet spot for 27B+ on
            # a 36 GB Mac during the bf16-streaming sensitivity probes:
            # longer sequences inflate per-probe activation memory enough to
            # trigger Jetsam OOM-kills.
            seq_len=512,
            mix=calibration_mix,
        )

        # dhara is tri-mode (causal AR + block-diffusion from one set of weights).
        # Calibrating only on the causal forward means sensitivity never observes the
        # diffusion path, so the optimizer crushes the layers only diffusion needs —
        # yielding a build with a pristine AR path and a diffusion path that collapses
        # into repetition. Probe BOTH modes so the bit allocation preserves both.
        # (DiffusionGemma solves the same problem in vlm/diffusion_gemma/calibration.py.)
        from optiq.mlx_lm_patches.dhara_calibration import (
            is_dhara, make_dhara_calibration,
        )
        if is_dhara(running_model):
            print("  dhara detected → tri-mode calibration (causal AR + block-diffusion)")
            calibration_fn = make_dhara_calibration(calibration_fn, running_model)

    # Step 4: sensitivity
    checkpoint_path = os.path.join(output_dir, "sensitivity_checkpoint.json")
    if method == "static":
        print(f"\n[2/4] Sensitivity analysis (method=static · structural rules, bits={candidate_bits})…")
        sensitivity_results = analyze_sensitivity_static(
            running_model,
            candidate_bits=candidate_bits,
            group_size=group_size,
        )
    else:
        print(f"\n[2/4] Sensitivity analysis (reference={mode}, bits={candidate_bits})…")
        sensitivity_results = analyze_sensitivity_exact(
            running_model,
            calibration_fn,
            candidate_bits=candidate_bits,
            group_size=group_size,
            n_calibration=n_calibration,
            checkpoint_path=checkpoint_path,
            bf16_source_dir=bf16_source_dir,
        )
    print_sensitivity_report(sensitivity_results)
    results["sensitivity"] = sensitivity_results

    del running_model
    import gc
    gc.collect()
    if baseline_dir_for_cleanup is not None:
        try:
            shutil.rmtree(baseline_dir_for_cleanup)
        except Exception:
            pass

    # Step 5: optimize
    print("\n[3/4] Optimizing multi-tier bit allocation...")
    opt_result = optimize_mixed_precision(
        sensitivity_results,
        target_bpw=target_bpw,
        candidate_bits=candidate_bits,
        group_size=group_size,
        n_floor_per_block=n_floor_per_block,
    )
    results["optimization"] = opt_result

    # Step 5b: optional end-to-end-KL iterative refinement of the allocation.
    # Only supported in bf16 reference mode (uniform_4bit mode would need to
    # stream bf16 weights from disk per swap candidate, which is too slow
    # for the iterative loop).
    # Step 6: convert
    print("\n[4/4] Converting to MLX...")

    _dirname = "static_mixed" if method == "static" else "optiq_mixed"
    _refmode = "structural_rules" if method == "static" else mode
    optiq_path = os.path.join(output_dir, _dirname)
    predicate = make_quant_predicate(opt_result)
    metadata = {
        "method": f"{method}_mixed_precision",
        "base_model": model_name,
        "reference": _refmode,
        "target_bpw": target_bpw,
        "achieved_bpw": opt_result.achieved_bpw,
        "n_high_bits": opt_result.n_high_bits,
        "n_low_bits": opt_result.n_low_bits,
        "threshold": opt_result.threshold,
        "per_layer": {
            c.layer_name: {"bits": c.bits, "group_size": c.group_size}
            for c in opt_result.configs
        },
    }
    convert_llm_to_mlx(
        model_name,
        optiq_path,
        quant_predicate=predicate,
        q_bits=low_bits,
        q_group_size=group_size,
        metadata=metadata,
    )

    # mlx-lm's convert drops the vision/audio towers, so the artifact is
    # language-only until they are put back. Do it here, in the pipeline, rather
    # than leaving it to whoever remembers to run a script -- that is how VLMs
    # came to be published with no tower. (MTP needs no call: convert_llm_to_mlx
    # already preserves it, and knows the host quant params to match.)
    from ..sidecars import attach_sidecars
    results["sidecars"] = attach_sidecars(model_name, optiq_path, mtp=False)
    results["optiq_path"] = optiq_path

    # The fast ``static`` method skips the uniform-4-bit + static byproducts
    # (each a full quantize of the base) that would defeat its purpose; only
    # the default optiq path emits comparison artifacts.
    if not skip_baselines and method == "optiq":
        uniform_path = os.path.join(output_dir, "uniform_4bit")
        convert_llm_uniform(model_name, uniform_path, bits=4, group_size=group_size)
        results["uniform_path"] = uniform_path

        try:
            static_path = os.path.join(output_dir, "static_mixed")
            convert_llm_static_mixed(model_name, static_path, recipe="mixed_3_6")
            results["static_path"] = static_path
        except Exception as e:
            print(f"  Static mixed recipe failed (may not support this model): {e}")

    return results


def load_sensitivity_cache(cache_path: str) -> Optional[list[SensitivityResult]]:
    if not os.path.exists(cache_path):
        return None
    with open(cache_path) as f:
        data = json.load(f)
    return [
        SensitivityResult(
            layer_name=d["layer_name"],
            sensitivities={int(k): v for k, v in d["sensitivities"].items()},
            param_count=d["param_count"],
        )
        for d in data
    ]


def save_sensitivity_cache(results: list[SensitivityResult], cache_path: str):
    data = [
        {
            "layer_name": r.layer_name,
            "sensitivities": {str(k): v for k, v in r.sensitivities.items()},
            "param_count": r.param_count,
        }
        for r in results
    ]
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
