"""Decode settings for the benchmark harness.

Every benchmark used to build ``make_sampler(temp=0.0)`` and pass no logits
processors. Greedy is the right call for a reproducible benchmark: it removes
sampling noise, so two quants of the same model differ only by their weights.

It is not the right call for a model that needs a repetition penalty to stay
coherent. dhara-250m is one. Under bare greedy it answers a GSM8K question with
"we need to find the number of clips ... $1 = 1 + 1 + 1 + 1 + 1 + ..." and never
stops; with ``repetition_penalty=1.3`` (what the reference dhara-chat Space
decodes with) the same weights produce structured reasoning.

So the six benchmarks were scoring a decode loop, not the model. Every dhara
build scored the same ~8.3 because they all looped, and we read that flat line as
"a 250M model has no benchmark headroom" when it actually meant "the harness
breaks this model". The Capability Score cannot compare two quants that are both
degenerate.

The penalty cannot be read off the model. dhara's ``generation_config.json`` is
empty apart from an eos id, so the 1.3 lives only in the Space's app.py. Hence
this table: an architecture whose intended decode config is not in its own repo
declares it here, once, and every benchmark picks it up.

Greedy still holds. A repetition penalty is orthogonal to temperature (the Space
runs ``do_sample=False`` with the penalty on), so runs stay deterministic and
quant-to-quant comparisons stay fair.
"""

from __future__ import annotations

# model_type -> repetition_penalty the model is meant to be decoded with.
#
# Only for architectures that do NOT declare it in generation_config.json. If a
# model ships a penalty in its own config, honour that instead of adding it here.
# Most models want 1.0 (off): a 9B does not loop at greedy, and a needless penalty
# would distort its benchmark answers.
ARCH_REPETITION_PENALTY: dict[str, float] = {
    # https://huggingface.co/spaces/codelion/dhara-chat app.py:
    #   GEN_GREEDY = dict(do_sample=False, repetition_penalty=1.3, ...)
    "dhara_ar": 1.3,
}

NO_PENALTY = 1.0


def _model_type(model) -> str | None:
    return getattr(model, "model_type", None) or getattr(
        getattr(model, "args", None), "model_type", None
    )


def repetition_penalty_for(model, override: float | None = None) -> float:
    """The penalty this model should be benchmarked with."""
    if override is not None:
        return override
    return ARCH_REPETITION_PENALTY.get(_model_type(model), NO_PENALTY)


def resolve_decode(model, repetition_penalty: float | None = None):
    """Return ``(sampler, logits_processors)`` for a benchmark run.

    Greedy always. Processors only when the model actually needs a penalty, so
    every model that was fine before is bit-for-bit unaffected.
    """
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    sampler = make_sampler(temp=0.0)
    penalty = repetition_penalty_for(model, repetition_penalty)
    if penalty == NO_PENALTY:
        return sampler, None
    return sampler, make_logits_processors(repetition_penalty=penalty)


def load_tolerant(model_path: str):
    """Load a model for benchmarking, tolerating a checkpoint that carries more
    weights than the text model has slots for.

    Several published checkpoints ship weights mlx-lm's text model has no module
    for, and a strict load rejects the whole thing:

      * gemma-4 unified quants keep `vision_embedder.*` inline. Loading one as a
        text model raises "Received 11 parameters not in model".
      * gemma-4 bf16 with KV-shared layers ships donor k/v_proj weights.

    The model is complete without them, so the strict load over-rejects.

    gsm8k.py and eval/kl.py each carried their own copy of this fallback. The
    other four benchmarks did not, so `optiq eval --task gsm8k` worked on
    gemma-4-12B-it-qat-4bit while `--task bfcl` died on it. One implementation
    now, used by all six.
    """
    import os
    from pathlib import Path

    from mlx_lm import load

    try:
        return load(model_path)
    except ValueError as e:
        if "not in model" not in str(e):
            raise
        from mlx_lm.utils import load_model, load_tokenizer

        local = model_path
        if not os.path.isdir(local):
            from huggingface_hub import snapshot_download
            local = snapshot_download(model_path)
        print("  (retrying non-strict: the checkpoint carries weights the text "
              "model has no slots for)")
        model, _ = load_model(Path(local), strict=False)
        return model, load_tokenizer(Path(local))


def seed_rng(seed: int) -> None:
    """Pin MLX's global RNG so a benchmark is reproducible.

    Autoregressive greedy decode is deterministic and needs none of this. A
    *diffusion* decode is not: DiffusionGemma denoises a canvas of random token
    ids (`_diffusion_initialize_canvas` -> `mx.random.randint`) and samples with
    `mx.random.categorical`. The benchmark harness never seeded either, so two
    runs of the same weights on the same code produced different scores.

    Measured on diffusiongemma-26B-A4B-it-OptiQ-4bit, HumanEval: 87.2 on one run,
    83.5 on another. Six problems' difference out of 164, from the dice. Every
    diffusion number we published was unreproducible, and any delta between two
    diffusion quants was confounded by canvas noise.

    The calibration path has seeded its canvas since the sweep learned to resume
    (an unseeded capture makes a checkpointed sensitivity run score its remaining
    layers against a different input). The benchmarks never got the same fix.
    """
    import mlx.core as mx

    mx.random.seed(int(seed))


# ── keeping the benchmarks off the machine's throat ───────────────────────────
# MLX does not free a GPU buffer when you drop the array; it parks it in a reuse
# pool. The pool's default ceiling is the whole machine (36.7 GB on a 36 GB M3),
# so a benchmark that runs a thousand differently-shaped prompts grows it without
# bound. Measured on dhara-250m MMLU: active memory 0.00 GB, process RSS 1.03 GB,
# and an MLX cache of 30.6 GB. Metal buffers do not show up in RSS, so `ps` says
# the process is tiny while the machine swaps and the kernel kills the run.
#
# serve, the trainer, the sensitivity sweep, moe_stream and the Lab all cap or
# drain this pool. The benchmark harness was the one path that never did.
_clear_threshold: int | None = None


def reclaim() -> bool:
    """Drop MLX's reuse pool if it has grown past the threshold.

    Call once per benchmark question. Cheap when there is nothing to do (a
    ``get_cache_memory()`` read), and the pool refills from the same buffers on
    the next forward, so throughput is unaffected.
    """
    global _clear_threshold
    from ..lab.mlx_cleanup import default_threshold_bytes, maybe_clear

    if _clear_threshold is None:
        _clear_threshold = default_threshold_bytes()   # 10% of RAM, floor 1 GB
    return maybe_clear(_clear_threshold)
