"""Masked-canvas calibration for DiffusionGemma sensitivity analysis.

OptiQ's sensitivity engine measures KL on per-position logits for a set of
calibration forwards. For an autoregressive LM those forwards are plain
``model(input_ids)`` calls. DiffusionGemma is a masked/block-diffusion model:
its meaningful forward is a single *denoising step* — the prompt is KV-cached in
the encoder, a block of canvas tokens (mostly ``[Mask]``) is fed to the decoder,
and the model predicts the canvas logits.

Reproducing that setup by hand (encoder prefill + canvas init + attention-mask
construction) would duplicate ~500 lines of the vendored decode loop. Instead we
drive the real loop (``stream_diffusion_generate``) and intercept the
``Model.__call__`` arguments at each denoising step. Replaying
``model(**canvas_kwargs)`` reproduces the canvas logits bit-for-bit (the forward
does not mutate the captured KV cache — re-verified: the cache ``offset`` is
unchanged across steps and across replays, and replayed logits are identical to
0 ulp), so the captured kwargs are stable, self-contained calibration samples
the standard engine can probe.

**Sampling the denoising schedule.** A diffusion model does not face one input
distribution, it faces a *schedule*: the first step sees a near-fully-masked
canvas ("write something from nothing"), later steps see a partially committed
canvas ("refine, given what is already there"). Calibrating only on the first
step measures sensitivity at one extreme of that schedule. We therefore capture
the canvas at several denoising steps per prompt and interleave the samples, so
the bit allocation sees the whole schedule. (dhara's tri-mode calibration probes
mid-refinement for the same reason — see ``mlx_lm_patches/dhara_calibration``.)

The result is a ``calibration_fn`` of the exact shape
``analyze_sensitivity_exact`` expects: a zero-arg callable returning
``[((), canvas_kwargs), ...]``, each consumed as ``model(*args, **kwargs)``.

Two known approximations, both measured on diffusiongemma-26B-A4B-it-OptiQ-4bit
so the next reader does not have to re-derive them:

* **The prompt KV cache is frozen.** ``analyze_sensitivity_exact`` calls
  ``calibration_fn`` once and replays the samples for every layer x bit probe, so
  the prefill is never re-run against the quantized layer: error a layer
  introduces while *encoding the prompt* is invisible to the KL. Crushing 14
  decoder layers to 2-bit and comparing the frozen-cache KL against a KL whose
  cache was recomputed under the same quantized layer: the frozen cache
  understates by a median of 1.1x (range 0.6x-1.8x, 10/14 understated). The worst
  cases are the projections that *write* the cache (v_proj 1.8x). Rank
  correlation stays high (Spearman 0.89) so most of the ordering survives, but
  the top-3 most-sensitive layers — the ones the knapsack actually acts on —
  agree only 1/3. Fixing it means re-running the prefill per probe (~2x sweep
  cost) and an invasive change to the shared engine, so it is left as a known
  bias rather than paid for on every convert.

* **Calibration is text-only**, so the vision tower's 164 layers score KL == 0
  and land at the floor bit-width by default rather than by measurement. Fine
  for text; if DiffusionGemma's image path matters, calibrate with image prompts
  (or keep the towers in a bf16 sidecar, as OptiQ's other VLMs do).
"""

from __future__ import annotations

import mlx.core as mx

from .._mlxvlm.generate.diffusion import stream_diffusion_generate

# A small domain mix (prose / code / reasoning / instruct) — enough to surface
# per-layer precision sensitivity without a long sweep. Mirrors the spirit of
# OptiQ's bundled ``optiq.jsonl`` LLM calibration mix.
DEFAULT_CALIBRATION_PROMPTS = [
    "Explain how photosynthesis converts sunlight into chemical energy.",
    "Write a Python function that reverses a singly linked list in place.",
    "A train leaves at 3pm going 60 mph; another leaves at 4pm going 80 mph "
    "from the same station. When does the second catch the first?",
    "Summarize the main causes of the First World War in a few sentences.",
]

# How many denoising steps to sample per prompt. 3 spans the schedule (noisy ->
# partly committed -> mostly committed) without a long sweep.
DEFAULT_DENOISE_STEPS = 3

# DiffusionGemma seeds its canvas with **random token ids** (see
# ``_diffusion_initialize_canvas``: ``mx.random.randint(0, vocab_size, ...)``) —
# it is a discrete-diffusion model denoising from noise, not one unmasking a
# ``[Mask]`` sentinel. So an unseeded capture returns a different canvas every
# call, which would make sensitivity irreproducible: the sweep checkpoints and
# resumes, and a resumed run would score the remaining layers against a
# *different* input than the layers already done, leaving the knapsack to
# allocate bits from an inconsistent table. Seed the canvas per prompt.
# (dhara's calibration seeds for the same reason.)
DEFAULT_CANVAS_SEED = 0


class _Captured(Exception):
    """Sentinel raised to unwind the decode loop once kwargs are captured."""


def _normalize_ids(ids):
    """Normalize a list[int] / BatchEncoding / [[int]] to a flat list of ids."""
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if len(ids) and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return list(ids)


def mask_ratio(model, canvas_kwargs: dict) -> float | None:
    """Fraction of the canvas still holding the ``[Mask]`` sentinel.

    ``None`` for DiffusionGemma, which denoises from *random tokens* rather than
    unmasking a sentinel — there is no mask id to count. Kept for masked-diffusion
    checkpoints that do declare ``mask_token_id``. Diagnostic only.
    """
    mask_id = getattr(getattr(model, "config", None), "mask_token_id", None)
    canvas = canvas_kwargs.get("canvas_ids")
    if mask_id is None or canvas is None or canvas.size == 0:
        return None
    return float(int((canvas == int(mask_id)).sum()) / canvas.size)


def canvas_drift(first: dict, other: dict) -> float | None:
    """Fraction of canvas positions that changed since the first denoising step.

    The usable progress signal for a denoise-from-noise model: 0.0 at step 1, and
    rising as the model commits tokens over the schedule.
    """
    a, b = first.get("canvas_ids"), other.get("canvas_ids")
    if a is None or b is None or a.size == 0 or a.shape != b.shape:
        return None
    return float(int((a != b).sum()) / a.size)


def _drive_steps(model, tokenizer, input_ids, n_steps: int) -> list[dict]:
    """Run the real decode loop for ``n_steps`` denoising steps, capturing the
    ``Model.__call__`` kwargs at each one.

    The captured samples share one KV-cache object (the prompt's), which is safe:
    the canvas forward reads the cache but never appends to it, so replaying any
    captured step is independent of the others.
    """
    captured: list[dict] = []
    model_cls = type(model)
    original_call = model_cls.__call__

    def capturing(self, *args, **kwargs):
        out = original_call(self, *args, **kwargs)
        # The decoder pass is the one carrying canvas_ids; the encoder prefill
        # passes input_ids instead.
        if kwargs.get("canvas_ids") is not None:
            captured.append(dict(kwargs))
            if len(captured) >= n_steps:
                raise _Captured
        return out

    model_cls.__call__ = capturing
    try:
        for _ in stream_diffusion_generate(
            model, tokenizer, tokenizer, input_ids, None, None,
            max_tokens=16, skip_special_token_ids=set(), temperature=0.0,
            max_denoising_steps=n_steps, diffusion_sampler="confidence-threshold",
        ):
            break
    except _Captured:
        pass
    finally:
        model_cls.__call__ = original_call

    return captured


def capture_canvas_from_ids(model, tokenizer, input_ids) -> dict | None:
    """Drive one denoising step on already-tokenized ``input_ids`` and capture
    the ``Model.__call__`` kwargs (cache / canvas_ids / masks)."""
    if not isinstance(input_ids, mx.array):
        input_ids = mx.array(_normalize_ids(input_ids), dtype=mx.int32)[None]
    steps = _drive_steps(model, tokenizer, input_ids, 1)
    return steps[0] if steps else None


def capture_canvas_kwargs(model, tokenizer, prompt: str,
                          n_steps: int = 1) -> dict | None:
    """Drive ``n_steps`` denoising steps and return the LAST step's kwargs.

    With the default ``n_steps=1`` this is the initial, near-fully-masked canvas.
    """
    steps = capture_canvas_steps(model, tokenizer, prompt, n_steps=n_steps)
    return steps[-1] if steps else None


def capture_canvas_steps(model, tokenizer, prompt: str,
                         n_steps: int = DEFAULT_DENOISE_STEPS,
                         seed: int | None = DEFAULT_CANVAS_SEED) -> list[dict]:
    """Capture the canvas kwargs at each of the first ``n_steps`` denoising steps.

    ``seed`` pins the random initial canvas so the samples — and therefore the
    sensitivity scores — are reproducible across runs and across a
    checkpoint/resume. Pass ``None`` to leave the global RNG alone.
    """
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True
    )
    input_ids = mx.array(_normalize_ids(ids), dtype=mx.int32)[None]
    if seed is not None:
        mx.random.seed(int(seed))
    return _drive_steps(model, tokenizer, input_ids, n_steps)


def _interleave(n_prompts: int, n_steps: int) -> list[tuple[int, int]]:
    """Order the (prompt, step) grid so that ANY prefix spans both axes.

    ``analyze_sensitivity_exact`` truncates to ``n_calibration`` samples, so a
    naive prompt-major order would hand it three canvases from one prompt, and a
    step-major order would hand it one denoising step across three prompts.
    Walking the grid diagonally keeps a short prefix diverse in both.
    """
    order = []
    for k in range(n_prompts * n_steps):
        p = k % n_prompts
        s = (k // n_prompts + p) % n_steps
        order.append((p, s))
    # The diagonal repeats when gcd(n_prompts, n_steps) > 1; drop duplicates and
    # append whatever the walk missed so the grid is still covered exactly once.
    seen, out = set(), []
    for pair in order:
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    for p in range(n_prompts):
        for s in range(n_steps):
            if (p, s) not in seen:
                seen.add((p, s))
                out.append((p, s))
    return out


def make_diffusion_calibration(model, tokenizer, prompts: list[str] | None = None,
                               n_denoise_steps: int = DEFAULT_DENOISE_STEPS,
                               seed: int = DEFAULT_CANVAS_SEED):
    """Build a ``calibration_fn`` for ``analyze_sensitivity_exact``.

    Samples the denoising schedule: for each prompt the canvas is captured at
    ``n_denoise_steps`` points along the trajectory (noisy → committed), and the
    ``prompt × step`` grid is interleaved so a truncated prefix still spans both.
    Each prompt's canvas is seeded, so the whole sample set is reproducible.

    Returns a zero-arg callable yielding ``[((), canvas_kwargs), ...]``; each
    sample replays as ``model(**canvas_kwargs)`` → ``LanguageModelOutput`` whose
    ``.logits`` the engine compares via KL.
    """
    prompts = prompts or DEFAULT_CALIBRATION_PROMPTS
    n_denoise_steps = max(1, int(n_denoise_steps))

    def calibration_fn():
        # step_captures[p] = [kwargs at step 1, step 2, ...] for prompt p.
        # A distinct seed per prompt: reproducible, but not the same canvas noise
        # for every prompt.
        step_captures: list[list[dict]] = [
            capture_canvas_steps(model, tokenizer, prompt,
                                 n_steps=n_denoise_steps, seed=seed + i)
            for i, prompt in enumerate(prompts)
        ]

        samples: list[tuple[tuple, dict]] = []
        for p, s in _interleave(len(prompts), n_denoise_steps):
            steps = step_captures[p]
            if s < len(steps):
                samples.append(((), steps[s]))

        if not samples:
            raise RuntimeError(
                "DiffusionGemma calibration captured no canvas forwards; "
                "check the vendored decode loop / tokenizer."
            )

        drifts = [
            d for d in (
                canvas_drift(step_captures[p][0], step_captures[p][s])
                for p, s in _interleave(len(prompts), n_denoise_steps)
                if s < len(step_captures[p])
            ) if d is not None
        ]
        span = f"; canvas drift {min(drifts):.0%}-{max(drifts):.0%}" if drifts else ""
        print(f"  [diffusion-calib] {len(samples)} samples over {len(prompts)} "
              f"prompts x {n_denoise_steps} denoising steps (seed={seed}){span}")
        return samples

    return calibration_fn
