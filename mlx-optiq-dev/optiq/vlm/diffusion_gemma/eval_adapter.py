"""Make OptiQ's eval harness work on DiffusionGemma.

Every eval module (gsm8k, mmlu, bfcl, humaneval, ifeval, hashhop) does
``from mlx_lm import load, generate`` and then ``load(path)`` + ``generate(...)``.
DiffusionGemma is not mlx-lm-loadable and decodes via the vendored masked-
diffusion loop, not mlx-lm's autoregressive ``generate``. ``install_diffusion_
eval_adapter`` monkeypatches ``mlx_lm.load`` / ``mlx_lm.generate`` so that, for a
``diffusion_gemma`` checkpoint, both route through the OptiQ diffusion API
(``optiq.vlm.diffusion_gemma.load`` / ``generate``). Non-diffusion models fall
through to the originals.

(MMLU is logit-based and the stock module reads AR next-token logits, which a
diffusion forward doesn't produce — ``scripts/diffusion_capability_score.py``
handles MMLU through the canvas position-0 path instead.)
"""

from __future__ import annotations

from .generate import generate as diffusion_generate
from .generate import load as diffusion_load
from .loader import load_config


def install_diffusion_eval_adapter() -> None:
    """Patch ``mlx_lm.load`` / ``mlx_lm.generate`` to be DiffusionGemma-aware.

    Idempotent. Safe for non-diffusion models (they fall through unchanged).
    """
    import mlx_lm

    if getattr(mlx_lm, "_optiq_diffusion_patched", False):
        return
    orig_load = mlx_lm.load
    orig_generate = mlx_lm.generate

    def load(path, *args, **kwargs):
        model_type = None
        try:
            model_type = load_config(path).get("model_type")
        except Exception:
            pass
        if model_type == "diffusion_gemma":
            model, tokenizer = diffusion_load(path)
            model._optiq_diffusion = True
            return model, tokenizer
        return orig_load(path, *args, **kwargs)

    def generate(model, tokenizer, *args, **kwargs):
        if getattr(model, "_optiq_diffusion", False):
            prompt = kwargs.pop("prompt", args[0] if args else None)
            return diffusion_generate(
                model, tokenizer, prompt, max_tokens=kwargs.get("max_tokens", 512)
            )
        return orig_generate(model, tokenizer, *args, **kwargs)

    mlx_lm.load = load
    mlx_lm.generate = generate
    mlx_lm._optiq_diffusion_patched = True


def evaluate_mmlu_diffusion(model_path: str, n_samples: int = 1000,
                            n_shots: int = 5, seed: int = 42, **_ignored) -> float:
    """5-shot MMLU scored through the diffusion canvas, not AR next-token logits.

    OptiQ's stock MMLU is logit-based: it reads the next-token distribution over
    " A"/" B"/" C"/" D". A diffusion forward does not produce one — it predicts a
    *canvas*. So encode the 5-shot prompt, take the first canvas position (the
    answer slot), and argmax the four letter logits there.

    This lived only in `scripts/diffusion_capability_score.py`, which the wheel does
    not ship, so `optiq eval --task all` on a DiffusionGemma model could not produce
    a Capability Score at all.
    """
    import mlx.core as mx
    import numpy as np
    from datasets import load_dataset

    from optiq.eval.mmlu import _build_prompt
    from optiq.eval.decode import seed_rng

    from .calibration import capture_canvas_from_ids
    from .generate import load as diffusion_load

    # The canvas is initialized from RANDOM token ids, so an unseeded run scores
    # a different canvas each time (the same reason every other benchmark seeds).
    seed_rng(seed)
    model, tokenizer = diffusion_load(model_path)
    test_ds = load_dataset("cais/mmlu", "all", split="test")
    dev_ds = load_dataset("cais/mmlu", "all", split="dev")

    dev_by_subject: dict[str, list] = {}
    for ex in dev_ds:
        dev_by_subject.setdefault(ex["subject"], []).append(ex)
    test_by_subject: dict[str, list] = {}
    for ex in test_ds:
        test_by_subject.setdefault(ex["subject"], []).append(ex)

    rng = np.random.RandomState(seed)
    subjects = sorted(test_by_subject)
    per_subject = max(1, n_samples // len(subjects))
    sampled = []
    for subj in subjects:
        pool = test_by_subject[subj]
        idxs = rng.choice(len(pool), size=min(per_subject, len(pool)), replace=False)
        sampled.extend(pool[int(i)] for i in idxs)
    if len(sampled) > n_samples:
        rng.shuffle(sampled)
        sampled = sampled[:n_samples]

    letter_ids = []
    for letter in ("A", "B", "C", "D"):
        ids = (tokenizer.encode(f" {letter}", add_special_tokens=False)
               or tokenizer.encode(letter, add_special_tokens=False))
        letter_ids.append(ids[0])

    from tqdm import tqdm

    n_correct = 0
    n_scored = 0
    n_skipped = 0
    for ex in tqdm(sampled, desc="MMLU (diffusion canvas)"):
        prompt = _build_prompt(
            ex["question"], ex["choices"], dev_by_subject.get(ex["subject"], []),
            ex["subject"], n_shots=n_shots,
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        canvas = capture_canvas_from_ids(
            model, tokenizer, mx.array(ids, dtype=mx.int32)[None]
        )
        if canvas is None:
            # Canvas capture failed — a harness miss, not a wrong answer. Excluded
            # from the denominator instead of counted incorrect (which understated).
            n_skipped += 1
            continue
        logits = model(**canvas).logits          # (1, canvas, vocab)
        p0 = logits[0, 0, :]                     # the answer slot
        pick = int(mx.argmax(mx.array([float(p0[t].item()) for t in letter_ids])).item())
        n_correct += int(pick == ex["answer"])
        n_scored += 1

    if n_skipped:
        print(f"  ⚠ MMLU (diffusion): {n_skipped}/{len(sampled)} questions had no "
              f"canvas capture and were excluded from the denominator")
    return 100.0 * n_correct / max(n_scored, 1)
