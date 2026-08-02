"""5-shot MMLU evaluation for quantized MLX LLMs.

MMLU (``cais/mmlu``) is the standard multi-domain knowledge benchmark
across 57 subjects: STEM, humanities, social sciences, etc. Each example
is a multiple-choice question with 4 options (A/B/C/D); the model picks
one. The 5-shot variant prepends 5 in-domain demonstrations from the
``dev`` split before each test question — this matches how everyone
else reports MMLU and lets us compare against published baselines.

Scoring is per-token first-letter argmax over A/B/C/D logits — the
deterministic, fast variant that doesn't require generation. This is
how Unsloth (and most quant-eval pipelines) measure MMLU; it's stable,
cheap, and matches the official methodology.

For a quant-eval pipeline our needs are:
  * Cheap — runs in ~30 min on a 27 B with 1000 samples
  * Sample-size-tunable so we can pick a CI we're happy with
  * Stable across runs (no temperature)
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import numpy as np

from ._harmony import strip_to_final_channel


def _extract_letter(text: str) -> int | None:
    """Pull an A/B/C/D answer index out of a (post-think) generation.

    The answer letter is uppercase and standalone — MMLU's few-shot demos show
    ``Answer: B`` — so the letter capture is case-sensitive and anchored between
    non-letters, while only the prefix words ("answer", "is") are matched
    case-insensitively. An earlier version matched ``[A-D]`` under
    ``re.IGNORECASE`` with no anchor, so "the answer is choice B" captured the
    'c' of "choice" and "definitely B" the 'd' — a wrong letter that scored
    correct only ~1/3 of the time, silently understating chatty models.
    """
    text = strip_to_final_channel(text)
    # A standalone uppercase A-D, not glued to a surrounding word.
    L = r"(?<![A-Za-z])([A-D])(?![A-Za-z])"
    # After an explicit "answer is …" we can also accept lowercase b/c/d — those
    # are never English words, so there is no filler-word ambiguity. Lowercase
    # 'a' is excluded: it is the article ("the answer is a bit tricky"), the
    # case that made the letter extractor wrong in the first place.
    La = r"(?<![A-Za-z])([A-Dbcd])(?![A-Za-z])"
    # Strongest signals first: "answer is (B)", "answer: b", "**C**", "(D)".
    for pat in (
        r"(?i:answer)\s*(?i:is|:)?\s*\(?\*{0,2}" + La,
        r"\*\*\s*([A-D])\s*\*\*",
        r"\(([A-D])\)",
        L,
    ):
        m = list(re.finditer(pat, text))
        if m:
            return ord(m[-1].group(1).upper()) - ord("A")
    return None


# 5-shot prompt template (canonical formulation matching Hendrycks et al.)
_HEADER = (
    "The following are multiple choice questions (with answers) about "
    "{subject_pretty}.\n\n"
)


def _format_subject(subject: str) -> str:
    """Convert e.g. 'high_school_us_history' → 'high school us history'."""
    return subject.replace("_", " ")


def _format_example(question: str, choices, answer_idx: int | None = None) -> str:
    """Format a single MMLU example.

    If ``answer_idx`` is given (for the 5-shot demonstrations), append the
    correct letter; otherwise leave open for the model to fill in.
    """
    s = question.strip() + "\n"
    for i, choice in enumerate(choices):
        s += f"{chr(ord('A') + i)}. {choice}\n"
    s += "Answer:"
    if answer_idx is not None:
        s += f" {chr(ord('A') + answer_idx)}\n"
    return s


def _build_prompt(
    test_q: str, test_choices, dev_examples: list, subject: str, n_shots: int = 5,
) -> str:
    """Build a 5-shot prompt: subject header → 5 demos → test question."""
    pretty = _format_subject(subject)
    prompt = _HEADER.format(subject_pretty=pretty)
    for ex in dev_examples[:n_shots]:
        prompt += _format_example(ex["question"], ex["choices"], ex["answer"])
        prompt += "\n"
    prompt += _format_example(test_q, test_choices)
    return prompt


@dataclass
class MMLUResult:
    n_correct: int
    n_total: int
    accuracy: float
    by_subject: dict[str, tuple[int, int]]  # subject -> (correct, total)
    elapsed_sec: float

    def __str__(self) -> str:
        ci = 1.96 * np.sqrt(
            (self.accuracy * (1 - self.accuracy)) / max(self.n_total, 1)
        )
        return (
            f"MMLU 5-shot accuracy: {self.n_correct}/{self.n_total} = "
            f"{self.accuracy * 100:.1f}% (95% CI ±{ci * 100:.1f}pp), "
            f"elapsed {self.elapsed_sec:.0f}s"
        )


def evaluate_mmlu(
    model_path: str,
    n_samples: int = 1000,
    n_shots: int = 5,
    seed: int = 42,
    reasoning: bool = False,
    max_tokens: int = 4096,  # reasoning-path budget (non-reasoning MMLU is argmax, generates nothing)
    repetition_penalty: float | None = None,
) -> MMLUResult:
    """5-shot MMLU on ``cais/mmlu`` test split.

    Args:
        model_path: Path or HF repo of the model.
        n_samples: Total test questions to sample across all 57 subjects.
            Default 1000 → ~17–18 questions per subject, 95 % CI ±~3 pp on
            the global accuracy.
        n_shots: Demos prepended per question. Standard is 5.
        seed: Reproducibility for sample selection.
    """
    import mlx.core as mx
    from mlx_lm import load
    from datasets import load_dataset
    from tqdm import tqdm

    from .decode import load_tolerant, reclaim, seed_rng

    t0 = time.time()

    if os.path.isdir(model_path):
        model_path = os.path.abspath(model_path)
    print(f"  Loading model from {model_path} …")
    seed_rng(seed)   # diffusion decode starts from a RANDOM canvas
    model, tokenizer = load_tolerant(model_path)

    print(f"  Loading MMLU test + dev splits …")
    test_ds = load_dataset("cais/mmlu", "all", split="test")
    dev_ds = load_dataset("cais/mmlu", "all", split="dev")

    # Group dev examples by subject for prompt construction
    dev_by_subject: dict[str, list] = {}
    for ex in dev_ds:
        dev_by_subject.setdefault(ex["subject"], []).append(ex)

    # Stratified sample across subjects
    rng = np.random.RandomState(seed)
    test_by_subject: dict[str, list] = {}
    for ex in test_ds:
        test_by_subject.setdefault(ex["subject"], []).append(ex)

    # Aim for roughly equal samples per subject
    subjects = sorted(test_by_subject.keys())
    per_subject = max(1, n_samples // len(subjects))
    sampled = []
    for subj in subjects:
        pool = test_by_subject[subj]
        idxs = rng.choice(len(pool), size=min(per_subject, len(pool)), replace=False)
        for i in idxs:
            sampled.append(pool[int(i)])
    if len(sampled) > n_samples:
        rng.shuffle(sampled)
        sampled = sampled[:n_samples]

    # Reasoning models can't be scored by first-letter argmax: a model trained
    # to think before answering doesn't put its answer-letter mass on the token
    # right after "Answer:". For those, generate WITH thinking (large budget),
    # strip the <think> block, and parse the letter out of the final answer.
    if reasoning:
        from mlx_lm import generate
        from .decode import reclaim, resolve_decode, seed_rng

        n_correct = 0
        by_subject = {s: [0, 0] for s in subjects}
        sampler, logits_processors = resolve_decode(model, repetition_penalty)
        for ex in tqdm(sampled, desc="MMLU (reasoning) eval"):
            subj = ex["subject"]
            opts = "\n".join(
                f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(ex["choices"])
            )
            body = (
                f"Answer this multiple-choice question about "
                f"{_format_subject(subj)}.\n\n{ex['question'].strip()}\n\n{opts}\n\n"
                f"Reason it through, then end your reply with "
                f"'The answer is X' where X is the letter A, B, C, or D."
            )
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": body}],
                tokenize=False, add_generation_prompt=True,
            )
            out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                           verbose=False, sampler=sampler, logits_processors=logits_processors)
            reclaim()
            pred_idx = _extract_letter(out)
            correct = int(pred_idx is not None and pred_idx == ex["answer"])
            n_correct += correct
            by_subject[subj][0] += correct
            by_subject[subj][1] += 1
        n_total = len(sampled)
        return MMLUResult(
            n_correct=n_correct, n_total=n_total,
            accuracy=n_correct / max(n_total, 1),
            by_subject={s: (c, t) for s, (c, t) in by_subject.items() if t > 0},
            elapsed_sec=time.time() - t0,
        )

    # Pre-compute the answer-letter token ids for fast scoring
    letter_ids = []
    for letter in ["A", "B", "C", "D"]:
        # Match how Hendrycks et al. score: " A", " B", ... after "Answer:"
        ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(letter, add_special_tokens=False)
        letter_ids.append(ids[0] if ids else None)
    if any(t is None for t in letter_ids):
        # Fallback to bare letter without leading space
        letter_ids = []
        for letter in ["A", "B", "C", "D"]:
            ids = tokenizer.encode(letter, add_special_tokens=False)
            letter_ids.append(ids[0] if ids else None)
    # A broken/incomplete tokenizer can resolve the four letters to fewer than
    # four distinct ids; argmax over the ties then predicts "A" for every
    # question and reports the ~23% A-gold rate as a real score. Fail loud
    # instead of scoring garbage.
    if len(set(letter_ids)) != 4 or None in letter_ids:
        raise RuntimeError(
            "MMLU: could not resolve four distinct A/B/C/D answer tokens "
            f"(got {letter_ids}). The tokenizer is likely incomplete — refusing "
            "to score, since this would silently report ~23% for any model."
        )

    n_correct = 0
    by_subject: dict[str, list[int]] = {s: [0, 0] for s in subjects}

    for ex in tqdm(sampled, desc="MMLU 5-shot eval"):
        subj = ex["subject"]
        prompt = _build_prompt(
            ex["question"], ex["choices"],
            dev_by_subject.get(subj, []),
            subj, n_shots=n_shots,
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        ids_arr = mx.array([ids], dtype=mx.int32)

        reclaim()
        out = model(ids_arr)
        logits = out.logits if hasattr(out, "logits") else out
        # Last-token logits — what would come after "Answer:"
        last = logits[:, -1, :]
        # Pick the letter with highest logit
        letter_logits = mx.array([float(last[0, t].item()) for t in letter_ids])
        pred_idx = int(mx.argmax(letter_logits).item())

        gold_idx = ex["answer"]
        correct = int(pred_idx == gold_idx)
        n_correct += correct
        by_subject[subj][0] += correct
        by_subject[subj][1] += 1

    n_total = len(sampled)
    return MMLUResult(
        n_correct=n_correct,
        n_total=n_total,
        accuracy=n_correct / max(n_total, 1),
        by_subject={s: (c, t) for s, (c, t) in by_subject.items() if t > 0},
        elapsed_sec=time.time() - t0,
    )
