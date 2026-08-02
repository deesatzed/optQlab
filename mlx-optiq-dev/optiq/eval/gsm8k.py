"""GSM8K evaluation for quantized MLX LLMs.

Measures grade-school math accuracy to verify quantized models
still reason correctly, not just produce fluent text.
"""

import os
import re
from ._harmony import strip_to_final_channel, concise_template_kwargs
from dataclasses import dataclass

import numpy as np


# 3-shot exemplars for chain-of-thought prompting
FEW_SHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted.\n#### 6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39",
    },
]


@dataclass
class GSM8KResult:
    n_correct: int
    n_total: int
    accuracy: float
    per_question: list[dict]


def _build_prompt(question: str, n_shots: int = 3) -> str:
    """Build a few-shot prompt for GSM8K."""
    prompt = ""
    for ex in FEW_SHOT_EXAMPLES[:n_shots]:
        prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    prompt += f"Q: {question}\nA:"
    return prompt


def _extract_answer(text: str) -> str | None:
    """Extract the numeric answer from a model output.

    Strategy:
      1. If the output contains a ``</think>`` tag, look only at the post-think
         portion (reasoning models like Qwen3.5/3.6 emit a ``<think>...</think>``
         block of working before the actual answer).
      2. Look for the canonical ``#### N`` marker first.
      3. Otherwise, look for ``\\boxed{N}`` (LaTeX-formatted final answer that
         many instruction-tuned models emit).
      4. As a last-resort fallback, take the last number in the (post-think)
         text. This is unreliable for long CoT outputs that get truncated, so
         the chat-template + thinking-disabled path above should usually keep
         the output short and answer-focused.
    """
    # Reduce channel-structured output (gpt-oss harmony / Gemma-4) to the final
    # answer before any number extraction.
    text = strip_to_final_channel(text)

    # Strip the thinking block if present
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    # Number pattern that REQUIRES a digit — the old `[\d,]+` matched a lone
    # comma, so a final answer stated mid-sentence ("18 apples, and...") left a
    # bare "," as the last "number", which parsed to None and scored wrong.
    num = r"-?\d[\d,]*(?:\.\d+)?"

    # Look for #### marker
    match = re.search(rf"####\s*({num})", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # \boxed{N} (LaTeX style)
    match = re.search(rf"\\boxed\{{\s*({num})\s*\}}", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Explicit "the answer is N" — preferred over the last-number fallback, which
    # otherwise grabs a trailing incidental number ("the answer is 42. That took
    # 3 steps." must return 42, not 3). Take the LAST such statement.
    stated = re.findall(rf"answer\s*(?:is|:)?\s*(?:of\s+)?\$?({num})", text, re.IGNORECASE)
    if stated:
        return stated[-1].replace(",", "").strip()

    # Fallback: the last number in the text.
    numbers = re.findall(num, text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None


def _extract_ground_truth(answer_text: str) -> str:
    """Extract numeric answer from GSM8K ground truth (after ####)."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return ""


def _normalize_number(s: str) -> float | None:
    """Parse a number string to float for comparison."""
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return None


def evaluate_gsm8k(
    model_path: str,
    n_samples: int = 200,
    n_shots: int = 3,
    max_tokens: int = 512,
    seed: int = 42,
    reasoning: bool = False,
    repetition_penalty: float | None = None,
) -> GSM8KResult:
    """Evaluate an MLX LLM on GSM8K test set.

    Args:
        model_path: Path to MLX model directory
        n_samples: Number of test questions to evaluate
        n_shots: Number of few-shot examples in prompt
        max_tokens: Max tokens to generate per question
        seed: Random seed for sample selection

    Returns:
        GSM8KResult with accuracy and per-question details
    """
    import mlx.core as mx
    from mlx_lm import load, generate
    from .decode import load_tolerant, reclaim, resolve_decode, seed_rng
    from datasets import load_dataset
    from tqdm import tqdm

    # Resolve to absolute path so mlx_lm treats it as local, not HF repo
    if os.path.isdir(model_path):
        model_path = os.path.abspath(model_path)
    print(f"  Loading model from {model_path}...")
    seed_rng(seed)   # diffusion decode starts from a RANDOM canvas
    model, tokenizer = load_tolerant(model_path)

    print(f"  Loading GSM8K test set...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    # Sample subset
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    indices.sort()

    n_correct = 0
    per_question = []

    # Detect whether the tokenizer's chat template accepts ``enable_thinking``
    # (Qwen3.5/3.6 reasoning models do — passing False keeps the
    # ``<think>...</think>`` block empty so the model emits a direct answer
    # within the budget). For models that don't support the kwarg or don't
    # have a chat template at all, fall back to the original raw few-shot
    # prompt.
    use_chat_template = bool(getattr(tokenizer, "chat_template", None))
    # Reasoning mode: let the model think (don't suppress) — the caller gives a
    # large max_tokens budget so the trace completes and the answer appears.
    concise_kwargs = {} if reasoning else concise_template_kwargs(tokenizer)
    print(
        f"  Evaluating {len(indices)} questions ({n_shots}-shot, "
        f"chat_template={use_chat_template}, concise={concise_kwargs})..."
    )

    for idx in tqdm(indices, desc="  GSM8K eval"):
        item = ds[int(idx)]
        question = item["question"]
        gt_answer = _extract_ground_truth(item["answer"])

        # Build the user-facing prompt body (few-shot exemplars + new question)
        body = _build_prompt(question, n_shots=n_shots)

        # Wrap into the model's chat template if available — Qwen3.5/3.6
        # reasoning models behave very poorly on raw prompts (inconsistent
        # think-mode triggering, output truncation mid-thought). For those,
        # we ALSO pass ``enable_thinking=False`` so they answer directly.
        if use_chat_template:
            template_kwargs = dict(
                tokenize=False, add_generation_prompt=True, **concise_kwargs,
            )
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": body}],
                **template_kwargs,
            )
        else:
            prompt = body

        # Generate with greedy decoding for deterministic math
        sampler, logits_processors = resolve_decode(model, repetition_penalty)
        output = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
            sampler=sampler, logits_processors=logits_processors,
        )

        reclaim()
        predicted = _extract_answer(output)

        gt_num = _normalize_number(gt_answer)
        pred_num = _normalize_number(predicted) if predicted else None

        correct = (gt_num is not None and pred_num is not None
                   and abs(gt_num - pred_num) < 1e-3)
        if correct:
            n_correct += 1

        per_question.append({
            "idx": int(idx),
            "question": question[:100] + "...",
            "ground_truth": gt_answer,
            "predicted": predicted,
            "correct": correct,
        })

    accuracy = n_correct / len(indices) if indices.size > 0 else 0.0

    return GSM8KResult(
        n_correct=n_correct,
        n_total=len(indices),
        accuracy=accuracy,
        per_question=per_question,
    )


def print_gsm8k_report(result: GSM8KResult):
    """Print GSM8K evaluation results."""
    print(f"\n  GSM8K Results")
    print(f"  {'=' * 50}")
    print(f"  Accuracy: {result.n_correct}/{result.n_total} "
          f"({result.accuracy:.1%})")

    # Show some examples
    wrong = [q for q in result.per_question if not q["correct"]]
    right = [q for q in result.per_question if q["correct"]]

    if right:
        print(f"\n  Correct examples:")
        for q in right[:3]:
            print(f"    Q: {q['question']}")
            print(f"    GT: {q['ground_truth']}, Pred: {q['predicted']}")

    if wrong:
        print(f"\n  Incorrect examples:")
        for q in wrong[:3]:
            print(f"    Q: {q['question']}")
            print(f"    GT: {q['ground_truth']}, Pred: {q['predicted']}")
