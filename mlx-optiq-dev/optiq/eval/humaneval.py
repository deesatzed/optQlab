"""HumanEval — code-generation evaluation for quantized MLX LLMs.

164 Python function-completion problems from
`openai_humaneval <https://huggingface.co/datasets/openai_humaneval>`_.
For each problem we:

  1. Render the function signature + docstring as the prompt.
  2. Generate a continuation up to a stop token.
  3. Concatenate the prompt and completion to form a candidate program.
  4. Append the dataset's test cases plus a ``check(<entry>)`` driver.
  5. Run the resulting script in a sandbox (apple/container if installed,
     else macOS sandbox-exec, else subprocess) with a 5 s timeout. The memory
     cap (512 MB) is enforced only on the apple/container tier; macOS rejects
     the setrlimit call, so the sandbox-exec/subprocess tiers run without a hard
     memory limit (the timeout still bounds runaway programs).

A problem is "passed" only if the program runs the full ``check(<entry>)``
driver to completion and prints the pass sentinel — a program that exits early
(e.g. ``sys.exit(0)``) before the asserts finish is NOT credited. The reported
metric is **pass@1** at temperature 0 — the canonical, strictest HumanEval
number.
"""

from __future__ import annotations

import os
import re
from ._harmony import strip_to_final_channel, concise_template_kwargs
import time
from dataclasses import dataclass

import numpy as np

from ..lab.sandbox import run_python, detect_sandbox_kind


# Stop sequences — mirror what the original HumanEval harness uses.
_STOP_SEQS = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\n```"]


def _extract_code_block(text: str) -> str | None:
    """If ``text`` contains a fenced ```python ... ``` block, return the
    last one (the most likely "final answer" when models repeat themselves).
    Returns ``None`` if there's no fenced block.
    """
    # gpt-oss harmony / Gemma-4: keep only the final channel so a code block the
    # model wrote while *reasoning* doesn't get picked over the answer.
    text = strip_to_final_channel(text)
    # Match ```python\n...\n``` or just ```\n...\n```
    matches = re.findall(
        r"```(?:python|py)?\n(.*?)\n```",
        text, re.DOTALL,
    )
    return matches[-1] if matches else None


def _strip_function_def(body: str, entry_point: str) -> str:
    """When the model emits a full function (header + body) inside a code
    block, the prompt also has the header so the program would have a
    duplicate ``def``. Strip everything up to and including the
    ``def entry_point(...):`` line + its docstring; keep just the body.
    Falls through if no ``def entry_point`` is found.
    """
    # Find the function def line
    pat = rf"def\s+{re.escape(entry_point)}\s*\([^)]*\)[^:]*:\s*\n"
    m = re.search(pat, body)
    if not m:
        return body
    # Preserve any module-level imports the model wrote ABOVE the def — the
    # prompt does not always carry them, so dropping them turns a correct
    # solution into a NameError. They ride at column 0 and _build_program
    # hoists them to the top of the assembled program.
    imports = [l for l in body[:m.start()].splitlines()
               if re.match(r"(import |from \w)", l)]
    after = body[m.end():]
    # Skip leading blank lines + a possible docstring
    rest = after.lstrip("\n")
    docstring_match = re.match(r"\s*([\"\']{3})(.*?)\1\s*\n", rest, re.DOTALL)
    if docstring_match:
        rest = rest[docstring_match.end():]
    if imports:
        rest = "\n".join(imports) + "\n" + rest
    return rest


def _truncate_completion(text: str, entry_point: str = "") -> str:
    """Reduce the model's response down to a function body suitable for
    concatenating onto the HumanEval prompt.

    Strategy:
      1. Strip thinking blocks.
      2. If the response contains fenced code blocks, take the last one;
         strip the ``def entry_point(...):`` header + docstring so what
         remains is purely the body.
      3. If no fenced block, fall back to the legacy stop-sequence cut
         (strict completion-style models).
    """
    out = text
    if "</think>" in out:
        out = out.split("</think>", 1)[1]
    if "<channel|>" in out:
        # Gemma-4: <|channel>thought ... <channel|> precedes the response
        out = out.split("<channel|>", 1)[1]

    # Try to extract from a fenced code block (covers conversational models
    # like Gemma-4 that won't follow "no markdown" instructions).
    block = _extract_code_block(out)
    if block is not None:
        if entry_point:
            return _strip_function_def(block, entry_point)
        return block

    # Legacy / completion-mode path: no fenced block.
    # Drop leading blank lines but KEEP indentation — lstrip() used to remove
    # the body's own indent, so an indented completion de-indented into an
    # IndentationError when concatenated onto the prompt.
    out = out.lstrip("\n")
    if out.startswith("```python\n"):
        out = out[len("```python\n"):]
    elif out.startswith("```\n"):
        out = out[len("```\n"):]
    # A model may emit the whole function (imports + `def entry_point`) with no
    # fence. Extract the body the same way the fenced path does, otherwise the
    # `\ndef ` stop sequence cuts at the def and discards the entire answer
    # (the imports-before-def case: 26 of 164 HumanEval prompts).
    if entry_point and re.search(rf"def\s+{re.escape(entry_point)}\s*\(", out):
        return _strip_function_def(out, entry_point)
    earliest = len(out)
    for stop in _STOP_SEQS:
        i = out.find(stop)
        if i != -1 and i < earliest:
            earliest = i
    return out[:earliest]


def _build_program(prompt: str, completion: str, test: str, entry_point: str) -> str:
    """Concatenate prompt + completion + test driver. Mirrors the official
    ``human_eval`` reference scoring protocol."""
    # Hoist any column-0 imports the completion carries (the body will be
    # appended INSIDE the prompt's function, so an import left there would be
    # indented/unreachable). Idempotent — a duplicate import is harmless.
    imports, rest = [], []
    for line in completion.split("\n"):
        (imports if re.match(r"(import |from \w[\w.]* import )", line) else rest).append(line)
    head = ("\n".join(imports) + "\n\n") if imports else ""
    # The trailing sentinel is what proves the asserts actually ran to
    # completion. Without it, a completion that reaches sys.exit(0)/quit()
    # (in the body or as trailing module code) exits 0 before check() finishes
    # and was scored PASSED — official human_eval catches SystemExit as failure.
    return (
        head
        + prompt
        + "\n".join(rest)
        + "\n\n"
        + test
        + "\n\n"
        + f"check({entry_point})\n"
        + "print('__OPTIQ_HE_PASS__', flush=True)\n"
    )


@dataclass
class HumanEvalResult:
    n_total: int
    n_passed: int
    pass_at_1: float
    sandbox_kind: str
    failures_by_reason: dict[str, int]
    elapsed_sec: float

    def __str__(self) -> str:
        ci = 1.96 * np.sqrt(
            (self.pass_at_1 * (1 - self.pass_at_1)) / max(self.n_total, 1)
        )
        s = (
            f"HumanEval pass@1: {self.n_passed}/{self.n_total} = "
            f"{self.pass_at_1 * 100:.1f}% (95% CI ±{ci * 100:.1f}pp), "
            f"sandbox={self.sandbox_kind}, elapsed {self.elapsed_sec:.0f}s"
        )
        if self.failures_by_reason:
            top = sorted(self.failures_by_reason.items(),
                         key=lambda kv: -kv[1])[:3]
            s += "\n  failure modes: " + \
                 ", ".join(f"{k}({v})" for k, v in top)
        return s


def evaluate_humaneval(
    model_path: str,
    n_samples: int | None = None,
    max_tokens: int = 512,
    timeout_sec: float = 5.0,
    memory_mb: int = 512,
    seed: int = 42,
    reasoning: bool = False,
    repetition_penalty: float | None = None,
) -> HumanEvalResult:
    """Run HumanEval pass@1 at temperature 0.

    Args:
        model_path: Path or HF repo of the model.
        n_samples: Number of problems to evaluate. None → full 164-problem set.
        max_tokens: Generation budget per problem.
        timeout_sec: Per-program execution timeout in the sandbox.
        memory_mb: Per-program memory cap in MB.
        seed: Reproducibility for sampling when ``n_samples < 164``.
    """
    import mlx.core as mx
    from mlx_lm import load, generate
    from .decode import load_tolerant, reclaim, resolve_decode, seed_rng
    from datasets import load_dataset
    from tqdm import tqdm

    t0 = time.time()
    if os.path.isdir(model_path):
        model_path = os.path.abspath(model_path)
    print(f"  Loading model from {model_path} …")
    seed_rng(seed)   # diffusion decode starts from a RANDOM canvas
    model, tokenizer = load_tolerant(model_path)

    print(f"  Loading openai/openai_humaneval (test split) …")
    # Legacy "openai_humaneval" id is rejected by huggingface-hub >=1.0
    # which now requires the namespaced "<org>/<name>" form. Same payload.
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = list(ds)

    rng = np.random.RandomState(seed)
    if n_samples is not None and n_samples < len(rows):
        idxs = rng.choice(len(rows), size=n_samples, replace=False)
        rows = [rows[int(i)] for i in idxs]

    sandbox_kind = detect_sandbox_kind()
    print(f"  sandbox kind: {sandbox_kind}")

    use_chat = bool(getattr(tokenizer, "chat_template", None))
    concise = {} if reasoning else concise_template_kwargs(tokenizer)
    sampler, logits_processors = resolve_decode(model, repetition_penalty)

    n_passed = 0
    failures: dict[str, int] = {}

    for ex in tqdm(rows, desc="HumanEval"):
        prompt_text = ex["prompt"]
        if use_chat:
            # Wrap as instruct: ask for the COMPLETE function inside a
            # python code block. We strip the header + docstring server-
            # side; what we want from the model is a complete, properly-
            # indented function we can extract verbatim. Asking for
            # "just the body" makes models drop indentation, which then
            # explodes when concatenated to the prompt.
            user_msg = (
                "Complete the following Python function. Output the "
                "complete function (header, docstring, body) inside a "
                "single ```python ... ``` code block. No commentary "
                "outside the code block.\n\n"
                f"```python\n{prompt_text}```"
            )
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True, **concise,
            )
        else:
            prompt = prompt_text

        completion = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=max_tokens, sampler=sampler, logits_processors=logits_processors, verbose=False,
        )
        reclaim()
        completion = _truncate_completion(completion, entry_point=ex["entry_point"])

        program = _build_program(
            ex["prompt"], completion, ex["test"], ex["entry_point"],
        )

        # strict=False: HumanEval's test driver mirrors the reference
        # scoring protocol; AST safety checks would reject legitimate
        # assertion patterns. Sandbox isolation alone is sufficient here.
        result = run_python(
            program, timeout=timeout_sec, memory_limit_mb=memory_mb, strict=False,
        )
        passed = (result.returncode == 0 and not result.timed_out
                  and "__OPTIQ_HE_PASS__" in (result.stdout or ""))
        if passed:
            n_passed += 1
        else:
            if result.returncode < 0 and not result.timed_out:
                # Sandbox could not RUN the program (launch/reject/cancel), which
                # is a harness failure, not a wrong answer. Bucket it distinctly
                # so it is visible instead of silently folded into "other".
                reason = "harness_error"
            elif result.timed_out:
                reason = "timeout"
            elif "AssertionError" in (result.stderr or ""):
                reason = "assertion_failed"
            elif "SyntaxError" in (result.stderr or ""):
                reason = "syntax_error"
            elif "NameError" in (result.stderr or ""):
                reason = "name_error"
            elif "TypeError" in (result.stderr or ""):
                reason = "type_error"
            else:
                reason = "other"
            failures[reason] = failures.get(reason, 0) + 1

    n_total = len(rows)
    return HumanEvalResult(
        n_total=n_total,
        n_passed=n_passed,
        pass_at_1=n_passed / max(n_total, 1),
        sandbox_kind=sandbox_kind,
        failures_by_reason=failures,
        elapsed_sec=time.time() - t0,
    )
