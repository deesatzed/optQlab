"""Basic validators for benchmark outputs."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    name: str
    passed: bool
    detail: str = ""


def validate_json_text(text: str) -> ValidationResult:
    try:
        json.loads(text)
        return ValidationResult("json", True)
    except Exception as exc:
        return ValidationResult("json", False, str(exc))


def validate_python_syntax(text: str) -> ValidationResult:
    try:
        ast.parse(text)
        return ValidationResult("python_syntax", True)
    except Exception as exc:
        return ValidationResult("python_syntax", False, str(exc))


def validate_balanced_delimiters(text: str) -> ValidationResult:
    """Catch obvious code-shape collapse without pretending to judge quality."""

    pairs = {")": "(", "]": "[", "}": "{"}
    opens = set(pairs.values())
    stack: list[str] = []
    in_string: str | None = None
    escaped = False

    for index, char in enumerate(text):
        if in_string is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            continue
        if char in {"'", '"'}:
            in_string = char
            continue
        if char in opens:
            stack.append(char)
        elif char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return ValidationResult(
                    "balanced_delimiters",
                    False,
                    f"unexpected {char!r} at character {index}",
                )
            stack.pop()

    if in_string is not None:
        return ValidationResult("balanced_delimiters", False, "unterminated string")
    if stack:
        return ValidationResult(
            "balanced_delimiters",
            False,
            f"unclosed delimiter(s): {''.join(stack[-8:])}",
        )
    return ValidationResult("balanced_delimiters", True)


def validate_contains_tool_call_shape(text: str) -> ValidationResult:
    lowered = text.lower()
    has_toolish_marker = "tool_call" in lowered or "function" in lowered
    return ValidationResult(
        "tool_call_shape",
        has_toolish_marker,
        "missing tool/function marker" if not has_toolish_marker else "",
    )


def validate_no_degenerate_loop(
    text: str,
    *,
    ngram_size: int = 8,
    max_ngram_repeats: int = 3,
) -> ValidationResult:
    """Catch pathological repeated-output loops without judging normal quality."""

    tokens = re.findall(r"\S+", text.lower())
    if len(tokens) < ngram_size * (max_ngram_repeats + 1):
        return ValidationResult("no_degenerate_loop", True)

    ngrams = zip(*(tokens[i:] for i in range(ngram_size)), strict=False)
    counts = Counter(ngrams)
    if not counts:
        return ValidationResult("no_degenerate_loop", True)

    ngram, repeats = counts.most_common(1)[0]
    if repeats <= max_ngram_repeats:
        return ValidationResult("no_degenerate_loop", True)

    preview = " ".join(ngram[: min(ngram_size, 12)])
    return ValidationResult(
        "no_degenerate_loop",
        False,
        f"repeated {ngram_size}-gram {repeats} times: {preview!r}",
    )


def validate_benchmark_output(
    text: str,
    *,
    category: str,
    prompt_id: str = "",
) -> list[ValidationResult]:
    """Return the standard validator set for a benchmark prompt row."""

    category_l = category.lower()
    prompt_l = prompt_id.lower()
    results = [validate_no_degenerate_loop(text)]
    if category_l == "json_tool":
        results.append(validate_json_text(text.strip()))
    if "coding" in category_l or "code" in prompt_l or "python" in prompt_l:
        results.append(validate_balanced_delimiters(text))
    if (
        category_l == "cold_coding"
        or "python_modules" in prompt_l
        or "full_python" in prompt_l
    ):
        results.append(validate_python_syntax(text))
    return results
