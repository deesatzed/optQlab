"""IFEval — instruction-following evaluation for quantized MLX LLMs.

Google's `IFEval <https://huggingface.co/datasets/google/IFEval>`_ probes
whether a model follows verifiable, mechanically-checkable instructions:
"respond in 3 bullet points", "include the keyword 'banana'", "respond in
JSON". Each example carries one or more constraint instructions; we
generate a response and run the corresponding verification function.

We implement the subset of constraint types that cover ~85 % of IFEval.
The rare constraints (forbidden words tied to specific languages, postscript
formatting, etc.) are still scored — they just default to ``True`` in the
verifier when not implemented, which biases our reported numbers slightly
high vs Google's reference impl. We log the unhandled instruction IDs so
the gap is auditable.

Two metrics, both reported (matches the official methodology):
  * **prompt_strict** — fraction of examples where ALL of the prompt's
    instructions pass.
  * **prompt_loose** — same but with response-cleaning preprocessing
    (strip leading/trailing whitespace, collapse newlines).
"""

from __future__ import annotations

import json
import os
import re
from ._harmony import strip_to_final_channel, concise_template_kwargs
import time
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Constraint verifiers — each takes (response, kwargs) and returns bool.
# ---------------------------------------------------------------------------

def _relation(kw: dict, key: str = "relation") -> str:
    """Return the comparison relation, tolerating IFEval's kwargs layout.

    Every IFEval instruction carries the full kwargs schema with most keys set
    to null, and each length/frequency constraint keeps its relation under its
    own key (``relation``, ``let_relation``, ``capital_relation``). ``dict.get``
    with a default does not help: the default only fires when the key is absent,
    not when it is present and null, so ``kw.get("relation", "at least")``
    returned ``None`` and ``"least" in None`` raised, failing the whole prompt.
    Coerce a missing-or-null value to the standard ``at least`` fallback.
    """
    return kw.get(key) or "at least"


def _check_length_words(response: str, **kw) -> bool:
    n = len(re.findall(r"\b\w+\b", response))
    rel = _relation(kw)
    target = int(kw.get("num_words") or 0)
    return n >= target if "least" in rel else n <= target


_SENT_ABBREV = ("mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
                "inc", "ltd", "co", "e.g", "i.e", "u.s", "a.m", "p.m", "no", "fig")


def _count_sentences(text: str) -> int:
    """Approximate sentence count, protecting decimals, single-letter initials
    and common abbreviations before splitting on sentence-final punctuation, so
    "Dr. Smith met Mr. Jones." is one sentence, not three. Official IFEval uses
    nltk punkt; this is a dependency-free approximation of it."""
    t = " " + text.strip()
    t = re.sub(r"(\d)\.(\d)", r"\1<dot>\2", t)          # decimals
    for ab in _SENT_ABBREV:
        t = re.sub(rf"(?i)\b{re.escape(ab)}\.", ab + "<dot>", t)
    t = re.sub(r"\b([A-Za-z])\.", r"\1<dot>", t)        # single-letter initials
    return len([p for p in re.split(r"[.!?]+(?:\s|$)", t) if p.strip()])


def _check_length_sentences(response: str, **kw) -> bool:
    n = _count_sentences(response)
    rel = _relation(kw)
    target = int(kw.get("num_sentences") or 0)
    return n >= target if "least" in rel else n <= target


def _check_length_paragraphs(response: str, **kw) -> bool:
    # number_paragraphs is an exact-count constraint whose instruction text says
    # "separate the paragraphs with the markdown divider: ***", and official
    # IFEval splits on that divider — not on blank lines.
    parts = re.split(r"\s*\*\s*\*\s*\*\s*", response)
    n = len([p for p in parts if p.strip()])
    target = int(kw.get("num_paragraphs") or 0)
    return n == target


def _check_keywords_existence(response: str, **kw) -> bool:
    keywords = kw.get("keywords") or []
    text = response.lower()
    return all(k.lower() in text for k in keywords)


def _check_keywords_forbidden(response: str, **kw) -> bool:
    # Word-boundary match, as official IFEval does — a plain substring test
    # wrongly failed "rocket" for a forbidden "rock".
    forbidden = kw.get("forbidden_words") or []
    text = response.lower()
    return not any(
        re.search(rf"\b{re.escape(w.lower())}\b", text) for w in forbidden if w
    )


def _check_keyword_frequency(response: str, **kw) -> bool:
    # Official IFEval counts raw (case-insensitive) occurrences, not whole words.
    kw_word = (kw.get("keyword") or "").lower()
    rel = _relation(kw)
    target = int(kw.get("frequency") or 0)
    n = len(re.findall(re.escape(kw_word), response.lower())) if kw_word else 0
    return n >= target if "least" in rel else n <= target


def _check_letter_frequency(response: str, **kw) -> bool:
    letter = (kw.get("letter") or "").lower()
    rel = _relation(kw, "let_relation")
    target = int(kw.get("let_frequency") or kw.get("frequency") or 0)
    n = response.lower().count(letter)
    return n >= target if "least" in rel else n <= target


def _check_capital_words_count(response: str, **kw) -> bool:
    n = sum(1 for w in re.findall(r"\b\w+\b", response)
            if w.isupper() and len(w) > 1)
    rel = _relation(kw, "capital_relation")
    target = int(kw.get("capital_frequency") or kw.get("frequency") or 0)
    return n >= target if "least" in rel else n <= target


def _check_change_case_capital(response: str, **kw) -> bool:
    return response.upper() == response


def _check_change_case_lowercase(response: str, **kw) -> bool:
    return response.lower() == response


def _check_response_language(response: str, **kw) -> bool:
    """Verify the response is in the target language. Official IFEval uses
    langdetect; when it is installed we use it. Without it we fall back to a
    script check (a non-Latin target must be predominantly non-ASCII), which
    stops the old behaviour of passing EVERY non-English target unconditionally
    — that was crediting models that answered in English."""
    target = (kw.get("language") or "en").lower()
    text = (response or "").strip()
    if not text:
        return False
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        try:
            return detect(text) == target
        except Exception:
            return True  # langdetect can't classify (degenerate text) — official passes
    except ImportError:
        pass
    # Fallback heuristic when langdetect is unavailable.
    ascii_ratio = sum(c < 128 for c in map(ord, text[:400])) / min(len(text), 400)
    latin_langs = {"en", "de", "fr", "es", "it", "pt", "nl", "sw", "vi", "id"}
    if target == "en":
        return ascii_ratio > 0.9
    if target in latin_langs:
        return True  # can't distinguish Latin-script languages without a detector
    return ascii_ratio < 0.5  # non-Latin target must be mostly non-ASCII


def _check_punctuation_no_comma(response: str, **kw) -> bool:
    return "," not in response


def _check_startend_quotation(response: str, **kw) -> bool:
    s = response.strip()
    return s.startswith('"') and s.endswith('"')


def _check_startend_end_phrase(response: str, **kw) -> bool:
    end = (kw.get("end_phrase") or "").strip().rstrip(".!?")
    return response.strip().rstrip(".!?").lower().endswith(end.lower())


def _check_format_number_bullets(response: str, **kw) -> bool:
    # Official IFEval counts markdown bullets (`*` or `-`) only — not numbered
    # lists and not `+`. Counting those inflated the score for a numbered answer.
    bullets = re.findall(r"^\s*\*[^*].*$", response, re.MULTILINE)
    bullets += re.findall(r"^\s*-.*$", response, re.MULTILINE)
    target = int(kw.get("num_bullets") or 0)
    return len(bullets) == target


def _check_format_number_highlighted(response: str, **kw) -> bool:
    n = len(re.findall(r"\*[^*]+\*", response))
    target = int(kw.get("num_highlights") or 0)
    return n >= target


def _check_format_title(response: str, **kw) -> bool:
    return bool(re.search(r"<<[^>]+>>", response))


def _check_format_constrained_response(response: str, **kw) -> bool:
    """One of a fixed set of responses (e.g. 'My answer is yes'). Official IFEval
    accepts the option as a SUBSTRING, so surrounding text is allowed."""
    valid = ("My answer is yes.", "My answer is no.", "My answer is maybe.")
    return any(v in response for v in valid)


def _check_format_json(response: str, **kw) -> bool:
    s = response.strip()
    if s.startswith("```"):
        s = re.sub(r"^```\w*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def _check_format_multiple_sections(response: str, **kw) -> bool:
    marker = (kw.get("section_spliter") or "Section").strip()
    target = int(kw.get("num_sections") or 0)
    n = len(re.findall(rf"{re.escape(marker)}\s*\d+", response, re.IGNORECASE))
    return n >= target


def _check_combination_two_responses(response: str, **kw) -> bool:
    # Official IFEval: exactly two non-empty, distinct responses separated by
    # ******. Empty leading/trailing segments (a model that also brackets the
    # pair with the divider) are allowed, so filter empties rather than demand
    # exactly two split parts.
    parts = [p.strip() for p in response.split("******")]
    nonempty = [p for p in parts if p]
    return len(nonempty) == 2 and nonempty[0] != nonempty[1]


def _check_combination_repeat_prompt(response: str, **kw) -> bool:
    # "First repeat the request word for word, then answer." The text to repeat
    # is in the kwargs, so this IS checkable — it used to return True
    # unconditionally, crediting a model that never repeated the prompt.
    to_repeat = (kw.get("prompt_to_repeat") or "").strip().lower()
    if not to_repeat:
        return True  # nothing to check against
    return response.strip().lower().startswith(to_repeat)


def _check_number_placeholders(response: str, **kw) -> bool:
    # "Contain at least N placeholders in square brackets, e.g. [address]."
    n = len(re.findall(r"\[[^\]]*\]", response))
    return n >= int(kw.get("num_placeholders") or 0)


def _check_postscript(response: str, **kw) -> bool:
    # "End with a postscript starting with P.S. / P.P.S."
    marker = (kw.get("postscript_marker") or "P.S.").strip().lower()
    text = response.lower()
    if marker in ("p.p.s", "p.p.s."):
        pat = r"p\.\s?p\.\s?s\.?"
    elif marker in ("p.s", "p.s."):
        pat = r"p\.\s?s\.?"
    else:
        pat = re.escape(marker)
    # Official IFEval finds the marker anywhere (a postscript is at the end, but
    # matching anywhere avoids understating an inline "P.S.").
    return bool(re.search(pat, text))


def _check_nth_paragraph_first_word(response: str, **kw) -> bool:
    # "N paragraphs (separated by two line breaks); the k-th starts with W."
    n_par = int(kw.get("num_paragraphs") or 0)
    nth = int(kw.get("nth_paragraph") or 0)
    first = (kw.get("first_word") or "").strip().lower()
    paras = [p.strip() for p in re.split(r"\n\n", response) if p.strip()]
    if len(paras) != n_par or not (1 <= nth <= len(paras)):
        return False
    words = re.findall(r"[\w']+", paras[nth - 1].lower())
    return bool(words) and words[0] == first


# Map IFEval instruction IDs → verifier
_VERIFIERS: dict[str, callable] = {
    "length_constraints:number_words": _check_length_words,
    "length_constraints:number_sentences": _check_length_sentences,
    "length_constraints:number_paragraphs": _check_length_paragraphs,
    "length_constraints:nth_paragraph_first_word": _check_nth_paragraph_first_word,
    "keywords:existence": _check_keywords_existence,
    "keywords:frequency": _check_keyword_frequency,
    "keywords:forbidden_words": _check_keywords_forbidden,
    "keywords:letter_frequency": _check_letter_frequency,
    "language:response_language": _check_response_language,
    "change_case:english_capital": _check_change_case_capital,
    "change_case:english_lowercase": _check_change_case_lowercase,
    "change_case:capital_word_frequency": _check_capital_words_count,
    "punctuation:no_comma": _check_punctuation_no_comma,
    "startend:quotation": _check_startend_quotation,
    "startend:end_checker": _check_startend_end_phrase,
    "detectable_format:number_bullet_lists": _check_format_number_bullets,
    "detectable_format:number_highlighted_sections": _check_format_number_highlighted,
    "detectable_format:title": _check_format_title,
    "detectable_format:constrained_response": _check_format_constrained_response,
    "detectable_format:json_format": _check_format_json,
    "detectable_format:multiple_sections": _check_format_multiple_sections,
    "combination:two_responses": _check_combination_two_responses,
    "combination:repeat_prompt": _check_combination_repeat_prompt,
    "detectable_content:number_placeholders": _check_number_placeholders,
    "detectable_content:postscript": _check_postscript,
}


def _verify_response(
    response: str, instruction_ids: list, kwargs_list: list,
) -> tuple[bool, list[str]]:
    """Return (all_pass, unhandled_ids)."""
    unhandled = []
    for iid, kw in zip(instruction_ids, kwargs_list):
        verifier = _VERIFIERS.get(iid)
        if verifier is None:
            unhandled.append(iid)
            continue  # treat unhandled as pass — bias is towards optimistic
        try:
            if not verifier(response, **(kw or {})):
                return False, unhandled
        except Exception as e:
            # A verifier raising is a harness defect, never a model failure.
            # Scoring it False (as this used to) silently understated IFEval:
            # a null-relation kwarg crashed three verifiers and failed 14% of
            # prompts outright. Surface it and treat the instruction as
            # unhandled instead of blaming the model.
            print(f"  ⚠ IFEval verifier {iid} raised {type(e).__name__}: {e}")
            unhandled.append(iid)
    return True, unhandled


def _loose_clean(response: str) -> str:
    """Loose-mode preprocessing: strip leading 'Sure, here is...' boilerplate
    and outer code fences."""
    s = response.strip()
    s = re.sub(r"^(Sure|Here|Of course)[,!.]\s*[^\n]*\n+", "", s, count=1)
    if s.startswith("```"):
        s = re.sub(r"^```\w*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


@dataclass
class IFEvalResult:
    n_total: int
    n_strict_pass: int
    n_loose_pass: int
    strict_acc: float
    loose_acc: float
    unhandled_instructions: dict[str, int]  # iid -> count
    elapsed_sec: float

    def __str__(self) -> str:
        s = (
            f"IFEval prompt-level pass rate: "
            f"strict={self.strict_acc * 100:.1f}% ({self.n_strict_pass}/{self.n_total}), "
            f"loose={self.loose_acc * 100:.1f}% ({self.n_loose_pass}/{self.n_total}), "
            f"elapsed {self.elapsed_sec:.0f}s"
        )
        if self.unhandled_instructions:
            top = sorted(self.unhandled_instructions.items(),
                         key=lambda kv: -kv[1])[:3]
            s += "\n  unhandled instruction ids (treated as pass): " + \
                 ", ".join(f"{k}({v})" for k, v in top)
        return s


def evaluate_ifeval(
    model_path: str,
    n_samples: int | None = None,
    max_tokens: int = 512,
    seed: int = 42,
    reasoning: bool = False,
    repetition_penalty: float | None = None,
) -> IFEvalResult:
    """Run IFEval on the ``google/IFEval`` validation set.

    Args:
        model_path: Path or HF repo of the model.
        n_samples: Number of prompts to score. None → entire 540-sample set.
        max_tokens: Generation budget per prompt.
        seed: Reproducibility for sampling when ``n_samples < 540``.
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

    print(f"  Loading IFEval (validation split) …")
    ds = load_dataset("google/IFEval", split="train")  # IFEval ships only one split
    rows = list(ds)

    rng = np.random.RandomState(seed)
    if n_samples is not None and n_samples < len(rows):
        idxs = rng.choice(len(rows), size=n_samples, replace=False)
        rows = [rows[int(i)] for i in idxs]

    use_chat = bool(getattr(tokenizer, "chat_template", None))
    concise = {} if reasoning else concise_template_kwargs(tokenizer)
    sampler, logits_processors = resolve_decode(model, repetition_penalty)

    n_strict = n_loose = 0
    unhandled: dict[str, int] = {}

    for ex in tqdm(rows, desc="IFEval"):
        prompt_text = ex["prompt"]
        if use_chat:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False, add_generation_prompt=True, **concise,
            )
        else:
            prompt = prompt_text

        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=max_tokens, sampler=sampler, logits_processors=logits_processors, verbose=False,
        )
        reclaim()
        # Reduce channel-structured output (gpt-oss harmony / Gemma-4) to the
        # final answer so the reasoning trace doesn't get scored against the
        # instruction constraints.
        response = strip_to_final_channel(response)
        # Strip thinking block
        if "</think>" in response:
            response = response.split("</think>", 1)[1]

        iids = ex.get("instruction_id_list") or []
        kw_list = ex.get("kwargs") or [{} for _ in iids]

        strict_pass, _unh = _verify_response(response, iids, kw_list)
        loose_pass, _ = _verify_response(_loose_clean(response), iids, kw_list)
        for u in _unh:
            unhandled[u] = unhandled.get(u, 0) + 1
        n_strict += int(strict_pass)
        n_loose += int(loose_pass)

    n_total = len(rows)
    return IFEvalResult(
        n_total=n_total,
        n_strict_pass=n_strict,
        n_loose_pass=n_loose,
        strict_acc=n_strict / max(n_total, 1),
        loose_acc=n_loose / max(n_total, 1),
        unhandled_instructions=unhandled,
        elapsed_sec=time.time() - t0,
    )
