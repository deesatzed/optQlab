"""BFCL-V3 — Berkeley Function Calling Leaderboard evaluation.

The full BFCL benchmark spans many sub-categories (simple / parallel /
multi-turn / executable / live / etc.). For mlx-optiq's quantization
eval we run the **simple non-live** subset:

  * Single-turn user query
  * One or more tool definitions
  * Ground truth: a single function call with name + arguments

Scoring is **AST equivalence**: parse the model's emitted call, compare
function name (exact match) and argument names + values (after
normalization — int/float coercion, string-vs-number tolerance for
canonical types).

This is the canonical BFCL "simple" metric, which Unsloth and most
quantization-eval pipelines report. The full multi-turn / live / agentic
categories require a separate eval framework that's out of scope here.
"""

from __future__ import annotations

import json
from ._harmony import concise_template_kwargs
import os
import re
import time
from dataclasses import dataclass

import numpy as np


def _extract_tool_call(response: str) -> dict | None:
    """Extract a single tool call from a model's response.

    Recognizes the four formats common in modern frontier models:
      * ``<tool_call>{"name": ..., "arguments": ...}</tool_call>``
        (Hermes / Qwen3 / many others)
      * ``<function=NAME>...</function>`` with ``<parameter=KEY>VALUE</parameter>``
        children (Qwen3.6 style)
      * ``<|tool_call>call:NAME{k:v,k:v,...}<tool_call|>`` with ``<|"|>``
        as the string-quote token (Gemma-4 style)
      * Plain JSON object with ``name`` + ``arguments`` keys at top level
    Returns ``{"name": str, "arguments": dict}`` or ``None`` if none found.
    """
    # Strip thinking block if present (Gemma-4 emits <|channel>thought ... </channel|>)
    text = response
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    if "<channel|>" in text:
        # Gemma-4 closes its thought channel with <channel|>; everything after
        # is the actual response.
        text = text.split("<channel|>", 1)[1]

    # Format 0: gpt-oss harmony — the call lands in the *commentary* channel as
    #   <|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{ARGS}<|call|>
    m = re.search(r"to=functions\.([\w.\-]+)\b", text)
    if m:
        name = m.group(1)
        after = text[m.end():]
        jm = re.search(r"<\|message\|>\s*(\{.*)", after, re.DOTALL)
        if jm:
            blob = jm.group(1)
            for stop in ("<|call|>", "<|end|>", "<|return|>"):
                blob = blob.split(stop)[0]
            try:
                args, _ = json.JSONDecoder().raw_decode(blob.strip())
                if isinstance(args, dict):
                    return {"name": name, "arguments": args}
            except json.JSONDecodeError:
                pass

    # Format 1: <tool_call>...</tool_call>
    m = re.search(r"<tool_call>\s*(.+?)\s*</tool_call>", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "name" in obj:
                args = obj.get("arguments", obj.get("parameters", {}))
                if isinstance(args, str):
                    args = json.loads(args)
                return {"name": obj["name"], "arguments": args or {}}
        except (json.JSONDecodeError, TypeError):
            pass

    # Format 1b: Laguna (poolside) — <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>
    # The name sits bare right after the tag and args are <arg_key>/<arg_value>
    # pairs, not JSON, so Format 1's json.loads fails and every call reads as
    # no_call — the same harness-bug-not-capability zero that Format 4 fixed for
    # Devstral. The leading char is a name (not '{'), which is how this is told
    # apart from the JSON form above.
    m = re.search(r"<tool_call>\s*([A-Za-z_][\w.\-]*)\s*(<arg_key>.*?</arg_value>)?\s*</tool_call>",
                  text, re.DOTALL)
    if m:
        name = m.group(1).strip()
        args = {}
        for pm in re.finditer(
            r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
            m.group(2) or "", re.DOTALL,
        ):
            raw = pm.group(2).strip()
            try:
                args[pm.group(1).strip()] = json.loads(raw)  # coerce numbers/bools/null
            except (json.JSONDecodeError, ValueError):
                args[pm.group(1).strip()] = raw
        return {"name": name, "arguments": args}

    # Format 2: <function=NAME><parameter=K>V</parameter>...</function>
    m = re.search(r"<function=([^>\s]+)>(.*?)</function>", text, re.DOTALL)
    if m:
        name = m.group(1).strip()
        body = m.group(2)
        args = {}
        for pm in re.finditer(
            r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", body, re.DOTALL
        ):
            key = pm.group(1).strip()
            val = pm.group(2).strip()
            args[key] = val
        return {"name": name, "arguments": args}

    # Format 3: Gemma-4 — <|tool_call>call:NAME{k:v,...}<tool_call|>
    # The model emits <|"|> as a string-quote token; replace with " before
    # parsing values.
    m = re.search(
        r"<\|tool_call>\s*call:(\S+?)\s*(?:\{(.*?)\})?\s*<tool_call\|>",
        text, re.DOTALL,
    )
    if m:
        name = m.group(1).strip().rstrip("{")
        body = (m.group(2) or "").replace('<|"|>', '"')
        args = _parse_gemma_args(body)
        return {"name": name, "arguments": args}

    # Format 4: Mistral / Devstral — [TOOL_CALLS]NAME[ARGS]{json}, and the older
    # array form [TOOL_CALLS][{"name": ..., "arguments": {...}}].
    #
    # Its absence scored Devstral-Small-2-24B a hard 0/200 here while the same
    # model tool-called correctly in production, which is how it was found: a
    # zero on a model whose whole point is agentic tool use is a harness bug,
    # not a capability result. Unlike the other families there is no closing
    # marker -- the call is terminated by EOS -- so match to the end of the
    # segment rather than to a sentinel.
    m = re.search(
        r"\[TOOL_CALLS\]\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\[ARGS\]\s*(\{.*?\})"
        r"(?=\s*(?:\[TOOL_CALLS\]|\[/?TOOL_RESULTS\]|</s>|$))",
        text, re.DOTALL,
    )
    if m:
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            args = {}
        return {"name": m.group(1),
                "arguments": args if isinstance(args, dict) else {}}

    m = re.search(r"\[TOOL_CALLS\]\s*(\[.*?\])(?=\s*(?:\[/?TOOL_RESULTS\]|</s>|$))",
                  text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(1))
        except json.JSONDecodeError:
            arr = None
        if isinstance(arr, list) and arr and isinstance(arr[0], dict):
            first = arr[0]
            args = first.get("arguments", first.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if "name" in first:
                return {"name": first["name"],
                        "arguments": args if isinstance(args, dict) else {}}

    # Format 5: a bare {"name": ..., "arguments": ...} object in ANY wrapper
    # (or none). Scan every '{' and try to decode a balanced JSON object there
    # via raw_decode — this handles nested arguments (e.g.
    # {"arguments": {"number": 5}}) that the old single-line regex could not,
    # and tolerates hallucinated wrappers (some models emit valid call JSON
    # inside <cancellable>…</cancellable> or other non-standard tags).
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj and (
            "arguments" in obj or "parameters" in obj
        ):
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return {"name": obj["name"], "arguments": args if isinstance(args, dict) else {}}

    return None


def _parse_gemma_args(body: str) -> dict:
    """Parse Gemma-4's tool-call argument body — a comma-separated list of
    ``key:value`` pairs where values are JSON-ish (numbers, "strings", true,
    [arrays], {nested}).

    Not a perfect parser, but handles flat dicts which is what BFCL-simple
    requires. We walk character by character to respect string quotes,
    bracket nesting, then JSON-decode each value.
    """
    args: dict = {}
    i = 0
    n = len(body)
    while i < n:
        # Skip whitespace + leading commas
        while i < n and body[i] in ", \t\n":
            i += 1
        if i >= n:
            break
        # Read key (up to colon)
        k_start = i
        while i < n and body[i] != ":":
            i += 1
        if i >= n:
            break
        key = body[k_start:i].strip().strip('"').strip("'")
        i += 1  # skip ':'
        # Skip whitespace
        while i < n and body[i] in " \t\n":
            i += 1
        # Read value — respect quotes and bracket nesting
        v_start = i
        depth = 0
        in_str = False
        str_q = ""
        while i < n:
            c = body[i]
            if in_str:
                if c == "\\" and i + 1 < n:
                    i += 2
                    continue
                if c == str_q:
                    in_str = False
                i += 1
                continue
            if c in ('"', "'"):
                in_str = True
                str_q = c
                i += 1
                continue
            if c in "[{":
                depth += 1
                i += 1
                continue
            if c in "]}":
                if depth == 0:
                    break
                depth -= 1
                i += 1
                continue
            if c == "," and depth == 0:
                break
            i += 1
        v_str = body[v_start:i].strip()
        try:
            args[key] = json.loads(v_str)
        except json.JSONDecodeError:
            # Bare token (variable name, unquoted string) → keep as-is
            args[key] = v_str.strip('"').strip("'")
    return args


def _canonicalize_value(v):
    """Normalize a value for comparison. Int/float-coerce numerics; lower-case
    strings only when a sibling enum is also strings (best-effort)."""
    if isinstance(v, str):
        s = v.strip()
        # Try numeric. Guard underscores: int("1_000") == 1000 in Python, so
        # "1_000" would wrongly match 1000.
        try:
            if "_" in s:
                raise ValueError
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except (ValueError, TypeError):
            pass
        # Try bool
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        # Try a bracketed collection: JSON first, then Python-literal (a model
        # that single-quotes an array — ['a', 'b'] — is not valid JSON but is a
        # correct answer; without this it stays a str and never matches a list).
        if s.startswith(("[", "{")):
            try:
                return _canonicalize_value(json.loads(s))
            except json.JSONDecodeError:
                pass
            try:
                import ast
                return _canonicalize_value(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                pass
        return s.strip()
    if isinstance(v, list):
        return [_canonicalize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _canonicalize_value(x) for k, x in v.items()}
    return v


def _tool_name(value) -> str:
    """A function name as a string, or "" if the model emitted something else.

    The prediction is untrusted: it is whatever the model wrote. dhara-250m
    emitted a *list* for "name", and ``.strip()`` on it raised AttributeError and
    killed the whole benchmark suite after MMLU, GSM8K and IFEval had already
    been paid for. A model producing nonsense must score zero, not crash the run.
    """
    return value.strip() if isinstance(value, str) else ""


def _schema_params_for(functions, fn_name: str) -> set | None:
    """Valid parameter names for ``fn_name`` from a BFCL function schema list,
    or ``None`` when the schema is unavailable (then no unexpected-param check)."""
    if not isinstance(functions, list):
        return None
    for f in functions:
        if isinstance(f, dict) and _tool_name(f.get("name")) == fn_name:
            props = (f.get("parameters") or {}).get("properties")
            if isinstance(props, dict):
                return set(props)
    return None


def _calls_match(predicted: dict, ground_truth: dict, functions=None) -> bool:
    """Compare two tool-call dicts. ``ground_truth`` may be:
      * a ``{name: {arg: value, ...}}`` mapping (BFCL native)
      * a ``{"name": ..., "arguments": ...}`` dict

    ``functions`` is the row's function schema; when given, a prediction that
    supplies an argument outside the function's parameter list is rejected, as
    official BFCL does. Without it the matcher only walked the ground-truth
    args, so a correct call with an extra hallucinated argument still passed.
    """
    if not isinstance(predicted, dict) or not predicted:
        return False

    pred_name = _tool_name(predicted.get("name"))
    if not pred_name:
        return False
    pred_args = _canonicalize_value(predicted.get("arguments", {}) or {})

    # Normalize ground truth shape
    if "name" in ground_truth and "arguments" in ground_truth:
        gt_name = _tool_name(ground_truth["name"])
        gt_args = _canonicalize_value(ground_truth["arguments"] or {})
    else:
        # BFCL native: {fn_name: {arg: value, ...}}
        if not ground_truth:
            return False
        gt_key = next(iter(ground_truth))
        gt_name = _tool_name(gt_key)
        gt_args = _canonicalize_value(ground_truth[gt_key] or {})

    if pred_name != gt_name:
        return False
    if not isinstance(pred_args, dict) or not isinstance(gt_args, dict):
        return False

    # Reject hallucinated parameters (official BFCL fails an "unexpected
    # parameter"). Checked against the function schema when available, so a
    # legitimate optional arg that isn't in the ground truth is not penalised.
    allowed = _schema_params_for(functions, gt_name)
    if allowed is not None and any(k not in allowed for k in pred_args):
        return False

    # All required ground-truth args must be present and match. BFCL stores
    # acceptable values as lists ([primary, alt1, alt2]); we accept the
    # prediction if it matches ANY list element.
    #
    # An arg is **optional** if its acceptable-values list includes ``''``
    # (empty-string sentinel BFCL uses for "may be omitted"). For optional
    # args, missing-from-prediction is also acceptable.
    for k, gt_v in gt_args.items():
        is_list = isinstance(gt_v, list)
        is_optional = is_list and any(
            (isinstance(x, str) and x == "") or x is None
            for x in gt_v
        )
        if k not in pred_args:
            if is_optional:
                continue
            return False
        pv = pred_args[k]
        if is_list:
            # BFCL wraps EVERY parameter's ground truth in a list of acceptable
            # values, and the prediction must match any one of them:
            #     "base":  [10]
            #     "unit":  ["units", ""]
            #     "stops": [["Santa Barbara", "Monterey"],
            #               ["Monterey", "Santa Barbara"]]
            #
            # For an array-typed argument each acceptable value is ITSELF a list.
            # This used to read `if is_list and not isinstance(pv, list)`, so a
            # list-valued prediction skipped the any() and fell through to
            # `_loose_eq(pv, gt_v)` -- comparing the predicted list against the
            # whole list of ALTERNATIVES. For "stops" both are length 2, so
            # _loose_eq zipped them and compared "Santa Barbara" against
            # ["Santa Barbara", "Monterey"]. False. Every array-typed argument
            # was unmatchable no matter what the model answered, and feeding
            # BFCL's own ground truth back in scored zero.
            #
            # 65 of the 400 BFCL-simple questions (16.2%) take an array argument,
            # so every BFCL score we have published is understated.
            if not any(_loose_eq(pv, gv) for gv in gt_v):
                return False
        else:
            if not _loose_eq(pv, gt_v):
                return False
    return True


def _loose_eq(a, b) -> bool:
    """Equality with int/float tolerance and case-insensitive strings."""
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_loose_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_loose_eq(a[k], b[k]) for k in a)
    return a == b


@dataclass
class BFCLResult:
    n_total: int
    n_correct: int
    accuracy: float
    n_no_call: int          # how often the model didn't emit a parseable call
    n_wrong_name: int       # right shape but wrong function name
    n_wrong_args: int       # right name but wrong arguments
    elapsed_sec: float

    def __str__(self) -> str:
        ci = 1.96 * np.sqrt(
            (self.accuracy * (1 - self.accuracy)) / max(self.n_total, 1)
        )
        return (
            f"BFCL-V3 simple AST accuracy: {self.n_correct}/{self.n_total} = "
            f"{self.accuracy * 100:.1f}% (95% CI ±{ci * 100:.1f}pp)\n"
            f"  failures: no_call={self.n_no_call}, "
            f"wrong_name={self.n_wrong_name}, wrong_args={self.n_wrong_args}, "
            f"elapsed {self.elapsed_sec:.0f}s"
        )


def _wrap_tools_for_chat(raw_tools) -> list:
    """Convert BFCL tool defs (list of {name, description, parameters}) into
    the OpenAI/Anthropic chat-template tools schema."""
    if isinstance(raw_tools, str):
        try:
            raw_tools = json.loads(raw_tools)
        except json.JSONDecodeError:
            raw_tools = []
    out = []
    for t in raw_tools or []:
        if not isinstance(t, dict):
            continue
        if "function" in t:
            out.append(t)
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return out


def _load_bfcl_v3_simple() -> list[dict]:
    """Fetch the BFCL-V3 simple split + ground-truth answers from HF directly.

    The gorilla-llm/Berkeley-Function-Calling-Leaderboard repo isn't a
    HF-datasets-compatible layout — it's a directory of JSONL files.
    Download the questions + ground-truth files via hf_hub_download and
    join on ``id``.
    """
    from huggingface_hub import hf_hub_download

    repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    q_path = hf_hub_download(repo_id=repo, filename="BFCL_v3_simple.json",
                             repo_type="dataset")
    a_path = hf_hub_download(repo_id=repo,
                             filename="possible_answer/BFCL_v3_simple.json",
                             repo_type="dataset")

    def _read_jsonl(p: str) -> list[dict]:
        rows = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    questions = _read_jsonl(q_path)
    answers = {a["id"]: a["ground_truth"] for a in _read_jsonl(a_path)}

    out = []
    for q in questions:
        gt = answers.get(q["id"])
        if gt is None:
            continue
        out.append({**q, "ground_truth": gt})
    return out


def evaluate_bfcl(
    model_path: str,
    n_samples: int = 200,
    max_tokens: int = 512,
    seed: int = 42,
    reasoning: bool = False,
    repetition_penalty: float | None = None,
) -> BFCLResult:
    """Run BFCL-V3 simple AST eval on a single-turn function-calling subset.

    Args:
        model_path: Path or HF repo of the model.
        n_samples: Number of simple-category questions to evaluate.
        max_tokens: Generation budget per question.
        seed: Reproducibility for sampling.
    """
    import mlx.core as mx
    from mlx_lm import load, generate
    from .decode import load_tolerant, reclaim, resolve_decode, seed_rng
    from tqdm import tqdm

    t0 = time.time()
    if os.path.isdir(model_path):
        model_path = os.path.abspath(model_path)
    print(f"  Loading model from {model_path} …")
    seed_rng(seed)   # diffusion decode starts from a RANDOM canvas
    model, tokenizer = load_tolerant(model_path)

    print(f"  Loading BFCL-V3 simple subset …")
    rows = _load_bfcl_v3_simple()
    rng = np.random.RandomState(seed)
    if n_samples is not None and n_samples < len(rows):
        idxs = rng.choice(len(rows), size=n_samples, replace=False)
        rows = [rows[int(i)] for i in idxs]

    use_chat = bool(getattr(tokenizer, "chat_template", None))
    concise = {} if reasoning else concise_template_kwargs(tokenizer)
    sampler, logits_processors = resolve_decode(model, repetition_penalty)

    n_correct = n_no_call = n_wrong_name = n_wrong_args = 0

    for ex in tqdm(rows, desc="BFCL-V3 simple"):
        # BFCL question shape: question is a list-of-list of messages
        question_field = ex.get("question") or ex.get("query") or ex.get("input")
        if isinstance(question_field, list) and question_field and isinstance(question_field[0], list):
            user_text = question_field[0][-1]["content"] if question_field[0] else ""
        elif isinstance(question_field, list) and question_field and isinstance(question_field[0], dict):
            user_text = question_field[-1].get("content", "")
        else:
            user_text = str(question_field) if question_field else ""

        tools = _wrap_tools_for_chat(ex.get("function") or ex.get("tools"))

        if use_chat and tools:
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tools=tools,
                    tokenize=False, add_generation_prompt=True, **concise,
                )
            except (TypeError, ValueError):
                # Some templates reject `tools`
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tokenize=False, add_generation_prompt=True, **concise,
                )
        else:
            # No chat template / no tools support — fall back to a textual
            # prompt asking the model to emit a tool_call block
            tools_json = json.dumps(tools, indent=2)
            prompt = (
                f"Available tools:\n{tools_json}\n\n"
                f"User: {user_text}\n\n"
                f"Respond with a single <tool_call>...</tool_call> block.\n"
            )

        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=max_tokens, sampler=sampler, logits_processors=logits_processors, verbose=False,
        )
        reclaim()
        predicted = _extract_tool_call(response)

        # Parse ground truth — BFCL stores `ground_truth` as a list-of-dict
        gt_raw = ex.get("ground_truth") or ex.get("answers") or ex.get("answer")
        if isinstance(gt_raw, str):
            try:
                gt_raw = json.loads(gt_raw)
            except json.JSONDecodeError:
                pass
        if isinstance(gt_raw, list) and gt_raw:
            gt = gt_raw[0]
        else:
            gt = gt_raw or {}

        if predicted is None:
            n_no_call += 1
            continue

        if not _calls_match(predicted, gt, functions=ex.get("function")):
            # Categorize the failure. Everything here is untrusted model output,
            # so it goes through _tool_name: a prediction whose "name" is a list
            # (dhara-250m emitted exactly that) is a wrong name, not a crash.
            if isinstance(predicted, dict) and "name" in predicted:
                if "name" in gt:
                    gt_name = _tool_name(gt["name"])
                elif isinstance(gt, dict) and gt:
                    gt_name = _tool_name(next(iter(gt)))
                else:
                    gt_name = ""
                if _tool_name(predicted["name"]) != gt_name:
                    n_wrong_name += 1
                else:
                    n_wrong_args += 1
            continue

        n_correct += 1

    n_total = len(rows)
    return BFCLResult(
        n_total=n_total,
        n_correct=n_correct,
        accuracy=n_correct / max(n_total, 1),
        n_no_call=n_no_call,
        n_wrong_name=n_wrong_name,
        n_wrong_args=n_wrong_args,
        elapsed_sec=time.time() - t0,
    )
