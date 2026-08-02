"""Fourteen dataset-building templates.

Each template takes some user input + an API client (pointing at our own
``optiq serve`` when LLM-driven generation is needed) and produces
JSONL files compatible with the formats the trainers accept:

  - {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
  - {"prompt": "...", "completion": "..."}
  - {"prompt": "...", "chosen": "...", "rejected": "..."}   (DPO)
  - {"text": "..."}
  - {"image": "...", "prompt": "...", "completion": "..."}   (VLM, for
    `optiq lora train --vision`; images standardized to a uniform canvas)

Templates write to ``<output_dir>/train.jsonl`` (and optionally
``valid.jsonl``); the fine-tune wizard can point straight at the parent.

Heavy LLM-driven templates (style transfer, self-instruct,
prompt reconstruction) defer to data-designer when it's importable; the
simpler ones don't need it.

Templates are listed non-LLM first, then LLM-driven, so the picker shows
fast/local templates before ones that need a served model.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass
class TemplateDef:
    id: str
    label: str
    description: str
    output_format: str   # "messages" / "prompt_completion" / "dpo" / "text"
    needs_llm: bool      # uses optiq serve for generation
    fields: list[dict]   # for the UI: [{name, type, label, hint?, required?}]


TEMPLATES: list[TemplateDef] = [
    # ---- non-LLM templates (fast, local) ----
    TemplateDef(
        id="sft_qa_pairs",
        label="SFT from QA pairs",
        description="Paste Q/A pairs or upload a .txt with `Q: …\\nA: …` blocks. Produces chat-format JSONL.",
        output_format="messages",
        needs_llm=False,
        fields=[
            {"name": "pairs_text", "type": "textarea",
             "label": "Pairs (Q: / A: blocks)",
             "hint": "Q: What is OptiQ?\\nA: A mixed-precision quantizer …",
             "required": True},
        ],
    ),
    TemplateDef(
        id="dpo_pref_pairs",
        label="DPO from preference pairs",
        description="Upload CSV with columns `prompt,chosen,rejected`. Emits DPO JSONL.",
        output_format="dpo",
        needs_llm=False,
        fields=[
            {"name": "csv_text", "type": "textarea",
             "label": "CSV (header `prompt,chosen,rejected`)",
             "required": True},
        ],
    ),
    TemplateDef(
        id="code_completion",
        label="Code completion",
        description="Walks a directory's .py files, splits each function at a random midpoint into prompt/completion pairs.",
        output_format="prompt_completion",
        needs_llm=False,
        fields=[
            {"name": "src_dir", "type": "text",
             "label": "Source directory (containing .py files)", "required": True},
            {"name": "max_pairs", "type": "number",
             "label": "Max pairs to produce", "default": 500},
        ],
    ),
    TemplateDef(
        id="hf_dataset_import",
        label="Hugging Face dataset import",
        description=(
            "Pull a public dataset from the Hugging Face Hub by id, optionally "
            "filter rows by a column value, slice to a row cap, and emit the "
            "chosen output format. Stand-alone (no LLM call) and idempotent — "
            "use it as the first step of any pipeline that starts from a "
            "published corpus (EditLens, no_robots, dolly, your own dataset)."
        ),
        output_format="text",
        needs_llm=False,
        fields=[
            {"name": "hf_id", "type": "text",
             "label": "Dataset id",
             "hint": "e.g. pangram/editlens_iclr or HuggingFaceH4/no_robots",
             "required": True},
            {"name": "config", "type": "text",
             "label": "Config / subset (optional)",
             "hint": "Some datasets have multiple configs (e.g. wikitext-2-raw-v1)."},
            {"name": "split", "type": "text",
             "label": "Split", "default": "train"},
            {"name": "text_column", "type": "text",
             "label": "Text column",
             "hint": "Field on each row that holds the body text.",
             "default": "text"},
            {"name": "label_column", "type": "text",
             "label": "Filter column (optional)",
             "hint": "e.g. text_type for EditLens. Leave blank for no filter."},
            {"name": "label_filter", "type": "text",
             "label": "Filter value (optional)",
             "hint": "Keep only rows where filter-column == this value. e.g. human_written."},
            {"name": "max_rows", "type": "number",
             "label": "Row cap (0 = no cap)", "default": 1000},
            {"name": "min_chars", "type": "number",
             "label": "Drop rows shorter than this many chars", "default": 200},
            {"name": "output_format", "type": "text",
             "label": "Output format",
             "hint": "One of: text | messages_user_only | prompt_completion. "
                     "'text' writes {\"text\": ...}; 'messages_user_only' writes a "
                     "messages row with the text as the user turn (downstream "
                     "templates can read either).",
             "default": "text"},
        ],
    ),
    TemplateDef(
        id="vlm_image_text",
        label="VLM image+text (vision fine-tune)",
        description=(
            "Import an image+text dataset from the Hugging Face Hub (ChartQA, "
            "DocVQA, LaTeX-OCR, or your own) and standardize it for vision LoRA. "
            "Every image is letterboxed to one fixed square canvas (uniform "
            "image shape is what keeps OptiQ's vision LoRA memory bounded on "
            "Apple Silicon), then emitted as {image, prompt, completion} JSONL "
            "that `optiq lora train --vision` reads directly."
        ),
        output_format="vlm",
        needs_llm=False,
        fields=[
            {"name": "hf_id", "type": "text",
             "label": "Dataset id",
             "hint": "e.g. HuggingFaceM4/ChartQA or unsloth/LaTeX_OCR",
             "required": True},
            {"name": "config", "type": "text",
             "label": "Config / subset (optional)"},
            {"name": "split", "type": "text", "label": "Split", "default": "train"},
            {"name": "image_column", "type": "text",
             "label": "Image column", "default": "image"},
            {"name": "prompt_column", "type": "text",
             "label": "Question/prompt column (optional)",
             "hint": "e.g. query for ChartQA. Leave blank to use the fixed "
                     "prompt below for every row (e.g. LaTeX-OCR captioning)."},
            {"name": "fixed_prompt", "type": "text",
             "label": "Fixed prompt (used when no prompt column)",
             "default": "Describe this image."},
            {"name": "answer_column", "type": "text",
             "label": "Answer / target column",
             "hint": "e.g. label for ChartQA, text for LaTeX-OCR.",
             "default": "label"},
            {"name": "instruction", "type": "text",
             "label": "Instruction suffix (optional)",
             "hint": "Appended to every prompt, e.g. "
                     "'Answer with just the value.'"},
            {"name": "image_size", "type": "number",
             "label": "Standardize to (px square)", "default": 512},
            {"name": "max_rows", "type": "number",
             "label": "Row cap (0 = no cap)", "default": 2000},
        ],
    ),
    TemplateDef(
        id="format_conversion",
        label="Format conversion",
        description="Upload existing JSONL in any shape + a key mapping. Emits OptiQ-expected JSONL.",
        output_format="messages",
        needs_llm=False,
        fields=[
            {"name": "input_jsonl", "type": "textarea",
             "label": "Existing JSONL", "required": True},
            {"name": "user_key", "type": "text", "label": "user field", "default": "input"},
            {"name": "assistant_key", "type": "text", "label": "assistant field", "default": "output"},
        ],
    ),
    # ---- LLM-driven templates (need a served model) ----
    TemplateDef(
        id="style_transfer",
        label="Style transfer",
        description="Provide reference samples and raw text. Uses the served model to rewrite in the reference style.",
        output_format="prompt_completion",
        needs_llm=True,
        fields=[
            {"name": "reference_samples", "type": "textarea",
             "label": "Reference style samples (separated by ---)", "required": True},
            {"name": "raw_text", "type": "textarea",
             "label": "Raw text to rewrite (one paragraph per row)", "required": True},
        ],
    ),
    TemplateDef(
        id="self_instruct",
        label="Self-instruct expansion",
        description="Upload seed instructions; uses the served model (via data-designer when available) to generate K variants per seed.",
        output_format="messages",
        needs_llm=True,
        fields=[
            {"name": "seeds", "type": "textarea",
             "label": "Seed instructions (one per line)", "required": True},
            {"name": "variants_per_seed", "type": "number",
             "label": "Variants per seed", "default": 5},
        ],
    ),
    TemplateDef(
        id="prompt_reconstruction",
        label="Prompt reconstruction",
        description=(
            "Build (AI draft → target) training pairs by working backwards: "
            "for each target paragraph the served model infers a likely prompt "
            "and writes a generic AI draft. The assistant target is the original "
            "paragraph verbatim, so facts and formatting are preserved by construction."
        ),
        output_format="messages",
        needs_llm=True,
        fields=[
            {"name": "target_text", "type": "textarea",
             "label": "Target paragraphs (separated by blank lines)",
             "hint": "Paste the text you want the model to learn to produce. "
                     "Posts, memos, brand-voice samples, edited drafts, etc.",
             "required": True},
            {"name": "style", "type": "text",
             "label": "Style label",
             "hint": "Free text, surfaced in the system prompt.",
             "default": "direct technical blog"},
            {"name": "tone", "type": "text",
             "label": "Tone", "default": "analytical, clear, non-corporate"},
            {"name": "preserve", "type": "text",
             "label": "Things to preserve (comma-separated)",
             "default": "facts, names, numbers, URLs, citations, code blocks, quotes"},
            {"name": "avoid", "type": "text",
             "label": "Things to avoid (comma-separated)",
             "default": "em dashes, generic transitions, marketing language"},
        ],
    ),
    TemplateDef(
        id="multi_turn_chat",
        label="Multi-turn chat synthesis",
        description=(
            "Take seed user prompts and a persona; expand each into an "
            "N-turn synthetic user/assistant conversation by alternating "
            "model-as-assistant and model-as-followup-user calls."
        ),
        output_format="messages",
        needs_llm=True,
        fields=[
            {"name": "seeds", "type": "textarea",
             "label": "Seed user prompts (one per line)", "required": True},
            {"name": "turns", "type": "number",
             "label": "Total turns per conversation (user + assistant pairs)",
             "default": 4},
            {"name": "persona", "type": "text",
             "label": "Assistant persona (system prompt)",
             "default": "You are a helpful, concise assistant."},
            {"name": "user_persona", "type": "text",
             "label": "Followup-user persona",
             "hint": "How follow-up questions should sound.",
             "default": "a curious developer probing for specifics"},
        ],
    ),
    TemplateDef(
        id="tool_use_traces",
        label="Tool-use traces",
        description=(
            "Generate (user, tool_call, tool_result, final) training traces "
            "in OpenAI tool-call shape. Provide a list of tool schemas plus "
            "scenario prompts; the model picks a tool, you supply a mocked "
            "result, the model writes the final answer."
        ),
        output_format="messages",
        needs_llm=True,
        fields=[
            {"name": "tools_json", "type": "textarea",
             "label": "Tool schemas (OpenAI tools format, JSON array)",
             "hint": "Paste a JSON array of {type, function:{name,description,parameters}}.",
             "required": True},
            {"name": "scenarios", "type": "textarea",
             "label": "User scenarios (one per line)", "required": True},
            {"name": "mock_results", "type": "textarea",
             "label": "Mocked tool results (one per line, matches scenarios order)",
             "hint": "Free text; gets passed back as the tool_result content.",
             "required": True},
        ],
    ),
    TemplateDef(
        id="rag_qa",
        label="RAG Q/A from documents",
        description=(
            "Chunk pasted text into passages; for each passage the model "
            "writes a question whose answer is grounded in that passage and "
            "then writes the grounded answer. Outputs messages-format rows "
            "where the passage is in a system message and the question/answer "
            "in user/assistant turns."
        ),
        output_format="messages",
        needs_llm=True,
        fields=[
            {"name": "documents", "type": "textarea",
             "label": "Source documents (paragraphs separated by blank lines)",
             "required": True},
            {"name": "questions_per_chunk", "type": "number",
             "label": "Questions per chunk", "default": 2},
            {"name": "min_chunk_chars", "type": "number",
             "label": "Min chunk size (chars)", "default": 200},
        ],
    ),
    TemplateDef(
        id="cot_synthesis",
        label="Reasoning trace (CoT) synthesis",
        description=(
            "For each question, the served model emits a step-by-step "
            "<think> trace followed by the final answer. Output is messages-"
            "format with the reasoning preserved inside the assistant content "
            "as a `<think>...</think>` block, matching the Qwen3 / GPT-OSS "
            "reasoning convention."
        ),
        output_format="messages",
        needs_llm=True,
        fields=[
            {"name": "questions", "type": "textarea",
             "label": "Questions (one per line)", "required": True},
            {"name": "category", "type": "text",
             "label": "Domain hint",
             "default": "math, logic, planning, technical analysis"},
        ],
    ),
    TemplateDef(
        id="verified_code",
        label="Verified code generation",
        description=(
            "For each natural-language spec, the served model writes Python "
            "code AND a set of `assert` checks. We run the assertions in the "
            "sandbox and keep only the pairs that pass. Output is messages-"
            "format; failed pairs are dropped before writing the JSONL."
        ),
        output_format="messages",
        needs_llm=True,
        fields=[
            {"name": "specs", "type": "textarea",
             "label": "Natural-language specs (one per line)",
             "hint": "e.g. `write a function that returns the n-th Fibonacci number`",
             "required": True},
            {"name": "language", "type": "text",
             "label": "Language", "default": "python"},
        ],
    ),
]


def get_template(template_id: str) -> TemplateDef | None:
    return next((t for t in TEMPLATES if t.id == template_id), None)


# ---------------------------------------------------------------------------
# Generator implementations
# ---------------------------------------------------------------------------


def generate(
    template_id: str,
    inputs: dict[str, Any],
    output_dir: Path,
    emit: Callable[[dict], None],
    api_url: str | None = None,
    auth_token: str | None = None,
    model_name: str | None = None,
) -> dict:
    """Run the named template with ``inputs`` → write JSONL into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if template_id == "sft_qa_pairs":
        rows = _gen_sft_qa(inputs)
    elif template_id == "dpo_pref_pairs":
        rows = _gen_dpo_pairs(inputs)
    elif template_id == "code_completion":
        rows = _gen_code_completion(inputs)
    elif template_id == "format_conversion":
        rows = _gen_format_conversion(inputs)
    elif template_id == "hf_dataset_import":
        rows = _gen_hf_dataset_import(inputs, emit)
    elif template_id == "vlm_image_text":
        rows = _gen_vlm_image_text(inputs, output_dir, emit)
    elif template_id == "style_transfer":
        rows = _gen_style_transfer(inputs, api_url, auth_token, model_name, emit)
    elif template_id == "self_instruct":
        rows = _gen_self_instruct(inputs, api_url, auth_token, model_name, emit)
    elif template_id == "prompt_reconstruction":
        rows = _gen_prompt_reconstruction(inputs, api_url, auth_token, model_name, emit)
    elif template_id == "multi_turn_chat":
        rows = _gen_multi_turn_chat(inputs, api_url, auth_token, model_name, emit)
    elif template_id == "tool_use_traces":
        rows = _gen_tool_use_traces(inputs, api_url, auth_token, model_name, emit)
    elif template_id == "rag_qa":
        rows = _gen_rag_qa(inputs, api_url, auth_token, model_name, emit)
    elif template_id == "cot_synthesis":
        rows = _gen_cot_synthesis(inputs, api_url, auth_token, model_name, emit)
    elif template_id == "verified_code":
        rows = _gen_verified_code(inputs, api_url, auth_token, model_name, emit)
    else:
        raise ValueError(f"unknown template {template_id!r}")

    rows = list(rows)
    n = len(rows)
    if n == 0:
        raise ValueError("generator produced 0 rows — check your inputs")

    # 90/10 split
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    split_idx = max(1, int(n * 0.9))

    with train_path.open("w") as f:
        for r in rows[:split_idx]:
            f.write(json.dumps(r) + "\n")
    with valid_path.open("w") as f:
        for r in rows[split_idx:] or rows[-1:]:
            f.write(json.dumps(r) + "\n")

    emit({"type": "stage", "stage": "done", "progress": 1.0,
          "message": f"Wrote {n} rows.",
          "output_dir": str(output_dir),
          "n_train": split_idx,
          "n_valid": max(1, n - split_idx)})

    return {"output_dir": str(output_dir),
            "n_total": n,
            "n_train": split_idx,
            "n_valid": max(1, n - split_idx)}


def _gen_sft_qa(inputs: dict) -> Iterable[dict]:
    text = (inputs.get("pairs_text") or "").strip()
    if not text:
        return []
    rows = []
    # Split on blank lines; each block expected to contain Q: / A:
    for block in re.split(r"\n\s*\n", text):
        q_match = re.search(r"^\s*Q\s*:\s*(.+)", block, re.MULTILINE | re.IGNORECASE)
        a_match = re.search(r"^\s*A\s*:\s*(.+)", block, re.MULTILINE | re.IGNORECASE | re.DOTALL)
        if q_match and a_match:
            rows.append({
                "messages": [
                    {"role": "user", "content": q_match.group(1).strip()},
                    {"role": "assistant", "content": a_match.group(1).strip()},
                ]
            })
    return rows


def _gen_dpo_pairs(inputs: dict) -> Iterable[dict]:
    import csv
    import io
    text = (inputs.get("csv_text") or "").strip()
    if not text:
        return []
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for r in reader:
        if r.get("prompt") and r.get("chosen") and r.get("rejected"):
            rows.append({
                "prompt": r["prompt"],
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            })
    return rows


def _gen_code_completion(inputs: dict) -> Iterable[dict]:
    src = Path(inputs.get("src_dir") or "").expanduser()
    max_pairs = int(inputs.get("max_pairs") or 500)
    if not src.is_dir():
        raise ValueError(f"src_dir not a directory: {src}")

    rng = random.Random(42)
    rows = []
    for py in src.rglob("*.py"):
        try:
            text = py.read_text(errors="ignore")
        except Exception:
            continue
        # Split on `def ` / `class ` boundaries — simple but works
        chunks = re.split(r"(?=^(?:def |class |async def ))", text, flags=re.MULTILINE)
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < 30:
                continue
            split_at = rng.randint(len(chunk) // 4, max(len(chunk) // 4 + 1, len(chunk) * 3 // 4))
            rows.append({
                "prompt": chunk[:split_at],
                "completion": chunk[split_at:],
            })
            if len(rows) >= max_pairs:
                return rows
    return rows


def _gen_format_conversion(inputs: dict) -> Iterable[dict]:
    text = (inputs.get("input_jsonl") or "").strip()
    user_key = inputs.get("user_key") or "input"
    asst_key = inputs.get("assistant_key") or "output"
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        u = obj.get(user_key)
        a = obj.get(asst_key)
        if u is None or a is None:
            continue
        rows.append({
            "messages": [
                {"role": "user", "content": str(u)},
                {"role": "assistant", "content": str(a)},
            ]
        })
    return rows


def _gen_hf_dataset_import(inputs: dict, emit) -> Iterable[dict]:
    """Pull a public HF dataset by id, optionally filter, slice, and emit rows.

    Generic counterpart to Unsloth Studio's Hub picker and NeMo Data Designer's
    ``HuggingFaceSeedSource``. Stand-alone (no served model), idempotent, and
    intended as a first-stage feed into the LLM-driven templates downstream
    (prompt_reconstruction, style_transfer, rag_qa, etc).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "hf_dataset_import requires the `datasets` library. "
            "Install with: pip install datasets"
        ) from e

    hf_id = (inputs.get("hf_id") or "").strip()
    if not hf_id:
        return []
    config = (inputs.get("config") or "").strip() or None
    split = (inputs.get("split") or "train").strip()
    text_column = (inputs.get("text_column") or "text").strip()
    label_column = (inputs.get("label_column") or "").strip() or None
    label_filter = (inputs.get("label_filter") or "").strip() or None
    max_rows = int(inputs.get("max_rows") or 0)
    min_chars = int(inputs.get("min_chars") or 0)
    output_format = (inputs.get("output_format") or "text").strip()

    emit({"type": "stage", "stage": "loading",
          "message": f"Loading {hf_id} ({split})…", "progress": 0.05})

    ds = load_dataset(hf_id, config, split=split) if config else load_dataset(hf_id, split=split)

    total = len(ds)
    emit({"type": "stage", "stage": "filtering",
          "message": f"Loaded {total} rows. Filtering…", "progress": 0.20})

    if text_column not in ds.column_names:
        raise ValueError(
            f"text_column={text_column!r} not in dataset. "
            f"Available columns: {list(ds.column_names)}"
        )
    if label_column and label_column not in ds.column_names:
        raise ValueError(
            f"label_column={label_column!r} not in dataset. "
            f"Available columns: {list(ds.column_names)}"
        )

    rows: list[dict] = []
    kept = 0
    rejected_short = 0
    rejected_filter = 0
    for i, row in enumerate(ds):
        if max_rows and kept >= max_rows:
            break
        text = (row.get(text_column) or "")
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if min_chars and len(text) < min_chars:
            rejected_short += 1
            continue
        if label_column and label_filter:
            if str(row.get(label_column, "")) != label_filter:
                rejected_filter += 1
                continue

        if output_format == "messages_user_only":
            rows.append({"messages": [{"role": "user", "content": text}]})
        elif output_format == "prompt_completion":
            rows.append({"prompt": text, "completion": ""})
        else:  # default: text
            rows.append({"text": text})
        kept += 1

        if kept and kept % 500 == 0:
            emit({"type": "stage", "stage": "filtering",
                  "message": f"kept {kept} rows…",
                  "progress": 0.20 + 0.70 * min(1.0, kept / max(1, max_rows or total))})

    emit({"type": "stage", "stage": "writing",
          "message": (f"kept {kept}; dropped {rejected_short} short, "
                      f"{rejected_filter} filtered out"),
          "progress": 0.95})
    return rows


def _gen_vlm_image_text(inputs: dict, output_dir: Path, emit) -> Iterable[dict]:
    """Import an image+text HF dataset and standardize it for vision LoRA.

    Streams the dataset, letterboxes every image to a fixed square canvas (the
    standardize step — uniform shape keeps OptiQ's vision LoRA memory bounded on
    Apple Silicon), saves the PNGs under ``<output_dir>/images``, and emits
    ``{image, prompt, completion}`` rows that ``optiq lora train --vision`` reads.
    Self-contained (no served model); reuses the trainer's own ``letterbox``.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "vlm_image_text requires the `datasets` library. "
            "Install with: pip install datasets"
        ) from e
    from optiq.vlm.lora import letterbox

    hf_id = (inputs.get("hf_id") or "").strip()
    if not hf_id:
        return []
    config = (inputs.get("config") or "").strip() or None
    split = (inputs.get("split") or "train").strip()
    image_col = (inputs.get("image_column") or "image").strip()
    prompt_col = (inputs.get("prompt_column") or "").strip() or None
    fixed_prompt = (inputs.get("fixed_prompt") or "Describe this image.").strip()
    answer_col = (inputs.get("answer_column") or "label").strip()
    instruction = (inputs.get("instruction") or "").strip()
    size = max(64, int(inputs.get("image_size") or 768))
    max_rows = int(inputs.get("max_rows") or 0)

    emit({"type": "stage", "stage": "loading",
          "message": f"Streaming {hf_id} ({split})…", "progress": 0.05})
    ds = (load_dataset(hf_id, config, split=split, streaming=True) if config
          else load_dataset(hf_id, split=split, streaming=True))

    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n = 0
    for row in ds:
        if max_rows and n >= max_rows:
            break
        img = row.get(image_col)
        if img is None:
            continue
        ans = row.get(answer_col)
        if isinstance(ans, list):
            ans = ans[0] if ans else ""
        ans = "" if ans is None else str(ans).strip()
        if not ans:
            continue
        question = (str(row.get(prompt_col)).strip() if prompt_col
                    and row.get(prompt_col) else fixed_prompt) or fixed_prompt
        prompt = (question + ("\n" + instruction if instruction else "")).strip()
        try:
            canvas = letterbox(img, size)
        except Exception:
            continue
        path = img_dir / f"{n:06d}.png"
        canvas.save(path)
        rows.append({"image": str(path.resolve()),
                     "prompt": prompt, "completion": ans})
        n += 1
        if n % 200 == 0:
            emit({"type": "stage", "stage": "standardizing",
                  "message": f"standardized {n} images to {size}×{size}px…",
                  "progress": 0.10 + 0.80 * min(1.0, n / max(1, max_rows or n + 1))})

    emit({"type": "stage", "stage": "writing",
          "message": f"prepared {n} image+text rows ({size}px uniform)",
          "progress": 0.95})
    return rows


def _gen_style_transfer(inputs: dict, api_url, auth_token, model_name, emit) -> Iterable[dict]:
    refs = (inputs.get("reference_samples") or "").strip()
    raw = (inputs.get("raw_text") or "").strip()
    if not (refs and raw and api_url):
        return []
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    rows = []
    for i, p in enumerate(paragraphs):
        emit({"type": "stage", "stage": "generating",
              "progress": 0.1 + 0.85 * (i / max(len(paragraphs), 1)),
              "message": f"rewriting paragraph {i+1}/{len(paragraphs)}"})
        prompt = (
            "Rewrite the text below in the same style and tone as these reference samples.\n\n"
            f"REFERENCES:\n{refs}\n\nTEXT TO REWRITE:\n{p}\n\nREWRITTEN:"
        )
        out = _llm_call(api_url, auth_token, model_name, prompt, max_tokens=512)
        rows.append({"prompt": p, "completion": out})
    return rows


def _gen_self_instruct(inputs: dict, api_url, auth_token, model_name, emit) -> Iterable[dict]:
    seeds = [s.strip() for s in (inputs.get("seeds") or "").splitlines() if s.strip()]
    k = int(inputs.get("variants_per_seed") or 5)
    if not seeds or not api_url:
        return []
    rows = []
    for i, seed in enumerate(seeds):
        emit({"type": "stage", "stage": "generating",
              "progress": 0.1 + 0.85 * (i / max(len(seeds), 1)),
              "message": f"seed {i+1}/{len(seeds)}"})
        prompt = (
            f"Generate {k} new instructions in the same spirit as the example. "
            "Return them as a JSON array of strings, no commentary. Example "
            'output format: ["instruction one", "instruction two"]\n\n'
            f"Example: {seed}\n\nArray:"
        )
        out = _llm_call(api_url, auth_token, model_name, prompt, max_tokens=512)
        variants = _parse_instruction_list(out, k)
        for v in variants[:k]:
            rows.append({
                "messages": [
                    {"role": "user", "content": str(v)},
                    {"role": "assistant", "content": ""},  # to be filled by a later pass
                ]
            })
    return rows


def _parse_instruction_list(text: str, k: int) -> list[str]:
    """Best-effort extract a list of K instructions from LLM output.

    Tries (in order): strict JSON array (shortest balanced [...]),
    brackets with comma-split (first [...] only, shortest match),
    newline-separated lines (with optional ``1.``/``-`` markers).
    """
    if not text:
        return []
    s = text.strip()
    # Find every balanced [ ... ] span and try strict JSON on each shortest
    # first. Stops greedy matching from consuming repetition artifacts.
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append((start, i + 1))
                start = -1
    # Try strict JSON on the first balanced span; fall through to looser
    # parsing only if it fails.
    for start_i, end_i in spans:
        try:
            v = json.loads(s[start_i:end_i])
            if isinstance(v, list):
                items = [str(x).strip() for x in v if str(x).strip()]
                if items:
                    return items[:k]
        except Exception:
            continue

    # First [ ... ] span, comma-split (handles unquoted strings).
    if spans:
        start_i, end_i = spans[0]
        inner = s[start_i + 1: end_i - 1]
        parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
        parts = [p for p in parts if p]
        if parts:
            return parts[:k]

    # Numbered / bulleted lines
    out: list[str] = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^(?:\d+[.)]|[-*])\s+", "", line)
        if line:
            out.append(line.strip('"').strip("'"))
    return out[:k]


def _gen_prompt_reconstruction(inputs: dict, api_url, auth_token, model_name, emit) -> Iterable[dict]:
    """Prompt-reconstruction training pairs.

    For each target paragraph:
      1. Call the served model to infer a plausible user prompt.
      2. Call the served model to write a generic AI draft from that prompt.
      3. Extract preservation locks (URLs, numbers, dates, citations, quotes, code)
         from the target paragraph.
      4. Emit one messages-format row whose assistant target is the original
         paragraph verbatim.

    The output is bit-compatible with the standalone prompt-reconstruction
    pipeline in the proofs repo's ``humanizer`` package, so a Labs run produces
    the same row shape as the CLI flow.
    """
    raw = (inputs.get("target_text") or inputs.get("human_text") or "").strip()
    style = (inputs.get("style") or "direct technical blog").strip()
    tone = (inputs.get("tone") or "analytical, clear").strip()
    preserve = (inputs.get("preserve") or "facts, names, numbers, URLs, citations").strip()
    avoid = (inputs.get("avoid") or "em dashes, generic transitions").strip()
    if not (raw and api_url):
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]

    # Cheap preservation extractors (mirror humanizer/extractors.py, kept here
    # so labs has zero external deps and the template is self-contained).
    url_re = re.compile(r"https?://[^\s)\]>]+")
    pct_re = re.compile(r"\b\d+(?:\.\d+)?\s?%")
    cur_re = re.compile(r"(?:\$|€|£|¥|USD\s*|EUR\s*|GBP\s*)\s?[\d,]+(?:\.\d+)?(?:[KMBkmb])?")
    date_re = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|(?:19|20)\d{2}|(?:Q[1-4]|FY)\s?\d{2,4})\b")
    code_re = re.compile(r"`([^`\n]{1,80})`")
    cite_re = re.compile(r"\[\d{1,4}\]|\[[A-Z][A-Za-z\-]+(?:\s+et\s+al\.?)?,?\s*\d{4}[a-z]?\]")

    def locks_of(t: str) -> dict:
        return {
            "urls": url_re.findall(t),
            "percentages": pct_re.findall(t),
            "currencies": cur_re.findall(t),
            "dates": date_re.findall(t),
            "code": code_re.findall(t),
            "citations": cite_re.findall(t),
        }

    rows: list[dict] = []
    for i, para in enumerate(paragraphs):
        emit({"type": "stage", "stage": "generating",
              "progress": 0.05 + 0.90 * (i / max(len(paragraphs), 1)),
              "message": f"reconstruct {i+1}/{len(paragraphs)}"})

        # 1. infer prompt (short response, low temperature)
        infer_prompt = (
            "Given the human-written text below, infer a realistic user prompt "
            "that could have led an AI assistant to produce this kind of text. "
            "Capture topic, intent, audience, and depth. Do NOT quote the text.\n\n"
            f"HUMAN TEXT:\n{para}\n\nPROMPT:"
        )
        inferred = _llm_call(api_url, auth_token, model_name, infer_prompt, max_tokens=160).strip()
        if not inferred:
            inferred = "Write a piece on the topic implied by the chunk."

        # 2. generate AI-ish draft
        draft_prompt = (
            "Answer the following user request in a polished but generic AI-assistant style. "
            "Be coherent, use common AI transitions, slightly over-explain, keep similar length. "
            "Do not introduce errors.\n\n"
            f"USER REQUEST:\n{inferred}\n\nSTYLE HINT:\n{style} / {tone}\n\nANSWER:"
        )
        draft = _llm_call(api_url, auth_token, model_name, draft_prompt, max_tokens=800).strip()
        if not draft:
            continue

        lk = locks_of(para)
        sys_prompt = (
            "You are a controlled rewrite model. Rewrite AI-generated drafts into "
            "natural human-style prose while preserving meaning, facts, names, "
            "numbers, citations, URLs, quotes, and formatting. Do not add unsupported "
            "claims."
        )
        user_msg = (
            f"STYLE:\n{style}\n\nTONE:\n{tone}\n\n"
            f"PRESERVE:\n{preserve}\n\nAVOID:\n{avoid}\n\n"
            f"USER PROMPT:\n{inferred}\n\nSOURCE AI DRAFT:\n{draft}\n\n"
            "TASK:\nRewrite only the SOURCE AI DRAFT into natural human prose. "
            "Do not introduce new facts."
        )
        rows.append({
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": para},
            ],
            "metadata": {
                "inferred_prompt": inferred,
                "source_ai_draft": draft,
                "preservation_locks": lk,
                "style": style,
                "tone": tone,
            },
        })
    return rows


def _gen_multi_turn_chat(inputs, api_url, auth_token, model_name, emit) -> Iterable[dict]:
    """For each seed user prompt, alternate model-as-assistant and
    model-as-followup-user calls to build an N-turn conversation."""
    seeds = [s.strip() for s in (inputs.get("seeds") or "").splitlines() if s.strip()]
    turns = max(2, int(inputs.get("turns") or 4))
    persona = (inputs.get("persona") or
               "You are a helpful, concise assistant.").strip()
    user_persona = (inputs.get("user_persona") or
                    "a curious developer probing for specifics").strip()
    if not (seeds and api_url):
        return []

    rows = []
    for i, seed in enumerate(seeds):
        emit({"type": "stage", "stage": "generating",
              "progress": 0.05 + 0.90 * (i / max(len(seeds), 1)),
              "message": f"seed {i+1}/{len(seeds)} ({turns} turns)"})
        convo = [{"role": "system", "content": persona},
                 {"role": "user", "content": seed}]

        for t in range(turns):
            is_user_turn = (t % 2 == 1)
            if is_user_turn:
                # Ask the model to pretend to be the user following up.
                followup_prompt = (
                    "Continue the conversation below by writing ONLY the next "
                    "user-turn message in character as " + user_persona + ". "
                    "Keep it short (one or two sentences), grounded in what "
                    "the assistant just said. Do not write the assistant's "
                    "next reply.\n\n"
                    + _format_convo_for_followup(convo)
                )
                followup = _llm_call(
                    api_url, auth_token, model_name, followup_prompt,
                    max_tokens=160,
                ).strip()
                if not followup:
                    break
                convo.append({"role": "user", "content": followup})
            else:
                # Assistant turn: use the chat completions endpoint with the
                # full convo so the assistant's reply is in-character.
                reply = _llm_chat(
                    api_url, auth_token, model_name, convo,
                    max_tokens=512,
                ).strip()
                if not reply:
                    break
                convo.append({"role": "assistant", "content": reply})

        # Drop the leading system message from the row if the trainer is
        # going to add its own; keep it for parity with the rest of our
        # messages-format output and let the trainer config decide.
        rows.append({"messages": convo})
    return rows


def _format_convo_for_followup(convo: list[dict]) -> str:
    """Render a chat history as plain text for a one-shot followup prompt."""
    lines = []
    for m in convo:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content or role == "system":
            continue
        lines.append(f"{role.upper()}:\n{content}\n")
    return "\n".join(lines)


def _gen_tool_use_traces(inputs, api_url, auth_token, model_name, emit) -> Iterable[dict]:
    """Generate (user, tool_call, tool_result, final) traces in OpenAI shape.

    The user supplies a list of OpenAI-format tool schemas plus parallel
    arrays of scenario prompts and mocked tool results. For each scenario
    we ask the model to pick a tool + args, then feed the mocked result
    back and ask for the final answer."""
    try:
        tools = json.loads(inputs.get("tools_json") or "[]")
    except json.JSONDecodeError:
        tools = []
    if not isinstance(tools, list) or not tools:
        return []
    scenarios = [s.strip() for s in (inputs.get("scenarios") or "").splitlines() if s.strip()]
    mocks = [m.strip() for m in (inputs.get("mock_results") or "").splitlines() if m.strip()]
    if not scenarios or len(mocks) != len(scenarios) or not api_url:
        return []

    rows = []
    for i, (scenario, mock) in enumerate(zip(scenarios, mocks)):
        emit({"type": "stage", "stage": "generating",
              "progress": 0.05 + 0.90 * (i / len(scenarios)),
              "message": f"scenario {i+1}/{len(scenarios)}"})

        # Turn 1: model picks a tool. We pass tools= so the model emits a
        # structured tool_calls reply via the chat completions API.
        first = _llm_chat_raw(api_url, auth_token, model_name,
                              messages=[{"role": "user", "content": scenario}],
                              tools=tools, max_tokens=512)
        first_msg = (first.get("choices") or [{}])[0].get("message", {})
        tool_calls = first_msg.get("tool_calls") or []
        if not tool_calls:
            continue  # model didn't pick a tool; skip this scenario

        tc = tool_calls[0]
        tc_id = tc.get("id") or f"call_{i}"

        # Turn 2: feed back the mocked result + ask for the final answer.
        messages_for_final = [
            {"role": "user", "content": scenario},
            {"role": "assistant", "content": first_msg.get("content") or "",
             "tool_calls": tool_calls},
            {"role": "tool", "tool_call_id": tc_id,
             "name": (tc.get("function") or {}).get("name", ""),
             "content": mock},
        ]
        final = _llm_chat(api_url, auth_token, model_name,
                          messages_for_final, max_tokens=512).strip()
        if not final:
            continue

        # Output: the full trace as a single training row.
        rows.append({
            "messages": [
                {"role": "user", "content": scenario},
                {"role": "assistant", "content": first_msg.get("content") or "",
                 "tool_calls": tool_calls},
                {"role": "tool", "tool_call_id": tc_id,
                 "name": (tc.get("function") or {}).get("name", ""),
                 "content": mock},
                {"role": "assistant", "content": final},
            ],
            "tools": tools,
        })
    return rows


def _gen_rag_qa(inputs, api_url, auth_token, model_name, emit) -> Iterable[dict]:
    """Chunk a document and ask the model to produce grounded (Q, A) pairs
    for each chunk. The chunk text goes into a system message; the
    question and grounded answer become the user/assistant turns."""
    docs = (inputs.get("documents") or "").strip()
    qpc = max(1, int(inputs.get("questions_per_chunk") or 2))
    min_chars = max(40, int(inputs.get("min_chunk_chars") or 200))
    if not (docs and api_url):
        return []

    # Split on blank lines but glue tiny fragments together until each chunk
    # has at least min_chars. Keeps single-sentence paragraphs from becoming
    # their own RAG context.
    raw_chunks = [c.strip() for c in re.split(r"\n\s*\n", docs) if c.strip()]
    chunks: list[str] = []
    buf = ""
    for c in raw_chunks:
        buf = (buf + "\n\n" + c).strip() if buf else c
        if len(buf) >= min_chars:
            chunks.append(buf); buf = ""
    if buf:
        if chunks:
            chunks[-1] = (chunks[-1] + "\n\n" + buf).strip()
        else:
            chunks.append(buf)

    rows = []
    for i, chunk in enumerate(chunks):
        emit({"type": "stage", "stage": "generating",
              "progress": 0.05 + 0.90 * (i / max(len(chunks), 1)),
              "message": f"chunk {i+1}/{len(chunks)} ({qpc} questions)"})
        for j in range(qpc):
            q_prompt = (
                "Write ONE concise, specific question whose answer is "
                "stated directly in the passage below. Do not answer; only "
                "write the question. Vary it from any other question you "
                "would write for the same passage.\n\n"
                f"PASSAGE:\n{chunk}\n\nQUESTION (question number {j+1}):"
            )
            q = _llm_call(api_url, auth_token, model_name, q_prompt,
                          max_tokens=120).strip()
            if not q:
                continue
            a_prompt = (
                "Answer the question using ONLY information from the passage. "
                "If the answer is not in the passage, reply exactly: "
                '"The passage does not say." Be concise.\n\n'
                f"PASSAGE:\n{chunk}\n\nQUESTION:\n{q}\n\nANSWER:"
            )
            a = _llm_call(api_url, auth_token, model_name, a_prompt,
                          max_tokens=400).strip()
            if not a:
                continue
            rows.append({
                "messages": [
                    {"role": "system",
                     "content": f"Use only the following passage to answer.\n\n{chunk}"},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "metadata": {"chunk_index": i, "passage_chars": len(chunk)},
            })
    return rows


def _gen_cot_synthesis(inputs, api_url, auth_token, model_name, emit) -> Iterable[dict]:
    """Ask the model to emit a step-by-step <think> trace plus a final
    answer for each question. Output preserves the <think> block inside
    assistant content, matching the Qwen3 / GPT-OSS reasoning convention."""
    qs = [q.strip() for q in (inputs.get("questions") or "").splitlines() if q.strip()]
    category = (inputs.get("category") or "general reasoning").strip()
    if not (qs and api_url):
        return []

    rows = []
    for i, q in enumerate(qs):
        emit({"type": "stage", "stage": "generating",
              "progress": 0.05 + 0.90 * (i / len(qs)),
              "message": f"question {i+1}/{len(qs)}"})
        prompt = (
            f"You are solving a {category} problem. Think step by step in "
            "a <think>...</think> block first, then write the final answer "
            "after the closing </think> tag. Keep the final answer concise.\n\n"
            f"QUESTION:\n{q}\n\nRESPONSE:"
        )
        out = _llm_call(api_url, auth_token, model_name, prompt,
                        max_tokens=1024).strip()
        if not out:
            continue
        # Normalize: ensure the output has <think>...</think>. If the model
        # forgot the tags, wrap the whole thing as the trace and use the
        # last paragraph as the answer.
        if "<think>" not in out:
            paras = [p for p in out.split("\n\n") if p.strip()]
            if len(paras) >= 2:
                trace = "\n\n".join(paras[:-1])
                ans = paras[-1]
                out = f"<think>\n{trace}\n</think>\n\n{ans}"
            else:
                out = f"<think>\n{out}\n</think>\n\n{out}"
        rows.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": out},
            ],
            "metadata": {"category": category},
        })
    return rows


def _gen_verified_code(inputs, api_url, auth_token, model_name, emit) -> Iterable[dict]:
    """For each NL spec, ask the model to write code + asserts; run the
    asserts in the sandbox; keep only the pairs whose asserts pass."""
    specs = [s.strip() for s in (inputs.get("specs") or "").splitlines() if s.strip()]
    language = (inputs.get("language") or "python").strip().lower()
    if not (specs and api_url):
        return []
    if language != "python":
        # We only know how to verify python via the sandbox right now.
        # For other languages, emit unverified rows (still useful as SFT)
        # but mark them.
        pass

    from .sandbox import run_python

    rows = []
    n_verified = 0
    n_total = 0
    for i, spec in enumerate(specs):
        emit({"type": "stage", "stage": "generating",
              "progress": 0.05 + 0.90 * (i / len(specs)),
              "message": f"spec {i+1}/{len(specs)} ({n_verified}/{n_total} verified so far)"})
        n_total += 1
        prompt = (
            "Write a Python function that satisfies the spec. Then write "
            "three `assert` statements at module level that verify the "
            "function. Output ONLY the code as a single fenced ```python "
            "block, no commentary.\n\n"
            f"SPEC:\n{spec}\n\nCODE:"
        )
        out = _llm_chat_raw(
            api_url, auth_token, model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200, temperature=0.2,
        )
        out_text = (out.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        code = _extract_python_block(out_text)
        if not code or len(code) < 10:
            continue

        verified = False
        verify_error: str | None = None
        if language == "python":
            result = run_python(code, timeout=15.0, memory_limit_mb=512,
                                strict=False)
            verified = (result.returncode == 0)
            if not verified:
                verify_error = (result.stderr or result.stdout or "")[:400]
        if verified:
            n_verified += 1

        rows.append({
            "messages": [
                {"role": "user", "content": spec},
                {"role": "assistant", "content": f"```python\n{code}\n```"},
            ],
            "metadata": {
                "verified": verified,
                "language": language,
                "verify_error": verify_error,
            },
        })
    emit({"type": "stage", "stage": "generating", "progress": 0.96,
          "message": f"{n_verified}/{n_total} specs verified (unverified rows kept with verified=false)"})
    return rows


def _extract_python_block(text: str) -> str:
    """Pull a ```python ... ``` block out of LLM output. Returns the code
    string with fences removed, or the whole text if no fence found."""
    m = re.search(r"```(?:python|py)?\n(.*?)\n```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _llm_chat(api_url, auth_token, model_name, messages, max_tokens=512) -> str:
    """Multi-turn chat completion. Returns the assistant content string."""
    resp = _llm_chat_raw(api_url, auth_token, model_name,
                         messages=messages, max_tokens=max_tokens)
    return (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""


def _llm_chat_raw(api_url, auth_token, model_name, *, messages, tools=None,
                  max_tokens=512, temperature=0.7,
                  enable_thinking: bool = False) -> dict:
    """Non-streaming chat call returning the full JSON body.

    Omits the ``model`` field when no name was passed so mlx-lm falls
    back to whatever model the server was started with; sending the
    literal string ``"default"`` makes mlx-lm respond 404.

    Reasoning models (Qwen3, GPT-OSS) default to spending the entire
    ``max_tokens`` budget on the ``<think>`` block and emitting an empty
    ``content``. Dataset generators need the content, so we explicitly
    disable thinking by default. The CoT-synthesis template wants the
    trace, but it constructs the reasoning prompt itself rather than
    relying on a hidden channel.
    """
    import urllib.request
    body: dict = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }
    if model_name:
        body["model"] = model_name
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        f"{api_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_token or 'sk-optiq-local'}",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def _llm_call(api_url, auth_token, model_name, prompt, max_tokens=512) -> str:
    """One-shot chat completion against the local API."""
    data = _llm_chat_raw(
        api_url, auth_token, model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
