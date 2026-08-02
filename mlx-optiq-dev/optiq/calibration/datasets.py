"""Calibration data loading for sensitivity analysis.

LLM calibration is MLX-native. The public entry ``load_llm_calibration``
returns a **callable** in the shape the analysers want:
``list[(args_tuple, kwargs_dict)]`` where each ``args_tuple`` has one
positional argument — an ``mx.array`` of shape ``(1, seq_len)`` of int32
token ids.

The default uses ``optiq.jsonl`` — a 32-sample hand-curated mix shipped
inside the package across five domains (prose, thinking traces, code
reasoning, agent loops, tool calling). Pass ``mix="/path/to/custom.jsonl"``
for a domain-specialised replacement.

A WikiText-2 single-domain helper (``_load_wikitext_calibration``) is
kept private — used internally for ablation comparisons against the
default mix; not exposed in the CLI surface.

VLM calibration still returns PyTorch tensors as the VLM pipeline hasn't
been ported.
"""

import json
import os
from pathlib import Path

import numpy as np


_OPTIQ_MIX_PATH = Path(__file__).parent / "data" / "optiq.jsonl"


def load_llm_calibration(
    tokenizer,
    n_samples: int = 32,
    seq_len: int = 1024,
    seed: int = 42,
    mix: str = "optiq",
):
    """Build an MLX calibration callable.

    Returns a zero-arg callable ``calibration_fn()`` that produces
    ``list[(args_tuple, kwargs_dict)]`` ready for
    ``analyze_sensitivity_exact``. Each ``args_tuple`` is
    ``(mx.array[(1, seq_len), int32],)``; ``kwargs`` is empty.

    Args:
        tokenizer: Hugging Face tokenizer (used to encode + apply chat
            template when present).
        n_samples: Number of fixed-length calibration sequences to emit.
            Each forward-pass through the model during sensitivity probing
            uses one of these. Trade-off: more samples → more stable
            per-layer ranking, linearly more probe compute.
        seq_len: Tokens per sequence.
        seed: Reproducibility for sample shuffling.
        mix: ``"optiq"`` (default) → use the bundled hand-curated 6-domain
            mix. Or any path to a JSONL file in the same schema as
            ``optiq.jsonl`` for a custom domain-specialised mix.
    """
    if mix == "optiq":
        return _load_optiq_calibration(
            tokenizer, n_samples, seq_len, seed, _OPTIQ_MIX_PATH
        )
    path = Path(mix)
    if not path.exists():
        raise FileNotFoundError(
            f"calibration mix not found at {path}. Use mix='optiq' "
            f"(default) or a path to a JSONL file."
        )
    return _load_optiq_calibration(
        tokenizer, n_samples, seq_len, seed, path
    )


def _load_optiq_calibration(
    tokenizer,
    n_samples: int,
    seq_len: int,
    seed: int,
    jsonl_path: Path,
):
    """Load the hand-curated multi-domain mix.

    Each line in the JSONL is one sample with schema:

        {"domain": "prose",   "text": "..."}              # raw text
        {"domain": "thought", "messages": [...]}          # multi-turn chat
        {"domain": "tool",    "messages": [...],          # chat + tool defs
                              "tools":    [...]}

    For chat-style samples we apply the tokenizer's chat template (with
    tools when supported); for raw text we encode directly. Then we
    concatenate everything into one long token stream and chunk into
    fixed-length sequences — same end-shape as the wikitext loader, just
    with vastly more diverse activation patterns.
    """
    import mlx.core as mx

    rng = np.random.RandomState(seed)

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"optiq calibration mix not found at {jsonl_path}. "
            f"Run `python scripts/build_calibration.py` to rebuild it."
        )

    if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token

    has_chat_template = bool(getattr(tokenizer, "chat_template", None))

    samples: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    by_domain: dict[str, int] = {}
    all_tokens: list[int] = []
    for s in samples:
        domain = s.get("domain", "?")
        text: str
        if "messages" in s:
            if has_chat_template:
                tools = s.get("tools")
                kwargs = dict(tokenize=False, add_generation_prompt=False)
                if tools:
                    kwargs["tools"] = tools
                try:
                    text = tokenizer.apply_chat_template(s["messages"], **kwargs)
                except Exception:
                    # Some models reject ``tools`` if they don't know about
                    # them; retry without.
                    try:
                        kwargs.pop("tools", None)
                        text = tokenizer.apply_chat_template(s["messages"], **kwargs)
                    except Exception:
                        # The template still rejects these messages — e.g.
                        # gpt-oss's harmony template raises on a ``tool`` turn
                        # that isn't preceded by an assistant ``tool_call``.
                        # Fall back to dumb role-prefixed concatenation so one
                        # strict template can't sink the whole calibration set.
                        text = "\n\n".join(
                            f"[{m['role']}]: {m['content']}" for m in s["messages"]
                        )
            else:
                # No chat template — concatenate role-prefixed turns
                text = "\n\n".join(
                    f"[{m['role']}]: {m['content']}" for m in s["messages"]
                )
        elif "text" in s:
            text = s["text"]
        else:
            continue

        toks = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(toks)
        by_domain[domain] = by_domain.get(domain, 0) + len(toks)

    if len(all_tokens) < seq_len:
        raise RuntimeError(
            f"Calibration mix produced only {len(all_tokens)} tokens "
            f"(< seq_len={seq_len}). Try a smaller seq_len or rebuild "
            f"the mix with longer samples."
        )

    starts = list(range(0, len(all_tokens) - seq_len, seq_len))
    rng.shuffle(starts)
    chunks: list[mx.array] = []
    for start in starts[:n_samples]:
        chunks.append(mx.array([all_tokens[start:start + seq_len]], dtype=mx.int32))

    if not chunks:
        raise RuntimeError("Could not create any calibration sequences.")

    domain_summary = " · ".join(f"{d} {t}t" for d, t in sorted(by_domain.items()))
    print(
        f"  loaded {len(chunks)} calibration sequences (seq_len={seq_len}) "
        f"from optiq mix [{domain_summary}, chat_template={has_chat_template}]"
    )

    def calibration_fn() -> list[tuple[tuple, dict]]:
        return [((c,), {}) for c in chunks]

    return calibration_fn


def _load_wikitext_calibration(
    tokenizer, n_samples: int, seq_len: int, seed: int,
):
    """Single-domain WikiText-2 calibration. Internal-only — used for
    ablation A/Bs against the optiq mix; not exposed in the public API."""
    import mlx.core as mx
    from datasets import load_dataset

    rng = np.random.RandomState(seed)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [t for t in ds["text"] if len(t.strip()) > 100]
    if not texts:
        raise RuntimeError(
            "WikiText-2 validation set returned no usable texts. "
            "Check datasets library + network."
        )

    if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token

    all_tokens: list[int] = []
    for text in texts:
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        if len(all_tokens) >= n_samples * seq_len * 2:
            break
    if len(all_tokens) < seq_len:
        raise RuntimeError(
            f"Not enough tokens from WikiText-2 ({len(all_tokens)} < {seq_len})."
        )

    starts = list(range(0, len(all_tokens) - seq_len, seq_len))
    rng.shuffle(starts)
    chunks: list[mx.array] = []
    for start in starts[:n_samples]:
        chunk = all_tokens[start:start + seq_len]
        chunks.append(mx.array([chunk], dtype=mx.int32))

    if not chunks:
        raise RuntimeError("Could not create any calibration sequences.")

    print(
        f"  loaded {len(chunks)} calibration sequences "
        f"(seq_len={seq_len}) from WikiText-2 (single-domain ablation mode)"
    )

    def calibration_fn() -> list[tuple[tuple, dict]]:
        return [((c,), {}) for c in chunks]

    return calibration_fn
