"""Lightweight, dependency-free RAG for chat-with-files.

Splits attached documents into overlapping chunks, retrieves the chunks most
relevant to the user's question with a small BM25 scorer (no embedding model,
no vector DB), and formats them into a context block with ``[n]`` citation
markers plus a sources list. The model is instructed to cite ``[n]``; only the
retrieved chunks enter the prompt, not the whole document, which is the point.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")

CHUNK_CHARS = 900          # ~200 tokens per chunk
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class Chunk:
    text: str
    source: str            # filename
    index: int             # chunk number within the source


def chunk_document(name: str, text: str) -> list[Chunk]:
    """Split one document into overlapping char-windows on paragraph-ish
    boundaries."""
    text = (text or "").strip()
    if not text:
        return []
    chunks: list[Chunk] = []
    start = 0
    n = len(text)
    i = 0
    while start < n:
        end = min(start + CHUNK_CHARS, n)
        # Prefer to break on a newline/sentence boundary near the window end.
        if end < n:
            window = text[start:end]
            brk = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
            if brk > CHUNK_CHARS // 2:
                end = start + brk + 1
        chunks.append(Chunk(text=text[start:end].strip(), source=name, index=i))
        i += 1
        if end >= n:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return [c for c in chunks if c.text]


def _bm25_rank(query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
    """Classic BM25 over the chunk set. Returns (chunk, score) sorted desc."""
    q_terms = set(_tokenize(query))
    if not q_terms or not chunks:
        return []
    docs = [_tokenize(c.text) for c in chunks]
    N = len(docs)
    avgdl = sum(len(d) for d in docs) / N if N else 0.0
    # Document frequency per query term.
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d) & q_terms:
            df[t] = df.get(t, 0) + 1
    k1, b = 1.5, 0.75
    scored: list[tuple[Chunk, float]] = []
    for c, d in zip(chunks, docs):
        if not d:
            continue
        dl = len(d)
        tf: dict[str, int] = {}
        for t in d:
            if t in q_terms:
                tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for t, f in tf.items():
            n_t = df.get(t, 0)
            idf = math.log(1 + (N - n_t + 0.5) / (n_t + 0.5))
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / (avgdl or 1)))
        if score > 0:
            scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def retrieve(query: str, documents: list[dict], top_k: int = DEFAULT_TOP_K):
    """Retrieve the top-k chunks across ``documents`` (each {name, text}).

    Returns ``(context_block, sources)`` where ``context_block`` is a string to
    prepend to the prompt (empty if nothing matched) and ``sources`` is a list
    of ``{n, source, chunk, preview}`` dicts for the UI's citation panel.
    """
    chunks: list[Chunk] = []
    for doc in documents or []:
        chunks.extend(chunk_document(doc.get("name") or "document",
                                     doc.get("text") or ""))
    ranked = _bm25_rank(query or "", chunks, top_k)
    if not ranked:
        return "", []

    lines = [
        "Use the following retrieved context to answer the question. Cite "
        "sources inline with bracketed numbers like [1], [2] that match the "
        "numbered snippets. If the context does not contain the answer, say so.",
        "",
        "Retrieved context:",
    ]
    q_terms = set(_tokenize(query or ""))
    sources = []
    for i, (c, _score) in enumerate(ranked, start=1):
        lines.append(f"[{i}] (from {c.source}) {c.text}")
        sources.append({
            "n": i, "source": c.source, "chunk": c.index,
            "preview": _snippet(c.text, q_terms),
        })
    return "\n".join(lines), sources


def _snippet(text: str, q_terms: set, width: int = 160) -> str:
    """A ~width-char preview centered on the most specific query-term match
    (longest term wins, so it skips stopwords like 'the')."""
    low = text.lower()
    pos = -1
    for t in sorted(q_terms, key=len, reverse=True):
        if len(t) < 3:
            continue
        p = low.find(t)
        if p != -1:
            pos = p
            break
    if pos == -1:
        return text[:width] + ("…" if len(text) > width else "")
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    snip = text[start:end].strip()
    return ("…" if start > 0 else "") + snip + ("…" if end < len(text) else "")
