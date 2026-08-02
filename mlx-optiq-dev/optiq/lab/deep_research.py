"""Deep research for OptiQ Lab — a TTD-DR (Test-Time Diffusion Deep Researcher)
workflow: local web search + a cloud/local model for synthesis, so the research
trail never leaves the machine and nothing is paid to an external deep-research
provider.

Draft-centric (arXiv:2507.16075): plan -> a first-pass ("noisy") draft -> a
denoise loop where the *current draft* drives each round's searches (targeting
its own gaps) and the retrieved, cited findings refine the draft -> a final
report with real citations. This bidirectional draft<->search loop is what beats
a naive search-then-summarize pipeline.

The model is injected as ``chat(system, user, ...)`` so the Lab can pass its
cloud or local client; ``on_event`` streams progress the UI renders as a
research-trace card. Search + fetch are the Lab's local web tools.
"""
from __future__ import annotations

import io
import json
import re
import urllib.request
from typing import Any, Callable

from .tools.web_search import fetch_page, search

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _strip_ascii_diagrams(md: str) -> str:
    """Remove fenced code blocks that are ASCII-art box/flow diagrams. The model
    loves drawing ``+----+`` boxes even when told not to, and they look amateurish
    in a report; a real fenced code block (actual code) never has a ``+---`` border,
    so that signature is a safe detector that leaves genuine code alone."""
    def is_box(block: str) -> bool:
        return (bool(re.search(r"^\s*\+[-+=]{3,}", block, re.MULTILINE))       # +---+ boxes
                or bool(re.search(r"[│├└┌┐┘┬┴┤┼►▶▼▲]|──►|──>|-->", block))     # tree/flow art
                or (len(re.findall(r"^\s*\|.*\|\s*$", block, re.MULTILINE)) >= 3
                    and "+" in block and "|" in block and "def " not in block))
    parts = re.split(r"(```.*?```)", md, flags=re.DOTALL)
    kept = [p for p in parts if not (p.startswith("```") and is_box(p))]
    return re.sub(r"\n{3,}", "\n\n", "".join(kept)).strip()


def _json_list(raw: str, key: str) -> list:
    """Pull ``key``'s list out of a model reply, tolerating ```json fences and a
    reasoning model that wraps the object in prose. Returns [] if nothing parses.
    (A reasoning model like gemini-flash often returns text around the JSON, which
    is why a bare json.loads dropped the draft-driven queries to a fallback.)"""
    if not raw:
        return []
    for candidate in (raw, *re.findall(r"\{.*?\}", raw, re.DOTALL)):
        try:
            v = json.loads(candidate).get(key)
            if isinstance(v, list):
                return v
        except Exception:
            continue
    # last resort: a bare ["a", "b"] array anywhere in the text
    m = re.search(r"\[\s*\".*?\"\s*\]", raw, re.DOTALL)
    if m:
        try:
            v = json.loads(m.group(0))
            return v if isinstance(v, list) else []
        except Exception:
            pass
    return []


# ── source fetching: arXiv HTML full-text, PDF extraction, canonicalization ──

def _arxiv_id(url: str) -> str | None:
    m = re.search(r"arxiv\.org/(?:abs|pdf|html|format)/(\d{4}\.\d{4,5})(v\d+)?", url)
    return (m.group(1) + (m.group(2) or "")) if m else None


def canonical_url(url: str) -> str:
    """Collapse the many arXiv URL forms (abs / pdf / html) for one paper to a
    single key so the same paper is never counted as several sources."""
    aid = _arxiv_id(url)
    if aid:
        return f"arxiv:{aid.split('v')[0]}"     # ignore version for dedup
    return url.split("#")[0].rstrip("/")


def _fetch_pdf(url: str, max_pages: int = 24) -> str:
    from pypdf import PdfReader
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read(12_000_000)
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((p.extract_text() or "") for p in reader.pages[:max_pages]).strip()


def fetch_source(url: str, max_chars: int = 9000) -> tuple[str, str]:
    """Return (text, effective_url). Prefers arXiv's HTML full text over the PDF
    (html2text can't read a PDF, which is why the naive fetch dropped most arXiv
    sources); extracts real PDFs with pypdf; otherwise fetches the page."""
    aid = _arxiv_id(url)
    if aid:
        html_url = f"https://arxiv.org/html/{aid}"
        try:
            txt = fetch_page(html_url)
        except Exception:
            txt = ""
        # arXiv serves a stub when no HTML build exists — fall back to the abstract.
        if txt and len(txt) > 800 and "no html" not in txt[:400].lower():
            return txt[:max_chars], html_url
        try:
            return fetch_page(f"https://arxiv.org/abs/{aid.split('v')[0]}")[:max_chars], \
                   f"https://arxiv.org/abs/{aid.split('v')[0]}"
        except Exception:
            return "", url
    if url.lower().endswith(".pdf") or "/pdf/" in url:
        try:
            return _fetch_pdf(url)[:max_chars], url
        except Exception:
            return "", url
    try:
        return fetch_page(url)[:max_chars], url
    except Exception:
        return "", url


# ── prompts ──────────────────────────────────────────────────────────────────

_PLAN = ("You are a research planner. Question:\n{q}\n\nWrite a tight outline (3-6 "
         "sections) of what a thorough, well-cited report must cover. Bullets only.")
_DRAFT0 = ("You are writing a first-pass research draft that will be refined with web "
           "sources. Question:\n{q}\nOutline:\n{plan}\n\nWrite ~300 words from what you "
           "already know; mark anything uncertain or needing a source with [?].")
_QUERIES = ("You turn the gaps in a research draft into web searches. "
            "Question:\n{q}\nCurrent draft:\n{draft}\n\nIdentify the {n} most important "
            "things still missing, uncertain, or marked [?]. Return JSON "
            '{{"queries": ["...", ...]}} — {n} specific, diverse web-search queries to '
            "fill those gaps. Strongly prefer primary and authoritative sources — papers "
            "(arXiv), official docs, standards, benchmark results — over generic blog "
            "listicles; phrase queries to surface them (add 'arxiv', 'paper', or 'docs').")
_EXTRACT = ("Extract citable evidence relevant to the question from this source. "
            "Question:\n{q}\nSource: {title} ({url})\n\n{page}\n\nReturn JSON "
            '{{"evidence": [{{"claim": "...", "quote": "..."}}]}} — 2-5 items. For each: '
            "'claim' is one concrete, self-contained sentence that bears on the question; "
            "'quote' is a SHORT span (<=30 words) copied VERBATIM from the source above that "
            "directly supports that claim (do not paraphrase the quote — it must appear "
            "word-for-word on the page). Only include a claim if you can back it with such a "
            "quote. If the page has nothing relevant, return "
            '{{"evidence": []}}.')
_REVISE = ("Revise a research draft using new sourced findings. "
           "Question:\n{q}\nCurrent draft:\n{draft}\n\nNew findings — each has a fixed [n] "
           "number and a verbatim quote from its source:\n{finds}\n\nFold the relevant "
           "findings into the draft. Ground every factual sentence in the findings: state "
           "only what a finding's quote supports, staying close to its wording, and cite the "
           "ONE finding [n] that claim comes from. Do NOT merge several findings into a "
           "single sweeping sentence that no individual source states — write separate, "
           "individually-cited claims instead. Resolve [?] where the findings allow, and "
           "keep all prior [n] citations intact. Stay specific and detailed.")
_REPORT = ("Polish this cited draft into the final research report in Markdown. "
           "Question:\n{q}\n\nCited draft — its [n] citations are already grounded in real "
           "sources (each was added when that source was read), so KEEP them:\n{draft}\n\n"
           "Sources (for the reference list):\n{srcs}\n\nWrite it the way a sharp domain "
           "analyst would:\n"
           "- Open with a crisp framing of the problem and why it matters (2-3 sentences), "
           "not a dictionary definition.\n"
           "- 4-6 sections (##) with concise, descriptive headers (no rigid '1. 2. 3.' "
           "numbering); keep the full depth and every concrete method name, bit-width, and "
           "measured number.\n"
           "- Write in flowing, ANALYTICAL prose that compares and weighs the approaches "
           "and their tradeoffs — not annotated bullet lists. Use a Markdown table where a "
           "true side-by-side comparison helps.\n"
           "- Do NOT draw ASCII-art or box/flow diagrams — prose, headers, and Markdown "
           "tables only.\n"
           "- End with a short '## Bottom line' that states the key takeaway / how to "
           "choose, in 2-4 sentences.\n"
           "- Keep every factual sentence faithful to the source it cites; when you tighten "
           "or combine the draft, do NOT create a broad claim that none of its cited sources "
           "individually supports.\n"
           "PRESERVE every [n] citation exactly as in the draft — do not add, drop, or "
           "renumber citations, and never cite a number not already in the draft. Do NOT "
           "write a Sources or References list yourself — the application appends it "
           "consistently, so just end after the Bottom line.")
_SUMMARY = ("Answer the question directly in 3-4 sentences from this report, then note "
            "that a full cited report is attached.\nQuestion: {q}\n\n{report}")


def _append_sources(report: str, sources: list[dict]) -> str:
    """Replace any model-written Sources/References section with a deterministic one
    built from the fetched sources, so the reference list ALWAYS matches the ``[n]``
    markers. A weak local model routinely garbles or truncates a hand-written list
    (drops entries, merges lines, misnumbers); the app owning it — as Unsloth does —
    makes ``[n]`` ↔ source exact regardless of the model."""
    # Cut anything from a trailing Sources/References/Bibliography heading onward.
    report = re.sub(r"\n#{1,4}\s*(?:Sources?|References?|Bibliography|Works\s+Cited)\b.*",
                    "", report, flags=re.IGNORECASE | re.DOTALL).rstrip()
    if not sources:
        return report
    lines = "\n".join(f"[{i + 1}] {s.get('title') or s.get('url')} — {s.get('url')}"
                      for i, s in enumerate(sources))
    return report + "\n\n## Sources\n" + lines


def _evidence_for(n: int, sources: list[dict]) -> str:
    """The verbatim quotes (falling back to claims) captured for source ``[n]`` —
    the real source words the FACT check will be graded against."""
    if not (1 <= n <= len(sources)):
        return ""
    ev = sources[n - 1].get("evidence") or []
    quotes = [e.get("quote") or e.get("claim") or "" for e in ev]
    text = " … ".join(q for q in quotes if q) or (sources[n - 1].get("extract") or "")
    return text[:700]


def _verify_and_repair(report: str, sources: list[dict],
                       chat: Callable[..., str]) -> tuple[str, int]:
    """Grade every cited sentence against the verbatim evidence of its cited
    sources and *repair* it — the FACT fix that actually moves the number.

    For each sentence carrying ``[n]`` markers, the judge sees the claim and the
    exact quotes captured from all of its cited sources. If the evidence supports
    the claim, it is left alone; if it only partly supports it, the sentence is
    REWRITTEN to state just what the quotes back (keeping the [n]); if nothing
    supports it, its citations are dropped. Rewriting (not merely pruning) is what
    lifts citation-support: a synthesized over-claim becomes a faithful one instead
    of an uncited orphan. Returns (report, num_changed)."""
    sents = re.split(r"(?<=[.!?])\s+", report)
    cited = [s for s in sents if re.search(r"\[\d+\]", s)]
    if not cited:
        return report, 0
    changed = 0
    for i0 in range(0, len(cited), 6):
        batch = cited[i0:i0 + 6]
        listing = []
        for j, s in enumerate(batch):
            ns = sorted({int(x) for x in re.findall(r"\[(\d+)\]", s)})
            ev = "\n".join(f"    [{n}]: {_evidence_for(n, sources)}" for n in ns
                           if _evidence_for(n, sources))
            listing.append(f"{j + 1}. CLAIM: {s.strip()}\n  EVIDENCE:\n{ev or '    (none)'}")
        raw = chat(
            "You are grading citation faithfulness in a research report. For each item, "
            "decide whether the cited EVIDENCE (verbatim source quotes) supports the CLAIM.\n"
            "Return JSON {{\"items\":[{{\"i\":<n>,\"action\":\"keep|rewrite|drop\","
            "\"text\":\"<rewritten sentence, only if action=rewrite>\"}}]}}.\n"
            "- keep: the evidence clearly supports the claim as written.\n"
            "- rewrite: the claim overreaches or blends unsupported detail — return a "
            "sentence that states ONLY what the quotes support, KEEPING the same [n] "
            "citation markers verbatim.\n"
            "- drop: no cited quote supports the claim at all.\n"
            "Be strict: numbers, named methods, and comparisons must actually appear in the "
            f"quotes.\n\n" + "\n\n".join(listing),
            json_mode=True, max_tokens=4000)
        for v in (_json_list(raw, "items") or []):
            try:
                k = int(v.get("i", 0)) - 1
                action = str(v.get("action", "keep")).lower()
            except (TypeError, ValueError, AttributeError):
                continue
            if not (0 <= k < len(batch)) or action == "keep":
                continue
            old = batch[k]
            if action == "rewrite" and v.get("text", "").strip():
                new_s = v["text"].strip()
                if not re.search(r"\[\d+\]", new_s):     # model dropped the markers — re-add
                    new_s += " " + "".join(sorted(set(re.findall(r"\[\d+\]", old))))
            elif action == "drop":
                new_s = re.sub(r"\s*\[\d+\]", "", old)
                new_s = re.sub(r"\s+([.,;:)])", r"\1", new_s)
            else:
                continue
            report = report.replace(old, new_s, 1)
            changed += 1
    return report, changed


# ── the workflow ─────────────────────────────────────────────────────────────

def deep_research(
    question: str,
    *,
    chat: Callable[..., str],
    on_event: Callable[[str, dict], None] | None = None,
    max_rounds: int = 3,
    queries_per_round: int = 3,
    sources_per_query: int = 3,
    report_tokens: int = 16000,     # deep research is long — give it OptiQ-Code-level
    revise_tokens: int = 8000,      # room; a reasoning model spends a lot before writing
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run the TTD-DR loop. ``chat(system, user, json_mode=False, max_tokens=...)``
    returns text. Returns {report, summary, sources, plan}."""
    def emit(kind: str, **data):
        if on_event:
            on_event(kind, data)

    def stop() -> bool:
        return bool(cancelled and cancelled())

    emit("phase", phase="planning")
    plan = chat(_PLAN.format(q=question), max_tokens=2000)   # above the reasoning floor
    emit("plan", outline=plan)

    emit("phase", phase="drafting", round=0)
    draft = chat(_DRAFT0.format(q=question, plan=plan), max_tokens=4000)

    sources: list[dict] = []
    seen: set[str] = set()
    for rd in range(1, max_rounds + 1):
        if stop():
            break
        emit("phase", phase="searching", round=rd)
        # Reasoning models spend tokens thinking before the JSON, so give room and
        # parse tolerantly; falling back to the raw question loses the draft-driven
        # diversity that is the whole point.
        raw = chat(_QUERIES.format(q=question, draft=draft, n=queries_per_round),
                   json_mode=True, max_tokens=2000)
        queries = [q for q in _json_list(raw, "queries") if isinstance(q, str) and q.strip()]
        queries = queries[:queries_per_round] or [question]

        round_finds: list[dict] = []
        for q in queries:
            if stop():
                break
            try:
                results = search(q, max_results=6)
            except Exception:
                results = []
            emit("search", query=q, results=[r.get("url", "") for r in results])
            picked = 0
            for res in results:
                if picked >= sources_per_query or stop():
                    break
                cu = canonical_url(res.get("url", ""))
                if not cu or cu in seen:
                    continue
                page, eff_url = fetch_source(res.get("url", ""))
                if not page or len(page) < 300:
                    continue
                seen.add(cu)
                picked += 1
                emit("source", url=eff_url, title=res.get("title", ""))
                # A reasoning model (gemini-flash) spends ~1.3-1.6k tokens THINKING before
                # it emits content; too small a ceiling truncates the JSON to nothing and
                # the source is silently lost. Keep extraction well above that floor.
                raw_ext = chat(_EXTRACT.format(q=question, title=res.get("title", ""),
                                               url=eff_url, page=page),
                               json_mode=True, max_tokens=4000)
                evidence = [{"claim": (e.get("claim") or "").strip(),
                             "quote": (e.get("quote") or "").strip()}
                            for e in _json_list(raw_ext, "evidence")
                            if isinstance(e, dict) and (e.get("claim") or "").strip()]
                if not evidence:
                    continue
                extract = " ".join(e["claim"] for e in evidence)
                round_finds.append({"url": eff_url, "title": res.get("title", ""),
                                    "extract": extract, "evidence": evidence})
        sources.extend(round_finds)

        if stop():
            break
        emit("phase", phase="refining", round=rd)
        base = len(sources) - len(round_finds)
        finds = "\n\n".join(
            f"[{base + i + 1}] {f['title']} ({f['url']})\n" +
            "\n".join(f"- {e['claim']}" + (f'  «{e["quote"]}»' if e["quote"] else "")
                      for e in f["evidence"])
            for i, f in enumerate(round_finds))
        draft = chat(_REVISE.format(q=question, draft=draft, finds=finds or "(none new)"),
                     max_tokens=revise_tokens)

    emit("phase", phase="writing")
    # The denoise loop already produced a *cited* draft — each [n] was attached when
    # that finding was read, so it is grounded. The final step polishes (structure,
    # depth) and PRESERVES those citations rather than re-synthesizing and re-citing
    # off the model's priors (which is what tanked FACT).
    src_list = "\n".join(f"[{i + 1}] {s['title']} — {s['url']}" for i, s in enumerate(sources))
    report = chat(_REPORT.format(q=question, draft=draft, srcs=src_list or "(none)"),
                  max_tokens=report_tokens)   # generous — deep research reports run long
    report = _strip_ascii_diagrams(report)    # the model won't stop drawing ASCII boxes

    # Grade each cited sentence against its sources' verbatim quotes and repair the
    # overreaches — the pass that actually lifts FACT (citation-support). Uses the
    # quotes captured at extract time, so it grades against real source words.
    emit("phase", phase="verifying")
    report, repaired = _verify_and_repair(report, sources, chat)
    if repaired:
        emit("verified", repaired=repaired)

    # Append the reference list deterministically so [n] ↔ source is always exact.
    report = _append_sources(report, sources)

    summary = chat(_SUMMARY.format(q=question, report=report), max_tokens=2000)
    emit("done", sources=len(sources), report_chars=len(report))
    return {"report": report, "summary": summary, "sources": sources, "plan": plan}
