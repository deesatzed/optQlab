"""Text extraction for chat file attachments.

The chat composer accepts these uploads:

  - plain text (txt, md, code, json, csv, ...),handled in JS, no
    server hop needed
  - PDF,extracted server-side via ``pypdf``
  - DOCX,extracted server-side via the ``docx2txt`` library

Everything else is rejected at the server boundary. We deliberately do
NOT support images or audio here,see CLAUDE.md and the v0.1.0 scope
note: VLM and audio are not on the chat path.

``extract(filename, raw_bytes) -> str`` returns the plain-text body. It
caps output at ``MAX_EXTRACT_CHARS`` so a 500-page PDF doesn't blow the
context window.
"""
from __future__ import annotations

import io
import os


MAX_EXTRACT_CHARS = 100_000   # cap inserted text per file
MAX_PAGES = 200               # cap PDF pages we process


class UnsupportedFile(Exception):
    """Raised when the file extension isn't on the allow-list."""


class ExtractFailed(Exception):
    """Raised when a supported file failed to parse."""


def _truncate(s: str) -> str:
    if len(s) <= MAX_EXTRACT_CHARS:
        return s
    return s[:MAX_EXTRACT_CHARS] + f"\n\n[...truncated, {len(s)} chars total]"


def _extract_pdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ExtractFailed(
            "PDF support not installed. `pip install pypdf` to enable."
        )
    try:
        reader = PdfReader(io.BytesIO(raw))
    except Exception as e:
        raise ExtractFailed(f"could not parse PDF: {e}") from e

    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        if i >= MAX_PAGES:
            parts.append(f"\n[...stopped at page {MAX_PAGES}]")
            break
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(t.strip())
    return "\n\n".join(parts).strip() or "(no extractable text in PDF)"


def _extract_docx(raw: bytes) -> str:
    try:
        import docx2txt
    except ImportError:
        raise ExtractFailed(
            "DOCX support not installed. `pip install docx2txt` to enable."
        )
    # docx2txt only takes a path or file-like with .name set; we write
    # to /tmp briefly. (It uses zipfile under the hood; can't avoid the
    # disk round-trip on older versions.)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        f.write(raw)
        tmp_path = f.name
    try:
        try:
            text = docx2txt.process(tmp_path) or ""
        except Exception as e:
            raise ExtractFailed(f"could not parse DOCX: {e}") from e
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return text.strip() or "(no extractable text in DOCX)"


def _extract_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def extract(filename: str, raw: bytes) -> str:
    """Top-level dispatcher. Returns plain text, capped, or raises.

    ``filename`` is used only for extension routing; we never write the
    file under that name on disk.
    """
    name = (filename or "").lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""

    if ext == "pdf":
        return _truncate(_extract_pdf(raw))
    if ext == "docx":
        return _truncate(_extract_docx(raw))

    text_exts = {
        "txt", "md", "py", "json", "jsonl", "yaml", "yml", "csv",
        "html", "htm", "css", "js", "ts", "tsx", "jsx",
        "go", "rs", "toml", "ini", "sh", "log", "rst",
    }
    if ext in text_exts:
        return _truncate(_extract_text(raw))

    raise UnsupportedFile(
        f"Unsupported file type: .{ext or '(none)'}. "
        "Supported: text/code files, PDF, DOCX."
    )
