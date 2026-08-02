"""The OptiQ Code tool set and their execution.

Tight and curated (design §3.2): a large tool catalog degrades weak-model tool
selection, so the set is small. `read_file`, `write_file`, `edit_file`, `bash`,
`run_tests`, `search`, `git`, `web_search`, `web_fetch`, `done`. No
property/contract tools (retired). `bash` already covers "run code" / "run the
app" (like Claude Code — it runs in the repo, gated by approval); `web_search` /
`web_fetch` add the web the way Claude Code's WebSearch/WebFetch do.

The paging, ANSI-stripping, and pytest-parsing helpers are the format-robustness
edge (design §3.3): a too-big file is navigable instead of a dead-end truncation,
and pytest output is stripped of color before parsing (a real bug that silently
broke score parsing).
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from .exec_env import as_executor

# Never read/write .pyc while the agent rewrites modules — stale bytecode could
# mask a real fix/break on a rapid same-second rewrite.
_SUBPROC_ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

MUTATING = {"write_file", "edit_file", "bash"}
_DEFAULT_MAX_TOOL_OUTPUT = 16000
_LINE_CAP = 2000

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": ("Read a file's contents. Large files come back paginated: "
                        "the header shows 'lines A-B of N' and, if there's more, the "
                        "offset to read next. Use offset/limit to page through."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "1-based start line (default 1)."},
            "limit": {"type": "integer", "description": "Max lines (default 2000)."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace an exact substring in a file (old must be unique).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
            "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command in the repo root.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "run_tests",
        "description": "Run the repository's test suite. Reports passed/failed.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "search",
        "description": ("Search the repo. Give `pattern` to grep file contents "
                        "(returns file:line matches), or `glob` to list files by "
                        "name pattern (e.g. '*.py')."),
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Regex to grep for in file contents."},
            "glob": {"type": "string", "description": "Filename glob, e.g. '**/*.py'."}}}}},
    {"type": "function", "function": {
        "name": "git",
        "description": ("Run a git subcommand in the repo, e.g. 'status', 'diff', "
                        "'branch', 'add -A', 'commit -m \"msg\"'."),
        "parameters": {"type": "object", "properties": {
            "args": {"type": "string"}}, "required": ["args"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": ("Search the web (DuckDuckGo). Returns the top results as "
                        "title + URL + snippet. Use for docs, API/library usage "
                        "you're unsure of, or an error message. Follow up with "
                        "web_fetch to read a result in full."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "description": "1-10 (default 5)."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "web_fetch",
        "description": ("Fetch an http(s) web page and return its main text as "
                        "markdown. Use to read a page (e.g. a doc or a "
                        "web_search result) in full."),
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "done",
        "description": "Signal the goal is complete. The human reviews the diff.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"}}, "required": ["summary"]}}},
]

_WEB_DEPS_MSG = ("ERROR: web tools need 'ddgs' and 'html2text'. Install with "
                 "`pip install ddgs html2text` (bundled in a normal mlx-optiq "
                 "install).")


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(s: str) -> str:
    return _ANSI.sub("", s)


def truncate(s: str, limit: int = _DEFAULT_MAX_TOOL_OUTPUT) -> str:
    return s if len(s) <= limit else s[:limit] + "\n…[truncated]"


def parse_pytest(output: str) -> tuple[int, int]:
    """Extract (passed, failed) from pytest's summary line. Best-effort.
    ANSI is stripped first — pytest colorizes, which otherwise breaks the regex."""
    passed = failed = 0
    for line in strip_ansi(output).splitlines()[::-1]:
        if " passed" in line or " failed" in line or " error" in line:
            for n, kind in re.findall(r"(\d+)\s+(passed|failed|error|errors)", line):
                if kind == "passed":
                    passed = int(n)
                elif kind.startswith("error"):
                    failed += int(n)
                else:
                    failed = int(n)
            if passed or failed:
                break
    return passed, failed


def page_text(text: str, name: str, offset: int = 1, limit: int = 2000,
              max_chars: int = _DEFAULT_MAX_TOOL_OUTPUT) -> str:
    """Window file content the way Claude Code's Read does: a header
    (lines A-B of N) and an explicit offset hint when more remains, so a too-big
    file is navigable rather than a dead-end truncation. Raw content (no line
    prefixes) so edit_file's exact-substring match still works."""
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return f"[{name}: empty file]"
    start = max(1, offset)
    if start > n:
        return f"[{name}: {n} lines total; offset {offset} is past end of file]"
    want_end = min(n, start - 1 + max(1, limit))
    rendered: list[str] = []
    used = 0
    shown_end = start - 1
    for i in range(start, want_end + 1):
        ln = lines[i - 1]
        if len(ln) > _LINE_CAP:
            ln = ln[:_LINE_CAP] + "…[line truncated]"
        if used + len(ln) + 1 > max_chars and i > start:
            break
        rendered.append(ln)
        used += len(ln) + 1
        shown_end = i
    header = f"[{name}: lines {start}-{shown_end} of {n}]"
    body = "\n".join(rendered)
    if shown_end < n:
        return (f"{header}\n{body}\n[…{n - shown_end} more lines. "
                f"Call read_file with offset={shown_end + 1} to continue.]")
    return f"{header}\n{body}"


def _search(ex, args: dict, max_out: int) -> str:
    glob = args.get("glob")
    pattern = args.get("pattern")
    if glob:
        hits = ex.rglob(glob if "/" not in glob else Path(glob).name)
        hits = [h for h in hits if "__pycache__" not in h][:200]
        return truncate("\n".join(hits) or "(no files match)", max_out)
    if pattern:
        # ripgrep if available, else grep -rn; exclude the usual noise.
        rc, out = ex.run(
            f"grep -rnI --exclude-dir=__pycache__ --exclude-dir=.git -e {_q(pattern)} . 2>/dev/null | head -200",
            timeout=60, env=_SUBPROC_ENV)
        return truncate(out or "(no matches)", max_out)
    return "ERROR: search needs `pattern` or `glob`"


def _q(s: str) -> str:
    import shlex
    return shlex.quote(s)


def _web_search(args: dict, max_out: int) -> str:
    """Web search via the Lab's DuckDuckGo helper (ddgs). Read-only, no approval."""
    try:
        from ..lab.tools.web_search import search as _search_web
        results = _search_web(args.get("query", ""), args.get("max_results", 5))
    except ImportError:
        return _WEB_DEPS_MSG
    except Exception as e:
        return f"ERROR: web_search failed: {type(e).__name__}: {e}"
    if not results:
        return "(no results)"
    lines = [f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}"
             for i, r in enumerate(results, 1)]
    return truncate("\n".join(lines), max_out)


def _web_fetch(args: dict, max_out: int) -> str:
    """Fetch a page as markdown via the Lab's html2text helper. Read-only."""
    try:
        from ..lab.tools.web_search import fetch_page as _fetch
        return truncate(_fetch(args.get("url", "")), max_out)
    except ImportError:
        return _WEB_DEPS_MSG
    except Exception as e:
        return f"ERROR: web_fetch failed: {type(e).__name__}: {e}"


def execute_tool(name: str, args: dict, repo, verify_command: str | None,
                 max_tool_output: int = _DEFAULT_MAX_TOOL_OUTPUT) -> str:
    """Run one tool against the repo. Returns a text result. `repo` is a Path
    (local) or an Executor (container) — the loop is identical either way."""
    ex = as_executor(repo)
    try:
        if name == "read_file":
            path = args["path"]
            return page_text(ex.read_text(path), Path(path).name,
                             offset=int(args.get("offset", 1) or 1),
                             limit=int(args.get("limit", 2000) or 2000),
                             max_chars=max_tool_output)
        if name == "write_file":
            ex.write_text(args["path"], args["content"])
            return f"wrote {args['path']} ({len(args['content'])} bytes)"
        if name == "edit_file":
            text = ex.read_text(args["path"])
            if args["old"] not in text:
                return ("ERROR: `old` string not found in file. Read the file to "
                        "copy the exact text, or use write_file to rewrite it.")
            if text.count(args["old"]) > 1:
                return "ERROR: `old` string is not unique; include more surrounding context."
            ex.write_text(args["path"], text.replace(args["old"], args["new"], 1))
            return f"edited {args['path']}"
        if name == "bash":
            rc, out = ex.run(args["command"], timeout=300, env=_SUBPROC_ENV)
            return truncate(f"$ {args['command']}\n[exit {rc}]\n{out}", max_tool_output)
        if name == "run_tests":
            cmd = verify_command or "pytest -q"
            rc, out = ex.run(cmd, timeout=600, env=_SUBPROC_ENV)
            p, f = parse_pytest(out)
            return truncate(f"$ {cmd}\n[exit {rc}] passed={p} failed={f}\n{out}", max_tool_output)
        if name == "search":
            return _search(ex, args, max_tool_output)
        if name == "git":
            rc, out = ex.run(f"git {args.get('args', 'status')}", timeout=120, env=_SUBPROC_ENV)
            return truncate(f"$ git {args.get('args','status')}\n[exit {rc}]\n{out}", max_tool_output)
        if name == "web_search":
            return _web_search(args, max_tool_output)
        if name == "web_fetch":
            return _web_fetch(args, max_tool_output)
        return f"ERROR: unknown tool {name}"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out"
    except Exception as e:  # noqa: BLE001 — surface any tool failure to the model
        return f"ERROR: {type(e).__name__}: {e}"
