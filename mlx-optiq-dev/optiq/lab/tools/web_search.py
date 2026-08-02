"""DuckDuckGo web search + page-fetch helper.

Two operating modes:

  - ``search(query, max_results)`` — returns a list of dicts with
    ``title``, ``url``, ``snippet``.
  - ``fetch_page(url)`` — returns the page body as compact markdown,
    truncated to ``MAX_PAGE_CHARS`` so tool output stays under the
    context budget.

Both are exposed by ``execute_web_search`` which dispatches based on the
``arguments`` shape from the model: ``url`` present means fetch, else
treat as a search query.
"""
from __future__ import annotations

import urllib.request
from typing import Any


MAX_PAGE_CHARS = 8000  # truncate fetched pages so they fit a tool reply
DEFAULT_MAX_RESULTS = 5


def search(query: str, max_results: int = DEFAULT_MAX_RESULTS,
           timeout: int = 15) -> list[dict[str, str]]:
    """Run a DuckDuckGo text search via the ``ddgs`` library."""
    from ddgs import DDGS  # imported lazily to avoid import cost at startup

    max_results = max(1, min(int(max_results or DEFAULT_MAX_RESULTS), 10))
    raw = DDGS(timeout=timeout).text(query, max_results=max_results)
    out: list[dict[str, str]] = []
    for r in raw or []:
        out.append(
            {
                "title": str(r.get("title") or ""),
                "url": str(r.get("href") or ""),
                "snippet": str(r.get("body") or ""),
            }
        )
    return out


def fetch_page(url: str, timeout: int = 20) -> str:
    """Download ``url`` and return its main text content as markdown.

    Uses a desktop UA, follows redirects, and runs the HTML through
    ``html2text`` for a compact markdown form. Output is hard-capped at
    ``MAX_PAGE_CHARS`` characters so tool replies stay bounded.
    """
    import html2text

    if not url.startswith(("http://", "https://")):
        return f"Refusing to fetch non-http(s) URL: {url}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "").lower()
            data = resp.read(2_000_000)  # 2 MB cap on raw body
    except Exception as e:
        return f"Fetch failed: {e.__class__.__name__}: {e}"

    if "text/" not in ctype and "html" not in ctype:
        return f"Non-text content type: {ctype}"

    try:
        html = data.decode("utf-8", errors="replace")
    except Exception:
        html = data.decode("latin-1", errors="replace")

    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.ignore_emphasis = False
    h.body_width = 0  # no wrapping
    md = h.handle(html).strip()

    if len(md) > MAX_PAGE_CHARS:
        md = md[:MAX_PAGE_CHARS] + f"\n\n[...truncated at {MAX_PAGE_CHARS} chars]"
    return md


def execute_web_search(arguments: dict[str, Any]) -> str:
    """Dispatcher used by the tool registry. Returns a string response.

    ``arguments`` shape from the model:
      - ``{"query": "..."}`` -> search and return formatted snippets
      - ``{"url": "..."}``   -> fetch the page text
      - ``{"query": "...", "url": "..."}`` -> url wins (snippets are
        usually what led the model to want the page)
    """
    url = (arguments.get("url") or "").strip()
    if url:
        return fetch_page(url)

    query = (arguments.get("query") or "").strip()
    if not query:
        return "Error: provide either `query` or `url`."

    max_results = arguments.get("max_results", DEFAULT_MAX_RESULTS)
    try:
        results = search(query, max_results=int(max_results))
    except Exception as e:
        return f"Search failed: {e.__class__.__name__}: {e}"

    if not results:
        return "No results found."

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r['title']}\n    {r['url']}\n    {r['snippet']}")
    body = "\n\n---\n\n".join(parts)
    body += (
        "\n\n---\n\n"
        "IMPORTANT: These are only short snippets, not full pages. If the "
        "snippets do not contain the answer you need, call web_search again "
        "with `{\"url\": \"<URL>\"}` to fetch the full text of one of the "
        "results above. Do not guess based on titles alone."
    )
    return body
