"""OpenAI tool schemas for the tools available in OptiQ Lab.

Each schema follows the OpenAI Chat Completions ``tools=[...]`` format:

  {
    "type": "function",
    "function": {
      "name": ...,
      "description": ...,
      "parameters": {...JSON schema...},
    },
  }
"""
from __future__ import annotations


WEB_SEARCH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web with DuckDuckGo, or fetch a specific page's "
            "text content. Two modes: pass `query` to get snippet results "
            "back; pass `url` to fetch the full text of a single page. "
            "Use this to answer questions about current events, look up "
            "documentation, or read a URL the user provided."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Omit when using `url`.",
                },
                "url": {
                    "type": "string",
                    "description": (
                        "A specific URL to fetch full page text from. "
                        "Use this after `query` returns snippets to read "
                        "a result in detail."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results when searching (default 5, max 10).",
                },
            },
            "required": [],
        },
    },
}


PYTHON_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Execute Python code in a sandboxed subprocess and return "
            "stdout / stderr. The sandbox has no network access and a "
            "wall-time + memory limit. Use this for calculations, data "
            "manipulation, plotting (matplotlib will write PNGs to the "
            "session workdir which the UI can render), and verifying "
            "code snippets you reason about."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python source to run.",
                },
            },
            "required": ["code"],
        },
    },
}


TERMINAL_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": (
            "Run a bash command in a sandboxed subprocess and return "
            "stdout / stderr. Common shell tools (ls, cat, grep, find, "
            "awk, sed, python, etc.) are available. Dangerous commands "
            "(rm, dd, sudo, curl, ssh, etc.) are blocked at command "
            "position. No network access; filesystem writes are limited "
            "to the session workdir."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run.",
                },
            },
            "required": ["command"],
        },
    },
}


ALL_TOOLS: list[dict] = [WEB_SEARCH_TOOL, PYTHON_TOOL, TERMINAL_TOOL]
