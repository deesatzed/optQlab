"""Tools available to chat models in OptiQ Lab.

Three tools, exposed as OpenAI tool-call schemas:

- ``web_search`` — DuckDuckGo via the ``ddgs`` library, plus a ``url``
  parameter to fetch full page text from a result.
- ``python`` — calls into ``optiq.lab.sandbox.run_python`` with strict
  AST checks.
- ``terminal`` — calls into ``optiq.lab.sandbox.run_terminal`` with strict
  blocked-command checks.

Public entry points:

- ``ALL_TOOLS`` — list of tool schemas to send in the ``tools=[...]`` body
- ``execute_tool(name, arguments, session_id=None) -> str`` — dispatches
  one tool call and returns a string result suitable for a ``role=tool``
  follow-up message.
"""
from __future__ import annotations

from .schemas import ALL_TOOLS, WEB_SEARCH_TOOL, PYTHON_TOOL, TERMINAL_TOOL
from .registry import execute_tool
from .healing import heal_tool_calls, parse_tool_arguments

__all__ = [
    "ALL_TOOLS",
    "WEB_SEARCH_TOOL",
    "PYTHON_TOOL",
    "TERMINAL_TOOL",
    "execute_tool",
    "heal_tool_calls",
    "parse_tool_arguments",
]


def tool_names() -> list[str]:
    """Convenience: list of available tool names."""
    return [t["function"]["name"] for t in ALL_TOOLS]
