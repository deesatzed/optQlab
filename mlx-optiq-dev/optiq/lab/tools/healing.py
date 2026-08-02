"""Tool-call healing for sloppy local model output.

Frontier models emit tool calls in the structured ``tool_calls`` field
of an assistant message. Smaller / quantized local models routinely
fail at this and emit the call in one of several broken shapes:

  1. JSON object embedded in ``content`` (no ``tool_calls`` at all):
       ``{"name": "python", "arguments": {"code": "..."}}``
  2. Tagged form used by Qwen / Hermes-style fine-tunes:
       ``<tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>``
  3. Fenced-code form:
       ``` ```json\n{"name": "..."}\n``` ```
  4. ``arguments`` as a JSON string rather than an object, sometimes
     with backslash-escape errors or trailing commas.
  5. ``parameters`` instead of ``arguments`` (Anthropic-style).
  6. Function-call style without the outer wrapper:
       ``python({"code": "..."})``

This module normalizes all of those into the OpenAI-format
``tool_calls`` list so the downstream orchestrator can stay simple.

Entry points:

- ``heal_tool_calls(message, tool_names) -> (healed_message, n_recovered)``
- ``parse_tool_arguments(arguments_str) -> dict`` for last-mile arg parsing.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any


_TOOL_CALL_TAG = re.compile(
    r"<\s*tool_call\s*>(.*?)<\s*/\s*tool_call\s*>",
    re.DOTALL | re.IGNORECASE,
)
_JSON_FENCE = re.compile(
    r"```(?:json|tool|tool_call)?\s*\n(.*?)\n```",
    re.DOTALL,
)
_FUNCTION_CALL_BARE = re.compile(
    # name({json...}) at top level
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(\{.*?\})\s*\)\s*$",
    re.DOTALL,
)
# Mistral / Devstral. Two shapes ship in the wild: the newer per-call form
# `[TOOL_CALLS]name[ARGS]{...}` and the older JSON-array form
# `[TOOL_CALLS][{"name": ..., "arguments": {...}}]`. mlx-lm's `mistral` tool
# parser handles both when it recognizes them; these catch the cases where a
# quantized model's output drifts enough that the parser leaves it in content.
_MISTRAL_NAMED_CALL = re.compile(
    r"\[TOOL_CALLS\]\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\[ARGS\]\s*(\{.*?\})(?=\s*(?:\[TOOL_CALLS\]|\[/?TOOL_RESULTS\]|</s>|$))",
    re.DOTALL,
)
_MISTRAL_ARRAY_CALL = re.compile(
    r"\[TOOL_CALLS\]\s*(\[.*?\])(?=\s*(?:\[/?TOOL_RESULTS\]|</s>|$))",
    re.DOTALL,
)


def _strip_trailing_commas(s: str) -> str:
    """Remove trailing commas before ``}`` / ``]`` — common local-model slop."""
    return re.sub(r",(\s*[\}\]])", r"\1", s)


def _try_loads(text: str) -> Any | None:
    """Try ``json.loads`` with progressive cleanups. Returns parsed value
    or None if no variant parses."""
    if not text or not text.strip():
        return None
    candidates = [text, _strip_trailing_commas(text)]
    # Replace fancy quotes with ASCII
    fancy = text.replace("“", '"').replace("”", '"').replace("’", "'")
    if fancy != text:
        candidates.append(fancy)
        candidates.append(_strip_trailing_commas(fancy))
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Best-effort decode of an ``arguments`` field.

    The model may send arguments as:
      - a JSON object (already a dict) -> returned as-is
      - a JSON-encoded string -> decoded
      - a JSON-encoded string with trailing commas / fancy quotes -> cleaned + decoded
      - a string that itself contains an outer ``{...}`` block -> extracted then decoded
    Returns ``{}`` if nothing parses.
    """
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {}
    v = _try_loads(arguments)
    if isinstance(v, dict):
        return v
    # Try to find a JSON object inside the string
    m = re.search(r"\{.*\}", arguments, re.DOTALL)
    if m:
        v = _try_loads(m.group(0))
        if isinstance(v, dict):
            return v
    return {}


def _new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


def _to_openai_tool_call(name: str, args: Any) -> dict[str, Any]:
    """Build an OpenAI-shape ``tool_calls`` entry from name + args."""
    if isinstance(args, dict):
        args_str = json.dumps(args)
    elif isinstance(args, str):
        # leave as-is if it parses; otherwise wrap as a code/command
        if _try_loads(args) is not None:
            args_str = args
        else:
            args_str = json.dumps({"code": args} if name == "python"
                                  else {"command": args} if name == "terminal"
                                  else {"query": args} if name == "web_search"
                                  else {"raw": args})
    else:
        args_str = json.dumps({})
    return {
        "id": _new_call_id(),
        "type": "function",
        "function": {"name": name, "arguments": args_str},
    }


def _extract_calls_from_object(obj: Any, tool_names: set[str]) -> list[dict[str, Any]]:
    """Walk a parsed JSON value and pull out anything that looks like a tool call.

    Recognized shapes:
      - ``{"name": <tool>, "arguments": {...}}`` or ``"parameters": {...}``
      - ``{"tool_calls": [{"function": {"name": ..., "arguments": ...}}, ...]}``
      - ``{"function": {"name": ..., "arguments": ...}}``
      - top-level dict where key IS the tool name and value is the arg dict:
        ``{"python": {"code": "..."}}``
      - list of any of the above
    """
    out: list[dict[str, Any]] = []
    if obj is None:
        return out

    if isinstance(obj, list):
        for item in obj:
            out.extend(_extract_calls_from_object(item, tool_names))
        return out

    if not isinstance(obj, dict):
        return out

    if "tool_calls" in obj and isinstance(obj["tool_calls"], list):
        for tc in obj["tool_calls"]:
            out.extend(_extract_calls_from_object(tc, tool_names))
        return out

    if "function" in obj and isinstance(obj["function"], dict):
        fn = obj["function"]
        name = fn.get("name")
        if isinstance(name, str) and name in tool_names:
            args = fn.get("arguments")
            out.append(_to_openai_tool_call(name, args))
            return out

    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    if isinstance(name, str) and name in tool_names:
        args = obj.get("arguments")
        if args is None:
            args = obj.get("parameters")
        if args is None:
            args = obj.get("args")
        if args is None:
            args = obj.get("input")
        out.append(_to_openai_tool_call(name, args))
        return out

    # Last shape: top-level key IS the tool name.
    for k, v in obj.items():
        if isinstance(k, str) and k in tool_names and isinstance(v, dict):
            out.append(_to_openai_tool_call(k, v))
            return out

    return out


def _scan_content_for_calls(
    content: str, tool_names: set[str],
) -> tuple[list[dict[str, Any]], str]:
    """Scan free-form content for malformed tool calls.

    Returns ``(calls, remaining_content)`` where remaining_content has the
    healed segments stripped out so the user doesn't see raw JSON in chat.
    """
    if not content:
        return [], content

    calls: list[dict[str, Any]] = []
    stripped = content

    # 1) <tool_call>...</tool_call> tags
    for m in _TOOL_CALL_TAG.finditer(content):
        inner = m.group(1)
        v = _try_loads(inner)
        new = _extract_calls_from_object(v, tool_names)
        if new:
            calls.extend(new)
            stripped = stripped.replace(m.group(0), "")

    # 2) ```json fenced blocks
    for m in _JSON_FENCE.finditer(stripped):
        inner = m.group(1)
        v = _try_loads(inner)
        new = _extract_calls_from_object(v, tool_names)
        if new:
            calls.extend(new)
            stripped = stripped.replace(m.group(0), "")

    # 3) Bare top-level JSON object that IS a tool call
    candidate = stripped.strip()
    if candidate.startswith("{") and candidate.endswith("}"):
        v = _try_loads(candidate)
        new = _extract_calls_from_object(v, tool_names)
        if new:
            calls.extend(new)
            stripped = ""

    # 4) Function-call style: python({"code": "..."})
    if not calls:
        m = _FUNCTION_CALL_BARE.match(stripped)
        if m and m.group(1) in tool_names:
            v = _try_loads(m.group(2))
            if isinstance(v, dict):
                calls.append(_to_openai_tool_call(m.group(1), v))
                stripped = ""

    # 5) Mistral / Devstral [TOOL_CALLS] forms, per-call then array.
    for m in _MISTRAL_NAMED_CALL.finditer(stripped):
        name = m.group(1)
        if name not in tool_names:
            continue
        v = _try_loads(m.group(2))
        if isinstance(v, dict):
            calls.append(_to_openai_tool_call(name, v))
            stripped = stripped.replace(m.group(0), "")
    if not calls:
        for m in _MISTRAL_ARRAY_CALL.finditer(stripped):
            v = _try_loads(m.group(1))
            new = _extract_calls_from_object(v, tool_names)
            if new:
                calls.extend(new)
                stripped = stripped.replace(m.group(0), "")

    return calls, stripped.strip()


def heal_tool_calls(
    message: dict[str, Any], tool_names: list[str] | set[str],
) -> tuple[dict[str, Any], int]:
    """Return a normalized copy of ``message`` plus a count of healed calls.

    Behavior:
      - If ``message["tool_calls"]`` is already present and well-formed,
        we still normalize its ``arguments`` to a JSON-encoded string and
        ensure each call has an ``id``.
      - If it's missing, we scan ``message["content"]`` for malformed
        calls (see ``_scan_content_for_calls``) and synthesize them.
      - Recovered calls have their leftover content stripped so the model's
        "I'll call X" preamble doesn't bleed into the user-visible reply.

    Args:
        message: Assistant message dict from a chat completion.
        tool_names: Set of valid tool names. Used to avoid hallucinating
          a tool call when the model just happens to emit a JSON object.
    """
    names = set(tool_names)
    healed = dict(message)
    n_recovered = 0

    # Path A: well-formed tool_calls already present
    if isinstance(healed.get("tool_calls"), list) and healed["tool_calls"]:
        new_calls: list[dict[str, Any]] = []
        for tc in healed["tool_calls"]:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name")
            if not isinstance(name, str):
                continue
            args = fn.get("arguments")
            if args is None:
                args = tc.get("arguments")
            # Normalize arguments to a JSON-encoded string (OpenAI shape)
            parsed = parse_tool_arguments(args)
            new_call = {
                "id": tc.get("id") or _new_call_id(),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(parsed),
                },
            }
            new_calls.append(new_call)
        healed["tool_calls"] = new_calls
        return healed, 0

    # Path B: scan content for malformed calls
    content = healed.get("content") or ""
    if not isinstance(content, str):
        return healed, 0

    calls, stripped = _scan_content_for_calls(content, names)
    if not calls:
        return healed, 0

    healed["tool_calls"] = calls
    healed["content"] = stripped or None
    n_recovered = len(calls)
    return healed, n_recovered
