"""Coalesce consecutive same-role messages before chat templating.

Mistral and Devstral templates enforce strict alternation and raise::

    After the optional system message, conversation roles must alternate
    user and assistant roles except for tool calls and results.

Qwen and Gemma templates are permissive, so agent loops that emit two user
turns in a row (a tool result followed by a nudge, a retry appended after a
prompt) work everywhere else and 404 only on Mistral. That is a client-shape
problem, not a model problem, so it is fixed once here rather than in each
agent.

Merging is safe for the permissive families too: their templates already
concatenate consecutive same-role turns when rendering, so the rendered prompt
is equivalent. ``tool`` messages are never merged — they carry a
``tool_call_id`` and the strict templates explicitly allow them between an
assistant turn and the next user turn.
"""

from __future__ import annotations

__all__ = ["normalize_role_alternation", "fold_user_after_tool"]

# Roles that must alternate. ``tool`` is exempt by the templates' own rules,
# and ``system`` only ever leads.
_MERGEABLE = ("user", "assistant")


def _text_of(content) -> str:
    """Message content as text, tolerating content-part lists and None."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _mergeable(message) -> bool:
    """Only merge plain text turns — never one carrying tool calls or parts.

    An assistant message with ``tool_calls`` is structurally significant, and a
    content-part list may hold images, so both are left as their own turn even
    when that means the alternation stays broken. Rendering a wrong prompt is
    worse than the template's error.
    """
    return (
        isinstance(message, dict)
        and message.get("role") in _MERGEABLE
        and not message.get("tool_calls")
        and isinstance(message.get("content"), str)
    )


def fold_user_after_tool(messages, separator: str = "\n\n"):
    """Fold a user turn that directly follows tool results into the last result.

    Agent harnesses interject after running tools ("you are not making
    progress", "the tests still fail"), which lands a ``user`` message straight
    after a ``tool`` message. Mistral and Devstral reject that shape: after tool
    results they want the assistant to speak next.

    Appending the text to the preceding tool result keeps the conversation
    valid without inventing an assistant turn. The alternative — bridging with a
    synthetic assistant message — would put words in the model's mouth, and an
    *empty* bridge does not work either (Mistral requires assistant content).
    The model still reads the instruction; it just arrives attached to the tool
    output that prompted it.

    Only plain-string turns are folded, so nothing carrying tool calls or
    content parts is disturbed.
    """
    if not isinstance(messages, list):
        return messages

    out: list = []
    for message in messages:
        prev = out[-1] if out else None
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and message.get("content").strip()
            and isinstance(prev, dict)
            and prev.get("role") == "tool"
            and isinstance(prev.get("content"), str)
        ):
            folded = dict(prev)
            folded["content"] = (
                f"{prev['content']}{separator}{message['content']}"
                if prev["content"] else message["content"]
            )
            out[-1] = folded
        else:
            out.append(message)
    return out


def normalize_role_alternation(messages, separator: str = "\n\n"):
    """Merge adjacent same-role user/assistant messages, in place-ish.

    Returns a new list; the caller decides whether to write it back. Anything
    unmergeable (tool calls, content parts, tool messages) passes through
    untouched and breaks the run of merges, so structure is never lost.
    """
    if not isinstance(messages, list):
        return messages

    out: list = []
    for message in messages:
        if (
            out
            and _mergeable(message)
            and _mergeable(out[-1])
            and out[-1].get("role") == message.get("role")
        ):
            merged = dict(out[-1])
            left, right = _text_of(merged.get("content")), _text_of(message.get("content"))
            merged["content"] = (
                f"{left}{separator}{right}" if left and right else (left or right)
            )
            out[-1] = merged
        else:
            out.append(message)
    return out
