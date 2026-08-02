"""Normalize ``tool_calls[].function.arguments`` before chat templating.

The OpenAI wire format carries tool-call arguments as a JSON **string**.
Chat templates want them as a **mapping**, and Gemma-4's canonical template
(Google, published 2026-07-09) makes that explicit: anything that is not a
mapping or ``None`` hits ``raise_exception``. So every path that renders
messages through ``apply_chat_template`` has to coerce first.

mlx-lm does coerce, in ``mlx_lm.server.process_message_content``::

    if args := func.get("arguments"):
        func["arguments"] = json.loads(args)

but it has two holes. ``json.loads`` raises ``TypeError`` when a client sends
arguments as an object rather than a string, and the ``if args:`` guard skips
falsy values, so ``""`` (a no-argument tool call) survives as a string and
reaches the template. Both end the request: the first as a 500, the second as
a template exception.

``coerce_arguments_to_json_string`` closes both holes ahead of mlx-lm's own
pass, leaving it a well-formed non-empty JSON object string in every case.
``normalize_tool_arguments`` is the direct form for templating paths that
don't run through mlx-lm at all (the cluster server).
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

__all__ = ["coerce_arguments_to_json_string", "normalize_tool_arguments"]


def _iter_functions(messages):
    """Yield every ``function`` dict on every tool call, defensively."""
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if isinstance(function, dict):
                yield function


def _as_mapping(arguments, name: str) -> dict:
    """Best-effort coercion of one ``arguments`` value to a mapping.

    Unparseable arguments degrade to ``{}`` with a warning rather than raising.
    A malformed tool call has already failed upstream by the time it is being
    replayed as history, and losing the turn's structure to a 500 is worse than
    rendering an empty argument list.
    """
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            logger.warning(
                "tool call %r: arguments are not valid JSON, rendering as empty. %r",
                name, arguments[:200],
            )
            return {}
        if isinstance(parsed, dict):
            return parsed
        logger.warning(
            "tool call %r: arguments decoded to %s, not an object, rendering as "
            "empty.", name, type(parsed).__name__,
        )
        return {}
    logger.warning(
        "tool call %r: arguments are %s, not a string or object, rendering as "
        "empty.", name, type(arguments).__name__,
    )
    return {}


def normalize_tool_arguments(messages):
    """Coerce every tool call's ``arguments`` to a mapping, in place.

    Use on templating paths that do not go through mlx-lm's own conversion.
    """
    for function in _iter_functions(messages):
        function["arguments"] = _as_mapping(
            function.get("arguments"), function.get("name") or "?",
        )
    return messages


def coerce_arguments_to_json_string(messages):
    """Coerce every tool call's ``arguments`` to a non-empty JSON object string.

    The inverse-looking sibling of :func:`normalize_tool_arguments`, and the one
    to use *ahead of* ``mlx_lm.server.process_message_content``: it hands the
    stock ``json.loads`` exactly the shape it expects, so mlx-lm performs the
    final conversion to a mapping itself. ``{}`` is emitted rather than ``""``
    so mlx-lm's falsy-skip cannot leave a string behind.
    """
    for function in _iter_functions(messages):
        mapping = _as_mapping(function.get("arguments"), function.get("name") or "?")
        try:
            function["arguments"] = json.dumps(mapping)
        except (TypeError, ValueError):
            logger.warning(
                "tool call %r: arguments are not JSON-serializable, rendering as "
                "empty.", function.get("name") or "?",
            )
            function["arguments"] = "{}"
    return messages
