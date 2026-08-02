"""OpenAI Responses API shim for ``mlx_lm.server``.

Adds a ``/v1/responses`` endpoint that speaks the OpenAI Responses API
so OptiQ-served models can be used as a drop-in for OpenAI's newer
agent stack. Codex switched to Responses exclusively (Chat Completions
deprecated for Codex), and other tools like Cursor, Continue, Cline,
and the OpenAI SDK speak Responses now too.

Architecture mirrors ``anthropic_shim.py``: translate the Responses
request to a Chat Completions request body, let mlx_lm.server's existing
handler (now MTP-aware via ``install_mtp_speculation``) do the work, then
translate the response back to Responses output shape.

Mapping (Responses ↔ OpenAI Chat Completions):

  Responses request               Chat Completions request
  -----------------               ------------------------
  instructions                    {role: "system"} prefix
  input (str or list of items)    messages
    - text item                     {role, content: text}
    - function_call item            {role: "assistant", tool_calls: [...]}
    - function_call_output item     {role: "tool", tool_call_id, content}
  tools (flat function shape)     tools (nested function shape)
  tool_choice                     tool_choice (translated)
  max_output_tokens               max_tokens
  temperature / top_p             same
  stream                          stream

  Chat Completions response       Responses response
  ------------------------        ------------------
  {choices: [{message: {           {id, object: "response",
    role, content,                  status: "completed",
    tool_calls: [...]}}]}           output: [
                                     {type: "message",
                                      content: [{type: "output_text",
                                                 text: "..."}]},
                                     {type: "function_call",
                                      call_id, name, arguments}
                                   ],
                                   usage: {input_tokens, output_tokens,
                                           total_tokens}}

Streaming events (SSE), in order:
  response.created
  response.in_progress
  response.output_item.added (per text/tool item)
  response.content_part.added (for text items)
  response.output_text.delta (per text token)
  response.output_text.done
  response.content_part.done
  response.output_item.done
  response.completed

Codex specifically listens for ``response.output_text.delta`` and
``response.completed`` so those two are non-negotiable. The other events
are emitted for spec compliance with the broader OpenAI SDK / Cursor /
Continue / Cline ecosystem.

Built-in Responses tools (web_search, file_search, mcp, computer_use)
are dropped silently. Only ``function`` tools are forwarded; built-ins
have server-side dependencies that local serving stacks don't provide.

Stateful response IDs (resumption via ``previous_response_id``) are not
implemented. Each request is treated as a new conversation.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional, Union


# ----------------------------------------------------------------------
# Request translation: Responses → Chat Completions
# ----------------------------------------------------------------------


def _coerce_text_content(content: Any) -> str:
    """Flatten a Responses content array (list of {type, text} parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            # Responses input parts: {type: "input_text", text: "..."}
            # Responses output parts: {type: "output_text", text: "..."}
            if part.get("text") is not None:
                parts.append(str(part["text"]))
    return "".join(parts)


def _translate_responses_tools_to_chat(
    tools: Optional[list[dict]],
) -> Optional[list[dict]]:
    """Flat Responses tool shape into nested Chat Completions shape.

    Responses::  {"type": "function", "name": "...", "description": "...",
                  "parameters": {...}, "strict": true}
    Chat::       {"type": "function",
                  "function": {"name": "...", "description": "...",
                               "parameters": {...}, "strict": true}}

    Built-in Responses tools (web_search, file_search, mcp, computer_use)
    are dropped. Forwarding them to llama-server would produce opaque 400s.
    """
    if not tools:
        return None
    out: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue  # silently drop built-ins
        fn: dict[str, Any] = {}
        for key in ("name", "description", "parameters", "strict"):
            if tool.get(key) is not None:
                fn[key] = tool[key]
        out.append({"type": "function", "function": fn})
    return out or None


def _translate_responses_tool_choice_to_chat(tool_choice: Any) -> Any:
    """Translate a Responses tool_choice to the Chat Completions shape."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        # "auto" / "none" / "required" pass through unchanged
        return tool_choice
    if (
        isinstance(tool_choice, dict)
        and tool_choice.get("type") == "function"
        and "name" in tool_choice
        and "function" not in tool_choice
    ):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return tool_choice


def output_items_to_input_items(output: list[dict]) -> list[dict]:
    """Convert a prior response's ``output`` array into ``input``-shaped
    items so it can be prepended to a new request when the client passes
    ``previous_response_id``.

    Mapping:
      message → message item (role=assistant, content preserved as-is —
        Responses input accepts the same content part types for prior
        assistant turns)
      function_call → function_call item (id rewritten as call_id)
      reasoning → dropped (the model produced this on the prior turn;
        replaying it would invite the model to repeat itself, and clients
        typically don't want reasoning fed back in)
    """
    items: list[dict] = []
    for item in output or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            items.append({
                "type": "message",
                "role": item.get("role", "assistant"),
                "content": item.get("content") or [],
            })
        elif itype == "function_call":
            items.append({
                "type": "function_call",
                "call_id": item.get("call_id") or item.get("id"),
                "name": item.get("name", ""),
                "arguments": item.get("arguments", ""),
            })
        # reasoning items intentionally dropped
    return items


def _normalize_input_to_messages(
    payload: dict,
) -> list[dict]:
    """Convert the Responses ``input`` (str OR list of items) plus
    ``instructions`` into a Chat Completions ``messages`` array.

    Handles the three Responses input item types:
      - ``message`` (or bare {role, content}): regular chat turn
      - ``function_call``: a prior assistant tool call being replayed
        on a follow-up turn. Becomes ``{role: "assistant", tool_calls: [...]}``
      - ``function_call_output``: a tool result the client is returning.
        Becomes ``{role: "tool", tool_call_id, content}``

    System messages from both ``instructions`` AND any ``role=system`` /
    ``role=developer`` items in input are merged and placed at the top.
    Strict chat templates (harmony / Qwen3) reject duplicate or
    non-leading system messages. Codex sends both ``instructions`` and a
    developer message in ``input``, so this merging is essential.
    """
    system_parts: list[str] = []
    messages: list[dict] = []

    instructions = payload.get("instructions")
    if instructions:
        system_parts.append(str(instructions))

    inp = payload.get("input")

    # Simple string input
    if isinstance(inp, str):
        if inp:
            messages.append({"role": "user", "content": inp})
        if system_parts:
            return [{"role": "system", "content": "\n\n".join(system_parts)}, *messages]
        return messages

    if not isinstance(inp, list):
        return messages

    for item in inp:
        if not isinstance(item, dict):
            continue

        itype = item.get("type")

        # Function call from a prior turn (assistant called a tool)
        if itype == "function_call":
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.get("call_id") or f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": item.get("arguments", ""),
                    },
                }],
            })
            continue

        # Function call output (tool result from client)
        if itype == "function_call_output":
            output = item.get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output)
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id") or "",
                "content": output,
            })
            continue

        # Regular message item (explicit type or implicit by having role)
        role = item.get("role", "user")
        content = _coerce_text_content(item.get("content"))

        if role in ("system", "developer"):
            if content:
                system_parts.append(content)
            continue

        if role not in ("user", "assistant", "tool"):
            role = "user"
        messages.append({"role": role, "content": content})

    if system_parts:
        return [{"role": "system", "content": "\n\n".join(system_parts)}, *messages]
    return messages


def responses_to_openai_body(responses_body: dict) -> dict:
    """Translate a Responses /v1/responses request body into the
    equivalent /v1/chat/completions body that mlx-lm's server understands.
    """
    oai: dict[str, Any] = {}
    oai["model"] = responses_body.get("model", "default_model")
    oai["messages"] = _normalize_input_to_messages(responses_body)

    if responses_body.get("max_output_tokens") is not None:
        oai["max_tokens"] = responses_body["max_output_tokens"]
    elif responses_body.get("max_tokens") is not None:
        oai["max_tokens"] = responses_body["max_tokens"]

    for key in ("temperature", "top_p", "top_k"):
        if responses_body.get(key) is not None:
            oai[key] = responses_body[key]

    tools = _translate_responses_tools_to_chat(responses_body.get("tools"))
    if tools:
        oai["tools"] = tools
    tc = _translate_responses_tool_choice_to_chat(responses_body.get("tool_choice"))
    if tc is not None:
        oai["tool_choice"] = tc

    if responses_body.get("stream"):
        oai["stream"] = True

    # Pass through chat_template_kwargs verbatim. We do NOT suppress
    # thinking for reasoning models — quality drops sharply when they
    # can't think. The reasoning emerges in the chat-completion
    # response's separate ``message.reasoning`` field, which we capture
    # into a Responses ``reasoning`` output item below.
    if responses_body.get("chat_template_kwargs") is not None:
        oai["chat_template_kwargs"] = responses_body["chat_template_kwargs"]

    return oai


# ----------------------------------------------------------------------
# Response translation: OpenAI → Responses (non-streaming)
# ----------------------------------------------------------------------


def _gen_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def _gen_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _gen_item_id(prefix: str = "msg") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


_STATUS_MAP = {
    "stop": "completed",
    "length": "incomplete",
    "tool_calls": "completed",
    "function_call": "completed",
    None: "completed",
}


def _chat_tool_calls_to_responses_items(tool_calls: list) -> list[dict]:
    """Translate Chat Completions tool_calls into Responses function_call items."""
    items: list[dict] = []
    for tc in tool_calls or []:
        fn = tc.get("function", {})
        items.append({
            "id": _gen_item_id("fc"),
            "type": "function_call",
            "status": "completed",
            "call_id": tc.get("id") or _gen_call_id(),
            "name": fn.get("name") or "",
            "arguments": fn.get("arguments") or "",
        })
    return items


def openai_to_responses_response(oai_resp: dict, model: str) -> dict:
    """Translate a non-streaming OpenAI completion response to Responses format.

    Reasoning models emit a separate ``message.reasoning`` field; we
    surface it as a Responses ``reasoning`` output item ahead of the
    message item. Codex and other clients that opt into reasoning will
    render it; clients that ignore the field still see the answer in
    the message output item.
    """
    choice = (oai_resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    reasoning_text = message.get("reasoning") or ""
    finish = choice.get("finish_reason")
    tool_calls = message.get("tool_calls") or []

    output: list[dict] = []
    if reasoning_text:
        output.append({
            "id": _gen_item_id("rs"),
            "type": "reasoning",
            "status": "completed",
            "summary": [{
                "type": "summary_text",
                "text": reasoning_text,
            }],
        })
    if text:
        output.append({
            "id": _gen_item_id("msg"),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": text,
                "annotations": [],
            }],
        })
    if tool_calls:
        output.extend(_chat_tool_calls_to_responses_items(tool_calls))

    usage = oai_resp.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens", 0))
    output_tokens = int(usage.get("completion_tokens", 0))
    reasoning_tokens = 0
    if reasoning_text:
        # Rough estimate — actual count is hidden inside mlx-lm; the
        # Responses spec only requires ``reasoning_tokens`` to be
        # non-zero when reasoning was emitted. Word-count as a proxy.
        reasoning_tokens = max(1, len(reasoning_text.split()))

    return {
        "id": _gen_response_id(),
        "object": "response",
        "created_at": int(time.time()),
        "status": _STATUS_MAP.get(finish, "completed"),
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": None,
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
        "user": None,
        "metadata": {},
    }


# ----------------------------------------------------------------------
# Streaming: SSE translation (OpenAI delta chunks → Responses events)
# ----------------------------------------------------------------------


def _sse(event: str, data: dict) -> bytes:
    """Format a single SSE frame as bytes."""
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, separators=(',', ':'))}\n\n"
    ).encode()


class ResponsesStreamTranslator:
    """Incremental translator: feed it OpenAI chat-completion stream chunks
    one at a time via ``add_chunk(chunk) -> Iterable[bytes]``, then call
    ``finalize() -> Iterable[bytes]`` once the upstream sends ``[DONE]``.

    The class wraps the generator in ``responses_stream_events_from_openai_chunks``
    so the SSE proxy can flush per-chunk instead of buffering the whole
    response. The lazy form is essential for thinking models — their
    first text token can arrive minutes after the request lands, and
    clients time out if they see no bytes during that window.
    """

    def __init__(self, model: str):
        self.model = model
        self.resp_id = _gen_response_id()
        self.created_at = int(time.time())
        self.text_item_id = _gen_item_id("msg")
        self.reasoning_item_id = _gen_item_id("rs")

        self.base_response = {
            "id": self.resp_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "in_progress",
            "error": None,
            "instructions": None,
            "max_output_tokens": None,
            "model": model,
            "output": [],
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "reasoning": None,
            "store": False,
            "temperature": None,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "truncation": "disabled",
            "usage": None,
            "user": None,
            "metadata": {},
        }

        self.text_item_started = False
        self.text_accum: list[str] = []
        self.reasoning_started = False
        self.reasoning_closed = False
        self.reasoning_output_index: Optional[int] = None
        self.reasoning_accum: list[str] = []
        self.tool_items: dict[int, dict] = {}
        self.tool_item_order: list[int] = []
        self.finish_reason: Optional[str] = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.output_index_counter = 0
        self._headers_sent = False

    def _headers(self):
        if self._headers_sent:
            return
        self._headers_sent = True
        yield _sse("response.created", {
            "type": "response.created", "response": self.base_response,
        })
        yield _sse("response.in_progress", {
            "type": "response.in_progress", "response": self.base_response,
        })

    def _close_reasoning_item(self):
        if not self.reasoning_started or self.reasoning_closed:
            return
        full = "".join(self.reasoning_accum)
        yield _sse("response.reasoning_summary_text.done", {
            "type": "response.reasoning_summary_text.done",
            "item_id": self.reasoning_item_id,
            "output_index": self.reasoning_output_index,
            "summary_index": 0,
            "text": full,
        })
        yield _sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": self.reasoning_output_index,
            "item": {
                "id": self.reasoning_item_id,
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": full}],
            },
        })
        self.reasoning_closed = True

    def _start_text_item(self):
        if self.text_item_started:
            return
        self.text_item_started = True
        yield _sse("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": self.output_index_counter,
            "item": {
                "id": self.text_item_id, "type": "message",
                "status": "in_progress", "role": "assistant", "content": [],
            },
        })
        yield _sse("response.content_part.added", {
            "type": "response.content_part.added",
            "item_id": self.text_item_id,
            "output_index": self.output_index_counter,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        })

    def add_chunk(self, chunk: dict):
        """Process one OpenAI chat-completion stream chunk; yield event bytes."""
        # Lazy header emission on first call.
        yield from self._headers()

        choices = chunk.get("choices") or []
        if not choices:
            usage = chunk.get("usage")
            if usage:
                self.input_tokens = int(usage.get("prompt_tokens", self.input_tokens))
                self.output_tokens = int(usage.get("completion_tokens", self.output_tokens))
            return
        choice = choices[0]
        delta = choice.get("delta") or {}
        fin = choice.get("finish_reason")
        if fin:
            self.finish_reason = fin

        reasoning_delta = delta.get("reasoning")
        if reasoning_delta:
            if not self.reasoning_started:
                self.reasoning_started = True
                self.reasoning_output_index = self.output_index_counter
                self.output_index_counter += 1
                yield _sse("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": self.reasoning_output_index,
                    "item": {
                        "id": self.reasoning_item_id, "type": "reasoning",
                        "status": "in_progress", "summary": [],
                    },
                })
            self.reasoning_accum.append(reasoning_delta)
            yield _sse("response.reasoning_summary_text.delta", {
                "type": "response.reasoning_summary_text.delta",
                "item_id": self.reasoning_item_id,
                "output_index": self.reasoning_output_index,
                "summary_index": 0,
                "delta": reasoning_delta,
            })

        token_text = delta.get("content")
        if token_text:
            yield from self._close_reasoning_item()
            yield from self._start_text_item()
            self.text_accum.append(token_text)
            self.output_tokens += 1
            yield _sse("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": self.text_item_id,
                "output_index": self.output_index_counter,
                "content_index": 0,
                "delta": token_text,
            })

        # Tool call deltas (streamed arguments).
        for tc_delta in delta.get("tool_calls", []) or []:
            tc_idx = tc_delta.get("index", 0)
            if tc_idx not in self.tool_items:
                self.tool_items[tc_idx] = {
                    "id": _gen_item_id("fc"),
                    "call_id": tc_delta.get("id") or _gen_call_id(),
                    "name": "",
                    "arguments": "",
                    "output_index": None,
                }
                self.tool_item_order.append(tc_idx)
            entry = self.tool_items[tc_idx]
            fn = tc_delta.get("function") or {}
            if fn.get("name"):
                entry["name"] += fn["name"]
            args_delta = fn.get("arguments") or ""
            if args_delta:
                entry["arguments"] += args_delta

            if entry["output_index"] is None:
                if self.text_item_started:
                    yield _sse("response.output_text.done", {
                        "type": "response.output_text.done",
                        "item_id": self.text_item_id,
                        "output_index": self.output_index_counter,
                        "content_index": 0,
                        "text": "".join(self.text_accum),
                    })
                    yield _sse("response.content_part.done", {
                        "type": "response.content_part.done",
                        "item_id": self.text_item_id,
                        "output_index": self.output_index_counter,
                        "content_index": 0,
                        "part": {"type": "output_text",
                                 "text": "".join(self.text_accum),
                                 "annotations": []},
                    })
                    yield _sse("response.output_item.done", {
                        "type": "response.output_item.done",
                        "output_index": self.output_index_counter,
                        "item": {
                            "id": self.text_item_id, "type": "message",
                            "status": "completed", "role": "assistant",
                            "content": [{
                                "type": "output_text",
                                "text": "".join(self.text_accum),
                                "annotations": [],
                            }],
                        },
                    })
                    self.output_index_counter += 1
                    self.text_item_started = False
                entry["output_index"] = self.output_index_counter
                self.output_index_counter += 1
                yield _sse("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": entry["output_index"],
                    "item": {
                        "id": entry["id"], "type": "function_call",
                        "status": "in_progress", "call_id": entry["call_id"],
                        "name": entry["name"], "arguments": "",
                    },
                })

            if args_delta:
                yield _sse("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "item_id": entry["id"],
                    "output_index": entry["output_index"],
                    "delta": args_delta,
                })

    def finalize(self):
        """Emit the trailing events: close open items + response.completed."""
        # In case no chunks ever arrived (empty response), emit headers.
        yield from self._headers()

        if self.text_item_started:
            yield _sse("response.output_text.done", {
                "type": "response.output_text.done",
                "item_id": self.text_item_id,
                "output_index": self.output_index_counter,
                "content_index": 0,
                "text": "".join(self.text_accum),
            })
            yield _sse("response.content_part.done", {
                "type": "response.content_part.done",
                "item_id": self.text_item_id,
                "output_index": self.output_index_counter,
                "content_index": 0,
                "part": {"type": "output_text",
                         "text": "".join(self.text_accum),
                         "annotations": []},
            })
            yield _sse("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": self.output_index_counter,
                "item": {
                    "id": self.text_item_id, "type": "message",
                    "status": "completed", "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": "".join(self.text_accum),
                        "annotations": [],
                    }],
                },
            })
            self.output_index_counter += 1

        yield from self._close_reasoning_item()

        for tc_idx in self.tool_item_order:
            entry = self.tool_items[tc_idx]
            yield _sse("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": entry["id"],
                "output_index": entry["output_index"],
                "arguments": entry["arguments"],
            })
            yield _sse("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": entry["output_index"],
                "item": {
                    "id": entry["id"], "type": "function_call",
                    "status": "completed", "call_id": entry["call_id"],
                    "name": entry["name"], "arguments": entry["arguments"],
                },
            })

        final_output: list[dict] = []
        if self.reasoning_started:
            final_output.append({
                "id": self.reasoning_item_id, "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text",
                             "text": "".join(self.reasoning_accum)}],
            })
        if self.text_accum:
            final_output.append({
                "id": self.text_item_id, "type": "message",
                "status": "completed", "role": "assistant",
                "content": [{"type": "output_text",
                             "text": "".join(self.text_accum),
                             "annotations": []}],
            })
        for tc_idx in self.tool_item_order:
            entry = self.tool_items[tc_idx]
            final_output.append({
                "id": entry["id"], "type": "function_call",
                "status": "completed", "call_id": entry["call_id"],
                "name": entry["name"], "arguments": entry["arguments"],
            })

        reasoning_tokens = (
            max(1, len("".join(self.reasoning_accum).split()))
            if self.reasoning_started else 0
        )
        completed = dict(self.base_response)
        completed.update({
            "status": _STATUS_MAP.get(self.finish_reason, "completed"),
            "output": final_output,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
            },
        })
        yield _sse("response.completed", {
            "type": "response.completed", "response": completed,
        })

    def final_response(self) -> dict:
        """Return the same dict that the ``response.completed`` event carries.

        Useful for the server to capture for ``previous_response_id`` after
        streaming has finished, without re-parsing its own SSE output.
        """
        final_output: list[dict] = []
        if self.reasoning_started:
            final_output.append({
                "id": self.reasoning_item_id, "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text",
                             "text": "".join(self.reasoning_accum)}],
            })
        if self.text_accum:
            final_output.append({
                "id": self.text_item_id, "type": "message",
                "status": "completed", "role": "assistant",
                "content": [{"type": "output_text",
                             "text": "".join(self.text_accum),
                             "annotations": []}],
            })
        for tc_idx in self.tool_item_order:
            entry = self.tool_items[tc_idx]
            final_output.append({
                "id": entry["id"], "type": "function_call",
                "status": "completed", "call_id": entry["call_id"],
                "name": entry["name"], "arguments": entry["arguments"],
            })
        completed = dict(self.base_response)
        completed.update({
            "status": _STATUS_MAP.get(self.finish_reason, "completed"),
            "output": final_output,
        })
        return completed


def responses_stream_events_from_openai_chunks(chunks, model: str):
    """Buffered convenience wrapper: feed all chunks at once, get all events.

    Used by callers that only need a one-shot translation (tests, batch
    code). Production streaming on the live server goes through
    ``ResponsesStreamTranslator`` directly so events flush per chunk.

    Emits at minimum (Codex requires these two):
      - ``response.created`` (initial response object)
      - ``response.output_text.delta`` (per token delta)
      - ``response.completed`` (final response object)

    Plus the surrounding spec-compliance events:
      - ``response.in_progress``
      - ``response.output_item.added`` (per text item or function_call)
      - ``response.content_part.added`` (for text items)
      - ``response.output_text.done`` / ``response.content_part.done`` /
        ``response.output_item.done`` (when items complete)
      - For tool calls: ``response.function_call_arguments.delta`` and ``.done``
    """
    translator = ResponsesStreamTranslator(model)
    for chunk in chunks:
        yield from translator.add_chunk(chunk)
    yield from translator.finalize()



