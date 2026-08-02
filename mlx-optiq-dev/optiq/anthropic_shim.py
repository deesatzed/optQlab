"""Anthropic Messages API shim for ``mlx_lm.server``.

Adds a ``/v1/messages`` endpoint that speaks Anthropic's Messages API so
OptiQ-served models can be used as a drop-in for Claude via
``ANTHROPIC_BASE_URL`` — e.g., inside Claude Code for a fully local
coding agent, or any other Anthropic-SDK-using tool.

Mapping (Anthropic ↔ OpenAI):

  Anthropic request               OpenAI request
  ----------------                ---------------
  system (top-level)              {role: "system"} prefix in messages
  messages[*].role (user/asst)    same
  messages[*].content (str|list)  flatten content-blocks → str
  max_tokens                      max_tokens
  temperature / top_p / top_k     same
  stop_sequences                  stop
  stream                          stream

  Anthropic response              OpenAI response
  ------------------              ----------------
  {id, type: "message",           {choices: [{message: {role,
   role: "assistant",              content}}], usage, ...}
   content: [{type:"text",
              text}],
   stop_reason, usage}

Streaming events translated 1:1:
  OpenAI delta ----→ Anthropic ``content_block_delta`` (text_delta)

Tool use: request ``tools`` (and ``tool_choice``) are forwarded to the model
as OpenAI function tools, so the model actually knows which tools exist;
the model's resulting tool calls are translated back to ``tool_use`` content
blocks. Prior ``tool_use``/``tool_result`` blocks in the conversation are
rendered inline as Qwen-style ``<tool_call>``/``<tool_response>`` text.
Because mlx-lm only parses tool calls on its non-streaming path, tool-bearing
requests are generated buffered and (when the client asked to stream)
replayed as SSE events — see ``anthropic_server._handle_stream_buffered``.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Optional


# --------------------------------------------------------------------------
# Request translation: Anthropic → OpenAI (internal)
# --------------------------------------------------------------------------

_VALID_ROLES = {"user", "assistant", "tool"}


def _normalize_content(content: Any) -> str:
    """Flatten Anthropic content (string or list of content blocks) to plain text.

    Text only. Tool blocks are *not* flattened here: they carry structure that
    every model family spells differently, so they are translated into real
    OpenAI ``tool_calls`` / ``role: "tool"`` messages by
    :func:`anthropic_to_openai_body` and rendered by the model's own chat
    template. Flattening them to one family's markup here is what broke
    non-Qwen models (see that function's docstring).

    Anthropic content blocks:
      {type: "text", text: "..."}
      {type: "image", source: {...}}     — not supported, returns placeholder
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "image":
            parts.append("[image omitted]")
        elif btype in ("tool_use", "tool_result"):
            # Only reachable for a stray tool block outside the message list
            # (e.g. in `system`). Family-neutral JSON beats one family's markup.
            parts.append(json.dumps(block))
        else:
            parts.append(block.get("text", "") or json.dumps(block))
    return "\n".join(parts)


def _translate_tools(tools: Any) -> list[dict]:
    """Anthropic tool defs → OpenAI chat-completions ``function`` tools.

    Anthropic custom tool:  {name, description, input_schema (JSON Schema)}
    OpenAI function tool:    {type: "function", function: {name, description,
                                                           parameters}}
    Anthropic server/built-in tools (web_search, computer_use, bash_20250124,
    …) carry a ``type`` and no ``input_schema``; the local model can't execute
    those, so they're dropped (mirrors the Responses shim).
    """
    out: list[dict] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        schema = t.get("input_schema")
        if not name or not isinstance(schema, dict):
            continue
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description", ""),
                "parameters": schema,
            },
        })
    return out


def _translate_tool_choice(tc: Any) -> Any:
    """Anthropic tool_choice → OpenAI tool_choice.

    auto→"auto", any→"required", none→"none", {type:tool,name}→
    {type:function, function:{name}}.
    """
    if tc is None:
        return None
    if isinstance(tc, str):
        return tc
    if isinstance(tc, dict):
        ctype = tc.get("type")
        if ctype == "auto":
            return "auto"
        if ctype == "any":
            return "required"
        if ctype == "none":
            return "none"
        if ctype == "tool" and tc.get("name"):
            return {"type": "function", "function": {"name": tc["name"]}}
    return None


def _split_blocks(content: Any) -> tuple[str, list[dict], list[dict]]:
    """One Anthropic message's blocks → (text, tool_use blocks, tool_result blocks)."""
    if not isinstance(content, list):
        return _normalize_content(content), [], []
    text_blocks, tool_uses, tool_results = [], [], []
    for block in content:
        if not isinstance(block, dict):
            text_blocks.append(block)
            continue
        btype = block.get("type")
        if btype == "tool_use":
            tool_uses.append(block)
        elif btype == "tool_result":
            tool_results.append(block)
        else:
            text_blocks.append(block)
    return _normalize_content(text_blocks), tool_uses, tool_results


def _tool_use_to_openai(block: dict) -> dict:
    """Anthropic tool_use block → OpenAI tool_call.

    ``arguments`` is serialized to a JSON string because that is the OpenAI wire
    format; mlx-lm (and :mod:`optiq.runtime.tool_args`) turn it back into a
    mapping before the chat template runs.
    """
    return {
        "id": block.get("id") or f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": block.get("name") or "",
            "arguments": json.dumps(block.get("input") or {}),
        },
    }


def anthropic_to_openai_body(anthropic_body: dict) -> dict:
    """Translate an Anthropic /v1/messages request body to the equivalent
    OpenAI /v1/chat/completions body that mlx-lm's server understands.

    Tool history is translated **structurally**, not flattened to text. Anthropic
    carries a tool call as a ``tool_use`` block inside an assistant message and
    its result as a ``tool_result`` block inside the *next user* message; OpenAI
    wants ``tool_calls`` on the assistant message and a separate ``role: "tool"``
    message. Emitting the structured form lets each model's own chat template
    render its native markup — ``[TOOL_CALLS]`` for Mistral/Devstral,
    ``<|tool_call>`` for Gemma-4, ``<tool_call>`` for Qwen, and so on.

    This used to inline Qwen-style ``<tool_call>`` XML into the message text,
    which meant every non-Qwen model saw tool history in a format it was never
    trained on, and saw tool *results* as if the user had typed them. Agent
    loops on Devstral, Gemma-4, GLM and Kimi degraded into repeated calls and
    ignored results.
    """
    oai: dict[str, Any] = {}
    oai["model"] = anthropic_body.get("model", "default_model")

    # Flatten messages — prepend system if provided.
    messages: list[dict] = []
    system = anthropic_body.get("system")
    if system:
        sys_text = _normalize_content(system) if isinstance(system, list) else str(system)
        messages.append({"role": "system", "content": sys_text})
    for m in anthropic_body.get("messages", []):
        role = m.get("role", "user")
        if role not in _VALID_ROLES:
            role = "user"
        text, tool_uses, tool_results = _split_blocks(m.get("content"))

        # Results answer the *previous* assistant turn, so they lead — OpenAI
        # requires each tool message to follow the assistant that called it.
        for block in tool_results:
            result = block.get("content", "")
            messages.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id") or "",
                "content": (_normalize_content(result)
                            if not isinstance(result, str) else result),
            })

        if tool_uses:
            # Assistant content may be empty when the turn is purely a call.
            messages.append({
                "role": "assistant",
                "content": text,
                "tool_calls": [_tool_use_to_openai(b) for b in tool_uses],
            })
        elif text or not tool_results:
            # Keep an empty message only when it carried nothing else, so a
            # user turn that was purely tool results doesn't become a blank one.
            messages.append({"role": role, "content": text})
    oai["messages"] = messages

    # Sampling / length.
    if "max_tokens" in anthropic_body:
        oai["max_tokens"] = anthropic_body["max_tokens"]
    for k in ("temperature", "top_p", "top_k"):
        if k in anthropic_body:
            oai[k] = anthropic_body[k]

    # Stop sequences.
    stops = anthropic_body.get("stop_sequences")
    if stops:
        oai["stop"] = stops

    # Tools — forward the tool definitions so the model actually knows which
    # tools exist (without this, tool-using clients like Claude Code get
    # text-only replies because the model sees no tools). Output-side
    # tool_calls → tool_use conversion is handled below.
    tools = _translate_tools(anthropic_body.get("tools"))
    if tools:
        oai["tools"] = tools
        choice = _translate_tool_choice(anthropic_body.get("tool_choice"))
        if choice is not None:
            oai["tool_choice"] = choice

    # Streaming passthrough.
    if anthropic_body.get("stream"):
        oai["stream"] = True

    # Pass through chat_template_kwargs verbatim. We intentionally do
    # NOT force enable_thinking=False — reasoning models perform much
    # better when allowed to think, and we capture the reasoning into
    # Anthropic-style ``thinking`` content blocks in the response below.
    if anthropic_body.get("chat_template_kwargs") is not None:
        oai["chat_template_kwargs"] = anthropic_body["chat_template_kwargs"]

    # Optional: disable the model's chain-of-thought on the Anthropic path.
    # Reasoning is great for one-shot answers but for agentic harnesses on a
    # local model (e.g. Claude Code) it dominates per-turn latency; this lets
    # an operator trade thinking for much faster tool-loop turns. Off by
    # default (reasoning stays on) — opt in with OPTIQ_ANTHROPIC_NO_THINK=1.
    if os.environ.get("OPTIQ_ANTHROPIC_NO_THINK") == "1":
        ctk = dict(oai.get("chat_template_kwargs") or {})
        ctk["enable_thinking"] = False
        oai["chat_template_kwargs"] = ctk

    return oai


# --------------------------------------------------------------------------
# Response translation: OpenAI → Anthropic
# --------------------------------------------------------------------------

_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    None: "end_turn",
}


def _gen_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def openai_to_anthropic_response(oai_resp: dict, model: str) -> dict:
    """Translate a non-streaming OpenAI completion response to Anthropic format.

    Reasoning models (Qwen3.5 / 3.6 / DeepSeek-R1) emit a separate
    ``message.reasoning`` field alongside ``message.content``. We map it
    to Claude 3.7's extended-thinking content block format so Anthropic
    clients that support it see the chain-of-thought, and clients that
    don't still see the final text block as the main answer.
    """
    choice = (oai_resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    reasoning = message.get("reasoning") or ""
    finish = choice.get("finish_reason")

    content_blocks: list[dict] = []
    if reasoning:
        content_blocks.append({
            "type": "thinking",
            "thinking": reasoning,
            "signature": "",
        })
    if text:
        content_blocks.append({"type": "text", "text": text})

    # OpenAI-style tool calls → Anthropic tool_use blocks.
    for tc in message.get("tool_calls", []) or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except (TypeError, json.JSONDecodeError):
            args = {"_raw": fn.get("arguments", "")}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}",
            "name": fn.get("name"),
            "input": args,
        })
        finish = "tool_calls"

    usage_oai = oai_resp.get("usage") or {}
    return {
        "id": _gen_message_id(),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": _STOP_REASON_MAP.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage_oai.get("prompt_tokens", 0)),
            "output_tokens": int(usage_oai.get("completion_tokens", 0)),
        },
    }


# --------------------------------------------------------------------------
# Streaming: SSE translation (OpenAI delta events → Anthropic events)
# --------------------------------------------------------------------------


class AnthropicStreamTranslator:
    """Incremental Anthropic SSE translator. Feed OpenAI chunks one at a
    time so each Anthropic event flushes to the client before the next
    upstream token arrives. Reasoning-model first-token latency on a
    laptop is well past Anthropic SDK timeouts otherwise.
    """

    def __init__(self, model: str):
        self.model = model
        self.msg_id = _gen_message_id()
        self.stop_reason = "end_turn"
        self.output_tokens = 0
        self.thinking_started = False
        self.thinking_closed = False
        self.text_started = False
        self.text_index = 0
        self._headers_sent = False

    def _headers(self):
        if self._headers_sent:
            return
        self._headers_sent = True
        yield _sse("message_start", {
            "type": "message_start",
            "message": {
                "id": self.msg_id, "type": "message", "role": "assistant",
                "model": self.model, "content": [],
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

    def _open_thinking(self):
        return _sse("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        })

    def _close_thinking(self):
        # signature is empty for unsigned local reasoning (the spec
        # allows it; clients render thinking deltas regardless).
        yield _sse("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": ""},
        })
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": 0,
        })

    def _open_text(self, idx):
        return _sse("content_block_start", {
            "type": "content_block_start",
            "index": idx,
            "content_block": {"type": "text", "text": ""},
        })

    def add_chunk(self, chunk: dict):
        yield from self._headers()
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        fin = choice.get("finish_reason")

        reasoning_text = delta.get("reasoning")
        if reasoning_text:
            if not self.thinking_started:
                self.thinking_started = True
                self.text_index = 1
                yield self._open_thinking()
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": reasoning_text},
            })

        token_text = delta.get("content")
        if token_text:
            if self.thinking_started and not self.thinking_closed:
                yield from self._close_thinking()
                self.thinking_closed = True
            if not self.text_started:
                self.text_started = True
                yield self._open_text(self.text_index)
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": self.text_index,
                "delta": {"type": "text_delta", "text": token_text},
            })
            self.output_tokens += 1

        if fin:
            self.stop_reason = _STOP_REASON_MAP.get(fin, "end_turn")

    def finalize(self):
        yield from self._headers()
        if self.thinking_started and not self.thinking_closed:
            yield from self._close_thinking()
            self.thinking_closed = True
        if self.text_started:
            yield _sse("content_block_stop", {
                "type": "content_block_stop",
                "index": self.text_index,
            })
        elif not self.thinking_started:
            yield self._open_text(0)
            yield _sse("content_block_stop", {
                "type": "content_block_stop",
                "index": 0,
            })
        yield _sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": self.output_tokens},
        })
        yield _sse("message_stop", {"type": "message_stop"})


def anthropic_stream_events_from_openai_chunks(chunks, model: str):
    """Buffered convenience wrapper: feed all chunks at once, get all bytes.

    Production serving uses ``AnthropicStreamTranslator`` directly so
    events flush per chunk. Kept for tests + offline translation.

    Emits in order:
      message_start
      (if model reasons)
        content_block_start (index=0, type=thinking)
        content_block_delta (thinking_delta, many)
        content_block_stop  (index=0)
      content_block_start (text, next index)
      content_block_delta (text_delta, many)
      content_block_stop
      message_delta (carries stop_reason + output_tokens)
      message_stop
    """
    translator = AnthropicStreamTranslator(model)
    for chunk in chunks:
        yield from translator.add_chunk(chunk)
    yield from translator.finalize()


def _sse(event: str, data: dict) -> bytes:
    """Format a single Server-Sent Events frame."""
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, separators=(',', ':'))}\n\n"
    ).encode()
