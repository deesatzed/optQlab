"""Emit an OptiQ Code session as a HuggingFace Session-Traces-Format file.

Spec: https://huggingface.co/docs/hub/session-traces-format
  line 1 : {"type":"session","harness":"optiq-code","id","name", ...meta}
  line N : {"type":"message","message":{role, content, toolCalls?, toolCallId?, model?}}
    toolCalls  = [{"id","function":{"name","arguments"(json str)}}]   # camelCase
    tool result = {role:"tool", toolCallId, content}

Ported from conjure/agent/trace_writer.py (harness renamed).
"""
from __future__ import annotations

import json


def _to_toolcalls(tcs):
    out = []
    for t in tcs or []:
        fn = t.get("function") or {}
        out.append({"id": t.get("id"),
                    "function": {"name": fn.get("name"),
                                 "arguments": fn.get("arguments") or "{}"}})
    return out


def message_to_stf(m: dict) -> dict:
    """One OpenAI-format message dict -> Session-Traces-Format envelope."""
    role = m.get("role")
    inner = {"role": role, "content": m.get("content") or ""}
    if m.get("reasoning_content"):
        inner["reasoningContent"] = m["reasoning_content"]
    if m.get("tool_calls"):
        inner["toolCalls"] = _to_toolcalls(m["tool_calls"])
    if role == "tool" and m.get("tool_call_id"):
        inner["toolCallId"] = m["tool_call_id"]
    if m.get("model"):
        inner["model"] = m["model"]
    return {"type": "message", "message": inner}


def write_stf(path, messages: list, *, session_id: str, name: str | None = None,
              model: str | None = None, meta: dict | None = None) -> None:
    """Write a full trajectory to `path` as Session-Traces-Format JSONL."""
    header = {"type": "session", "harness": "optiq-code", "id": session_id}
    if name:
        header["name"] = name
    if model:
        header["model"] = model
    for k, v in (meta or {}).items():
        if v is not None:
            header[k] = v
    lines = [json.dumps(header)]
    lines += [json.dumps(message_to_stf(m)) for m in messages]
    text = "\n".join(lines) + "\n"
    if path in ("-", None):
        import sys
        sys.stdout.write(text)
    else:
        with open(path, "w") as f:
            f.write(text)
