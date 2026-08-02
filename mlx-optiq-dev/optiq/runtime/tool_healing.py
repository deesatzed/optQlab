"""Server-side tool-call healing for ``optiq serve`` (and the Lab's server).

mlx-lm's server only turns model output into OpenAI ``tool_calls`` when the
tokenizer's tool parser recognizes the format. Quantized open-weight models
often emit a malformed shape instead (Hermes ``<tool_call>`` tags, fenced JSON,
bare JSON, trailing commas, fancy quotes, function-call form, the
``{"python": {...}}`` key-is-tool-name pattern), which leaks into ``content``
rather than being parsed. This installs a post-process that runs OptiQ's
``heal_tool_calls`` on the final completion message, recovering those into
proper ``tool_calls`` so any API client (agents, Claude Code via the OpenAI
endpoint) gets clean calls.

It is the same healer the Lab chat orchestrator uses; here it runs at the
server layer so it applies to every client, not just the Lab UI. Healing runs
only on **non-streaming** completions, where the full content is available (a
streamed malformed call has already left the wire token by token). Requests
without ``tools`` are untouched.

Wired into ``mlx_lm.server`` with two small monkeypatches (see ``install``):

* ``APIHandler.handle_completion`` — stash this request's tool names on a
  ContextVar (the handler thread that builds the response reads it).
* ``APIHandler.generate_response`` — for a non-stream message, heal it and
  promote ``finish_reason`` to ``tool_calls`` when a call is recovered.
"""

from __future__ import annotations

import contextvars

_REQ_TOOL_NAMES: contextvars.ContextVar = contextvars.ContextVar(
    "optiq_req_tool_names", default=None)
_installed = False


def _tool_names(tools) -> list[str]:
    """Pull the function names out of an OpenAI ``tools`` array."""
    names: list[str] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else None
        name = (fn or {}).get("name") or t.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def install(server_mod) -> None:
    """Patch ``mlx_lm.server.APIHandler`` to heal tool calls in completions."""
    global _installed
    if _installed:
        return
    try:
        from optiq.lab.tools.healing import heal_tool_calls
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, never break serve
        print(f"[optiq] tool-call healing unavailable: {exc}", flush=True)
        return

    Handler = server_mod.APIHandler
    _orig_handle = Handler.handle_completion
    _orig_generate_response = Handler.generate_response

    def handle_completion(self, request, stop_words):
        names = _tool_names(getattr(request, "tools", None))
        token = _REQ_TOOL_NAMES.set(names or None)
        try:
            return _orig_handle(self, request, stop_words)
        finally:
            _REQ_TOOL_NAMES.reset(token)

    def generate_response(self, text, finish_reason, *args, **kwargs):
        resp = _orig_generate_response(self, text, finish_reason, *args, **kwargs)
        names = _REQ_TOOL_NAMES.get()
        # Heal only the final, non-streaming message (full content in hand).
        if names and not getattr(self, "stream", False) and isinstance(resp, dict):
            for choice in resp.get("choices", []):
                msg = choice.get("message")
                if isinstance(msg, dict):
                    healed, n_recovered = heal_tool_calls(msg, names)
                    if n_recovered:
                        choice["message"] = healed
                        choice["finish_reason"] = "tool_calls"
        return resp

    Handler.handle_completion = handle_completion
    Handler.generate_response = generate_response
    _installed = True
