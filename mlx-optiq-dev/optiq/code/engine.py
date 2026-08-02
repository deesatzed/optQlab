"""The OptiQ Code model client — a thin OpenAI-compatible client over `optiq serve`.

OptiQ Code is a *consumer* (design §2): it talks to whatever model the OptiQ
server is serving over the OpenAI Chat Completions API. This replaces conjure's
`llm/engine.py`; the `chat(messages, tools, max_tokens)` -> response contract is
the same, so the scripted test engine is a drop-in double.

Speed features (design §5.2) come from the server: prompt caching (the stable
system+context prefix is reused across turns, ~0.4s TTFT), and MTP if the served
model supports it. The client just structures a stable prefix and lets the server
cache it.
"""
from __future__ import annotations

import os

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"


class OptiqEngine:
    """OpenAI-compatible chat client pointed at the OptiQ server."""

    def __init__(
        self,
        model_id: str,
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        context_window: int | None = None,
        request_timeout: float | None = None,
        max_retries: int | None = None,
        temperature: float = 0.6,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - openai is a core dep
            raise ImportError("OptiQ Code requires the 'openai' package.") from e

        # Settings resolve through optiq.code.config (flag > env > repo > user
        # > default); an explicit argument here is the caller's own override.
        # `optiq code` resolves the config once *with the repo* and passes every
        # value in explicitly, so the repo tier reaches this client. This bare
        # load (no repo) is the fallback for a programmatic/test caller that
        # constructs the engine with just a model_id -- it still honors user
        # config + env, and any explicit argument wins over it.
        from .config import load as _load_config
        cfg, _ = _load_config()

        self.model_id = model_id
        self.base_url = base_url or cfg.base_url or DEFAULT_BASE_URL
        # The OptiQ server wants a Bearer token starting with 'sk-optiq-'.
        self.api_key = api_key or cfg.api_key or "sk-optiq-local"
        # A configured window wins over the server probe below -- set it when
        # pointing at a cloud endpoint that has no /optiq/context, so compaction
        # still has a budget to work from.
        self._context_window_override = (context_window if context_window is not None
                                          else cfg.context_window)
        self.reasoning_effort = reasoning_effort or cfg.reasoning_effort
        # A local 26B prefilling a few thousand tokens on a 24 GB Mac routinely
        # exceeds the SDK default; a too-short timeout reports a failed agent
        # run against a server that never erred.
        request_timeout = (cfg.request_timeout if request_timeout is None
                           else request_timeout)
        max_retries = cfg.max_retries if max_retries is None else max_retries
        self.temperature = temperature
        # max_retries above the SDK default absorbs a server that is still warming
        # up (Metal kernel compile) or a transient blip, rather than recording it
        # as a model failure (design §3.3 infra resilience).
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                             max_retries=max_retries,
                             timeout=request_timeout)

    def context_window(self) -> int | None:
        """The window the server will actually accept, or None if unknown.

        Read from the server rather than the model's config, because
        `optiq serve --max-context` may have capped it well below native —
        and that cap engages exactly on the tight-RAM machines where staying
        inside the window matters most. Best-effort: a stock mlx_lm.server or
        an older OptiQ has no such endpoint, and the caller falls back.

        A configured ``context_window`` (e.g. for a cloud endpoint that has no
        /optiq/context) wins over the probe -- that is the whole point of the
        override, so return it without touching the network.
        """
        if self._context_window_override and self._context_window_override > 0:
            return int(self._context_window_override)
        import json
        import urllib.error
        import urllib.request
        url = self.base_url.rstrip("/") + "/optiq/context"
        try:
            with urllib.request.urlopen(url, timeout=3.0) as r:
                win = json.loads(r.read()).get("context_window")
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return None
        try:
            win = int(win)
        except (TypeError, ValueError):
            return None
        return win if win > 0 else None

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int = 8192, temperature: float | None = None,
             on_token=None, on_reasoning=None):
        """Full chat completion. Returns a response whose
        `choices[0].message.{content,tool_calls}` and `usage` the loop reads.

        When ``on_token`` is given, the request streams and ``on_token(n)`` is
        called as deltas arrive (n = deltas seen so far, ~= tokens generated) so
        a caller can show live progress instead of a frozen screen. The streamed
        deltas are reassembled into the same response shape the non-streaming
        path returns, so the loop is unchanged.
        """
        temp = self.temperature if temperature is None else temperature
        kwargs = dict(model=self.model_id, messages=messages,
                     max_tokens=max_tokens, temperature=temp, top_p=0.95)
        if tools:
            kwargs["tools"] = tools
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if on_token is None:
            return self.client.chat.completions.create(**kwargs)

        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        content: list[str] = []
        calls: dict[int, dict] = {}
        usage = None
        finish_reason = None            # "stop" | "length" | "tool_calls" | ...
        chars = 0                       # total characters streamed this turn
        for chunk in self.client.chat.completions.create(**kwargs):
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not getattr(chunk, "choices", None):
                continue
            # Capture the terminal finish_reason so the loop can tell a completed
            # turn from one the server cut off at the output-token limit
            # ("length"). Without this the streamed path reports None and the loop
            # is blind to truncation -- a truncated tool call then looks like a
            # turn that simply made no tool call, and the model retries the same
            # oversized edit forever.
            if getattr(chunk.choices[0], "finish_reason", None):
                finish_reason = chunk.choices[0].finish_reason
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                content.append(piece)
                chars += len(piece)
            reasoning = (getattr(delta, "reasoning", None)
                         or getattr(delta, "reasoning_content", None))
            if reasoning:
                chars += len(reasoning)
                if on_reasoning is not None:
                    on_reasoning(reasoning)   # ephemeral thinking peek (not stored)
            deltas_tc = getattr(delta, "tool_calls", None)
            if deltas_tc:
                for tc in deltas_tc:
                    slot = calls.setdefault(getattr(tc, "index", 0),
                                            {"id": None, "name": "", "args": ""})
                    if getattr(tc, "id", None):
                        slot["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            slot["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            slot["args"] += fn.arguments; chars += len(fn.arguments)
            if piece or reasoning or deltas_tc:
                # mlx-lm buffers tool calls into 1-2 big deltas, so counting deltas
                # sticks at 1-2 during a write. Estimate tokens from characters
                # (~4 chars/token) so the counter climbs smoothly for tool calls too.
                on_token(max(1, chars // 4))
        # once the real usage lands, report the exact completion-token count
        if usage is not None and getattr(usage, "completion_tokens", None):
            on_token(usage.completion_tokens)
        return _StreamedResponse("".join(content), calls, usage, finish_reason)


class _StreamedFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _StreamedToolCall:
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.type = "function"
        self.function = _StreamedFunction(name, arguments)


class _StreamedMessage:
    def __init__(self, content: str, tool_calls: list):
        self.content = content
        self.role = "assistant"
        self.tool_calls = tool_calls or None


class _StreamedChoice:
    def __init__(self, message, finish_reason=None):
        self.message = message
        self.finish_reason = finish_reason


class _StreamedResponse:
    """Reassembles streamed deltas into the response shape the loop reads:
    ``choices[0].message.{content,tool_calls}``, ``finish_reason`` and ``usage``."""
    def __init__(self, content: str, calls: dict, usage, finish_reason=None):
        tool_calls = [
            _StreamedToolCall(c["id"], c["name"], c["args"])
            for c in calls.values() if c.get("id")
        ]
        self.choices = [
            _StreamedChoice(_StreamedMessage(content, tool_calls), finish_reason)]
        self.usage = usage
