"""Server-side chat orchestrator with tool execution loop.

Why this exists
---------------
The plain "browser streams from mlx-lm" path works fine for chat with no
tools. The moment tools enter the picture, the orchestration has to
happen somewhere: someone has to detect tool calls, execute them, append
``role=tool`` results, and re-call the model. Doing this in JavaScript
means trusting the browser to run sandboxed Python — that's a non-starter.

So when the user enables tools, the browser hits ``/api/chat/stream``
(an SSE endpoint backed by this module) instead of going direct. The
orchestrator:

  1. Streams a chat-completions request with ``tools=[...]`` from the local
     mlx-lm API, forwarding each delta as an SSE ``token`` event and
     reassembling the turn into the same envelope the loop below consumes.
     (It used to buffer the whole turn — ``"stream": False`` — which meant
     that with tools on, the default, chat never streamed token by token on
     any backend, and a 4000-token answer looked like a four-minute hang.)
  2. Heals any malformed tool calls in the assistant message. When healing
     rewrites what was already streamed — pulling a tool call out of the
     prose — a ``replace`` event replays the corrected transcript.
  3. If there are tool calls, executes them with ``execute_tool`` and
     appends ``role=tool`` messages.
  4. Loops up to ``MAX_TOOL_TURNS`` times, then forces one final
     tools-disabled turn (also streamed) so the model commits to an answer.

Each step emits an SSE event the UI can render as a tool card:

  - ``token``        ``{"text": "...", "reasoning": "..."}``
  - ``replace``      ``{"text": "..."}``  (corrected transcript for this turn)
  - ``tool_call``    ``{"id": "...", "name": "...", "arguments": {...}, "turn": 1}``
  - ``tool_result``  ``{"id": "...", "name": "...", "result": "...", "turn": 1}``
  - ``error``        ``{"message": "..."}``
  - ``done``         ``{}``
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Iterator

from .tools import ALL_TOOLS, execute_tool, heal_tool_calls, tool_names


# Session id -> threading.Event. The /api/chat/stream endpoint registers
# an entry when a stream begins; the /api/chat/cancel endpoint sets it
# when the user clicks Stop. Cleaned up when the stream ends.
_CANCEL_EVENTS: dict[str, threading.Event] = {}
_CANCEL_LOCK = threading.Lock()


def register_session(session_id: str | None = None) -> tuple[str, threading.Event]:
    """Register a new chat session. Returns ``(session_id, cancel_event)``.
    If ``session_id`` is None we mint a fresh one."""
    sid = session_id or "sess_" + uuid.uuid4().hex[:16]
    ev = threading.Event()
    with _CANCEL_LOCK:
        _CANCEL_EVENTS[sid] = ev
    return sid, ev


def cancel_session(session_id: str) -> bool:
    """Trigger the cancel event for ``session_id``. Returns True iff a live
    event was found (i.e. the cancel actually has somewhere to go)."""
    with _CANCEL_LOCK:
        ev = _CANCEL_EVENTS.get(session_id)
    if ev is None:
        return False
    ev.set()
    return True


def cleanup_session(session_id: str) -> None:
    """Drop the event from the registry once the stream ends."""
    with _CANCEL_LOCK:
        _CANCEL_EVENTS.pop(session_id, None)


MAX_TOOL_TURNS = 25                # Matches Unsloth Studio's default; well
                                   # above the 1-3 turns a normal session
                                   # needs but stops genuinely runaway models.
DEFAULT_TIMEOUT = 600.0            # Seconds for one mlx-lm call
DEFAULT_TEMP = 0.7
DEFAULT_MAX_TOKENS = 1024


DUPLICATE_NUDGE = (
    "You just called this tool with identical arguments and the previous "
    "call succeeded. Use that result directly instead of re-running."
)
BUDGET_EXHAUSTED_NUDGE = (
    "You have used the tool budget for this turn. Stop calling tools and "
    "answer the user's question now using only the information you have."
)
TOOL_ERROR_NUDGE = (
    "\n\nThe tool call encountered an issue. Try a different approach, "
    "different arguments, or pick a different tool."
)
# Self-healing: how many times the SAME (name, args) call may be re-run after
# it has failed. Beyond this the loop stops burning the sandbox on a call that
# clearly will not work and tells the model to change course.
MAX_RETRIES_PER_CALL = 3
REPEATED_FAILURE_NUDGE = (
    "This exact tool call has now failed {n} times. Do NOT call it again with "
    "the same arguments. Either call a different tool, change the arguments "
    "substantially, or answer the user with what you already have."
)

# Tool results that begin with one of these prefixes are treated as errors.
# Matches the pattern Unsloth Studio uses; lets us surface a recovery nudge
# to the model without parsing tool-specific formats.
_TOOL_ERROR_PREFIXES: tuple[str, ...] = (
    "Error",
    "Error:",
    "error:",
    "Failed",
    "Blocked:",
    "Exit code",
    "Search failed",
    "Fetch failed",
    "sandbox: rejected",
    "sandbox: blocked",
    "Tool '",  # registry's "Tool 'X' raised ..." path
    "Refusing to fetch",
    "No results found.",
    "No query provided",
)


def _extract_images(result: str) -> tuple[str, list[dict[str, str]]]:
    """Pull the __IMAGES__: sentinel (if any) out of a tool result string.

    Returns ``(cleaned_result, images)``. The cleaned result has the
    sentinel line removed and replaced with a short ``[N images attached]``
    note. The base64 payload then never enters the model's context (and
    is shipped to the UI via a separate SSE field).
    """
    sentinel = "__IMAGES__:"
    idx = result.find(sentinel)
    if idx == -1:
        return result, []
    line_end = result.find("\n", idx)
    line_end = len(result) if line_end == -1 else line_end
    payload = result[idx + len(sentinel): line_end].strip()
    try:
        images = json.loads(payload)
        if not isinstance(images, list):
            images = []
    except json.JSONDecodeError:
        images = []
    if not images:
        cleaned = result[:idx] + result[line_end:]
        return cleaned.rstrip(), []
    note = f"[{len(images)} image{'s' if len(images) != 1 else ''} attached to the chat]"
    cleaned = result[:idx].rstrip() + ("\n" + note if result[:idx].strip() else note) + result[line_end:]
    return cleaned.rstrip(), images


def _is_tool_error(result: str) -> bool:
    """Best-effort: did this tool call fail? Matches result prefix against
    the known error sigils. Used to decide whether to (a) suppress a
    consecutive identical call and (b) append the recovery nudge."""
    if not isinstance(result, str):
        return False
    head = result.lstrip()
    if not head:
        return False
    for p in _TOOL_ERROR_PREFIXES:
        if head.startswith(p):
            return True
    return "rc=-2" in head[:200]


def _call_signature(tool_call: dict) -> tuple[str, str]:
    """Stable identity for de-dupe detection: (name, normalized args)."""
    fn = tool_call.get("function") or {}
    return (fn.get("name") or "", (fn.get("arguments") or "").strip())


def _sse(event: str, data: dict[str, Any]) -> bytes:
    """Format a Server-Sent Event line."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


def _chat_completions_url(api_url: str) -> str:
    """Build the chat-completions URL from a base.

    The local ``optiq serve`` base has no version segment (``http://host:port``),
    so we append ``/v1/chat/completions``. A cloud base the user pastes usually
    already ends in ``/v1`` (``https://openrouter.ai/api/v1``); appending another
    ``/v1`` would 404, so only add the version segment when it is missing.
    """
    base = api_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _request(api_url: str, body: dict[str, Any], api_key: str | None):
    return urllib.request.Request(
        _chat_completions_url(api_url),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or 'sk-optiq-local'}",
        },
        method="POST",
    )


def _call_mlx_lm(
    api_url: str, body: dict[str, Any], timeout: float, api_key: str | None,
) -> dict[str, Any]:
    """Single non-streaming chat-completions call. Returns parsed JSON."""
    with urllib.request.urlopen(_request(api_url, body, api_key), timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _stream_mlx_lm(
    api_url: str, body: dict[str, Any], timeout: float, api_key: str | None,
) -> Iterator[dict[str, Any]]:
    """One chat-completions turn, streamed.

    Yields ``{"type": "delta"|"reasoning", "text": str}`` as tokens arrive and
    finally ``{"type": "final", "resp": ...}`` carrying a response shaped like
    the non-streaming envelope, so the tool loop downstream is unchanged.

    The turn is streamed rather than awaited because the caller has a live SSE
    connection to a browser: waiting for the whole completion before emitting
    anything makes a 4000-token answer look like a four-minute hang.
    """
    b = dict(body)
    b["stream"] = True
    content, reasoning, finish = "", "", None
    # Streamed tool calls arrive as fragments across many deltas, keyed by
    # ``index``: the id and name land once, the arguments string is chunked and
    # must be concatenated. (mlx-lm buffers each call into one delta so a plain
    # extend happened to work locally, but every OpenAI-compatible cloud endpoint
    # -- OpenRouter, OpenAI -- fragments the arguments, which arrived here as a
    # pile of separate calls with empty arguments.) Merge by index instead.
    tool_slots: dict[int, dict[str, Any]] = {}

    with urllib.request.urlopen(_request(api_url, b, api_key), timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    idx = tc.get("index", 0)
                    slot = tool_slots.setdefault(idx, {
                        "id": None, "type": "function",
                        "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
            if delta.get("reasoning"):
                reasoning += delta["reasoning"]
                yield {"type": "reasoning", "text": delta["reasoning"]}
            if delta.get("content"):
                content += delta["content"]
                yield {"type": "delta", "text": delta["content"]}

    tool_calls = [tool_slots[i] for i in sorted(tool_slots)]

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    yield {"type": "final", "resp": {"choices": [
        {"index": 0, "finish_reason": finish or "stop", "message": message}]}}


def _strip_tool_artifacts(content: str | None) -> str:
    """When healing pulls a tool call out of content, sometimes a stray
    "Calling tool..." / "Let me execute..." preamble is left over. Trim
    obvious filler down to nothing if the rest is purely transitional."""
    if not content:
        return ""
    s = content.strip()
    if not s:
        return ""
    filler_starts = (
        "let me", "i'll", "i will", "i need to", "calling", "first,",
        "to answer", "to help", "let's", "i'm going to",
    )
    low = s.lower()
    if len(s) < 80 and any(low.startswith(p) for p in filler_starts):
        return ""
    return s


def run_chat(
    *,
    api_url: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = DEFAULT_TEMP,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    enable_thinking: bool | None = None,
    tools_enabled: bool = True,
    max_turns: int = MAX_TOOL_TURNS,
    cancel: threading.Event | None = None,
    adapter: str | None = None,
    response_format: dict[str, Any] | None = None,
) -> Iterator[bytes]:
    """Yield SSE byte frames for one chat orchestration.

    ``cancel``: optional threading.Event. Set externally (via the
    ``/api/chat/cancel`` endpoint, which calls ``cancel_session``) to stop
    the loop. Polled between turns; also forwarded into ``execute_tool``
    so a long-running tool call can be killed mid-flight.
    """

    convo: list[dict[str, Any]] = list(messages)
    names = tool_names()
    # Track only successful calls for dedup: a failed call is allowed to be
    # retried with the same args (the model might be iterating on a fix).
    last_successful_sig: tuple[str, str] | None = None
    # Self-healing: per-signature failure counter. A call that keeps failing
    # with identical args is allowed MAX_RETRIES_PER_CALL attempts, then the
    # loop refuses to re-run it (saves the sandbox + breaks retry loops).
    failed_sig_counts: dict[tuple[str, str], int] = {}

    def _is_cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    # Text the browser has been shown and kept, across all turns so far. A turn
    # whose content is rewritten by healing (a tool call pulled out of the prose)
    # replays `shown + corrected` so the stripped blob does not linger on screen.
    shown = ""

    base_body: dict[str, Any] = {
        "messages": convo,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if model:
        base_body["model"] = model
    if enable_thinking is not None:
        base_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if adapter:
        # optiq serve's mounted-LoRA path picks an adapter by name (the
        # adapter directory's basename). Setting this flips the active
        # adapter ContextVar for the duration of each turn.
        base_body["adapters"] = adapter
    # Structured / JSON-constrained output. lm-format-enforcer constrains the
    # decode to the schema, which is incompatible with free tool-calling, so
    # JSON mode takes precedence over tools.
    if response_format:
        base_body["response_format"] = response_format
    elif tools_enabled:
        base_body["tools"] = ALL_TOOLS

    for turn in range(1, max_turns + 1):
        if _is_cancelled():
            yield _sse("cancelled", {"turn": turn})
            yield _sse("done", {})
            return

        body = dict(base_body)
        body["messages"] = convo

        resp, streamed = None, ""
        try:
            for ev in _stream_mlx_lm(api_url, body, DEFAULT_TIMEOUT, api_key):
                if _is_cancelled():
                    yield _sse("cancelled", {"turn": turn})
                    yield _sse("done", {})
                    return
                if ev["type"] == "delta":
                    streamed += ev["text"]
                    yield _sse("token", {"text": ev["text"]})
                elif ev["type"] == "reasoning":
                    # No visible text, but it is a token: lets the client's
                    # decode-speed readout tick while the model is thinking.
                    yield _sse("token", {"text": "", "reasoning": ev["text"]})
                else:
                    resp = ev["resp"]
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")[:500]
            yield _sse("error", {"message": f"HTTP {e.code}: {err}"})
            yield _sse("done", {})
            return
        except (urllib.error.URLError, TimeoutError) as e:
            yield _sse("error", {"message": f"API unreachable: {e}"})
            yield _sse("done", {})
            return

        if resp is None:  # stream died before [DONE]; one non-streaming retry
            try:
                resp = _call_mlx_lm(api_url, body, DEFAULT_TIMEOUT, api_key)
                streamed = ""
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                yield _sse("error", {"message": f"API unreachable: {e}"})
                yield _sse("done", {})
                return

        choice = (resp.get("choices") or [{}])[0]
        message = choice.get("message") or {}

        healed, n_recovered = heal_tool_calls(message, names) \
            if tools_enabled else (message, 0)

        tool_calls = healed.get("tool_calls") or []

        # Even on a healed message we want the user to see whatever
        # plain content survived (e.g. a brief "let me check…" preamble).
        content_for_user = _strip_tool_artifacts(healed.get("content"))
        reasoning = healed.get("reasoning") or ""

        # Empty content + empty tool_calls is the truncated-tool-call case:
        # mlx-lm stripped the broken <tool_call> block and gave us nothing.
        # Surface a useful message rather than a silent dead-end.
        finish_reason = (choice.get("finish_reason") or "").lower()
        is_truncated_tool_call = (
            tools_enabled
            and not tool_calls
            and not (healed.get("content") or "").strip()
            and (finish_reason == "length" or finish_reason == "")
        )
        if is_truncated_tool_call:
            msg = (
                "(model started a tool call but ran out of tokens before the "
                "JSON closed. Raise `max_tokens` in Model & params, or "
                "rephrase the question to be more specific.)"
            )
            yield _sse("replace", {"text": shown + msg})
            yield _sse("assistant", {
                "content": msg, "reasoning": reasoning, "tool_calls": [],
            })
            yield _sse("done", {})
            return

        # The browser has already seen `streamed`. Healing may have rewritten it
        # (pulling a tool call out of the prose), so replay the corrected text.
        if content_for_user != streamed:
            yield _sse("replace", {"text": shown + content_for_user})
        shown += content_for_user

        if not tool_calls:
            yield _sse("assistant", {
                "content": healed.get("content") or "",
                "reasoning": reasoning,
                "tool_calls": [],
            })
            yield _sse("done", {})
            return

        convo.append({
            "role": "assistant",
            "content": healed.get("content"),
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            arg_str = fn.get("arguments") or "{}"
            try:
                args_disp = json.loads(arg_str)
            except json.JSONDecodeError:
                args_disp = arg_str

            sig = _call_signature(tc)
            # Only consider it a duplicate if the LAST SUCCESSFUL call had
            # the same signature. Lets a failed call be retried with the
            # same arguments (the model may be iterating on a fix).
            duplicate = (sig == last_successful_sig)
            # Self-healing: this exact call already failed its retry budget.
            retry_exhausted = (
                failed_sig_counts.get(sig, 0) >= MAX_RETRIES_PER_CALL
            )

            yield _sse("tool_call", {
                "id": tc.get("id") or "",
                "name": name,
                "arguments": args_disp,
                "turn": turn,
                "healed": bool(n_recovered),
                "duplicate": duplicate,
                "retry_exhausted": retry_exhausted,
            })

            images: list[dict[str, str]] = []
            if duplicate:
                # Don't burn the sandbox on a re-run; the previous result is
                # already in convo. Hand back a nudge.
                result = DUPLICATE_NUDGE
                elapsed = 0.0
                errored = False
            elif retry_exhausted:
                # Self-healing budget spent: refuse to re-run a call that keeps
                # failing identically. Push the model to change course instead.
                result = REPEATED_FAILURE_NUDGE.format(n=failed_sig_counts[sig])
                elapsed = 0.0
                errored = True
            else:
                t0 = time.time()
                raw_result = execute_tool(name, arg_str, cancel=cancel)
                elapsed = time.time() - t0
                # Strip image-data sentinel before either the model or the
                # UI sees a 100 KB blob of base64 in the result body.
                result, images = _extract_images(raw_result)
                errored = _is_tool_error(result)
                if errored:
                    failed_sig_counts[sig] = failed_sig_counts.get(sig, 0) + 1
                else:
                    last_successful_sig = sig
                    failed_sig_counts.pop(sig, None)  # recovered; reset budget

            # If the cancel event fired during the tool call, surface that
            # and bail rather than handing the broken result back to the
            # model for another turn.
            if _is_cancelled():
                yield _sse("tool_result", {
                    "id": tc.get("id") or "", "name": name, "result": result,
                    "elapsed_sec": round(elapsed, 3), "turn": turn,
                    "errored": True,
                })
                yield _sse("cancelled", {"turn": turn})
                yield _sse("done", {})
                return

            # If the tool errored, append the recovery nudge so the model
            # gets an explicit signal to try a different approach. The
            # retry-exhausted result already carries its own (stronger) nudge,
            # so don't double up.
            convo_result = result
            if errored and not retry_exhausted:
                attempt = failed_sig_counts.get(sig, 0)
                convo_result = (
                    result + TOOL_ERROR_NUDGE
                    + (f" (attempt {attempt} of {MAX_RETRIES_PER_CALL})"
                       if attempt else "")
                )

            yield _sse("tool_result", {
                "id": tc.get("id") or "",
                "name": name,
                "result": result,
                "elapsed_sec": round(elapsed, 3),
                "turn": turn,
                "errored": errored,
                "images": images,
            })

            convo.append({
                "role": "tool",
                "tool_call_id": tc.get("id") or "",
                "name": name,
                "content": convo_result,
            })

    # Budget exhausted. Instead of erroring, force one final tools-disabled
    # turn so the model has to commit to a text answer using what it has.
    yield _sse("token", {
        "text": "(budget exhausted, asking model to finish without tools)",
        "reasoning": "",
    })
    convo.append({"role": "user", "content": BUDGET_EXHAUSTED_NUDGE})
    final_body = dict(base_body)
    final_body["messages"] = convo
    final_body.pop("tools", None)
    # Stream it: this is the turn the user is actually waiting on.
    try:
        resp = None
        for ev in _stream_mlx_lm(api_url, final_body, DEFAULT_TIMEOUT, api_key):
            if ev["type"] == "delta":
                yield _sse("token", {"text": ev["text"]})
            elif ev["type"] == "reasoning":
                yield _sse("token", {"text": "", "reasoning": ev["text"]})
            else:
                resp = ev["resp"]
        message = (resp.get("choices") or [{}])[0].get("message") or {} if resp else {}
        yield _sse("assistant", {
            "content": message.get("content") or "",
            "reasoning": message.get("reasoning") or "",
            "tool_calls": [],
        })
    except Exception as e:
        yield _sse("error", {"message": f"final reprompt failed: {e}"})
    yield _sse("done", {})
