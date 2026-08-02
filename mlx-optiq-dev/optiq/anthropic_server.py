"""Install an Anthropic ``/v1/messages`` endpoint on mlx-lm's APIHandler.

Usage (from optiq.serve or optiq.cli):

    from optiq.anthropic_server import install_anthropic_endpoint
    install_anthropic_endpoint()
    # then start mlx_lm.server.main() as usual

The install is idempotent. It works by monkey-patching
``mlx_lm.server.APIHandler.do_POST`` to recognize the new path; every
other route behaves identically to upstream.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .anthropic_shim import (
    anthropic_to_openai_body,
    openai_to_anthropic_response,
    AnthropicStreamTranslator,
    _sse,
)


_INSTALLED = False


def install_anthropic_endpoint() -> None:
    """Monkey-patch mlx_lm.server.APIHandler to serve /v1/messages."""
    global _INSTALLED
    if _INSTALLED:
        return

    import mlx_lm.server as server_mod

    original_do_POST = server_mod.APIHandler.do_POST

    def patched_do_POST(self):
        # Strip query string so /v1/messages?beta=true matches too
        # (Claude Code sends ?beta=true for the prompt-caching beta;
        # other clients may pass other query params we should ignore).
        path = self.path.split("?", 1)[0]
        if path == "/v1/messages":
            return _handle_anthropic_messages(self)
        return original_do_POST(self)

    server_mod.APIHandler.do_POST = patched_do_POST
    _INSTALLED = True
    logging.info("[optiq] Anthropic /v1/messages endpoint installed")


# --------------------------------------------------------------------------
# /v1/messages handler
# --------------------------------------------------------------------------


def _read_body(handler) -> dict[str, Any]:
    """Read JSON body or raise a handler-returning ValueError with status."""
    length = handler.headers.get("Content-Length")
    if length is None:
        _write_error(handler, 411, "Content-Length required")
        raise _HandledError()
    try:
        length = int(length)
    except ValueError:
        _write_error(handler, 400, "Invalid Content-Length")
        raise _HandledError()
    raw = handler.rfile.read(length)
    try:
        body = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        _write_error(handler, 400, f"Invalid JSON: {e}")
        raise _HandledError()
    if not isinstance(body, dict):
        _write_error(handler, 400, "Request body must be a JSON object")
        raise _HandledError()
    return body


def _write_error(handler, status: int, message: str) -> None:
    body = json.dumps({
        "type": "error",
        "error": {"type": "api_error", "message": message},
    }).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


class _HandledError(Exception):
    pass


def _handle_anthropic_messages(handler):
    """Handle a POST /v1/messages request.

    Strategy: rewrite the request body in-place to the OpenAI /v1/chat/completions
    shape, then reuse the stock mlx-lm handler machinery to generate. For
    streaming, intercept the SSE output and translate to Anthropic events.
    """
    try:
        anthropic_body = _read_body(handler)
    except _HandledError:
        return

    # Hold onto the requested model so the response reflects it even after
    # we mutate the body.
    model_name = anthropic_body.get("model", "default_model")
    wants_stream = bool(anthropic_body.get("stream"))

    # Transform request body → OpenAI shape, assign to handler.body so
    # mlx-lm's internal handlers use it.
    oai_body = anthropic_to_openai_body(anthropic_body)
    handler.body = oai_body

    # Mimic mlx-lm's do_POST field extraction (see server.py do_POST).
    handler.stream = wants_stream
    handler.stream_options = oai_body.get("stream_options", None)
    handler.requested_model = oai_body.get("model", "default_model")
    handler.requested_draft_model = oai_body.get("draft_model", "default_model")
    handler.num_draft_tokens = oai_body.get(
        "num_draft_tokens", handler.response_generator.cli_args.num_draft_tokens
    )
    handler.adapter = oai_body.get("adapters")
    handler.max_tokens = oai_body.get("max_completion_tokens") or oai_body.get(
        "max_tokens", handler.response_generator.cli_args.max_tokens
    )
    handler.temperature = oai_body.get(
        "temperature", handler.response_generator.cli_args.temp
    )
    handler.top_p = oai_body.get("top_p", handler.response_generator.cli_args.top_p)
    handler.top_k = oai_body.get("top_k", handler.response_generator.cli_args.top_k)
    handler.min_p = oai_body.get("min_p", handler.response_generator.cli_args.min_p)
    handler.repetition_penalty = oai_body.get("repetition_penalty", 0.0)
    handler.repetition_context_size = oai_body.get("repetition_context_size", 20)
    handler.presence_penalty = oai_body.get("presence_penalty", 0.0)
    handler.presence_context_size = oai_body.get("presence_context_size", 20)
    handler.frequency_penalty = oai_body.get("frequency_penalty", 0.0)
    handler.frequency_context_size = oai_body.get("frequency_context_size", 20)
    handler.xtc_probability = oai_body.get("xtc_probability", 0.0)
    handler.xtc_threshold = oai_body.get("xtc_threshold", 0.0)
    handler.logit_bias = oai_body.get("logit_bias")
    handler.logprobs = oai_body.get("logprobs", False)
    handler.top_logprobs = oai_body.get("top_logprobs", -1)
    handler.seed = oai_body.get("seed")
    handler.chat_template_kwargs = oai_body.get("chat_template_kwargs")
    try:
        handler.validate_model_parameters()
    except Exception as e:
        _write_error(handler, 400, str(e))
        return

    stop_words = oai_body.get("stop") or []
    if isinstance(stop_words, str):
        stop_words = [stop_words]

    # Dispatch. Tool-using requests are buffered even when streaming: mlx-lm
    # only parses <tool_call> tags into structured tool_calls on the
    # NON-streaming path, so a streamed tool call would otherwise leak through
    # as raw text and the client (e.g. Claude Code) would never see a tool_use
    # block. So: tools + stream → generate buffered, then replay as SSE.
    if not wants_stream:
        _handle_nonstream(handler, stop_words, model_name)
    elif oai_body.get("tools"):
        _handle_stream_buffered(handler, stop_words, model_name)
    else:
        _handle_stream(handler, stop_words, model_name)


def _generate_oai_response(handler, stop_words):
    """Run mlx-lm generation NON-streaming, capturing the OpenAI JSON it would
    have written. Returns (status_code, parsed_json | None); writes nothing to
    the real client. Shared by the non-stream and buffered-stream paths."""
    import io
    buf = io.BytesIO()
    orig = (handler.wfile, handler.send_response, handler.send_header,
            handler.end_headers, getattr(handler, "stream", False))
    status = [200]
    handler.wfile = buf
    handler.send_response = lambda code, message=None: status.__setitem__(0, code)
    handler.send_header = lambda *a, **k: None
    handler.end_headers = lambda: None
    handler.stream = False  # force buffered generation regardless of client ask
    if isinstance(getattr(handler, "body", None), dict):
        handler.body["stream"] = False
    try:
        request = handler.handle_chat_completions()
        handler.handle_completion(request, stop_words)
    finally:
        (handler.wfile, handler.send_response, handler.send_header,
         handler.end_headers, handler.stream) = orig
    try:
        return status[0], json.loads(buf.getvalue().decode("utf-8"))
    except Exception:
        return 500, None


def _handle_nonstream(handler, stop_words, model_name: str) -> None:
    """Non-streaming: generate, translate to Anthropic, write JSON."""
    status, oai_resp = _generate_oai_response(handler, stop_words)
    if oai_resp is None:
        _write_error(handler, 500, "upstream returned non-JSON response")
        return
    if status != 200:
        msg = oai_resp.get("error", {}) if isinstance(oai_resp, dict) else {}
        _write_error(handler, status, msg.get("message", "generation failed"))
        return
    anth = openai_to_anthropic_response(oai_resp, model_name)
    body = json.dumps(anth).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _message_to_sse(anth: dict):
    """Replay a complete Anthropic message dict as the SSE event sequence a
    streaming client expects. Used for buffered (tool) streaming so tool_use
    blocks reach the client intact (mlx-lm only parses tool_calls when
    generating non-streaming)."""
    usage = anth.get("usage", {}) or {}
    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": anth.get("id"), "type": "message", "role": "assistant",
            "model": anth.get("model"), "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": int(usage.get("input_tokens", 0)), "output_tokens": 0},
        },
    })
    for i, block in enumerate(anth.get("content", []) or []):
        bt = block.get("type")
        if bt == "thinking":
            yield _sse("content_block_start", {"type": "content_block_start", "index": i,
                "content_block": {"type": "thinking", "thinking": ""}})
            yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                "delta": {"type": "thinking_delta", "thinking": block.get("thinking", "")}})
            yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                "delta": {"type": "signature_delta", "signature": block.get("signature", "")}})
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})
        elif bt == "text":
            yield _sse("content_block_start", {"type": "content_block_start", "index": i,
                "content_block": {"type": "text", "text": ""}})
            yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                "delta": {"type": "text_delta", "text": block.get("text", "")}})
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})
        elif bt == "tool_use":
            yield _sse("content_block_start", {"type": "content_block_start", "index": i,
                "content_block": {"type": "tool_use", "id": block.get("id"),
                                  "name": block.get("name"), "input": {}}})
            yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                "delta": {"type": "input_json_delta",
                          "partial_json": json.dumps(block.get("input", {}))}})
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": anth.get("stop_reason", "end_turn"), "stop_sequence": None},
        "usage": {"output_tokens": int(usage.get("output_tokens", 0))},
    })
    yield _sse("message_stop", {"type": "message_stop"})


def _handle_stream_buffered(handler, stop_words, model_name: str) -> None:
    """Tools + stream: generate buffered so <tool_call> tags parse into
    structured tool_calls, then replay the Anthropic message as SSE events."""
    status, oai_resp = _generate_oai_response(handler, stop_words)
    if oai_resp is None or status != 200:
        msg = (oai_resp.get("error", {}) if isinstance(oai_resp, dict) else {}) or {}
        _write_error(handler, status if oai_resp is not None else 500,
                     msg.get("message", "generation failed"))
        return
    anth = openai_to_anthropic_response(oai_resp, model_name)
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    for ev in _message_to_sse(anth):
        handler.wfile.write(ev)
    try:
        handler.wfile.flush()
    except Exception:
        pass


def _handle_stream(handler, stop_words, model_name: str) -> None:
    """Streaming: capture each SSE 'data: {...}' line from mlx-lm's output,
    parse the OpenAI chunk, translate to Anthropic event, emit."""
    import io

    class _StreamProxy(io.RawIOBase):
        """Incremental SSE proxy: each upstream OpenAI chunk passes through
        the translator immediately, so the client sees Anthropic events
        flowing while the model is still generating."""

        def __init__(self, real_wfile, model_name):
            self.real = real_wfile
            self.model = model_name
            self._buf = b""
            self._translator = AnthropicStreamTranslator(model_name)
            self._done = False

        def writable(self):
            return True

        def write(self, data):
            self._buf += data
            while b"\n\n" in self._buf:
                frame, self._buf = self._buf.split(b"\n\n", 1)
                self._handle_frame(frame)
            return len(data)

        def _emit(self, ev_bytes: bytes):
            self.real.write(ev_bytes)
            try:
                self.real.flush()
            except Exception:
                pass

        def _handle_frame(self, frame: bytes):
            for line in frame.splitlines():
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    self._finalize()
                    return
                try:
                    chunk = json.loads(payload.decode())
                except json.JSONDecodeError:
                    continue
                for ev in self._translator.add_chunk(chunk):
                    self._emit(ev)

        def _finalize(self):
            if self._done:
                return
            self._done = True
            for ev in self._translator.finalize():
                self._emit(ev)

        def flush(self):
            try:
                self.real.flush()
            except Exception:
                pass

        def close(self):
            self._finalize()

    # Set up Anthropic-appropriate SSE response headers.
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()

    # Redirect mlx-lm writes through the proxy.
    proxy = _StreamProxy(handler.wfile, model_name)
    orig_wfile = handler.wfile
    orig_send_response = handler.send_response
    orig_send_header = handler.send_header
    orig_end_headers = handler.end_headers
    handler.wfile = proxy
    handler.send_response = lambda *a, **k: None
    handler.send_header = lambda *a, **k: None
    handler.end_headers = lambda *a, **k: None
    try:
        request = handler.handle_chat_completions()
        handler.handle_completion(request, stop_words)
    finally:
        handler.wfile = orig_wfile
        handler.send_response = orig_send_response
        handler.send_header = orig_send_header
        handler.end_headers = orig_end_headers
