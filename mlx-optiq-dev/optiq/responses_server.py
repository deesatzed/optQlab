"""Install an OpenAI ``/v1/responses`` endpoint on mlx-lm's APIHandler.

Usage (from optiq.serve or optiq.cli):

    from optiq.responses_server import install_responses_endpoint
    install_responses_endpoint()
    # then start mlx_lm.server.main() as usual

The install is idempotent. Works by monkey-patching
``mlx_lm.server.APIHandler.do_POST`` to recognise ``/v1/responses``; every
other route behaves identically to upstream.

Translation strategy mirrors ``anthropic_server.py``: rewrite the
Responses request body to the OpenAI Chat Completions shape, reuse
mlx-lm's existing handler machinery to generate, capture the response,
and translate back to Responses output format. Streaming uses an SSE
proxy that re-emits the OpenAI deltas as Responses-flavored events
(``response.created`` / ``response.output_text.delta`` /
``response.completed``).

When ``install_mtp_speculation`` is also installed, MTP-aware generation
applies automatically to Responses requests (the MTP patch hooks
``mlx_lm.server.stream_generate``, which both endpoints route through).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .response_store import get_store
from .responses_shim import (
    responses_to_openai_body,
    openai_to_responses_response,
    output_items_to_input_items,
    ResponsesStreamTranslator,
)


_INSTALLED = False


def install_responses_endpoint() -> None:
    """Monkey-patch mlx_lm.server.APIHandler to serve /v1/responses."""
    global _INSTALLED
    if _INSTALLED:
        return

    import mlx_lm.server as server_mod

    original_do_POST = server_mod.APIHandler.do_POST

    def patched_do_POST(self):
        # Strip query string so /v1/responses?foo=bar matches too.
        path = self.path.split("?", 1)[0]
        if path == "/v1/responses":
            return _handle_responses(self)
        return original_do_POST(self)

    server_mod.APIHandler.do_POST = patched_do_POST
    _INSTALLED = True
    logging.info("[optiq] OpenAI /v1/responses endpoint installed")


# ----------------------------------------------------------------------
# /v1/responses handler
# ----------------------------------------------------------------------


def _read_body(handler) -> dict[str, Any]:
    """Read JSON body or raise a handler-returning ``_HandledError``."""
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


def _handle_responses(handler):
    """Handle a POST /v1/responses request.

    Strategy: translate body to OpenAI Chat Completions shape, reuse
    mlx-lm's internal handlers to generate, capture the response, then
    re-emit as Responses output. Streaming proxies SSE writes and
    translates each OpenAI delta into a Responses event chain.
    """
    try:
        responses_body = _read_body(handler)
    except _HandledError:
        return

    model_name = responses_body.get("model", "default_model")
    wants_stream = bool(responses_body.get("stream"))

    # Resolve previous_response_id by prepending the prior conversation to
    # this request's input.
    prev_id = responses_body.get("previous_response_id")
    prepended_input_items: list = []
    if prev_id:
        store = get_store()
        prior = store.get(str(prev_id))
        if prior is None:
            _write_error(handler, 404, f"previous_response_id {prev_id!r} not found or expired")
            return
        prepended_input_items = list(prior["input"]) + output_items_to_input_items(prior["output"])
        # Carry instructions forward only if the new request omits them.
        if not responses_body.get("instructions") and prior.get("instructions"):
            responses_body = dict(responses_body)
            responses_body["instructions"] = prior["instructions"]

    if prepended_input_items:
        # Splice prior history in before the user's new input items.
        new_input = responses_body.get("input")
        if isinstance(new_input, str):
            new_input_items = [{"type": "message", "role": "user", "content": new_input}]
        elif isinstance(new_input, list):
            new_input_items = list(new_input)
        else:
            new_input_items = []
        responses_body = dict(responses_body)
        responses_body["input"] = prepended_input_items + new_input_items

    # Remember the input we (effectively) sent to the model so a later
    # follow-up that chains off this response sees the full history.
    effective_input = responses_body.get("input")
    if isinstance(effective_input, str):
        captured_input_items = [{"type": "message", "role": "user", "content": effective_input}]
    elif isinstance(effective_input, list):
        captured_input_items = list(effective_input)
    else:
        captured_input_items = []
    captured_instructions = responses_body.get("instructions")

    oai_body = responses_to_openai_body(responses_body)
    handler.body = oai_body

    # Mimic mlx-lm's do_POST field extraction (server.py do_POST).
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

    if not wants_stream:
        _handle_nonstream(handler, stop_words, model_name, captured_input_items, captured_instructions)
    else:
        _handle_stream(handler, stop_words, model_name, captured_input_items, captured_instructions)


def _handle_nonstream(handler, stop_words, model_name: str,
                      captured_input_items: list, captured_instructions) -> None:
    """Non-streaming: buffer mlx-lm's OpenAI JSON response, translate to
    Responses output format, write out."""
    import io

    class _Buffer(io.BytesIO):
        pass

    buf = _Buffer()
    orig_wfile = handler.wfile
    orig_send_response = handler.send_response
    orig_send_header = handler.send_header
    orig_end_headers = handler.end_headers

    captured_status = [200]

    def _capture_status(code, message=None):
        captured_status[0] = code

    def _noop(*a, **k):
        pass

    handler.wfile = buf
    handler.send_response = _capture_status
    handler.send_header = _noop
    handler.end_headers = _noop

    try:
        request = handler.handle_chat_completions()
        handler.handle_completion(request, stop_words)
    finally:
        handler.wfile = orig_wfile
        handler.send_response = orig_send_response
        handler.send_header = orig_send_header
        handler.end_headers = orig_end_headers

    try:
        oai_resp = json.loads(buf.getvalue().decode("utf-8"))
    except Exception:
        _write_error(handler, 500, "upstream returned non-JSON response")
        return
    if captured_status[0] != 200:
        msg = oai_resp.get("error", {}) if isinstance(oai_resp, dict) else {}
        _write_error(handler, captured_status[0],
                     msg.get("message", "generation failed"))
        return

    resp = openai_to_responses_response(oai_resp, model_name)
    get_store().put(
        resp["id"],
        input_items=captured_input_items,
        output_items=resp.get("output") or [],
        instructions=captured_instructions,
    )
    body = json.dumps(resp).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _handle_stream(handler, stop_words, model_name: str,
                   captured_input_items: list, captured_instructions) -> None:
    """Streaming: capture each OpenAI SSE chunk from mlx-lm's output,
    accumulate, then emit Responses-shaped events at flush."""
    import io

    class _StreamProxy(io.RawIOBase):
        """Incremental SSE proxy. Each upstream OpenAI chunk passes through
        the translator immediately, so the client sees bytes flowing while
        the model is still generating (vs. waiting for [DONE] to flush).

        Reasoning-model first-token latency can be tens of seconds — without
        per-chunk flushing the client times out before any byte arrives.
        """

        def __init__(self, real_wfile, model_name):
            self.real = real_wfile
            self.model = model_name
            self._buf = b""
            self._translator = ResponsesStreamTranslator(model_name)
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
            # Capture final response for previous_response_id resumption.
            try:
                final = self._translator.final_response()
                rid = final.get("id")
                if rid:
                    get_store().put(
                        rid,
                        input_items=captured_input_items,
                        output_items=final.get("output") or [],
                        instructions=captured_instructions,
                    )
            except Exception:
                pass

        def flush(self):
            try:
                self.real.flush()
            except Exception:
                pass

        def close(self):
            # If the upstream closes without [DONE] (rare; partial response),
            # still emit a response.completed so the client doesn't hang.
            self._finalize()

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()

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
