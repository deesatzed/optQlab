"""Distributed OpenAI-compatible serving across an OptiQ cluster.

Rank 0 runs a thin HTTP layer that drives mlx_lm's own ``stream_generate``
through the ring (``DistributedModel``), so sampling, structured output
(logits processors), streaming and stop tokens all come from the SAME code path
a single Mac uses. Generation runs on rank 0's MAIN thread (MLX distributed ops
must stay on the init thread); the HTTP handler threads hand work to it via a
queue and drain tokens. The other ranks run ``run_worker``.

Why a thin layer and not ``mlx_lm.server`` itself: under ``mlx.launch`` the
server sees ``group.size() > 1`` and switches to its OWN pipeline-parallel mode,
which fights OptiQ's RAM-weighted ring. So the cluster can't just host the
server. To keep responses byte-identical anyway, this layer reuses the server's
exact response shaping — mlx_lm's ``<think>`` reasoning split (via the tokenizer
markers) and its native ``tokenizer.tool_parser`` + ``ToolCallFormatter`` for
tool_calls, with OptiQ's ``heal_tool_calls`` as the malformed-shape fallback —
so ``reasoning`` / ``tool_calls`` / structured output match single-Mac serve.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .pipeline import DistributedModel, _ShardContext, _bcast, run_worker


def _flatten_content(message):
    """One message's content as plain text, tolerating None and content parts."""
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _encode_prompt(tokenizer, body, structured=False):
    """messages (+ tools + chat_template_kwargs) or a raw prompt -> token ids."""
    msgs = body.get("messages")
    if msgs:
        kwargs = {}
        if body.get("tools"):
            kwargs["tools"] = body["tools"]
        if isinstance(body.get("chat_template_kwargs"), dict):
            kwargs.update(body["chat_template_kwargs"])
        # Model-id variants (match the server's install_variants): a trailing
        # :think / :no-think on the model id toggles thinking without the
        # non-standard chat_template_kwargs field.
        model_id = body.get("model") or ""
        if model_id.endswith(":no-think"):
            kwargs.setdefault("enable_thinking", False)
        elif model_id.endswith(":think"):
            kwargs.setdefault("enable_thinking", True)
        # A structured spec forbids free text, so a reasoning model can't think
        # here; without this its template opens a <think> block and the
        # constrained output is mislabeled. Match the single-Mac serve: default
        # thinking off (an explicit client value still wins).
        if structured:
            kwargs.setdefault("enable_thinking", False)
        # Tool-call arguments come off the wire as a JSON string; chat templates
        # want a mapping, and Gemma-4's canonical template raises on anything
        # else. Single-Mac serve gets this from mlx-lm; the cluster layer has to
        # do it itself.
        from optiq.runtime.tool_args import normalize_tool_arguments
        normalize_tool_arguments(msgs)
        try:
            text = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, **kwargs)
            return tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            # Last resort: the template failed, so rebuild a plain transcript.
            # Content can be None or a list of parts, so flatten defensively —
            # the old str-join raised TypeError here and turned a degraded
            # prompt into a 500.
            text = "\n".join(_flatten_content(m) for m in msgs)
            return tokenizer.encode(text)
    return tokenizer.encode(body.get("prompt", ""))


def _parse_spec(body):
    try:
        from optiq.runtime.structured import parse_spec
        return parse_spec(body)
    except Exception:
        return None


def _gen_kwargs(body, tokenizer, spec):
    """Sampler + structured-output logits processors from the request, reusing
    mlx_lm's sampler and OptiQ's lm-format-enforcer processor."""
    from mlx_lm.sample_utils import make_sampler
    temp = float(body.get("temperature", 0.0) or 0.0)
    tp = float(body.get("top_p", 1.0) or 1.0)
    top_k = int(body.get("top_k", 0) or 0)
    sampler = make_sampler(temp=temp, top_p=(tp if 0.0 < tp < 1.0 else 0.0),
                           top_k=top_k)
    procs = []
    if spec:
        try:
            from optiq.runtime.structured import make_processor
            procs.append(make_processor(tokenizer, spec))
        except Exception:
            pass
    return {"max_tokens": int(body.get("max_tokens", 256)),
            "sampler": sampler, "logits_processors": procs}


# --- response shaping: match the single-Mac mlx_lm.server output exactly -------
# The cluster drives generation via stream_generate (ring-safe), so it must apply
# the same reasoning split (mlx_lm's <think> markers) and tool-call healing
# (OptiQ's heal_tool_calls) the server applies, so `reasoning` / `tool_calls`
# fields are identical whether a request hits one Mac or the cluster.

def _starts_in_reasoning(prompt_ids, tokenizer):
    """True if the rendered prompt left a <think> block open (thinking enabled),
    so generation begins as reasoning — mlx_lm's own rule."""
    if not getattr(tokenizer, "has_thinking", False):
        return False
    try:
        ts = tokenizer.rfind_think_start(prompt_ids)
        te = tokenizer.rfind_think_end(prompt_ids)
        return ts >= 0 and ts > te
    except Exception:
        return False


def _tool_names(body):
    tools = body.get("tools")
    if not isinstance(tools, list):
        return []
    return [t["function"]["name"] for t in tools
            if isinstance(t, dict) and isinstance(t.get("function"), dict)
            and t["function"].get("name")]


def _native_tool_calls(content, tokenizer, tools):
    """Extract OpenAI tool_calls from ``content`` using mlx_lm's OWN tokenizer
    tool_parser + ToolCallFormatter (the exact path the single-Mac server uses),
    and strip the tool-call spans from the content. Returns (calls, content)."""
    if not getattr(tokenizer, "has_tool_calling", False):
        return None, content
    start = getattr(tokenizer, "tool_call_start", None)
    end = getattr(tokenizer, "tool_call_end", None)
    if not start or start not in content:
        return None, content
    segments, spans, i = [], [], 0
    while True:
        s = content.find(start, i)
        if s < 0:
            break
        e = content.find(end, s + len(start)) if end else -1
        if e < 0:
            segments.append(content[s + len(start):])
            spans.append((s, len(content)))
            break
        segments.append(content[s + len(start):e])
        spans.append((s, e + len(end)))
        i = e + len(end)
    if not segments:
        return None, content
    try:
        from mlx_lm.server import ToolCallFormatter
        calls = ToolCallFormatter(tokenizer.tool_parser, tools, streaming=False)(segments)
    except Exception:
        return None, content
    if not calls:
        return None, content
    kept, last = [], 0
    for a, b in spans:
        kept.append(content[last:a])
        last = b
    kept.append(content[last:])
    return calls, "".join(kept).strip("\n")


def _heal(msg, body, tokenizer=None):
    """Populate tool_calls the way the single-Mac server does: mlx_lm's native
    tokenizer parser first, then OptiQ's heal_tool_calls for malformed shapes."""
    tools = body.get("tools")
    if not tools:
        return msg
    if tokenizer is not None:
        calls, content = _native_tool_calls(msg.get("content") or "", tokenizer, tools)
        if calls:
            msg = dict(msg, tool_calls=calls, content=content or None)
            return msg
    names = _tool_names(body)
    if names:
        try:
            from optiq.lab.tools.healing import heal_tool_calls
            healed, _ = heal_tool_calls(msg, names)
            return healed
        except Exception:
            pass
    return msg


def _finalize(text, tokenizer, body, in_reasoning):
    """Split reasoning from content and heal tool calls -> an assistant message
    with the same shape the single-Mac server returns."""
    reasoning, content = None, text
    if in_reasoning:
        end = getattr(tokenizer, "think_end", None)
        if end and end in text:
            i = text.index(end)
            reasoning, content = text[:i], text[i + len(end):]
        else:
            reasoning, content = text, ""
    msg = {"role": "assistant", "content": content.strip("\n")}
    if reasoning and reasoning.strip("\n"):
        msg["reasoning"] = reasoning.strip("\n")
    return _heal(msg, body, tokenizer)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, model, tokenizer, info):
        super().__init__(addr, _Handler)
        self.model = model
        self.tokenizer = tokenizer
        self.info = info
        self.model_id = info.get("model", "optiq-cluster")
        self._lock = threading.Lock()          # one generation at a time (lockstep)
        self._req = threading.Event()
        self._pending = None
        self._finish_reason = "stop"           # set by the generate loop per request

    def run(self, prompt_ids, gen_kwargs):
        """Hand a request to rank 0's main generate loop; yield text pieces."""
        with self._lock:
            q = queue.Queue()
            self._pending = (prompt_ids, gen_kwargs, q)
            self._req.set()
            while True:
                item = q.get()
                if item is None:
                    break
                yield item


class _Handler(BaseHTTPRequestHandler):
    server_version = "OptiQCluster/0.3"

    def log_message(self, *a):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.rstrip("/")
        if p == "/cluster/info":
            self._json(200, self.server.info)
        elif p in ("/v1/models", "/models"):
            self._json(200, {"object": "list",
                             "data": [{"id": self.server.model_id, "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._json(400, {"error": "invalid json"})
            return
        spec = _parse_spec(body)
        tok = self.server.tokenizer
        prompt_ids = _encode_prompt(tok, body, structured=bool(spec))
        gk = _gen_kwargs(body, tok, spec)
        in_reasoning = _starts_in_reasoning(prompt_ids, tok)
        t0 = time.time()
        if body.get("stream"):
            self._stream(prompt_ids, gk, t0, body, in_reasoning)
            return
        text = "".join(self.server.run(prompt_ids, gk))
        msg = _finalize(text, tok, body, in_reasoning)
        finish = "tool_calls" if msg.get("tool_calls") else self.server._finish_reason
        self._json(200, {
            "id": f"optiq-cluster-{int(t0)}", "object": "chat.completion",
            "model": self.server.model_id,
            "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
            "usage": {"generation_seconds": round(time.time() - t0, 2)},
        })

    def _stream(self, prompt_ids, gk, t0, body, in_reasoning):
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # An SSE body carries no Content-Length, and BaseHTTPRequestHandler
        # speaks HTTP/1.0, so it cannot frame the body with chunked encoding
        # either. The end of the response *is* the close of the socket. Answer
        # the browser's `Connection: keep-alive` with `close`, or fetch()'s
        # reader never reports done and the caller hangs after [DONE].
        # curl hides this: it reads until EOF regardless.
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        cid, model = f"optiq-cluster-{int(t0)}", self.server.model_id
        tok = self.server.tokenizer
        end = getattr(tok, "think_end", None) or "</think>"

        def sse(delta, finish=None):
            chunk = {"id": cid, "object": "chat.completion.chunk",
                     "created": int(t0), "model": model,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.flush()

        try:
            sse({"role": "assistant"})
            buf, content = "", ""          # buf: reasoning tail (detect </think>)
            for piece in self.server.run(prompt_ids, gk):
                if not piece:
                    continue
                if in_reasoning:
                    buf += piece
                    if end in buf:         # think block closed mid-stream
                        before, after = buf.split(end, 1)
                        if before:
                            sse({"reasoning": before})
                        in_reasoning, buf = False, ""
                        if after:
                            content += after
                            sse({"content": after})
                    elif len(buf) > len(end):
                        # emit reasoning but hold back a possible partial </think>
                        safe, buf = buf[:-len(end)], buf[-len(end):]
                        sse({"reasoning": safe})
                else:
                    content += piece
                    sse({"content": piece})
            if buf:                        # flush leftover reasoning tail
                sse({"reasoning": buf})
            healed = _heal({"role": "assistant", "content": content}, body, tok)
            tc = healed.get("tool_calls")
            if tc:
                sse({"tool_calls": tc}, finish="tool_calls")
            else:
                sse({}, finish=self.server._finish_reason)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client hung up; rank 0 still drains the generation


def _serve_rank0(ctx, host, port, info):
    from mlx_lm import stream_generate

    model = DistributedModel(ctx)
    server = _Server((host, port), model, ctx.tokenizer, info)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[cluster] OpenAI endpoint: http://{host}:{port}/v1/chat/completions",
          flush=True)
    print(f"[cluster] {ctx.size} nodes, model={info.get('model')}. Ctrl+C to stop.",
          flush=True)
    try:
        while True:
            server._req.wait()
            server._req.clear()
            prompt_ids, gk, q = server._pending
            reason = "stop"
            try:
                for resp in stream_generate(model, ctx.tokenizer, prompt_ids, **gk):
                    q.put(resp.text)
                    if resp.finish_reason:
                        reason = resp.finish_reason
            except Exception as e:  # surface the error to the client, keep serving
                q.put(f"[cluster error: {e}]")
            finally:
                server._finish_reason = reason
                q.put(None)
    except KeyboardInterrupt:
        _bcast(ctx, 0)  # tell the workers to exit
        print("\n[cluster] stopped.", flush=True)


def cluster_serve(model_path: str, host: str = "127.0.0.1", port: int = 8100,
                  cluster_info: dict | None = None, verbose: bool = True):
    """Entry point run on every rank (via mlx.launch ring backend)."""
    ctx = _ShardContext(model_path, verbose=verbose)
    if not ctx.is_first:
        run_worker(ctx)
        return
    info = {**(cluster_info or {}), "model": model_path,
            "nodes": ctx.size, "world_size": ctx.size}
    _serve_rank0(ctx, host, port, info)


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8100
    cluster_serve(model, port=port)
