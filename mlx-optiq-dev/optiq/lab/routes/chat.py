"""Chat workflow: page + chat-history persistence + proxy to API.

The browser streams tokens directly from the local mlx-lm API for the
fast tool-less path. When tools are enabled, the browser instead hits
``/api/chat/stream``, which runs a server-side multi-turn loop in
``optiq.lab.chat_orchestrator``.
"""

from __future__ import annotations

import json

from flask import (
    Blueprint, Response, abort, current_app, jsonify, render_template,
    request, stream_with_context,
)

from .. import chat_store, local_quants
from ..chat_orchestrator import (
    cancel_session, cleanup_session, register_session, run_chat,
)
from ..file_extract import ExtractFailed, UnsupportedFile, extract as extract_file
from ..provenance_capture import apply_stream_provenance, build_partial_provenance
from ..sandbox import detect_sandbox_kind


bp = Blueprint("chat", __name__)


MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB cap per upload


@bp.route("/chat")
def chat_page():
    chats = chat_store.list_chat_records()
    paths = current_app.config["OPTIQ_LAB_PATHS"]
    loaded = current_app.config.get("OPTIQ_LOADED_MODEL") or ""
    # Build a deduped picker: model loaded at lab startup, models the lab
    # built locally, and mlx-community models in the HF cache. Each entry
    # is a {name, path, source} dict the UI uses to populate a datalist.
    picker: list[dict] = []
    seen: set[str] = set()

    def _add(name: str, path: str, source: str) -> None:
        key = path or name
        if not name or key in seen:
            return
        seen.add(key)
        picker.append({"name": name, "path": path or name, "source": source})

    if loaded:
        _add(loaded, loaded, "loaded")
    for q in local_quants.discover(paths.models_dir):
        _add(q.display_name, q.path, "local")
    for q in local_quants.discover_hf_cache():
        _add(q.display_name, q.path, "hf_cache")
    return render_template("chat.html", page_title="Chat", section="chat",
                           chats=chats, model_picker=picker, loaded_model=loaded)


@bp.route("/api/chats", methods=["GET"])
def list_chats():
    return jsonify({"chats": chat_store.list_chat_records()})


@bp.route("/api/chats", methods=["POST"])
def save_chat():
    """Persist a chat thread. Body: {id?, title, messages, model}."""
    data = request.get_json(force=True) or {}
    chat_id = chat_store.save_chat_record(data)
    return jsonify({"ok": True, "id": chat_id})


@bp.route("/api/chats/<chat_id>", methods=["GET"])
def load_chat(chat_id):
    if not _is_chat_id(chat_id):
        abort(404)
    record = chat_store.load_chat_record(chat_id)
    if record is None:
        abort(404)
    return jsonify(record)


@bp.route("/api/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    if not _is_chat_id(chat_id):
        abort(400)
    chat_store.delete_chat_record(chat_id)
    return jsonify({"ok": True})


def _resolve_chat_endpoint(data, default_url, default_key):
    """Pick the chat backend (url, key) for one request.

    A non-blank ``base_url`` in the body means the user configured a cloud /
    third-party OpenAI-compatible endpoint (OpenRouter, etc.) in the Chat UI: we
    proxy there with the key they supplied, falling back to the local key only if
    they left it blank. No ``base_url`` -> the local ``optiq serve`` defaults. The
    key is used per-request and never persisted server-side; it lives only in the
    user's browser localStorage.
    """
    base = (data.get("base_url") or "").strip()
    if base:
        return base, ((data.get("api_key") or "").strip() or default_key)
    return default_url, default_key


@bp.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """SSE endpoint for the tool-enabled chat path.

    Body shape::

      {
        "model": "...",
        "messages": [...],
        "temperature": 0.7,
        "max_tokens": 1024,
        "enable_thinking": false,
        "tools_enabled": true
      }
    """
    data = request.get_json(force=True) or {}
    api_url, api_key = _resolve_chat_endpoint(
        data,
        current_app.config.get("OPTIQ_API_URL", "http://127.0.0.1:8080"),
        current_app.config.get("OPTIQ_API_KEY", "sk-optiq-local"),
    )

    et = data.get("enable_thinking")
    enable_thinking = None if et is None else bool(et)

    # Register a cancel event for this stream; emit it back to the client
    # as the very first SSE event so the Stop button knows what to cancel.
    session_id, cancel_event = register_session(data.get("session_id"))

    # Chat-with-files RAG: if the request carries attached documents, retrieve
    # the chunks most relevant to the user's question and prepend them (with
    # [n] citation markers) to the final user turn, instead of dumping whole
    # documents into the context. Sources are streamed to the UI separately.
    messages = data.get("messages") or []
    documents = data.get("documents") or []
    rag_sources: list = []
    if documents and messages:
        from .. import rag as _rag
        last = messages[-1]
        q = last.get("content")
        if isinstance(q, list):
            q = " ".join(p.get("text", "") for p in q
                         if isinstance(p, dict) and p.get("type") == "text")
        ctx, rag_sources = _rag.retrieve(q or "", documents)
        if ctx and isinstance(last.get("content"), str):
            messages = list(messages)
            messages[-1] = {**last,
                            "content": ctx + "\n\nQuestion: " + last["content"]}

    def gen():
        tools_called: list = []
        assistant_content: str | None = None
        healed_any = False
        retry_hits = 0
        try:
            yield (
                f"event: session\ndata: {json.dumps({'session_id': session_id})}\n\n"
                .encode("utf-8")
            )
            if rag_sources:
                yield (
                    f"event: sources\ndata: {json.dumps({'sources': rag_sources})}\n\n"
                    .encode("utf-8")
                )
            for chunk in run_chat(
                api_url=api_url,
                api_key=api_key,
                model=str(data.get("model") or ""),
                messages=messages,
                temperature=float(data.get("temperature", 0.7)),
                max_tokens=int(data.get("max_tokens", 1024)),
                enable_thinking=enable_thinking,
                tools_enabled=bool(data.get("tools_enabled", True)),
                cancel=cancel_event,
                adapter=(data.get("adapter") or None) or None,
                response_format=(data.get("response_format") or None),
            ):
                yield chunk
                event, payload = _parse_sse_chunk(chunk)
                if event == "tool_call" and isinstance(payload, dict):
                    entry = {
                        "name": payload.get("name") or "",
                        "healed": bool(payload.get("healed")),
                    }
                    tools_called.append(entry)
                    if entry["healed"]:
                        healed_any = True
                    if payload.get("retry_exhausted"):
                        retry_hits += 1
                elif event == "assistant" and isinstance(payload, dict):
                    assistant_content = payload.get("content") or ""

            # Partial provenance on stream end when client supplied chat_id.
            chat_id = data.get("chat_id")
            if chat_id:
                final_msgs = list(data.get("messages") or [])
                if assistant_content is not None:
                    final_msgs = list(final_msgs) + [{
                        "role": "assistant",
                        "content": assistant_content,
                    }]
                sampler = {
                    "temperature": float(data.get("temperature", 0.7)),
                    "max_tokens": int(data.get("max_tokens", 1024)),
                }
                if enable_thinking is not None:
                    sampler["enable_thinking"] = enable_thinking
                # context_window only if client/server actually provided it
                ctx_win = data.get("context_window")
                try:
                    ctx_win = int(ctx_win) if ctx_win is not None else None
                except (TypeError, ValueError):
                    ctx_win = None
                # tools_enabled names / tok_per_sec / peak_mem_gb stay null —
                # never invent unmeasured fields.
                prov = build_partial_provenance(
                    model=str(data.get("model") or "") or None,
                    sampler=sampler,
                    tools_called=tools_called,
                    healed=healed_any if tools_called else None,
                    retry_hits=retry_hits if tools_called else None,
                    context_window=ctx_win,
                )
                try:
                    apply_stream_provenance(chat_id, final_msgs, prov)
                except Exception:  # noqa: BLE001 — never break the SSE stream
                    pass
        finally:
            cleanup_session(session_id)

    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _last_user_text(messages: list) -> str:
    """Pull the plain-text question out of the last user turn (handles the
    multimodal content-array shape the composer sends)."""
    for m in reversed(messages or []):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c.strip()
        if isinstance(c, list):
            return " ".join(p.get("text", "") for p in c
                            if isinstance(p, dict) and p.get("type") == "text").strip()
    return ""


@bp.route("/api/research/stream", methods=["POST"])
def research_stream():
    """SSE endpoint for the Deep Research workflow.

    Runs ``optiq.lab.deep_research.deep_research`` (a TTD-DR plan -> draft ->
    search/extract -> revise -> report -> verify loop) against the resolved chat
    endpoint, streaming its ``on_event`` trace so the UI can render a research
    card, then a final ``report`` event carrying the Markdown report + summary +
    sources for the artifact pane.

    Body: same shape as ``/api/chat/stream`` (model, messages, base_url, api_key,
    temperature). Deep research ignores tools/JSON-mode.
    """
    import queue
    import threading

    from ..chat_orchestrator import _call_mlx_lm
    from ..deep_research import deep_research

    data = request.get_json(force=True) or {}
    api_url, api_key = _resolve_chat_endpoint(
        data,
        current_app.config.get("OPTIQ_API_URL", "http://127.0.0.1:8080"),
        current_app.config.get("OPTIQ_API_KEY", "sk-optiq-local"),
    )
    model = str(data.get("model") or "")
    temperature = float(data.get("temperature", 0.4))
    question = _last_user_text(data.get("messages") or [])
    session_id, cancel_event = register_session(data.get("session_id"))

    # Research budget — request-tunable, but defaulted lighter than the library
    # defaults because the Lab commonly drives a *local* model where a 3-round /
    # 16k-token run costs 30+ minutes. These give a strong report in ~half that.
    def _budget(key: str, default: int, lo: int, hi: int) -> int:
        try:
            return max(lo, min(hi, int((data.get("research") or {}).get(key, default))))
        except (TypeError, ValueError):
            return default
    b_rounds = _budget("max_rounds", 2, 1, 4)
    b_queries = _budget("queries_per_round", 2, 1, 5)
    b_sources = _budget("sources_per_query", 2, 1, 5)
    b_report_tokens = _budget("report_tokens", 8000, 2000, 16000)

    def chat(prompt: str, json_mode: bool = False, max_tokens: int = 2000) -> str:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        # Deep-research calls can be long (a 16k-token report); give each a wide
        # per-call ceiling. The whole run is user-cancellable via cancel_event.
        resp = _call_mlx_lm(api_url, body, timeout=1800.0, api_key=api_key)
        return (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""

    def gen():
        try:
            yield f"event: session\ndata: {json.dumps({'session_id': session_id})}\n\n".encode()
            if not question:
                yield ("event: error\ndata: "
                       f"{json.dumps({'message': 'Deep research needs a question.'})}\n\n").encode()
                yield b"event: done\ndata: {}\n\n"
                return

            events: "queue.Queue" = queue.Queue()
            result: dict = {}

            def on_event(kind: str, payload: dict) -> None:
                events.put((kind, payload))

            def run() -> None:
                try:
                    result["out"] = deep_research(
                        question, chat=chat, on_event=on_event,
                        cancelled=cancel_event.is_set,
                        max_rounds=b_rounds, queries_per_round=b_queries,
                        sources_per_query=b_sources, report_tokens=b_report_tokens,
                    )
                except Exception as exc:                        # noqa: BLE001
                    result["error"] = str(exc)
                finally:
                    events.put(("__end__", {}))

            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            while True:
                kind, payload = events.get()
                if kind == "__end__":
                    break
                if kind == "done":                              # internal end marker
                    continue                                    # we send our own below
                yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n".encode()

            if cancel_event.is_set():
                yield b"event: cancelled\ndata: {}\n\n"
            elif "error" in result:
                yield ("event: error\ndata: "
                       f"{json.dumps({'message': result['error']})}\n\n").encode()
            else:
                out = result.get("out") or {}
                yield ("event: report\ndata: " + json.dumps({
                    "report": out.get("report", ""),
                    "summary": out.get("summary", ""),
                    "sources": out.get("sources", []),
                    "plan": out.get("plan", ""),
                }) + "\n\n").encode()
            yield b"event: done\ndata: {}\n\n"
        finally:
            cleanup_session(session_id)

    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@bp.route("/api/chat/cancel", methods=["POST"])
def chat_cancel():
    """Cancel an in-flight chat stream by session_id."""
    data = request.get_json(silent=True) or {}
    sid = (data.get("session_id") or "").strip()
    if not sid:
        return jsonify({"error": "session_id required"}), 400
    ok = cancel_session(sid)
    return jsonify({"ok": ok})


@bp.route("/api/sandbox/info", methods=["GET"])
def sandbox_info():
    """Report which sandbox backend is currently active."""
    return jsonify({"kind": detect_sandbox_kind()})


@bp.route("/api/files/extract", methods=["POST"])
def file_extract_endpoint():
    """Accept one file upload, return its extracted text body."""
    f = request.files.get("file")
    if f is None:
        return jsonify({"error": "no file"}), 400

    raw = f.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"error": f"file exceeds {MAX_UPLOAD_BYTES} bytes"}), 413

    try:
        text = extract_file(f.filename or "", raw)
    except UnsupportedFile as e:
        return jsonify({"error": str(e)}), 415
    except ExtractFailed as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"text": text, "filename": f.filename, "size": len(raw)})


# ---------------------------------------------------------------------------


def _parse_sse_chunk(chunk: bytes) -> tuple[str | None, dict | None]:
    """Best-effort parse of a single SSE frame yielded by the orchestrator."""
    try:
        text = chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
    except Exception:  # noqa: BLE001
        return None, None
    event: str | None = None
    data: dict | None = None
    for line in text.split("\n"):
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            raw = line[5:].strip()
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = None
            data = parsed if isinstance(parsed, dict) else None
    return event, data


def _is_chat_id(s: str) -> bool:
    """Tight allowlist — defense against path traversal in chat_id."""
    return s.startswith("chat_") and s[5:].isalnum() and len(s) <= 32
