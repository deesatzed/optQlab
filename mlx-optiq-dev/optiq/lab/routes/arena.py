"""Model Arena: load two models and compare their answers to the same prompt
side by side, with tokens/sec for each.

Reuses the running serve infra. Model A is the Lab's main API server (loaded
via the existing Settings -> Server / Hub flow). Model B runs in a second
``ApiSupervisor`` on ``main_port + 1``, started on demand. Both are queried in
parallel through their OpenAI-compatible ``/v1/chat/completions`` endpoints.

Best used with small/fast models (two models are resident at once). The page
warns about this; loading a big model as B on a tight machine is the user's
call.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

from flask import Blueprint, current_app, jsonify, render_template, request

bp = Blueprint("arena", __name__)

# Second, on-demand supervisor for model B (port = main api port + 1).
_b_sup = None
_b_lock = threading.Lock()


def _main_api_url() -> str:
    return current_app.config["OPTIQ_API_URL"].rstrip("/")


def _b_api_url() -> str:
    base = _main_api_url()
    host, port = base.rsplit(":", 1)
    return f"{host}:{int(port) + 1}"


def _ensure_b_supervisor():
    """Lazily create the second supervisor on main_port + 1."""
    global _b_sup
    with _b_lock:
        if _b_sup is None:
            from ..api_supervisor import ApiSupervisor
            from ..config import lab_paths

            base = current_app.config["OPTIQ_API_URL"].rstrip("/")
            host = base.split("://")[-1].rsplit(":", 1)[0]
            port = int(base.rsplit(":", 1)[1]) + 1
            try:
                cache_dir = lab_paths().cache_dir
            except Exception:
                cache_dir = None
            _b_sup = ApiSupervisor(host=host, port=port, log_dir=cache_dir)
        return _b_sup


@bp.route("/arena")
def arena_page():
    return render_template("arena.html", page_title="Arena", section="arena")


@bp.route("/api/arena/load_b", methods=["POST"])
def arena_load_b():
    """Start/swap model B in the second supervisor."""
    data = request.get_json(force=True) or {}
    model = (data.get("model") or "").strip()
    if not model:
        return jsonify({"ok": False, "error": "model is required"}), 400
    try:
        sup = _ensure_b_supervisor()
        sup.start(model=model)  # idempotent swap if already running another
        return jsonify({"ok": True, "model": model, "api_url": _b_api_url()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def _complete(api_url: str, model: str, messages: list, max_tokens: int,
              temperature: float, out: dict) -> None:
    """One non-streaming completion; records text + tok/s into ``out``."""
    body = {
        "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "stream": False,
        # Arena is a quick head-to-head, not a deep-reasoning bench: ask
        # reasoning models to answer directly so the panes show a real answer
        # rather than burning the budget on hidden chain-of-thought.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if model:
        body["model"] = model
    req = urllib.request.Request(
        f"{api_url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer sk-optiq-local"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - t0
        msg = (d.get("choices") or [{}])[0].get("message") or {}
        usage = d.get("usage") or {}
        ct = usage.get("completion_tokens") or 0
        out.update({
            "ok": True,
            "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning") or "",
            "tokens": ct,
            "elapsed_sec": round(elapsed, 2),
            "tok_per_sec": round(ct / elapsed, 1) if elapsed > 0 and ct else 0.0,
        })
    except Exception as e:
        out.update({"ok": False, "error": str(e)[:300]})


@bp.route("/api/arena/run", methods=["POST"])
def arena_run():
    """Fire the same prompt at model A (main server) and model B (second
    supervisor) in parallel; return both answers + tokens/sec."""
    data = request.get_json(force=True) or {}
    messages = data.get("messages") or []
    max_tokens = int(data.get("max_tokens", 512))
    temperature = float(data.get("temperature", 0.7))
    model_a = (data.get("model_a") or "").strip()
    model_b = (data.get("model_b") or "").strip()
    if not messages:
        return jsonify({"ok": False, "error": "messages required"}), 400

    a_out: dict = {}
    b_out: dict = {}
    ta = threading.Thread(target=_complete, args=(
        _main_api_url(), model_a, messages, max_tokens, temperature, a_out))
    tb = threading.Thread(target=_complete, args=(
        _b_api_url(), model_b, messages, max_tokens, temperature, b_out))
    ta.start(); tb.start()
    ta.join(); tb.join()
    return jsonify({"ok": True, "a": a_out, "b": b_out})
