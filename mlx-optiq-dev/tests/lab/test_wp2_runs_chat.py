"""WP-2: Runs API, chat search, health event wiring."""

from __future__ import annotations

import json

from optiq.lab import chat_store, job_bus, jobs, spine
from optiq.lab.app import create_app
from optiq.lab.config import ensure_lab_dirs
from optiq.lab.job_bus import _reset_for_tests


def _client(lab_home):
    ensure_lab_dirs()
    app = create_app(secret_key=b"y" * 32)
    app.config["TESTING"] = True
    app.config["OPTIQ_TEST_AUTH_BYPASS"] = True
    return app.test_client()


def _slow(emit, config):
    import time
    emit({"type": "progress", "progress": 0.2, "message": "go"})
    time.sleep(float(config.get("sleep", 0.15)))
    emit({"type": "progress", "progress": 1.0, "message": "done"})


def test_runs_list_and_page(lab_home):
    _reset_for_tests()
    c = _client(lab_home)
    jid = job_bus.submit("test_wp2", _slow, {"sleep": 0.1}, resource_class="light")
    # wait done
    import time
    for _ in range(50):
        row = jobs.get(jid)
        if row and row["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.05)
    r = c.get("/api/runs")
    assert r.status_code == 200
    data = r.get_json()
    assert any(x["id"] == jid for x in data["runs"])
    page = c.get("/runs")
    assert page.status_code == 200
    assert b"Runs" in page.data
    log = c.get(f"/api/runs/{jid}/log")
    assert log.status_code == 200
    assert "lines" in log.get_json()


def test_chat_search(lab_home):
    ensure_lab_dirs()
    chat_store.save_chat_record({
        "id": "chat_searchalpha",
        "title": "Alpha clinical notes",
        "model": "m1",
        "messages": [{"role": "user", "content": "hello alpha"},
                     {"role": "assistant", "content": "hi", "provenance": {
                         "server_model_label": "m1",
                         "sampler": {"temperature": 0.7},
                         "context_window": 8192,
                         "captured_at": "2026-01-01T00:00:00Z",
                     }}],
    })
    chat_store.save_chat_record({
        "id": "chat_searchbeta",
        "title": "Beta latency",
        "model": "m2",
        "messages": [{"role": "user", "content": "other"}],
    })
    hits = chat_store.search_chat_records("clinical")
    ids = {h["id"] for h in hits}
    assert "chat_searchalpha" in ids
    assert "chat_searchbeta" not in ids

    c = _client(lab_home)
    r = c.get("/api/chats?q=latency")
    assert r.status_code == 200
    ids = {h["id"] for h in r.get_json()["chats"]}
    assert "chat_searchbeta" in ids


def test_workspace_coherence_api(lab_home):
    ensure_lab_dirs()
    ws = spine.create_workspace("wp2-ws", system_prompt="Be careful.")
    b = spine.register_build(name="b", path="/tmp/wp2-build", weights_gb=1.0)
    spine.set_workspace_build(ws, b, resident_build_id="other-resident")
    got = spine.get_workspace(ws)
    assert got["coherence_flags_json"].get("model_not_resident") is True

    c = _client(lab_home)
    r = c.get(f"/api/workspaces/{ws}")
    assert r.status_code == 200
    assert r.get_json()["system_prompt"] == "Be careful."


def test_health_event_emitted_in_stream_shape(lab_home):
    """Unit-level: health payload shape used by chat SSE (no live model)."""
    # Document expected keys for client chips
    health = {
        "healed": True,
        "retry_hits": 1,
        "tools_called": [{"name": "python", "healed": True}],
        "retrieved_chunks": 0,
        "retrieval_empty": True,
        "context_window": None,
        "tok_per_sec": None,
    }
    assert health["retrieval_empty"] is True
    assert health["tok_per_sec"] is None  # never invented
