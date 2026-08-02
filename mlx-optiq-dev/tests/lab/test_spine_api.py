"""Spine API routes — events, bus SSE, provenance, workspace CRUD."""

from __future__ import annotations

import json

from optiq.lab import events, spine
from optiq.lab.app import create_app


def _client(lab_home):
    """Flask test client with auth bypass against the lab_home OPTIQ_HOME."""
    app = create_app(secret_key=b"x" * 32)
    app.config["TESTING"] = True
    app.config["OPTIQ_TEST_AUTH_BYPASS"] = True
    return app.test_client()


def test_workspace_create_and_list(lab_home):
    client = _client(lab_home)

    resp = client.post(
        "/api/workspaces",
        json={
            "name": "Main lab",
            "system_prompt": "Be brief.",
            "sampler_json": {"temperature": 0.2},
        },
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["id"].startswith("ws_")
    assert body["name"] == "Main lab"
    assert body["system_prompt"] == "Be brief."
    assert body["sampler_json"] == {"temperature": 0.2}
    ws_id = body["id"]

    listed = client.get("/api/workspaces")
    assert listed.status_code == 200
    workspaces = listed.get_json()["workspaces"]
    assert any(w["id"] == ws_id for w in workspaces)

    one = client.get(f"/api/workspaces/{ws_id}")
    assert one.status_code == 200
    assert one.get_json()["name"] == "Main lab"

    missing = client.get("/api/workspaces/ws_missing")
    assert missing.status_code == 404


def test_workspace_patch_name_and_build(lab_home):
    client = _client(lab_home)
    ws_id = spine.create_workspace("Before")
    bld_id = spine.register_build(name="A", path="/m/a")
    other = spine.register_build(name="B", path="/m/b")

    resp = client.patch(
        f"/api/workspaces/{ws_id}",
        json={
            "name": "After",
            "default_build_id": bld_id,
            "resident_build_id": other,
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["name"] == "After"
    assert body["default_build_id"] == bld_id
    assert body["coherence_flags_json"]["model_not_resident"] is True


def test_events_list_after_append(lab_home):
    client = _client(lab_home)

    eid = events.append(
        type="test.ping",
        entity_type="test",
        entity_id="t1",
        payload={"n": 1},
    )

    resp = client.get("/api/events?after_id=0&limit=500")
    assert resp.status_code == 200
    rows = resp.get_json()["events"]
    assert any(e["id"] == eid and e["type"] == "test.ping" for e in rows)

    after = client.get(f"/api/events?after_id={eid}")
    assert after.status_code == 200
    assert after.get_json()["events"] == []


def test_provenance_export(lab_home):
    client = _client(lab_home)

    conv_id = spine.upsert_conversation_from_chat_payload(
        {
            "title": "prov",
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "yo",
                    "id": "msg_prov_api_1",
                },
            ],
        }
    )
    assert conv_id

    envelope = {
        "build_id": "bld_x",
        "sampler": {"temperature": 0.5},
        "context_window": 8192,
        "captured_at": "2026-08-02T12:00:00Z",
    }
    spine.set_message_provenance("msg_prov_api_1", envelope)

    resp = client.get("/api/messages/msg_prov_api_1/provenance")
    assert resp.status_code == 200
    assert resp.get_json() == envelope

    missing = client.get("/api/messages/msg_nope/provenance")
    assert missing.status_code == 404


def test_bus_stream_max_events(lab_home):
    client = _client(lab_home)

    events.append(
        type="bus.test",
        entity_type="test",
        entity_id="bus1",
        payload={"k": "v"},
    )

    resp = client.get("/api/bus/stream?after_id=0&max_events=1")
    assert resp.status_code == 200
    assert resp.mimetype == "text/event-stream"
    text = resp.data.decode("utf-8")
    assert "data: " in text

    # Parse first data line
    data_lines = [
        line[len("data: ") :]
        for line in text.splitlines()
        if line.startswith("data: ")
    ]
    assert data_lines
    payload = json.loads(data_lines[0])
    assert payload["type"] == "bus.test"
    assert payload["payload"] == {"k": "v"}
