"""Phase 0 acceptance suite — end-to-end real SQLite under lab_home.

Covers schema v2, dual-write chat, provenance, sequential job bus,
events cursor resume, workspace coherence, and zombie job bookkeeping.
No mocks; process tests use polling with timeouts up to 15s.
"""

from __future__ import annotations

import time

import pytest

from optiq.lab import chat_store, db, events, job_bus, jobs, spine


# ---------------------------------------------------------------------------
# Helpers (module-level so Process targets stay picklable)
# ---------------------------------------------------------------------------

SPINE_TABLES = [
    "workspaces",
    "builds",
    "adapters",
    "datasets",
    "runs",
    "conversations",
    "messages",
    "message_provenance",
    "evals",
    "artifacts",
    "events",
]


def _acc_slow(emit, config):
    """Picklable slow job target for sequential-bus acceptance."""
    import time as _time

    emit({"type": "progress", "progress": 0.1, "message": "acc start"})
    _time.sleep(float(config.get("sleep", 0.3)))
    emit({"type": "progress", "progress": 1.0, "message": "acc done"})


def _table_names(conn) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {
        r["name"] if isinstance(r, dict) or hasattr(r, "keys") else r[0]
        for r in rows
    }


def _reset_db_conn() -> None:
    if getattr(db._local, "conn", None) is not None:
        try:
            db._local.conn.close()
        except Exception:
            pass
        db._local.conn = None


def _wait_status(job_id: str, want, timeout: float = 15.0) -> str:
    if isinstance(want, str):
        want = {want}
    else:
        want = set(want)
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        row = jobs.get(job_id)
        last = row["status"] if row else None
        if last in want:
            return last
        time.sleep(0.05)
    raise AssertionError(
        f"job {job_id} status={last!r} not in {want} within {timeout}s"
    )


# ---------------------------------------------------------------------------
# 1. Schema v2 on fresh home
# ---------------------------------------------------------------------------


def test_schema_v2_on_fresh_home(lab_home):
    conn = db.get_conn()
    version = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'version'"
    ).fetchone()
    assert version is not None
    assert version["value"] == "2"

    names = _table_names(conn)
    for table in SPINE_TABLES:
        assert table in names, f"missing spine table: {table}"


# ---------------------------------------------------------------------------
# 2. Schema v2 on v1-populated DB (data preserved across reconnect/migrate)
# ---------------------------------------------------------------------------


def test_schema_v2_preserves_data_across_reconnect(lab_home):
    conn = db.get_conn()
    assert (
        conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()["value"]
        == "2"
    )

    conn.execute(
        """
        INSERT INTO jobs (id, kind, status, config_json, log_path)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("job_acc_persist", "quantize", "done", "{}", "/tmp/acc.log"),
    )
    conn.execute(
        """
        INSERT INTO models_local (path, source_hf_id, bpw, mtp_present)
        VALUES (?, ?, ?, ?)
        """,
        ("/models/acc-qwen", "Qwen/Acc", 4.0, 0),
    )

    # Simulate process restart: drop thread-local connection, reconnect.
    _reset_db_conn()
    conn2 = db.get_conn()

    version = conn2.execute(
        "SELECT value FROM schema_meta WHERE key = 'version'"
    ).fetchone()
    assert version["value"] == "2"

    job_row = conn2.execute(
        "SELECT id, status FROM jobs WHERE id = ?",
        ("job_acc_persist",),
    ).fetchone()
    assert job_row is not None
    assert job_row["id"] == "job_acc_persist"
    assert job_row["status"] == "done"

    model_row = conn2.execute(
        "SELECT path FROM models_local WHERE path = ?",
        ("/models/acc-qwen",),
    ).fetchone()
    assert model_row is not None

    names = _table_names(conn2)
    for table in SPINE_TABLES:
        assert table in names, f"spine table missing after reconnect: {table}"


# ---------------------------------------------------------------------------
# 3. Dual-write chat round-trip
# ---------------------------------------------------------------------------


def test_dual_write_chat_round_trip(lab_home):
    messages = [
        {"role": "user", "content": "acceptance hello"},
        {"role": "assistant", "content": "acceptance hi"},
    ]
    chat_id = chat_store.save_chat_record(
        {
            "title": "Phase0 acceptance chat",
            "model": "acc-model",
            "messages": messages,
        }
    )
    assert chat_id.startswith("chat_")

    loaded = chat_store.load_chat_record(chat_id)
    assert loaded is not None
    assert loaded["id"] == chat_id
    assert loaded["title"] == "Phase0 acceptance chat"
    assert loaded["model"] == "acc-model"
    assert [m["role"] for m in loaded["messages"]] == ["user", "assistant"]
    assert [m["content"] for m in loaded["messages"]] == [
        "acceptance hello",
        "acceptance hi",
    ]

    # Spine mirror also matches
    conv = spine.get_conversation(chat_id)
    assert conv is not None
    assert [m["content"] for m in conv["messages"]] == [
        "acceptance hello",
        "acceptance hi",
    ]


# ---------------------------------------------------------------------------
# 4. Provenance export round-trip
# ---------------------------------------------------------------------------


def test_provenance_export_round_trip(lab_home):
    conv_id = spine.upsert_conversation_from_chat_payload(
        {
            "title": "prov acceptance",
            "model": "acc-model",
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "a",
                    "id": "msg_acc_prov_1",
                },
            ],
        }
    )
    assert conv_id

    envelope = {
        "build_id": "bld_acc",
        "sampler": {"temperature": 0.3, "max_tokens": 512},
        "context_window": 8192,
        "captured_at": "2026-08-02T20:00:00Z",
        "tok_per_sec": None,
        "peak_mem_gb": None,
    }
    spine.set_message_provenance("msg_acc_prov_1", envelope)

    got = spine.get_provenance("msg_acc_prov_1")
    assert got == envelope
    assert spine.provenance_complete(got)

    # Optional API surface (real Flask app, real SQLite)
    from optiq.lab.app import create_app

    app = create_app(secret_key=b"x" * 32)
    app.config["TESTING"] = True
    app.config["OPTIQ_TEST_AUTH_BYPASS"] = True
    client = app.test_client()

    resp = client.get("/api/messages/msg_acc_prov_1/provenance")
    assert resp.status_code == 200
    assert resp.get_json() == envelope


# ---------------------------------------------------------------------------
# 5. Sequential bus: two heavies never both running
# ---------------------------------------------------------------------------


@pytest.fixture()
def _reset_bus(lab_home):
    job_bus._reset_for_tests()
    yield
    job_bus._reset_for_tests()


def test_sequential_two_heavies_never_both_running(lab_home, _reset_bus):
    j1 = job_bus.submit(
        "test",
        _acc_slow,
        config={"sleep": 0.5},
        resource_class="memory_heavy",
    )
    j2 = job_bus.submit(
        "test",
        _acc_slow,
        config={"sleep": 0.2},
        resource_class="memory_heavy",
    )

    _wait_status(j1, "running")

    saw_j2_queued_while_j1_running = False
    deadline = time.time() + 5.0
    while time.time() < deadline:
        s1 = jobs.get(j1)["status"]
        s2 = jobs.get(j2)["status"]
        if s1 == "running" and s2 == "queued":
            saw_j2_queued_while_j1_running = True
            break
        if s1 == "done":
            break
        assert not (s1 == "running" and s2 == "running")
        time.sleep(0.05)

    assert saw_j2_queued_while_j1_running, (
        f"expected j2 queued while j1 running; j1={jobs.get(j1)}; j2={jobs.get(j2)}"
    )

    deadline = time.time() + 15.0
    while time.time() < deadline:
        s1 = jobs.get(j1)["status"]
        s2 = jobs.get(j2)["status"]
        assert not (s1 == "running" and s2 == "running")
        if s1 == "done" and s2 == "done":
            break
        time.sleep(0.05)
    else:
        raise AssertionError(
            f"jobs did not finish: j1={jobs.get(j1)}; j2={jobs.get(j2)}"
        )


# ---------------------------------------------------------------------------
# 6. Events after_id resume
# ---------------------------------------------------------------------------


def test_events_after_id_resume(lab_home):
    id1 = events.append(
        type="acc.one",
        entity_type="acc",
        entity_id="e1",
        payload={"n": 1},
    )
    id2 = events.append(
        type="acc.two",
        entity_type="acc",
        entity_id="e2",
        payload={"n": 2},
    )
    id3 = events.append(
        type="acc.three",
        entity_type="acc",
        entity_id="e3",
        payload={"n": 3},
    )
    assert id1 < id2 < id3

    remainder = events.iter_after(after_id=id1)
    assert len(remainder) == 2
    assert [r["id"] for r in remainder] == [id2, id3]
    assert [r["type"] for r in remainder] == ["acc.two", "acc.three"]

    after_mid = events.iter_after(after_id=id2)
    assert len(after_mid) == 1
    assert after_mid[0]["id"] == id3
    assert after_mid[0]["payload"] == {"n": 3}


# ---------------------------------------------------------------------------
# 7. Workspace coherence
# ---------------------------------------------------------------------------


def test_workspace_coherence_model_not_resident(lab_home):
    ws_id = spine.create_workspace("Acceptance WS")
    bld_id = spine.register_build(name="Build A", path="/m/a")
    other = spine.register_build(name="Build B", path="/m/b")

    updated = spine.set_workspace_build(
        ws_id, bld_id, resident_build_id=other
    )
    assert updated["default_build_id"] == bld_id
    assert updated["coherence_flags_json"]["model_not_resident"] is True

    row = spine.get_workspace(ws_id)
    assert row["coherence_flags_json"]["model_not_resident"] is True

    coh = [
        e for e in events.iter_after(0) if e["type"] == "workspace.coherence"
    ]
    assert any(
        e["entity_id"] == ws_id
        and e["payload"].get("model_not_resident") is True
        for e in coh
    )


# ---------------------------------------------------------------------------
# 8. mark_zombies still works
# ---------------------------------------------------------------------------


def test_mark_zombies_still_works(lab_home):
    conn = db.get_conn()
    conn.execute(
        """
        INSERT INTO jobs (id, kind, status, config_json, log_path)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("job_acc_zombie", "test", "running", "{}", "/tmp/acc_zombie.log"),
    )
    assert jobs.get("job_acc_zombie")["status"] == "running"

    jobs.mark_zombies()
    assert jobs.get("job_acc_zombie")["status"] == "zombie"
