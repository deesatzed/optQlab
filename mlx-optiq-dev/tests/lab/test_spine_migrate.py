"""Backfill models_local + chat_*.json into spine (Phase 0 Task 8)."""

from __future__ import annotations

import json

from optiq.lab import config, db, events, spine, spine_migrate


def test_backfill_builds_from_models_local(lab_home):
    conn = db.get_conn()
    path = "/models/qwen-4bit-optiq"
    conn.execute(
        """
        INSERT INTO models_local (path, source_hf_id, bpw, mtp_present, kv_config_path)
        VALUES (?, ?, ?, ?, ?)
        """,
        (path, "Qwen/Qwen2.5-7B", 4.5, 1, "/models/qwen/kv.json"),
    )

    # Not yet a build
    assert conn.execute(
        "SELECT id FROM builds WHERE path = ?", (path,)
    ).fetchone() is None

    n = spine_migrate.backfill_builds_from_models_local()
    assert n == 1

    row = conn.execute(
        "SELECT * FROM builds WHERE path = ?", (path,)
    ).fetchone()
    assert row is not None
    assert row["name"] == "Qwen/Qwen2.5-7B"
    assert row["source_hf_id"] == "Qwen/Qwen2.5-7B"
    assert row["bpw"] == 4.5
    assert row["path"] == path

    bld = spine.get_build(row["id"])
    assert bld is not None
    assert bld["metadata"]["mtp_present"] is True
    assert bld["metadata"]["kv_config_path"] == "/models/qwen/kv.json"

    registered = [
        e for e in events.iter_after(0) if e["type"] == "build.registered"
    ]
    assert any(e["payload"].get("path") == path for e in registered)

    # Idempotent
    assert spine_migrate.backfill_builds_from_models_local() == 0
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM builds WHERE path = ?", (path,)
    ).fetchone()["n"] == 1


def test_backfill_chats_from_disk(lab_home):
    # Open DB first so the one-shot migrate backfill runs against an empty
    # chats_dir; then plant a legacy file for the explicit backfill call.
    db.get_conn()

    chats_dir = config.ensure_lab_dirs().chats_dir
    chat_id = "chat_import01"
    payload = {
        "id": chat_id,
        "title": "Legacy import",
        "model": "old-model",
        "messages": [
            {"role": "user", "content": "from disk"},
            {"role": "assistant", "content": "imported reply"},
        ],
        "updated_at": 1.0,
    }
    (chats_dir / f"{chat_id}.json").write_text(json.dumps(payload, indent=2))

    assert spine.get_conversation(chat_id) is None

    n = spine_migrate.backfill_chats_from_disk()
    assert n == 1

    conv = spine.get_conversation(chat_id)
    assert conv is not None
    assert conv["id"] == chat_id
    assert conv["title"] == "Legacy import"
    assert conv["model"] == "old-model"
    assert [m["content"] for m in conv["messages"]] == [
        "from disk",
        "imported reply",
    ]

    imported = [
        e for e in events.iter_after(0) if e["type"] == "conversation.imported"
    ]
    assert len(imported) == 1
    assert imported[0]["entity_type"] == "conversation"
    assert imported[0]["entity_id"] == chat_id
    assert imported[0]["payload"]["title"] == "Legacy import"

    # Idempotent
    assert spine_migrate.backfill_chats_from_disk() == 0


def test_run_all_idempotent_via_meta_key_migrate(lab_home):
    """Second migrate via get_conn does not create duplicates (meta key)."""
    conn = db.get_conn()
    meta = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'spine_backfill_v1'"
    ).fetchone()
    assert meta is not None
    assert meta["value"] == "done"

    # Seed after the empty first-pass backfill
    model_path = "/models/meta-backfill"
    conn.execute(
        """
        INSERT INTO models_local (path, source_hf_id, bpw)
        VALUES (?, ?, ?)
        """,
        (model_path, "org/meta-model", 5.0),
    )
    chat_id = "chat_metabf01"
    chats_dir = config.ensure_lab_dirs().chats_dir
    (chats_dir / f"{chat_id}.json").write_text(
        json.dumps(
            {
                "id": chat_id,
                "title": "Meta path chat",
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "updated_at": 2.0,
            }
        )
    )

    # Clear meta so next connect re-runs backfill
    conn.execute("DELETE FROM schema_meta WHERE key = 'spine_backfill_v1'")
    conn.close()
    db._local.conn = None

    conn2 = db.get_conn()
    meta2 = conn2.execute(
        "SELECT value FROM schema_meta WHERE key = 'spine_backfill_v1'"
    ).fetchone()
    assert meta2 is not None
    assert meta2["value"] == "done"

    builds = conn2.execute(
        "SELECT * FROM builds WHERE path = ?", (model_path,)
    ).fetchall()
    assert len(builds) == 1
    assert spine.get_conversation(chat_id) is not None

    n_builds = conn2.execute("SELECT COUNT(*) AS n FROM builds").fetchone()["n"]
    n_convs = conn2.execute(
        "SELECT COUNT(*) AS n FROM conversations"
    ).fetchone()["n"]

    # Reconnect: meta key present → backfill not re-run; counts stable
    conn2.close()
    db._local.conn = None
    conn3 = db.get_conn()
    assert (
        conn3.execute("SELECT COUNT(*) AS n FROM builds").fetchone()["n"]
        == n_builds
    )
    assert (
        conn3.execute("SELECT COUNT(*) AS n FROM conversations").fetchone()["n"]
        == n_convs
    )

    # Direct second run_all also creates zero new
    result = spine_migrate.run_all()
    assert result == {"builds": 0, "chats": 0}
