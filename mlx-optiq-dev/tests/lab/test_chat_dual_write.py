"""Dual-write chat store: JSON file + spine DB, migrate-on-read (Phase 0)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from optiq.lab import chat_store, config, db, spine


def _chats_dir() -> Path:
    return config.ensure_lab_dirs().chats_dir


def test_save_writes_file_and_db(lab_home):
    chat_id = chat_store.save_chat_record({
        "title": "Dual write",
        "model": "qwen-local",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
    })
    assert chat_id.startswith("chat_")

    path = _chats_dir() / f"{chat_id}.json"
    assert path.is_file()
    file_data = json.loads(path.read_text())
    assert file_data["id"] == chat_id
    assert file_data["title"] == "Dual write"
    assert file_data["model"] == "qwen-local"
    assert len(file_data["messages"]) == 2
    assert file_data["messages"][0]["content"] == "Hello"
    assert "updated_at" in file_data

    conv = spine.get_conversation(chat_id)
    assert conv is not None
    assert conv["id"] == chat_id
    assert conv["title"] == "Dual write"
    assert conv["model"] == "qwen-local"
    assert len(conv["messages"]) == 2
    assert conv["messages"][0]["role"] == "user"
    assert conv["messages"][0]["content"] == "Hello"
    assert conv["messages"][1]["role"] == "assistant"
    assert conv["messages"][1]["content"] == "Hi"


def test_load_from_db_returns_same_messages(lab_home):
    chat_id = chat_store.save_chat_record({
        "title": "Load test",
        "model": "m",
        "messages": [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
        ],
    })
    # Remove file so load must come from DB
    path = _chats_dir() / f"{chat_id}.json"
    path.unlink()
    assert not path.is_file()

    loaded = chat_store.load_chat_record(chat_id)
    assert loaded is not None
    assert loaded["id"] == chat_id
    assert loaded["title"] == "Load test"
    assert loaded["model"] == "m"
    contents = [m["content"] for m in loaded["messages"]]
    roles = [m["role"] for m in loaded["messages"]]
    assert contents == ["ping", "pong"]
    assert roles == ["user", "assistant"]


def test_migrate_on_read_imports_file_only_chat(lab_home):
    # Initialize DB first so the one-shot spine backfill does not consume
    # this file; migrate-on-read is what we are exercising here.
    db.get_conn()

    chat_id = "chat_migrate01"
    path = _chats_dir() / f"{chat_id}.json"
    payload = {
        "id": chat_id,
        "title": "Legacy file",
        "model": "old-model",
        "messages": [
            {"role": "user", "content": "from file only"},
            {"role": "assistant", "content": "legacy reply"},
        ],
        "updated_at": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2))

    # Not in DB yet
    assert spine.get_conversation(chat_id) is None

    loaded = chat_store.load_chat_record(chat_id)
    assert loaded is not None
    assert loaded["id"] == chat_id
    assert loaded["title"] == "Legacy file"
    assert [m["content"] for m in loaded["messages"]] == [
        "from file only",
        "legacy reply",
    ]

    # Migrated into spine
    conv = spine.get_conversation(chat_id)
    assert conv is not None
    assert conv["title"] == "Legacy file"
    assert len(conv["messages"]) == 2
    assert conv["messages"][0]["content"] == "from file only"


def test_assistant_provenance_persists(lab_home):
    envelope = {
        "build_id": "bld_chat1",
        "sampler": {"temperature": 0.5},
        "context_window": 8192,
        "captured_at": "2026-08-02T12:00:00Z",
    }
    chat_id = chat_store.save_chat_record({
        "title": "With provenance",
        "model": "qwen-local",
        "messages": [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "a",
                "provenance": envelope,
            },
        ],
    })
    conv = spine.get_conversation(chat_id)
    assert conv is not None
    asst = conv["messages"][1]
    assert asst["role"] == "assistant"
    assert asst.get("provenance") == envelope
    assert spine.get_provenance(asst["id"]) == envelope


def test_delete_removes_file_and_db(lab_home):
    chat_id = chat_store.save_chat_record({
        "title": "To delete",
        "model": "m",
        "messages": [
            {"role": "user", "content": "bye"},
            {
                "role": "assistant",
                "content": "ok",
                "provenance": {
                    "server_model_label": "m",
                    "sampler": {},
                    "context_window": 1,
                    "captured_at": "t",
                },
            },
        ],
    })
    path = _chats_dir() / f"{chat_id}.json"
    assert path.is_file()
    conv = spine.get_conversation(chat_id)
    assert conv is not None
    asst_id = conv["messages"][1]["id"]
    assert spine.get_provenance(asst_id) is not None

    deleted = chat_store.delete_chat_record(chat_id)
    assert deleted is True
    assert not path.is_file()
    assert spine.get_conversation(chat_id) is None
    assert spine.get_provenance(asst_id) is None

    conn = db.get_conn()
    msg_count = conn.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
        (chat_id,),
    ).fetchone()["n"]
    assert msg_count == 0

    # Second delete is a no-op
    assert chat_store.delete_chat_record(chat_id) is False
