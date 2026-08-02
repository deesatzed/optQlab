"""One-shot backfill of legacy Lab state into the spine tables.

Imports ``models_local`` rows into ``builds`` and on-disk ``chat_*.json``
files into ``conversations`` / ``messages``. Safe to re-run: existing
paths and conversation ids are skipped. Called from ``db._migrate`` once
(schema_meta key ``spine_backfill_v1``).
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config, db, events, spine


def backfill_builds_from_models_local() -> int:
    """For each models_local row, register_build if no build with same path.

    Return count newly created. Emit build.registered via spine.register_build.
    """
    conn = db.get_conn()
    rows = conn.execute(
        """
        SELECT path, source_hf_id, bpw, mtp_present, kv_config_path
        FROM models_local
        """
    ).fetchall()

    created = 0
    for row in rows:
        path = row["path"]
        if not path:
            continue
        existing = conn.execute(
            "SELECT id FROM builds WHERE path = ?",
            (path,),
        ).fetchone()
        if existing is not None:
            continue

        source_hf_id = row["source_hf_id"]
        name = source_hf_id or Path(path).name or path
        metadata: dict = {}
        if row["mtp_present"] is not None:
            metadata["mtp_present"] = bool(row["mtp_present"])
        if row["kv_config_path"]:
            metadata["kv_config_path"] = row["kv_config_path"]

        spine.register_build(
            name=name,
            path=path,
            source_hf_id=source_hf_id,
            bpw=row["bpw"],
            metadata=metadata or None,
        )
        created += 1
    return created


def backfill_chats_from_disk() -> int:
    """Scan ensure_lab_dirs().chats_dir for chat_*.json; if conversation id
    not in DB, migrate via spine.upsert. Return count imported.

    Emit conversation.imported events.
    """
    chats_dir = config.ensure_lab_dirs().chats_dir
    if not chats_dir.is_dir():
        return 0

    conn = db.get_conn()
    created = 0
    for path in sorted(chats_dir.glob("chat_*.json")):
        try:
            raw = path.read_text()
            data = json.loads(raw)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue

        chat_id = data.get("id") or path.stem
        if not chat_id:
            continue
        data["id"] = chat_id

        existing = conn.execute(
            "SELECT id FROM conversations WHERE id = ?",
            (chat_id,),
        ).fetchone()
        if existing is not None:
            continue

        spine.upsert_conversation_from_chat_payload(data)
        events.append(
            type="conversation.imported",
            entity_type="conversation",
            entity_id=chat_id,
            payload={
                "path": str(path),
                "title": data.get("title"),
                "model": data.get("model"),
                "message_count": len(data.get("messages") or []),
            },
            workspace_id=data.get("workspace_id"),
        )
        created += 1
    return created


def run_all() -> dict:
    """Run both; return ``{"builds": n, "chats": n}``."""
    builds = backfill_builds_from_models_local()
    chats = backfill_chats_from_disk()
    return {"builds": builds, "chats": chats}
