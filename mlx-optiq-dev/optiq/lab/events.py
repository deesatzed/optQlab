"""Append-only event log repository for the Lab spine.

Events are the durable bus for SSE polling and audit trails. Writers call
``append``; consumers poll with ``iter_after(after_id=...)`` using the
integer primary key as a cursor.
"""

from __future__ import annotations

import json
from typing import Any

from . import db


def append(
    *,
    type: str,
    entity_type: str,
    entity_id: str,
    payload: dict | None = None,
    workspace_id: str | None = None,
) -> int:
    """Insert event; return new integer id."""
    payload_json = json.dumps(payload if payload is not None else {})
    with db.transaction() as conn:
        cur = conn.execute(
            """
            INSERT INTO events (type, entity_type, entity_id, payload_json, workspace_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (type, entity_type, entity_id, payload_json, workspace_id),
        )
        return int(cur.lastrowid)


def iter_after(after_id: int = 0, limit: int = 500) -> list[dict]:
    """Return events with id > after_id ordered by id ASC, max limit rows.

    Each dict: id, ts, type, entity_type, entity_id, payload (parsed dict),
    workspace_id.
    """
    conn = db.get_conn()
    rows = conn.execute(
        """
        SELECT id, ts, type, entity_type, entity_id, payload_json, workspace_id
        FROM events
        WHERE id > ?
        ORDER BY id ASC
        LIMIT ?
        """,
        (after_id, limit),
    ).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        raw = row["payload_json"]
        if raw is None or raw == "":
            payload: dict = {}
        else:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                payload = {}
        out.append(
            {
                "id": row["id"],
                "ts": row["ts"],
                "type": row["type"],
                "entity_type": row["entity_type"],
                "entity_id": row["entity_id"],
                "payload": payload,
                "workspace_id": row["workspace_id"],
            }
        )
    return out
