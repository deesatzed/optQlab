"""Dual-write chat persistence: JSON files + spine SQLite (Phase 0).

JSON under ``chats_dir`` remains the client-facing file format. Spine
conversations/messages/provenance are written on every save and on
migrate-on-read when a legacy file is loaded without a DB row.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from . import config, db, spine


def _chats_dir() -> Path:
    d = config.ensure_lab_dirs().chats_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _chat_path(chat_id: str) -> Path:
    return _chats_dir() / f"{chat_id}.json"


def _mint_chat_id() -> str:
    return f"chat_{uuid.uuid4().hex[:12]}"


def _write_chat_file(chat_id: str, title: str, model: str, messages: list) -> dict:
    record = {
        "id": chat_id,
        "title": title,
        "model": model,
        "messages": messages,
        "updated_at": time.time(),
    }
    _chat_path(chat_id).write_text(json.dumps(record, indent=2))
    return record


def _conversation_to_api(conv: dict) -> dict:
    """Shape spine.get_conversation result for chat API clients."""
    messages: list[dict[str, Any]] = []
    for m in conv.get("messages") or []:
        item: dict[str, Any] = {
            "role": m.get("role"),
            "content": m.get("content"),
        }
        if m.get("id") is not None:
            item["id"] = m["id"]
        if "seq" in m:
            item["seq"] = m["seq"]
        if m.get("provenance") is not None:
            item["provenance"] = m["provenance"]
        messages.append(item)
    out: dict[str, Any] = {
        "id": conv["id"],
        "title": conv.get("title") or "Untitled chat",
        "model": conv.get("model") or "",
        "messages": messages,
    }
    # Prefer file updated_at when present (float epoch clients already use).
    path = _chat_path(conv["id"])
    if path.is_file():
        try:
            file_data = json.loads(path.read_text())
            if "updated_at" in file_data:
                out["updated_at"] = file_data["updated_at"]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return out


def save_chat_record(data: dict) -> str:
    """Dual-write a chat: JSON file + spine conversation/messages.

    1. Determine chat_id from data.get('id') or mint chat_...
    2. Write JSON file (id, title, model, messages, updated_at)
    3. spine.upsert_conversation_from_chat_payload with full data including id
    4. Return chat_id
    """
    if not isinstance(data, dict):
        data = {}
    chat_id = data.get("id") or _mint_chat_id()
    title = (data.get("title") or "Untitled chat")[:80]
    messages = data.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    model = data.get("model") or ""

    _write_chat_file(chat_id, title, model, messages)

    payload = dict(data)
    payload["id"] = chat_id
    payload["title"] = title
    payload["model"] = model
    payload["messages"] = messages
    spine.upsert_conversation_from_chat_payload(payload)
    return chat_id


def load_chat_record(chat_id: str) -> dict | None:
    """Load a chat: prefer DB, else migrate-on-read from JSON file.

    If conversation exists in DB via spine.get_conversation, return API-shaped
    dict. Else if JSON file exists, read it, upsert into spine (migrate-on-read),
    return file content. Else None.
    """
    conv = spine.get_conversation(chat_id)
    if conv is not None:
        return _conversation_to_api(conv)

    path = _chat_path(chat_id)
    if not path.is_file():
        return None

    try:
        file_data = json.loads(path.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(file_data, dict):
        return None

    # Ensure id is set for spine upsert
    if not file_data.get("id"):
        file_data["id"] = chat_id
    spine.upsert_conversation_from_chat_payload(file_data)
    return file_data


def list_chat_records() -> list[dict]:
    """List chats: prefer DB conversations if any, else scan JSON files.

    Each item: id, title, model, updated_at if available, n_messages.
    """
    conn = db.get_conn()
    rows = conn.execute(
        """
        SELECT c.id, c.title, c.model, c.updated_at,
               (SELECT COUNT(*) FROM messages m
                WHERE m.conversation_id = c.id) AS n_messages
        FROM conversations c
        ORDER BY c.updated_at DESC
        """
    ).fetchall()

    if rows:
        out: list[dict] = []
        for r in rows:
            item: dict[str, Any] = {
                "id": r["id"],
                "title": r["title"] or "Untitled chat",
                "model": r["model"] or "",
                "n_messages": int(r["n_messages"] or 0),
            }
            # Prefer float epoch from file when dual-written; else DB datetime.
            path = _chat_path(r["id"])
            if path.is_file():
                try:
                    file_data = json.loads(path.read_text())
                    if "updated_at" in file_data:
                        item["updated_at"] = file_data["updated_at"]
                    else:
                        item["updated_at"] = r["updated_at"]
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    item["updated_at"] = r["updated_at"]
            else:
                item["updated_at"] = r["updated_at"]
            out.append(item)
        # Sort by updated_at descending when values are comparable epochs
        def _sort_key(item: dict) -> float:
            u = item.get("updated_at", 0)
            if isinstance(u, (int, float)):
                return float(u)
            return 0.0

        if all(isinstance(i.get("updated_at"), (int, float)) for i in out):
            out.sort(key=_sort_key, reverse=True)
        return out

    # File fallback (legacy / empty DB)
    out = []
    for p in sorted(
        _chats_dir().glob("chat_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        try:
            data = json.loads(p.read_text())
            out.append({
                "id": data["id"],
                "title": data["title"],
                "model": data.get("model", ""),
                "updated_at": data.get("updated_at", 0),
                "n_messages": len(data.get("messages") or []),
            })
        except Exception:
            continue
    return out


def search_chat_records(query: str, *, limit: int = 50) -> list[dict]:
    """Search chats by title, model, or message content (case-insensitive).

    Returns the same list shape as ``list_chat_records``, filtered.
    """
    q = (query or "").strip().lower()
    if not q:
        return list_chat_records()[:limit]

    # Prefer DB full-text-ish LIKE over messages + conversations
    conn = db.get_conn()
    rows = conn.execute(
        """
        SELECT DISTINCT c.id
        FROM conversations c
        LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE lower(c.title) LIKE ?
           OR lower(ifnull(c.model, '')) LIKE ?
           OR lower(ifnull(m.content, '')) LIKE ?
        ORDER BY c.updated_at DESC
        LIMIT ?
        """,
        (f"%{q}%", f"%{q}%", f"%{q}%", int(limit)),
    ).fetchall()
    if rows:
        out: list[dict] = []
        for r in rows:
            rec = load_chat_record(r["id"])
            if not rec:
                continue
            out.append({
                "id": rec["id"],
                "title": rec.get("title") or "Untitled chat",
                "model": rec.get("model") or "",
                "updated_at": rec.get("updated_at"),
                "n_messages": len(rec.get("messages") or []),
            })
        return out

    # File fallback
    hits = []
    for item in list_chat_records():
        blob = f"{item.get('title','')} {item.get('model','')}".lower()
        path = _chat_path(item["id"])
        if path.is_file():
            try:
                blob += " " + path.read_text().lower()
            except OSError:
                pass
        if q in blob:
            hits.append(item)
        if len(hits) >= limit:
            break
    return hits


def delete_chat_record(chat_id: str) -> bool:
    """Delete JSON file and conversation/messages/provenance from DB.

    Return True if anything was deleted.
    """
    deleted = False

    path = _chat_path(chat_id)
    if path.is_file():
        path.unlink()
        deleted = True

    conn = db.get_conn()
    row = conn.execute(
        "SELECT id FROM conversations WHERE id = ?",
        (chat_id,),
    ).fetchone()
    if row is not None:
        with db.transaction() as txn:
            msg_rows = txn.execute(
                "SELECT id FROM messages WHERE conversation_id = ?",
                (chat_id,),
            ).fetchall()
            msg_ids = [r["id"] for r in msg_rows]
            if msg_ids:
                placeholders = ",".join("?" * len(msg_ids))
                txn.execute(
                    f"DELETE FROM message_provenance WHERE message_id IN ({placeholders})",
                    msg_ids,
                )
            txn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (chat_id,),
            )
            txn.execute(
                "DELETE FROM conversations WHERE id = ?",
                (chat_id,),
            )
        deleted = True

    return deleted
