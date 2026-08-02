"""Session persistence + resume — Claude-Code-style.

A *session* is one OptiQ Code conversation in a repo. It is stored globally under
``~/.optiq/code/sessions/<repo-key>/`` (design §1), keyed by the repo's absolute
path so ``optiq code -c`` resumes the most recent session *in this repo*:

  * ``<id>.jsonl``          — the event log (meta / input / stream lines)
  * ``<id>.messages.json``  — the full OpenAI-format trajectory (for export/resume)

Launch behaviour mirrors Claude Code:
  optiq code -c / --continue   → resume the most recent session in this repo
  optiq code -r / --resume ID  → resume a specific session

Kept UI-agnostic and dependency-free so it unit-tests without Textual.
Ported from conjure/sessions.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


def _sessions_root() -> Path:
    home = Path(os.environ.get("OPTIQ_HOME") or (Path.home() / ".optiq")) / "code"
    return home / "sessions"


def repo_key(repo: Path | str) -> str:
    """A filesystem-safe key for a repo: its name plus a short hash of the
    absolute path (so two same-named repos don't collide)."""
    p = Path(repo).resolve()
    return f"{p.name}-{hashlib.sha1(str(p).encode()).hexdigest()[:8]}"


@dataclass
class SessionEvent:
    kind: str          # "meta" | "input" (a goal) | "stream" (rendered line)
    text: str
    ts: float = 0.0

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, "ts": self.ts}

    @classmethod
    def from_dict(cls, d: dict) -> "SessionEvent":
        return cls(kind=d.get("kind", "stream"), text=d.get("text", ""),
                   ts=float(d.get("ts", 0.0)))


@dataclass
class SessionMeta:
    id: str
    title: str
    created_at: float
    updated_at: float


class Session:
    """One conversation's append-only event log + trajectory sidecar."""

    def __init__(self, store: "SessionStore", sid: str):
        self.store = store
        self.id = sid
        self.path = store.dir / f"{sid}.jsonl"

    def append(self, kind: str, text: str) -> None:
        self.store.dir.mkdir(parents=True, exist_ok=True)
        ev = SessionEvent(kind=kind, text=text, ts=time.time())
        with self.path.open("a") as f:
            f.write(json.dumps(ev.to_dict()) + "\n")

    def events(self) -> list[SessionEvent]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                try:
                    out.append(SessionEvent.from_dict(json.loads(line)))
                except json.JSONDecodeError:
                    pass
        return out

    # ── the full trajectory (system/user/assistant/tool), for export + resume ──
    def save_messages(self, messages: list) -> None:
        self.store.dir.mkdir(parents=True, exist_ok=True)
        self.path.with_suffix(".messages.json").write_text(json.dumps(messages))

    def load_messages(self) -> list:
        side = self.path.with_suffix(".messages.json")
        if not side.exists():
            return []
        try:
            return json.loads(side.read_text())
        except (OSError, ValueError):
            return []

    def last_goal(self) -> str:
        for ev in reversed(self.events()):
            if ev.kind == "input":
                return ev.text
        return ""

    def meta(self) -> dict:
        for ev in self.events():
            if ev.kind == "meta":
                try:
                    return json.loads(ev.text)
                except json.JSONDecodeError:
                    return {}
        return {}


class SessionStore:
    """Manages one repo's sessions under ~/.optiq/code/sessions/<repo-key>/."""

    def __init__(self, repo: Path | str):
        self.repo = Path(repo).resolve()
        self.dir = _sessions_root() / repo_key(self.repo)

    @staticmethod
    def _gen_id() -> str:
        # 8 hex chars: the timestamp resolves only to the second, so a burst of
        # sessions in one second relies on the suffix to stay unique.
        return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]

    def new(self, session_id: str | None = None) -> Session:
        return Session(self, session_id or self._gen_id())

    def load(self, sid: str) -> Session:
        return Session(self, sid)

    def list(self) -> list[SessionMeta]:
        """All sessions in this repo, most-recently-updated first."""
        out: list[SessionMeta] = []
        if not self.dir.is_dir():
            return out
        for p in sorted(self.dir.glob("*.jsonl")):
            try:
                rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
            except (OSError, json.JSONDecodeError):
                rows = []
            title = next((d["text"][:60] for d in rows if d.get("kind") == "input"), "")
            if not title and rows:
                title = rows[0].get("text", "")[:60]
            created = float(rows[0].get("ts", 0.0)) if rows else p.stat().st_mtime
            updated = float(rows[-1].get("ts", 0.0)) if rows else p.stat().st_mtime
            out.append(SessionMeta(id=p.stem, title=title or "(empty session)",
                                   created_at=created, updated_at=updated))
        out.sort(key=lambda m: m.updated_at, reverse=True)
        return out

    def latest(self) -> SessionMeta | None:
        sessions = self.list()
        return sessions[0] if sessions else None
