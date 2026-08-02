"""Background job runner with line-delimited JSON progress logs.

Jobs run in ``multiprocessing.Process`` because MLX state isn't safe to
share between concurrent operations. Each job writes line-delimited JSON
events to ``<jobs_dir>/<job_id>.log`` and the Lab UI tails that file via
SSE. ``jobs`` table tracks queued / running / done / failed and persists
across Lab restarts (a running job at restart shows as ``zombie`` — the
UI offers to clean it up).
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator

from . import db
from .config import ensure_lab_dirs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def new_job_id() -> str:
    return f"job_{uuid.uuid4().hex[:16]}"


def submit(kind: str, target: Callable[..., None], config: dict, **kwargs) -> str:
    """Spawn a background process to run ``target(emit, config, **kwargs)``.

    ``target`` is called with an ``emit(event: dict)`` callable as its
    first arg. ``emit`` writes one JSON object per call to the job log
    and updates the DB row's ``progress`` / ``message`` columns.

    Returns the job id. The caller subscribes to events via ``tail()``.
    """
    paths = ensure_lab_dirs()
    job_id = new_job_id()
    log_path = paths.jobs_dir / f"{job_id}.log"
    log_path.touch()

    db.get_conn().execute(
        """
        INSERT INTO jobs (id, kind, status, config_json, log_path)
        VALUES (?, ?, 'queued', ?, ?)
        """,
        (job_id, kind, json.dumps(config), str(log_path)),
    )

    proc = mp.Process(
        target=_job_main,
        args=(job_id, str(log_path), target, config, kwargs),
        name=f"lab-job-{job_id}",
        daemon=True,
    )
    proc.start()
    return job_id


def tail(job_id: str, follow: bool = True, poll_interval: float = 0.2) -> Iterator[dict]:
    """Yield event dicts from a job's log file in order.

    If ``follow`` is True (SSE use case), yields ``None`` heartbeats while
    waiting for new lines and exits cleanly when the job status transitions
    to ``done`` / ``failed``. If False, yields current events and stops.
    """
    row = db.get_conn().execute(
        "SELECT log_path FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    if row is None:
        raise KeyError(job_id)
    log_path = Path(row["log_path"])

    pos = 0
    while True:
        try:
            with log_path.open("r") as f:
                f.seek(pos)
                while True:
                    line = f.readline()
                    if not line:
                        pos = f.tell()
                        break
                    line = line.rstrip("\n")
                    if not line:
                        pos = f.tell()
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # tolerate partial writes; will re-read on next pass
                        break
                    pos = f.tell()
        except FileNotFoundError:
            pass

        if not follow:
            return

        status = _job_status(job_id)
        if status in ("done", "failed", "zombie"):
            return
        time.sleep(poll_interval)


def get(job_id: str) -> dict | None:
    row = db.get_conn().execute(
        "SELECT * FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    return dict(row) if row else None


def recent(limit: int = 10, kind: str | None = None) -> list[dict]:
    sql = "SELECT * FROM jobs"
    args: tuple = ()
    if kind:
        sql += " WHERE kind = ?"
        args = (kind,)
    sql += " ORDER BY started_at DESC LIMIT ?"
    args = (*args, limit)
    rows = db.get_conn().execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def mark_zombies() -> None:
    """Called at Lab startup: jobs that were 'running' at shutdown are
    marked as 'zombie' so the UI can surface them for cleanup."""
    db.get_conn().execute(
        "UPDATE jobs SET status = 'zombie' WHERE status IN ('queued', 'running')"
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _job_status(job_id: str) -> str:
    row = db.get_conn().execute(
        "SELECT status FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    return row["status"] if row else "unknown"


def _job_main(job_id: str, log_path: str, target: Callable, config: dict, kwargs: dict) -> None:
    """Subprocess entry point. Wraps target with emit + status tracking.

    Lives in a separate process so MLX state is isolated and a crashed
    job can't take down the Lab server. Reopens its own DB connection
    because SQLite connections don't cross fork().
    """
    # Re-init the per-thread DB connection in the child process
    db._local.conn = None  # type: ignore[attr-defined]

    log_file = open(log_path, "w", buffering=1)  # line-buffered

    def emit(event: dict) -> None:
        log_file.write(json.dumps(event) + "\n")
        log_file.flush()
        # Update DB row's progress/message if event carries them
        progress = event.get("progress")
        message = event.get("message")
        if progress is not None or message is not None:
            sets = []
            args: list[Any] = []
            if progress is not None:
                sets.append("progress = ?")
                args.append(float(progress))
            if message is not None:
                sets.append("message = ?")
                args.append(str(message))
            args.append(job_id)
            db.get_conn().execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", args,
            )

    db.get_conn().execute(
        "UPDATE jobs SET status = 'running' WHERE id = ?", (job_id,),
    )
    emit({"type": "started", "ts": time.time()})

    try:
        target(emit, config, **kwargs)
        db.get_conn().execute(
            "UPDATE jobs SET status = 'done', ended_at = datetime('now'), progress = 1.0 "
            "WHERE id = ?",
            (job_id,),
        )
        emit({"type": "done", "ts": time.time()})
    except Exception as exc:  # noqa: BLE001 - subprocess boundary
        err = f"{type(exc).__name__}: {exc}"
        db.get_conn().execute(
            "UPDATE jobs SET status = 'failed', ended_at = datetime('now'), error = ? "
            "WHERE id = ?",
            (err, job_id),
        )
        emit({"type": "failed", "error": err, "ts": time.time()})
    finally:
        with contextlib.suppress(Exception):
            log_file.close()
