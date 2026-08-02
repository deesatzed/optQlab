"""Sequential JobBus: admit light jobs immediately; serialize memory_heavy.

Wraps the process-based job runner so at most one ``memory_heavy`` job runs
at a time (quantize / finetune / dataset). ``light`` jobs start immediately
and may overlap a heavy job. Every submission dual-writes ``jobs`` + ``runs``
and emits lifecycle events on the spine event log.
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import threading
import time
from typing import Any, Callable

from . import db, events, jobs
from .config import ensure_lab_dirs


TERMINAL = frozenset({"done", "failed", "cancelled", "zombie"})

# ---------------------------------------------------------------------------
# In-process admission state (parent Lab process only)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_processes: dict[str, mp.Process] = {}
_pending_heavy: list[dict[str, Any]] = []
_running_heavy_id: str | None = None


def _reset_for_tests() -> None:
    """Clear admission state and terminate any live children. Test-only."""
    global _running_heavy_id
    with _lock:
        procs = list(_processes.items())
        _processes.clear()
        _pending_heavy.clear()
        _running_heavy_id = None
    for _jid, proc in procs:
        if proc.is_alive():
            with contextlib.suppress(Exception):
                proc.terminate()
            with contextlib.suppress(Exception):
                proc.join(timeout=2)
            if proc.is_alive():
                with contextlib.suppress(Exception):
                    proc.kill()
                    proc.join(timeout=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def submit(
    kind: str,
    target: Callable[..., None],
    config: dict,
    *,
    resource_class: str = "memory_heavy",
    workspace_id: str | None = None,
    build_id: str | None = None,
    **kwargs,
) -> str:
    """Enqueue or start a job. Returns job_id (same format as jobs.new_job_id).

    memory_heavy: at most one running at a time; others stay status='queued'
    until pump() starts them FIFO.
    light: start immediately (can run alongside a heavy job).
    """
    if resource_class not in ("memory_heavy", "light"):
        raise ValueError(
            f"resource_class must be 'memory_heavy' or 'light', got {resource_class!r}"
        )

    paths = ensure_lab_dirs()
    job_id = jobs.new_job_id()
    log_path = paths.jobs_dir / f"{job_id}.log"
    log_path.touch()
    config_json = json.dumps(config)

    conn = db.get_conn()
    conn.execute(
        """
        INSERT INTO jobs (id, kind, status, config_json, log_path)
        VALUES (?, ?, 'queued', ?, ?)
        """,
        (job_id, kind, config_json, str(log_path)),
    )
    conn.execute(
        """
        INSERT INTO runs (id, kind, status, workspace_id, build_id, config_json, progress)
        VALUES (?, ?, 'queued', ?, ?, ?, 0.0)
        """,
        (job_id, kind, workspace_id, build_id, config_json),
    )
    events.append(
        type="job.queued",
        entity_type="job",
        entity_id=job_id,
        payload={"kind": kind, "resource_class": resource_class},
        workspace_id=workspace_id,
    )

    if resource_class == "light":
        _launch(
            job_id=job_id,
            target=target,
            config=config,
            kwargs=kwargs,
            workspace_id=workspace_id,
            build_id=build_id,
            is_heavy=False,
            log_path=str(log_path),
        )
    else:
        with _lock:
            _pending_heavy.append(
                {
                    "job_id": job_id,
                    "target": target,
                    "config": config,
                    "kwargs": kwargs,
                    "workspace_id": workspace_id,
                    "build_id": build_id,
                    "log_path": str(log_path),
                }
            )
        pump()

    return job_id


def cancel(job_id: str) -> bool:
    """Terminate process if running; set status cancelled; event job.cancelled.

    Return True if cancelled, False if not found or already terminal.
    """
    global _running_heavy_id
    with _lock:
        for i, item in enumerate(_pending_heavy):
            if item["job_id"] == job_id:
                _pending_heavy.pop(i)
                return _mark_cancelled(job_id)
        proc = _processes.get(job_id)
        heavy_id = _running_heavy_id

    row = jobs.get(job_id)
    if row is None:
        return False
    if row["status"] in TERMINAL:
        return False

    # Process may still be launching — brief wait for registry entry.
    if proc is None:
        for _ in range(40):
            time.sleep(0.05)
            with _lock:
                proc = _processes.get(job_id)
            if proc is not None:
                break
            row = jobs.get(job_id)
            if row is None or row["status"] in TERMINAL:
                return False
            # Still queued with no process and not pending → mark cancelled.
            if row["status"] == "queued" and proc is None:
                with _lock:
                    still_pending = any(
                        p["job_id"] == job_id for p in _pending_heavy
                    )
                if not still_pending:
                    # Could be mid-launch; one more short wait then cancel.
                    time.sleep(0.1)
                    with _lock:
                        proc = _processes.get(job_id)
                    if proc is None:
                        ok = _mark_cancelled(job_id)
                        if ok and heavy_id == job_id:
                            with _lock:
                                if _running_heavy_id == job_id:
                                    _running_heavy_id = None
                            pump()
                        return ok

    if proc is not None and proc.is_alive():
        with contextlib.suppress(Exception):
            proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            with contextlib.suppress(Exception):
                proc.kill()
            proc.join(timeout=2)

    row = jobs.get(job_id)
    if row is None:
        return False
    if row["status"] in TERMINAL:
        return row["status"] == "cancelled"

    ok = _mark_cancelled(job_id)
    with _lock:
        _processes.pop(job_id, None)
        if _running_heavy_id == job_id:
            _running_heavy_id = None
    if ok:
        pump()
    return ok


def pump() -> None:
    """Start next queued heavy job if none is running.

    Called by submit and when a heavy job ends (watcher thread).
    """
    global _running_heavy_id
    item: dict[str, Any] | None = None
    with _lock:
        if _running_heavy_id is not None:
            proc = _processes.get(_running_heavy_id)
            if proc is not None and proc.is_alive():
                return
            # Slot free (finished, cancelled, or never registered).
            _running_heavy_id = None

        if not _pending_heavy:
            return

        item = _pending_heavy.pop(0)
        # Reserve so concurrent pump/submit won't double-admit.
        _running_heavy_id = item["job_id"]

    assert item is not None
    try:
        _launch(
            job_id=item["job_id"],
            target=item["target"],
            config=item["config"],
            kwargs=item["kwargs"],
            workspace_id=item["workspace_id"],
            build_id=item["build_id"],
            is_heavy=True,
            log_path=item["log_path"],
            reserved=True,
        )
    except Exception:
        with _lock:
            if _running_heavy_id == item["job_id"]:
                _running_heavy_id = None
        raise


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _launch(
    *,
    job_id: str,
    target: Callable,
    config: dict,
    kwargs: dict,
    workspace_id: str | None,
    build_id: str | None,
    is_heavy: bool,
    log_path: str,
    reserved: bool = False,
) -> None:
    global _running_heavy_id
    proc = mp.Process(
        target=_bus_job_main,
        args=(job_id, log_path, target, config, kwargs, workspace_id),
        name=f"lab-job-{job_id}",
        daemon=True,
    )
    proc.start()
    with _lock:
        _processes[job_id] = proc
        if is_heavy:
            if not reserved:
                _running_heavy_id = job_id
            elif _running_heavy_id != job_id:
                _running_heavy_id = job_id

    t = threading.Thread(
        target=_watch_process,
        args=(job_id, proc, is_heavy),
        name=f"lab-job-watch-{job_id}",
        daemon=True,
    )
    t.start()


def _watch_process(job_id: str, proc: mp.Process, is_heavy: bool) -> None:
    global _running_heavy_id
    proc.join()
    with _lock:
        cur = _processes.get(job_id)
        if cur is proc:
            _processes.pop(job_id, None)
        if is_heavy and _running_heavy_id == job_id:
            _running_heavy_id = None
    if is_heavy:
        pump()


def _mark_cancelled(job_id: str) -> bool:
    row = jobs.get(job_id)
    if row is None:
        return False
    if row["status"] in TERMINAL:
        return row["status"] == "cancelled"

    workspace_id = _run_workspace(job_id)
    db.get_conn().execute(
        """
        UPDATE jobs SET status = 'cancelled', ended_at = datetime('now')
        WHERE id = ? AND status NOT IN ('done', 'failed', 'cancelled', 'zombie')
        """,
        (job_id,),
    )
    db.get_conn().execute(
        """
        UPDATE runs SET status = 'cancelled', ended_at = datetime('now')
        WHERE id = ? AND status NOT IN ('done', 'failed', 'cancelled', 'zombie')
        """,
        (job_id,),
    )
    # Confirm we actually flipped it (or it was already cancelled by us).
    row = jobs.get(job_id)
    if row is None or row["status"] != "cancelled":
        return False
    events.append(
        type="job.cancelled",
        entity_type="job",
        entity_id=job_id,
        payload={"kind": row.get("kind")},
        workspace_id=workspace_id,
    )
    return True


def _run_workspace(job_id: str) -> str | None:
    row = db.get_conn().execute(
        "SELECT workspace_id FROM runs WHERE id = ?", (job_id,)
    ).fetchone()
    return row["workspace_id"] if row else None


def _bus_job_main(
    job_id: str,
    log_path: str,
    target: Callable,
    config: dict,
    kwargs: dict,
    workspace_id: str | None,
) -> None:
    """Subprocess entry: emit → JSONL + jobs/runs + throttled job.progress events."""
    # Fresh DB connection in the child (connections don't cross fork/spawn).
    db._local.conn = None  # type: ignore[attr-defined]

    log_file = open(log_path, "w", buffering=1)  # line-buffered
    last_progress_event = 0.0

    def emit(event: dict) -> None:
        nonlocal last_progress_event
        log_file.write(json.dumps(event) + "\n")
        log_file.flush()

        progress = event.get("progress")
        message = event.get("message")
        if progress is None and message is None:
            return

        sets: list[str] = []
        args: list[Any] = []
        if progress is not None:
            sets.append("progress = ?")
            args.append(float(progress))
        if message is not None:
            sets.append("message = ?")
            args.append(str(message))
        set_sql = ", ".join(sets)
        args_jobs = [*args, job_id]
        args_runs = [*args, job_id]
        conn = db.get_conn()
        conn.execute(f"UPDATE jobs SET {set_sql} WHERE id = ?", args_jobs)
        conn.execute(f"UPDATE runs SET {set_sql} WHERE id = ?", args_runs)

        now = time.time()
        # Throttle progress events; always emit on 0 or 1.0 milestones.
        should_event = (
            progress is not None
            and (
                float(progress) <= 0.0
                or float(progress) >= 1.0
                or (now - last_progress_event) >= 0.5
            )
        )
        if should_event or (message is not None and progress is None and (now - last_progress_event) >= 0.5):
            events.append(
                type="job.progress",
                entity_type="job",
                entity_id=job_id,
                payload={
                    k: v
                    for k, v in {
                        "progress": float(progress) if progress is not None else None,
                        "message": str(message) if message is not None else None,
                    }.items()
                    if v is not None
                },
                workspace_id=workspace_id,
            )
            last_progress_event = now

    conn = db.get_conn()
    conn.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (job_id,))
    conn.execute("UPDATE runs SET status = 'running' WHERE id = ?", (job_id,))
    events.append(
        type="job.started",
        entity_type="job",
        entity_id=job_id,
        payload={},
        workspace_id=workspace_id,
    )
    emit({"type": "started", "ts": time.time()})

    try:
        target(emit, config, **kwargs)
        # Don't overwrite a cancel that landed while we were finishing.
        row = conn.execute(
            "SELECT status FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row and row["status"] == "cancelled":
            return
        conn.execute(
            "UPDATE jobs SET status = 'done', ended_at = datetime('now'), progress = 1.0 "
            "WHERE id = ? AND status = 'running'",
            (job_id,),
        )
        conn.execute(
            "UPDATE runs SET status = 'done', ended_at = datetime('now'), progress = 1.0 "
            "WHERE id = ? AND status = 'running'",
            (job_id,),
        )
        events.append(
            type="job.done",
            entity_type="job",
            entity_id=job_id,
            payload={},
            workspace_id=workspace_id,
        )
        emit({"type": "done", "ts": time.time()})
    except Exception as exc:  # noqa: BLE001 - subprocess boundary
        err = f"{type(exc).__name__}: {exc}"
        row = conn.execute(
            "SELECT status FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row and row["status"] == "cancelled":
            return
        conn.execute(
            "UPDATE jobs SET status = 'failed', ended_at = datetime('now'), error = ? "
            "WHERE id = ? AND status = 'running'",
            (err, job_id),
        )
        conn.execute(
            "UPDATE runs SET status = 'failed', ended_at = datetime('now'), error = ? "
            "WHERE id = ? AND status = 'running'",
            (err, job_id),
        )
        events.append(
            type="job.failed",
            entity_type="job",
            entity_id=job_id,
            payload={"error": err},
            workspace_id=workspace_id,
        )
        emit({"type": "failed", "error": err, "ts": time.time()})
    finally:
        with contextlib.suppress(Exception):
            log_file.close()
