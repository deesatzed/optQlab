"""Machine state for the Lab strip — real memory and port probes."""

from __future__ import annotations

import socket
from typing import Any
from urllib.parse import urlparse

from . import db, jobs
from .config import ensure_lab_dirs
from . import fit_engine


def probe_tcp(host: str, port: int, timeout: float = 0.4) -> bool:
    if not host or not port:
        return False
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def memory_info() -> dict[str, Any]:
    total_gb, free_gb = fit_engine._memory_snapshot()
    used_gb = max(0.0, total_gb - free_gb)
    return {
        "total_ram_gb": round(total_gb, 3),
        "free_ram_gb": round(free_gb, 3),
        "used_ram_gb": round(used_gb, 3),
        "used_pct": round(100.0 * used_gb / total_gb, 1) if total_gb else 0.0,
        "source": "psutil.virtual_memory",
    }


def running_job_count() -> int:
    ensure_lab_dirs()
    try:
        conn = db.get_conn()
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status IN ('queued', 'running')"
        ).fetchone()
        return int(row["n"]) if row else 0
    except Exception:
        return 0


def recent_active_jobs(limit: int = 5) -> list[dict]:
    try:
        rows = jobs.recent(limit=limit)
        return [
            {
                "id": r["id"],
                "kind": r["kind"],
                "status": r["status"],
                "progress": r.get("progress"),
                "message": r.get("message"),
            }
            for r in rows
            if r.get("status") in ("queued", "running", "zombie")
        ]
    except Exception:
        return []


def machine_state(
    *,
    api_url: str,
    lab_port: int | None = None,
    model: str | None = None,
    api_reachable: bool | None = None,
    api_status: str | None = None,
    adapters: list | None = None,
    mtp_enabled: bool = False,
) -> dict[str, Any]:
    """Assemble strip payload. Prefer caller-supplied API probe when known."""
    parsed = urlparse(api_url if "://" in api_url else f"http://{api_url}")
    host = parsed.hostname or "127.0.0.1"
    serve_port = parsed.port or 8080
    lab_port = lab_port or 7860

    serve_up = probe_tcp(host, serve_port) if api_reachable is None else bool(api_reachable)
    lab_up = probe_tcp(host, lab_port)

    mem = memory_info()
    cal = fit_engine.load_calibration()

    return {
        "model": model,
        "adapters": adapters or [],
        "mtp_enabled": bool(mtp_enabled),
        "memory": mem,
        "ports": {
            "lab": {"port": lab_port, "healthy": lab_up},
            "serve": {
                "port": serve_port,
                "healthy": serve_up,
                "status": api_status or ("ready" if serve_up else "down"),
            },
        },
        "running_jobs": running_job_count(),
        "active_jobs": recent_active_jobs(),
        "fit_calibrated": bool(cal.get("reserved_gb")),
        "api_url": api_url,
    }
