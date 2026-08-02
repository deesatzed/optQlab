"""Single-line live progress for Hugging Face model downloads.

Replaces the dual-stream chaos that quickstart used to render — Hugging
Face tqdm bars on stderr fighting MTPLX's text heartbeat lines on stdout.
This module owns one terminal line and updates it in place via
``rich.progress``. When ``rich`` is unavailable (minimal venv, JSON mode,
piped stdout) it falls back to plain-stdlib status prints.

Usage:

    progress = RichDownloadProgress(repo_id="org/repo", total_bytes=29.1e9)
    progress.start(started_bytes=10_400_000_000)
    progress.update(current_bytes=12_000_000_000)
    progress.update(current_bytes=14_000_000_000)
    progress.complete(final_bytes=29_100_000_000)

The class is reentrancy-safe: ``stop``/``complete`` are idempotent and
``update`` is a no-op after a stop. It does not start its own polling
thread — the caller is expected to drive ``update`` from the outside,
typically from ``mtplx.hf_loader`` heartbeat callbacks.
"""

from __future__ import annotations

import time
from typing import Any


def _format_bytes(size: float | int | None) -> str:
    if not isinstance(size, (int, float)):
        return "? B"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1000.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1000.0
    return f"{value:.1f} TB"


class RichDownloadProgress:
    """One-line live progress for a Hugging Face download.

    The instance is created cheaply and stays inert until ``start`` is
    called. After ``start`` it owns one terminal line until ``complete``
    or ``stop`` is invoked. Every ``update`` mutates that single line.
    """

    def __init__(self, repo_id: str, total_bytes: int | None = None) -> None:
        self.repo_id = repo_id
        self.total_bytes = int(total_bytes) if total_bytes else None
        self._fallback = False
        self._progress: Any = None
        self._task_id: Any = None
        self._console: Any = None
        self._started = False
        self._stopped = False
        self._last_log_at: float = 0.0
        self._last_log_bytes: int = 0
        try:
            from rich.console import Console

            self._console = Console()
        except ImportError:
            self._fallback = True
        # Rich handles non-terminal output gracefully — it emits static
        # final frames instead of carriage-return animations — so we don't
        # disable it just because stdout is captured. The fallback path
        # is reserved for environments where ``rich`` itself can't import.

    # ---------- lifecycle --------------------------------------------------
    def start(self, *, started_bytes: int = 0) -> None:
        if self._started or self._stopped:
            return
        self._started = True
        self._last_log_at = time.monotonic()
        self._last_log_bytes = int(started_bytes or 0)

        if self._fallback:
            self._fallback_start(started_bytes=started_bytes)
            return

        try:
            from rich.progress import (
                BarColumn,
                DownloadColumn,
                Progress,
                TextColumn,
                TimeRemainingColumn,
                TransferSpeedColumn,
            )
        except ImportError:
            self._fallback = True
            self._fallback_start(started_bytes=started_bytes)
            return

        try:
            columns = [
                TextColumn("[bold cyan]downloading[/]  [bold]{task.description}[/]"),
                BarColumn(bar_width=24),
                TextColumn("[dim]{task.percentage:>5.1f}%[/]"),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(elapsed_when_finished=True),
            ]
            self._progress = Progress(
                *columns,
                console=self._console,
                transient=False,
                refresh_per_second=8,
                expand=False,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                description=self.repo_id,
                total=self.total_bytes,
                completed=int(started_bytes or 0),
            )
        except Exception:
            # Any rich init hiccup → fall back gracefully rather than
            # leaving the user with no progress UX at all.
            self._progress = None
            self._task_id = None
            self._fallback = True
            self._fallback_start(started_bytes=started_bytes)

    def update(self, current_bytes: int) -> None:
        if not self._started or self._stopped:
            return
        current_bytes = int(current_bytes or 0)
        if self._fallback:
            self._fallback_update(current_bytes)
            return
        if self._progress is None or self._task_id is None:
            return
        try:
            kwargs = {"completed": current_bytes}
            # If the total grew (rare — repo metadata sometimes lags) extend
            # the bar so it doesn't look "over 100%".
            if self.total_bytes is not None and current_bytes > self.total_bytes:
                kwargs["total"] = current_bytes
            self._progress.update(self._task_id, **kwargs)
        except Exception:
            # Rich can blow up if the console is closed mid-update; degrade.
            self._fallback_update(current_bytes)

    def complete(self, *, final_bytes: int | None = None) -> None:
        self.stop(final_bytes=final_bytes)

    def stop(self, *, final_bytes: int | None = None) -> None:
        if self._stopped:
            return
        self._stopped = True
        if final_bytes is not None and not self._fallback and self._progress is not None and self._task_id is not None:
            try:
                kwargs = {"completed": int(final_bytes)}
                if self.total_bytes is None:
                    kwargs["total"] = int(final_bytes)
                self._progress.update(self._task_id, **kwargs)
            except Exception:
                pass
        if self._progress is not None:
            try:
                self._progress.stop()
            except Exception:
                pass
            self._progress = None
            self._task_id = None
        if self._fallback and self._started:
            label = _format_bytes(final_bytes if final_bytes is not None else self._last_log_bytes)
            print(f"  download complete: {self.repo_id} ({label})", flush=True)

    # ---------- stdlib fallback --------------------------------------------
    def _fallback_start(self, *, started_bytes: int) -> None:
        if started_bytes and started_bytes > 0:
            print(
                f"  resuming {self.repo_id} ({_format_bytes(started_bytes)} of "
                f"{_format_bytes(self.total_bytes) if self.total_bytes else 'unknown total'} on disk)",
                flush=True,
            )
        else:
            total_label = _format_bytes(self.total_bytes) if self.total_bytes else "unknown total"
            print(
                f"  downloading {self.repo_id} ({total_label})",
                flush=True,
            )

    def _fallback_update(self, current_bytes: int) -> None:
        # Coalesce: only print every ~5s in the fallback path so non-tty
        # logs don't get spammed once per 0.4s heartbeat.
        now = time.monotonic()
        if now - self._last_log_at < 5.0:
            return
        delta_t = now - self._last_log_at
        delta_b = current_bytes - self._last_log_bytes
        speed = delta_b / delta_t if delta_t > 0 else 0
        if self.total_bytes:
            pct = 100.0 * current_bytes / max(self.total_bytes, 1)
            print(
                f"  {self.repo_id}: {pct:5.1f}%  "
                f"{_format_bytes(current_bytes)} / {_format_bytes(self.total_bytes)}  "
                f"@ {_format_bytes(speed)}/s",
                flush=True,
            )
        else:
            print(
                f"  {self.repo_id}: {_format_bytes(current_bytes)} downloaded  "
                f"@ {_format_bytes(speed)}/s",
                flush=True,
            )
        self._last_log_at = now
        self._last_log_bytes = current_bytes


def from_progress_event_callback(
    *,
    progress: RichDownloadProgress,
    fallback_printer: Any = None,
):
    """Adapt a ``RichDownloadProgress`` instance to the event-callback API
    used by ``mtplx.hf_loader.pull_model``.

    Returns a callable that consumes the event dicts emitted by
    ``pull_model`` and forwards them to the rich progress bar. Events with
    keys it doesn't understand are silently dropped — never break the
    download because of a UX hiccup.
    """

    def emit(event: dict[str, Any]) -> None:
        try:
            kind = event.get("event")
            size = int(event.get("size_bytes") or 0)
            if kind in ("start", "resume"):
                # The total may have arrived in the event payload (queried by
                # pull_model from HfApi). Adopt it if we don't have one yet.
                total = event.get("total_bytes")
                if total and not progress.total_bytes:
                    progress.total_bytes = int(total)
                progress.start(started_bytes=size)
            elif kind == "progress":
                progress.update(size)
            elif kind == "complete":
                progress.complete(final_bytes=size)
        except Exception as exc:  # noqa: BLE001
            # Last-resort fallback so a UX glitch never breaks the download.
            if callable(fallback_printer):
                try:
                    fallback_printer(f"  [progress error: {exc}]")
                except Exception:
                    pass

    return emit


__all__ = [
    "RichDownloadProgress",
    "from_progress_event_callback",
]
