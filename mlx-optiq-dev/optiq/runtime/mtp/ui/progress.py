"""Model-load progress indicator.

Uses plain terminal lines plus a tiny heartbeat subprocess while MLX maps model
weights. A separate process keeps progress visible even if the main Python
thread is busy inside model-loading code.

Usage::

    with ModelLoadProgress("Loading model", quiet=False) as progress:
        progress.set_subtitle(f"resolving {repo}")
        runtime = load(model_path, mtp=True)
        progress.set_subtitle("ready")
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from typing import Any


class ModelLoadProgress:
    """Plain heartbeat for long-running model loads."""

    def __init__(
        self,
        title: str = "Loading model",
        *,
        quiet: bool = False,
        console: Any | None = None,
        interval_s: float = 10.0,
    ) -> None:
        self._title = title
        self._quiet = quiet
        self._console = console
        self._interval_s = float(interval_s)
        self._heartbeat_proc: subprocess.Popen[str] | None = None
        self._fallback_thread: threading.Thread | None = None
        self._fallback_stop = threading.Event()
        self._started_at: float = 0.0

    def __enter__(self) -> "ModelLoadProgress":
        self._started_at = time.perf_counter()
        if self._quiet:
            return self
        self._write("[mtplx] Model load in progress (this may take a minute).")
        self._start_heartbeat()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._heartbeat_proc is not None:
            proc = self._heartbeat_proc
            self._heartbeat_proc = None
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
        if self._fallback_thread is not None:
            self._fallback_stop.set()
            self._fallback_thread.join(timeout=1.0)
            self._fallback_thread = None

    def set_subtitle(self, subtitle: str) -> None:
        if self._quiet:
            return
        text = str(subtitle).strip()
        if not text or text == "ready":
            return
        self._write(f"[mtplx] {self._title}: {text}")

    def _write(self, text: str) -> None:
        if self._console is not None:
            try:
                self._console.print(text, highlight=False)
                return
            except Exception:
                pass
        print(text, flush=True)

    def _start_heartbeat(self) -> None:
        if self._quiet:
            return
        script = (
            "import os,signal,sys,time\n"
            "interval=float(sys.argv[1])\n"
            "parent=int(sys.argv[2])\n"
            "running=True\n"
            "def stop(_signum,_frame):\n"
            "    global running\n"
            "    running=False\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "elapsed=0.0\n"
            "while running:\n"
            "    time.sleep(interval)\n"
            "    if not running:\n"
            "        break\n"
            "    if parent and os.getppid() != parent:\n"
            "        break\n"
            "    elapsed += interval\n"
            "    print(f'[mtplx] Model still loading... {elapsed:.0f}s elapsed', flush=True)\n"
        )
        try:
            self._heartbeat_proc = subprocess.Popen(
                [sys.executable, "-c", script, str(self._interval_s), str(os.getpid())],
                stdout=None,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                text=True,
            )
        except Exception:
            self._start_fallback_thread()

    def _start_fallback_thread(self) -> None:
        stop = self._fallback_stop
        interval = self._interval_s

        def heartbeat() -> None:
            elapsed = 0.0
            while not stop.wait(interval):
                elapsed += interval
                print(f"[mtplx] Model still loading... {int(elapsed)}s elapsed", flush=True)

        thread = threading.Thread(target=heartbeat, name="mtplx-load-heartbeat", daemon=True)
        thread.start()
        self._fallback_thread = thread
