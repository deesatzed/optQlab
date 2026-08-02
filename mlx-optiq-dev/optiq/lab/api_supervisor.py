"""Subprocess supervisor for the model-serving API.

Hot-swap workflow:

    sup = ApiSupervisor(host="127.0.0.1", port=8080)
    sup.start(model="<path>", mtp=False)        # boot
    sup.is_alive()                              # → True
    sup.state()                                 # → {model, mtp, status, ...}
    sup.restart(model="<new>", mtp=True)        # SIGTERM old, spawn new
    sup.stop()                                  # cleanup at Lab exit

The supervisor lives in the Lab parent process. Children are spawned via
``python -m optiq.lab.api_runner`` so they're isolated — a model that
crashes the API doesn't take the Lab UI down with it.

Thread-safe (callers can hit /api/server/apply from any Flask request).
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


log = logging.getLogger(__name__)


Status = Literal["idle", "starting", "ready", "stopping", "error", "crashed"]


@dataclass
class ApiState:
    status: Status
    model: str | None
    mtp_enabled: bool
    mtp_depth: int
    drafter_id: str | None
    prompt_cache_bytes: int | None
    # User-set sampler overrides currently applied to the child mlx_lm.server.
    # ``None`` for a given key means "use the model's recommended default from
    # generation_config.json"; the api_runner injects those at child startup.
    sampler: dict | None
    # LoRA adapter paths currently mounted on the served model. Empty = no
    # adapter. One = classic single-adapter boot. Two or more = OptiQ
    # mounted-LoRA mode (per-request switching via the 'adapters' body field;
    # adapter names are the directory basenames).
    adapters: list[str]
    pid: int | None
    uptime_s: float | None
    last_error: str | None


class ApiSupervisor:
    """Manages a single mlx_lm.server child process for the Lab."""

    # How long to wait for /v1/models to start returning 200 after spawn.
    # Generous: a big quant (or one streamed from SSD) can take minutes to
    # load. The probe already fails FAST if the child process dies, so a long
    # cap only affects a genuinely-slow load, not a broken one. Override with
    # OPTIQ_LAB_BOOT_TIMEOUT.
    BOOT_TIMEOUT_S = int(os.environ.get("OPTIQ_LAB_BOOT_TIMEOUT", "900"))
    # How long to wait between SIGTERM and SIGKILL.
    STOP_GRACE_S = 6
    # Auto-restart parameters. We respawn the child up to N times with
    # exponential backoff when it crashes unexpectedly; past N we give up
    # so a permanently-broken model (wrong arch, OOM) doesn't loop forever.
    MAX_AUTO_RESTARTS = 3
    AUTO_RESTART_BACKOFF_S = (2, 5, 15)
    WATCHDOG_POLL_S = 1.0

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        log_dir: Path | None = None,
        clear_cache_threshold_bytes: int | None = None,
        auto_restart: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.log_dir = log_dir
        self.clear_cache_threshold_bytes = clear_cache_threshold_bytes
        self.auto_restart = auto_restart
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._model: str | None = None
        self._mtp_enabled = False
        self._mtp_depth = 0
        self._drafter_id: str | None = None
        self._prompt_cache_bytes: int | None = None
        self._sampler: dict | None = None
        self._adapters: list[str] = []
        self._stream_experts: str = "auto"
        self._status: Status = "idle"
        self._started_at: float | None = None
        self._last_error: str | None = None
        # Watchdog state
        self._restart_count = 0
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        # Install cleanup so the child doesn't orphan when the Lab process
        # exits. start_new_session=True keeps SIGTERM-to-parent from leaking
        # into the child by accident, but it also means SIGTERM-to-parent
        # leaves the child alive. The atexit + signal handlers paper over
        # this by explicitly killing the child during shutdown.
        self._install_shutdown_hooks()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def state(self) -> ApiState:
        with self._lock:
            uptime = (time.time() - self._started_at) if (self._started_at and self._status == "ready") else None
            # If we thought the child was running, verify it actually is.
            if self._proc and self._status in ("ready", "starting") and self._proc.poll() is not None:
                self._status = "crashed"
                self._last_error = f"child exited with code {self._proc.returncode}"
            return ApiState(
                status=self._status,
                model=self._model,
                mtp_enabled=self._mtp_enabled,
                mtp_depth=self._mtp_depth,
                drafter_id=self._drafter_id,
                prompt_cache_bytes=self._prompt_cache_bytes,
                sampler=dict(self._sampler) if self._sampler else None,
                adapters=list(self._adapters),
                pid=self._proc.pid if self._proc else None,
                uptime_s=uptime,
                last_error=self._last_error,
            )

    def is_alive(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def start(self, *, model: str, mtp: bool = False, mtp_depth: int = 2,
              drafter_id: str | None = None,
              prompt_cache_bytes: int | None = None,
              sampler: dict | None = None,
              adapters: list[str] | None = None,
              stream_experts: str = "auto") -> None:
        """Boot the API server. Raises if a child is already running
        (use restart() to swap). ``sampler`` is a partial dict of
        ``{temp, top_p, top_k, min_p}``; any key set overrides the
        model's recommended default; absent keys fall through to the
        model's ``generation_config.json``.

        Pass ``drafter_id`` to enable -assistant drafter speculative
        decoding (Gemma-4 family). Mutually exclusive with ``mtp``.

        Pass ``adapters`` to mount one or more LoRA adapters at startup.
        With a single adapter the classic mlx-lm single-adapter boot
        runs; with two or more, OptiQ's mounted-LoRA mode kicks in and
        clients pick an adapter per request via the 'adapters' field."""
        if mtp and drafter_id:
            raise RuntimeError("mtp and drafter_id are mutually exclusive")
        with self._lock:
            if self._proc and self._proc.poll() is None:
                raise RuntimeError("API server already running; use restart() to swap")
            self._restart_count = 0
            self._stream_experts = stream_experts
            self._spawn_locked(
                model=model, mtp=mtp, mtp_depth=mtp_depth,
                drafter_id=drafter_id,
                prompt_cache_bytes=prompt_cache_bytes,
                sampler=sampler,
                adapters=adapters or [],
                stream_experts=stream_experts,
            )
        self._start_watchdog()

    def restart(self, *, model: str, mtp: bool = False, mtp_depth: int = 2,
                drafter_id: str | None = None,
                prompt_cache_bytes: int | None = None,
                sampler: dict | None = None,
                adapters: list[str] | None = None,
                stream_experts: str = "auto") -> None:
        """Atomic swap: stop the old child + spawn a new one."""
        if mtp and drafter_id:
            raise RuntimeError("mtp and drafter_id are mutually exclusive")
        self.stop()
        with self._lock:
            self._restart_count = 0
            self._stream_experts = stream_experts
            self._spawn_locked(
                model=model, mtp=mtp, mtp_depth=mtp_depth,
                drafter_id=drafter_id,
                prompt_cache_bytes=prompt_cache_bytes,
                sampler=sampler,
                adapters=adapters or [],
                stream_experts=stream_experts,
            )
        self._start_watchdog()

    def stop(self) -> None:
        """Graceful SIGTERM → SIGKILL fallback. Also frees the port.

        Sends the signal to the entire process group so any helper
        processes the child spawned (mlx workers, anthropic_server
        threads, etc.) die with the parent. start_new_session=True at
        spawn time gave the child its own session, so the pgid == pid."""
        # Stop the watchdog first so it can't race to respawn the child
        # while we're tearing it down.
        self._watchdog_stop.set()

        with self._lock:
            proc = self._proc
            if proc is None:
                self._status = "idle"
                return
            self._status = "stopping"

        if proc.poll() is None:
            # Try TERM at process-group level first; falls back to plain
            # terminate() if killpg fails for any reason.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                proc.terminate()
            try:
                proc.wait(timeout=self.STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
                proc.wait(timeout=2)

        with self._lock:
            self._proc = None
            self._model = None
            self._mtp_enabled = False
            self._mtp_depth = 0
            self._drafter_id = None
            self._prompt_cache_bytes = None
            self._sampler = None
            self._started_at = None
            self._status = "idle"

        # Wait for the port to actually free up so the next start() can bind.
        for _ in range(20):
            if self._port_free():
                break
            time.sleep(0.2)

    # ------------------------------------------------------------------
    # Shutdown hooks + watchdog
    # ------------------------------------------------------------------

    def _install_shutdown_hooks(self) -> None:
        """Best-effort cleanup of the child when the Lab process exits.

        ``atexit`` covers normal termination (Ctrl-C raises KeyboardInterrupt
        which Flask catches and exits cleanly) and tools that call
        ``sys.exit()``. The SIGTERM/SIGINT handlers cover the case where a
        signal arrives outside a Python try/except scope (kill from another
        terminal, `optiq lab` being killed by its parent shell, etc.).
        SIGKILL is uncatchable; a hard kill -9 will still orphan the child.
        """
        atexit.register(self._on_shutdown)

        def _handler(signum, frame):
            log.info("API supervisor: caught signal %d, cleaning up child", signum)
            self._on_shutdown()
            # Re-raise as the default disposition so the parent actually exits.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                # signal.signal raises if we're not on the main thread.
                # In Flask debug mode this happens; the atexit hook still
                # fires so we're not totally exposed.
                pass

    def _on_shutdown(self) -> None:
        """Idempotent cleanup. Called from atexit + signal handlers."""
        self._watchdog_stop.set()
        try:
            self.stop()
        except Exception as e:
            log.warning("API supervisor: stop() during shutdown raised: %s", e)

    def _start_watchdog(self) -> None:
        """Start (or restart) the watchdog thread that polls the child and
        auto-respawns it on crash. Idempotent."""
        if not self.auto_restart:
            return
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True,
            name="api-supervisor-watchdog",
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        """Poll the child every ``WATCHDOG_POLL_S``. If it has exited
        unexpectedly (we weren't in 'stopping' / 'idle'), respawn it with
        backoff up to ``MAX_AUTO_RESTARTS`` times. Stops once the supervisor
        is shutting down (``_watchdog_stop`` set)."""
        while not self._watchdog_stop.is_set():
            time.sleep(self.WATCHDOG_POLL_S)
            with self._lock:
                proc = self._proc
                status = self._status
                model = self._model
                mtp = self._mtp_enabled
                mtp_depth = self._mtp_depth
                drafter_id = self._drafter_id
                pc_bytes = self._prompt_cache_bytes
                sampler = self._sampler
                adapters = list(self._adapters)

            # Skip if the supervisor is idle, stopping, or already restarting.
            if proc is None or model is None or status in ("idle", "stopping"):
                continue

            if proc.poll() is None:
                # Child still running. If it's ready, reset the restart counter
                # so we have a full budget for future crashes.
                if status == "ready":
                    self._restart_count = 0
                continue

            # Child exited. If we were in a healthy state, it crashed; try
            # to respawn.
            if self._restart_count >= self.MAX_AUTO_RESTARTS:
                log.error(
                    "API supervisor: child crashed %d times; giving up "
                    "(model=%s, rc=%s)",
                    self._restart_count, model, proc.returncode,
                )
                with self._lock:
                    self._status = "crashed"
                    self._last_error = (
                        f"crashed {self._restart_count} times in a row "
                        f"(last exit code {proc.returncode}). Check api.log."
                    )
                return

            delay = self.AUTO_RESTART_BACKOFF_S[
                min(self._restart_count, len(self.AUTO_RESTART_BACKOFF_S) - 1)
            ]
            log.warning(
                "API supervisor: child exited with code %s; respawn #%d in %ds",
                proc.returncode, self._restart_count + 1, delay,
            )
            self._restart_count += 1
            time.sleep(delay)
            try:
                with self._lock:
                    self._spawn_locked(
                        model=model, mtp=mtp, mtp_depth=mtp_depth,
                        drafter_id=drafter_id,
                        prompt_cache_bytes=pc_bytes, sampler=sampler,
                        adapters=adapters,
                    )
            except Exception as e:
                log.error("API supervisor: respawn failed: %s", e)
                with self._lock:
                    self._status = "crashed"
                    self._last_error = f"respawn failed: {e}"
                return

    # ------------------------------------------------------------------
    # Internals (lock must be held)
    # ------------------------------------------------------------------

    def _spawn_locked(self, *, model: str, mtp: bool, mtp_depth: int,
                      drafter_id: str | None = None,
                      prompt_cache_bytes: int | None = None,
                      sampler: dict | None = None,
                      adapters: list[str] | None = None,
                      stream_experts: str = "auto") -> None:
        argv = [
            sys.executable, "-m", "optiq.lab.api_runner",
            "--model", model,
            "--host", self.host,
            "--port", str(self.port),
            "--stream-experts", stream_experts,
        ]
        if self.clear_cache_threshold_bytes is not None:
            argv += ["--clear-cache-threshold-bytes",
                     str(self.clear_cache_threshold_bytes)]
        if mtp:
            argv += ["--mtp", "--mtp-depth", str(mtp_depth)]
        elif drafter_id:
            argv += ["--drafter", drafter_id]
        if prompt_cache_bytes is not None:
            # Forwarded straight to mlx_lm.server via api_runner's argv
            # passthrough. inject_prompt_cache_bytes() in api_runner
            # respects this (won't overwrite if already present).
            argv += ["--prompt-cache-bytes", str(prompt_cache_bytes)]
        # Forward each LoRA adapter path. api_runner picks classic
        # single-adapter mode for one entry and mounted-LoRA mode for
        # two or more.
        for ad in (adapters or []):
            argv += ["--adapter", ad]
        # Forward sampler overrides as the same CLI flags mlx_lm.server
        # accepts. api_runner's merge_into_argv leaves these alone and
        # only injects model defaults for keys we didn't pass.
        if sampler:
            for cli_flag, key in (
                ("--temp", "temp"),
                ("--top-p", "top_p"),
                ("--top-k", "top_k"),
                ("--min-p", "min_p"),
            ):
                v = sampler.get(key)
                if v is not None:
                    argv += [cli_flag, str(v)]

        log_path: Path | None = None
        out_handle = subprocess.PIPE
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / "api.log"
            out_handle = open(log_path, "wb")

        # New session so SIGTERM to parent doesn't kill child accidentally,
        # and the child doesn't receive parent's stdin.
        self._proc = subprocess.Popen(
            argv,
            stdout=out_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._model = model
        self._mtp_enabled = mtp
        self._mtp_depth = mtp_depth if mtp else 0
        self._drafter_id = drafter_id if (drafter_id and not mtp) else None
        self._prompt_cache_bytes = prompt_cache_bytes
        self._sampler = dict(sampler) if sampler else None
        self._adapters = list(adapters or [])
        self._status = "starting"
        self._started_at = None
        self._last_error = None
        log.info("API supervisor: spawned PID %d for %s", self._proc.pid, model)

        # Hand off readiness probe to a worker so start() returns fast and
        # the UI can show "starting…" while the model loads.
        threading.Thread(target=self._probe_readiness, daemon=True).start()

    def _probe_readiness(self) -> None:
        url = f"http://{self.host}:{self.port}/v1/models"
        deadline = time.time() + self.BOOT_TIMEOUT_S
        while time.time() < deadline:
            with self._lock:
                proc = self._proc
            if proc is None or proc.poll() is not None:
                with self._lock:
                    self._status = "crashed"
                    self._last_error = (
                        f"child exited during boot with code "
                        f"{proc.returncode if proc else '?'}"
                    )
                return
            try:
                req = urllib.request.Request(
                    url, headers={"Authorization": "Bearer sk-optiq-local"},
                )
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if 200 <= resp.status < 300:
                        with self._lock:
                            self._status = "ready"
                            self._started_at = time.time()
                        log.info("API supervisor: ready on %s", url)
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)

        # Timeout
        with self._lock:
            self._status = "error"
            self._last_error = f"never returned 200 from {url} within {self.BOOT_TIMEOUT_S}s"

    def _port_free(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.host, self.port)) != 0
