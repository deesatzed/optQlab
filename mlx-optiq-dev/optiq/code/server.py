"""Discover / attach / spawn the OptiQ server (design §5.1) — the zero-config edge.

On `optiq code` startup we want a served model with no configuration:
  1. Probe the default OptiQ server; if it answers, ATTACH and use whatever model
     it is serving (the Lab/CLI control plane chose it).
  2. If nothing is listening, SPAWN `optiq serve` with the requested/last/default
     model, wait until ready, and tear it down on exit (only because we spawned it).
  3. If no model is available at all, a friendly message points at
     `optiq quantize` / `optiq lab`.
"""
from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


def _base_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    return f"http://{host}:{port}/v1"


def discover_model(base_url: str, timeout: float = 2.0) -> str | None:
    """Return the id of the model the OptiQ server is actually serving, or None.

    `optiq serve` lists the whole local cache in `/v1/models` (LM-Studio style)
    but exposes the SERVED model's variant labels (`:think` / `:no-think` / ...).
    So the served model is the base of any variant id; fall back to the sole
    entry when the server lists exactly one, then to the first entry.
    """
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as r:
            import json
            data = json.loads(r.read())
        ids = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    if not ids:
        return None
    for suffix in (":no-think", ":think"):
        for i in ids:
            if i.endswith(suffix):
                return i[: -len(suffix)]
    return ids[0] if len(ids) == 1 else ids[0]


@dataclass
class ServerHandle:
    base_url: str
    model_id: str
    proc: subprocess.Popen | None = None   # set only if we spawned it

    def close(self) -> None:
        """Tear down only a server we spawned; leave an attached one running."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class NoModelError(RuntimeError):
    pass


def ensure_server(
    *,
    model: str | None = None,
    base_url: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    ready_timeout: float = 180.0,
    spawn: bool = True,
) -> ServerHandle:
    """Attach to a running OptiQ server, or spawn one. Returns a ServerHandle.

    If a ``base_url`` is configured we attach there and never spawn — the user
    (or a custom port) manages that server. That covers a cloud / third-party
    OpenAI-compatible endpoint (OpenRouter, etc.): set ``base_url`` (in
    ``~/.optiq/code/config.json`` or ``$OPTIQ_BASE_URL``) and name the model with
    ``--model`` / ``model`` in config / ``OPTIQ_CODE_MODEL``. The resolved config
    is passed as ``base_url`` here; the env var is the fallback so a bare call
    still honors it. When the model is given explicitly we do not require
    ``/models`` to answer — a cloud endpoint may gate it behind auth or list
    hundreds of models — so discovery is best-effort and only the
    no-model-and-no-discovery case is an error.
    """
    env_base = base_url or os.environ.get("OPTIQ_BASE_URL")
    if env_base:
        served = discover_model(env_base)
        model_id = model or served
        if model_id is None:
            raise NoModelError(
                f"OPTIQ_BASE_URL={env_base} is set but no server answered "
                f"at {env_base}/models and no --model / OPTIQ_CODE_MODEL was "
                f"given. For a cloud endpoint, name the model explicitly, e.g. "
                f"--model qwen/qwen3.5-397b-a17b.")
        return ServerHandle(base_url=env_base, model_id=model_id, proc=None)

    base = _base_url(host, port)

    # 1. attach if something is already serving
    served = discover_model(base)
    if served is not None:
        return ServerHandle(base_url=base, model_id=model or served, proc=None)

    if not spawn:
        raise NoModelError(f"no OptiQ server reachable at {base}")

    # 2. spawn `optiq serve`
    target = model or os.environ.get("OPTIQ_CODE_MODEL")
    if not target:
        raise NoModelError(
            "No OptiQ server is running and no model was given.\n"
            "Serve one first (`optiq serve --model <repo>`), or pass `--model`,\n"
            "or get a quant with `optiq quantize` / `optiq lab`.")

    argv = ["optiq", "serve", "--model", target, "--host", host, "--port", str(port)]
    proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 3. wait until it answers /v1/models (Metal kernel compile can take a while)
    deadline = time.monotonic() + ready_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise NoModelError(f"`optiq serve --model {target}` exited before becoming ready")
        served = discover_model(base)
        if served is not None:
            return ServerHandle(base_url=base, model_id=model or served, proc=proc)
        time.sleep(1.0)

    proc.terminate()
    raise NoModelError(f"`optiq serve` did not become ready within {ready_timeout:.0f}s")
