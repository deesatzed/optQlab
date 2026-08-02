"""Advertise the served model's effective context window over HTTP.

A client that has to stay inside the window (``optiq code``, which compacts its
message history before it overruns) can't work the number out for itself. The
model's native context from ``config.json`` is the wrong answer whenever
``--max-context`` engaged, and that is exactly the case where staying inside
the window matters: a tight Mac where the full native context would OOM.

So the server reports it. ``GET /v1/optiq/context`` returns::

    {"context_window": 32768, "native": 131072, "capped": true}

``context_window`` is the effective one -- the cap if a cap is installed, the
model's native context otherwise. Clients that get a connection error, a 404
(an older OptiQ, or stock ``mlx_lm.server``), or a malformed body should fall
back to their own default rather than fail; the endpoint is an optimization,
not a contract the client depends on.
"""

from __future__ import annotations

import json
from typing import Optional

_NATIVE: Optional[int] = None
_CAP: Optional[int] = None


def set_native(n: Optional[int]) -> None:
    """Record the model's native context length (from its config)."""
    global _NATIVE
    _NATIVE = int(n) if n else None


def set_cap(n: Optional[int]) -> None:
    """Record the installed context cap, if `--max-context` engaged."""
    global _CAP
    _CAP = int(n) if n else None


def effective_window() -> Optional[int]:
    """The window a client should actually plan against."""
    if _CAP and _NATIVE:
        return min(_CAP, _NATIVE)
    return _CAP or _NATIVE


def payload() -> dict:
    return {"context_window": effective_window(),
            "native": _NATIVE,
            "capped": _CAP is not None}


def install() -> bool:
    """Add ``GET /v1/optiq/context`` to the running mlx-lm server.

    Additive: any other GET route (including ones another OptiQ patch added)
    still reaches the previous handler. Idempotent.
    """
    try:
        import mlx_lm.server as server_mod
    except Exception:
        return False
    if getattr(server_mod, "_optiq_ctx_report_installed", False):
        return True

    orig_do_GET = getattr(server_mod.APIHandler, "do_GET", None)

    def patched_do_GET(self):
        if self.path.rstrip("/") == "/v1/optiq/context":
            body = json.dumps(payload()).encode()
            self._set_completion_headers(200)
            self.end_headers()
            self.wfile.write(body)
            return
        if orig_do_GET is not None:
            return orig_do_GET(self)
        self._set_completion_headers(404)
        self.end_headers()
        self.wfile.write(b"Not Found")

    server_mod.APIHandler.do_GET = patched_do_GET
    server_mod._optiq_ctx_report_installed = True
    return True
