"""Authentication helpers for ``optiq serve``.

Mirrors Unsloth's ``sk-unsloth-*`` pattern: clients pass a Bearer token
that starts with ``sk-optiq-`` (any suffix). This is prefix-validation
only, not real auth, but it gives every integration a stable token
shape to put in their configs and trips clients that forget the header
entirely.

Install via ``install_auth()`` before starting the server. Idempotent.
"""

from __future__ import annotations

import json
import logging


_REQUIRED_PREFIX = "sk-optiq-"
_INSTALLED = False


def install_auth() -> None:
    """Wrap ``mlx_lm.server.APIHandler.do_POST`` to enforce the prefix.

    Behavior:
      - Authorization header absent: allowed (warned in log). Lets local
        ``curl`` calls work without ceremony.
      - Authorization header present with ``Bearer sk-optiq-<anything>``:
        allowed.
      - Otherwise: 401 with a JSON error matching the OpenAI error shape.

    Install AFTER the endpoint installers (anthropic, responses) so this
    layer wraps them. Order in ``serve_cmd``:
      1. install_quantized_kv / install_mixed_kv
      2. install_anthropic_endpoint
      3. install_responses_endpoint
      4. install_mtp_speculation
      5. install_auth   <- last, so it wraps everything
    """
    global _INSTALLED
    if _INSTALLED:
        return

    import mlx_lm.server as server_mod

    current_do_POST = server_mod.APIHandler.do_POST

    def patched_do_POST(self):
        auth = self.headers.get("Authorization")
        if auth is not None:
            # Strip ``Bearer `` prefix if present (case-insensitive scheme).
            token = auth.strip()
            if token.lower().startswith("bearer "):
                token = token[7:].strip()
            if not token.startswith(_REQUIRED_PREFIX):
                _write_unauthorized(self)
                return
        # else: no header, allow (local-dev tolerance)
        return current_do_POST(self)

    server_mod.APIHandler.do_POST = patched_do_POST
    _INSTALLED = True
    logging.info(
        "[optiq] auth installed (accepts Bearer %s* tokens; missing header allowed)",
        _REQUIRED_PREFIX,
    )


def _write_unauthorized(handler) -> None:
    body = json.dumps({
        "error": {
            "type": "invalid_request_error",
            "message": (
                f"Authorization token must be a Bearer token starting with "
                f"'{_REQUIRED_PREFIX}'. Example: 'Bearer sk-optiq-local'. "
                f"Or omit the Authorization header for local development."
            ),
        }
    }).encode()
    handler.send_response(401)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)
