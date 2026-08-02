"""Flask application factory for OptiQ Lab.

The Lab is a separate web UI from the model-serving API (which stays
``mlx_lm.server`` on its own port). ``optiq lab`` boots both in the same
process — Flask in the main thread, mlx_lm.server in a daemon thread —
so users only run one command but each surface keeps its own port for
clarity. The Lab sidebar shows the API URL so users have something
concrete to paste into Claude Code / Codex / etc.
"""

from __future__ import annotations

import secrets
from typing import Any

from flask import Flask, redirect, render_template, request, url_for

from . import auth, db, jobs
from .config import ensure_lab_dirs
from .routes import api as api_routes
from .routes import arena as arena_routes
from .routes import auth as auth_routes
from .routes import chat as chat_routes
from .routes import cluster as cluster_routes
from .routes import dataset as dataset_routes
from .routes import finetune as finetune_routes
from .routes import hub as hub_routes
from .routes import quantize as quantize_routes
from .routes import settings as settings_routes
from .routes import spine_api as spine_api_routes


# Path prefixes that do not require an authenticated session — auth pages
# themselves, static assets, and the health probe.
PUBLIC_PREFIXES = ("/setup", "/login", "/static/", "/healthz")


def create_app(
    api_url: str | None = None,
    secret_key: bytes | None = None,
    mtp_enabled: bool = False,
    mtp_depth: int = 0,
    drafter_id: str | None = None,
    loaded_model: str | None = None,
    supervisor=None,  # ApiSupervisor | None — passed by `optiq lab` CLI
) -> Flask:
    """Build a Flask app instance.

    ``api_url`` is the URL where the model-serving API lives (e.g.
    ``http://127.0.0.1:8080``). Threaded through templates so the
    sidebar can display copy-paste configs for each integration.

    ``mtp_enabled`` / ``mtp_depth`` and ``drafter_id`` reflect the
    speculative-decoding flags passed to ``optiq lab`` (``--mtp`` or
    ``--drafter``); the API server doesn't expose this over the wire,
    so the Lab tracks it locally for the sidebar status pill.
    """
    paths = ensure_lab_dirs()

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/static",
    )
    app.config.update(
        SECRET_KEY=secret_key or _load_or_create_secret(paths.root / "secret.key"),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=False,  # localhost
        MAX_CONTENT_LENGTH=512 * 1024 * 1024,  # 512MB for dataset uploads
        OPTIQ_API_URL=api_url or "http://127.0.0.1:8080",
        OPTIQ_LAB_PATHS=paths,
        OPTIQ_MTP_ENABLED=bool(mtp_enabled),
        OPTIQ_MTP_DEPTH=int(mtp_depth or 0),
        OPTIQ_DRAFTER_ID=drafter_id or None,
        OPTIQ_LOADED_MODEL=loaded_model,
        OPTIQ_API_SUPERVISOR=supervisor,
    )

    # Initialise DB (creates tables on first connect) and mark zombie jobs
    with app.app_context():
        db.get_conn()
        jobs.mark_zombies()

    # Blueprints
    app.register_blueprint(auth_routes.bp)
    app.register_blueprint(api_routes.bp)
    app.register_blueprint(spine_api_routes.bp)
    app.register_blueprint(chat_routes.bp)
    app.register_blueprint(cluster_routes.bp)
    app.register_blueprint(arena_routes.bp)
    app.register_blueprint(hub_routes.bp)
    app.register_blueprint(quantize_routes.bp)
    app.register_blueprint(finetune_routes.bp)
    app.register_blueprint(dataset_routes.bp)
    app.register_blueprint(settings_routes.bp)

    # Make config + helpers available in every template
    @app.context_processor
    def inject_globals() -> dict[str, Any]:
        # Cache-bust static CSS by its mtime so edits show on a normal refresh
        # (no stale-stylesheet "why don't my changes appear" surprises).
        import os as _os
        try:
            _v = str(int(_os.path.getmtime(
                _os.path.join(app.static_folder, "css", "lab.min.css"))))
        except OSError:
            _v = "1"
        return {
            "api_url": app.config["OPTIQ_API_URL"],
            "credentials_exist": db.credentials_exist(),
            "asset_v": _v,
        }

    # Auth gate. Runs before every view. Redirects unauthenticated
    # requests to /setup (no creds yet) or /login (creds exist).
    @app.before_request
    def _require_auth():
        # Test-only bypass: only active when both TESTING and the explicit
        # OPTIQ_TEST_AUTH_BYPASS flag are set. Production never sets these.
        if app.config.get("TESTING") and app.config.get("OPTIQ_TEST_AUTH_BYPASS"):
            return None
        path = request.path
        # Public assets / probe / auth pages bypass.
        if any(path == p or path.startswith(p) for p in PUBLIC_PREFIXES):
            return None
        token = auth.current_session_token_from_cookies(request.cookies)
        if auth.verify_session_token(token or ""):
            return None
        if not auth.has_password():
            return redirect(url_for("auth.setup"))
        return redirect(url_for("auth.login"))

    # Health endpoint (used by the test fixture to wait for boot)
    @app.route("/healthz")
    def healthz():
        return {"ok": True}

    @app.route("/")
    def home():
        # Mirrors Unsloth Studio: first paint is Chat.
        return redirect(url_for("chat.chat_page"))

    @app.route("/integrations")
    def integrations():
        return render_template(
            "integrations.html",
            page_title="Integrations",
            section="integrations",
        )

    return app


def _load_or_create_secret(path) -> bytes:
    """Persistent Flask SECRET_KEY so sessions survive Lab restarts."""
    if path.exists():
        return path.read_bytes()
    secret = secrets.token_bytes(32)
    path.write_bytes(secret)
    path.chmod(0o600)
    return secret
