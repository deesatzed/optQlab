"""Settings → Hugging Face + Account."""

from __future__ import annotations

import importlib.metadata
import platform
import sqlite3

from flask import (
    Blueprint, current_app, flash, jsonify, redirect, render_template,
    request, url_for,
)
from huggingface_hub.utils import HfHubHTTPError

from .. import auth, db, hf, local_quants


bp = Blueprint("settings", __name__, url_prefix="/settings")


@bp.route("/huggingface", methods=["GET"])
def hf_index():
    tokens = hf.list_tokens()
    return render_template(
        "settings_hf.html",
        page_title="Hugging Face",
        section="settings_hf",
        tokens=tokens,
    )


@bp.route("/huggingface", methods=["POST"])
def hf_add():
    name = (request.form.get("name") or "").strip()
    token = (request.form.get("token") or "").strip()
    password = request.form.get("password") or ""

    if not name or not token or not password:
        flash("Name, token, and password are all required.", "err")
        return redirect(url_for("settings.hf_index"))

    if not auth.verify_password(password):
        flash("Wrong Lab password.", "err")
        return redirect(url_for("settings.hf_index"))

    try:
        token_id = hf.save_token(name, token, password)
    except HfHubHTTPError as e:
        flash(f"Hugging Face rejected the token: {e}", "err")
        return redirect(url_for("settings.hf_index"))
    except Exception as e:
        flash(f"Couldn't save token: {e}", "err")
        return redirect(url_for("settings.hf_index"))

    flash(f"Saved token #{token_id}.", "ok")
    return redirect(url_for("settings.hf_index"))


@bp.route("/huggingface/<int:token_id>/delete", methods=["POST"])
def hf_delete(token_id):
    hf.delete_token(token_id)
    flash(f"Deleted token #{token_id}.", "ok")
    return redirect(url_for("settings.hf_index"))


@bp.route("/server", methods=["GET"])
def server():
    """Show the current model + MTP + a quant picker. Apply-without-
    restart requires running mlx_lm.server as a subprocess (not a thread)
    and isn't in v0.1.0; for now we show the exact CLI command."""
    paths = current_app.config["OPTIQ_LAB_PATHS"]
    api_url = current_app.config["OPTIQ_API_URL"]
    api_port = api_url.rsplit(":", 1)[-1] if ":" in api_url else "8080"
    loaded = current_app.config.get("OPTIQ_LOADED_MODEL")
    mtp_on = current_app.config.get("OPTIQ_MTP_ENABLED", False)
    mtp_depth = current_app.config.get("OPTIQ_MTP_DEPTH", 0)
    drafter_id = current_app.config.get("OPTIQ_DRAFTER_ID")
    # Suggested -assistant drafters per Gemma-4 target. The Lab picks
    # the matching drafter automatically when a Gemma-4 model is selected;
    # the user can still type any other HF id in the text field to override.
    # Non-Gemma families (Qwen) use MTP, not a separate drafter — for them
    # the picker stays hidden.
    known_drafters = {
        "gemma-4-e2b":     "mlx-community/gemma-4-e2b-it-assistant-bf16",
        "gemma-4-e4b":     "mlx-community/gemma-4-e4b-it-assistant-bf16",
        "gemma-4-26b-a4b": "mlx-community/gemma-4-26B-A4B-it-assistant-bf16",
        "gemma-4-31b":     "mlx-community/gemma-4-31B-it-assistant-bf16",
    }

    def _summarize(q, source: str) -> dict:
        return {
            "name": q.display_name,
            "path": q.path,
            "bits_label": q.bits_label,
            "bpw_detail": q.bpw_detail,
            "bpw_label": q.bpw_label,
            "has_mtp": q.has_mtp,
            "size_gb": round(q.size_bytes / 1e9, 2) if q.size_bytes else None,
            "source": source,
        }

    local_built = [_summarize(q, "local")
                   for q in local_quants.discover(paths.models_dir)]
    # Also surface mlx-community models the user has pulled into the HF
    # cache (~/.cache/huggingface/hub). Anything mlx-lm can load shows up.
    hf_cached = [_summarize(q, "hf_cache")
                 for q in local_quants.discover_hf_cache()]
    # De-dup by path so a model that's both built locally and HF-cached
    # shows up once (local takes precedence since it has more metadata).
    seen_paths = {q["path"] for q in local_built}
    quants = local_built + [q for q in hf_cached if q["path"] not in seen_paths]

    from ..mlx_cleanup import default_prompt_cache_bytes
    default_pc_gb = round(default_prompt_cache_bytes() / 1024**3, 2)

    return render_template(
        "settings_server.html",
        page_title="Server",
        section="settings_server",
        loaded=loaded,
        mtp_on=mtp_on,
        mtp_depth=mtp_depth,
        drafter_id=drafter_id,
        known_drafters=known_drafters,
        api_port=api_port,
        quants=quants,
        default_pc_gb=default_pc_gb,
    )


@bp.route("/account", methods=["GET"])
def account():
    """Account page: change password + Lab metadata."""
    paths = current_app.config["OPTIQ_LAB_PATHS"]
    conn = db.get_conn()
    row = conn.execute(
        "SELECT created_at FROM credentials WHERE id = 1"
    ).fetchone()
    password_created = row["created_at"] if row else None

    n_jobs = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
    n_models = conn.execute(
        "SELECT COUNT(*) AS n FROM models_local"
    ).fetchone()["n"]
    n_tokens = conn.execute(
        "SELECT COUNT(*) AS n FROM hf_tokens"
    ).fetchone()["n"]

    try:
        optiq_version = importlib.metadata.version("mlx-optiq")
    except importlib.metadata.PackageNotFoundError:
        optiq_version = "unknown"

    return render_template(
        "settings_account.html",
        page_title="Account",
        section="settings_account",
        password_created=password_created,
        optiq_version=optiq_version,
        python_version=platform.python_version(),
        os_label=f"{platform.system()} {platform.release()} ({platform.machine()})",
        sqlite_version=sqlite3.sqlite_version,
        lab_root=str(paths.root),
        n_jobs=n_jobs,
        n_models=n_models,
        n_tokens=n_tokens,
    )


@bp.route("/account/password", methods=["POST"])
def change_password():
    old = request.form.get("old_password") or ""
    new = request.form.get("new_password") or ""
    confirm = request.form.get("new_password_confirm") or ""
    if not (old and new and confirm):
        flash("All three password fields are required.", "warn")
        return redirect(url_for("settings.account"))
    if new != confirm:
        flash("New passwords don't match.", "err")
        return redirect(url_for("settings.account"))
    try:
        auth.change_password(old, new)
    except PermissionError:
        flash("Current password is wrong.", "err")
        return redirect(url_for("settings.account"))
    except ValueError as e:
        flash(str(e), "err")
        return redirect(url_for("settings.account"))

    flash("Password changed. Sign in again with the new one.", "ok")
    # Force re-login since the salt-derived key (used for HF token
    # encryption) is now invalidated.
    resp = redirect(url_for("auth.logout"))
    return resp


@bp.route("/huggingface/validate", methods=["POST"])
def hf_validate():
    """AJAX endpoint — checks a token without persisting it.
    Returns the parsed user info so the user sees who they'd push as."""
    data = request.get_json(force=True) or {}
    token = (data.get("token") or "").strip()
    if not token:
        return jsonify({"ok": False, "error": "no token provided"}), 400
    try:
        info = hf.whoami(token)
    except HfHubHTTPError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({
        "ok": True,
        "username": info.get("name") or info.get("fullname"),
        "orgs": [o.get("name") for o in (info.get("orgs") or []) if o.get("name")],
        "type": info.get("type"),
    })
