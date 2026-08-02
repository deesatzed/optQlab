"""/setup, /login, /logout routes."""

from __future__ import annotations

from flask import (
    Blueprint, current_app, flash, make_response, redirect,
    render_template, request, url_for,
)

from .. import auth


bp = Blueprint("auth", __name__)


@bp.route("/setup", methods=["GET", "POST"])
def setup():
    if auth.has_password():
        return redirect(url_for("auth.login"))

    if request.method == "POST":
        p1 = request.form.get("password", "")
        p2 = request.form.get("password_confirm", "")
        if not p1:
            flash("Pick a password to continue.", "warn")
        elif p1 != p2:
            flash("Passwords don't match.", "err")
        elif len(p1) < 8:
            flash("Password must be at least 8 characters.", "err")
        else:
            auth.set_password(p1)
            token = auth.issue_session_token()
            resp = make_response(redirect(url_for("home")))
            _set_session_cookie(resp, token)
            flash("Password set. Welcome to OptiQ Lab.", "ok")
            return resp

    return render_template("setup.html", page_title="First-run setup")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if not auth.has_password():
        return redirect(url_for("auth.setup"))

    if request.method == "POST":
        if auth.verify_password(request.form.get("password", "")):
            token = auth.issue_session_token()
            resp = make_response(redirect(url_for("home")))
            _set_session_cookie(resp, token)
            return resp
        flash("Wrong password.", "err")

    return render_template("login.html", page_title="Sign in")


@bp.route("/logout", methods=["POST", "GET"])
def logout():
    resp = make_response(redirect(url_for("auth.login")))
    resp.delete_cookie(auth.COOKIE_NAME, path="/")
    flash("Signed out.", "ok")
    return resp


def _set_session_cookie(resp, token: str) -> None:
    resp.set_cookie(
        auth.COOKIE_NAME,
        token,
        max_age=auth.JWT_TTL_SECONDS,
        httponly=True,
        samesite="Lax",
        secure=False,  # Lab is localhost
        path="/",
    )
