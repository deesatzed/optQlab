"""Password (argon2id) + session (JWT cookie) for the Lab.

Mirrors Unsloth Studio's auth model: on first launch the user lands on
``/setup`` to pick a password; thereafter ``/login``. Successful login
sets ``optiq_lab_session``, a 24-hour HS256 JWT in an httpOnly cookie.
The middleware in ``app.py`` redirects unauthenticated requests to the
appropriate page.

Token-binding intentionally simple: there's exactly one local user. JWT
sub is always ``"local"`` — we use a token (instead of a server-side
session) so the salt-derived encryption key for ``hf_tokens`` can be
re-derived on every request without keeping plaintext around.
"""

from __future__ import annotations

import base64
import os
import secrets
import time

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from . import db
from .config import ensure_lab_dirs


JWT_ALG = "HS256"
JWT_TTL_SECONDS = 24 * 60 * 60   # 24 hours
COOKIE_NAME = "optiq_lab_session"

_hasher = PasswordHasher()


# ---------------------------------------------------------------------------
# Password
# ---------------------------------------------------------------------------


def has_password() -> bool:
    return db.credentials_exist()


def set_password(plain: str) -> None:
    """Set the (single) Lab user password. Fails if one already exists."""
    if has_password():
        raise RuntimeError("password already set; use change_password()")
    if len(plain) < 8:
        raise ValueError("password must be at least 8 characters")
    salt = secrets.token_bytes(16)  # used by hf.py to derive an encryption key
    hashed = _hasher.hash(plain)
    db.get_conn().execute(
        "INSERT INTO credentials (id, password_hash, salt) VALUES (1, ?, ?)",
        (hashed, salt),
    )


def verify_password(plain: str) -> bool:
    """Constant-time-ish password check via argon2-cffi."""
    row = db.get_conn().execute(
        "SELECT password_hash FROM credentials WHERE id = 1",
    ).fetchone()
    if row is None:
        return False
    try:
        _hasher.verify(row["password_hash"], plain)
    except VerifyMismatchError:
        return False
    # If argon2 parameters changed, opportunistically rehash. Cheap.
    if _hasher.check_needs_rehash(row["password_hash"]):
        new_hash = _hasher.hash(plain)
        db.get_conn().execute(
            "UPDATE credentials SET password_hash = ? WHERE id = 1", (new_hash,),
        )
    return True


def change_password(old: str, new: str) -> None:
    """Replace the existing password. ``old`` must match.

    The salt stays the same, so the Fernet key derived from the password
    changes. Saved HF tokens are re-encrypted with the new key before
    the password hash is updated, so the user keeps access to them.
    """
    if not verify_password(old):
        raise PermissionError("current password is wrong")
    if len(new) < 8:
        raise ValueError("password must be at least 8 characters")

    # Re-encrypt existing HF tokens with the new password (same salt).
    # Imported here to avoid a circular import at module load.
    from . import hf
    conn = db.get_conn()
    rows = conn.execute("SELECT id, encrypted_token FROM hf_tokens").fetchall()
    new_blobs: list[tuple[bytes, int]] = []
    for row in rows:
        plain = hf._decrypt(row["encrypted_token"], old)
        if plain is None:
            # Old token unreadable already; leave it alone (a later list
            # call will just show it can't be decrypted).
            continue
        new_blobs.append((hf._encrypt(plain, new), row["id"]))

    conn.execute(
        "UPDATE credentials SET password_hash = ? WHERE id = 1",
        (_hasher.hash(new),),
    )
    for blob, token_id in new_blobs:
        conn.execute(
            "UPDATE hf_tokens SET encrypted_token = ? WHERE id = ?",
            (blob, token_id),
        )


def get_salt() -> bytes:
    """Return the per-install salt — used by hf.py to derive an
    encryption key for storing the HF token at rest. Raises if no
    password has been set yet."""
    row = db.get_conn().execute("SELECT salt FROM credentials WHERE id = 1").fetchone()
    if row is None:
        raise RuntimeError("no credentials row; call set_password() first")
    return row["salt"]


# ---------------------------------------------------------------------------
# JWT cookie
# ---------------------------------------------------------------------------


def _jwt_secret() -> bytes:
    """Persistent HS256 secret. Co-located with the Flask SECRET_KEY but
    kept distinct so rotating one doesn't invalidate the other."""
    path = ensure_lab_dirs().root / "jwt.key"
    if path.exists():
        return path.read_bytes()
    secret = secrets.token_bytes(32)
    path.write_bytes(secret)
    path.chmod(0o600)
    return secret


def issue_session_token() -> str:
    now = int(time.time())
    payload = {"sub": "local", "iat": now, "exp": now + JWT_TTL_SECONDS}
    return jwt.encode(payload, _jwt_secret(), algorithm=JWT_ALG)


def verify_session_token(token: str) -> bool:
    """True if ``token`` is a valid, unexpired JWT issued by us."""
    if not token:
        return False
    try:
        jwt.decode(
            token, _jwt_secret(),
            algorithms=[JWT_ALG],
            options={"require": ["exp", "iat", "sub"]},
        )
        return True
    except jwt.PyJWTError:
        return False


def current_session_token_from_cookies(cookies) -> str | None:
    """Flask request.cookies dict-like → cookie value or None."""
    val = cookies.get(COOKIE_NAME)
    if isinstance(val, bytes):
        val = val.decode("utf-8", "ignore")
    return val or None
