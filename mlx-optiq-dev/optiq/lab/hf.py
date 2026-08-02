"""Hugging Face token storage + push helpers.

Tokens are encrypted at rest with a Fernet key derived from the user's
Lab password (via PBKDF2-HMAC-SHA256 over the per-install salt). This
means: token plaintext only exists in memory during a logged-in session.
A user who forgets their password loses access to their stored HF tokens
(by design — they'd need to re-paste from huggingface.co/settings/tokens).
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Iterable, Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from . import auth, db


# A note on threat model: this is local-only Lab on macOS. The threat we're
# defending against is "someone reads ~/.optiq/lab.db". By encrypting the
# token with a key derived from the user's password (which is never stored
# in plaintext), a stolen DB file is useless without the password.
PBKDF2_ITERATIONS = 200_000


@dataclass
class HFTokenInfo:
    id: int
    name: str
    username: str | None
    orgs: list[str]
    scope: str | None
    created_at: str


# ---------------------------------------------------------------------------
# Encryption — Fernet key derived from password + per-install salt
# ---------------------------------------------------------------------------


def _derive_fernet_key(password: str) -> bytes:
    """PBKDF2-HMAC-SHA256(password, salt, 200k) → urlsafe-base64 key."""
    salt = auth.get_salt()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))


def _encrypt(token: str, password: str) -> bytes:
    return Fernet(_derive_fernet_key(password)).encrypt(token.encode("utf-8"))


def _decrypt(blob: bytes, password: str) -> Optional[str]:
    try:
        return Fernet(_derive_fernet_key(password)).decrypt(blob).decode("utf-8")
    except InvalidToken:
        return None


# ---------------------------------------------------------------------------
# HF API integration
# ---------------------------------------------------------------------------


def whoami(token: str) -> dict:
    """Validate a token against the HF API. Returns the parsed user info
    or raises huggingface_hub.utils.HfHubHTTPError on a bad token."""
    from huggingface_hub import HfApi
    return HfApi(token=token).whoami()


# ---------------------------------------------------------------------------
# Store / retrieve
# ---------------------------------------------------------------------------


def save_token(name: str, token: str, password: str) -> int:
    """Encrypt + persist. Validates via whoami first."""
    if not name:
        raise ValueError("name is required")
    info = whoami(token)
    username = info.get("name") or info.get("fullname")
    orgs = [o.get("name") for o in (info.get("orgs") or []) if o.get("name")]
    # token role only present on fine-grained / write tokens
    scope = (info.get("auth") or {}).get("accessToken", {}).get("role") or info.get("type")

    blob = _encrypt(token, password)
    cur = db.get_conn().execute(
        """
        INSERT INTO hf_tokens (name, encrypted_token, username, orgs, scope)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, blob, username, ",".join(orgs) if orgs else None, scope),
    )
    return cur.lastrowid


def list_tokens() -> list[HFTokenInfo]:
    rows = db.get_conn().execute(
        "SELECT id, name, username, orgs, scope, created_at FROM hf_tokens ORDER BY id DESC"
    ).fetchall()
    return [
        HFTokenInfo(
            id=r["id"], name=r["name"], username=r["username"],
            orgs=(r["orgs"] or "").split(",") if r["orgs"] else [],
            scope=r["scope"], created_at=r["created_at"],
        )
        for r in rows
    ]


def get_decrypted_token(token_id: int, password: str) -> str | None:
    row = db.get_conn().execute(
        "SELECT encrypted_token FROM hf_tokens WHERE id = ?", (token_id,),
    ).fetchone()
    if row is None:
        return None
    return _decrypt(row["encrypted_token"], password)


def delete_token(token_id: int) -> None:
    db.get_conn().execute("DELETE FROM hf_tokens WHERE id = ?", (token_id,))


def get_first_token_decrypted(password: str) -> tuple[int, str] | None:
    """Convenience for Workflows E + G + H: return the most recently
    added token's plaintext. Returns None if no token is saved."""
    row = db.get_conn().execute(
        "SELECT id, encrypted_token FROM hf_tokens ORDER BY id DESC LIMIT 1",
    ).fetchone()
    if row is None:
        return None
    plain = _decrypt(row["encrypted_token"], password)
    if plain is None:
        return None
    return (row["id"], plain)


# ---------------------------------------------------------------------------
# Push helpers used by E, G, H workflows
# ---------------------------------------------------------------------------


# Funnel banner stamped onto every OptiQ quant's model card at push time, so
# anyone who finds the model on Hugging Face has a path back to the tool that
# made it. Only added when the folder is an OptiQ quant (has optiq_metadata.json).
OPTIQ_CARD_BANNER = (
    "> **Built with [mlx-optiq](https://mlx-optiq.com)**, the MLX-native toolkit "
    "to quantize, fine-tune, and serve LLMs locally on Apple Silicon (no PyTorch, "
    "no cloud). [Try the Lab](https://mlx-optiq.com/docs/lab/) · "
    "[All OptiQ quants](https://mlx-optiq.com/models) · "
    "[Docs](https://mlx-optiq.com/docs/)\n>\n"
    "> **Supported loaders:** [mlx-optiq](https://mlx-optiq.com) (text, vision, and "
    "MTP) and stock [mlx-lm](https://github.com/ml-explore/mlx-lm) (text). Other "
    "front-ends load MLX weights through their own stack, so support there depends "
    "on that stack rather than on these files."
)


def _ensure_optiq_banner(folder: str) -> None:
    """If ``folder`` is an OptiQ quant, make sure its README carries the
    mlx-optiq banner (inserted after the H1, or after YAML frontmatter).
    Idempotent and frontmatter-aware. No-op for non-OptiQ folders."""
    import os
    if not os.path.exists(os.path.join(folder, "optiq_metadata.json")):
        return  # not an OptiQ quant; don't claim authorship
    readme = os.path.join(folder, "README.md")
    s = open(readme).read() if os.path.exists(readme) else ""
    if "Built with [mlx-optiq]" in s:
        return  # already stamped
    lines = s.split("\n") if s.strip() else ["# OptiQ quant"]
    # Skip past YAML frontmatter if present.
    fm_end = 0
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                fm_end = i + 1
                break
    # Insert after the first H1 (preferred), else right after frontmatter.
    ins = fm_end
    for i in range(fm_end, len(lines)):
        if lines[i].startswith("# "):
            ins = i + 1
            break
    new = lines[:ins] + ["", OPTIQ_CARD_BANNER] + lines[ins:]
    with open(readme, "w") as f:
        f.write("\n".join(new))


def push_folder(
    *,
    folder: str,
    repo_id: str,
    token: str,
    repo_type: str = "model",
    private: bool = True,
    create_if_missing: bool = True,
    commit_message: str | None = None,
) -> str:
    """Upload a local directory as an HF repo. Idempotent — creates the
    repo if needed, otherwise pushes on top. OptiQ quants get the mlx-optiq
    funnel banner stamped onto their model card automatically."""
    from huggingface_hub import HfApi, create_repo
    if repo_type == "model":
        try:
            _ensure_optiq_banner(folder)
        except Exception:
            pass  # never block a push on the cosmetic banner
    if create_if_missing:
        create_repo(
            repo_id=repo_id, token=token, repo_type=repo_type,
            private=private, exist_ok=True,
        )
    api = HfApi(token=token)
    api.upload_large_folder(
        folder_path=folder,
        repo_id=repo_id,
        repo_type=repo_type,
        # upload_large_folder doesn't support commit_message in all versions;
        # if it raises, fall back to upload_folder.
    )
    return f"https://huggingface.co/{repo_id}"
