"""Swival integration helpers.

Swival's generic provider is configured entirely through CLI flags, so MTPLX
does not write a config file. The product surface returns the exact command a
user can run in another terminal while the local MTPLX server stays open.
"""

from __future__ import annotations

import shlex
import shutil
from typing import Any


SWIVAL_DEFAULT_CONTEXT_WINDOW = 262_144


def detect_swival_cli() -> dict[str, Any]:
    path = shutil.which("swival")
    if path:
        return {"available": True, "kind": "cli", "path": path}
    return {"available": False, "kind": "not_found"}


def build_swival_command(
    *,
    base_url: str,
    model_id: str,
    context_window: int = SWIVAL_DEFAULT_CONTEXT_WINDOW,
) -> list[str]:
    return [
        "swival",
        "--provider",
        "generic",
        "--base-url",
        str(base_url).rstrip("/"),
        "--model",
        str(model_id),
        "--max-context-tokens",
        str(int(context_window or SWIVAL_DEFAULT_CONTEXT_WINDOW)),
    ]


def shell_swival_command(
    *,
    base_url: str,
    model_id: str,
    context_window: int = SWIVAL_DEFAULT_CONTEXT_WINDOW,
) -> str:
    return " ".join(
        shlex.quote(part)
        for part in build_swival_command(
            base_url=base_url,
            model_id=model_id,
            context_window=context_window,
        )
    )
