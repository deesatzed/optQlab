"""ASCII banner for the MTPLX CLI.

The banner uses ``ANSI Shadow``-style block letters spelling ``MTPLX``.
``rich`` is used for color when available; the function gracefully falls back
to plain stdlib ``print`` so the CLI works without ``rich`` installed.
"""

from __future__ import annotations

import os
from typing import Any


# ANSI Shadow-style block letters for "MTPLX". Letter widths chosen so the
# banner fits comfortably on an 80-column terminal.
_MTPLX_BANNER = [
    "███╗   ███╗ ████████╗ ██████╗  ██╗      ██╗  ██╗",
    "████╗ ████║ ╚══██╔══╝ ██╔══██╗ ██║      ╚██╗██╔╝",
    "██╔████╔██║    ██║    ██████╔╝ ██║       ╚███╔╝ ",
    "██║╚██╔╝██║    ██║    ██╔═══╝  ██║       ██╔██╗ ",
    "██║ ╚═╝ ██║    ██║    ██║      ███████╗ ██╔╝ ██╗",
    "╚═╝     ╚═╝    ╚═╝    ╚═╝      ╚══════╝ ╚═╝  ╚═╝",
]

_TAGLINE = "Native MTP speculative decoding · Apple Silicon"


def shell_banner_already_shown() -> bool:
    """Return whether the shell hook already printed the session banner."""

    value = os.environ.get("MTPLX_SHELL_BANNER_SHOWN") or os.environ.get("MTPLX_NO_BANNER")
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def banner_text(*, indent: int = 2) -> str:
    """Return the plain ASCII banner with tagline, no color codes.

    ``indent`` controls left padding so the banner doesn't hug column 0.
    """

    pad = " " * max(0, int(indent))
    rows = [pad + line for line in _MTPLX_BANNER]
    rows.append("")
    rows.append(pad + _TAGLINE)
    return "\n".join(rows)


def render_banner(*, console: Any | None = None, no_color: bool = False) -> None:
    """Print the banner. Uses ``rich`` for color when available.

    Falls back to plain text when ``rich`` is not importable or when
    ``no_color`` is set, so the CLI banner survives without dependencies.
    """

    if shell_banner_already_shown():
        return

    if no_color:
        print(banner_text())
        return

    try:
        from rich.console import Console
        from rich.text import Text
    except ImportError:
        print(banner_text())
        return

    target = console or Console()
    pad = "  "
    for line in _MTPLX_BANNER:
        target.print(Text(pad + line, style="bold cyan"))
    target.print()
    target.print(Text(pad + _TAGLINE, style="dim"))
    target.print()
