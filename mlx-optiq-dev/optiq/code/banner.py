"""The OptiQ Code start-up banner.

A wordmark shown on launch. Emerald on warm paper, matching the OptiQ brand
(`#03684c` on `#f6f2ea`). Monochrome-safe: it must read in any terminal, so the
color is a single accent applied over an ASCII wordmark, and there is a plain
no-color path for dumb terminals / pipes.

Ported and rebranded from `conjure/tui/banner.py` (which was a cyan→magenta
"Conjure" gradient with a formal-methods tagline — both retired).
"""
from __future__ import annotations

# The OptiQ emerald accent and warm paper, shared across the brand.
EMERALD = "#03684c"

# "OptiQ" in a blocky ASCII wordmark. The trailing Q carries the brand glyph;
# the code caret ›_ after it is the Code facet's differentiator (design §7.1).
_WORDMARK = [
    r" ___        _   _  ___    ___         _      ",
    r"/ _ \ _ __ | |_(_)/ _ \  / __|___  __| |___  ",
    r"| (_) | '_ \|  _| | (_) || (__/ _ \/ _` / -_) ",
    r" \___/| .__/ \__|_|\__\_\ \___\___/\__,_\___| ",
    r"      |_|                          " + "›_",
]

_TAGLINE = "the coding agent for local models on your Mac"


def banner_text(*, repo_name: str, model: str | None = None, color: bool = True) -> str:
    """Return the launch banner as a string.

    ``color`` uses ANSI truecolor for the emerald accent; set it False for a
    plain-text banner (pipes, ``--no-color``, dumb terminals).
    """
    def em(s: str) -> str:
        if not color:
            return s
        r, g, b = 0x03, 0x68, 0x4c
        return f"\x1b[38;2;{r};{g};{b}m{s}\x1b[0m"

    lines = [""]
    for row in _WORDMARK:
        lines.append("  " + em(row))
    lines.append("")
    lines.append("  " + em("❖") + f"  {_TAGLINE}")
    lines.append("")
    served = model or "auto (whatever OptiQ is serving)"
    lines.append(f"  repo  {repo_name}")
    lines.append(f"  model {served}")
    lines.append("")
    return "\n".join(lines)


def banner_markup(*, repo_name: str, model: str | None = None) -> str:
    """Rich-markup variant for rendering inside the Textual TUI header."""
    served = model or "auto (whatever OptiQ is serving)"
    lines = [""]
    for row in _WORDMARK:
        lines.append(f"  [{EMERALD}]{row}[/]")
    lines.append("")
    lines.append(f"  [{EMERALD}]❖[/]  [b italic]{_TAGLINE}[/]")
    lines.append("")
    lines.append(f"  [dim]repo[/]  [b]{repo_name}[/]")
    lines.append(f"  [dim]model[/] {served}")
    lines.append("")
    return "\n".join(lines)
