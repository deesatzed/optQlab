"""MTPLX terminal UI helpers.

This package is import-on-demand from command handlers. It is intentionally
NOT imported at the top of ``mtplx/__init__.py`` so the package can survive in
fresh venvs that do not have ``rich`` installed.

Each module gracefully degrades to plain stdlib ``print`` when ``rich`` is not
available, so the CLI never hard-fails on a missing dependency.
"""

from __future__ import annotations

from .banner import banner_text, render_banner
from .chat_printer import ChatPrinter
from .panels import render_startup_panel
from .progress import ModelLoadProgress

__all__ = [
    "banner_text",
    "render_banner",
    "render_startup_panel",
    "ChatPrinter",
    "ModelLoadProgress",
]
