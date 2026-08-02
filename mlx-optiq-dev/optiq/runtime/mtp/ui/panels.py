"""Framed status panels for MTPLX startup output.

Uses ``rich.panel.Panel`` when available, falls back to a small plain-text
box-drawing renderer otherwise.
"""

from __future__ import annotations

from typing import Any, Iterable


def render_startup_panel(
    *,
    version: str,
    model: str,
    profile: str,
    profile_summary: str | None = None,
    api_url: str,
    chat_url: str | None = None,
    mode_label: str | None = None,
    thermal_label: str | None = None,
    extra_lines: Iterable[tuple[str, str]] | None = None,
    console: Any | None = None,
) -> None:
    """Print a framed status panel summarizing the running configuration."""

    rows: list[tuple[str, str]] = [
        ("Model", model),
        ("Profile", _profile_label(profile, profile_summary)),
        ("API", api_url),
    ]
    if mode_label:
        rows.append(("Mode", mode_label))
    if thermal_label:
        rows.append(("Thermal", thermal_label))
    if chat_url:
        rows.append(("Browser", chat_url))
    if extra_lines:
        rows.extend(list(extra_lines))
    rows.append(("Stop", "Ctrl-C"))

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        _print_plain_panel(version, rows)
        return

    target = console or Console()

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right", no_wrap=True)
    table.add_column(no_wrap=False)
    for label, value in rows:
        table.add_row(f"{label}", str(value))

    panel = Panel(
        table,
        title=Text(f"MTPLX {version}", style="bold cyan"),
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
        expand=False,
    )
    target.print(panel)
    target.print()


def _profile_label(name: str, summary: str | None) -> str:
    if summary:
        return f"{name} ({summary})"
    return name


def _print_plain_panel(version: str, rows: list[tuple[str, str]]) -> None:
    label_width = max((len(label) for label, _ in rows), default=0)
    body_lines = [f"  {label.rjust(label_width)}  {value}" for label, value in rows]
    inner_width = max(
        (len(line) for line in body_lines),
        default=0,
    )
    title = f" MTPLX {version} "
    inner_width = max(inner_width, len(title) + 4)

    print("┌" + "─" * 2 + title + "─" * (inner_width - len(title) - 2) + "┐")
    print("│" + " " * inner_width + "│")
    for line in body_lines:
        padded = line + " " * (inner_width - len(line))
        print("│" + padded + "│")
    print("│" + " " * inner_width + "│")
    print("└" + "─" * inner_width + "┘")
    print()
