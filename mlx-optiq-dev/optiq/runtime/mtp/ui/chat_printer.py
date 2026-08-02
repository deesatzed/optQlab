"""Streaming chat printer for the MTPLX terminal CLI.

Wraps ``rich.console.Console`` to color role labels, dim reasoning, and
highlight stats. Falls back to plain ``print`` when ``rich`` is unavailable
so the chat REPL survives in minimal environments.
"""

from __future__ import annotations

from typing import Any


class ChatPrinter:
    """Stream chat output with role headers, color-coded stats, and graceful fallback.

    Streaming is intentionally plain-text. Mid-stream markdown rendering tends
    to flicker or eat partial fences, so we keep the live output as text and
    rely on terminal soft-wrap for readability. Markdown-style content (fenced
    code blocks, headings) still appears literally during streaming, which is
    a familiar OpenCode/Claude Code convention.
    """

    def __init__(self, *, console: Any | None = None) -> None:
        try:
            from rich.console import Console
        except ImportError:  # pragma: no cover - exercised only without rich
            self._console = None
        else:
            self._console = console or Console()
        self._streaming = False

    # ------- general printing --------------------------------------------------
    def print_info(self, text: str, *, dim: bool = False) -> None:
        if self._console is None:
            print(text)
            return
        if dim:
            self._console.print(f"[dim]{text}[/dim]", highlight=False)
        else:
            self._console.print(text, highlight=False)

    def print_warning(self, text: str) -> None:
        if self._console is None:
            print(f"warning: {text}")
            return
        self._console.print(f"[bold yellow]warning:[/bold yellow] {text}", highlight=False)

    def print_error(self, text: str) -> None:
        if self._console is None:
            print(f"error: {text}")
            return
        self._console.print(f"[bold red]error:[/bold red] {text}", highlight=False)

    def rule(self, text: str | None = None) -> None:
        if self._console is None:
            print("─" * 40)
            return
        from rich.rule import Rule  # type: ignore

        self._console.print(Rule(text or "", style="dim"))

    # ------- conversational turns ---------------------------------------------
    def print_user(self, text: str) -> None:
        if self._console is None:
            print()
            print(f"you: {text}")
            return
        self._console.print()
        self._console.print("[bold]you[/bold]", highlight=False)
        self._console.print(text, highlight=False, soft_wrap=True)

    def begin_assistant(self) -> None:
        self._streaming = True
        if self._console is None:
            print()
            print("MTPLX:")
            return
        self._console.print()
        self._console.print("[bold cyan]MTPLX[/bold cyan]", highlight=False)

    def stream_chunk(self, text: str) -> None:
        if not text:
            return
        if self._console is None:
            print(text, end="", flush=True)
            return
        # ``soft_wrap=True`` lets the terminal break at word boundaries while we
        # stream raw text; ``highlight=False`` prevents rich's auto-highlighter
        # from re-styling parts of the response.
        self._console.print(text, end="", soft_wrap=True, highlight=False, markup=False)

    def end_assistant(self) -> None:
        self._streaming = False
        if self._console is None:
            print()
            return
        self._console.print()

    # ------- reasoning channel ------------------------------------------------
    def begin_reasoning(self) -> None:
        if self._console is None:
            print("[reasoning]")
            return
        self._console.print("[dim italic]reasoning[/dim italic]", highlight=False)

    def stream_reasoning_chunk(self, text: str) -> None:
        if not text:
            return
        if self._console is None:
            print(text, end="", flush=True)
            return
        self._console.print(
            f"[dim]{text}[/dim]" if "[" not in text else text,
            end="",
            soft_wrap=True,
            highlight=False,
            markup=("[" not in text),
        )

    def end_reasoning(self) -> None:
        if self._console is None:
            print()
            return
        self._console.print()

    # ------- stats footer -----------------------------------------------------
    def print_stats(
        self,
        *,
        tok_s: float | None = None,
        accept_rate: float | None = None,
        depth: int | None = None,
        validator: str | None = None,
        time_to_first_token_s: float | None = None,
        elapsed_s: float | None = None,
    ) -> None:
        parts: list[str] = []
        if tok_s is not None:
            parts.append(f"{tok_s:.1f} tok/s")
        if accept_rate is not None:
            parts.append(f"{accept_rate * 100:.0f}% accept")
        if depth is not None:
            parts.append(f"depth {depth}")
        if time_to_first_token_s is not None:
            parts.append(f"ttft {time_to_first_token_s * 1000:.0f}ms")
        if elapsed_s is not None:
            parts.append(f"elapsed {elapsed_s:.1f}s")
        if validator:
            parts.append(validator)
        if not parts:
            return

        if self._console is None:
            print("  " + " · ".join(parts))
            print()
            return

        rich_parts: list[str] = []
        for part in parts:
            if "tok/s" in part:
                rich_parts.append(f"[green]{part}[/green]")
            elif "accept" in part:
                rich_parts.append(f"[yellow]{part}[/yellow]")
            elif part in {"pass", "validator pass"}:
                rich_parts.append(f"[green]{part}[/green]")
            elif part in {"fail", "validator fail"}:
                rich_parts.append(f"[red]{part}[/red]")
            else:
                rich_parts.append(f"[dim]{part}[/dim]")
        self._console.print("  " + " · ".join(rich_parts), highlight=False)
        self._console.print()
