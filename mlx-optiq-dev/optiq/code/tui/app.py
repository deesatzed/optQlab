"""The OptiQ Code TUI — a Textual app that drives the agent loop (design §3.4).

Streaming transcript, tool-call display, an approval modal with a diff preview
for mutating tools, git awareness, and the emerald banner. The loop runs as a
worker so the UI stays responsive; blocking engine/pytest calls are pushed to a
thread via `run_sync`.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from rich.markup import escape
from textual.widgets import Button, Label, OptionList, RichLog, Static, TextArea
from textual.widgets.option_list import Option

from ..approval import (ApprovalMode, ApprovalPolicy, Decision, ToolCall)
from ..banner import EMERALD, banner_markup
from ..exec_env import as_executor
from ..loop import run_agent


#: Slash commands the TUI understands. name -> (aliases, one-line help).
SLASH_COMMANDS: dict = {
    "help":    (("?", "h"),    "show this help"),
    "init":    ((),            "analyze the repo and write an AGENTS.md guide"),
    "model":   (("status",),   "model, endpoint, and context window"),
    "resume":  ((),            "list past conversations here; /resume N loads one"),
    "copy":    (("yank",),     "copy the agent's last reply to the clipboard"),
    "compact": ((),            "compact the context now (drop old tool output)"),
    "clear":   ((),            "clear the screen and start a fresh context"),
    "quit":    (("exit", "q"), "exit (also Ctrl-C)"),
}
_SLASH_ALIAS = {a: name for name, (aliases, _) in SLASH_COMMANDS.items()
                for a in (name, *aliases)}


def parse_slash(text: str) -> str | None:
    """Return the canonical command for a ``/command`` line, or None.

    None means "not a command" (empty, or does not start with '/') and the text
    should go to the model as a goal. An unrecognized ``/foo`` returns the
    literal ``"foo"`` so the caller can say "unknown command" rather than send
    it to the model -- a user who typed a slash meant a command, not a task.
    """
    s = (text or "").strip()
    if not s.startswith("/"):
        return None
    word = s[1:].split()[0].lower() if s[1:].split() else ""
    return _SLASH_ALIAS.get(word, word)


def _fmt_tokens(n: int) -> str:
    """Compact token count: 1420 -> '1.4k'."""
    return f"{n / 1000:.1f}k" if n >= 1000 else str(int(n))


def _count_changed_files(patch: str) -> int:
    """Number of files in a unified diff, by counting ``diff --git`` headers."""
    if not patch:
        return 0
    return sum(1 for ln in patch.splitlines() if ln.startswith("diff --git"))


class PromptArea(TextArea):
    """The prompt input — a TextArea so it holds multiple lines (a paste keeps its
    newlines), but Enter submits like a chat box. Shift+Enter / Ctrl+J insert a
    newline; ↑/↓ move between lines and recall history only at the top/bottom edge;
    Tab completes an open autocomplete menu; Shift+Tab cycles approval. All of
    these delegate to methods on the app, so the app owns the behavior and the
    widget owns only the key routing."""

    async def _on_key(self, event: events.Key) -> None:
        app = self.app
        key = event.key
        if key == "enter":
            event.prevent_default(); event.stop()
            app.submit_prompt(self.text)
            return
        if key in ("shift+enter", "ctrl+j"):
            event.prevent_default(); event.stop()
            self.insert("\n")
            return
        if key in ("shift+tab", "backtab"):
            event.prevent_default(); event.stop()
            app.action_cycle_approval()
            return
        # Keyboard scroll of the transcript. With the mouse released (native
        # selection, the default), the wheel doesn't scroll the app, so PageUp/
        # PageDown (and Ctrl+U / Ctrl+D) drive the log instead.
        if key in ("pageup", "pagedown", "ctrl+u", "ctrl+d"):
            event.prevent_default(); event.stop()
            app.scroll_transcript(up=key in ("pageup", "ctrl+u"))
            return
        if key in ("ctrl+home", "ctrl+end"):
            event.prevent_default(); event.stop()
            app.scroll_transcript_edge(top=key == "ctrl+home")
            return
        if key == "tab" and getattr(app, "_menu_mode", None):
            event.prevent_default(); event.stop()
            app.complete_menu()
            return
        row = self.cursor_location[0]
        if key == "up" and row == 0:
            event.prevent_default(); event.stop()
            app.history_prev()
            return
        last_line = getattr(getattr(self, "document", None), "line_count", 1) - 1
        if key == "down" and row >= last_line:
            event.prevent_default(); event.stop()
            app.history_next()
            return
        await super()._on_key(event)


def _git_status(repo) -> str:
    """`branch · clean|N changed` for the status bar. Best-effort.

    Change count is scoped to the launch folder (``-- .``) so a project inside a
    larger repo reports its own changes, not the whole monorepo's."""
    ex = as_executor(repo)
    rc, br = ex.run("git rev-parse --abbrev-ref HEAD 2>/dev/null", timeout=10)
    branch = br.strip() if rc == 0 and br.strip() else "—"
    rc, out = ex.run("git status --porcelain -- . 2>/dev/null", timeout=10)
    n = len([l for l in out.splitlines() if l.strip()]) if rc == 0 else 0
    return f"{branch} · {'clean' if n == 0 else f'{n} changed'}"


class ApprovalModal(ModalScreen[Decision]):
    """Ask the human before a mutating tool runs, showing what it will do."""

    BINDINGS = [
        Binding("enter,y", "allow_once", "Approve"),
        Binding("a", "allow_session", "Approve for session"),
        Binding("n,escape", "deny", "Deny"),
    ]

    def __init__(self, call: ToolCall):
        super().__init__()
        self._call = call

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-modal"):
            yield Label(f"[b]Approve[/]  [{EMERALD}]{escape(self._call.summary(80))}[/]", id="approval-title")
            # the preview is code/diff (contains [..]); render it as literal text
            yield Static(self._call.preview(max_lines=18, width=76), id="approval-preview", markup=False)
            yield Label("[dim][b]Enter[/b] approve · [b]a[/b] approve for session · [b]Esc[/b] deny[/]", id="approval-help")

    def action_allow_once(self) -> None:
        self.dismiss(Decision(allow=True))

    def action_allow_session(self) -> None:
        self.dismiss(Decision(allow=True, remember=True))

    def action_deny(self) -> None:
        self.dismiss(Decision(allow=False, note="denied by user"))


class PickerModal(ModalScreen):
    """A keyboard-navigable list picker (↑/↓ to move, Enter to pick, Esc to
    cancel). Dismisses with the chosen option's id, or None on cancel. Reused for
    /resume (pick a conversation) and double-Esc rewind (pick a turn)."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, title: str, options: list, hint: str = "") -> None:
        super().__init__()
        self._title = title
        self._options = options            # list[(id, label)]
        self._hint = hint or "↑/↓ to choose · Enter to select · Esc to cancel"

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-modal"):
            yield Label(f"[b]{escape(self._title)}[/]", id="picker-title")
            yield OptionList(*[Option(label, id=oid) for oid, label in self._options],
                             id="picker-list")
            yield Label(f"[dim]{self._hint}[/]", id="picker-help")

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class OptiqCodeApp(App):
    """The interactive OptiQ Code app."""

    TITLE = "OptiQ Code"

    CSS = """
    Screen { layout: vertical; }
    /* The conversation fills the screen; the welcome scrolls away as you work. */
    #log { height: 1fr; padding: 1 2; }
    /* A single full-width input bar (Claude-Code style); grows with its content
       (multiline paste / Shift+Enter). */
    #goal { border: round #03684c; height: auto; max-height: 12; margin: 0 1; }
    /* Slash-command autocomplete: sits just above the status bar, hidden until
       the input starts with '/'. */
    #slash-menu { height: auto; max-height: 8; padding: 0 3; color: $text-muted;
        background: $panel; display: none; }
    #slash-menu.open { display: block; }
    #status { height: 1; padding: 0 2; color: $text-muted; background: $panel; }
    #approval-modal { align: center middle; width: 84; height: auto; max-height: 80%;
        border: thick #03684c; background: $surface; padding: 1 2; }
    #approval-title { padding-bottom: 1; }
    #picker-modal { align: center middle; width: 90; height: auto; max-height: 80%;
        border: thick #03684c; background: $surface; padding: 1 2; }
    #picker-title { padding-bottom: 1; }
    #picker-list { height: auto; max-height: 20; }
    #picker-help { padding-top: 1; }
    #approval-preview { border: round $panel; padding: 0 1; height: auto; max-height: 24; }
    #approval-help { padding-top: 1; }
    """

    BINDINGS = [
        # ctrl+c: clear the line, else interrupt a run, else press-twice to exit
        # (Claude-Code-style) — priority so it beats Textual's built-in quit.
        Binding("ctrl+c", "ctrl_c", "Quit", show=False, priority=True),
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("escape", "interrupt", "Interrupt", show=False),
    ]

    def __init__(self, repo, engine, *, verify_command: str | None = None,
                 mode: ApprovalMode = ApprovalMode.APPROVE, model_name: str | None = None,
                 max_turns: int = 20, max_tokens: int = 16384,
                 max_tool_output: int | None = None, session=None,
                 compact_at: int | None = None):
        super().__init__()
        self.repo = Path(repo)
        self.engine = engine
        self.verify_command = verify_command
        self.policy = ApprovalPolicy(mode=mode)
        self.model_name = model_name or getattr(engine, "model_id", None)
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.max_tool_output = max_tool_output
        self.compact_at = compact_at
        self.session = session
        # A resumed session seeds the trajectory so the model keeps its context.
        self._messages: list = session.load_messages() if session is not None else []
        self._busy = False
        self._slash_menu_items: list = []
        self._history: list = []            # past inputs, for ↑/↓ recall
        self._hist_pos: int | None = None   # cursor into _history while browsing
        self._draft = ""                    # in-progress line, saved while browsing
        self._reasoning_buf: list = []      # live thinking deltas (thread-appended)
        self._cost = 0.0                    # cumulative $ this session (if reported)
        self._in_tokens = 0                 # prompt tokens of the last model call (↑)
        self._menu_mode: str | None = None  # "slash" | "at" — what the menu shows
        self.sub_title = self.repo.name

    def compose(self) -> ComposeResult:
        yield RichLog(id="log", markup=True, wrap=True, highlight=False)
        yield PromptArea(id="goal", soft_wrap=True, show_line_numbers=False,
                         tab_behavior="focus",
                         placeholder="Describe a task — or /help for commands")
        yield Static(id="slash-menu")
        yield Static(self._status_text(), id="status")

    def on_mount(self) -> None:
        # The prompt starts focused. The transcript keeps Textual's native mouse
        # selection (so you can select + copy from it); a stray keystroke while
        # the log is focused is redirected back into the prompt (see on_key), so
        # typing still "just works" WITHOUT disabling selection.
        self.query_one("#goal", PromptArea).focus()
        try:
            self._ctx_window = self.engine.context_window()   # cache once for the ctx gauge
        except Exception:
            self._ctx_window = None
        if self._messages:
            self._seed_history_from_messages()
            self._replay()                      # resumed session
        else:
            self._welcome()
            if self.session is not None:
                import json
                self.session.append("meta", json.dumps(
                    {"model": self.model_name, "repo": str(self.repo)}))

    def _welcome(self) -> None:
        """The full banner, written into the transcript at session start so it
        scrolls away as the conversation grows (Claude-Code-style), not a fixed
        header."""
        log = self.query_one("#log", RichLog)
        log.write(banner_markup(repo_name=self.repo.name, model=self.model_name))
        log.write(f"  [dim]git[/]    [b]{_git_status(self.repo)}[/]")
        log.write("")
        log.write("  [dim]Enter to send · [/][b]/help[/][dim] for commands · "
                  "trackpad/wheel to scroll · Fn- or ⌥-drag to select (terminal-"
                  "dependent) · Ctrl-C to quit.[/]")

    def _replay(self) -> None:
        """Render the resumed conversation into the transcript so `-c` / `-r`
        show the prior turns — goals, the model's replies, tool calls, and tool
        output — not just a one-line summary. This is what "see the previous
        conversation" needs; the trajectory itself still seeds the model context."""
        log = self.query_one("#log", RichLog)
        log.write(banner_markup(repo_name=self.repo.name, model=self.model_name))
        sid = self.session.id if self.session is not None else "?"
        log.write(f"  [dim]resumed[/] [b]{sid}[/] [dim]· {len(self._messages)} messages[/]")
        for m in self._messages:
            self._replay_message(log, m)
        log.write("")
        log.write("  [dim]Continue — describe the next task, or press ↑ for a "
                  "previous one.[/]")

    def _replay_message(self, log, m: dict) -> None:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "user":
            # Only real user turns are wrapped as "GOAL:"; a plain user message
            # is an internal harness nudge ("You did not call any tool…") and
            # must not render as a ▶ turn.
            if not content.startswith("GOAL:"):
                return
            goal = content.split("GOAL:", 1)[1].split("\n\nPROJECT")[0].strip()
            if goal:
                log.write(f"\n[{EMERALD}]▶[/] [b]{escape(goal)}[/]")
        elif role == "assistant":
            if content:
                log.write(escape(content))
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
                name = (fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)) or "tool"
                if name == "done":
                    self._replay_done_summary(log, fn)   # the agent's answer
                else:
                    log.write(f"[dim]  ⚙ {escape(str(name))}[/]")
        elif role == "tool" and content:
            first = content.splitlines()[0] if content.splitlines() else ""
            log.write(f"[dim]  ⎿ {escape(first[:100])}[/]")

    def _replay_done_summary(self, log, fn) -> None:
        """Render the summary the agent finished with — for a question this IS the
        answer, which would otherwise be hidden inside the done tool call."""
        import json as _json
        raw = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", "")
        try:
            summ = (_json.loads(raw or "{}") or {}).get("summary", "").strip()
        except Exception:
            summ = ""
        if not summ or summ == "done":
            return
        from rich.markdown import Markdown
        try:
            log.write(Markdown(summ))
        except Exception:
            log.write(escape(summ))

    def _seed_history_from_messages(self) -> None:
        """Populate ↑/↓ history with the real goals from a resumed session (the
        ones wrapped as GOAL:), so a user can recall and re-run a previous prompt
        across restarts — never the internal harness nudges."""
        for m in self._messages:
            if m.get("role") != "user":
                continue
            c = (m.get("content") or "").strip()
            if not c.startswith("GOAL:"):
                continue
            goal = c.split("GOAL:", 1)[1].split("\n\nPROJECT")[0].strip()
            if goal and (not self._history or self._history[-1] != goal):
                self._history.append(goal)

    def on_click(self, event) -> None:
        """Return focus to the prompt on a plain click — UNLESS a text selection
        is active, so a mouse drag to select (and Ctrl-C to copy) in the transcript
        is preserved. A selection ends its own click with text selected, so we see
        it and leave focus alone; a bare click has none, and typing resumes."""
        try:
            if self.screen.get_selected_text():
                return
            ta = self._prompt()
            if self.focused is not ta:
                ta.focus()
        except Exception:
            pass

    # ── interrupt / quit / approval-mode ──────────────────────────────────────

    def scroll_transcript(self, *, up: bool) -> None:
        """Page the transcript with the keyboard (PageUp/PageDown, Ctrl+U/D) —
        the way to scroll back when the mouse is released for native selection."""
        try:
            log = self.query_one("#log", RichLog)
            log.scroll_page_up() if up else log.scroll_page_down()
        except Exception:
            pass

    def scroll_transcript_edge(self, *, top: bool) -> None:
        """Jump to the very top (Ctrl+Home) or bottom (Ctrl+End) of the full,
        uncapped transcript buffer."""
        try:
            log = self.query_one("#log", RichLog)
            log.scroll_home() if top else log.scroll_end()
        except Exception:
            pass

    def _interrupt(self) -> None:
        """Cancel the running turn without leaving the session. The worker gets
        CancelledError; `_run` catches it, resets state, and says so. A tool
        already running in a thread finishes detached (its result is discarded)."""
        self.workers.cancel_all()

    def action_interrupt(self) -> None:
        """Esc — interrupt a running turn; clear a typed line; else double-Esc to
        rewind to the previous turn (Claude-Code style)."""
        if self._busy:
            self._interrupt()
            return
        if self._prompt().text:
            self._set_prompt("")
            self._hist_pos = None
            self._close_slash_menu()
            return
        import time
        now = time.monotonic()
        if now - getattr(self, "_last_esc", 0.0) < 0.6:      # second Esc → rewind
            self._last_esc = 0.0
            self.action_rewind()
        else:
            self._last_esc = now
            self._write("  [dim]press Esc again to pick a turn to rewind to[/]")

    def action_rewind(self) -> None:
        """Open an ↑/↓ picker of the conversation's turns; the chosen turn (and
        everything after it) is dropped and its goal restored to the prompt to
        edit and resend — jump back to any point, not just the last turn."""
        turns = []
        for i, m in enumerate(self._messages):
            if m.get("role") == "user" and (m.get("content") or "").startswith("GOAL:"):
                goal = m["content"].split("GOAL:", 1)[1].split("\n\nPROJECT")[0].strip()
                turns.append((str(i), goal))
        if not turns:
            self._write("  [dim]nothing to rewind to.[/]")
            return
        opts = [(idx, goal[:80]) for idx, goal in reversed(turns)]   # newest first
        self.push_screen(
            PickerModal("Rewind to a turn", opts,
                        hint="↑/↓ · Enter to rewind here · Esc to cancel"),
            lambda idx: self._rewind_to(int(idx)) if idx is not None else None)

    def _rewind_to(self, index: int) -> None:
        """Drop message ``index`` and everything after it, restoring its goal to
        the prompt."""
        goal = (self._messages[index].get("content", "")
                .split("GOAL:", 1)[1].split("\n\nPROJECT")[0].strip())
        self._messages = self._messages[:index]
        self._rerender()
        self._set_prompt(goal)
        self._in_tokens = 0
        self._refresh_status()
        self._write("  [dim]rewound — edit and press Enter, or Esc·Esc to pick "
                    "another point.[/]")

    def _rerender(self) -> None:
        """Clear the transcript and repaint it from the current trajectory, then
        return focus to the prompt (with the mouse released you can't click back,
        so a lost focus would strand you needing Ctrl-C to quit)."""
        log = self.query_one("#log", RichLog)
        log.clear()
        log.write(banner_markup(repo_name=self.repo.name, model=self.model_name))
        if self._messages:
            for m in self._messages:
                self._replay_message(log, m)
        log.write("")
        try:
            self.query_one("#goal", PromptArea).focus()
        except Exception:
            pass

    def action_ctrl_c(self) -> None:
        """Ctrl-C — copy the transcript selection if there is one (terminal-style);
        else clear the line, else interrupt a run, else press twice to exit."""
        try:
            selected = self.screen.get_selected_text()
        except Exception:
            selected = None
        if selected:
            self.copy_to_clipboard(selected)
            try:
                self.screen.clear_selection()
            except Exception:
                pass
            self._write("  [dim]copied selection to clipboard[/]")
            return
        if self._prompt().text:
            self._set_prompt("")
            self._hist_pos = None
            self._close_slash_menu()
            return
        if self._busy:
            self._interrupt()
            return
        import time
        now = time.monotonic()
        if now - getattr(self, "_last_ctrl_c", 0.0) < 1.5:
            self.exit()
        else:
            self._last_ctrl_c = now
            self._write("  [dim]press Ctrl-C again to exit[/]")

    def action_cycle_approval(self) -> None:
        """Shift-Tab — toggle between approving each edit and auto-approving all
        for the session, so you can drop the modal once you trust the run."""
        self.policy.mode = (ApprovalMode.AUTO
                            if self.policy.mode == ApprovalMode.APPROVE
                            else ApprovalMode.APPROVE)
        # Reflect the mode in the status bar only — writing a line to the
        # transcript on every Shift-Tab press spammed the panel.
        try:
            self.query_one("#status", Static).update(self._status_text())
        except Exception:
            pass

    def _ctx_str(self) -> str:
        """`ctx N% (used/window)` from the last call's prompt tokens — so growth
        and the drop after /compact (or auto-compaction) are both visible."""
        used = self._in_tokens
        if not used:
            return ""
        win = getattr(self, "_ctx_window", None)
        if win and win > 0:
            pct = min(100, int(round(used * 100 / win)))
            return f"   [dim]ctx {pct}% ({_fmt_tokens(used)}/{_fmt_tokens(win)})[/]"
        return f"   [dim]ctx {_fmt_tokens(used)}[/]"

    def _status_text(self) -> str:
        auto = self.policy.mode == ApprovalMode.AUTO
        appr = "auto-approve" if auto else "approve edits"
        cost = f"   [dim]${self._cost:.4f}[/]" if self._cost > 0 else ""
        return (f"repo [b]{self.repo.name}[/]   model [b]{self.model_name or '—'}[/]   "
                f"git [b]{_git_status(self.repo)}[/]   [dim]{appr} · shift+tab[/]"
                f"{self._ctx_str()}{cost}")

    def _write(self, msg: str) -> None:
        """Write the app's OWN markup (welcome, status, decorations)."""
        self.query_one("#log", RichLog).write(msg)

    def _log(self, msg: str) -> None:
        """Write loop output — model text, tool results — which is DATA, not
        markup, so escape it (code contains [..] that Rich would otherwise eat)."""
        self.query_one("#log", RichLog).write(escape(str(msg)))

    def _reasoning(self, delta: str) -> None:
        """Collect thinking deltas. Fired from the engine's worker THREAD, so this
        only appends (atomic) — the _tick timer on the UI thread renders it into
        the ephemeral status line. Thinking is never written to the transcript, so
        a verbose reasoning model can't crowd out tool calls and answers."""
        self._reasoning_buf.append(delta)

    def _assistant_text(self, text: str) -> None:
        """Render the model's message text as Markdown (headings, lists, and
        syntax-highlighted code), uncapped — replaces the old 300-char plain-text
        clip. Runs on the UI event loop (loop body), so writing is safe."""
        if not text:
            return
        from rich.markdown import Markdown
        try:
            self.query_one("#log", RichLog).write(Markdown(text))
        except Exception:
            self._log(text)          # fall back to plain if markdown chokes

    def _usage(self, u: dict) -> None:
        """Record usage from a completed model call: accumulate cost and remember
        the prompt-token count so the status bar can show ↑ input (paid cloud
        endpoints report both; local servers may report neither)."""
        try:
            self._cost += float(u.get("cost") or 0.0)
            if u.get("prompt_tokens"):
                self._in_tokens = int(u["prompt_tokens"])
        except (TypeError, ValueError, AttributeError):
            pass

    def _progress(self, elapsed: float, tokens: int) -> None:
        """Record the current token count as the response streams. Fired from the
        worker thread; a plain int store is all it does, so no marshaling is
        needed. The elapsed clock is driven by _tick on a timer instead, because
        a tool-call turn arrives as a single delta at the end — so a
        delta-driven clock would not tick during the (long) generation."""
        self._tok = tokens

    async def _on_edit(self, name: str, path: str, code: str) -> None:
        """Reveal the code of a write/edit tool call as it lands, a few lines at a
        time, so the terminal shows the edit take shape (the model buffers the
        tool call, so the whole file arrives at once — this animates it) instead
        of only a 'wrote N bytes' summary. Runs on the event loop, so RichLog
        writes are safe."""
        import asyncio
        log = self.query_one("#log", RichLog)
        verb = "writing" if name == "write_file" else "editing"
        log.write(f"[dim]  ┌─ {verb} [/][b]{escape(path)}[/]")
        lines = code.splitlines()
        shown, cap = lines[:80], len(lines)
        buf = []
        for i, ln in enumerate(shown):
            buf.append(ln)
            if len(buf) >= 3 or i == len(shown) - 1:
                log.write("[#7ee787]" + escape("\n".join(buf)) + "[/]")
                buf = []
                await asyncio.sleep(0.035)
        if cap > 80:
            log.write(f"[dim]  … (+{cap - 80} more lines)[/]")
        log.write("[dim]  └─[/]")

    def _tick(self) -> None:
        """Timer callback (UI thread): refresh the live 'working' status so the
        elapsed clock ticks even while the server buffers a tool call. Both
        elapsed and tokens are cumulative for the whole turn (the loop banks each
        sub-step's tokens), Claude-Code-style."""
        import time
        elapsed = time.monotonic() - self._gen_start
        up = f"↑ [b]{_fmt_tokens(self._in_tokens)}[/]  " if self._in_tokens else ""
        down = _fmt_tokens(self._tok)
        # Ephemeral thinking peek: the tail of the reasoning stream on ONE line,
        # so a reasoning model shows live progress without filling the transcript.
        think = ""
        if self._reasoning_buf:
            tail = "".join(self._reasoning_buf)[-80:].replace("\n", " ").strip()
            if tail:
                think = f"   [dim]✻ {escape(tail)}[/]"
        icon = "✻" if self._reasoning_buf else "●"
        cost = f"   [dim]${self._cost:.4f}[/]" if self._cost > 0 else ""
        try:
            self.query_one("#status", Static).update(
                f"[{EMERALD}]{icon}[/] {'thinking' if self._reasoning_buf else 'working'}   "
                f"[b]{elapsed:.0f}s[/]   {up}↓ [b]{down}[/] tokens{cost}{think}")
        except Exception:
            pass

    # ── prompt access ─────────────────────────────────────────────────────────

    def _prompt(self) -> "PromptArea":
        return self.query_one("#goal", PromptArea)

    def _set_prompt(self, text: str) -> None:
        ta = self._prompt()
        ta.text = text
        try:
            ta.move_cursor(ta.document.end)   # cursor to end after a set/recall
        except Exception:
            pass

    def submit_prompt(self, text: str) -> None:
        """Enter — dispatch the prompt (called by PromptArea on Enter). Records it
        for ↑/↓ recall, clears the box, and runs it as a slash command or a goal."""
        self._close_slash_menu()
        if text.strip():                     # remember for ↑/↓ recall (goals + commands)
            if not self._history or self._history[-1] != text:
                self._history.append(text)
            self._hist_pos = None
            self._draft = ""
        self._set_prompt("")
        cmd = parse_slash(text)
        if cmd is not None:
            after = text.strip()[1:].split(None, 1)   # '/cmd rest' -> ['cmd', 'rest']
            arg = after[1].strip() if len(after) > 1 else ""
            self._handle_slash(cmd, arg)
            return
        self.start(text)

    def history_prev(self) -> None:
        """↑ (on the first line) — recall an older input (shell-style)."""
        if not self._history:
            return
        if self._hist_pos is None:
            self._draft = self._prompt().text    # save the in-progress line
            self._hist_pos = len(self._history)
        if self._hist_pos > 0:
            self._hist_pos -= 1
            self._set_prompt(self._history[self._hist_pos])

    def history_next(self) -> None:
        """↓ (on the last line) — toward the newest input, then the saved draft."""
        if self._hist_pos is None:
            return
        self._hist_pos += 1
        if self._hist_pos >= len(self._history):
            self._hist_pos = None
            self._set_prompt(self._draft)
        else:
            self._set_prompt(self._history[self._hist_pos])

    # ── slash-command autocomplete ────────────────────────────────────────────

    def _slash_prefix(self, value: str) -> str | None:
        """The command fragment being typed (``/mo`` -> ``mo``), or None if the
        line isn't a bare command word (empty, multiline, no '/', or past the
        word). Slash commands are single-line, so a newline rules one out."""
        s = (value or "").lstrip()
        if "\n" in s or not s.startswith("/") or " " in s.strip():
            return None
        return s[1:].lower()

    def _slash_matches(self, prefix: str) -> list[str]:
        return [n for n in SLASH_COMMANDS if n.startswith(prefix)]

    def _repo_files(self) -> list[str]:
        """Tracked repo files (git ls-files), cached once, for @-mention
        completion. Falls back to a shallow walk when the folder isn't git."""
        cached = getattr(self, "_repo_files_cache", None)
        if cached is not None:
            return cached
        files: list[str] = []
        try:
            rc, out = as_executor(self.repo).run("git ls-files 2>/dev/null", timeout=10)
            if rc == 0 and out.strip():
                files = [ln.strip() for ln in out.splitlines() if ln.strip()]
        except Exception:
            files = []
        if not files:
            import os
            skip = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}
            for root, dirs, names in os.walk(self.repo):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in skip]
                for n in names:
                    files.append(os.path.relpath(os.path.join(root, n), self.repo))
                if len(files) > 5000:
                    break
        self._repo_files_cache = files
        return files

    def _at_prefix(self, value: str) -> str | None:
        """The @-mention fragment at the end of the current token (text after the
        last '@' on the current line), or None. '' right after '@' opens the list."""
        if not value:
            return None
        line = value.split("\n")[-1]         # the line being edited
        tail = line.rsplit(" ", 1)[-1]
        at = tail.rfind("@")
        return tail[at + 1:] if at >= 0 else None

    def _file_matches(self, prefix: str) -> list[str]:
        p = (prefix or "").lower()
        files = self._repo_files()
        if not p:
            return files[:8]
        starts = [f for f in files if f.lower().startswith(p)]
        contains = [f for f in files if p in f.lower() and not f.lower().startswith(p)]
        return (starts + contains)[:8]

    def _complete_at(self, value: str, path: str) -> str:
        """Replace the trailing @fragment in `value` with `@<path> `."""
        at = value.rfind("@")
        return value[:at] + "@" + path + " " if at >= 0 else value

    def on_text_area_changed(self, event) -> None:
        if getattr(event.text_area, "id", None) != "goal":
            return
        self._update_menus(event.text_area.text)

    def _update_menus(self, val: str) -> None:
        try:
            menu = self.query_one("#slash-menu", Static)
        except Exception:
            return          # a late Changed during teardown — nothing to update
        # 1) slash-command menu (line starts with '/')
        sp = self._slash_prefix(val)
        if sp is not None:
            matches = self._slash_matches(sp)
            if matches:
                self._menu_mode = "slash"
                self._slash_menu_items = matches         # exposed for tests
                rows = [f"[b]/{n}[/][dim]{' ' * max(1, 12 - len(n))}"
                        f"{SLASH_COMMANDS[n][1]}[/]" for n in matches]
                rows.append("[dim]Tab to complete · Enter to run[/]")
                menu.update("\n".join(rows))
                menu.add_class("open")
                return
        # 2) @-file mention menu (a '@fragment' anywhere in the line)
        ap = self._at_prefix(val)
        if ap is not None:
            files = self._file_matches(ap)
            if files:
                self._menu_mode = "at"
                self._slash_menu_items = files
                rows = [f"[b]@{escape(f)}[/]" for f in files]
                rows.append("[dim]Tab to complete[/]")
                menu.update("\n".join(rows))
                menu.add_class("open")
                return
        self._menu_mode = None
        self._slash_menu_items = []
        self._close_slash_menu()

    def _close_slash_menu(self) -> None:
        try:
            self.query_one("#slash-menu", Static).remove_class("open")
        except Exception:
            pass

    def complete_menu(self) -> None:
        """Tab — complete the open autocomplete menu (slash command or @-file)
        into the prompt. Called by PromptArea when a menu is open."""
        text = self._prompt().text
        if self._menu_mode == "slash":
            matches = self._slash_matches(self._slash_prefix(text) or "")
            if matches:
                self._set_prompt(f"/{matches[0]} ")
                self._close_slash_menu()
        elif self._menu_mode == "at":
            ap = self._at_prefix(text)
            files = self._file_matches(ap) if ap is not None else []
            if files:
                self._set_prompt(self._complete_at(text, files[0]))
                self._close_slash_menu()

    def _handle_slash(self, cmd: str, arg: str = "") -> None:
        """Run an in-TUI slash command. ``cmd`` is already canonicalized by
        :func:`parse_slash`; ``arg`` is any text after it. An unknown name gets a
        hint, not sent to the model."""
        if self._busy:
            self._write("  [dim]busy — wait for the current task to finish.[/]")
            return
        if cmd == "help":
            self._cmd_help()
        elif cmd == "init":
            self._cmd_init()
        elif cmd == "model":
            self._cmd_model()
        elif cmd == "resume":
            self._cmd_resume(arg)
        elif cmd == "copy":
            self._cmd_copy()
        elif cmd == "compact":
            self._cmd_compact()
        elif cmd == "clear":
            self._cmd_clear()
        elif cmd == "quit":
            self.exit()
            return
        else:
            self._write(f"  [dim]unknown command [/][b]/{escape(cmd)}[/][dim] — "
                        f"try [/][b]/help[/]")
        # A command must never leave focus off the prompt — with the mouse
        # released the user can't click back in.
        try:
            self.query_one("#goal", PromptArea).focus()
        except Exception:
            pass

    def _cmd_help(self) -> None:
        self._write("")
        self._write("  [b]Commands[/]")
        for name, (aliases, desc) in SLASH_COMMANDS.items():
            # Show word aliases (e.g. /exit) so they're discoverable; hide the
            # single-letter shortcuts (/q /h /?) to keep the list clean.
            shown = "/" + name + "".join(f", /{a}" for a in aliases if len(a) > 1)
            self._write(f"  [b]{shown}[/][dim]{' ' * max(1, 16 - len(shown))}{desc}[/]")
        self._write("  [dim]Anything else you type is a task for the agent.[/]")
        self._write("")

    def _cmd_model(self) -> None:
        try:
            win = self.engine.context_window()
        except Exception:
            win = None
        base = getattr(self.engine, "base_url", "—")
        self._write("")
        self._write(f"  [b]model[/]     {escape(str(self.model_name or '—'))}")
        self._write(f"  [b]endpoint[/]  {escape(str(base))}")
        win_txt = f"{win} tokens" if win else "unknown"
        if win and self.compact_at:
            win_txt += f"  [dim]· compacting past ~{self.compact_at}[/]"
        self._write(f"  [b]context[/]   {win_txt}")
        self._write("")

    def _cmd_init(self) -> None:
        """Generate/refresh AGENTS.md by running the agent on the repo (Claude
        Code's /init). It runs as a normal goal, so it uses the write_file tool
        and the usual approval flow; ``AGENTS.md`` is then read into the system
        prompt at the start of every later run (loop._load_project_guidance)."""
        self.start(
            "Initialize agent guidance for this repository. First read any "
            "existing AGENTS.md or CLAUDE.md and skim the README, key config, and "
            "project layout for context. Then write a concise AGENTS.md at the "
            "repo root: what the project is, the main languages/frameworks, how to "
            "build and run the tests, and any conventions worth knowing. Fold in "
            "what those existing files already say — if AGENTS.md exists, refine "
            "it in place; if only CLAUDE.md exists, base the AGENTS.md on it. Keep "
            "it short and factual. Use write_file for AGENTS.md, then finish.")

    def _cmd_resume(self, arg: str = "") -> None:
        """Pick a past conversation in this repo and load it (Claude-Code's
        /resume). Bare `/resume` opens an ↑/↓ picker; `/resume N` loads the Nth
        directly. Switches the active session so further turns append to it."""
        from ..sessions import SessionStore
        store = SessionStore(self.repo)
        cur = self.session.id if self.session is not None else None
        sessions = [s for s in store.list() if s.id != cur][:20]
        if not sessions:
            self._write("  [dim]no other conversations in this repo to resume.[/]")
            return
        if arg.strip().isdigit():                       # typed shortcut: /resume N
            n = int(arg.strip())
            if 1 <= n <= len(sessions):
                self._load_session(sessions[n - 1].id)
            else:
                self._write(f"  [dim]pick 1–{len(sessions)}.[/]")
            return
        opts = [(s.id, s.title or "(empty)") for s in sessions]
        self.push_screen(PickerModal("Resume a conversation", opts),
                         lambda sid: self._load_session(sid) if sid else None)

    def _load_session(self, sid: str) -> None:
        from ..sessions import SessionStore
        sess = SessionStore(self.repo).load(sid)
        self.session = sess
        self._messages = sess.load_messages()
        self._history = []
        self._seed_history_from_messages()
        self._rerender()
        self._in_tokens = 0
        self._refresh_status()
        self._write(f"  [dim]resumed {sid} · {len(self._messages)} messages "
                    f"— continue below.[/]")

    def _cmd_copy(self) -> None:
        """Copy the agent's last text reply to the clipboard. Reliable in any
        terminal (native mouse-drag selection is captured by the TUI); for an
        arbitrary sub-selection, Option-drag uses the terminal's own selection."""
        last = ""
        for m in reversed(self._messages):
            if m.get("role") == "assistant" and (m.get("content") or "").strip():
                last = m["content"].strip()
                break
        if not last:
            self._write("  [dim]nothing to copy yet — no assistant reply.[/]")
            return
        self.copy_to_clipboard(last)
        self._write(f"  [dim]copied the last reply ({len(last)} chars) to the "
                    f"clipboard. (Option-drag to select a portion instead.)[/]")

    def _cmd_compact(self) -> None:
        """Compact the context now — the same drop-old-tool-output the loop does
        automatically at the budget, run on demand (Claude-Code's /compact). Reuses
        the loop primitive so manual and automatic compaction behave identically."""
        from ..loop import _approx_tokens, _compact
        if not self._messages:
            self._write("  [dim]nothing to compact yet.[/]")
            return
        before = _approx_tokens(self._messages)
        self._messages, dropped = _compact(self._messages)
        after = _approx_tokens(self._messages)
        self._in_tokens = after            # reflect the freed context in the ctx gauge
        self._refresh_status()
        if dropped:
            self._write(f"  [dim]compacted — dropped {len(dropped)} old tool "
                        f"result(s), ~{before - after} tokens freed.[/]")
        else:
            self._write("  [dim]already compact — nothing old enough to drop.[/]")

    def _cmd_clear(self) -> None:
        """Drop the in-memory trajectory so the next task starts a fresh context,
        and repaint. The on-disk session log is append-only and is left intact."""
        self._messages = []
        self._in_tokens = 0                # context reset — clear the ctx gauge
        log = self.query_one("#log", RichLog)
        log.clear()
        self._welcome()
        self._refresh_status()
        self._write("  [dim]context cleared — starting fresh.[/]")

    def _refresh_status(self) -> None:
        try:
            self.query_one("#status", Static).update(self._status_text())
        except Exception:
            pass

    def start(self, goal: str) -> bool:
        """Begin a run toward `goal` (used by the input handler and by tests)."""
        goal = (goal or "").strip()
        if not goal or self._busy:
            return False
        self._busy = True
        import time
        self._tok = 0
        self._reasoning_buf = []            # fresh thinking peek for this turn
        self._gen_start = time.monotonic()
        self._prog_timer = self.set_interval(0.5, self._tick)
        self._write(f"\n[{EMERALD}]▶[/] [b]{escape(goal)}[/]")
        if self.session is not None:
            self.session.append("input", goal)
        self.run_worker(self._run(goal), exclusive=True)
        return True

    async def _run(self, goal: str) -> None:
        import asyncio
        interrupted = False
        try:
            tool_output_kw = ({} if self.max_tool_output is None
                              else {"max_tool_output": self.max_tool_output})
            result = await run_agent(
                engine=self.engine, repo=self.repo, goal=goal,
                verify_command=self.verify_command, approve=self._approve,
                log=self._log, run_sync=self._run_sync, on_progress=self._progress,
                on_reasoning=self._reasoning, on_assistant_text=self._assistant_text,
                on_usage=self._usage, on_edit=self._on_edit, max_turns=self.max_turns,
                max_tokens=self.max_tokens, compact_at=self.compact_at,
                prior_messages=self._messages or None, **tool_output_kw)
        except asyncio.CancelledError:
            interrupted = True
        finally:
            self._busy = False
            if getattr(self, "_prog_timer", None) is not None:
                self._prog_timer.stop()
                self._prog_timer = None
        if interrupted:
            # Prior turns stay in self._messages, so context up to the last
            # completed turn is preserved; the cut-off turn is simply dropped.
            self._write("[yellow]⨯ interrupted[/] [dim]— type another task to continue.[/]")
            try:
                self.query_one("#status", Static).update(self._status_text())
            except Exception:
                pass
            return
        # accumulate the trajectory so the next goal keeps context; persist it
        self._messages = result.messages
        if self.session is not None:
            self.session.save_messages(self._messages)
        # No trailing combined-diff dump: edits are already shown inline as they
        # land (_on_edit), and the working tree is left changed for you to review
        # with your own git/editor — the same as Claude Code. The summary folds in
        # how many files changed; nothing is committed.
        color = EMERALD if result.succeeded else "yellow"
        changed = _count_changed_files(result.patch)
        # Only report what actually happened. A question (no tests, no file
        # changes) shows just "■ done" — the answer was already rendered.
        bits = []
        if result.passed or result.failed:
            bits.append(f"{result.passed} passed / {result.failed} failed")
        if changed:
            bits.append(f"{changed} file{'s' if changed != 1 else ''} changed")
        tail = ("  —  " + "  ·  ".join(bits)) if bits else ""
        self._write(f"[{color}]■ {result.stop_reason}[/]{tail}")
        self.query_one("#status", Static).update(self._status_text())

    async def _approve(self, call: ToolCall) -> bool:
        if not self.policy.needs_approval(call.name):
            return True
        decision: Decision = await self.push_screen_wait(ApprovalModal(call))
        if decision.allow and decision.remember:
            self.policy.allow_for_session(call.name)
        if not decision.allow:
            self._write(f"[yellow]✗ denied[/] {call.summary(70)}")
        return decision.allow

    async def _run_sync(self, fn):
        # Keep the UI responsive: blocking engine/pytest calls run in a thread.
        return await asyncio.get_running_loop().run_in_executor(None, fn)
