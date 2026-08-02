"""One place every OptiQ Code setting is defined, and one precedence order.

Settings had grown up ad hoc: seven environment variables read at six call
sites, some at import time (so a config file could never override them), CLI
flags whose names didn't match their env vars, and no config file at all. Every
new knob invented its own convention.

So the knobs are declared once, in :data:`FIELDS`, and resolved once:

    CLI flag  >  environment  >  repo config  >  user config  >  default

Config files are JSON:

* repo:  ``<repo>/.optiq/code.json``   -- checked in, shared by the team
* user:  ``~/.optiq/code/config.json`` -- personal, applies everywhere

Both hold a flat object keyed by the field names below::

    {"max_turns": 60, "compact_at": 24000, "reasoning_effort": "high"}

Every field gets a canonical ``OPTIQ_CODE_<NAME>`` environment variable.
Several shipped under other names first; those still work (see ``aliases``) and
win over config files, but the canonical name wins over them.

A value of ``None`` means "not set" and lets the next source down decide.
Unknown keys in a config file are reported, not ignored -- a typoed key that
silently does nothing is worse than a warning.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields as dc_fields
from pathlib import Path
from typing import Any, Callable, Optional


def _as_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return None


def _as_str(v: Any) -> Optional[str]:
    s = str(v).strip() if v is not None else ""
    return s or None


@dataclass(frozen=True)
class Field:
    name: str
    cast: Callable[[Any], Any]
    default: Any
    help: str
    aliases: tuple = ()      # legacy env vars, still honored

    @property
    def env(self) -> str:
        return f"OPTIQ_CODE_{self.name.upper()}"


FIELDS: tuple = (
    Field("base_url", _as_str, None,
          "OpenAI-compatible endpoint of the OptiQ server. Default: discover "
          "a local server, else http://127.0.0.1:8080/v1.",
          aliases=("OPTIQ_BASE_URL",)),
    Field("api_key", _as_str, None,
          "Bearer token for the server. Default: sk-optiq-local.",
          aliases=("OPTIQ_API_KEY",)),
    Field("model", _as_str, None,
          "Which served model to use. Default: whatever the server serves.",
          aliases=("OPTIQ_CODE_MODEL",)),
    Field("reasoning_effort", _as_str, None,
          "Reasoning effort for thinking models (low | medium | high).",
          aliases=("OPTIQ_REASONING_EFFORT",)),
    Field("max_turns", _as_int, 40,
          "Tool-calling turns before the agent stops."),
    Field("max_tokens", _as_int, 16384,
          "Token cap per model response. Thinking models need thousands: a "
          "short cap returns empty content with everything in `reasoning`. It "
          "also caps a single edit -- a tool call that outgrows this is "
          "truncated mid-JSON and cannot be applied, so keep it generous."),
    Field("context_window", _as_int, None,
          "Context window (tokens) of the served model. OptiQ serve reports "
          "this automatically; set it explicitly when pointing at a cloud or "
          "third-party OpenAI-compatible endpoint that has no /optiq/context, "
          "so compaction still knows the budget.",
          aliases=("OPTIQ_CONTEXT_WINDOW",)),
    Field("compact_at", _as_int, None,
          "Approximate token count at which old tool output is compacted out "
          "of the window. Default: 80%% of the server's context window, which "
          "the client reads from the server at startup."),
    Field("compact_headroom", _as_float, 0.8,
          "Fraction of the server's context window to fill before compacting. "
          "Ignored when compact_at is set explicitly."),
    Field("max_tool_output", _as_int, None,
          "Bytes of a single tool result kept before truncation."),
    Field("wall_clock", _as_float, None,
          "Seconds before the agent stops, regardless of turns. Default: none."),
    Field("request_timeout", _as_float, 900.0,
          "Seconds to wait for one model response. A big model on a small Mac "
          "prefills slowly: gemma-4-26B on a 24 GB M4 needs well over the "
          "OpenAI SDK's default on a few-thousand-token agent turn, and the "
          "timeout surfaces as a failed run against a perfectly healthy "
          "server."),
    Field("max_retries", _as_int, 6,
          "Retries for a request that errors or times out."),
    Field("color", _as_bool, True,
          "ANSI color in output.", aliases=("OPTIQ_CODE_COLOR",)),
    Field("mouse", _as_bool, True,
          "Capture the mouse in the TUI (default true, like OpenCode / Codex / "
          "Gemini CLI and other full-screen TUI agents). Enables two-finger "
          "trackpad + wheel scrolling of the transcript and click-to-focus; "
          "select text by holding your terminal's bypass key while dragging "
          "(Fn on macOS Terminal.app, Option on iTerm2) or use /copy. "
          "Set false for native click-drag selection with no modifier instead, "
          "at the cost of no trackpad/wheel scroll (PageUp/PageDown, Ctrl+Home/"
          "Ctrl+End still work)."),
)

_BY_NAME = {f.name: f for f in FIELDS}


@dataclass
class Config:
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    max_turns: int = 40
    max_tokens: int = 16384
    context_window: Optional[int] = None
    compact_at: Optional[int] = None
    compact_headroom: float = 0.8
    max_tool_output: Optional[int] = None
    wall_clock: Optional[float] = None
    request_timeout: float = 900.0
    max_retries: int = 6
    color: bool = True
    mouse: bool = True

    # Where each value came from, for `optiq code config`. Not a setting.
    sources: dict = None

    def resolve_compact_at(self, context_window: Optional[int]) -> Optional[int]:
        """The token budget to compact at, given the server's actual window.

        An explicit ``compact_at`` wins -- someone who set a number meant it.
        Otherwise take a fraction of the server's window, so the default
        tracks the model being served instead of a hardcoded guess. With no
        window reported (older server, or one that doesn't answer) there is
        nothing to take a fraction OF, so return None and let the caller skip
        compaction rather than invent a threshold.
        """
        if self.compact_at:
            return int(self.compact_at)
        if not context_window or context_window <= 0:
            return None
        return max(2048, int(context_window * self.compact_headroom))


def user_config_path() -> Path:
    home = Path(os.environ.get("OPTIQ_HOME") or (Path.home() / ".optiq")) / "code"
    return home / "config.json"


def repo_config_path(repo: Path | str) -> Path:
    return Path(repo) / ".optiq" / "code.json"


def _read(path: Path) -> tuple[dict, list[str]]:
    """Return (values, warnings). A broken config file must not be fatal."""
    if not path.is_file():
        return {}, []
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        return {}, [f"{path}: ignored ({e})"]
    if not isinstance(raw, dict):
        return {}, [f"{path}: ignored (expected a JSON object)"]
    out, warn = {}, []
    for k, v in raw.items():
        if k not in _BY_NAME:
            warn.append(f"{path}: unknown key {k!r}")
            continue
        cast = _BY_NAME[k].cast(v)
        if cast is None and v is not None:
            warn.append(f"{path}: bad value for {k!r}: {v!r}")
            continue
        out[k] = cast
    return out, warn


def load(repo: Path | str | None = None, **overrides) -> tuple[Config, list[str]]:
    """Resolve every setting. Returns (config, warnings).

    ``overrides`` are the CLI flags: a value of ``None`` means the flag was not
    passed and does not override anything.
    """
    warnings: list[str] = []
    user_vals, w = _read(user_config_path()); warnings += w
    repo_vals: dict = {}
    if repo is not None:
        repo_vals, w = _read(repo_config_path(repo)); warnings += w

    values, sources = {}, {}
    for f in FIELDS:
        val, src = f.default, "default"
        if f.name in user_vals:
            val, src = user_vals[f.name], "user config"
        if f.name in repo_vals:
            val, src = repo_vals[f.name], "repo config"
        for alias in f.aliases:                      # legacy env names
            if os.environ.get(alias) is not None:
                cast = f.cast(os.environ[alias])
                if cast is None:
                    warnings.append(f"${alias}: bad value {os.environ[alias]!r}")
                else:
                    val, src = cast, f"${alias}"
        if os.environ.get(f.env) is not None:        # canonical env name wins
            cast = f.cast(os.environ[f.env])
            if cast is None:
                warnings.append(f"${f.env}: bad value {os.environ[f.env]!r}")
            else:
                val, src = cast, f"${f.env}"
        if overrides.get(f.name) is not None:        # CLI flag wins over all
            val, src = f.cast(overrides[f.name]), "flag"
        values[f.name], sources[f.name] = val, src

    known = {f.name for f in dc_fields(Config)}
    for k in overrides:
        if k not in known:
            warnings.append(f"unknown setting {k!r}")
    return Config(**values, sources=sources), warnings
