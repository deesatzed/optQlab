"""Tool-call approval — two modes, like Claude Code.

`approval` (TUI default): pause before any side-effecting tool call
(`write_file`, `edit_file`, `bash`) and ask the human. Read-only / reversible
tools run freely. `auto` (headless default): execute everything without pausing.

UI-agnostic: the loop calls `gate(call, policy, approver)`; the front-end
supplies the `approver` (interactive modal in the TUI, never invoked in auto).
The policy remembers per-tool "always allow" decisions for the session.

Ported from conjure/agent/approval.py (the spec-level contract gate it mentioned
is retired; this action-level gate is all that remains).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class ApprovalMode(str, Enum):
    APPROVE = "approval"
    AUTO = "auto"


# Tools that mutate the filesystem or run arbitrary commands — gated in
# `approval` mode. Everything else runs freely.
DEFAULT_GATED_TOOLS = frozenset({"write_file", "edit_file", "bash"})


@dataclass
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)

    def summary(self, width: int = 100) -> str:
        """A one-line, human-readable description for the approval card."""
        a = self.arguments or {}
        if self.name in ("write_file", "edit_file"):
            path = a.get("path") or a.get("file_path") or "?"
            s = f"{self.name} → {path}"
        elif self.name == "bash":
            s = "$ " + str(a.get("command") or a.get("cmd") or "")
        else:
            args = ", ".join(f"{k}={v!r}" for k, v in a.items())
            s = f"{self.name}({args})"
        s = " ".join(s.split())
        return s if len(s) <= width else s[: width - 1] + "…"

    def preview(self, max_lines: int = 16, width: int = 64) -> str:
        """Multi-line detail for the approval modal: content / diff / command."""
        a = self.arguments or {}

        def clip(text: str) -> str:
            lines = str(text).splitlines()
            out = [ln[:width] for ln in lines[:max_lines]]
            if len(lines) > max_lines:
                out.append(f"… (+{len(lines) - max_lines} more lines)")
            return "\n".join(out)

        if self.name == "write_file":
            return clip(a.get("content", ""))
        if self.name == "edit_file":
            return "- " + clip(a.get("old", "")).replace("\n", "\n- ") + \
                   "\n+ " + clip(a.get("new", "")).replace("\n", "\n+ ")
        if self.name == "bash":
            return "$ " + str(a.get("command") or a.get("cmd") or "")
        return clip(repr(a))


@dataclass
class Decision:
    allow: bool
    remember: bool = False
    note: str = ""


Approver = Callable[[ToolCall], Decision]


@dataclass
class ApprovalPolicy:
    mode: ApprovalMode = ApprovalMode.APPROVE
    gated: frozenset = DEFAULT_GATED_TOOLS
    session_allow: set = field(default_factory=set)

    def needs_approval(self, name: str) -> bool:
        if self.mode == ApprovalMode.AUTO:
            return False
        if name in self.session_allow:
            return False
        return name in self.gated

    def allow_for_session(self, name: str) -> None:
        self.session_allow.add(name)


@dataclass
class GateResult:
    allowed: bool
    asked: bool
    decision: Decision | None = None


def gate(call: ToolCall, policy: ApprovalPolicy, approver: Approver) -> GateResult:
    """Decide whether `call` may execute, asking the user only when required."""
    if not policy.needs_approval(call.name):
        return GateResult(allowed=True, asked=False)
    decision = approver(call)
    if decision.allow and decision.remember:
        policy.allow_for_session(call.name)
    return GateResult(allowed=decision.allow, asked=True, decision=decision)


def auto_approver(_call: ToolCall) -> Decision:
    return Decision(allow=True)


def deny_approver(_call: ToolCall) -> Decision:
    return Decision(allow=False, note="denied by policy")
