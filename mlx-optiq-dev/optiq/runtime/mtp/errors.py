"""User-facing MTPLX error types.

The CLI should only show tracebacks when the user explicitly asks for debug
output.  These exceptions carry enough structure for both human and JSON
responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EXIT_GENERAL = 1
EXIT_INVALID_MTP = 2
EXIT_EXACTNESS = 3
EXIT_UNSUPPORTED_ARCHITECTURE = 4
EXIT_ENVIRONMENT = 5
EXIT_RESOURCE = 6
EXIT_INTEGRATION = 7
EXIT_THERMAL = 8


@dataclass
class MTPLXError(Exception):
    message: str
    code: int = EXIT_GENERAL
    kind: str = "mtplx_error"
    detail: str | None = None
    fix: str | None = None
    command: str | None = None

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": self.kind,
            "message": self.message,
            "exit_code": self.code,
        }
        if self.detail:
            payload["detail"] = self.detail
        if self.fix:
            payload["fix"] = self.fix
        if self.command:
            payload["command"] = self.command
        return payload


class MissingMLXError(MTPLXError):
    def __init__(self, detail: str | None = None) -> None:
        super().__init__(
            "MLX is not installed for this native arm64 Python environment.",
            code=EXIT_ENVIRONMENT,
            kind="missing_mlx",
            detail=detail,
            fix=(
                "Install MLX from a native arm64 Python on Apple Silicon. "
                "If platform.processor() reports i386, switch out of Rosetta."
            ),
            command="python3 -m pip install mlx",
        )


class MissingModelError(MTPLXError):
    def __init__(self, model: str, pull_command: str) -> None:
        super().__init__(
            f"Model {model} is not cached locally.",
            code=EXIT_RESOURCE,
            kind="missing_model",
            fix="Download the verified MTPLX model before running.",
            command=pull_command,
        )


class IntegrationError(MTPLXError):
    def __init__(self, message: str, *, fix: str | None = None, command: str | None = None) -> None:
        super().__init__(
            message,
            code=EXIT_INTEGRATION,
            kind="integration_error",
            fix=fix,
            command=command,
        )

