"""Helpers for separating server bind addresses from client URLs."""

from __future__ import annotations


def _clean_host(host: str | None) -> str:
    return str(host or "").strip()


def _unbracket_host(host: str | None) -> str:
    cleaned = _clean_host(host)
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return cleaned[1:-1]
    return cleaned


def is_wildcard_bind(host: str | None) -> bool:
    return _unbracket_host(host).lower() in {"0.0.0.0", "::"}


def connect_host_for_bind(host: str | None) -> str:
    normalized = _unbracket_host(host).lower()
    if normalized in {"", "0.0.0.0", "::", "localhost"}:
        return "127.0.0.1"
    return _clean_host(host)


def url_host(host: str | None) -> str:
    cleaned = _clean_host(host)
    raw = _unbracket_host(cleaned)
    if ":" in raw and not (cleaned.startswith("[") and cleaned.endswith("]")):
        return f"[{raw}]"
    return cleaned


def local_url_for_bind(host: str | None, port: int, *, path: str = "") -> str:
    suffix = path if path.startswith("/") or not path else f"/{path}"
    return f"http://{url_host(connect_host_for_bind(host))}:{int(port)}{suffix}"


def bind_label(host: str | None, port: int) -> str:
    raw = _unbracket_host(host)
    display = "[::]" if raw == "::" else (_clean_host(host) or "127.0.0.1")
    label = f"{display}:{int(port)}"
    if is_wildcard_bind(host):
        label += " (all interfaces)"
    return label
