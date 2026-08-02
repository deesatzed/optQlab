"""Copy-paste config snippets for each documented integration.

Same configs as site/docs/integrations/*.html — the live ones we
verified end-to-end. Kept in code (not jinja) so they can be unit
tested and referenced from API endpoints too.

Each snippet is parameterised on ``api_url`` and a representative
``model_name`` so the user can paste it without further edits.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Snippet:
    label: str
    language: str          # "bash" / "toml" / "json"
    body: str
    description: str = ""


DEFAULT_MODEL = "mlx-community/Qwen3.5-9B-OptiQ-4bit"
DEFAULT_TOKEN = "sk-optiq-local"


def claude_code(api_url: str, model: str = DEFAULT_MODEL) -> Snippet:
    return Snippet(
        label="Claude Code",
        language="bash",
        description="Set these env vars, then run `claude` in any project.",
        body=(
            f"export ANTHROPIC_BASE_URL={api_url}\n"
            f"export ANTHROPIC_API_KEY={DEFAULT_TOKEN}\n"
            f"export ANTHROPIC_MODEL={model}"
        ),
    )


def codex(api_url: str, model: str = DEFAULT_MODEL) -> Snippet:
    return Snippet(
        label="Codex",
        language="toml",
        description="Append to ~/.codex/config.toml, then `codex -p optiq`.",
        body=(
            "[model_providers.optiq]\n"
            '  name                 = "OptiQ Local"\n'
            f'  base_url             = "{api_url}/v1"\n'
            '  env_key              = "OPTIQ_AUTH_TOKEN"\n'
            '  wire_api             = "responses"\n'
            "  requires_openai_auth = false\n"
            "\n"
            "[profiles.optiq]\n"
            '  model_provider = "optiq"\n'
            f'  model          = "{model}"'
        ),
    )


def opencode(api_url: str, model: str = DEFAULT_MODEL) -> Snippet:
    return Snippet(
        label="OpenCode",
        language="json",
        description="Save as opencode.jsonc (per-project) or ~/.config/opencode/opencode.jsonc (global).",
        body=(
            "{\n"
            '  "$schema": "https://opencode.ai/config.json",\n'
            '  "provider": {\n'
            '    "optiq": {\n'
            '      "npm": "@ai-sdk/openai-compatible",\n'
            '      "name": "OptiQ Local",\n'
            '      "options": {\n'
            f'        "baseURL": "{api_url}/v1",\n'
            f'        "apiKey": "{DEFAULT_TOKEN}"\n'
            "      },\n"
            '      "models": {\n'
            '        "qwen": {"name": "' + model + '"}\n'
            "      }\n"
            "    }\n"
            "  },\n"
            '  "model": "optiq/qwen"\n'
            "}"
        ),
    )


def openclaw(api_url: str, model: str = DEFAULT_MODEL) -> Snippet:
    return Snippet(
        label="OpenClaw",
        language="json",
        description="Save as ~/.openclaw/openclaw.json. Validate with `openclaw config validate`.",
        body=(
            "{\n"
            '  "models": {\n'
            '    "providers": {\n'
            '      "optiq": {\n'
            f'        "baseUrl": "{api_url}/v1",\n'
            f'        "apiKey": "{DEFAULT_TOKEN}",\n'
            '        "api": "anthropic-messages",\n'
            '        "models": [\n'
            '          { "id": "qwen", "name": "' + model + '" }\n'
            "        ]\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "}"
        ),
    )


def hermes_agent(api_url: str, model: str = DEFAULT_MODEL) -> Snippet:
    return Snippet(
        label="Hermes Agent",
        language="bash",
        description="Set env vars and use `hermes chat --provider custom`.",
        body=(
            f"export CUSTOM_BASE_URL={api_url}/v1\n"
            f"export OPENAI_API_KEY={DEFAULT_TOKEN}\n"
            f"\n"
            f'hermes chat -m "{model}" --provider custom'
        ),
    )


ALL = (claude_code, codex, opencode, openclaw, hermes_agent)


def all_snippets(api_url: str, model: str = DEFAULT_MODEL) -> list[Snippet]:
    return [fn(api_url, model) for fn in ALL]
