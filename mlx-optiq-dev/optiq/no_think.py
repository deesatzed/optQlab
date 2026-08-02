"""Global no-think toggle for ``optiq serve``.

Disables the model's chain-of-thought across ALL chat endpoints (native
``/v1/chat/completions``, the Anthropic ``/v1/messages`` shim, and the OpenAI
``/v1/responses`` shim) by forcing ``enable_thinking=False`` into the chat
template. Two motivations:

  * Agentic harnesses on a local model: reasoning tokens dominate per-turn
    latency (e.g. Claude Code went from ~5 min/turn to ~9 s/turn with thinking
    off), so a no-think mode makes local agentic loops usable.
  * Fair harness comparisons: thinking must be held constant across harnesses,
    otherwise some endpoints reason and others don't and the comparison is
    confounded.

All three endpoints route generation through
``mlx_lm.server.APIHandler.handle_chat_completions``, so a single patch there
covers them uniformly. Opt in with ``OPTIQ_NO_THINK=1`` at serve time.
"""
from __future__ import annotations

import logging

_INSTALLED = False


def install_no_think() -> None:
    """Patch mlx-lm's chat handler so every chat request generates with
    ``enable_thinking=False``. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return

    import mlx_lm.server as server_mod

    original = server_mod.APIHandler.handle_chat_completions

    def patched_handle_chat_completions(self):
        # Force thinking off regardless of what the client (or shim) set.
        ctk = dict(getattr(self, "chat_template_kwargs", None) or {})
        ctk["enable_thinking"] = False
        self.chat_template_kwargs = ctk
        body = getattr(self, "body", None)
        if isinstance(body, dict):
            body["chat_template_kwargs"] = ctk
        return original(self)

    server_mod.APIHandler.handle_chat_completions = patched_handle_chat_completions
    _INSTALLED = True
    logging.info("[optiq] global no-think enabled (enable_thinking=False on all chat requests)")


_VARIANTS_INSTALLED = False


def install_thinking_variants(served_model: str | None = None) -> None:
    """Per-request thinking control via a model-name suffix.

    A request for ``model="<id>:no-think"`` generates with
    ``enable_thinking=False``; ``":think"`` forces it on. The suffix is stripped
    before mlx-lm matches the served model, so ``optiq code launch --model
    <id>:no-think`` works with no global flag and no server env var. When
    ``served_model`` is given, ``/v1/models`` also advertises that model with
    both ``:think`` and ``:no-think`` variant ids, so clients (and OptiQ Code's
    ``discover_model``) can see and select them. Idempotent; always safe to
    install (a request without a suffix is untouched).
    """
    global _VARIANTS_INSTALLED
    if _VARIANTS_INSTALLED:
        return
    import mlx_lm.server as server_mod

    _orig_chat = server_mod.APIHandler.handle_chat_completions

    def _chat_with_variants(self):
        body = getattr(self, "body", None)
        if isinstance(body, dict):
            model = body.get("model")
            if isinstance(model, str):
                for suffix, enabled in ((":no-think", False), (":think", True)):
                    if model.endswith(suffix):
                        body["model"] = model[: -len(suffix)]
                        ctk = dict(body.get("chat_template_kwargs") or {})
                        ctk["enable_thinking"] = enabled
                        body["chat_template_kwargs"] = ctk
                        self.chat_template_kwargs = ctk
                        break
        return _orig_chat(self)

    server_mod.APIHandler.handle_chat_completions = _chat_with_variants

    # The stock /v1/models handler builds+writes its list inline (scan_cache_dir),
    # so replace it: advertise the served model and its `:think` / `:no-think`
    # variants first, then the same cache list (which preserves on-demand loading).
    if served_model:
        def _models_with_variants(self):
            import json
            self._set_completion_headers(200)
            self.end_headers()
            created = getattr(self, "created", 0)
            models = [
                {"id": vid, "object": "model", "created": created}
                for vid in (served_model, f"{served_model}:think", f"{served_model}:no-think")
            ]
            seen = {m["id"] for m in models}
            try:
                from huggingface_hub import scan_cache_dir
                need = {"config.json", "model.safetensors.index.json", "tokenizer_config.json"}
                for repo in scan_cache_dir().repos:
                    if repo.repo_type != "model" or "main" not in repo.refs:
                        continue
                    names = {f.file_path.name for f in repo.refs["main"].files}
                    if need <= names and repo.repo_id not in seen:
                        models.append({"id": repo.repo_id, "object": "model", "created": created})
                        seen.add(repo.repo_id)
            except Exception:
                pass
            self.wfile.write(json.dumps({"object": "list", "data": models}).encode())

        server_mod.APIHandler.handle_models_request = _models_with_variants

    _VARIANTS_INSTALLED = True
    logging.info("[optiq] thinking variants enabled (model id suffix :think / :no-think)")
