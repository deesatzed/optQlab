"""Extend ``optiq serve``'s ``/v1/models`` to advertise locally-built quants.

mlx-lm's server already (a) lists every MLX model in the HuggingFace cache and
the ``--model`` that was served, and (b) hot-swaps to whatever ``model`` a
request asks for (``ModelProvider.load``). What it does *not* surface is a quant
sitting in a local directory (e.g. ``optiq_output/<name>``) that was never
pushed to the hub. A coding harness pointed at the server can't discover or
switch to those.

``install(server_mod, models_dir)`` replaces ``handle_models_request`` with a
version that lists, in OpenAI ``/v1/models`` form: the served model, every quant
under ``models_dir`` (by resolved path, which is directly usable as the request
``model`` field), and every MLX model in the hub cache. No-op when ``models_dir``
is falsy.
"""

from __future__ import annotations

import json
from pathlib import Path


def install(server_mod, models_dir):
    if not models_dir:
        return
    root = Path(models_dir)

    from ..lab import local_quants

    def handle_models_request(self):
        models = []
        seen = set()

        def add(mid):
            if mid and mid not in seen:
                seen.add(mid)
                models.append(
                    {"id": mid, "object": "model", "created": self.created}
                )

        # The model the server booted with.
        cli_model = getattr(self.response_generator.cli_args, "model", None)
        if cli_model:
            p = Path(cli_model)
            add(str(p.resolve()) if p.exists() else cli_model)

        # Locally-built quants under --models-dir (switchable by their path).
        try:
            for q in local_quants.discover(root):
                add(str(Path(q.path).resolve()))
        except Exception:
            pass

        # MLX models already in the HuggingFace cache (switchable by repo id).
        try:
            for q in local_quants.discover_hf_cache():
                add(q.name)
        except Exception:
            pass

        payload = json.dumps({"object": "list", "data": models}).encode()
        self._set_completion_headers(200)
        self.end_headers()
        self.wfile.write(payload)

    server_mod.APIHandler.handle_models_request = handle_models_request
