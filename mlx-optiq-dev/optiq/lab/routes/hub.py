"""Hub: browse + discover models (published OptiQ quants, Hugging Face search,
and locally-converted models) with one-click load-to-server / chat.

Reuses the existing serving plumbing: published quants come from
``/api/server/published`` and loading a model reuses ``/api/server/apply``.
This page only adds discovery (HF search + local scan).
"""

from __future__ import annotations

import os

from flask import Blueprint, current_app, jsonify, render_template, request

bp = Blueprint("hub", __name__)


@bp.route("/hub")
def hub_page():
    return render_template("hub.html", page_title="Hub", section="hub")


@bp.route("/api/hub/search")
def hub_search():
    """Search Hugging Face for MLX-compatible models. Returns lightweight
    cards. Defaults to OptiQ quants when the query is blank."""
    query = (request.args.get("q") or "").strip()
    only_optiq = request.args.get("optiq") == "1"
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        search = query or ("OptiQ" if only_optiq else "")
        models = list(
            api.list_models(
                search=search or None,
                library="mlx",
                sort="downloads",
                direction=-1,
                limit=40,
            )
        )
        out = []
        for m in models:
            rid = m.id
            if only_optiq and "optiq" not in rid.lower():
                continue
            out.append({
                "repo_id": rid,
                "downloads": getattr(m, "downloads", 0) or 0,
                "likes": getattr(m, "likes", 0) or 0,
                "tags": [t for t in (getattr(m, "tags", []) or [])
                         if t in ("mlx", "4-bit", "8-bit", "qat", "optiq",
                                  "image-text-to-text", "text-generation")][:6],
            })
        return jsonify({"ok": True, "models": out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.route("/api/hub/local")
def hub_local():
    """List locally-converted OptiQ models (dirs containing
    optiq_metadata.json) under the lab's output roots."""
    roots = []
    out_root = os.environ.get("OPTIQ_OUTPUT_DIR") or os.path.join(os.getcwd(), "optiq_output")
    roots.append(out_root)
    found = []
    seen = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if "optiq_metadata.json" in files and "config.json" in files:
                if dirpath in seen:
                    continue
                seen.add(dirpath)
                name = os.path.basename(dirpath)
                if name == "optiq_mixed":
                    name = os.path.basename(os.path.dirname(dirpath))
                from ...sidecar_layout import exists as _sidecar_exists
                has_vision = ("optiq_vision.safetensors" in files
                              or _sidecar_exists(dirpath, "optiq_vision.safetensors"))
                found.append({
                    "path": os.path.abspath(dirpath),
                    "name": name,
                    "vision": has_vision,
                })
    found.sort(key=lambda d: d["name"].lower())
    return jsonify({"ok": True, "models": found})
