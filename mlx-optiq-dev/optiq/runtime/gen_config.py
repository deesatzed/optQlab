"""Read a model's recommended sampling settings from generation_config.json.

Model authors publish recommended ``temperature``/``top_p``/``top_k``/``min_p``
in a ``generation_config.json`` next to ``config.json`` in the HF repo.
Following these is closer to what the model was tuned for and — for MTP
spec decoding specifically — matches the distributions our verify uses
rejection sampling against, dramatically improving draft acceptance over
pure temperature-only sampling.

OptiQ reads these defaults when ``optiq serve`` / ``optiq lab`` boots
and injects them into ``mlx_lm.server``'s argv unless the user has
already passed the corresponding flag explicitly. Users opt out by
passing ``--temp 0`` or any of the other flags directly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


# Keys we surface. mlx_lm.server uses lowercase hyphen forms on the CLI
# (``--top-p``); the JSON keys here are the underscore form HF uses.
_SAMPLER_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")


def read_recommended_sampling(model_path_or_id: str,
                              allow_hf_fetch: bool = False) -> dict:
    """Return a dict of recommended sampler settings for ``model``.

    Looks at ``generation_config.json`` in the model directory (or its
    HF cache snapshot for repo ids). Returns an empty dict if no file
    exists or no recognised keys are present. Never raises.

    If ``allow_hf_fetch`` is True and the model isn't cached locally,
    pull just the ``generation_config.json`` file from the HF repo. Tiny
    (a few hundred bytes), useful for the Lab's "preview defaults"
    endpoint where we want to surface recommended sampling before the
    user has actually loaded the model.
    """
    cfg_path = _locate_generation_config(model_path_or_id)
    cfg = None
    if cfg_path is not None and cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            cfg = None
    if cfg is None and allow_hf_fetch and "/" in model_path_or_id \
            and not Path(model_path_or_id).is_absolute():
        cfg = _fetch_generation_config_from_hf(model_path_or_id)
    if cfg is None:
        return {}
    out: dict = {}
    for k in _SAMPLER_KEYS:
        v = cfg.get(k)
        if v is not None:
            out[k] = v
    return out


def _fetch_generation_config_from_hf(repo_id: str) -> Optional[dict]:
    """Pull just ``generation_config.json`` from an HF repo (no model
    download). Returns the parsed dict, or None on any failure (404,
    network error, malformed JSON). Never raises."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id,
                               filename="generation_config.json")
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _locate_generation_config(model_path_or_id: str) -> Optional[Path]:
    """Resolve to the on-disk ``generation_config.json`` path, or None.

    Accepts a local directory, a file path, or an HF repo id. For HF
    repo ids we look in the local snapshot cache; if not cached we
    return None (avoid a network call from this helper).
    """
    p = Path(model_path_or_id)
    if p.is_dir():
        cand = p / "generation_config.json"
        return cand if cand.exists() else None
    if p.is_file():
        return None
    # Treat as HF repo id; resolve via the local cache layout
    repo_dir = Path.home() / ".cache/huggingface/hub" / (
        "models--" + str(model_path_or_id).replace("/", "--")
    )
    snaps_dir = repo_dir / "snapshots"
    if snaps_dir.is_dir():
        try:
            snap = sorted(snaps_dir.iterdir())[-1]
        except (StopIteration, IndexError):
            return None
        cand = snap / "generation_config.json"
        return cand if cand.exists() else None
    return None


def merge_into_argv(
    argv: list[str],
    recommended: dict,
    *,
    prefix_log: str = "[optiq] applying recommended sampling:",
) -> list[str]:
    """Append CLI flags for any recommended sampler keys that aren't
    already present in ``argv``. Only forwards keys ``mlx_lm.server``
    actually accepts: ``--temp``, ``--top-p``, ``--top-k``, ``--min-p``.

    Other recommended-sampling keys (``repetition_penalty``,
    ``presence_penalty``, ``do_sample``, etc.) are read from
    ``generation_config.json`` and exposed to clients via the Lab
    sampler-preview endpoint, but they are NOT forwarded to
    ``mlx_lm.server`` because that argparse rejects unknown flags
    and the server falls back to printing --help.

    Returns a new argv list and prints a one-line note when anything
    was injected.
    """
    if not recommended:
        return argv
    # Only the flags ``mlx_lm.server`` understands. Keep this list in
    # sync with mlx_lm.server's argparse (see :--temp, --top-p, --top-k,
    # --min-p) — if upstream adds more, extend here.
    flag_for = {
        "temperature": "--temp",
        "top_p": "--top-p",
        "top_k": "--top-k",
        "min_p": "--min-p",
    }
    applied: list[str] = []
    out = list(argv)
    for key, value in recommended.items():
        flag = flag_for.get(key)
        if flag is None:
            continue
        # Don't override user-provided values (either ``--flag VAL`` or ``--flag=VAL``).
        already = any(a == flag or a.startswith(flag + "=") for a in out)
        if already:
            continue
        out += [flag, str(value)]
        applied.append(f"{key}={value}")
    if applied:
        print(f"{prefix_log} {', '.join(applied)}", flush=True)
    return out
