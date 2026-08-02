"""The one place a converted artifact gets its sidecars attached.

Every OptiQ quant is a *language* artifact plus zero or more sidecars: the bf16
vision/audio towers (``optiq_vision.safetensors``) and the speculative-decoding
MTP head (``mtp.safetensors``). Both used to be bolted on after the fact --
``build_vision_sidecar`` and ``preserve_mtp`` existed, but nothing in the convert
pipeline called them, so whether a published model got its towers depended on
someone remembering to run a script. DiffusionGemma was the exception: its
pipeline built the sidecar inline, which is how the two paths drifted apart, and
how a Qwen3.5 VLM could be converted straight to a silently text-only artifact.

``attach_sidecars`` is that step, shared by every pipeline. It is:

* **Detect-and-skip.** A text-only model has no tower and no MTP head, so it
  simply gets neither; callers do not have to know what kind of model they hold.
* **Family-agnostic.** Which weights *are* the tower is data
  (``vlm.sidecar._MM_PREFIXES``), not control flow. Adding a family means adding
  a prefix, not a new call site.
* **Fail-loud on a real failure.** A model that advertises ``vision_config`` but
  whose towers cannot be extracted is a broken artifact, not a text model, and
  raising here is what stops it reaching the Hub.

``tests/test_sidecar_pipeline.py`` pins all three properties.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def has_vision_config(model: str | Path) -> bool:
    """Whether the model *claims* image support, per its config.json.

    ``model`` may be a local directory or a Hub repo id; a repo id is resolved
    against the local cache (never downloaded -- by the time this is called the
    base has already been fetched by the convert).

    Ask this of the BASE, never of the freshly-converted quant: ``mlx_lm.convert``
    strips ``vision_config`` out of the quant's config, so the quant is the one
    place where the answer has been erased. Reading it there is what made
    ``attach_sidecars`` skip silently and ship a VLM with no tower.

    This is the claim; the sidecar is the delivery. ``attach_sidecars`` raising
    when the two disagree is the entire point of the check.
    """
    from .sidecar_layout import local_model_dir

    d = local_model_dir(model)
    if d is None:
        return False
    cfg_path = Path(d) / "config.json"
    if not cfg_path.is_file():
        return False
    try:
        cfg = json.load(open(cfg_path))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(cfg.get("vision_config") or cfg.get("audio_config"))


def _base_has_mm_weights(model: str | Path) -> bool:
    """Whether the BASE actually ships any multimodal tower weights.

    A model can declare ``vision_config`` yet ship none -- e.g. a text-only
    coder finetuned from a VLM base, which keeps the vision fields in config.json
    but drops the tower. That is text-only, not a broken VLM, so it must not be
    forced through the sidecar path."""
    from .sidecar_layout import local_model_dir
    from .vlm.sidecar import _is_mm_key

    d = local_model_dir(model)
    if d is None:
        return False
    d = Path(d)
    idx = d / "model.safetensors.index.json"
    if idx.is_file():
        try:
            wm = json.load(open(idx)).get("weight_map", {})
            return any(_is_mm_key(k) for k in wm)
        except (json.JSONDecodeError, OSError):
            pass
    from safetensors import safe_open
    for f in sorted(d.glob("*.safetensors")):
        try:
            with safe_open(f, framework="numpy") as st:
                if any(_is_mm_key(k) for k in st.keys()):
                    return True
        except Exception:
            continue
    return False


def _strip_vision_claims(quant_dir: str | Path) -> None:
    """Drop vestigial vision/audio fields from a text-only quant's config so it
    does not advertise images it cannot see."""
    cfg_path = Path(quant_dir) / "config.json"
    try:
        cfg = json.load(open(cfg_path))
    except (json.JSONDecodeError, OSError):
        return
    for k in ("vision_config", "audio_config", "image_token_id", "video_token_id",
              "vision_start_token_id", "vision_end_token_id", "image_token_index"):
        cfg.pop(k, None)
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)


def attach_sidecars(
    source: str,
    quant_dir: str | Path,
    *,
    vision: bool = True,
    mtp: bool = True,
    vision_dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Attach every sidecar ``source`` warrants to the freshly-converted quant.

    Args:
        source: the bf16 base (Hub repo id or local dir) that still has the
            towers and the MTP head. The quant no longer does -- that is why
            this reads from the base, not from ``quant_dir``.
        quant_dir: the converted language artifact. Sidecars land under its
            ``optiq/`` subfolder (see ``sidecar_layout``), out of reach of the
            ``*.safetensors`` glob that a stock loader does.
        vision: set False to quantize the towers inline instead (DiffusionGemma's
            ``--quantize-vision``). The release contract wants bf16 towers, so
            this is not the default.
        mtp: attach the speculative-decoding head. The LLM pipeline passes
            False, not because it wants no MTP but because ``convert_llm_to_mlx``
            already preserves it one layer down -- and does it better, since it
            knows the host's ``q_bits``/``q_group_size`` and matches the sidecar
            to them. Pass True only from a pipeline that does not go through
            that backend.

    Returns:
        ``{"vision": summary|None, "mtp": bool}``.

    Raises:
        RuntimeError: the model declares ``vision_config`` but no tower weights
            could be extracted. That is a broken artifact -- most likely the
            family's weight prefix is missing from ``vlm.sidecar._MM_PREFIXES``
            -- and shipping it would produce a model that advertises images and
            cannot see.
    """
    report: dict[str, Any] = {"vision": None, "mtp": False}
    # Always the base: the quant's config no longer has vision_config (convert
    # strips it), so asking the quant would answer "no tower" for every VLM.
    claims_vision = has_vision_config(source)

    if vision and claims_vision and not _base_has_mm_weights(source):
        # Declares vision but ships no tower -> genuinely text-only (e.g. a coder
        # finetuned from a VLM base). Strip the vestigial vision fields so the
        # quant does not advertise images it cannot see, and skip the sidecar
        # instead of failing loud.
        _strip_vision_claims(quant_dir)
        print("  [sidecar] config declares vision but base ships no tower "
              "weights -> text-only (vision_config stripped)")
    elif vision and claims_vision:
        from .vlm.sidecar import build_vision_sidecar

        try:
            report["vision"] = build_vision_sidecar(
                str(source), quant_dir, dtype=vision_dtype, force=True)
        except Exception as exc:
            raise RuntimeError(
                f"{source} declares a vision/audio tower but the sidecar could "
                f"not be built: {exc}. The quant would ship advertising images "
                f"it cannot see. If this is a new model family, its weight "
                f"prefix is probably missing from vlm.sidecar._MM_PREFIXES."
            ) from exc
        n = report["vision"]["n_tensors"]
        mb = report["vision"]["bytes"] / 1e6
        print(f"  [sidecar] vision: {n} tensors, {mb:.0f} MB {vision_dtype}")

    if mtp:
        from .runtime.mtp_convert import preserve_mtp

        report["mtp"] = preserve_mtp(str(source), str(quant_dir))
        if report["mtp"]:
            print("  [sidecar] mtp: speculative-decoding head attached")

    return report
