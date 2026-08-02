"""MLX backend: convert models to MLX format with mixed-precision quantization."""

import json
import os
import shutil
from contextlib import contextmanager
from typing import Callable, Optional


def _fix_mistral_tokenizer_regex(output_dir: str) -> None:
    """Bake transformers' ``fix_mistral_regex`` correction into a converted
    model's tokenizer, but only when the tokenizer actually triggers it.

    Mistral-family fast tokenizers (poolside Laguna, Devstral, …) ship a buggy
    ``Split`` pretokenizer regex. transformers detects it, warns, and — unless
    told ``fix_mistral_regex=True`` — leaves it in place, which mis-tokenizes.
    OptiQ copies tokenizer files verbatim, so the bug would propagate to every
    published quant.

    Detection compares the **serialized backend tokenizer**: load the copied
    tokenizer once plainly and once with ``fix_mistral_regex=True``, and if the
    two ``backend_tokenizer.to_str()`` blobs differ the flag changed the
    pretokenizer (the corrected Mistral ``Split`` regex) — re-save the fixed one
    so downstream loads are clean. This is robust where the earlier checks were
    not: the corrected tokenizer exposes no reliable Python flag (the
    ``fix_mistral_regex`` attribute is simply absent on a normal load); the
    transformers warning goes through ``warning_once`` so it can be silently
    deduplicated when the source tokenizer was already loaded earlier in the same
    process; and a probe-string diff misses inputs the buggy regex tokenizes the
    same. Comparing the serialized pretokenizer depends on none of those. No-op
    when the flag changes nothing (non-Mistral tokenizers, or already-correct
    ones); best-effort, never raises.
    """
    try:
        if not os.path.exists(os.path.join(output_dir, "tokenizer.json")):
            return
        import warnings

        from transformers import AutoTokenizer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = AutoTokenizer.from_pretrained(output_dir)
            try:
                fixed = AutoTokenizer.from_pretrained(output_dir, fix_mistral_regex=True)
            except TypeError:
                return          # this transformers/tokenizer doesn't know the flag
            try:
                same = (plain.backend_tokenizer.to_str()
                        == fixed.backend_tokenizer.to_str())
            except AttributeError:
                return          # slow tokenizer, no rust backend to compare/fix
            if same:
                return          # fix changes nothing -> tokenizer is fine as copied
            fixed.save_pretrained(output_dir)
    except Exception:
        # Tokenizer stays as the verbatim copy — never block a conversion on this.
        pass


def _eager_requantize_experts(model, quant_predicate, def_group_size, def_bits):
    """Re-quantize each pre-quantized fused-expert tensor (e.g. gpt-oss MXFP4)
    to the OptiQ-assigned bit-width, processing **one expert slice at a time**.

    A fused expert tensor is ``[num_experts, out, in]`` — for a 120B that's ~1 B
    elements, which (a) OOMs if dequantized to bf16 all at once and (b) trips the
    Metal command-buffer GPU timeout if quantized in a single op. So for each
    tensor we loop over the experts, dequantize+requantize each ``[out, in]``
    slice (~16 MB bf16) with an eager ``mx.eval`` so it frees immediately, then
    stack the small quantized slices and assemble a ``QuantizedSwitchLinear``
    directly (bypassing its ``__init__``, which would re-quantize a random
    full-size tensor). Memory stays bounded to the accumulating output.

    Returns ``{path: {group_size, bits, mode}}`` to register in the output config.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    # Return freed buffers to the OS instead of caching, to bound RSS.
    try:
        mx.set_cache_limit(2 * 1024 ** 3)
    except Exception:
        pass

    targets = [(p, m) for p, m in model.named_modules()
               if isinstance(m, QuantizedSwitchLinear)]
    new_mods, layer_cfg = [], {}
    for ti, (path, module) in enumerate(targets):
        gs, bits = def_group_size, def_bits
        if quant_predicate is not None:
            bp = quant_predicate(path, module)
            if isinstance(bp, dict):
                gs = bp.get("group_size", def_group_size)
                bits = bp.get("bits", def_bits)
            elif not bp:
                continue
        n_experts = module.weight.shape[0]
        src_biases = getattr(module, "biases", None)
        qws, qss, qbs = [], [], []
        for e in range(n_experts):
            we = mx.dequantize(
                module.weight[e], module.scales[e],
                src_biases[e] if src_biases is not None else None,
                module.group_size, module.bits, module.mode,
            )
            parts = mx.quantize(we, group_size=gs, bits=bits, mode="affine")
            qw, qs = parts[0], parts[1]
            qb = parts[2] if len(parts) > 2 else None
            mx.eval(qw, qs, *((qb,) if qb is not None else ()))
            qws.append(qw)
            qss.append(qs)
            if qb is not None:
                qbs.append(qb)
            del we
        qsl = QuantizedSwitchLinear.__new__(QuantizedSwitchLinear)
        nn.Module.__init__(qsl)
        qsl.weight = mx.stack(qws)
        qsl.scales = mx.stack(qss)
        if qbs:
            qsl.biases = mx.stack(qbs)
        qsl.group_size, qsl.bits, qsl.mode = gs, bits, "affine"
        if "bias" in module:
            qsl.bias = module.bias
        mx.eval(*([qsl.weight, qsl.scales] + ([qsl.biases] if qbs else [])))
        qsl.freeze()
        del qws, qss, qbs
        mx.clear_cache()
        new_mods.append((path, qsl))
        layer_cfg[path] = {"group_size": gs, "bits": bits, "mode": "affine"}
        print(f"    [requantize] {ti + 1}/{len(targets)} {path} -> {bits}-bit", flush=True)
    if new_mods:
        model.update_modules(tree_unflatten(new_mods))
    return layer_cfg


@contextmanager
def _requantize_prequantized():
    """Let mlx-lm re-quantize a source whose weights are already quantized.

    Some bases ship pre-quantized weights that mlx-lm loads as ``Quantized*``
    modules — notably gpt-oss, whose MoE experts are native MXFP4 (4-bit). Those
    modules have no ``to_quantized`` method, so ``nn.quantize`` skips them and
    they pass through at their original bit-width: a 2-bit OptiQ target then only
    touches the (small) non-expert layers and the model never shrinks.

    This patch eagerly re-quantizes the pre-quantized fused-expert tensors to the
    OptiQ bit-widths (memory-bounded — see ``_eager_requantize_experts``) and
    registers them in a fresh affine quantization config, then lets the original
    ``quantize_model`` handle the remaining bf16 layers. No-op for ordinary bf16
    sources.
    """
    import sys
    import mlx.nn as nn
    import mlx_lm.convert  # ensure the submodule is imported
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    # ``convert`` binds ``quantize_model`` locally, so patch the convert *module*
    # (fetched via sys.modules — ``mlx_lm.convert`` the attr is the function).
    _mc = sys.modules["mlx_lm.convert"]
    _orig = _mc.quantize_model
    _qtypes = (nn.QuantizedLinear, nn.QuantizedEmbedding, QuantizedSwitchLinear)

    def _patched(model, config, group_size=None, bits=None, mode="affine",
                 quant_predicate=None, **kwargs):
        if any(isinstance(m, _qtypes) for _, m in model.named_modules()):
            print("  [requantize] source is pre-quantized; re-quantizing experts "
                  "to OptiQ bit-widths (mxfp4 -> bf16 -> affine, per-tensor)")
            def_gs, def_bits = group_size or 64, bits or 4
            config.pop("quantization_config", None)
            config["quantization"] = {
                "group_size": def_gs, "bits": def_bits, "mode": "affine"}
            expert_cfg = _eager_requantize_experts(
                model, quant_predicate, def_gs, def_bits)
            config["quantization"].update(expert_cfg)
        return _orig(model, config, group_size, bits, mode, quant_predicate, **kwargs)

    _mc.quantize_model = _patched
    try:
        yield
    finally:
        _mc.quantize_model = _orig


def convert_llm_to_mlx(
    hf_path: str,
    output_path: str,
    quant_predicate: Optional[Callable] = None,
    q_bits: int = 4,
    q_group_size: int = 64,
    metadata: Optional[dict] = None,
    preserve_mtp: bool = True,
) -> str:
    """Convert a HuggingFace LLM to quantized MLX format.

    Uses mlx-lm's convert() with optional custom quant_predicate for mixed-precision.
    The output config.json is left exactly as mlx-lm wrote it. Multi-modal base
    models that are being quantized as pure text are loaded by mlx-lm via their
    VLM-arch class, which delegates to the language sub-module for inference.

    When ``preserve_mtp`` is True (the default), any ``mtp.*`` tensors in the
    source — DeepSeek-V3-style heads found in Qwen3.5/3.6, MiMo, GLM,
    Nemotron-H, etc. — are fetched separately, quantized at the same
    (bits, group_size) the host was converted with, and written as a
    ``mtp.safetensors`` sidecar registered in ``config.json``. mlx-lm's
    Qwen3.5 loader drops these tensors silently, so without this step the
    converted artifact loses its in-checkpoint speculation head.
    """
    from mlx_lm import convert

    # mlx-lm refuses to overwrite existing dirs, so remove first
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    print(f"  Converting {hf_path} → {output_path}")
    print(f"  Quantization: {'mixed-precision (OptiQ)' if quant_predicate else f'uniform {q_bits}-bit'}")

    # mlx_lm.convert internally calls load(...) which uses strict=True; any
    # weight tensor in the bf16 source that the current mlx-lm model class
    # doesn't have a slot for (e.g. per-layer k_proj/v_proj on KV-shared
    # Gemma-4 layers, which were per-layer in the original Google release
    # but treated as shared in current mlx-lm) raises and aborts the
    # convert. Temporarily monkey-patch load_model to strict=False around
    # the convert call so extras are dropped silently. They're not in the
    # model anyway, so dropping is correct behavior.
    import mlx_lm.utils as _mlx_utils
    _orig_load_model = _mlx_utils.load_model
    def _load_model_lax(*args, **kwargs):
        kwargs["strict"] = False
        return _orig_load_model(*args, **kwargs)
    _mlx_utils.load_model = _load_model_lax
    try:
        with _requantize_prequantized():
            convert(
                hf_path=hf_path,
                mlx_path=output_path,
                quantize=True,
                q_bits=q_bits,
                q_group_size=q_group_size,
                quant_predicate=quant_predicate,
            )
    finally:
        _mlx_utils.load_model = _orig_load_model

    # Preserve MTP head tensors (silent no-op if source has none)
    if preserve_mtp:
        try:
            from ..runtime.mtp_convert import preserve_mtp as _preserve
            _preserve(
                source_repo=hf_path,
                output_path=output_path,
                bits=q_bits,
                group_size=q_group_size,
            )
        except Exception as exc:
            # Non-fatal: convert succeeded; only MTP preservation failed.
            print(f"  [mtp] preservation skipped: {exc}")

    # Ensure a recommended-sampling generation_config.json is present.
    # Qwen3.5 / Qwen3.6 don't publish one with their base models even
    # though their model cards recommend specific values; we write the
    # documented "instruct / non-thinking general" recipe so downstream
    # tools (Lab UI, mlx_lm.server's recommended-sampler injector) can
    # surface them. Other families fall through to whatever the source
    # already publishes.
    try:
        _write_recommended_generation_config(hf_path, output_path)
    except Exception as exc:
        print(f"  [gen_config] write skipped: {exc}")

    # Save OptiQ metadata alongside the model
    if metadata:
        meta_path = os.path.join(output_path, "optiq_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"  Saved optimization metadata to {meta_path}")

    # Report model size
    total_size = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if f.endswith((".safetensors", ".npz"))
    )
    print(f"  Output model size: {total_size / 1024 / 1024:.1f} MB")

    return output_path


# Per-family recommended sampling — the "non-thinking instruct general"
# recipe from each model author's published model card. Used when the
# source repo doesn't ship a generation_config.json with the same keys.
# Add new families here as we support them.
_FAMILY_RECOMMENDED_SAMPLING: dict[str, dict] = {
    "qwen3.5": {
        # From the Qwen3.5 model card "Instruct (or non-thinking) mode for
        # general tasks" recipe. Lifted verbatim.
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
    },
    "qwen3.6": {
        # Qwen3.6 publishes the same recipe in its README.
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
    },
    "minicpm5": {
        # From the MiniCPM5-1B model card: "No Think mode" recipe for
        # general / fast-assistant use. Think mode uses temperature=0.9,
        # top_p=0.95 but is selected at runtime via chat_template_kwargs,
        # not via generation_config defaults.
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
    },
}


def _family_from_hf_path(hf_path: str) -> str | None:
    """Map an HF repo id (e.g. ``Qwen/Qwen3.6-27B``) to a family key
    matching ``_FAMILY_RECOMMENDED_SAMPLING``. Returns None if no match."""
    n = hf_path.lower().rsplit("/", 1)[-1]
    if n.startswith("qwen3.5"):
        return "qwen3.5"
    if n.startswith("qwen3.6"):
        return "qwen3.6"
    if n.startswith("minicpm5"):
        return "minicpm5"
    return None


def _write_recommended_generation_config(hf_path: str, output_path: str) -> None:
    """Ensure the converted model directory has a generation_config.json
    with this family's recommended sampling values.

    If the file already exists (because mlx_lm.convert pulled it from the
    source), we merge our defaults under the existing values so any field
    the upstream author already published wins. If the file is absent
    (Qwen3.5 / 3.6 base repos don't ship one), we write a fresh file.
    Other families fall through silently.
    """
    family = _family_from_hf_path(hf_path)
    if family is None:
        return
    rec = _FAMILY_RECOMMENDED_SAMPLING.get(family)
    if not rec:
        return
    cfg_path = os.path.join(output_path, "generation_config.json")
    existing: dict = {}
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    # Recommended values fill in only what's missing — never overwrite
    # an upstream-published value.
    merged = {**rec, **existing}
    with open(cfg_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"  [gen_config] wrote recommended sampling defaults ({family})")


def convert_llm_uniform(
    hf_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 64,
) -> str:
    """Convert LLM with uniform quantization (baseline)."""
    return convert_llm_to_mlx(
        hf_path=hf_path,
        output_path=output_path,
        quant_predicate=None,
        q_bits=bits,
        q_group_size=group_size,
    )


def convert_llm_static_mixed(
    hf_path: str,
    output_path: str,
    recipe: str = "mixed_3_6",
    preserve_mtp: bool = True,
) -> str:
    """Convert LLM with mlx-lm's built-in static mixed-precision recipe.

    With ``preserve_mtp=True`` (default) the convert step is followed by
    the same MTP-tensor preservation that ``convert_llm_to_mlx`` runs, so
    static-recipe outputs also retain in-checkpoint MTP heads.
    """
    from mlx_lm import convert

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    print(f"  Converting {hf_path} → {output_path} (static recipe: {recipe})")

    with _requantize_prequantized():
        convert(
            hf_path=hf_path,
            mlx_path=output_path,
            quantize=True,
            quant_predicate=recipe,
        )

    if preserve_mtp:
        try:
            from ..runtime.mtp_convert import preserve_mtp as _preserve
            _preserve(source_repo=hf_path, output_path=output_path)
        except Exception as exc:
            print(f"  [mtp] preservation skipped: {exc}")

    total_size = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if f.endswith((".safetensors", ".npz"))
    )
    print(f"  Output model size: {total_size / 1024 / 1024:.1f} MB")
    return output_path


def _pt_to_mlx_vlm_path(pt_name: str) -> str:
    """Map PyTorch layer name to mlx-vlm layer path.

    PyTorch (HuggingFace):          MLX (mlx-vlm):
      model.visual.blocks.X.Y    → vision_tower.blocks.X.Y
      model.visual.merger.Y      → vision_tower.merger.Y
      model.language_model.Y     → language_model.model.Y
      lm_head                    → (tied, not quantized)
    """
    if pt_name == "lm_head":
        return "lm_head"
    if pt_name.startswith("model.visual."):
        return pt_name.replace("model.visual.", "vision_tower.", 1)
    if pt_name.startswith("model.language_model."):
        return pt_name.replace("model.language_model.", "language_model.model.", 1)
    return pt_name


def make_vlm_quant_predicate(
    configs: dict[str, int],
    default_bits: int = 4,
    group_size: int = 64,
) -> Callable:
    """Build a quant_predicate for mlx-vlm from OptiQ per-layer configs.

    Args:
        configs: Dict mapping PyTorch layer names to bit-widths.
        default_bits: Default bits for layers not in configs.
        group_size: Quantization group size.

    Returns:
        Callable suitable for mlx_vlm.convert(quant_predicate=...).
    """
    # Build MLX path → bits mapping
    mlx_configs = {}
    for pt_name, bits in configs.items():
        mlx_path = _pt_to_mlx_vlm_path(pt_name)
        mlx_configs[mlx_path] = bits

    def predicate(path: str, module) -> bool | dict:
        if path in mlx_configs:
            bits = mlx_configs[path]
            return {"bits": bits, "group_size": group_size}
        # For unmapped layers, use default
        return {"bits": default_bits, "group_size": group_size}

    return predicate


def convert_vlm_to_mlx(
    hf_path: str,
    output_path: str,
    quant_predicate: Optional[Callable] = None,
    q_bits: int = 4,
    q_group_size: int = 64,
    metadata: Optional[dict] = None,
) -> str:
    """Convert a HuggingFace VLM to quantized MLX format.

    Uses mlx-vlm's convert() with optional custom quant_predicate for mixed-precision.
    """
    from mlx_vlm import convert

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    print(f"  Converting VLM {hf_path} → {output_path}")
    print(f"  Quantization: {'mixed-precision (OptiQ)' if quant_predicate else f'uniform {q_bits}-bit'}")

    convert(
        hf_path=hf_path,
        mlx_path=output_path,
        quantize=True,
        q_bits=q_bits,
        q_group_size=q_group_size,
        quant_predicate=quant_predicate,
    )

    if metadata:
        meta_path = os.path.join(output_path, "optiq_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"  Saved optimization metadata to {meta_path}")

    total_size = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if f.endswith((".safetensors", ".npz"))
    )
    print(f"  Output model size: {total_size / 1024 / 1024:.1f} MB")

    return output_path


def make_audio_quant_predicate(
    configs: dict[str, int],
    default_bits: int = 4,
    group_size: int = 64,
    skip_prefixes: list[str] | None = None,
) -> "Callable":
    """Build a class_predicate for mx.nn.quantize from OptiQ per-layer configs.

    Layer names are MLX-native (e.g. model.layers.0.mlp.gate_proj).
    By default, skips audio_tower layers (kept in BF16).

    Returns:
        Callable suitable for mlx.nn.quantize(class_predicate=...).
    """
    import mlx.nn as nn

    if skip_prefixes is None:
        skip_prefixes = ["audio_tower."]

    def predicate(path: str, module):
        if not isinstance(module, nn.Linear):
            return False
        # Skip layers that should stay in full precision
        for prefix in skip_prefixes:
            if path.startswith(prefix):
                return False
        if path in configs:
            bits = configs[path]
            return {"bits": bits, "group_size": group_size}
        return {"bits": default_bits, "group_size": group_size}

    return predicate


def convert_audio_to_mlx(
    model,
    output_path: str,
    class_predicate: "Optional[Callable]" = None,
    q_bits: int = 4,
    q_group_size: int = 64,
    metadata: Optional[dict] = None,
    source_config_path: str | None = None,
) -> str:
    """Quantize an MLX audio model (Qwen3-ASR) and save to disk.

    Takes an already-loaded MLX model, applies quantization,
    and saves weights + config.

    Args:
        model: MLX model (e.g. from mlx_audio.stt.utils.load_model)
        output_path: Directory to save quantized model
        class_predicate: Per-layer quantization predicate
        q_bits: Default bit-width for uniform quantization
        q_group_size: Quantization group size
        metadata: OptiQ metadata to save alongside model
        source_config_path: Path to source config.json to copy
    """
    import mlx.nn as nn
    import mlx.core as mx
    from mlx.utils import tree_flatten

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    print(f"  Converting audio model → {output_path}")
    print(f"  Quantization: {'mixed-precision (OptiQ)' if class_predicate else f'uniform {q_bits}-bit'}")

    # Apply quantization (skip audio_tower by default to match mlx-audio format)
    if class_predicate:
        nn.quantize(model, class_predicate=class_predicate)
    else:
        def uniform_pred(path, module):
            if not isinstance(module, nn.Linear):
                return False
            if path.startswith("audio_tower."):
                return False
            return {"bits": q_bits, "group_size": q_group_size}
        nn.quantize(model, class_predicate=uniform_pred)

    # Save weights (as model.safetensors to match mlx-audio's expected naming)
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(os.path.join(output_path, "model.safetensors"), weights)

    # Collect per-layer quantization info from the quantized model
    per_layer_bits = {}
    if class_predicate:
        for path, module in model.named_modules():
            if hasattr(module, "bits"):
                per_layer_bits[path] = int(module.bits)

    # Copy config and tokenizer files from source
    if source_config_path and os.path.exists(source_config_path):
        src_dir = os.path.dirname(source_config_path)
        # Copy config with quantization info
        with open(source_config_path) as f:
            config = json.load(f)
        quant_config = {"bits": q_bits, "group_size": q_group_size}
        # Embed per-layer overrides so mlx-audio's apply_quantization()
        # creates QuantizedLinear with the correct bits for each layer
        for layer_path, bits in per_layer_bits.items():
            if bits != q_bits:
                quant_config[layer_path] = {"bits": bits, "group_size": q_group_size}
        config["quantization"] = quant_config
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        # Copy tokenizer and other necessary files (skip weight/index files)
        copy_exts = (".json", ".txt", ".model", ".tiktoken", ".py")
        skip_files = {"config.json", "model.safetensors"}
        for fname in os.listdir(src_dir):
            if fname in skip_files:
                continue
            # Skip safetensors index and shard files (we save our own weights)
            if "safetensors" in fname:
                continue
            if any(fname.endswith(ext) for ext in copy_exts):
                src_file = os.path.join(src_dir, fname)
                dst_file = os.path.join(output_path, fname)
                if os.path.isfile(src_file) and not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
        # Bake in the transformers Mistral pretokenizer-regex fix when the copied
        # tokenizer triggers it (Mistral-family tokenizers — e.g. poolside Laguna —
        # ship a buggy Split regex that transformers warns about and that mis-
        # tokenizes). No-op for every other tokenizer; best-effort so it can never
        # break a convert.
        _fix_mistral_tokenizer_regex(output_path)

    # Save per-layer quantization map in metadata
    if class_predicate and per_layer_bits:
        if metadata is None:
            metadata = {}
        metadata["per_layer_bits"] = per_layer_bits

    # Save metadata
    if metadata:
        meta_path = os.path.join(output_path, "optiq_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    total_size = get_model_size_mb(output_path)
    print(f"  Output model size: {total_size:.1f} MB")

    return output_path


def get_model_size_mb(path: str) -> float:
    """Get total model file size in MB."""
    total = 0
    for f in os.listdir(path):
        fp = os.path.join(path, f)
        if os.path.isfile(fp) and f.endswith((".safetensors", ".npz", ".bin")):
            total += os.path.getsize(fp)
    return total / (1024 * 1024)
