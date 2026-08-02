"""Convert-time MTP-tensor preservation.

mlx-lm's Qwen3.5 / Qwen3.6 model loaders don't declare an MTP submodule,
so any ``mtp.*`` tensors present in the source ``.safetensors`` are
silently dropped by ``mlx_lm.convert``. This module is the integrated
post-convert step that:

1. Inspects the source ``model.safetensors.index.json`` for ``mtp.*``
   keys (DeepSeek-V3-style heads found in Qwen3.5/3.6 family + future
   variants like MiMo / GLM / Nemotron-H).
2. Fetches only the shard(s) that contain those tensors (typically
   ~1-2 files, not the whole model).
3. Quantizes the projection weights at the same bits/group_size the host
   was converted with (so mlx-lm's ``QuantizedLinear`` accepts them at
   runtime); leaves ``mtp.fc.weight`` and norms at BF16 — empirically
   best on M-series M4 24GB (3-run median on Qwen3.5-9B, see
   project_optiq_mtp_qwen35_results.md).
4. Writes ``mtp.safetensors`` next to the main shards and registers
   it in ``config.json`` via the MTPLX sidecar convention
   (``mtp_file`` + ``mtplx_mtp_quantization`` block).

Called from ``backends/mlx_backend.convert_llm_to_mlx`` as a no-op when
the source has no MTP tensors. Users get MTP-preserving converts for
free; no separate ``mtp-patch`` command needed.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Optional


# fc + norms stay at BF16 — verified to be the empirical sweet spot
# (Qwen3.5-9B on M4: variant A "4-bit proj + fc BF16" was fastest).
_MTP_QUANT_SKIPLIST = {"mtp.fc.weight"}


def _is_quantize_target(name: str) -> bool:
    """True if this MTP tensor should be quantized (projection weight)."""
    if not name.endswith(".weight"):
        return False
    if name in _MTP_QUANT_SKIPLIST:
        return False
    leaf = name[: -len(".weight")].split(".")[-1].lower()
    return "norm" not in leaf


def _read_host_quant(model_dir: str) -> tuple[Optional[int], Optional[int]]:
    """Read the host model's group_size / bits from its config.json.
    The MTP sidecar MUST match these so mlx-lm's QuantizedLinear, which
    uses one global (bits, group_size), doesn't shape-mismatch."""
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        return None, None
    with open(cfg_path) as f:
        cfg = json.load(f)
    qc = cfg.get("quantization") or cfg.get("quantization_config")
    if not isinstance(qc, dict):
        return None, None
    return qc.get("group_size"), qc.get("bits")


def _safetensors_header_keys(path: str) -> list[str]:
    """Tensor names in a .safetensors file, read from its header alone.

    The header is a JSON blob whose byte-length is the leading u64, so this
    never loads a weight -- it is cheap even on an 18 GB single-shard model.
    """
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return [k for k in hdr if k != "__metadata__"]


def _find_mtp_in_source(source_repo: str, cache_dir: Optional[str] = None):
    """Return (shard_filenames, mtp_tensor_keys) for the source repo.

    Source can be either an HF repo id or a local directory. Returns
    ``([], [])`` if no MTP tensors are present.

    Sharded checkpoints are described by ``model.safetensors.index.json``, but a
    model small enough to fit one file ships a bare ``model.safetensors`` with no
    index. This used to read the index and nothing else, so a single-shard model
    reported "no MTP" and had its speculative-decoding head silently dropped --
    empero-ai/Qwythos-9B-v2 ships 18 GB in one file, with 15 mtp.* tensors that
    never made it into the quant. Fall back to the shard's own header.
    """
    def _keys_from_index(idx_path):
        with open(idx_path) as f:
            weight_map = json.load(f).get("weight_map", {})
        mtp_keys = sorted(k for k in weight_map if k.startswith("mtp."))
        return sorted({weight_map[k] for k in mtp_keys}), mtp_keys

    def _keys_from_single_shard(shard_path):
        mtp_keys = sorted(k for k in _safetensors_header_keys(shard_path)
                          if k.startswith("mtp."))
        return ([os.path.basename(shard_path)] if mtp_keys else []), mtp_keys

    if os.path.isdir(source_repo):
        idx = os.path.join(source_repo, "model.safetensors.index.json")
        if os.path.exists(idx):
            return _keys_from_index(idx)
        shard = os.path.join(source_repo, "model.safetensors")
        return _keys_from_single_shard(shard) if os.path.exists(shard) else ([], [])

    from huggingface_hub import hf_hub_download
    try:
        idx = hf_hub_download(source_repo, "model.safetensors.index.json",
                              cache_dir=cache_dir)
        return _keys_from_index(idx)
    except Exception:
        pass
    try:
        shard = hf_hub_download(source_repo, "model.safetensors",
                                cache_dir=cache_dir)
    except Exception:
        return [], []
    return _keys_from_single_shard(shard)


def _resolve_shard_paths(source_repo: str, shards: list[str],
                         cache_dir: Optional[str] = None) -> list[str]:
    if os.path.isdir(source_repo):
        return [os.path.join(source_repo, s) for s in shards]
    from huggingface_hub import hf_hub_download
    return [
        hf_hub_download(source_repo, s, cache_dir=cache_dir) for s in shards
    ]


def _extract_mtp_tensors(shard_paths: list[str], mtp_keys: list[str]) -> dict:
    """Memory-mapped load of just the MTP tensors. Skips non-MTP entries."""
    import mlx.core as mx
    keys = set(mtp_keys)
    out: dict[str, "mx.array"] = {}
    for path in shard_paths:
        shard = mx.load(path)
        if not isinstance(shard, dict):
            raise RuntimeError(f"mx.load({path}) returned {type(shard).__name__}")
        for k in list(shard.keys()):
            if k in keys:
                out[k] = shard[k]
        del shard
    missing = keys - set(out.keys())
    if missing:
        raise RuntimeError(
            f"Missing MTP tensors after extract: {sorted(missing)[:5]}"
        )
    return out


def _quantize_mtp(tensors: dict, bits: int, group_size: int) -> tuple[dict, list[str]]:
    """Quantize projection weights; pass norms + skiplist through at BF16.

    Returns (output_tensors, list_of_quantized_tensor_keys).
    """
    import mlx.core as mx
    out: dict[str, "mx.array"] = {}
    quantized_keys: list[str] = []
    for name, arr in tensors.items():
        if _is_quantize_target(name):
            if arr.dtype not in (mx.float16, mx.bfloat16, mx.float32):
                arr = arr.astype(mx.bfloat16)
            w_q, scales, biases = mx.quantize(arr, group_size=group_size, bits=bits)
            base = name[: -len(".weight")]
            out[f"{base}.weight"] = w_q
            out[f"{base}.scales"] = scales
            out[f"{base}.biases"] = biases
            quantized_keys.append(name)
        else:
            out[name] = arr
    mx.eval(*out.values())
    return out, quantized_keys


def _register_mtp_sidecar(
    model_dir: str,
    mtp_filename: str,
    mtp_tensor_count: int,
    bits: int,
    group_size: int,
    quantized_keys: list[str],
    source_mtp_layers: int,
) -> dict:
    """Write the MTPLX-compatible sidecar registration into config.json."""
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    written = {
        "mtp_file": mtp_filename,
        "mtp_tensor_count": mtp_tensor_count,
        "mtp_policy": f"optiq-int{bits}-prequantized-gs{group_size}",
        "mtplx_mtp_quantization": {
            "bits": bits,
            "group_size": group_size,
            "mode": "affine",
            "policy": "cyankiwi" if "mtp.fc.weight" not in quantized_keys else "all",
            "prequantized": True,
        },
    }
    cfg.update(written)

    # The OptiQ runtime resolver reads the head's path from here first, so a
    # sidecar under the optiq/ subfolder is still found by name.
    extra = cfg.get("mlx_lm_extra_tensors")
    if not isinstance(extra, dict):
        extra = {}
    extra["mtp_file"] = mtp_filename
    cfg["mlx_lm_extra_tensors"] = extra

    # If source declared mtp_num_hidden_layers, propagate it.
    # If the source config didn't declare it (Qwen3.5 case — has weights
    # but no config field), add it based on what we saw in the tensors.
    if "mtp_num_hidden_layers" not in cfg:
        cfg["mtp_num_hidden_layers"] = source_mtp_layers
        written["mtp_num_hidden_layers"] = source_mtp_layers

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return written


def _infer_n_mtp_layers(mtp_keys: list[str]) -> int:
    """Count distinct mtp.layers.{n} prefixes."""
    indices = set()
    for k in mtp_keys:
        parts = k.split(".")
        if len(parts) >= 3 and parts[0] == "mtp" and parts[1] == "layers":
            try:
                indices.add(int(parts[2]))
            except ValueError:
                continue
    return max(len(indices), 1)


def preserve_mtp(
    source_repo: str,
    output_path: str,
    bits: Optional[int] = None,
    group_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> bool:
    """If the source has ``mtp.*`` tensors, fetch + quantize them and
    write a ``mtp.safetensors`` sidecar into ``output_path``.

    Auto-detects host quant params (bits/group_size) from the output's
    ``config.json`` so the sidecar matches mlx-lm's runtime expectations.
    Explicit ``bits`` / ``group_size`` override the auto-detected values.

    Returns ``True`` if a sidecar was written, ``False`` otherwise.
    """
    shards, mtp_keys = _find_mtp_in_source(source_repo, cache_dir=cache_dir)
    if not mtp_keys:
        return False  # no MTP — silent no-op

    print(f"  [mtp] Found {len(mtp_keys)} MTP tensors in source across "
          f"{len(shards)} shard(s).")

    # Auto-detect host quant params for shape compatibility
    host_gs, host_bits = _read_host_quant(output_path)
    if group_size is None:
        group_size = host_gs if host_gs is not None else 64
    if bits is None:
        bits = host_bits if host_bits is not None else 4

    print(f"  [mtp] Fetching shards + extracting MTP tensors...")
    shard_paths = _resolve_shard_paths(source_repo, shards, cache_dir=cache_dir)
    tensors = _extract_mtp_tensors(shard_paths, mtp_keys)
    n_params = sum(int(t.size) for t in tensors.values())
    print(f"  [mtp] Extracted {len(tensors)} tensors, {n_params:,} params.")

    print(f"  [mtp] Quantizing projections at {bits}-bit "
          f"(group_size={group_size}); norms + mtp.fc.weight stay at BF16.")
    output_tensors, quantized_keys = _quantize_mtp(tensors, bits, group_size)

    import mlx.core as mx
    from ..sidecar_layout import MTP_BASENAME, canonical_rel, write_path
    # Under optiq/ so a `*.safetensors` glob (mlx-vlm, LM Studio) can't pick it
    # up and fail a strict load; config records this relative path.
    mtp_filename = canonical_rel(MTP_BASENAME)
    out_safetensors = str(write_path(output_path, MTP_BASENAME))
    mx.save_safetensors(out_safetensors, output_tensors)
    size_mb = os.path.getsize(out_safetensors) / 1024 / 1024
    print(f"  [mtp] Wrote {mtp_filename} ({size_mb:.1f} MB).")

    n_layers = _infer_n_mtp_layers(mtp_keys)
    written = _register_mtp_sidecar(
        output_path,
        mtp_filename=mtp_filename,
        mtp_tensor_count=len(output_tensors),
        bits=bits,
        group_size=group_size,
        quantized_keys=quantized_keys,
        source_mtp_layers=n_layers,
    )
    print(f"  [mtp] Registered sidecar in config.json: {list(written.keys())}")
    return True
