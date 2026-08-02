"""Image+text LoRA fine-tuning for OptiQ-quantized VLMs (Gemma-4, Qwen3.5/3.6).

OptiQ keeps the vision tower at bf16 in the ``optiq_vision`` sidecar and quantizes
only the language tower. This module fine-tunes the *language* tower on
image+text data: the frozen vision frontend encodes each image into soft tokens
and scatters them into the language embedding sequence (exactly as inference
does), and we optimise a next-token cross-entropy loss on the **assistant target
only** (the prompt and the image-token positions are masked out).

The vision tower stays **frozen** — this is the common, low-risk VLM adaptation
(domain VQA / OCR / captioning), not vision-encoder tuning. LoRA is injected on
the language tower's attention + dense-MLP projections via mlx-lm's
``linear_to_lora_layers``; the base 4-bit weights stay frozen.

Because the merged input embeddings depend only on *frozen* params (the vision
tower and ``embed_tokens``), they are computed **outside** the differentiated
function as a constant — the gradient flows only through the LoRA-adapted
transformer blocks. The adapter is saved in mlx-lm/PEFT layout
(``adapters.safetensors`` + ``adapter_config.json``) and reloads with
:func:`load_vlm_lora`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# Attention + dense-MLP projections. The Qwen3.5/3.6 family is *hybrid*
# (full_attention_interval=4): every 4th layer is standard full-attention
# (self_attn.*); the rest are linear/gated-delta attention (linear_attn.*).
#
# We adapt MLP (all layers) + full-attention (the ~25% full-attn layers) ONLY.
# We deliberately do NOT put LoRA on the linear_attn (gated-delta) projections:
# differentiating w.r.t. those weights forces the gated-delta recurrence's
# weight-gradient backward, which blows up training memory (OOMs even small
# models here, ~19 GB on a 0.8B) without the custom O(sqrt(T)) gated-delta
# training kernel (optiq.ops.enable_gated_delta_training). MLP-everywhere +
# full-attn carries the adaptation signal and trains in ~12 GB. (Adding
# gated-delta LoRA is a future item gated on wiring that kernel into this path.)
# MoE experts / routers (SwitchLinear) are left to the frozen base.
DEFAULT_LORA_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


# ---------------------------------------------------------------------------
# Model + frontend loading
# ---------------------------------------------------------------------------

def _resolve_dir(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path)


def _load_model_and_frontend(model_path: str):
    """Load the VLM quant (language tower via mlx-lm) and its vision frontend.

    Returns ``(model, language_model, tokenizer, frontend, config)``. Raises if
    the artifact has no ``optiq_vision`` sidecar / registered frontend."""
    from mlx_lm import load

    from . import get_frontend, has_vision_sidecar

    model_dir = _resolve_dir(model_path)
    if not has_vision_sidecar(model_dir):
        raise FileNotFoundError(
            f"{model_dir} has no optiq_vision sidecar — not a VLM artifact. "
            f"For text-only LoRA use `optiq lora train`."
        )
    config = json.load(open(os.path.join(model_dir, "config.json")))
    factory = get_frontend(config.get("model_type", ""))
    if factory is None:
        raise ValueError(
            f"no vision frontend registered for model_type "
            f"{config.get('model_type')!r}"
        )

    model, tokenizer = load(model_dir)
    language_model = getattr(model, "language_model", model)
    frontend = factory(model_dir, language_model)
    return model, language_model, tokenizer, frontend, config


def _lm_head(language_model):
    """Return a callable hidden -> logits for either an untied lm_head or a
    weight-tied embedding (``embed_tokens.as_linear``)."""
    head = getattr(language_model, "lm_head", None)
    if head is not None:
        return head
    embed = language_model.model.embed_tokens
    return lambda h, _e=embed: _e.as_linear(h)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _read_examples(path: str) -> list[dict]:
    """Read an image+text jsonl. Each line is one of:

      * ``{"image": <path|url>, "prompt": <str>, "completion": <str>}``
      * ``{"images": [<path|url>...], "prompt": <str>, "completion": <str>}``
      * ``{"messages": [{role, content[...] with image parts}, ...]}`` where the
        final assistant turn is the target.

    Returns normalised dicts ``{"messages": <user turn list>, "target": <str>}``.
    """
    out: list[dict] = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "messages" in d:
            msgs = d["messages"]
            target = next((m["content"] for m in reversed(msgs)
                           if m["role"] == "assistant"), None)
            if target is None:
                continue
            user_turns = [m for m in msgs if m["role"] != "assistant"]
            out.append({"messages": user_turns,
                        "target": _as_text(target)})
            continue
        images = d.get("images") or ([d["image"]] if d.get("image") else [])
        prompt = d.get("prompt") or d.get("question") or ""
        target = d.get("completion") or d.get("answer") or d.get("text") or ""
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        out.append({"messages": [{"role": "user", "content": content}],
                    "target": _as_text(target)})
    return out


def _as_text(content) -> str:
    if isinstance(content, str):
        return content
    # list of parts -> concat text parts
    return "".join(p.get("text", "") for p in content
                   if isinstance(p, dict) and p.get("type") == "text")


def letterbox(img, size: int):
    """Aspect-preserving resize onto a fixed ``size``x``size`` white canvas.

    Training MUST see images of a UNIFORM shape: with variable image sizes the
    Metal unified-memory allocator ratchets its peak allocation up on every
    larger image and never returns it, so memory climbs unbounded and swaps the
    machine (a 0.8B run hit ~20 GB resident on a 24 GB Mac). A constant shape
    keeps the per-step allocation constant and bounded — the same reason a
    uniform-crop dataset (LaTeX-OCR) trained cleanly while variable charts did
    not. This is also the Lab's dataset "standardize" step."""
    from PIL import Image

    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def _uniform_messages(messages: list[dict], size: int) -> list[dict]:
    """Return a copy of ``messages`` with every image part replaced by a
    letterboxed PIL image of fixed ``size``x``size`` (so all examples share one
    sequence shape). Non-image content passes through untouched."""
    from PIL import Image

    out = []
    for m in messages:
        c = m.get("content")
        if not isinstance(c, list):
            out.append(m)
            continue
        nc = []
        for part in c:
            if isinstance(part, dict) and part.get("type") in ("image", "image_url"):
                src = part.get("image") or part.get("image_url") or part.get("url")
                if isinstance(src, str):
                    src = Image.open(src)
                nc.append({"type": "image", "image": letterbox(src, size)})
            else:
                nc.append(part)
        out.append({"role": m["role"], "content": nc})
    return out


def _build_forward_inputs(frontend, tokenizer, example: dict, *, max_target: int,
                          image_size: int | None = None):
    """Build the constant merged embeddings + loss mask for one example.

    Returns ``(merged, extra, full_ids, loss_mask)`` where ``merged`` are the
    input embeddings for ``[prompt(+image tokens) | target | eos]`` and
    ``loss_mask`` marks the *label* positions that fall in the target region.
    All four are constants w.r.t. the LoRA params (vision tower + embed_tokens
    are frozen)."""
    messages = example["messages"]
    if image_size:
        messages = _uniform_messages(messages, image_size)
    pre = frontend.preprocess(messages, tokenizer=tokenizer,
                              enable_thinking=False)
    prompt_ids = pre["input_ids"]                       # (1, Lp)
    Lp = int(prompt_ids.shape[1])

    eos = tokenizer.eos_token_id
    tgt = tokenizer.encode(example["target"], add_special_tokens=False)[:max_target]
    if eos is not None:
        tgt = tgt + [eos]
    target_ids = mx.array(tgt, dtype=prompt_ids.dtype)[None]   # (1, Lt)
    Lt = int(target_ids.shape[1])

    full_ids = mx.concatenate([prompt_ids, target_ids], axis=1)  # (1, Lp+Lt)
    full_inputs = dict(pre)
    full_inputs["input_ids"] = full_ids

    merged, extra = frontend.merged_embeddings(full_inputs)      # (1, L, H)
    merged = mx.stop_gradient(merged)

    # Causal-LM loss mask over LABEL positions (labels = full_ids[:, 1:]).
    # Label j (predicting full token j+1) is in the target region iff j+1 >= Lp.
    L = Lp + Lt
    label_idx = mx.arange(L - 1) + 1                             # full index of each label
    loss_mask = (label_idx >= Lp).astype(mx.float32)[None]      # (1, L-1)
    return merged, extra, full_ids, loss_mask


def _loss(language_model, merged, extra, full_ids, loss_mask):
    """Next-token CE on the masked (target) positions. The lm-head is resolved
    from the traced ``language_model`` so the matmul uses the differentiated
    module's (frozen) weights."""
    inner = language_model.model
    kw = {"input_embeddings": merged}
    pli = extra.get("per_layer_inputs") if extra else None
    if pli is not None:
        kw["per_layer_inputs"] = pli
    hidden = inner(None, **kw)                          # (1, L, H)
    logits = _lm_head(language_model)(hidden).astype(mx.float32)   # (1, L, V)

    logits = logits[:, :-1, :]
    labels = full_ids[:, 1:]
    ce = nn.losses.cross_entropy(logits, labels, reduction="none")  # (1, L-1)
    denom = mx.maximum(loss_mask.sum(), 1.0)
    return (ce * loss_mask).sum() / denom


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_vlm_lora(
    model_path: str,
    data_path: str,
    adapter_path: str,
    *,
    rank: int = 8,
    scale: float = 8.0,            # the Qwen3.5/3.6 hybrid family collapses at
                                    # the mlx-lm AR default of 20 (cf. the
                                    # diffusion_gemma trainer); ~4-8 is stable.
    dropout: float = 0.0,
    num_layers: int = -1,
    iters: int = 200,
    learning_rate: float = 1e-4,
    max_target: int = 256,
    image_size: int = 512,         # letterbox every image to this fixed square;
                                    # uniform size is REQUIRED for bounded memory
                                    # (see letterbox()). 512 keeps the gated-delta
                                    # backward comfortably under 24 GB.
    report_every: int = 10,
    keys: list[str] | None = None,
    grad_checkpoint: bool = True,  # recompute layer activations in the backward;
                                    # ESSENTIAL for the qwen3.5/3.6 hybrid
                                    # (gated-delta) backward to fit on 24 GB.
    grad_clip: float = 1.0,        # clip global grad norm; prevents the mode
                                    # collapse seen with short targets + high LR.
    progress_callback=None,        # called every report with {step, loss, iters}
) -> str:
    """Fine-tune a LoRA adapter on an OptiQ VLM quant with image+text data.

    ``data_path`` is a jsonl (see :func:`_read_examples`). The vision tower is
    frozen; only the language tower's LoRA params train. Writes the adapter to
    ``adapter_path`` and returns it."""
    model, lm, tokenizer, frontend, config = _load_model_and_frontend(model_path)
    keys = keys or DEFAULT_LORA_KEYS

    lm.freeze()
    lora_cfg = {"rank": rank, "scale": scale, "dropout": dropout, "keys": keys}
    n_blocks = len(lm.model.layers) if num_layers < 0 else num_layers
    from mlx_lm.tuner.utils import linear_to_lora_layers
    linear_to_lora_layers(lm, n_blocks, lora_cfg)
    # MoE experts only stop_gradient their routing indices when training; also
    # enables dropout. Harmless for dense models.
    lm.train()

    # Gradient checkpointing: recompute each decoder block's activations during
    # the backward instead of storing them. All qwen3.5/3.6 blocks share one
    # DecoderLayer class, so checkpointing layer[0]'s type covers all of them —
    # including the gated-delta (linear_attn) blocks whose backward is the
    # memory hog. Without this the hybrid backward needs ~21 GB on a 0.8B (too
    # tight on 24 GB); with it the working set drops to a few GB.
    if grad_checkpoint:
        from mlx_lm.tuner.trainer import grad_checkpoint as _grad_ckpt
        _grad_ckpt(lm.model.layers[0])

    trainable = [(k, v) for k, v in tree_flatten(lm.trainable_parameters())]
    n_train = sum(v.size for _, v in trainable)
    print(f"  [vlm-lora] {len(trainable)} LoRA tensors, "
          f"{n_train/1e6:.2f}M trainable params over {n_blocks} blocks"
          f"{' · grad-checkpoint ON' if grad_checkpoint else ''}")

    examples = _read_examples(data_path)
    if not examples:
        raise ValueError(f"no training examples in {data_path}")
    print(f"  [vlm-lora] {len(examples)} examples, model_type="
          f"{config.get('model_type')}")

    opt = optim.AdamW(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(lm, _loss)

    losses: list[float] = []
    t0 = time.time()
    for it in range(iters):
        ex = examples[it % len(examples)]
        try:
            merged, extra, full_ids, mask = _build_forward_inputs(
                frontend, tokenizer, ex, max_target=max_target,
                image_size=image_size)
        except Exception as exc:  # noqa: BLE001 — skip a malformed example
            print(f"  [vlm-lora] skip example {it % len(examples)}: {exc}")
            continue
        if float(mask.sum()) == 0.0:
            continue
        loss, grads = loss_and_grad(lm, merged, extra, full_ids, mask)
        if grad_clip and grad_clip > 0:
            grads, _ = optim.clip_grad_norm(grads, grad_clip)
        opt.update(lm, grads)
        mx.eval(lm.trainable_parameters(), opt.state, loss)
        losses.append(float(loss))
        # Free this step's big intermediates. The real memory fix is UNIFORM
        # image sizes (the letterbox above): with one shape the buffer pool
        # reuses the same buffers and stays bounded, so a periodic trim is
        # plenty. Per-iter clear_cache is counter-productive — it defeats buffer
        # reuse and tanks throughput (~3-4x slower). Variable sizes are what
        # ratchet memory unbounded; the answer is uniform shapes, not constant
        # cache-clearing.
        del merged, grads
        if (it + 1) % 50 == 0:
            mx.clear_cache()
        if (it + 1) % report_every == 0:
            window = losses[-report_every:]
            avg = sum(window) / len(window)
            print(f"  [vlm-lora] iter {it+1}/{iters}  "
                  f"loss {avg:.4f}  "
                  f"({(it+1)/(time.time()-t0):.2f} it/s, "
                  f"{mx.get_active_memory()/1e9:.1f}GB active)", flush=True)
            if progress_callback is not None:
                try:
                    progress_callback({"step": it + 1, "loss": avg, "iters": iters})
                except Exception:
                    pass

    _save_adapter(lm, adapter_path, lora_cfg, model_path, config)
    print(f"  [vlm-lora] saved adapter -> {adapter_path}")
    return adapter_path


def _save_adapter(language_model, adapter_path: str, lora_cfg: dict,
                  base: str, config: dict) -> None:
    out = Path(adapter_path)
    out.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(language_model.trainable_parameters()))
    mx.save_safetensors(str(out / "adapters.safetensors"), weights)
    json.dump(
        {
            "base_model": base,
            "fine_tune_type": "lora",
            "model_type": config.get("model_type", "vlm"),
            "vlm": True,
            "lora_parameters": {
                "rank": lora_cfg["rank"], "scale": lora_cfg["scale"],
                "dropout": lora_cfg["dropout"], "keys": lora_cfg["keys"],
            },
        },
        open(out / "adapter_config.json", "w"), indent=2,
    )


def load_vlm_lora(model_path: str, adapter_path: str):
    """Load an OptiQ VLM quant + trained LoRA adapter.

    Returns ``(model, language_model, tokenizer, frontend)`` ready for image+text
    inference (e.g. via ``OptiqEngine.from_loaded(model, tokenizer, model_path)``)."""
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, lm, tokenizer, frontend, _ = _load_model_and_frontend(model_path)
    cfg = json.load(open(os.path.join(adapter_path, "adapter_config.json")))
    lp = cfg["lora_parameters"]
    lm.freeze()
    linear_to_lora_layers(
        lm, len(lm.model.layers),
        {"rank": lp["rank"], "scale": lp["scale"],
         "dropout": lp.get("dropout", 0.0), "keys": lp["keys"]},
    )
    weights = mx.load(os.path.join(adapter_path, "adapters.safetensors"))
    lm.update(tree_unflatten(list(weights.items())))
    mx.eval(lm.parameters())
    lm.eval()
    return model, lm, tokenizer, frontend
