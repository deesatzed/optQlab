"""LoRA fine-tuning for OptiQ-quantized DiffusionGemma.

DiffusionGemma is a discrete-diffusion LM, not autoregressive, so mlx-lm's LoRA
trainer (which loads via ``mlx_lm.load`` and optimizes a next-token cross-entropy
loss) does not apply. This module trains LoRA with the model's native
**denoising objective**: corrupt the target tokens to a random noise level
t ∈ [t_min, t_max] (replace each with prob t by a uniform-random token — the same
corruption the inference canvas starts from), run one encoder+decoder forward,
and minimise cross-entropy on the corrupted positions (predict the clean token).

LoRA is injected with mlx-lm's ``linear_to_lora_layers`` on the model's
``.layers`` (the decoder blocks). The encoder reuses those same blocks (weight-
tied via a weakref; only per-layer scalars differ), so a single injection trains
both the encode and decode paths. The base quantized weights stay frozen.

The adapter is saved as ``adapters.safetensors`` + ``adapter_config.json``
(mlx-lm/PEFT layout) and reloaded with :func:`load_diffusion_lora`.
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

from .generate import load

# Standard attention + dense-MLP projections (Unsloth-style 7). Experts/routers
# (SwitchLinear) are left to the frozen base — LoRA on fused MoE experts is a
# separate concern and the dense path carries most finetuning signal.
DEFAULT_LORA_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def _read_pairs(path: str) -> list[tuple[str, str]]:
    """Read a jsonl of {prompt, completion} or {messages:[...]} into pairs."""
    pairs = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "prompt" in d and "completion" in d:
            pairs.append((d["prompt"], d["completion"]))
        elif "messages" in d:
            msgs = d["messages"]
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            pairs.append((user, asst))
        elif "text" in d:  # split on a separator-free fallback: whole text is target
            pairs.append(("", d["text"]))
    return pairs


def _encode_pair(tokenizer, prompt: str, completion: str, max_canvas: int):
    """Chat-template the prompt (encoder) and tokenize the completion (canvas)."""
    try:
        p_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=False,
        )
        prompt_ids = tokenizer.encode(p_ids, add_special_tokens=False)
    except Exception:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    target_ids = tokenizer.encode(completion, add_special_tokens=False)[:max_canvas]
    return (
        mx.array(prompt_ids, dtype=mx.int32)[None],
        mx.array(target_ids, dtype=mx.int32)[None],
    )


def diffusion_loss(model, prompt_ids, target_ids, *, t_min, t_max, vocab_size):
    """Denoising CE on corrupted canvas positions."""
    B, L = target_ids.shape
    t = mx.random.uniform(low=t_min, high=t_max)
    corrupt = mx.random.uniform(shape=(B, L)) < t
    noise = mx.random.randint(0, vocab_size, (B, L)).astype(target_ids.dtype)
    canvas = mx.where(corrupt, noise, target_ids)

    logits = model(input_ids=prompt_ids, canvas_ids=canvas).logits  # (B, L, V)
    ce = nn.losses.cross_entropy(logits, target_ids, reduction="none")  # (B, L)
    denom = mx.maximum(corrupt.sum(), 1)
    return (ce * corrupt).sum() / denom


def train_diffusion_lora(
    model_path: str,
    data_dir: str,
    adapter_path: str,
    *,
    rank: int = 8,
    scale: float = 8.0,            # diffusion is more scale-sensitive than AR;
                                    # 20 (mlx-lm AR default) collapses, ~4-8 is stable
    dropout: float = 0.0,
    num_layers: int = -1,          # -1 = all decoder blocks
    iters: int = 200,
    learning_rate: float = 1e-4,
    max_canvas: int = 256,
    report_every: int = 10,
    seed: int = 0,
) -> str:
    """Train a LoRA adapter on DiffusionGemma and write it to ``adapter_path``."""
    model, tokenizer = load(model_path)
    vocab_size = model.config.text_config.vocab_size
    t_cfg = _diffusion_noise_range(model_path)

    model.freeze()
    lora_cfg = {"rank": rank, "scale": scale, "dropout": dropout, "keys": DEFAULT_LORA_KEYS}
    n_blocks = len(model.layers) if num_layers < 0 else num_layers
    from mlx_lm.tuner.utils import linear_to_lora_layers
    linear_to_lora_layers(model, n_blocks, lora_cfg)

    # Training mode is REQUIRED: the MoE Experts only stop_gradient their routing
    # indices when self.training is True; otherwise the backward hits
    # GatherQMM::vjp on the (non-differentiable) top-k indices and raises.
    model.train()

    trainable = [(k, v) for k, v in tree_flatten(model.trainable_parameters())]
    n_train = sum(v.size for _, v in trainable)
    print(f"  [diffusion-lora] {len(trainable)} LoRA tensors, {n_train/1e6:.2f}M trainable params")

    pairs = _read_pairs(os.path.join(data_dir, "train.jsonl"))
    if not pairs:
        raise ValueError(f"no training pairs in {data_dir}/train.jsonl")
    print(f"  [diffusion-lora] {len(pairs)} training examples, "
          f"noise t∈[{t_cfg[0]:.2f},{t_cfg[1]:.2f}]")

    opt = optim.AdamW(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, diffusion_loss)

    # deterministic-ish example order via index arithmetic (no Math.random in MLX scripts)
    losses = []
    for it in range(iters):
        prompt, completion = pairs[it % len(pairs)]
        prompt_ids, target_ids = _encode_pair(tokenizer, prompt, completion, max_canvas)
        if target_ids.shape[1] == 0:
            continue
        loss, grads = loss_and_grad(
            model, prompt_ids, target_ids,
            t_min=t_cfg[0], t_max=t_cfg[1], vocab_size=vocab_size,
        )
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state, loss)
        losses.append(float(loss))
        if (it + 1) % report_every == 0:
            window = losses[-report_every:]
            print(f"  [diffusion-lora] iter {it+1}/{iters}  loss {sum(window)/len(window):.4f}",
                  flush=True)

    _save_adapter(model, adapter_path, lora_cfg, model_path)
    print(f"  [diffusion-lora] saved adapter → {adapter_path}")
    return adapter_path


def _diffusion_noise_range(model_path: str) -> tuple[float, float]:
    """Read t_min/t_max from generation_config.json (defaults 0.4/0.8)."""
    from .loader import _resolve_dir
    try:
        gc = json.load(open(os.path.join(_resolve_dir(model_path), "generation_config.json")))
        return float(gc.get("t_min", 0.4)), float(gc.get("t_max", 0.8))
    except Exception:
        return 0.4, 0.8


def _save_adapter(model, adapter_path: str, lora_cfg: dict, base: str) -> None:
    out = Path(adapter_path)
    out.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out / "adapters.safetensors"), weights)
    json.dump(
        {
            "base_model": base,
            "fine_tune_type": "lora",
            "model_type": "diffusion_gemma",
            "lora_parameters": {"rank": lora_cfg["rank"], "scale": lora_cfg["scale"],
                                "dropout": lora_cfg["dropout"], "keys": lora_cfg["keys"]},
        },
        open(out / "adapter_config.json", "w"), indent=2,
    )


def load_diffusion_lora(model_path: str, adapter_path: str):
    """Load a DiffusionGemma quant and apply a trained LoRA adapter → (model, tokenizer)."""
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, tokenizer = load(model_path)
    cfg = json.load(open(os.path.join(adapter_path, "adapter_config.json")))
    lp = cfg["lora_parameters"]
    model.freeze()
    linear_to_lora_layers(
        model, len(model.layers),
        {"rank": lp["rank"], "scale": lp["scale"], "dropout": lp.get("dropout", 0.0),
         "keys": lp["keys"]},
    )
    weights = mx.load(os.path.join(adapter_path, "adapters.safetensors"))
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    model.eval()
    return model, tokenizer
