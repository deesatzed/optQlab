"""Fuse per-expert MoE weights that mlx-lm silently drops.

mlx-lm's ``qwen3_5_moe.Model.sanitize`` builds the fused
``switch_mlp.{gate,up,down}_proj`` tensors that ``QuantizedSwitchLinear``
expects, but it can only do so from an ALREADY-FUSED checkpoint::

    if f"{prefix}.experts.gate_up_proj" in new_weights:      # <-- the whole gate
        ...split gate_up in half, build switch_mlp...
    # else: nothing. No warning, no error.

``Qwen/Qwen3.5-35B-A3B`` ships that fused layout, so it works. But a MoE may
equally ship its experts SPLIT -- one tensor per (expert, projection)::

    model.language_model.layers.0.mlp.experts.0.gate_proj.weight
    model.language_model.layers.0.mlp.experts.0.up_proj.weight
    model.language_model.layers.0.mlp.experts.0.down_proj.weight
    ... x 256 experts x 40 layers = 30,720 tensors

``InternScience/Agents-A1`` does. The gate above is then False for every layer,
no ``switch_mlp.*`` weight is ever produced, and ``load_weights(strict=False)``
happily leaves every ``QuantizedSwitchLinear`` at its RANDOM INITIALISATION
while the 30,720 real expert tensors are discarded.

Nothing fails. The model loads, quantizes, serves, and passes every structural
check -- and emits fluent noise, because its experts are random numbers. We
measured 357% relative error against the true expert-0 weights, with double the
standard deviation: the fingerprint of untrained init.

This patch stacks the per-expert tensors into the fused
``[num_experts, out, in]`` layout before mlx-lm's sanitize runs, so the normal
path finds what it expects. Already-fused checkpoints are untouched.
"""

from __future__ import annotations


def _fuse_unfused_experts(weights: dict) -> dict:
    """Stack ``experts.{i}.{proj}.weight`` into ``experts.gate_up_proj`` /
    ``experts.down_proj``, the layout mlx-lm's sanitize consumes.

    Returns ``weights`` unchanged when the checkpoint is already fused (or has
    no experts at all), so this is safe to run on every model.
    """
    import re

    import mlx.core as mx

    pat = re.compile(r"^(.*\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
    # prefix -> proj -> {expert_index: tensor}
    found: dict[str, dict[str, dict[int, "mx.array"]]] = {}
    for key in list(weights):
        m = pat.match(key)
        if m:
            prefix, idx, proj = m.group(1), int(m.group(2)), m.group(3)
            found.setdefault(prefix, {}).setdefault(proj, {})[idx] = weights.pop(key)

    for prefix, projs in found.items():
        # Every projection must be present for every expert, or we would fuse a
        # partial tensor and corrupt the model in a subtler way than the bug we
        # are fixing. Refuse instead.
        counts = {p: len(d) for p, d in projs.items()}
        if set(projs) != {"gate", "up", "down"} or len(set(counts.values())) != 1:
            raise ValueError(
                f"{prefix}: cannot fuse per-expert weights -- expected gate/up/down "
                f"for the same set of experts, got {counts}")
        n = next(iter(counts.values()))

        def stack(proj: str):
            d = projs[proj]
            missing = [i for i in range(n) if i not in d]
            if missing:
                raise ValueError(f"{prefix}.{proj}_proj: experts {missing[:4]} missing")
            return mx.stack([d[i] for i in range(n)], axis=0)

        gate, up = stack("gate"), stack("up")
        # mlx-lm splits gate_up on the -2 axis, so concatenate there.
        weights[f"{prefix}.experts.gate_up_proj"] = mx.concatenate([gate, up], axis=-2)
        weights[f"{prefix}.experts.down_proj"] = stack("down")

    return weights


def install() -> None:
    """Wrap ``qwen3_5_moe.Model.sanitize`` so it also accepts unfused experts.

    Idempotent.
    """
    from mlx_lm.models import qwen3_5_moe as _m

    if getattr(_m.Model.sanitize, "_optiq_unfused", False):
        return

    _orig = _m.Model.sanitize

    def sanitize(self, weights):
        return _orig(self, _fuse_unfused_experts(dict(weights)))

    sanitize._optiq_unfused = True  # type: ignore[attr-defined]
    _m.Model.sanitize = sanitize  # type: ignore[assignment]
