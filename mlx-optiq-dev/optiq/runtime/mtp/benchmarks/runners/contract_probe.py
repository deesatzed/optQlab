"""Hidden-state and concat-order MTP contract probes."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from optiq.runtime.mtp.benchmarks.schema import encode_prompt_case, load_prompt_suite
from optiq.runtime.mtp.runtime import load

CONTRACT_VARIANTS = (
    ("post_norm", "embedding_hidden"),
    ("post_norm", "hidden_embedding"),
    ("pre_norm", "embedding_hidden"),
    ("pre_norm", "hidden_embedding"),
)


def run_contract_probe(
    model_path: Path | str,
    prompt_suite: Path | str,
    *,
    max_prompt_tokens: int = 256,
    chat_template: bool = True,
    enable_thinking: bool | None = None,
) -> dict[str, Any]:
    """Measure greedy MTP agreement at the scheduler-aligned first MTP step."""
    rt = load(model_path, mtp=True)
    prompts = load_prompt_suite(prompt_suite)
    all_started = time.perf_counter()
    results = []

    for hidden_variant, concat_order in CONTRACT_VARIANTS:
        matches = 0
        rows = []
        elapsed_s = 0.0
        for case in prompts:
            ids = encode_prompt_case(
                rt.tokenizer,
                case,
                chat_template=chat_template,
                enable_thinking=enable_thinking,
            )[-max_prompt_tokens:]
            cache = rt.make_cache()
            started = time.perf_counter()

            # Reference scheduler skips MTP on multi-token prefill.
            prefill_logits = rt.forward_ar(mx.array([ids]), cache=cache, return_hidden=False)
            mx.eval(prefill_logits)
            first = int(mx.argmax(prefill_logits[:, -1, :], axis=-1).item())

            step_logits, step_hidden = rt.forward_ar(
                mx.array([[first]]),
                cache=cache,
                return_hidden=True,
                hidden_variant=hidden_variant,
            )
            mx.eval(step_logits, step_hidden)
            second = int(mx.argmax(step_logits[:, -1, :], axis=-1).item())

            draft_logits = rt.draft_mtp(
                step_hidden[:, -1:, :],
                mx.array([[second]]),
                mtp_cache=rt.make_mtp_cache(),
                concat_order=concat_order,
            )
            mx.eval(draft_logits)
            draft_third = int(mx.argmax(draft_logits[:, -1, :], axis=-1).item())

            verify_logits = rt.forward_ar(
                mx.array([[second]]),
                cache=cache,
                return_hidden=False,
            )
            mx.eval(verify_logits)
            target_third = int(mx.argmax(verify_logits[:, -1, :], axis=-1).item())

            dt = time.perf_counter() - started
            elapsed_s += dt
            match = draft_third == target_third
            matches += int(match)
            rows.append(
                {
                    "prompt_id": case.id,
                    "first": first,
                    "second": second,
                    "draft_third": draft_third,
                    "target_third": target_third,
                    "match": match,
                    "elapsed_s": round(dt, 4),
                }
            )

        results.append(
            {
                "hidden_variant": hidden_variant,
                "concat_order": concat_order,
                "matches": matches,
                "total": len(prompts),
                "agreement": round(matches / len(prompts), 4) if prompts else 0.0,
                "elapsed_s": round(elapsed_s, 4),
                "rows": rows,
            }
        )

    return {
        "model_path": str(model_path),
        "prompt_suite": str(prompt_suite),
        "max_prompt_tokens": max_prompt_tokens,
        "chat_template": chat_template,
        "enable_thinking": enable_thinking,
        "probe_elapsed_s": round(time.perf_counter() - all_started, 4),
        "results": results,
    }
