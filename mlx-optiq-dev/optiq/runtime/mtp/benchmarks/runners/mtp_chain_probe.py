"""Recursive native-MTP agreement probes.

This runner is deliberately not a speed benchmark. It isolates whether deeper
MTP collapse comes from the head itself, the recursive MTP hidden state, or the
MTP cache/history schedule.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from optiq.runtime.mtp.benchmarks.schema import encode_prompt_case, load_prompt_suite
from optiq.runtime.mtp.mtp_patch import MTPContract
from optiq.runtime.mtp.runtime import load

ANCHORS = {"prompt_boundary", "after_one_target"}
BASE_HIDDEN_VARIANTS = {"post_norm", "pre_norm"}
MTP_HIDDEN_VARIANTS = {"post_norm", "pre_norm", "fc", "embedding", "prev"}
CACHE_POLICIES = {"fresh", "persistent"}
CONCAT_ORDERS = {"embedding_hidden", "hidden_embedding"}
POSITION_MODES = {"local", "absolute"}
HISTORY_MODES = {
    "recursive",
    "target_forced",
    "target_token_recursive_hidden",
}


def _parse_csv(value: str | None, allowed: set[str], *, name: str) -> list[str]:
    if not value:
        return sorted(allowed)
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(item for item in set(parsed) - allowed if not item.startswith("mix:"))
    if unknown:
        raise ValueError(f"unknown {name}: {unknown}")
    return parsed


def _parse_int_csv(value: str | None, *, name: str) -> list[int]:
    if not value:
        return []
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(item < 1 for item in parsed):
        raise ValueError(f"{name} values must be >= 1")
    return sorted(set(parsed))


def _argmax_token(logits: mx.array) -> int:
    mx.eval(logits)
    return int(mx.argmax(logits, axis=-1).item())


def _target_rank(logits: mx.array, target_token: int, *, max_rank: int) -> int | None:
    flat = logits.astype(mx.float32).reshape(-1)
    top_indices = mx.argpartition(-flat, kth=max_rank - 1, axis=-1)[:max_rank]
    top_values = flat[top_indices]
    order = mx.argsort(-top_values)
    ranked = top_indices[order]
    mx.eval(ranked)
    ranked_ids = np.asarray(ranked, dtype=np.int64).reshape(-1)
    found = np.where(ranked_ids == int(target_token))[0]
    if found.size == 0:
        return None
    return int(found[0]) + 1


def _prefill_prompt(rt, ids: list[int], *, base_hidden_variant: str):
    cache = rt.make_cache()
    if len(ids) > 1:
        logits = rt.forward_ar(
            mx.array([ids[:-1]]),
            cache=cache,
            return_hidden=False,
        )
        mx.eval(logits)
    logits, hidden = rt.forward_ar(
        mx.array([[ids[-1]]]),
        cache=cache,
        return_hidden=True,
        hidden_variant=base_hidden_variant,
    )
    mx.eval(logits, hidden)
    return cache, logits[:, -1, :], hidden[:, -1:, :]


def _target_trace(
    rt,
    ids: list[int],
    *,
    target_token_count: int,
    base_hidden_variant: str,
    include_logits: bool = False,
) -> dict[str, Any]:
    cache, logits, prompt_hidden = _prefill_prompt(
        rt,
        ids,
        base_hidden_variant=base_hidden_variant,
    )
    tokens: list[int] = []
    hiddens: list[mx.array] = []
    target_logits: list[mx.array] = []
    for _ in range(target_token_count):
        if include_logits:
            target_logits.append(logits)
        token = _argmax_token(logits[0])
        tokens.append(token)
        logits_next, hidden_next = rt.forward_ar(
            mx.array([[token]]),
            cache=cache,
            return_hidden=True,
            hidden_variant=base_hidden_variant,
        )
        mx.eval(logits_next, hidden_next)
        hiddens.append(hidden_next[:, -1:, :])
        logits = logits_next[:, -1, :]
    return {
        "prompt_len": len(ids),
        "prompt_hidden": prompt_hidden,
        "target_tokens": tokens,
        "target_hiddens": hiddens,
        "target_logits": target_logits,
    }


def _input_pair(trace: dict[str, Any], token_index: int) -> tuple[mx.array, int]:
    if token_index == 0:
        hidden = trace["prompt_hidden"]
    else:
        hidden = trace["target_hiddens"][token_index - 1]
    return hidden, int(trace["target_tokens"][token_index])


def _run_variant_on_trace(
    rt,
    trace: dict[str, Any],
    *,
    depth: int,
    anchor: str,
    mtp_hidden_variant: str,
    cache_policy: str,
    concat_order: str,
    mtp_position_mode: str,
    history_mode: str,
    window_start: int = 0,
    max_rank: int = 8,
) -> dict[str, Any]:
    anchor_offset = window_start + (0 if anchor == "prompt_boundary" else 1)
    hidden, next_token = _input_pair(trace, anchor_offset)
    mtp_cache = rt.make_mtp_cache() if cache_policy == "persistent" else None
    rows: list[dict[str, Any]] = []
    prefix = 0
    still_prefix = True

    for depth_index in range(depth):
        target_index = anchor_offset + depth_index + 1
        target_token = int(trace["target_tokens"][target_index])
        step_cache = mtp_cache if cache_policy == "persistent" else rt.make_mtp_cache()
        position_offset = None
        if mtp_position_mode == "absolute":
            position_offset = int(trace["prompt_len"]) + anchor_offset + depth_index
        draft_logits, draft_hidden = rt.draft_mtp(
            hidden,
            mx.array([[next_token]]),
            mtp_cache=step_cache,
            concat_order=concat_order,
            return_hidden=True,
            mtp_hidden_variant=mtp_hidden_variant,
            position_offset=position_offset,
        )
        mx.eval(draft_logits, draft_hidden)
        draft_token = _argmax_token(draft_logits[:, -1, :][0])
        target_rank = _target_rank(
            draft_logits[:, -1, :][0],
            target_token,
            max_rank=max_rank,
        )
        match = draft_token == target_token
        if still_prefix and match:
            prefix += 1
        else:
            still_prefix = False
        rows.append(
            {
                "depth": depth_index + 1,
                "input_token": int(next_token),
                "draft_token": draft_token,
                "target_token": target_token,
                "target_rank": target_rank,
                "match": match,
                "position_offset": position_offset,
            }
        )

        if history_mode == "target_forced":
            if depth_index + 1 < depth:
                hidden, next_token = _input_pair(trace, anchor_offset + depth_index + 1)
        elif history_mode == "target_token_recursive_hidden":
            hidden = draft_hidden[:, -1:, :]
            next_token = int(trace["target_tokens"][target_index])
        else:
            hidden = draft_hidden[:, -1:, :]
            next_token = draft_token

    return {"prefix": prefix, "rows": rows}


def run_mtp_chain_probe(
    model_path: Path | str,
    prompt_suite: Path | str,
    *,
    depth: int = 5,
    limit: int | None = None,
    max_prompt_tokens: int = 256,
    chat_template: bool = True,
    enable_thinking: bool | None = None,
    windows: int = 1,
    stride: int = 1,
    top_ranks: str | None = "1,2,4,8",
    mtp_quant_bits: int | None = None,
    mtp_quant_group_size: int = 64,
    mtp_quant_mode: str = "affine",
    base_hidden_variants: str | None = "post_norm",
    mtp_hidden_variants: str | None = "post_norm,pre_norm,fc",
    cache_policies: str | None = "fresh,persistent",
    concat_orders: str | None = "embedding_hidden",
    mtp_position_modes: str | None = "local",
    history_modes: str | None = "recursive,target_forced,target_token_recursive_hidden",
    anchors: str | None = "prompt_boundary,after_one_target",
) -> dict[str, Any]:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if windows < 1:
        raise ValueError("windows must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    selected_base = _parse_csv(base_hidden_variants, BASE_HIDDEN_VARIANTS, name="base hidden variants")
    selected_mtp = _parse_csv(mtp_hidden_variants, MTP_HIDDEN_VARIANTS, name="MTP hidden variants")
    selected_cache = _parse_csv(cache_policies, CACHE_POLICIES, name="cache policies")
    selected_concat = _parse_csv(concat_orders, CONCAT_ORDERS, name="concat orders")
    selected_position_modes = _parse_csv(mtp_position_modes, POSITION_MODES, name="MTP position modes")
    selected_history = _parse_csv(history_modes, HISTORY_MODES, name="history modes")
    selected_anchors = _parse_csv(anchors, ANCHORS, name="anchors")
    selected_top_ranks = _parse_int_csv(top_ranks, name="top ranks")
    max_rank = max(selected_top_ranks) if selected_top_ranks else 1

    rt = load(
        model_path,
        mtp=True,
        contract=MTPContract(
            mtp_quant_bits=mtp_quant_bits,
            mtp_quant_group_size=mtp_quant_group_size,
            mtp_quant_mode=mtp_quant_mode,
        ),
    )
    prompts = load_prompt_suite(prompt_suite)
    if limit is not None:
        prompts = prompts[:limit]

    all_started = time.perf_counter()
    traces: dict[tuple[str, str], dict[str, Any]] = {}
    prompt_meta = []
    target_token_count = ((windows - 1) * stride) + depth + 3
    for case in prompts:
        ids = encode_prompt_case(
            rt.tokenizer,
            case,
            chat_template=chat_template,
            enable_thinking=enable_thinking,
        )[-max_prompt_tokens:]
        prompt_meta.append(
            {
                "prompt_id": case.id,
                "category": case.category,
                "prompt_tokens": len(ids),
            }
        )
        for base_hidden in selected_base:
            traces[(case.id, base_hidden)] = _target_trace(
                rt,
                ids,
                target_token_count=target_token_count,
                base_hidden_variant=base_hidden,
            )

    variants = []
    for base_hidden in selected_base:
        for mtp_hidden in selected_mtp:
            for cache_policy in selected_cache:
                for concat_order in selected_concat:
                    for mtp_position_mode in selected_position_modes:
                        for history_mode in selected_history:
                            for anchor in selected_anchors:
                                matches_by_depth = [0 for _ in range(depth)]
                                totals_by_depth = [0 for _ in range(depth)]
                                topk_hits_by_depth = {
                                    str(rank): [0 for _ in range(depth)]
                                    for rank in selected_top_ranks
                                }
                                prefixes: list[int] = []
                                rows = []
                                started = time.perf_counter()
                                for case in prompts:
                                    trace = traces[(case.id, base_hidden)]
                                    for window_index in range(windows):
                                        window_start = window_index * stride
                                        result = _run_variant_on_trace(
                                            rt,
                                            trace,
                                            depth=depth,
                                            anchor=anchor,
                                            mtp_hidden_variant=mtp_hidden,
                                            cache_policy=cache_policy,
                                            concat_order=concat_order,
                                            mtp_position_mode=mtp_position_mode,
                                            history_mode=history_mode,
                                            window_start=window_start,
                                            max_rank=max_rank,
                                        )
                                        prefixes.append(int(result["prefix"]))
                                        for row in result["rows"]:
                                            idx = int(row["depth"]) - 1
                                            totals_by_depth[idx] += 1
                                            matches_by_depth[idx] += int(row["match"])
                                            target_rank = row.get("target_rank")
                                            for rank in selected_top_ranks:
                                                if target_rank is not None and int(target_rank) <= rank:
                                                    topk_hits_by_depth[str(rank)][idx] += 1
                                        rows.append(
                                            {
                                                "prompt_id": case.id,
                                                "window_index": window_index,
                                                "window_start": window_start,
                                                "prefix": int(result["prefix"]),
                                                "drafts": result["rows"],
                                            }
                                        )
                                elapsed = time.perf_counter() - started
                                agreement_by_depth = [
                                    (matches / total if total else None)
                                    for matches, total in zip(matches_by_depth, totals_by_depth)
                                ]
                                topk_rates_by_depth = {
                                    rank: [
                                        (hits / total if total else None)
                                        for hits, total in zip(hits_by_depth, totals_by_depth)
                                    ]
                                    for rank, hits_by_depth in topk_hits_by_depth.items()
                                }
                                variants.append(
                                    {
                                        "base_hidden_variant": base_hidden,
                                        "mtp_hidden_variant": mtp_hidden,
                                        "cache_policy": cache_policy,
                                        "concat_order": concat_order,
                                        "mtp_position_mode": mtp_position_mode,
                                        "history_mode": history_mode,
                                        "anchor": anchor,
                                        "matches_by_depth": matches_by_depth,
                                        "totals_by_depth": totals_by_depth,
                                        "agreement_by_depth": agreement_by_depth,
                                        "topk_hits_by_depth": topk_hits_by_depth,
                                        "topk_rates_by_depth": topk_rates_by_depth,
                                        "mean_prefix": sum(prefixes) / len(prefixes) if prefixes else 0.0,
                                        "prefixes": prefixes,
                                        "elapsed_s": elapsed,
                                        "rows": rows,
                                    }
                                )

    variants.sort(
        key=lambda item: (
            item["mean_prefix"],
            sum(x or 0.0 for x in item["agreement_by_depth"]),
        ),
        reverse=True,
    )
    return {
        "model_path": str(model_path),
        "prompt_suite": str(prompt_suite),
        "depth": depth,
        "limit": limit,
        "max_prompt_tokens": max_prompt_tokens,
        "windows": windows,
        "stride": stride,
        "target_token_count": target_token_count,
        "top_ranks": selected_top_ranks,
        "mtp_quant_bits": rt.contract.mtp_quant_bits,
        "mtp_quant_group_size": rt.contract.mtp_quant_group_size,
        "mtp_quant_mode": rt.contract.mtp_quant_mode,
        "mtp_quant_policy": rt.contract.mtp_quant_policy,
        "mtp_position_modes": selected_position_modes,
        "chat_template": chat_template,
        "enable_thinking": enable_thinking,
        "prompt_meta": prompt_meta,
        "elapsed_s": time.perf_counter() - all_started,
        "variants": variants,
    }


def write_mtp_chain_probe(path: Path | str, result: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
