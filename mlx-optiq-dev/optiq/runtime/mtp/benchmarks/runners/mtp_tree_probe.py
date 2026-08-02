"""Native-MTP tree coverage probes.

This estimates whether branching over native MTP candidates can recover target
paths that the linear recursive chain misses. It does not run target verify and
does not claim speed; it answers whether a tree verifier is worth building.
"""

from __future__ import annotations

import heapq
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from optiq.runtime.mtp.benchmarks.runners.mtp_chain_probe import (
    _input_pair,
    _parse_int_csv,
    _target_trace,
)
from optiq.runtime.mtp.benchmarks.schema import encode_prompt_case, load_prompt_suite
from optiq.runtime.mtp.mtp_patch import MTPContract
from optiq.runtime.mtp.runtime import load


@dataclass
class _Node:
    token: int
    depth: int
    parent: int
    score: float
    hidden: mx.array


def _path_to_root(nodes: list[_Node], node_id: int) -> list[_Node]:
    path: list[_Node] = []
    while node_id >= 0:
        node = nodes[node_id]
        path.append(node)
        node_id = node.parent
    path.reverse()
    return path


def _replay_persistent_path(
    rt,
    *,
    nodes: list[_Node],
    parent_id: int,
    root_hidden: mx.array,
    root_token: int,
    mtp_hidden_variant: str,
):
    """Rebuild the persistent MTP cache for one branch path.

    This is deliberately an offline probe helper, not a runtime implementation.
    It lets tree coverage use the same within-cycle persistent MTP contract as
    live D2/D3 generation without needing unsafe mutable-cache copies.
    """
    cache = rt.make_mtp_cache()
    hidden = root_hidden
    token = int(root_token)
    for node in _path_to_root(nodes, parent_id):
        _, hidden_next = rt.draft_mtp(
            hidden,
            mx.array([[token]]),
            mtp_cache=cache,
            return_hidden=True,
            mtp_hidden_variant=mtp_hidden_variant,
        )
        mx.eval(hidden_next)
        hidden = hidden_next[:, -1:, :]
        token = int(node.token)
    return cache, hidden, token


def _top_candidates(logits: mx.array, count: int) -> list[tuple[int, float]]:
    flat = logits.astype(mx.float32).reshape(-1)
    top_indices = mx.argpartition(-flat, kth=count - 1, axis=-1)[:count]
    top_values = flat[top_indices]
    order = mx.argsort(-top_values)
    ranked_indices = top_indices[order]
    ranked_values = top_values[order]
    norm = mx.logsumexp(flat)
    log_probs = ranked_values - norm
    mx.eval(ranked_indices, log_probs)
    ids = np.asarray(ranked_indices, dtype=np.int64).reshape(-1)
    scores = np.asarray(log_probs, dtype=np.float32).reshape(-1)
    return [(int(tok), float(score)) for tok, score in zip(ids, scores)]


def _build_tree(
    rt,
    *,
    root_hidden: mx.array,
    root_token: int,
    depth: int,
    budget: int,
    branch_factor: int,
    mtp_hidden_variant: str,
    mtp_cache_policy: str,
) -> list[_Node]:
    if mtp_cache_policy not in {"fresh", "persistent_path"}:
        raise ValueError("mtp_cache_policy must be 'fresh' or 'persistent_path'")

    nodes: list[_Node] = []
    queue: list[tuple[float, int, int, mx.array, int]] = []
    counter = 0
    # Heap payload: negative cumulative score, tiebreaker, parent id, hidden, token.
    heapq.heappush(queue, (0.0, counter, -1, root_hidden, int(root_token)))

    while queue and len(nodes) < budget:
        neg_score, _, parent_id, hidden, token = heapq.heappop(queue)
        parent_score = -neg_score
        parent_depth = 0 if parent_id < 0 else nodes[parent_id].depth
        if parent_depth >= depth:
            continue

        if mtp_cache_policy == "persistent_path":
            mtp_cache, hidden, token = _replay_persistent_path(
                rt,
                nodes=nodes,
                parent_id=parent_id,
                root_hidden=root_hidden,
                root_token=root_token,
                mtp_hidden_variant=mtp_hidden_variant,
            )
        else:
            mtp_cache = rt.make_mtp_cache()

        logits, next_hidden = rt.draft_mtp(
            hidden,
            mx.array([[token]]),
            mtp_cache=mtp_cache,
            return_hidden=True,
            mtp_hidden_variant=mtp_hidden_variant,
        )
        mx.eval(logits, next_hidden)
        for child_token, child_logp in _top_candidates(logits[:, -1, :][0], branch_factor):
            if len(nodes) >= budget:
                break
            node = _Node(
                token=child_token,
                depth=parent_depth + 1,
                parent=parent_id,
                score=parent_score + child_logp,
                hidden=next_hidden[:, -1:, :],
            )
            node_id = len(nodes)
            nodes.append(node)
            if node.depth < depth:
                counter += 1
                heapq.heappush(
                    queue,
                    (-node.score, counter, node_id, node.hidden, node.token),
                )
    return nodes


def _accepted_prefix(
    nodes: list[_Node],
    target_tokens: list[int],
    *,
    anchor_offset: int,
    depth: int,
) -> int:
    prefix = 0
    parent = -1
    for depth_index in range(depth):
        target = int(target_tokens[anchor_offset + depth_index + 1])
        match_id = None
        for idx, node in enumerate(nodes):
            if node.parent == parent and node.token == target:
                match_id = idx
                break
        if match_id is None:
            break
        prefix += 1
        parent = match_id
    return prefix


def run_mtp_tree_probe(
    model_path: Path | str,
    prompt_suite: Path | str,
    *,
    depth: int = 5,
    budgets: str = "1,2,4,8,16",
    branch_factor: int = 4,
    limit: int | None = None,
    windows: int = 32,
    stride: int = 1,
    max_prompt_tokens: int = 256,
    chat_template: bool = True,
    enable_thinking: bool | None = None,
    mtp_quant_bits: int | None = None,
    mtp_quant_group_size: int = 64,
    mtp_quant_mode: str = "affine",
    base_hidden_variant: str = "post_norm",
    mtp_hidden_variant: str = "pre_norm",
    mtp_cache_policy: str = "fresh",
    anchor: str = "prompt_boundary",
) -> dict[str, Any]:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if branch_factor < 1:
        raise ValueError("branch_factor must be >= 1")
    if windows < 1:
        raise ValueError("windows must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if anchor not in {"prompt_boundary", "after_one_target"}:
        raise ValueError("anchor must be 'prompt_boundary' or 'after_one_target'")
    if mtp_cache_policy not in {"fresh", "persistent_path"}:
        raise ValueError("mtp_cache_policy must be 'fresh' or 'persistent_path'")

    selected_budgets = _parse_int_csv(budgets, name="budgets")
    if not selected_budgets:
        raise ValueError("at least one budget is required")
    max_budget = max(selected_budgets)
    target_token_count = ((windows - 1) * stride) + depth + 3
    anchor_base = 0 if anchor == "prompt_boundary" else 1

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

    started_all = time.perf_counter()
    budget_rows = {
        budget: {
            "budget": budget,
            "prefixes": [],
            "prefix_histogram": {str(i): 0 for i in range(depth + 1)},
            "rows": [],
        }
        for budget in selected_budgets
    }
    prompt_meta = []

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
        trace = _target_trace(
            rt,
            ids,
            target_token_count=target_token_count,
            base_hidden_variant=base_hidden_variant,
        )
        for window_index in range(windows):
            window_start = window_index * stride
            anchor_offset = window_start + anchor_base
            root_hidden, root_token = _input_pair(trace, anchor_offset)
            nodes = _build_tree(
                rt,
                root_hidden=root_hidden,
                root_token=root_token,
                depth=depth,
                budget=max_budget,
                branch_factor=branch_factor,
                mtp_hidden_variant=mtp_hidden_variant,
                mtp_cache_policy=mtp_cache_policy,
            )
            for budget in selected_budgets:
                prefix = _accepted_prefix(
                    nodes[:budget],
                    trace["target_tokens"],
                    anchor_offset=anchor_offset,
                    depth=depth,
                )
                row = budget_rows[budget]
                row["prefixes"].append(prefix)
                row["prefix_histogram"][str(prefix)] += 1
                row["rows"].append(
                    {
                        "prompt_id": case.id,
                        "window_index": window_index,
                        "window_start": window_start,
                        "prefix": prefix,
                    }
                )

    results = []
    for budget in selected_budgets:
        row = budget_rows[budget]
        prefixes = row["prefixes"]
        row["mean_prefix"] = sum(prefixes) / len(prefixes) if prefixes else 0.0
        row["full_depth_rate"] = (
            sum(1 for prefix in prefixes if prefix >= depth) / len(prefixes)
            if prefixes
            else 0.0
        )
        results.append(row)

    return {
        "model_path": str(model_path),
        "prompt_suite": str(prompt_suite),
        "depth": depth,
        "budgets": selected_budgets,
        "branch_factor": branch_factor,
        "limit": limit,
        "windows": windows,
        "stride": stride,
        "target_token_count": target_token_count,
        "max_prompt_tokens": max_prompt_tokens,
        "chat_template": chat_template,
        "enable_thinking": enable_thinking,
        "mtp_quant_bits": mtp_quant_bits,
        "mtp_quant_group_size": mtp_quant_group_size,
        "mtp_quant_mode": mtp_quant_mode,
        "base_hidden_variant": base_hidden_variant,
        "mtp_hidden_variant": mtp_hidden_variant,
        "mtp_cache_policy": mtp_cache_policy,
        "anchor": anchor,
        "prompt_meta": prompt_meta,
        "elapsed_s": time.perf_counter() - started_all,
        "results": results,
    }


def write_mtp_tree_probe(path: Path | str, result: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
