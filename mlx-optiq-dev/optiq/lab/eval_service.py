"""Eval storage, BYO scoring, compare, and promote gate (WP-3 / G2).

All scores stored here are either:
  * produced by ``optiq eval`` (CLI JSON), or
  * BYO exact/substring match against user-provided expected strings.

Never invents Capability Scores.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from . import db, events, spine
from .config import ensure_lab_dirs


def eval_sets_dir() -> Path:
    d = ensure_lab_dirs().root / "eval_sets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_eval_id() -> str:
    return f"eval_{uuid.uuid4().hex[:12]}"


def new_set_id() -> str:
    return f"evset_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# BYO prompt sets
# ---------------------------------------------------------------------------


def save_eval_set(name: str, items: list[dict], *, set_id: str | None = None) -> str:
    """Persist a BYO set. Each item: ``{id?, prompt, expected}``."""
    if not name or not isinstance(name, str):
        raise ValueError("name required")
    cleaned: list[dict] = []
    for i, raw in enumerate(items or []):
        if not isinstance(raw, dict):
            continue
        prompt = (raw.get("prompt") or raw.get("question") or "").strip()
        expected = (raw.get("expected") or raw.get("answer") or "").strip()
        if not prompt:
            continue
        cleaned.append({
            "id": raw.get("id") or f"p{i+1}",
            "prompt": prompt,
            "expected": expected,
        })
    if not cleaned:
        raise ValueError("eval set needs at least one item with a prompt")

    sid = set_id or new_set_id()
    path = eval_sets_dir() / f"{sid}.json"
    path.write_text(json.dumps({
        "id": sid,
        "name": name.strip(),
        "items": cleaned,
        "created_at": time.time(),
    }, indent=2))
    events.append(
        type="eval_set.created",
        entity_type="eval_set",
        entity_id=sid,
        payload={"name": name, "n": len(cleaned)},
    )
    return sid


def list_eval_sets() -> list[dict]:
    out = []
    for p in sorted(eval_sets_dir().glob("evset_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text())
            out.append({
                "id": data["id"],
                "name": data.get("name") or data["id"],
                "n_items": len(data.get("items") or []),
                "created_at": data.get("created_at"),
            })
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return out


def load_eval_set(set_id: str) -> dict | None:
    path = eval_sets_dir() / f"{set_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def normalize_answer(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def score_pair(expected: str, actual: str) -> bool:
    """Real match: exact after normalize, or expected contained in actual."""
    e = normalize_answer(expected)
    a = normalize_answer(actual)
    if not e:
        return False
    if e == a:
        return True
    # Allow expected as substring (models often add punctuation)
    if e in a:
        return True
    return False


def score_byo_pairs(pairs: list[dict]) -> dict[str, Any]:
    """Score list of ``{expected, actual, id?, prompt?}``.

    Pure function — no model. Used by job after generation and by tests.
    """
    results = []
    n_ok = 0
    for p in pairs:
        ok = score_pair(str(p.get("expected") or ""), str(p.get("actual") or ""))
        if ok:
            n_ok += 1
        results.append({
            "id": p.get("id"),
            "prompt": p.get("prompt"),
            "expected": p.get("expected"),
            "actual": p.get("actual"),
            "correct": ok,
        })
    n = len(results)
    pct = (100.0 * n_ok / n) if n else 0.0
    return {
        "suite": "byo",
        "n_total": n,
        "n_correct": n_ok,
        "accuracy_pct": round(pct, 2),
        "capability_score": round(pct, 2),  # single-metric set: mean = accuracy
        "components": {"BYO": round(pct, 2)},
        "items": results,
    }


# ---------------------------------------------------------------------------
# Stored eval results
# ---------------------------------------------------------------------------


def store_eval_result(
    *,
    build_id: str | None,
    model_path: str,
    suite: str,
    scores: dict[str, Any],
    job_id: str | None = None,
    eval_set_id: str | None = None,
    metadata: dict | None = None,
) -> str:
    eid = new_eval_id()
    meta = dict(metadata or {})
    meta["model_path"] = model_path
    if job_id:
        meta["job_id"] = job_id
    if eval_set_id:
        meta["eval_set_id"] = eval_set_id
    scores_out = dict(scores)
    scores_out.setdefault("suite", suite)

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO evals (id, build_id, suite, scores_json, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                eid,
                build_id,
                suite,
                json.dumps(scores_out),
                json.dumps(meta),
            ),
        )
    events.append(
        type="eval.stored",
        entity_type="eval",
        entity_id=eid,
        payload={"suite": suite, "build_id": build_id, "score": scores_out.get("capability_score")},
    )
    return eid


def list_evals(*, build_id: str | None = None, limit: int = 50) -> list[dict]:
    conn = db.get_conn()
    if build_id:
        rows = conn.execute(
            "SELECT * FROM evals WHERE build_id = ? ORDER BY created_at DESC LIMIT ?",
            (build_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM evals ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["scores"] = json.loads(d.pop("scores_json") or "{}")
        d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
        out.append(d)
    return out


def get_eval(eval_id: str) -> dict | None:
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM evals WHERE id = ?", (eval_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["scores"] = json.loads(d.pop("scores_json") or "{}")
    d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
    return d


def compare_evals(eval_a_id: str, eval_b_id: str) -> dict[str, Any]:
    a = get_eval(eval_a_id)
    b = get_eval(eval_b_id)
    if not a or not b:
        raise KeyError("eval not found")
    sa = a["scores"]
    sb = b["scores"]
    comps_a = sa.get("components") or {}
    comps_b = sb.get("components") or {}
    keys = sorted(set(comps_a) | set(comps_b))
    rows = []
    for k in keys:
        va = comps_a.get(k)
        vb = comps_b.get(k)
        delta = None
        if va is not None and vb is not None:
            delta = round(float(vb) - float(va), 3)
        rows.append({"name": k, "a": va, "b": vb, "delta": delta})
    score_a = sa.get("capability_score")
    score_b = sb.get("capability_score")
    overall_delta = None
    if score_a is not None and score_b is not None:
        overall_delta = round(float(score_b) - float(score_a), 3)
    return {
        "a": a,
        "b": b,
        "rows": rows,
        "score_a": score_a,
        "score_b": score_b,
        "overall_delta": overall_delta,
    }


def promote_allowed(
    baseline_eval_id: str,
    candidate_eval_id: str,
    *,
    min_delta: float = 0.0,
) -> dict[str, Any]:
    """Regression gate: promote candidate only if overall score does not fall.

    ``min_delta`` default 0.0 means candidate must be ≥ baseline.
    """
    cmp_ = compare_evals(baseline_eval_id, candidate_eval_id)
    delta = cmp_["overall_delta"]
    if delta is None:
        return {
            "allowed": False,
            "reason": "missing capability_score on one or both evals",
            "compare": cmp_,
        }
    allowed = delta >= min_delta
    return {
        "allowed": allowed,
        "reason": (
            f"candidate Δcapability = {delta:+.2f} "
            f"({'pass' if allowed else 'blocked'}; min_delta={min_delta})"
        ),
        "overall_delta": delta,
        "compare": cmp_,
    }


def parse_cli_eval_json(path: str | Path) -> dict[str, Any]:
    """Normalize ``optiq eval --output-json`` payloads into scores dict."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("eval JSON must be an object")

    # Single-task form: {task, score_pct, model_path}
    if "score_pct" in data and "components" not in data:
        task = data.get("task") or "task"
        pct = float(data["score_pct"])
        return {
            "suite": task,
            "capability_score": pct,
            "components": {str(task).upper(): pct},
            "raw": data,
        }

    # Full suite form from _run_all
    components: dict[str, float] = {}
    for key, label in (
        ("mmlu", "MMLU"),
        ("gsm8k", "GSM8K"),
        ("ifeval", "IFEval"),
        ("bfcl", "BFCL"),
        ("humaneval", "HumanEval"),
        ("hashhop", "HashHop"),
        ("gsm8k_50", "GSM8K-50"),
        ("kl", "KL"),
    ):
        if key in data and data[key] is not None:
            try:
                components[label] = float(data[key])
            except (TypeError, ValueError):
                pass
    # Nested metrics
    if "metrics" in data and isinstance(data["metrics"], dict):
        for k, v in data["metrics"].items():
            if isinstance(v, (int, float)):
                components[str(k)] = float(v)

    cap = data.get("capability_score") or data.get("Capability_Score") or data.get("score")
    if cap is None and components:
        # Mean of components that look like percentages (0-100)
        pcts = [v for v in components.values() if 0 <= v <= 100]
        cap = sum(pcts) / len(pcts) if pcts else None
    if cap is not None:
        cap = float(cap)

    return {
        "suite": data.get("suite") or data.get("task") or "suite",
        "capability_score": cap,
        "components": components,
        "disk_gb": data.get("disk_gb"),
        "raw": data,
    }


def resolve_build_id_for_path(model_path: str) -> str | None:
    """Find spine build id by path, or register a lightweight row."""
    conn = db.get_conn()
    row = conn.execute(
        "SELECT id FROM builds WHERE path = ? LIMIT 1", (model_path,)
    ).fetchone()
    if row:
        return row["id"]
    # Register so evals attach to something real
    try:
        return spine.register_build(
            name=Path(model_path).name or model_path,
            path=model_path,
        )
    except Exception:
        return None
