"""WP-3 measurement: BYO score, store, compare, promote, consequences API."""

from __future__ import annotations

import json
from pathlib import Path

from optiq.lab import eval_service, spine
from optiq.lab.app import create_app
from optiq.lab.config import ensure_lab_dirs


def _client(lab_home):
    ensure_lab_dirs()
    app = create_app(secret_key=b"z" * 32)
    app.config["TESTING"] = True
    app.config["OPTIQ_TEST_AUTH_BYPASS"] = True
    return app.test_client()


def test_byo_score_pair_real_match():
    assert eval_service.score_pair("4", "4") is True
    assert eval_service.score_pair("4", "The answer is 4.") is True
    assert eval_service.score_pair("4", "5") is False


def test_score_byo_pairs_and_store(lab_home):
    ensure_lab_dirs()
    scores = eval_service.score_byo_pairs([
        {"id": "1", "prompt": "2+2", "expected": "4", "actual": "4"},
        {"id": "2", "prompt": "3+3", "expected": "6", "actual": "seven"},
    ])
    assert scores["n_total"] == 2
    assert scores["n_correct"] == 1
    assert scores["accuracy_pct"] == 50.0
    assert scores["capability_score"] == 50.0

    bid = spine.register_build(name="t", path="/tmp/wp3-model", weights_gb=1.0)
    eid = eval_service.store_eval_result(
        build_id=bid,
        model_path="/tmp/wp3-model",
        suite="byo",
        scores=scores,
    )
    got = eval_service.get_eval(eid)
    assert got is not None
    assert got["scores"]["capability_score"] == 50.0


def test_promote_gate_blocks_regression(lab_home):
    ensure_lab_dirs()
    bid = spine.register_build(name="t2", path="/tmp/wp3-b", weights_gb=1.0)
    a = eval_service.store_eval_result(
        build_id=bid, model_path="/tmp/wp3-b", suite="byo",
        scores={"capability_score": 90.0, "components": {"BYO": 90.0}},
    )
    b = eval_service.store_eval_result(
        build_id=bid, model_path="/tmp/wp3-b", suite="byo",
        scores={"capability_score": 80.0, "components": {"BYO": 80.0}},
    )
    gate = eval_service.promote_allowed(a, b)
    assert gate["allowed"] is False
    assert gate["overall_delta"] == -10.0

    gate2 = eval_service.promote_allowed(b, a)
    assert gate2["allowed"] is True


def test_parse_cli_eval_json(lab_home, tmp_path):
    p = tmp_path / "out.json"
    p.write_text(json.dumps({
        "model_path": "/m",
        "task": "gsm8k",
        "score_pct": 88.5,
    }))
    scores = eval_service.parse_cli_eval_json(p)
    assert scores["capability_score"] == 88.5
    assert "GSM8K" in scores["components"] or "gsm8k" in str(scores["components"]).lower()


def test_eval_set_import_api(lab_home):
    c = _client(lab_home)
    r = c.post("/api/eval/sets", json={
        "name": "tiny",
        "jsonl": '{"prompt":"hi","expected":"hello"}\n{"prompt":"2+2","expected":"4"}\n',
    })
    assert r.status_code == 201
    data = r.get_json()
    assert data["ok"] is True
    sid = data["id"]
    g = c.get(f"/api/eval/sets/{sid}")
    assert g.status_code == 200
    assert len(g.get_json()["items"]) == 2


def test_promote_api(lab_home):
    ensure_lab_dirs()
    bid = spine.register_build(name="t3", path="/tmp/wp3-c", weights_gb=1.0)
    a = eval_service.store_eval_result(
        build_id=bid, model_path="/tmp/wp3-c", suite="x",
        scores={"capability_score": 70, "components": {"MMLU": 70}},
    )
    b = eval_service.store_eval_result(
        build_id=bid, model_path="/tmp/wp3-c", suite="x",
        scores={"capability_score": 75, "components": {"MMLU": 75}},
    )
    c = _client(lab_home)
    r = c.post("/api/eval/promote", json={"baseline": a, "candidate": b})
    assert r.status_code == 200
    assert r.get_json()["allowed"] is True


def test_eval_page_and_consequences(lab_home):
    c = _client(lab_home)
    page = c.get("/eval")
    assert page.status_code == 200
    assert b"Promote" in page.data or b"promote" in page.data
    r = c.post("/api/fit/consequences", json={"weights_gb": 4, "ctx": 8192, "kv_bits": 8})
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True
    assert d["variants"]
    assert any(v.get("estimate") for v in d["variants"])


def test_run_eval_submits_job(lab_home):
    c = _client(lab_home)
    # Create set for byo so validation passes; job may fail later without model — we only check submit
    sid = eval_service.save_eval_set("j", [{"prompt": "q", "expected": "a"}])
    r = c.post("/api/eval/run", json={
        "model_path": "/nonexistent/model",
        "suite": "byo",
        "eval_set_id": sid,
    })
    assert r.status_code == 200
    assert r.get_json()["job_id"].startswith("job_")
