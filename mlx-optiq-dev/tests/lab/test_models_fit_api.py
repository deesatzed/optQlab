"""API tests for Fit predict and Fit-gated load (critical gaps G1/G7)."""

from __future__ import annotations

from optiq.lab.app import create_app
from optiq.lab.config import ensure_lab_dirs


def _client(lab_home):
    ensure_lab_dirs()
    app = create_app(secret_key=b"x" * 32)
    app.config["TESTING"] = True
    app.config["OPTIQ_TEST_AUTH_BYPASS"] = True
    return app.test_client()


def test_fit_predict_api(lab_home):
    c = _client(lab_home)
    r = c.post(
        "/api/fit/predict",
        json={"weights_gb": 4.0, "ctx": 8192, "kv_bits": 8},
    )
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert "fit" in data
    assert data["fit"]["verdict"] in (
        "comfortable", "degraded", "will_not_fit", "hard_fail",
    )
    assert "Capability" not in data["fit"]["detail"]


def test_fit_blocks_load_returns_409(lab_home):
    c = _client(lab_home)
    r = c.post(
        "/api/models/load",
        json={
            "model": "/nonexistent/model-path-for-test",
            "weights_gb": 50.0,
            "ctx": 32768,
            "kv_bits": 8,
        },
    )
    # Either 409 (blocked) or 500 if path handling differs — with huge weights must block
    data = r.get_json()
    assert r.status_code == 409
    assert data["ok"] is False
    assert data.get("error") == "fit_blocks_load"
    assert data["fit"]["blocks_load"] is True


def test_load_without_supervisor_when_fits(lab_home):
    c = _client(lab_home)
    r = c.post(
        "/api/models/load",
        json={
            "model": "/tmp/tiny-fit-model",
            "weights_gb": 1.0,
            "ctx": 2048,
            "kv_bits": 8,
        },
    )
    data = r.get_json()
    assert r.status_code == 200
    assert data["ok"] is True
    assert data["model"] == "/tmp/tiny-fit-model"


def test_models_page_renders(lab_home):
    c = _client(lab_home)
    r = c.get("/models")
    assert r.status_code == 200
    assert b"Load with Fit" in r.data


def test_machine_api(lab_home):
    c = _client(lab_home)
    r = c.get("/api/machine")
    assert r.status_code == 200
    data = r.get_json()
    assert "memory" in data
    assert data["memory"]["total_ram_gb"] > 0
    assert "ports" in data
