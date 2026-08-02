"""Fit Engine unit tests — real arithmetic + injectable memory."""

from __future__ import annotations

from pathlib import Path

import pytest

from optiq.lab import fit_engine
from optiq.lab.config import ensure_lab_dirs


def test_estimate_kv_scales_with_ctx_and_bits():
    low = fit_engine.estimate_kv_gb(4096, 4)
    high = fit_engine.estimate_kv_gb(32768, 8)
    assert high > low
    assert low > 0


def test_predict_comfortable_when_plenty_of_ram(lab_home, tmp_path):
    ensure_lab_dirs()
    # 8 GB weights on a 64 GB free machine
    weights = tmp_path / "m"
    weights.mkdir()
    # skip disk: pass explicit weights
    r = fit_engine.predict(
        path=None,
        weights_gb=8.0,
        ctx=8192,
        kv_bits=8,
        free_ram_gb=40.0,
        total_ram_gb=64.0,
    )
    assert r.verdict == "comfortable"
    assert r.blocks_load is False
    assert r.weights_gb == 8.0
    assert "Capability" not in r.detail


def test_predict_will_not_fit_when_over_budget(lab_home):
    ensure_lab_dirs()
    r = fit_engine.predict(
        weights_gb=30.0,
        ctx=32768,
        kv_bits=8,
        free_ram_gb=8.0,
        total_ram_gb=16.0,
    )
    assert r.verdict in ("will_not_fit", "hard_fail")
    assert r.blocks_load is True


def test_predict_hard_fail_mtl_prior(lab_home):
    ensure_lab_dirs()
    r = fit_engine.predict(
        weights_gb=4.0,
        ctx=65536,
        kv_bits=4,
        free_ram_gb=48.0,
        total_ram_gb=64.0,
    )
    assert r.verdict == "hard_fail"
    assert r.blocks_load is True


def test_weights_from_safetensors_disk(lab_home, tmp_path):
    ensure_lab_dirs()
    model = tmp_path / "model"
    model.mkdir()
    blob = model / "w.safetensors"
    blob.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MiB
    gb, src = fit_engine.weights_gb_for_path(model)
    assert src == "disk_safetensors"
    assert gb > 0
    assert gb < 0.01


def test_calibration_persists(lab_home):
    ensure_lab_dirs()
    data = fit_engine.run_calibration_snapshot()
    assert data["total_ram_gb"] > 0
    assert fit_engine.calibration_path().is_file()
    cal = fit_engine.load_calibration()
    assert cal.get("reserved_gb")
    r = fit_engine.predict(weights_gb=1.0, free_ram_gb=20.0, total_ram_gb=32.0)
    assert r.calibrated is True
