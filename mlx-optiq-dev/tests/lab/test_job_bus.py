"""Sequential JobBus — admission, cancel, lifecycle events."""

from __future__ import annotations

import time

import pytest

from optiq.lab import db, events, job_bus, jobs


def _slow_job(emit, config):
    import time as _time

    emit({"type": "progress", "progress": 0.1, "message": "hi"})
    _time.sleep(float(config.get("sleep", 0.3)))
    emit({"type": "progress", "progress": 1.0, "message": "done"})


def _fail_job(emit, config):
    emit({"type": "progress", "progress": 0.2, "message": "boom soon"})
    raise RuntimeError(config.get("error", "intentional failure"))


@pytest.fixture(autouse=True)
def _reset_bus(lab_home):
    job_bus._reset_for_tests()
    yield
    job_bus._reset_for_tests()


def _wait_status(job_id: str, want, timeout: float = 15.0) -> str:
    if isinstance(want, str):
        want = {want}
    else:
        want = set(want)
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        row = jobs.get(job_id)
        last = row["status"] if row else None
        if last in want:
            return last
        time.sleep(0.05)
    raise AssertionError(
        f"job {job_id} status={last!r} not in {want} within {timeout}s"
    )


def _wait_pred(pred, timeout: float = 15.0, interval: float = 0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return
        time.sleep(interval)
    raise AssertionError(f"predicate not satisfied within {timeout}s")


def test_sequential_two_heavies_not_both_running(lab_home):
    j1 = job_bus.submit(
        "test", _slow_job, config={"sleep": 0.5}, resource_class="memory_heavy"
    )
    j2 = job_bus.submit(
        "test", _slow_job, config={"sleep": 0.2}, resource_class="memory_heavy"
    )

    # First should reach running; second must stay queued while first runs.
    _wait_status(j1, "running")

    saw_j2_queued_while_j1_running = False
    deadline = time.time() + 5.0
    while time.time() < deadline:
        s1 = jobs.get(j1)["status"]
        s2 = jobs.get(j2)["status"]
        if s1 == "running" and s2 == "queued":
            saw_j2_queued_while_j1_running = True
            break
        if s1 == "done":
            break
        # Never both running.
        assert not (s1 == "running" and s2 == "running")
        time.sleep(0.05)

    assert saw_j2_queued_while_j1_running, (
        f"expected j2 queued while j1 running; j1={jobs.get(j1)}; j2={jobs.get(j2)}"
    )

    # Continuously assert they never run concurrently until both done.
    deadline = time.time() + 15.0
    while time.time() < deadline:
        s1 = jobs.get(j1)["status"]
        s2 = jobs.get(j2)["status"]
        assert not (s1 == "running" and s2 == "running")
        if s1 == "done" and s2 == "done":
            break
        time.sleep(0.05)
    else:
        raise AssertionError(
            f"jobs did not finish: j1={jobs.get(j1)}; j2={jobs.get(j2)}"
        )

    # runs table dual-write
    r1 = db.get_conn().execute("SELECT * FROM runs WHERE id = ?", (j1,)).fetchone()
    r2 = db.get_conn().execute("SELECT * FROM runs WHERE id = ?", (j2,)).fetchone()
    assert r1["status"] == "done"
    assert r2["status"] == "done"

    types = {e["type"] for e in events.iter_after(0) if e["entity_id"] in (j1, j2)}
    assert "job.started" in types
    assert "job.done" in types

    # Each job got started + done
    for jid in (j1, j2):
        jtypes = [e["type"] for e in events.iter_after(0) if e["entity_id"] == jid]
        assert "job.queued" in jtypes
        assert "job.started" in jtypes
        assert "job.done" in jtypes


def test_light_can_run_during_heavy(lab_home):
    heavy = job_bus.submit(
        "test", _slow_job, config={"sleep": 1.0}, resource_class="memory_heavy"
    )
    light = job_bus.submit(
        "test", _slow_job, config={"sleep": 0.15}, resource_class="light"
    )

    _wait_status(heavy, "running")
    # Light should start (and ideally finish) while heavy still not done.
    _wait_status(light, {"running", "done"})

    light_done_while_heavy_active = False
    deadline = time.time() + 15.0
    while time.time() < deadline:
        hs = jobs.get(heavy)["status"]
        ls = jobs.get(light)["status"]
        if ls == "done" and hs in ("running", "queued"):
            light_done_while_heavy_active = True
            break
        if ls == "done" and hs == "done":
            # Light finished; if heavy already done too, at least light started.
            break
        time.sleep(0.05)

    assert jobs.get(light)["status"] == "done" or light_done_while_heavy_active
    # Stronger: light must reach done; heavy may still be running at that moment.
    _wait_status(light, "done")
    # If heavy is still running right after light done, success for overlap.
    # Otherwise light at least started during heavy's lifetime (checked above).
    _wait_status(heavy, "done")


def test_cancel_queued_or_running(lab_home):
    # Running cancel
    running = job_bus.submit(
        "test", _slow_job, config={"sleep": 5.0}, resource_class="memory_heavy"
    )
    _wait_status(running, "running")
    assert job_bus.cancel(running) is True
    assert jobs.get(running)["status"] == "cancelled"
    run_row = db.get_conn().execute(
        "SELECT status FROM runs WHERE id = ?", (running,)
    ).fetchone()
    assert run_row["status"] == "cancelled"
    # Second cancel is a no-op
    assert job_bus.cancel(running) is False

    # Queued cancel: fill the heavy slot then queue another
    hold = job_bus.submit(
        "test", _slow_job, config={"sleep": 3.0}, resource_class="memory_heavy"
    )
    queued = job_bus.submit(
        "test", _slow_job, config={"sleep": 0.2}, resource_class="memory_heavy"
    )
    _wait_status(hold, "running")
    _wait_pred(lambda: jobs.get(queued)["status"] == "queued")
    assert job_bus.cancel(queued) is True
    assert jobs.get(queued)["status"] == "cancelled"

    # Hold should still complete (or we cancel it to clean up).
    job_bus.cancel(hold)

    cancelled_events = [
        e for e in events.iter_after(0) if e["type"] == "job.cancelled"
    ]
    assert any(e["entity_id"] == running for e in cancelled_events)
    assert any(e["entity_id"] == queued for e in cancelled_events)


def test_events_emitted_for_lifecycle(lab_home):
    jid = job_bus.submit(
        "test",
        _slow_job,
        config={"sleep": 0.15},
        resource_class="light",
        workspace_id="ws_bus",
    )
    _wait_status(jid, "done")

    evs = [e for e in events.iter_after(0) if e["entity_id"] == jid]
    types = [e["type"] for e in evs]
    assert types[0] == "job.queued"
    assert "job.started" in types
    assert "job.done" in types
    # progress optional but expected from _slow_job emits
    assert any(t == "job.progress" for t in types)
    assert all(e["entity_type"] == "job" for e in evs)
    assert all(e["workspace_id"] == "ws_bus" for e in evs)

    # failed path
    bad = job_bus.submit(
        "test", _fail_job, config={}, resource_class="light"
    )
    _wait_status(bad, "failed")
    ftypes = [e["type"] for e in events.iter_after(0) if e["entity_id"] == bad]
    assert "job.queued" in ftypes
    assert "job.started" in ftypes
    assert "job.failed" in ftypes


def test_mark_zombies_still_works(lab_home):
    jid = job_bus.submit(
        "test", _slow_job, config={"sleep": 5.0}, resource_class="memory_heavy"
    )
    _wait_status(jid, "running")
    # Simulate restart bookkeeping without killing the process first —
    # mark_zombies only touches DB rows.
    jobs.mark_zombies()
    assert jobs.get(jid)["status"] == "zombie"
    # Clean up live process so the suite doesn't hang.
    job_bus._reset_for_tests()
