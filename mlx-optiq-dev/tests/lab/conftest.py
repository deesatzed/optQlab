import pytest


@pytest.fixture()
def lab_home(tmp_path, monkeypatch):
    home = tmp_path / "optiq_home"
    home.mkdir()
    monkeypatch.setenv("OPTIQ_HOME", str(home))
    from optiq.lab import db
    if getattr(db._local, "conn", None) is not None:
        try:
            db._local.conn.close()
        except Exception:
            pass
        db._local.conn = None
    yield home
    if getattr(db._local, "conn", None) is not None:
        try:
            db._local.conn.close()
        except Exception:
            pass
        db._local.conn = None
