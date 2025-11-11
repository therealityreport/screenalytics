from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("DB_URL", "postgresql://demo")
sys.path.append(str(REPO_ROOT))

from mcps.postgres import server  # noqa: E402


class DummyCursor:
    def __init__(self, rows, recorder):
        self._rows = rows
        self._recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self._recorder["executed"].append({"query": query, "params": params})

    def fetchall(self):
        return self._rows


class DummyConn:
    def __init__(self, rows, recorder):
        self._rows = rows
        self._recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return DummyCursor(self._rows, self._recorder)


def test_episodes_by_show_returns_ids(monkeypatch):
    recorder = {"executed": []}
    rows = [("ep-a",), ("ep-b",)]
    monkeypatch.setattr(server, "_conn", lambda: DummyConn(rows, recorder))

    data = server.episodes_by_show("show-slug")
    assert data["ep_ids"] == ["ep-a", "ep-b"]
    params = recorder["executed"][0]["params"]
    assert params == ("show-slug", "show-slug")


def test_presence_by_person_supports_alias_and_slug(monkeypatch):
    recorder = {"executed": []}
    rows = [("ep_demo", 10, 5, 4)]
    monkeypatch.setattr(server, "_conn", lambda: DummyConn(rows, recorder))

    payload = server.presence_by_person("demo_person", "show_demo")
    assert payload["rows"][0]["visual_s"] == 10.0

    params = recorder["executed"][0]["params"]
    assert params["person"] == "demo_person"
    assert params["show"] == "show_demo"
