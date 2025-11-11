from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import episodes as episodes_router
from apps.api.services.episodes import EpisodeStore


def _reset_episode_store(monkeypatch, tmp_path) -> Path:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    episodes_router.EPISODE_STORE = EpisodeStore()
    return data_root


def test_upsert_by_id_creates_episode(monkeypatch, tmp_path) -> None:
    data_root = _reset_episode_store(monkeypatch, tmp_path)
    client = TestClient(app)
    payload = {
        "ep_id": "rhobh-s05e17",
        "show_slug": "rhobh",
        "season": 5,
        "episode": 17,
        "title": "Amster-damn!",
    }

    resp = client.post("/episodes/upsert_by_id", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["created"] is True
    assert body["ep_id"] == payload["ep_id"]
    assert body["season"] == payload["season"]
    store_file = data_root / "meta" / "episodes.json"
    assert store_file.exists()
    stored = json.loads(store_file.read_text(encoding="utf-8"))
    record = stored[payload["ep_id"]]
    assert record["show_ref"] == payload["show_slug"]
    assert record["title"] == payload["title"]


def test_upsert_by_id_is_idempotent(monkeypatch, tmp_path) -> None:
    data_root = _reset_episode_store(monkeypatch, tmp_path)
    client = TestClient(app)
    payload = {
        "ep_id": "rhoslc-s02e03",
        "show_slug": "rhoslc",
        "season": 2,
        "episode": 3,
        "title": "Cracks in the Ice",
    }
    first = client.post("/episodes/upsert_by_id", json=payload)
    assert first.status_code == 200
    assert first.json()["created"] is True

    duplicate = payload | {"title": "New Title"}
    second = client.post("/episodes/upsert_by_id", json=duplicate)
    assert second.status_code == 200
    assert second.json()["created"] is False

    store_file = data_root / "meta" / "episodes.json"
    stored = json.loads(store_file.read_text(encoding="utf-8"))
    record = stored[payload["ep_id"]]
    assert record["title"] == payload["title"]  # unchanged by duplicate upsert
