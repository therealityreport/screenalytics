from __future__ import annotations

from typing import Dict, Iterable

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import episodes as episodes_router


class _FakeStorage:
    def __init__(self, items: Iterable[Dict[str, object]]) -> None:
        self._items = list(items)

    def list_episode_videos_s3(self, limit: int = 1000, **_) -> list[Dict[str, object]]:
        return self._items[:limit]


class _FakeStore:
    def __init__(self, existing: set[str]):
        self._existing = existing

    def exists(self, ep_id: str) -> bool:
        return ep_id in self._existing


def test_s3_videos_endpoint(monkeypatch) -> None:
    items = [
        {
            "bucket": "screenalytics",
            "key": "raw/videos/rhoa/s01/e01/episode.mp4",
            "ep_id": "rhoa-s01e01",
            "show": "rhoa",
            "season": 1,
            "episode": 1,
            "size": 123456,
            "last_modified": "2025-01-02T03:04:05Z",
            "etag": "etag-1",
            "key_version": "v2",
        },
        {
            "bucket": "screenalytics",
            "key": "raw/videos/rhobh-s05e17/episode.mp4",
            "ep_id": "rhobh-s05e17",
            "size": 654321,
            "last_modified": "2025-01-03T04:05:06Z",
            "etag": "etag-2",
            "key_version": "v1",
        },
        {
            "bucket": "screenalytics",
            "key": "raw/videos/random.txt",
            "ep_id": "random",
        },
    ]

    fake_storage = _FakeStorage(items)
    fake_store = _FakeStore({"rhoa-s01e01"})

    monkeypatch.setattr(episodes_router, "STORAGE", fake_storage)
    monkeypatch.setattr(episodes_router, "EPISODE_STORE", fake_store)

    client = TestClient(app)
    resp = client.get("/episodes/s3_videos")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    ep_ids = {item["ep_id"] for item in data["items"]}
    assert ep_ids == {"rhoa-s01e01", "rhobh-s05e17"}

    rhoa_entry = next(item for item in data["items"] if item["ep_id"] == "rhoa-s01e01")
    assert rhoa_entry["exists_in_store"] is True
    assert rhoa_entry["show"] == "rhoa"
    assert rhoa_entry["season"] == 1
    assert rhoa_entry["episode"] == 1
    assert rhoa_entry["key_version"] == "v2"
    rhobh_entry = next(item for item in data["items"] if item["ep_id"] == "rhobh-s05e17")
    assert rhobh_entry["exists_in_store"] is False
