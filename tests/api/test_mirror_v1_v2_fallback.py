from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import episodes as episodes_router


class _StubStorage:
    bucket = "screenalytics"

    def __init__(self) -> None:
        self.mirror_calls = []

    def video_object_key_v2(self, show_slug: str, season: int, episode: int) -> str:
        return f"raw/videos/{show_slug}/s{season:02d}/e{episode:02d}/episode.mp4"

    def video_object_key_v1(self, ep_id: str) -> str:
        return f"raw/videos/{ep_id}/episode.mp4"

    def object_exists(self, key: str) -> bool:
        return key.endswith("legacy/episode.mp4")

    def presign_episode_video(self, *_, **__):
        raise NotImplementedError

    def ensure_local_mirror(self, ep_id: str, **kwargs):
        self.mirror_calls.append((ep_id, kwargs))
        return {
            "local_video_path": f"/tmp/{ep_id}.mp4",
            "bytes": 100,
            "etag": "etag",
            "used_key_version": "v1",
        }


def test_mirror_falls_back_to_v1(monkeypatch, tmp_path):
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path / "data"))
    client = TestClient(app)

    create_resp = client.post(
        "/episodes",
        json={
            "show_slug_or_id": "legacy",
            "season_number": 1,
            "episode_number": 1,
        },
    )
    ep_id = create_resp.json()["ep_id"]

    stub_storage = _StubStorage()
    monkeypatch.setattr(episodes_router, "STORAGE", stub_storage)

    resp = client.post(f"/episodes/{ep_id}/mirror")
    assert resp.status_code == 200
    data = resp.json()
    assert data["used_key_version"] == "v1"
    assert stub_storage.mirror_calls
    _, kwargs = stub_storage.mirror_calls[0]
    assert kwargs["show_ref"] == "legacy"
*** End Patch
