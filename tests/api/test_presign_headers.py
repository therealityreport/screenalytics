from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import episodes as episodes_router
from apps.api.services.storage import PresignedUpload, StorageService


class _FakeStorage:
    def __init__(self) -> None:
        self._calls = []
        self.backend = "s3"
        self._client = object()
        self.bucket = "screenalytics"
        self.init_error = None

    def video_object_key_v2(self, show_slug: str, season: int, episode: int) -> str:
        return f"raw/videos/{show_slug}/s{season:02d}/e{episode:02d}/episode.mp4"

    def presign_episode_video(self, ep_id: str, *, object_key: str, **_) -> PresignedUpload:
        self._calls.append(ep_id)
        return PresignedUpload(
            ep_id=ep_id,
            bucket="screenalytics",
            object_key=object_key,
            upload_url="https://storage/upload",
            expires_in=900,
            headers={"Content-Type": "video/mp4"},
            method="PUT",
        )


def _create_episode(client: TestClient) -> str:
    resp = client.post(
        "/episodes",
        json={
            "show_slug_or_id": "salt-lake",
            "season_number": 7,
            "episode_number": 3,
        },
    )
    assert resp.status_code == 200
    return resp.json()["ep_id"]


def test_presign_returns_file_path_for_local(monkeypatch, tmp_path):
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setattr(episodes_router, "STORAGE", StorageService())
    client = TestClient(app)
    ep_id = _create_episode(client)

    resp = client.post(f"/episodes/{ep_id}/assets")
    assert resp.status_code == 503
    assert "S3 misconfigured" in resp.json().get("message", "")


def test_presign_put_headers(monkeypatch, tmp_path):
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("STORAGE_BACKEND", "s3")
    fake_storage = _FakeStorage()
    monkeypatch.setattr(episodes_router, "STORAGE", fake_storage)
    client = TestClient(app)
    ep_id = _create_episode(client)

    resp = client.post(f"/episodes/{ep_id}/assets")
    assert resp.status_code == 200
    data = resp.json()
    assert data["method"] == "PUT"
    assert data["headers"]["Content-Type"] == "video/mp4"
    assert data["upload_url"].startswith("https://storage")
    assert data["key"].startswith("raw/videos/")
