import os

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import episodes as episodes_router
from apps.api.services.storage import PresignedUpload, StorageService


def _create_episode(client: TestClient) -> str:
    resp = client.post(
        "/episodes",
        json={
            "show_slug_or_id": "matrix-show",
            "season_number": 1,
            "episode_number": 1,
        },
    )
    assert resp.status_code == 200
    return resp.json()["ep_id"]


class _FakeS3Storage:
    backend = "s3"
    bucket = "screenalytics"
    _client = object()
    init_error = None

    def video_object_key_v2(self, show_slug: str, season: int, episode: int) -> str:
        return f"raw/videos/{show_slug}/s{season:02d}/e{episode:02d}/episode.mp4"

    def presign_episode_video(self, ep_id: str, *, object_key: str, **_) -> PresignedUpload:
        bucket = os.getenv("AWS_S3_BUCKET", "screenalytics")
        key = object_key
        return PresignedUpload(
            ep_id=ep_id,
            bucket=bucket,
            object_key=key,
            upload_url="https://example.com/upload",
            expires_in=900,
            headers={"Content-Type": "video/mp4"},
            method="PUT",
        )


def test_presign_matrix_local(monkeypatch, tmp_path):
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setattr(episodes_router, "STORAGE", StorageService())
    client = TestClient(app)
    ep_id = _create_episode(client)

    resp = client.post(f"/episodes/{ep_id}/assets")
    assert resp.status_code == 503
    assert "S3 misconfigured" in resp.json().get("message", "")


def test_presign_matrix_s3(monkeypatch, tmp_path):
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("STORAGE_BACKEND", "s3")
    monkeypatch.setenv("AWS_S3_BUCKET", "screenalytics")
    monkeypatch.setattr(episodes_router, "STORAGE", _FakeS3Storage())
    client = TestClient(app)
    ep_id = _create_episode(client)

    resp = client.post(f"/episodes/{ep_id}/assets")
    assert resp.status_code == 200
    data = resp.json()
    assert data["method"] == "PUT"
    assert data["bucket"] == "screenalytics"
    assert data["key"].startswith("raw/videos/")
    assert data["headers"]["Content-Type"] == "video/mp4"
