import os
import re
from urllib.parse import urlparse

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import episodes as episodes_router
from apps.api.services.storage import PresignedUpload


def extract_episode_number(ep_id: str) -> int:
    match = re.search(r"e(\d+)$", ep_id)
    if not match:
        raise ValueError(f"Unable to parse episode number from {ep_id}")
    return int(match.group(1))


class _FakeS3Storage:
    def __init__(self) -> None:
        self.backend = "s3"
        self._client = object()
        self.bucket = "screenalytics"
        self.init_error = None

    def video_object_key_v2(self, show_slug: str, season: int, episode: int) -> str:
        return f"raw/videos/{show_slug}/s{season:02d}/e{episode:02d}/episode.mp4"

    def presign_episode_video(self, ep_id: str, *, object_key: str, **_) -> PresignedUpload:
        return PresignedUpload(
            ep_id=ep_id,
            bucket=self.bucket,
            object_key=object_key,
            upload_url="https://storage/upload",
            expires_in=900,
            headers={"Content-Type": "video/mp4"},
            method="PUT",
        )


def test_create_episode_and_presign(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "s3")
    monkeypatch.setattr(episodes_router, "STORAGE", _FakeS3Storage())
    client = TestClient(app)

    payload = {
        "show_slug_or_id": "rhoslc",
        "season_number": 5,
        "episode_number": 7,
        "title": "Don't Ice Me Bro",
    }
    resp = client.post("/episodes", json=payload)
    assert resp.status_code == 200
    ep_id = resp.json()["ep_id"]
    assert ep_id.startswith("rhoslc-")

    presign = client.post(f"/episodes/{ep_id}/assets")
    assert presign.status_code == 200
    data = presign.json()
    assert data["ep_id"] == ep_id
    assert data["method"] == "PUT"
    expected_bucket = os.getenv("AWS_S3_BUCKET", "screenalytics")
    assert data["bucket"] in {expected_bucket}
    number = extract_episode_number(ep_id)
    assert data["key"] == f"raw/videos/rhoslc/s05/e{number:02}/episode.mp4"
    parsed = urlparse(data["upload_url"])
    assert parsed.scheme in {"http", "https"}
    assert data["headers"]["Content-Type"] == "video/mp4"

    assert data["local_video_path"].endswith("episode.mp4")
