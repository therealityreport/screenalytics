import os
import re
from urllib.parse import urlparse

from fastapi.testclient import TestClient

from apps.api.main import app


def extract_episode_number(ep_id: str) -> int:
    match = re.search(r"e(\d+)$", ep_id)
    if not match:
        raise ValueError(f"Unable to parse episode number from {ep_id}")
    return int(match.group(1))


def test_create_episode_and_presign(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
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
    if data["method"] == "PUT":
        expected_bucket = os.getenv("AWS_S3_BUCKET", "screenalytics")
        assert data["bucket"] in {expected_bucket}
        number = extract_episode_number(ep_id)
        assert data["key"] == f"raw/videos/rhoslc/s05/e{number:02}/episode.mp4"
        parsed = urlparse(data["upload_url"])
        assert parsed.scheme in {"http", "https"}
        assert data["headers"]["Content-Type"] == "video/mp4"
    else:
        assert data["method"] == "FILE"
        assert data["bucket"] == "local"
        assert data["key"].startswith("videos/")
        assert data["path"].endswith("episode.mp4")

    assert data["local_video_path"].endswith("episode.mp4")
