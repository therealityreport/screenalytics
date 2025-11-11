from urllib.parse import urlparse

from fastapi.testclient import TestClient

from apps.api.main import app


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
    assert data["object_key"].startswith(f"videos/{ep_id}")
    assert data["method"] == "FILE"
    assert data["local_video_path"].endswith("episode.mp4")
    assert data["path"].endswith("episode.mp4")

    parsed = urlparse(data["upload_url"])
    assert parsed.scheme in {"http", "https"}
    assert data["bucket"]
    assert data["headers"]["Content-Type"] == "video/mp4"
