import os
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import app


def test_detect_job_stub(tmp_path, monkeypatch):
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    client = TestClient(app)
    payload = {
        "ep_id": "ep_demo",
        "video": "/path/to/video.mp4",
        "stride": 4,
        "fps": 2.0,
        "stub": True,
    }
    resp = client.post("/jobs/detect", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["job"] == "detect"
    assert "--stub" in data["command"]
    assert data["artifacts"]["detections"].startswith(str(tmp_path))


def test_track_job_stub(tmp_path, monkeypatch):
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    client = TestClient(app)
    resp = client.post("/jobs/track", json={"ep_id": "ep_demo"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["job"] == "track"
    assert data["artifacts"]["tracks"].startswith(str(tmp_path))
    assert "--ep-id" in data["command"]
