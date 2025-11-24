from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import app


def _write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_episode_progress_returns_payload(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    progress_path = data_root / "manifests" / "ep-demo" / "progress.json"
    progress_payload = {
        "phase": "detect",
        "frames_done": 25,
        "frames_total": 100,
        "secs_done": 12.5,
        "secs_total": 50.0,
    }
    _write_progress(progress_path, progress_payload)

    client = TestClient(app)
    resp = client.get("/episodes/ep-demo/progress")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ep_id"] == "ep-demo"
    assert body["progress"] == progress_payload


def test_episode_progress_handles_missing(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    client = TestClient(app)
    resp = client.get("/episodes/missing-ep/progress")
    assert resp.status_code == 404
