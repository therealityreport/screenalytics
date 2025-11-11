from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import app
from py_screenalytics.artifacts import ensure_dirs, get_path


def _bootstrap_tracks(ep_id: str) -> None:
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    manifests_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = manifests_dir / "tracks.jsonl"
    sample_track = {
        "ep_id": ep_id,
        "track_id": 1,
        "bboxes_sampled": [
            {"frame_idx": 10, "ts": 0.4, "bbox_xyxy": [0, 0, 50, 50]},
            {"frame_idx": 20, "ts": 0.8, "bbox_xyxy": [5, 5, 60, 60]},
        ],
    }
    tracks_path.write_text(json.dumps(sample_track) + "\n", encoding="utf-8")


def test_faces_embed_and_cluster_stub(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "show-s01e01"
    _bootstrap_tracks(ep_id)

    client = TestClient(app)
    faces_resp = client.post(
        "/jobs/faces_embed",
        json={"ep_id": ep_id, "stub": True, "save_crops": False},
    )
    assert faces_resp.status_code == 200
    faces_payload = faces_resp.json()
    assert faces_payload["faces_count"] > 0

    cluster_resp = client.post(
        "/jobs/cluster",
        json={"ep_id": ep_id, "stub": True},
    )
    assert cluster_resp.status_code == 200
    cluster_payload = cluster_resp.json()
    assert cluster_payload["identities_count"] >= 1
    identities_path = get_path(ep_id, "detections").parent / "identities.json"
    assert identities_path.exists()


def test_faces_embed_requires_tracks(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "missing-s01e01"
    client = TestClient(app)
    resp = client.post("/jobs/faces_embed", json={"ep_id": ep_id, "stub": True})
    assert resp.status_code == 400


def test_cluster_requires_faces(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "show-s02e01"
    _bootstrap_tracks(ep_id)
    client = TestClient(app)
    resp = client.post("/jobs/cluster", json={"ep_id": ep_id, "stub": True})
    assert resp.status_code == 400
