from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import app
from py_screenalytics.artifacts import ensure_dirs, get_path


def _write_tracks(ep_id: str) -> None:
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    tracks_path = manifests_dir / "tracks.jsonl"
    sample = {
        "ep_id": ep_id,
        "track_id": 7,
        "bboxes_sampled": [{"frame_idx": 12, "ts": 0.5, "bbox_xyxy": [1, 1, 10, 10]}],
    }
    tracks_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")


def _progress_path(ep_id: str) -> Path:
    return get_path(ep_id, "detections").parent / "progress.json"


def test_progress_updates_for_faces_and_cluster(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "show-s03e02"
    _write_tracks(ep_id)
    client = TestClient(app)

    # Faces stage
    resp_faces = client.post("/jobs/faces_embed", json={"ep_id": ep_id, "stub": True})
    assert resp_faces.status_code == 200
    progress_payload = json.loads(_progress_path(ep_id).read_text(encoding="utf-8"))
    assert progress_payload["phase"] == "done"
    assert progress_payload.get("summary", {}).get("stage") == "faces_embed"

    # Cluster stage overrides progress with new summary
    resp_cluster = client.post("/jobs/cluster", json={"ep_id": ep_id, "stub": True})
    assert resp_cluster.status_code == 200
    progress_payload = json.loads(_progress_path(ep_id).read_text(encoding="utf-8"))
    assert progress_payload["phase"] == "done"
    assert progress_payload.get("summary", {}).get("stage") == "cluster"
