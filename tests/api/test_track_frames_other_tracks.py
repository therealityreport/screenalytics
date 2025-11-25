import json

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from py_screenalytics.artifacts import ensure_dirs, get_path


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    return data_root


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_track_frames_include_other_tracks_hint(tmp_path):
    client = TestClient(app)
    ep_id = "demo-s01e03"
    ensure_dirs(ep_id)
    frames_root = get_path(ep_id, "frames_root")
    crop_dir = frames_root / "crops" / "track_0001"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / "frame_000010.jpg"
    crop_path.write_bytes(b"x")

    manifests_dir = get_path(ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
    tracks_path = get_path(ep_id, "tracks")

    _write_jsonl(
        faces_path,
        [
            {
                "track_id": 1,
                "frame_idx": 10,
                "face_id": "face_0001_000010",
                "ts": 1.0,
                "conf": 0.9,
                "bbox": [0, 0, 10, 10],
                "crop_rel_path": "crops/track_0001/frame_000010.jpg",
            },
            {
                # Second face in the same frame belongs to a different track
                "track_id": 2,
                "frame_idx": 10,
                "face_id": "face_0002_000010",
                "ts": 1.0,
                "conf": 0.8,
                "bbox": [10, 10, 20, 20],
                "skip": "blurry",
            },
        ],
    )
    _write_jsonl(
        tracks_path,
        [
            {"track_id": 1, "frame_count": 1, "faces_count": 1, "best_crop_rel_path": "crops/track_0001/frame_000010.jpg"},
            {"track_id": 2, "frame_count": 1, "faces_count": 1},
        ],
    )

    resp = client.get(f"/episodes/{ep_id}/tracks/1/frames")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total"] == 1
    assert payload["items"], "expected at least one frame entry"
    frame_entry = payload["items"][0]
    assert frame_entry["frame_idx"] == 10
    assert frame_entry["other_tracks"] == [2]
