import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import episodes as episodes_router
from apps.api.services.storage import StorageService
from py_screenalytics.artifacts import ensure_dirs, get_path


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-jpeg-bytes")


def test_list_track_crops_and_frames(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    episodes_router.STORAGE = StorageService()

    ep_id = "testshow-s01e01"
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
    crops_dir = get_path(ep_id, "frames_root") / "crops" / "track_0001"
    frames_dir = get_path(ep_id, "frames_root") / "frames"

    rows = []
    for idx in range(10):
        _write_image(crops_dir / f"frame_{idx:06d}.jpg")
        if idx < 5:
            _write_image(frames_dir / f"frame_{idx:06d}.jpg")
        rows.append({"track_id": 1, "frame_idx": idx, "ts": round(idx * 0.25, 3)})
    faces_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    client = TestClient(app)

    resp = client.get(
        f"/episodes/{ep_id}/tracks/1/crops",
        params={"sample": 2, "limit": 3, "offset": 1},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    assert all(item["track_id"] == 1 for item in data)
    frame_indices = [item["frame_idx"] for item in data]
    assert frame_indices == [2, 4, 6]
    assert all(item["url"] for item in data)

    resp_frames = client.get(
        f"/episodes/{ep_id}/tracks/1/frames",
        params={"sample": 1, "limit": 2, "offset": 1},
    )
    assert resp_frames.status_code == 200
    frames_payload = resp_frames.json()
    assert len(frames_payload) == 2
    assert [item["frame_idx"] for item in frames_payload] == [1, 2]
    assert all(item["url"] for item in frames_payload)
