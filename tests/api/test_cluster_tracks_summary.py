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


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_cluster_summary_endpoint(tmp_path):
    client = TestClient(app)
    ep_id = "demo-s01e01"
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
    tracks_path = get_path(ep_id, "tracks")
    identities_path = manifests_dir / "identities.json"
    frames_root = get_path(ep_id, "frames_root")
    thumbs_dir = frames_root / "thumbs" / "track_0001"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    (thumbs_dir / "thumb_000000.jpg").write_bytes(b"x")

    _write_jsonl(
        faces_path,
        [
            {"track_id": 1, "frame_idx": 0, "face_id": "face_0"},
            {"track_id": 1, "frame_idx": 1, "face_id": "face_1"},
        ],
    )
    _write_jsonl(
        tracks_path,
        [
            {
                "track_id": 1,
                "faces_count": 2,
                "frame_count": 2,
                "thumb_rel_path": "track_0001/thumb_000000.jpg",
            }
        ],
    )
    _write_json(
        identities_path,
        {"ep_id": ep_id, "identities": [{"identity_id": "id_0001", "track_ids": [1], "name": "Test"}]},
    )

    resp = client.get(f"/episodes/{ep_id}/cluster_tracks")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ep_id"] == ep_id
    assert len(data["clusters"]) == 1
    cluster = data["clusters"][0]
    assert cluster["identity_id"] == "id_0001"
    assert cluster["counts"]["tracks"] == 1
    assert cluster["tracks"][0]["faces"] == 2
