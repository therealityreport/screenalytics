from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import app
from py_screenalytics.artifacts import ensure_dirs, get_path


def _bootstrap_facebank(ep_id: str, data_root: Path, *, run_id: str | None = None) -> None:
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    manifests_dir.mkdir(parents=True, exist_ok=True)
    target_dir = manifests_dir / "runs" / run_id if run_id else manifests_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    faces_path = target_dir / "faces.jsonl"
    track_path = target_dir / "tracks.jsonl"
    identities_path = target_dir / "identities.json"

    faces_rows = [
        {
            "ep_id": ep_id,
            "face_id": "face_0001_000010",
            "track_id": 1,
            "frame_idx": 10,
            "ts": 0.1,
            "thumb_rel_path": "track_0001/thumb_000010.jpg",
        },
        {
            "ep_id": ep_id,
            "face_id": "face_0001_000020",
            "track_id": 1,
            "frame_idx": 20,
            "ts": 0.2,
            "thumb_rel_path": "track_0001/thumb_000020.jpg",
        },
        {
            "ep_id": ep_id,
            "face_id": "face_0002_000030",
            "track_id": 2,
            "frame_idx": 30,
            "ts": 0.3,
            "thumb_rel_path": "track_0002/thumb_000030.jpg",
        },
    ]
    faces_path.write_text("\n".join(json.dumps(row) for row in faces_rows) + "\n", encoding="utf-8")

    tracks_rows = [
        {
            "track_id": 1,
            "faces_count": 2,
            "thumb_rel_path": "track_0001/thumb_000010.jpg",
        },
        {
            "track_id": 2,
            "faces_count": 1,
            "thumb_rel_path": "track_0002/thumb_000030.jpg",
        },
    ]
    track_path.write_text("\n".join(json.dumps(row) for row in tracks_rows) + "\n", encoding="utf-8")

    identities_payload = {
        "ep_id": ep_id,
        "identities": [
            {
                "identity_id": "id_0001",
                "label": None,
                "track_ids": [1],
                "size": 2,
                "rep_thumb_rel_path": "track_0001/thumb_000010.jpg",
            },
            {
                "identity_id": "id_0002",
                "label": "Guest",
                "track_ids": [2],
                "size": 1,
                "rep_thumb_rel_path": "track_0002/thumb_000030.jpg",
            },
        ],
        "stats": {"faces": 3, "clusters": 2},
    }
    identities_path.write_text(json.dumps(identities_payload, indent=2), encoding="utf-8")


def test_identity_rename_and_merge(monkeypatch, tmp_path) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "test-s01e01"
    run_id = "attempt-1"
    _bootstrap_facebank(ep_id, data_root, run_id=run_id)
    client = TestClient(app)

    rename_resp = client.post(
        f"/episodes/{ep_id}/identities/id_0001/rename",
        params={"run_id": run_id},
        json={"label": "Lead"},
    )
    assert rename_resp.status_code == 200
    merge_resp = client.post(
        f"/episodes/{ep_id}/identities/merge",
        params={"run_id": run_id},
        json={"source_id": "id_0002", "target_id": "id_0001"},
    )
    assert merge_resp.status_code == 200

    identities_path = get_path(ep_id, "detections").parent / "runs" / run_id / "identities.json"
    identities_doc = json.loads(identities_path.read_text(encoding="utf-8"))
    assert len(identities_doc["identities"]) == 1
    identity = identities_doc["identities"][0]
    assert identity["label"] == "Lead"
    assert sorted(identity["track_ids"]) == [1, 2]


def test_move_track_and_delete(monkeypatch, tmp_path) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "test-s01e02"
    _bootstrap_facebank(ep_id, data_root)
    client = TestClient(app)

    move_resp = client.post(
        f"/episodes/{ep_id}/tracks/1/move",
        json={"target_identity_id": None},
    )
    assert move_resp.status_code == 200

    delete_resp = client.request(
        "DELETE",
        f"/episodes/{ep_id}/tracks/1",
        json={"delete_faces": True},
    )
    assert delete_resp.status_code == 200

    faces_path = get_path(ep_id, "detections").parent / "faces.jsonl"
    remaining_faces = [line for line in faces_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(remaining_faces) == 1  # only track 2 remains


def test_delete_single_frame(monkeypatch, tmp_path) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "test-s01e03"
    _bootstrap_facebank(ep_id, data_root)
    client = TestClient(app)

    resp = client.request(
        "DELETE",
        f"/episodes/{ep_id}/frames",
        json={"track_id": 1, "frame_idx": 10, "delete_assets": False},
    )
    assert resp.status_code == 200
    faces_path = get_path(ep_id, "detections").parent / "faces.jsonl"
    faces_rows = [json.loads(line) for line in faces_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert all(not (row["track_id"] == 1 and row["frame_idx"] == 10) for row in faces_rows)
