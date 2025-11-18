import json
from pathlib import Path

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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_reassign_frames_creates_new_track_and_identity(tmp_path):
    client = TestClient(app)
    ep_id = "demo-s01e01"
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
    tracks_path = get_path(ep_id, "tracks")
    identities_path = manifests_dir / "identities.json"

    frames_root = get_path(ep_id, "frames_root")
    crops_dir = frames_root / "crops" / "track_0001"
    thumbs_dir = frames_root / "thumbs" / "track_0001"
    crops_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in (0, 1):
        (crops_dir / f"frame_{frame_idx:06d}.jpg").write_bytes(b"x")
        (thumbs_dir / f"thumb_{frame_idx:06d}.jpg").write_bytes(b"x")

    faces_rows = [
        {
            "ep_id": ep_id,
            "face_id": f"face_{frame_idx}",
            "track_id": 1,
            "frame_idx": frame_idx,
            "ts": float(frame_idx) / 10.0,
            "bbox_xyxy": [0, 0, 10, 10],
            "crop_rel_path": f"crops/track_0001/frame_{frame_idx:06d}.jpg",
            "thumb_rel_path": f"track_0001/thumb_{frame_idx:06d}.jpg",
            "crop_s3_key": f"artifacts/crops/demo/s01/e01/tracks/track_0001/frame_{frame_idx:06d}.jpg",
            "thumb_s3_key": f"artifacts/thumbs/demo/s01/e01/tracks/track_0001/thumb_{frame_idx:06d}.jpg",
        }
        for frame_idx in (0, 1)
    ]
    _write_jsonl(faces_path, faces_rows)
    _write_jsonl(
        tracks_path,
        [
            {
                "track_id": 1,
                "class": "face",
                "frame_count": 2,
                "faces_count": 2,
                "first_ts": 0.0,
                "last_ts": 0.1,
                "bboxes_sampled": [],
                "ep_id": ep_id,
                "thumb_rel_path": "track_0001/thumb_000000.jpg",
            }
        ],
    )
    _write_json(
        identities_path,
        {
            "ep_id": ep_id,
            "identities": [{"identity_id": "id_0001", "track_ids": [1]}],
            "stats": {},
        },
    )

    payload = {
        "from_track_id": 1,
        "face_ids": ["face_0"],
        "new_identity_name": "New Cast Member",
        "show_id": "demo",
    }
    resp = client.post(f"/episodes/{ep_id}/faces/move_frames", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["new_track_id"] == 2
    assert data["moved_faces"] == 1
    assert any(cluster["identity_id"] == data["target_identity_id"] for cluster in data.get("clusters", []))

    faces_after = [json.loads(line) for line in faces_path.read_text(encoding="utf-8").splitlines() if line]
    moved = next(row for row in faces_after if row["frame_idx"] == 0)
    assert moved["track_id"] == 2
    assert moved["crop_rel_path"].startswith("crops/track_0002/")
    assert (frames_root / moved["crop_rel_path"]).exists()

    tracks_after = [json.loads(line) for line in tracks_path.read_text(encoding="utf-8").splitlines() if line]
    assert any(row["track_id"] == 2 for row in tracks_after)

    identities_after = json.loads(identities_path.read_text(encoding="utf-8"))
    names = [ident.get("name") for ident in identities_after["identities"] if ident.get("name")]
    assert "New Cast Member" in names
    track_lists = [sorted(ident.get("track_ids", [])) for ident in identities_after["identities"]]
    assert [2] in track_lists

    roster_path = tmp_path / "data" / "rosters" / "demo.json"
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    assert roster["names"] == ["New Cast Member"]


def test_track_frame_move_endpoint_accepts_frame_ids(tmp_path):
    client = TestClient(app)
    ep_id = "demo-s01e02"
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
    tracks_path = get_path(ep_id, "tracks")
    identities_path = manifests_dir / "identities.json"
    frames_root = get_path(ep_id, "frames_root")
    (frames_root / "crops" / "track_0001").mkdir(parents=True, exist_ok=True)
    (frames_root / "thumbs" / "track_0001").mkdir(parents=True, exist_ok=True)
    for frame_idx in (0, 1):
        (frames_root / "crops" / "track_0001" / f"frame_{frame_idx:06d}.jpg").write_bytes(b"x")
        (frames_root / "thumbs" / "track_0001" / f"thumb_{frame_idx:06d}.jpg").write_bytes(b"x")
    faces_rows = [
        {
            "ep_id": ep_id,
            "face_id": f"face_{frame_idx}",
            "track_id": 1,
            "frame_idx": frame_idx,
            "ts": float(frame_idx),
            "crop_rel_path": f"crops/track_0001/frame_{frame_idx:06d}.jpg",
            "thumb_rel_path": f"track_0001/thumb_{frame_idx:06d}.jpg",
        }
        for frame_idx in (0, 1)
    ]
    _write_jsonl(faces_path, faces_rows)
    _write_jsonl(
        tracks_path,
        [
            {
                "track_id": 1,
                "faces_count": 2,
                "thumb_rel_path": "track_0001/thumb_000000.jpg",
            }
        ],
    )
    _write_json(
        identities_path,
        {
            "ep_id": ep_id,
            "identities": [{"identity_id": "id_0001", "track_ids": [1]}],
            "stats": {},
        },
    )

    payload = {"frame_ids": [0], "new_identity_name": "FrameMove", "show_id": "demo"}
    resp = client.post(f"/episodes/{ep_id}/tracks/1/frames/move", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["moved"] == 1
    ops_path = manifests_dir / "faces_ops.jsonl"
    assert ops_path.exists()
    ops_entries = [json.loads(line) for line in ops_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(entry.get("op") == "move_frame" and entry.get("frame_idx") == 0 for entry in ops_entries)


def test_track_frame_delete_endpoint_removes_multiple_frames(tmp_path):
    client = TestClient(app)
    ep_id = "demo-s01e03"
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
    tracks_path = get_path(ep_id, "tracks")
    identities_path = manifests_dir / "identities.json"
    frames_root = get_path(ep_id, "frames_root")
    crops_dir = frames_root / "crops" / "track_0001"
    thumbs_dir = frames_root / "thumbs" / "track_0001"
    crops_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in (0, 1, 2):
        (crops_dir / f"frame_{frame_idx:06d}.jpg").write_bytes(b"x")
        (thumbs_dir / f"thumb_{frame_idx:06d}.jpg").write_bytes(b"x")
    faces_rows = [
        {
            "ep_id": ep_id,
            "face_id": f"face_{frame_idx}",
            "track_id": 1,
            "frame_idx": frame_idx,
            "crop_rel_path": f"crops/track_0001/frame_{frame_idx:06d}.jpg",
            "thumb_rel_path": f"track_0001/thumb_{frame_idx:06d}.jpg",
        }
        for frame_idx in (0, 1, 2)
    ]
    _write_jsonl(faces_path, faces_rows)
    _write_jsonl(tracks_path, [{"track_id": 1, "faces_count": 3}])
    _write_json(
        identities_path,
        {
            "ep_id": ep_id,
            "identities": [{"identity_id": "id_0001", "track_ids": [1]}],
            "stats": {},
        },
    )

    resp = client.request(
        "DELETE",
        f"/episodes/{ep_id}/tracks/1/frames",
        json={"frame_ids": [0, 1], "delete_assets": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] == 2
    assert not (crops_dir / "frame_000000.jpg").exists()
    assert (crops_dir / "frame_000002.jpg").exists()
    faces_after = [json.loads(line) for line in faces_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    remaining_idxs = {row["frame_idx"] for row in faces_after}
    assert remaining_idxs == {2}
    ops_path = manifests_dir / "faces_ops.jsonl"
    entries = [json.loads(line) for line in ops_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert sum(1 for entry in entries if entry.get("op") == "delete_frame") == 2
