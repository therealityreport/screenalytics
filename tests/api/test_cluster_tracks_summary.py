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
        {
            "ep_id": ep_id,
            "identities": [{"identity_id": "id_0001", "track_ids": [1], "name": "Test"}],
        },
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


def test_cluster_tracks_include_skipped_face_previews(tmp_path):
    client = TestClient(app)
    ep_id = "demo-s01e02"
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
    tracks_path = get_path(ep_id, "tracks")
    identities_path = manifests_dir / "identities.json"
    frames_root = get_path(ep_id, "frames_root")

    crop_dir = frames_root / "crops" / "track_0002"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_file = crop_dir / "frame_000007.jpg"
    crop_file.write_bytes(b"x")

    _write_jsonl(
        faces_path,
        [
            {
                "track_id": 2,
                "frame_idx": 7,
                "face_id": "face_skip",
                "skip": True,
                "crop_rel_path": "crops/track_0002/frame_000007.jpg",
            }
        ],
    )
    _write_jsonl(
        tracks_path,
        [
            {
                "track_id": 2,
                "faces_count": 1,
                "frame_count": 1,
            }
        ],
    )
    _write_json(
        identities_path,
        {
            "ep_id": ep_id,
            "identities": [
                {
                    "identity_id": "id_skip",
                    "track_ids": [2],
                }
            ],
        },
    )

    resp = client.get(f"/episodes/{ep_id}/cluster_tracks")
    assert resp.status_code == 200
    track = resp.json()["clusters"][0]["tracks"][0]
    assert track["track_id"] == 2
    preview_url = track.get("rep_thumb_url")
    assert preview_url, "cluster preview URL should be hydrated even when faces are skipped"
    assert preview_url.endswith("frame_000007.jpg")


def test_cluster_tracks_prefer_best_quality_thumb(tmp_path):
    client = TestClient(app)
    ep_id = "demo-s01e03"
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
    tracks_path = get_path(ep_id, "tracks")
    identities_path = manifests_dir / "identities.json"
    frames_root = get_path(ep_id, "frames_root")
    thumbs_dir = frames_root / "thumbs" / "track_0003"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    low_thumb = thumbs_dir / "thumb_000005.jpg"
    high_thumb = thumbs_dir / "thumb_000050.jpg"
    low_thumb.write_bytes(b"low")
    high_thumb.write_bytes(b"high")

    _write_jsonl(
        faces_path,
        [
            {
                "track_id": 3,
                "frame_idx": 5,
                "quality": 0.2,
                "thumb_rel_path": "track_0003/thumb_000005.jpg",
                "crop_rel_path": "crops/track_0003/frame_000005.jpg",
            },
            {
                "track_id": 3,
                "frame_idx": 50,
                "quality": 0.95,
                "thumb_rel_path": "track_0003/thumb_000050.jpg",
                "crop_rel_path": "crops/track_0003/frame_000050.jpg",
            },
        ],
    )
    _write_jsonl(
        tracks_path,
        [
            {
                "track_id": 3,
                "faces_count": 2,
                "frame_count": 10,
            }
        ],
    )
    _write_json(
        identities_path,
        {
            "ep_id": ep_id,
            "identities": [
                {
                    "identity_id": "id_best_thumb",
                    "track_ids": [3],
                }
            ],
        },
    )

    resp = client.get(f"/episodes/{ep_id}/cluster_tracks")
    assert resp.status_code == 200
    track = resp.json()["clusters"][0]["tracks"][0]
    assert track["track_id"] == 3
    thumb_url = track.get("rep_thumb_url")
    media_url = track.get("rep_media_url")
    assert thumb_url and thumb_url.endswith("thumb_000050.jpg")
    assert media_url and media_url.endswith("thumb_000050.jpg")
