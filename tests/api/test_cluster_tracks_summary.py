import json

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import episodes as episodes_router
from py_screenalytics import run_layout
from py_screenalytics.artifacts import ensure_dirs, get_path


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "s3")
    return data_root


class _FakeStorage:
    backend = "s3"
    bucket = "demo-bucket"
    _client = object()
    init_error = None

    def presign_get(self, key: str, expires_in: int = 3600, content_type: str | None = None) -> str:
        return f"https://example.com/{key}"


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_cluster_summary_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(episodes_router, "STORAGE", _FakeStorage())
    client = TestClient(app)
    ep_id = "demo-s01e01"
    run_id = "run-summary-1"
    manifests_dir = run_layout.run_root(ep_id, run_id)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    faces_path = manifests_dir / "faces.jsonl"
    tracks_path = manifests_dir / "tracks.jsonl"
    identities_path = manifests_dir / "identities.json"

    _write_jsonl(
        faces_path,
        [
            {
                "track_id": 1,
                "frame_idx": 0,
                "face_id": "face_0",
                "crop_s3_key": "artifacts/crops/demo-s01e01/track_0001/frame_000000.jpg",
            },
            {
                "track_id": 1,
                "frame_idx": 1,
                "face_id": "face_1",
                "crop_s3_key": "artifacts/crops/demo-s01e01/track_0001/frame_000000.jpg",
            },
        ],
    )
    _write_jsonl(
        tracks_path,
        [
            {
                "track_id": 1,
                "faces_count": 2,
                "frame_count": 2,
                "best_crop_s3_key": "artifacts/crops/demo-s01e01/track_0001/frame_000000.jpg",
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

    resp = client.get(f"/episodes/{ep_id}/cluster_tracks", params={"run_id": run_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ep_id"] == ep_id
    assert len(data["clusters"]) == 1
    assert "singleton_merge" not in data or data.get("singleton_merge") is None
    cluster = data["clusters"][0]
    assert cluster["identity_id"] == "id_0001"
    assert cluster["counts"]["tracks"] == 1
    assert cluster["tracks"][0]["faces"] == 2


def test_cluster_tracks_include_skipped_face_previews(tmp_path, monkeypatch):
    monkeypatch.setattr(episodes_router, "STORAGE", _FakeStorage())
    client = TestClient(app)
    ep_id = "demo-s01e02"
    run_id = "run-skipped-1"
    manifests_dir = run_layout.run_root(ep_id, run_id)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    faces_path = manifests_dir / "faces.jsonl"
    tracks_path = manifests_dir / "tracks.jsonl"
    identities_path = manifests_dir / "identities.json"

    _write_jsonl(
        faces_path,
        [
            {
                "track_id": 2,
                "frame_idx": 7,
                "face_id": "face_skip",
                "skip": True,
                "crop_s3_key": "artifacts/crops/demo-s01e02/track_0002/frame_000007.jpg",
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

    resp = client.get(f"/episodes/{ep_id}/cluster_tracks", params={"run_id": run_id})
    assert resp.status_code == 200
    track = resp.json()["clusters"][0]["tracks"][0]
    assert track["track_id"] == 2
    preview_url = track.get("rep_thumb_url")
    assert preview_url, "cluster preview URL should be hydrated even when faces are skipped"
    assert preview_url.endswith("frame_000007.jpg")


def test_cluster_tracks_prefer_best_quality_thumb(tmp_path, monkeypatch):
    monkeypatch.setattr(episodes_router, "STORAGE", _FakeStorage())
    client = TestClient(app)
    ep_id = "demo-s01e03"
    run_id = "run-best-thumb-1"
    manifests_dir = run_layout.run_root(ep_id, run_id)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    faces_path = manifests_dir / "faces.jsonl"
    tracks_path = manifests_dir / "tracks.jsonl"
    identities_path = manifests_dir / "identities.json"

    _write_jsonl(
        faces_path,
        [
            {
                "track_id": 3,
                "frame_idx": 5,
                "quality": 0.2,
                "crop_s3_key": "artifacts/crops/demo-s01e03/track_0003/frame_000005.jpg",
            },
            {
                "track_id": 3,
                "frame_idx": 50,
                "quality": 0.95,
                "crop_s3_key": "artifacts/crops/demo-s01e03/track_0003/frame_000050.jpg",
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

    resp = client.get(f"/episodes/{ep_id}/cluster_tracks", params={"run_id": run_id})
    assert resp.status_code == 200
    track = resp.json()["clusters"][0]["tracks"][0]
    assert track["track_id"] == 3
    thumb_url = track.get("rep_thumb_url")
    media_url = track.get("rep_media_url")
    assert thumb_url and thumb_url.endswith("frame_000050.jpg")
    assert media_url and media_url.endswith("frame_000050.jpg")


def test_cluster_status_includes_singleton_merge_metrics(tmp_path):
    client = TestClient(app)
    ep_id = "demo-s01e04"
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    identities_path = manifests_dir / "identities.json"
    stats = {
        "faces": 4,
        "clusters": 3,
        "singleton_stats": {
            "enabled": True,
            "threshold": 0.5,
            "before": {"singleton_fraction": 0.7, "cluster_count": 4, "singleton_count": 3},
            "after": {"singleton_fraction": 0.3, "cluster_count": 3, "singleton_count": 1, "merge_count": 4},
        },
        "singleton_merge": {
            "enabled": True,
            "singleton_fraction_before": 0.7,
            "singleton_fraction_after": 0.3,
            "total_clusters_before": 4,
            "total_clusters_after": 3,
            "similarity_thresh": 0.62,
            "neighbor_top_k": 10,
            "num_singleton_merges": 4,
        },
    }
    identities_payload = {
        "ep_id": ep_id,
        "stats": stats,
        "identities": [
            {"identity_id": "id_0001", "track_ids": [1]},
            {"identity_id": "id_0002", "track_ids": [2]},
            {"identity_id": "id_0003", "track_ids": [3]},
        ],
    }
    identities_path.write_text(json.dumps(identities_payload), encoding="utf-8")

    resp = client.get(f"/episodes/{ep_id}/status")
    assert resp.status_code == 200
    cluster = resp.json().get("cluster", {})
    assert cluster.get("singleton_merge_enabled") is True
    assert cluster.get("singleton_merge_neighbor_top_k") == 10
    assert cluster.get("singleton_merge_merge_count") == 4
    assert cluster.get("singleton_merge_similarity_thresh") == 0.62
    assert cluster.get("singleton_fraction_after") == 0.3
    assert cluster.get("singleton_stats", {}).get("after", {}).get("singleton_fraction") == 0.3
