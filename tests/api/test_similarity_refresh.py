"""Regression tests to ensure similarity indexes stay current after edits."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.services.people import PeopleService
from apps.api.services.track_reps import generate_track_reps_and_centroids
from py_screenalytics.artifacts import ensure_dirs, get_path
from py_screenalytics import run_layout


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@pytest.fixture()
def data_root(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    from apps.api.services import people as people_module

    monkeypatch.setattr(people_module, "DEFAULT_DATA_ROOT", data_root, raising=False)
    return data_root


def test_clusters_summary_updates_after_track_delete(data_root):
    client = TestClient(app)
    ep_id = "demo-s01e01"
    run_id = "run-1"
    ensure_dirs(ep_id)
    manifests_dir = run_layout.run_root(ep_id, run_id)
    faces_path = manifests_dir / "faces.jsonl"
    tracks_path = manifests_dir / "tracks.jsonl"
    identities_path = manifests_dir / "identities.json"
    frames_root = get_path(ep_id, "frames_root")
    run_crops_root = frames_root / "runs" / run_id / "crops"

    # Create crops so representative selection passes quality gates.
    for track_id in (1, 2):
        crop_dir = run_crops_root / f"track_{track_id:04d}"
        crop_dir.mkdir(parents=True, exist_ok=True)
        (crop_dir / "frame_000000.jpg").write_bytes(b"x")

    embeddings = {
        1: [1.0, 0.0, 0.0],
        2: [0.0, 1.0, 0.0],
    }
    face_rows = []
    for track_id, vector in embeddings.items():
        face_rows.append(
            {
                "face_id": f"face_{track_id}",
                "track_id": track_id,
                "frame_idx": 0,
                "conf": 0.99,
                "crop_std": 5.0,
                "embedding": vector,
            }
        )
    _write_jsonl(faces_path, face_rows)

    _write_jsonl(
        tracks_path,
        [
            {"track_id": 1, "faces_count": 1, "frame_count": 1},
            {"track_id": 2, "faces_count": 1, "frame_count": 1},
        ],
    )

    people_service = PeopleService(data_root)
    person = people_service.create_person(
        "DEMO",
        name="Test Person",
        cluster_ids=[f"{ep_id}:{run_id}:id_0001"],
    )
    person_id = person["person_id"]

    _write_json(
        identities_path,
        {
            "ep_id": ep_id,
            "identities": [
                {
                    "identity_id": "id_0001",
                    "track_ids": [1, 2],
                    "person_id": person_id,
                    "name": "Test Person",
                }
            ],
            "stats": {"faces": 2, "clusters": 1},
        },
    )

    generate_track_reps_and_centroids(ep_id, run_id=run_id)

    def _similarities():
        resp = client.get(
            f"/episodes/{ep_id}/people/{person_id}/clusters_summary",
            params={"run_id": run_id},
        )
        assert resp.status_code == 200
        payload = resp.json()
        clusters = payload["clusters"]
        assert clusters, "expected at least one cluster"
        track_entries = clusters[0].get("track_reps", [])
        return {entry["track_id"]: entry["similarity"] for entry in track_entries}, track_entries

    initial_map, initial_tracks = _similarities()
    assert "track_0001" in initial_map and "track_0002" in initial_map
    assert initial_map["track_0001"] == pytest.approx(0.707, rel=0.01)
    assert initial_map["track_0002"] == pytest.approx(0.707, rel=0.01)
    assert len(initial_tracks) == 2

    resp = client.request(
        "DELETE",
        f"/episodes/{ep_id}/tracks/2?run_id={run_id}",
        json={"delete_faces": True},
    )
    assert resp.status_code == 200

    updated_map, updated_tracks = _similarities()
    assert list(updated_map.keys()) == ["track_0001"]
    assert updated_map["track_0001"] == pytest.approx(1.0, rel=0.01)
    assert len(updated_tracks) == 1
