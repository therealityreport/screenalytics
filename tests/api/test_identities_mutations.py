import json
import os
from pathlib import Path

os.environ.setdefault("STORAGE_BACKEND", "local")

from fastapi.testclient import TestClient

from apps.api.main import app
from py_screenalytics.artifacts import ensure_dirs, get_path


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_identity_endpoints(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    ep_id = "demo-s01e01"
    run_id = "attempt-1"
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    run_dir = manifests_dir / "runs" / run_id
    tracks_path = run_dir / "tracks.jsonl"
    faces_path = run_dir / "faces.jsonl"
    identities_path = run_dir / "identities.json"

    _write_jsonl(
        tracks_path,
        [
            {"track_id": 1, "class": "face", "frame_count": 10},
            {"track_id": 2, "class": "face", "frame_count": 5},
        ],
    )
    _write_jsonl(
        faces_path,
        [
            {"track_id": 1, "frame_idx": 0, "ts": 0.0},
            {"track_id": 2, "frame_idx": 1, "ts": 0.5},
        ],
    )
    _write_json(
        identities_path,
        {
            "ep_id": ep_id,
            "identities": [
                {"identity_id": "id_0001", "label": "One", "track_ids": [1]},
                {"identity_id": "id_0002", "label": "Two", "track_ids": [2]},
            ],
            "stats": {},
        },
    )

    client = TestClient(app)

    resp = client.post(
        f"/identities/{ep_id}/rename",
        params={"run_id": run_id},
        json={"identity_id": "id_0001", "new_label": "Lead"},
    )
    assert resp.status_code == 200
    assert resp.json()["label"] == "Lead"

    resp = client.post(
        f"/identities/{ep_id}/merge",
        params={"run_id": run_id},
        json={"source_id": "id_0002", "target_id": "id_0001"},
    )
    assert resp.status_code == 200
    assert sorted(resp.json()["track_ids"]) == [1, 2]

    resp = client.post(
        f"/identities/{ep_id}/move_track",
        params={"run_id": run_id},
        json={"track_id": 2, "target_identity_id": None},
    )
    assert resp.status_code == 200

    resp = client.post(f"/identities/{ep_id}/drop_track", params={"run_id": run_id}, json={"track_id": 2})
    assert resp.status_code == 200
    assert resp.json()["track_id"] == 2

    resp = client.post(
        f"/identities/{ep_id}/drop_frame",
        params={"run_id": run_id},
        json={"track_id": 1, "frame_idx": 0, "delete_assets": False},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["removed"] == 1

    updated_identities = json.loads(identities_path.read_text(encoding="utf-8"))
    assert updated_identities["identities"][0]["label"] == "Lead"
