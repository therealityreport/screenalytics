from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from apps.api.main import app
from py_screenalytics.artifacts import get_path
from tests.api._sse_utils import write_sample_faces, write_sample_tracks


def test_episode_status_from_run_markers_and_outputs(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "status-demo"
    write_sample_tracks(ep_id, sample_count=6)
    write_sample_faces(ep_id, face_count=6)

    client = TestClient(app)

    manifests_dir = get_path(ep_id, "detections").parent
    run_dir = manifests_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Make marker timestamps slightly *newer* than the manifest mtimes so the
    # status endpoint does not mark downstream phases as stale.
    now = datetime.now(timezone.utc).replace(microsecond=0)
    finished_dt = now + timedelta(seconds=30)
    started_dt = finished_dt - timedelta(minutes=5)
    started_at = started_dt.isoformat().replace("+00:00", "Z")
    finished_at = finished_dt.isoformat().replace("+00:00", "Z")
    (run_dir / "faces_embed.json").write_text(
        json.dumps(
            {
                "phase": "faces_embed",
                "status": "success",
                "faces": 6,
                "started_at": started_at,
                "finished_at": finished_at,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "cluster.json").write_text(
        json.dumps(
            {
                "phase": "cluster",
                "status": "success",
                "faces": 6,
                "identities": 2,
                "started_at": started_at,
                "finished_at": finished_at,
            }
        ),
        encoding="utf-8",
    )

    identities_payload = {
        "ep_id": ep_id,
        "stats": {"faces": 6, "clusters": 2},
        "identities": [
            {"identity_id": "id_0001", "track_ids": [1], "size": 3},
            {"identity_id": "id_0002", "track_ids": [2], "size": 3},
        ],
    }
    (manifests_dir / "identities.json").write_text(json.dumps(identities_payload), encoding="utf-8")

    status_resp = client.get(f"/episodes/{ep_id}/status")
    assert status_resp.status_code == 200
    payload = status_resp.json()

    faces_status = payload.get("faces_embed", {})
    assert faces_status.get("status") == "success"
    assert isinstance(faces_status.get("faces"), int) and faces_status["faces"] > 0
    assert faces_status.get("finished_at")
    assert faces_status.get("runtime_sec") == 300.0
    assert faces_status.get("source") == "marker"

    cluster_status = payload.get("cluster", {})
    assert cluster_status.get("status") == "success"
    assert isinstance(cluster_status.get("identities"), int) and cluster_status["identities"] > 0
    assert cluster_status.get("runtime_sec") == 300.0
    assert cluster_status.get("source") == "marker"

    if run_dir.exists():
        shutil.rmtree(run_dir)

    inferred_resp = client.get(f"/episodes/{ep_id}/status")
    assert inferred_resp.status_code == 200
    inferred = inferred_resp.json()
    assert inferred["faces_embed"]["status"] == "success"
    assert inferred["faces_embed"].get("source") == "output"
    assert inferred["faces_embed"].get("runtime_sec") is None
    assert inferred["cluster"]["status"] == "success"
    assert inferred["cluster"].get("source") == "output"
    assert inferred["cluster"].get("runtime_sec") is None


def test_detect_track_tracks_only_marks_stale(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "stale-tracks-only"

    manifests_dir = get_path(ep_id, "detections").parent
    tracks_path = manifests_dir / "tracks.jsonl"
    tracks_path.parent.mkdir(parents=True, exist_ok=True)
    tracks_path.write_text(
        json.dumps(
            {
                "ep_id": ep_id,
                "track_id": 1,
                "detector": "retinaface",
                "tracker": "bytetrack",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    client = TestClient(app)
    status_resp = client.get(f"/episodes/{ep_id}/status")
    assert status_resp.status_code == 200
    payload = status_resp.json()

    detect_status = payload.get("detect_track", {})
    assert detect_status.get("status") == "stale"
    assert payload.get("tracks_ready") is False
    # Scenes are considered ready when tracks manifest exists even if detections are missing
    assert payload.get("scenes_ready") is True


def test_episode_status_scopes_to_run_id(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "status-run-id-demo"
    run_id = "attempt-1"

    client = TestClient(app)

    manifests_dir = get_path(ep_id, "detections").parent
    runs_dir = manifests_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Ensure both legacy and run-scoped manifests exist so detect_track doesn't get marked stale.
    (manifests_dir / "detections.jsonl").write_text('{"d":1}\n', encoding="utf-8")
    (manifests_dir / "tracks.jsonl").write_text(
        json.dumps({"ep_id": ep_id, "track_id": 1, "detector": "retinaface", "tracker": "bytetrack"}) + "\n",
        encoding="utf-8",
    )
    (manifests_dir / "faces.jsonl").write_text('{"f":1}\n', encoding="utf-8")
    (manifests_dir / "identities.json").write_text(json.dumps({"identities": []}), encoding="utf-8")

    (run_dir / "detections.jsonl").write_text('{"d":1}\n', encoding="utf-8")
    (run_dir / "tracks.jsonl").write_text(
        json.dumps({"ep_id": ep_id, "track_id": 1, "detector": "retinaface", "tracker": "bytetrack"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "faces.jsonl").write_text('{"f":1}\n{"f":2}\n{"f":3}\n', encoding="utf-8")
    (run_dir / "identities.json").write_text(json.dumps({"identities": [{}, {}, {}, {}]}), encoding="utf-8")

    now = datetime.now(timezone.utc).replace(microsecond=0)
    detect_finished = now.isoformat().replace("+00:00", "Z")
    faces_finished = (now + timedelta(seconds=10)).isoformat().replace("+00:00", "Z")
    cluster_finished = (now + timedelta(seconds=20)).isoformat().replace("+00:00", "Z")

    # Legacy markers (runs/*.json)
    (runs_dir / "detect_track.json").write_text(
        json.dumps(
            {
                "phase": "detect_track",
                "status": "success",
                "run_id": "legacy",
                "tracks": 1,
                "detections": 1,
                "finished_at": detect_finished,
            }
        ),
        encoding="utf-8",
    )
    (runs_dir / "faces_embed.json").write_text(
        json.dumps(
            {
                "phase": "faces_embed",
                "status": "success",
                "run_id": "legacy",
                "faces": 1,
                "finished_at": faces_finished,
            }
        ),
        encoding="utf-8",
    )
    (runs_dir / "cluster.json").write_text(
        json.dumps(
            {
                "phase": "cluster",
                "status": "success",
                "run_id": "legacy",
                "faces": 1,
                "identities": 0,
                "finished_at": cluster_finished,
            }
        ),
        encoding="utf-8",
    )

    # Run-scoped markers (runs/{run_id}/*.json)
    (run_dir / "detect_track.json").write_text(
        json.dumps(
            {
                "phase": "detect_track",
                "status": "success",
                "run_id": run_id,
                "tracks": 2,
                "detections": 2,
                "finished_at": detect_finished,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "faces_embed.json").write_text(
        json.dumps(
            {
                "phase": "faces_embed",
                "status": "success",
                "run_id": run_id,
                "faces": 3,
                "finished_at": faces_finished,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "cluster.json").write_text(
        json.dumps(
            {
                "phase": "cluster",
                "status": "success",
                "run_id": run_id,
                "faces": 3,
                "identities": 4,
                "finished_at": cluster_finished,
            }
        ),
        encoding="utf-8",
    )

    run_resp = client.get(f"/episodes/{ep_id}/status", params={"run_id": run_id})
    assert run_resp.status_code == 200
    run_payload = run_resp.json()
    assert run_payload["detect_track"]["run_id"] == run_id
    assert run_payload["detect_track"]["tracks"] == 2
    assert run_payload["faces_embed"]["run_id"] == run_id
    assert run_payload["faces_embed"]["faces"] == 3
    assert run_payload["cluster"]["run_id"] == run_id
    assert run_payload["cluster"]["identities"] == 4

    legacy_resp = client.get(f"/episodes/{ep_id}/status")
    assert legacy_resp.status_code == 200
    legacy_payload = legacy_resp.json()
    assert legacy_payload["detect_track"]["run_id"] == "legacy"
    assert legacy_payload["faces_embed"]["run_id"] == "legacy"
    assert legacy_payload["faces_embed"]["faces"] == 1
