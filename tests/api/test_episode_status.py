from __future__ import annotations

import json
import shutil

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
    (run_dir / "faces_embed.json").write_text(
        '{"phase": "faces_embed", "status": "success", "faces": 6, "started_at": "2024-12-01T09:55:00Z", "finished_at": "2024-12-01T10:00:00Z"}',
        encoding="utf-8",
    )
    (run_dir / "cluster.json").write_text(
        '{"phase": "cluster", "status": "success", "faces": 6, "identities": 2, "started_at": "2024-12-01T10:00:00Z", "finished_at": "2024-12-01T10:05:00Z"}',
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
