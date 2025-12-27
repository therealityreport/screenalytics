from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from tests.api._sse_utils import write_sample_faces, write_sample_tracks

pytest.importorskip("numpy")


def test_cluster_accepts_event_stream_header(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "cluster-sse"
    run_id = "run-sse-cluster"
    write_sample_tracks(ep_id, sample_count=8, run_id=run_id)
    write_sample_faces(ep_id, face_count=8, run_id=run_id)

    def _fake_enqueue(*, ep_id: str, run_id: str, stage: str, params: dict | None, source: str = "test") -> dict:
        return {
            "status": "queued",
            "ep_id": ep_id,
            "run_id": run_id,
            "stage": stage,
            "job_id": "job-cluster-sse",
            "params_hash": "hash",
        }

    monkeypatch.setattr("apps.api.routers.jobs._enqueue_run_stage_job", _fake_enqueue)

    client = TestClient(app)
    headers = {"accept": "text/event-stream"}
    response = client.post("/jobs/cluster", headers=headers, json={"ep_id": ep_id, "run_id": run_id})
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "queued"
    assert data["job_id"] == "job-cluster-sse"
