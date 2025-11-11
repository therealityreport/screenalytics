from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import jobs as jobs_router
from apps.api.services.jobs import JobService


def _write_job_record(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


def test_progress_endpoint_tracks_updated_frames(tmp_path, monkeypatch) -> None:
    data_root = tmp_path
    jobs_dir = data_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = data_root / "manifests" / "ep-test"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    progress_path = manifests_dir / "progress.json"

    job_id = "job-progress"
    job_record = {
        "job_id": job_id,
        "ep_id": "ep-test",
        "job_type": "detect_track",
        "pid": 1234,
        "state": "running",
        "command": ["python", "tools/episode_run.py"],
        "started_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "ended_at": None,
        "progress_file": str(progress_path),
        "requested": {"stride": 5, "fps": None, "stub": False, "device": "auto"},
        "summary": None,
        "error": None,
        "return_code": None,
        "data_root": str(data_root),
    }
    _write_job_record(jobs_dir / f"{job_id}.json", job_record)

    first_progress = {
        "frames_done": 10,
        "frames_total": 100,
        "elapsed_sec": 5.0,
        "state": "running",
    }
    progress_path.write_text(json.dumps(first_progress), encoding="utf-8")

    service = JobService(data_root=data_root)
    monkeypatch.setattr(jobs_router, "JOB_SERVICE", service)

    client = TestClient(app)

    resp1 = client.get(f"/jobs/{job_id}/progress")
    assert resp1.status_code == 200
    payload1 = resp1.json()
    assert payload1["state"] == "running"
    assert payload1["progress"]["frames_done"] == 10

    updated_progress = first_progress | {"frames_done": 30, "elapsed_sec": 12.0}
    progress_path.write_text(json.dumps(updated_progress), encoding="utf-8")

    resp2 = client.get(f"/jobs/{job_id}/progress")
    assert resp2.status_code == 200
    payload2 = resp2.json()
    assert payload2["progress"]["frames_done"] == 30
    assert payload2["progress"]["elapsed_sec"] == 12.0
