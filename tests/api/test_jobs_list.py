from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import jobs as jobs_router
from apps.api.services.jobs import JobService


def _write_job_record(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


def test_list_jobs_empty(tmp_path, monkeypatch) -> None:
    """Test that list_jobs returns empty list when no jobs exist."""
    data_root = tmp_path / "data"
    jobs_dir = data_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    service = JobService(data_root=data_root)
    result = service.list_jobs()

    assert result == []


def test_list_jobs_basic(tmp_path, monkeypatch) -> None:
    """Test that list_jobs returns all jobs."""
    data_root = tmp_path / "data"
    jobs_dir = data_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    # Create some test jobs
    job1 = {
        "job_id": "job-001",
        "ep_id": "test-s01e01",
        "job_type": "screen_time_analyze",
        "state": "succeeded",
        "started_at": "2025-01-01T10:00:00Z",
        "ended_at": "2025-01-01T10:01:00Z",
        "pid": 1234,
        "command": ["python", "analyze.py"],
        "requested": {},
        "progress_file": "progress.json",
        "data_root": str(data_root),
        "summary": None,
        "error": None,
        "return_code": 0,
    }
    job2 = {
        "job_id": "job-002",
        "ep_id": "test-s01e02",
        "job_type": "detect_track",
        "state": "running",
        "started_at": "2025-01-01T11:00:00Z",
        "ended_at": None,
        "pid": 5678,
        "command": ["python", "detect.py"],
        "requested": {},
        "progress_file": "progress.json",
        "data_root": str(data_root),
        "summary": None,
        "error": None,
        "return_code": None,
    }
    job3 = {
        "job_id": "job-003",
        "ep_id": "test-s01e01",
        "job_type": "cluster",
        "state": "failed",
        "started_at": "2025-01-01T09:00:00Z",
        "ended_at": "2025-01-01T09:05:00Z",
        "pid": 9999,
        "command": ["python", "cluster.py"],
        "requested": {},
        "progress_file": "progress.json",
        "data_root": str(data_root),
        "summary": None,
        "error": "Test error",
        "return_code": 1,
    }

    _write_job_record(jobs_dir / "job-001.json", job1)
    _write_job_record(jobs_dir / "job-002.json", job2)
    _write_job_record(jobs_dir / "job-003.json", job3)

    service = JobService(data_root=data_root)

    # Test: get all jobs (should be sorted by started_at desc)
    all_jobs = service.list_jobs()
    assert len(all_jobs) == 3
    assert all_jobs[0]["job_id"] == "job-002"  # Latest
    assert all_jobs[1]["job_id"] == "job-001"
    assert all_jobs[2]["job_id"] == "job-003"  # Oldest


def test_list_jobs_filter_by_ep_id(tmp_path, monkeypatch) -> None:
    """Test that list_jobs filters by ep_id correctly."""
    data_root = tmp_path / "data"
    jobs_dir = data_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    job1 = {
        "job_id": "job-001",
        "ep_id": "test-s01e01",
        "job_type": "screen_time_analyze",
        "state": "succeeded",
        "started_at": "2025-01-01T10:00:00Z",
        "ended_at": "2025-01-01T10:01:00Z",
        "pid": 1234,
        "command": [],
        "requested": {},
        "progress_file": "",
        "data_root": str(data_root),
        "summary": None,
        "error": None,
        "return_code": 0,
    }
    job2 = {
        "job_id": "job-002",
        "ep_id": "test-s01e02",
        "job_type": "screen_time_analyze",
        "state": "succeeded",
        "started_at": "2025-01-01T11:00:00Z",
        "ended_at": "2025-01-01T11:01:00Z",
        "pid": 5678,
        "command": [],
        "requested": {},
        "progress_file": "",
        "data_root": str(data_root),
        "summary": None,
        "error": None,
        "return_code": 0,
    }

    _write_job_record(jobs_dir / "job-001.json", job1)
    _write_job_record(jobs_dir / "job-002.json", job2)

    service = JobService(data_root=data_root)

    # Filter by ep_id
    filtered = service.list_jobs(ep_id="test-s01e01")
    assert len(filtered) == 1
    assert filtered[0]["job_id"] == "job-001"


def test_list_jobs_filter_by_job_type(tmp_path, monkeypatch) -> None:
    """Test that list_jobs filters by job_type correctly."""
    data_root = tmp_path / "data"
    jobs_dir = data_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    job1 = {
        "job_id": "job-001",
        "ep_id": "test-s01e01",
        "job_type": "screen_time_analyze",
        "state": "succeeded",
        "started_at": "2025-01-01T10:00:00Z",
        "ended_at": "2025-01-01T10:01:00Z",
        "pid": 1234,
        "command": [],
        "requested": {},
        "progress_file": "",
        "data_root": str(data_root),
        "summary": None,
        "error": None,
        "return_code": 0,
    }
    job2 = {
        "job_id": "job-002",
        "ep_id": "test-s01e01",
        "job_type": "detect_track",
        "state": "succeeded",
        "started_at": "2025-01-01T11:00:00Z",
        "ended_at": "2025-01-01T11:01:00Z",
        "pid": 5678,
        "command": [],
        "requested": {},
        "progress_file": "",
        "data_root": str(data_root),
        "summary": None,
        "error": None,
        "return_code": 0,
    }

    _write_job_record(jobs_dir / "job-001.json", job1)
    _write_job_record(jobs_dir / "job-002.json", job2)

    service = JobService(data_root=data_root)

    # Filter by job_type
    filtered = service.list_jobs(job_type="screen_time_analyze")
    assert len(filtered) == 1
    assert filtered[0]["job_id"] == "job-001"


def test_list_jobs_limit(tmp_path, monkeypatch) -> None:
    """Test that list_jobs respects limit parameter."""
    data_root = tmp_path / "data"
    jobs_dir = data_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    # Create 5 jobs
    for i in range(5):
        job = {
            "job_id": f"job-{i:03d}",
            "ep_id": "test-s01e01",
            "job_type": "screen_time_analyze",
            "state": "succeeded",
            "started_at": f"2025-01-01T{10+i:02d}:00:00Z",
            "ended_at": f"2025-01-01T{10+i:02d}:01:00Z",
            "pid": 1000 + i,
            "command": [],
            "requested": {},
            "progress_file": "",
            "data_root": str(data_root),
            "summary": None,
            "error": None,
            "return_code": 0,
        }
        _write_job_record(jobs_dir / f"job-{i:03d}.json", job)

    service = JobService(data_root=data_root)

    # Test limit
    limited = service.list_jobs(limit=2)
    assert len(limited) == 2
    assert limited[0]["job_id"] == "job-004"  # Most recent
    assert limited[1]["job_id"] == "job-003"


def test_list_jobs_endpoint(tmp_path, monkeypatch) -> None:
    """Test GET /jobs endpoint."""
    data_root = tmp_path / "data"
    jobs_dir = data_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    job1 = {
        "job_id": "job-001",
        "ep_id": "test-s01e01",
        "job_type": "screen_time_analyze",
        "state": "succeeded",
        "started_at": "2025-01-01T10:00:00Z",
        "ended_at": "2025-01-01T10:01:00Z",
        "pid": 1234,
        "command": [],
        "requested": {},
        "progress_file": "",
        "data_root": str(data_root),
        "summary": None,
        "error": None,
        "return_code": 0,
    }

    _write_job_record(jobs_dir / "job-001.json", job1)

    service = JobService(data_root=data_root)
    monkeypatch.setattr(jobs_router, "JOB_SERVICE", service)

    client = TestClient(app)
    resp = client.get("/jobs")

    assert resp.status_code == 200
    data = resp.json()
    assert "jobs" in data
    assert "count" in data
    assert data["count"] == 1
    assert len(data["jobs"]) == 1
    assert data["jobs"][0]["job_id"] == "job-001"
    assert data["jobs"][0]["job_type"] == "screen_time_analyze"
    assert data["jobs"][0]["state"] == "succeeded"


def test_list_jobs_endpoint_with_filters(tmp_path, monkeypatch) -> None:
    """Test GET /jobs endpoint with query filters."""
    data_root = tmp_path / "data"
    jobs_dir = data_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    job1 = {
        "job_id": "job-001",
        "ep_id": "test-s01e01",
        "job_type": "screen_time_analyze",
        "state": "succeeded",
        "started_at": "2025-01-01T10:00:00Z",
        "ended_at": "2025-01-01T10:01:00Z",
        "pid": 1234,
        "command": [],
        "requested": {},
        "progress_file": "",
        "data_root": str(data_root),
        "summary": None,
        "error": None,
        "return_code": 0,
    }
    job2 = {
        "job_id": "job-002",
        "ep_id": "test-s01e02",
        "job_type": "detect_track",
        "state": "running",
        "started_at": "2025-01-01T11:00:00Z",
        "ended_at": None,
        "pid": 5678,
        "command": [],
        "requested": {},
        "progress_file": "",
        "data_root": str(data_root),
        "summary": None,
        "error": None,
        "return_code": None,
    }

    _write_job_record(jobs_dir / "job-001.json", job1)
    _write_job_record(jobs_dir / "job-002.json", job2)

    service = JobService(data_root=data_root)
    monkeypatch.setattr(jobs_router, "JOB_SERVICE", service)

    client = TestClient(app)

    # Filter by ep_id
    resp = client.get("/jobs?ep_id=test-s01e01")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["jobs"][0]["job_id"] == "job-001"

    # Filter by job_type
    resp = client.get("/jobs?job_type=detect_track")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["jobs"][0]["job_id"] == "job-002"

    # Filter by both
    resp = client.get("/jobs?ep_id=test-s01e01&job_type=screen_time_analyze")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["jobs"][0]["job_id"] == "job-001"

    # No matches
    resp = client.get("/jobs?ep_id=non-existent")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 0
