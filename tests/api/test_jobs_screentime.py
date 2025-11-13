from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import jobs as jobs_router
from apps.api.services.episodes import EpisodeStore
from apps.api.services.jobs import JobService
from py_screenalytics.artifacts import ensure_dirs, get_path


def test_start_screen_time_job_happy_path(tmp_path, monkeypatch):
    """Test that JobService.start_screen_time_job builds correct command."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "test-s01e01"
    show_id = "TEST"

    ensure_dirs(ep_id)
    manifests_dir = data_root / "manifests" / ep_id
    manifests_dir.mkdir(parents=True, exist_ok=True)

    faces_path = manifests_dir / "faces.jsonl"
    faces_path.write_text(json.dumps({"track_id": 1}) + "\n", encoding="utf-8")

    tracks_path = get_path(ep_id, "tracks")
    tracks_path.write_text(json.dumps({"track_id": 1}) + "\n", encoding="utf-8")

    identities_path = manifests_dir / "identities.json"
    identities_path.write_text(json.dumps({"identities": []}), encoding="utf-8")

    shows_dir = data_root / "shows" / show_id
    shows_dir.mkdir(parents=True, exist_ok=True)
    people_path = shows_dir / "people.json"
    people_path.write_text(json.dumps({"people": []}), encoding="utf-8")

    jobs_router.EPISODE_STORE = EpisodeStore()
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="test", season=1, episode=1)

    service = JobService(data_root=data_root)
    captured: dict = {}

    def _fake_launch(**kwargs):
        captured.update(kwargs)
        return {
            "job_id": "test-job",
            "ep_id": ep_id,
            "state": "running",
            "started_at": "now",
            "job_type": kwargs["job_type"],
        }

    monkeypatch.setattr(service, "_launch_job", _fake_launch)

    job = service.start_screen_time_job(
        ep_id=ep_id,
        quality_min=0.8,
        gap_tolerance_s=1.0,
        use_video_decode=False,
    )

    assert job["job_type"] == "screen_time_analyze"
    assert captured["requested"]["quality_min"] == 0.8
    assert captured["requested"]["gap_tolerance_s"] == 1.0
    assert captured["requested"]["use_video_decode"] is False

    command = captured["command"]
    assert "--ep-id" in command
    assert ep_id in command
    assert "--quality-min" in command
    assert "--gap-tolerance-s" in command
    assert "--use-video-decode" in command
    assert "false" in command  # use_video_decode=False


def test_start_screen_time_job_default_params(tmp_path, monkeypatch):
    """Test that start_screen_time_job works with default parameters."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "test-s01e02"
    show_id = "TEST"

    ensure_dirs(ep_id)
    manifests_dir = data_root / "manifests" / ep_id
    manifests_dir.mkdir(parents=True, exist_ok=True)

    faces_path = manifests_dir / "faces.jsonl"
    faces_path.write_text(json.dumps({"track_id": 1}) + "\n", encoding="utf-8")

    tracks_path = get_path(ep_id, "tracks")
    tracks_path.write_text(json.dumps({"track_id": 1}) + "\n", encoding="utf-8")

    identities_path = manifests_dir / "identities.json"
    identities_path.write_text(json.dumps({"identities": []}), encoding="utf-8")

    shows_dir = data_root / "shows" / show_id
    shows_dir.mkdir(parents=True, exist_ok=True)
    people_path = shows_dir / "people.json"
    people_path.write_text(json.dumps({"people": []}), encoding="utf-8")

    jobs_router.EPISODE_STORE = EpisodeStore()
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="test", season=1, episode=2)

    service = JobService(data_root=data_root)
    captured: dict = {}

    def _fake_launch(**kwargs):
        captured.update(kwargs)
        return {
            "job_id": "test-job-2",
            "ep_id": ep_id,
            "state": "running",
            "job_type": kwargs["job_type"],
        }

    monkeypatch.setattr(service, "_launch_job", _fake_launch)

    job = service.start_screen_time_job(ep_id=ep_id)

    assert job["job_type"] == "screen_time_analyze"
    assert captured["requested"] == {}

    command = captured["command"]
    assert "--quality-min" not in command
    assert "--gap-tolerance-s" not in command


def test_start_screen_time_job_missing_faces(tmp_path, monkeypatch):
    """Test that start_screen_time_job fails when faces.jsonl is missing."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "test-s01e03"
    show_id = "TEST"

    ensure_dirs(ep_id)
    manifests_dir = data_root / "manifests" / ep_id
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Skip faces.jsonl
    tracks_path = get_path(ep_id, "tracks")
    tracks_path.write_text(json.dumps({"track_id": 1}) + "\n", encoding="utf-8")

    identities_path = manifests_dir / "identities.json"
    identities_path.write_text(json.dumps({"identities": []}), encoding="utf-8")

    shows_dir = data_root / "shows" / show_id
    shows_dir.mkdir(parents=True, exist_ok=True)
    people_path = shows_dir / "people.json"
    people_path.write_text(json.dumps({"people": []}), encoding="utf-8")

    service = JobService(data_root=data_root)

    with pytest.raises(FileNotFoundError, match="faces.jsonl"):
        service.start_screen_time_job(ep_id=ep_id)


def test_start_screen_time_job_missing_tracks(tmp_path, monkeypatch):
    """Test that start_screen_time_job fails when tracks.jsonl is missing."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "test-s01e04"
    show_id = "TEST"

    ensure_dirs(ep_id)
    manifests_dir = data_root / "manifests" / ep_id
    manifests_dir.mkdir(parents=True, exist_ok=True)

    faces_path = manifests_dir / "faces.jsonl"
    faces_path.write_text(json.dumps({"track_id": 1}) + "\n", encoding="utf-8")

    # Skip tracks.jsonl
    identities_path = manifests_dir / "identities.json"
    identities_path.write_text(json.dumps({"identities": []}), encoding="utf-8")

    shows_dir = data_root / "shows" / show_id
    shows_dir.mkdir(parents=True, exist_ok=True)
    people_path = shows_dir / "people.json"
    people_path.write_text(json.dumps({"people": []}), encoding="utf-8")

    service = JobService(data_root=data_root)

    with pytest.raises(FileNotFoundError, match="tracks.jsonl"):
        service.start_screen_time_job(ep_id=ep_id)


def test_start_screen_time_job_missing_people(tmp_path, monkeypatch):
    """Test that start_screen_time_job fails when people.json is missing."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "test-s01e05"

    ensure_dirs(ep_id)
    manifests_dir = data_root / "manifests" / ep_id
    manifests_dir.mkdir(parents=True, exist_ok=True)

    faces_path = manifests_dir / "faces.jsonl"
    faces_path.write_text(json.dumps({"track_id": 1}) + "\n", encoding="utf-8")

    tracks_path = get_path(ep_id, "tracks")
    tracks_path.write_text(json.dumps({"track_id": 1}) + "\n", encoding="utf-8")

    identities_path = manifests_dir / "identities.json"
    identities_path.write_text(json.dumps({"identities": []}), encoding="utf-8")

    # Don't create people.json

    service = JobService(data_root=data_root)

    with pytest.raises(FileNotFoundError, match="people.json"):
        service.start_screen_time_job(ep_id=ep_id)


def test_analyze_screen_time_endpoint(tmp_path, monkeypatch):
    """Test POST /jobs/screen_time/analyze endpoint returns correct job record."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "test-s01e06"
    show_id = "TEST"

    ensure_dirs(ep_id)
    manifests_dir = data_root / "manifests" / ep_id
    manifests_dir.mkdir(parents=True, exist_ok=True)

    faces_path = manifests_dir / "faces.jsonl"
    faces_path.write_text(json.dumps({"track_id": 1}) + "\n", encoding="utf-8")

    tracks_path = get_path(ep_id, "tracks")
    tracks_path.write_text(json.dumps({"track_id": 1}) + "\n", encoding="utf-8")

    identities_path = manifests_dir / "identities.json"
    identities_path.write_text(json.dumps({"identities": []}), encoding="utf-8")

    shows_dir = data_root / "shows" / show_id
    shows_dir.mkdir(parents=True, exist_ok=True)
    people_path = shows_dir / "people.json"
    people_path.write_text(json.dumps({"people": []}), encoding="utf-8")

    jobs_router.EPISODE_STORE = EpisodeStore()
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="test", season=1, episode=6)

    service = JobService(data_root=data_root)

    def _fake_launch(**kwargs):
        return {
            "job_id": "endpoint-test-job",
            "ep_id": ep_id,
            "state": "running",
            "started_at": "2025-01-01T00:00:00Z",
            "progress_file": "/tmp/progress.json",
            "requested": kwargs["requested"],
            "job_type": kwargs["job_type"],
        }

    monkeypatch.setattr(service, "_launch_job", _fake_launch)
    monkeypatch.setattr(jobs_router, "JOB_SERVICE", service)

    client = TestClient(app)
    resp = client.post(
        "/jobs/screen_time/analyze",
        json={
            "ep_id": ep_id,
            "quality_min": 0.75,
            "gap_tolerance_s": 0.8,
        }
    )

    assert resp.status_code == 200
    data = resp.json()

    assert data["ep_id"] == ep_id
    assert data["state"] == "running"
    assert "job_id" in data
    assert "started_at" in data


def test_analyze_screen_time_endpoint_missing_artifacts(tmp_path, monkeypatch):
    """Test POST /jobs/screen_time/analyze returns 400 when artifacts are missing."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "test-s01e07"

    ensure_dirs(ep_id)

    jobs_router.EPISODE_STORE = EpisodeStore()
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="test", season=1, episode=7)

    service = JobService(data_root=data_root)
    monkeypatch.setattr(jobs_router, "JOB_SERVICE", service)

    client = TestClient(app)
    resp = client.post(
        "/jobs/screen_time/analyze",
        json={"ep_id": ep_id}
    )

    assert resp.status_code == 400
    assert "detail" in resp.json()


def test_analyze_screen_time_endpoint_minimal_payload(tmp_path, monkeypatch):
    """Test POST /jobs/screen_time/analyze works with only ep_id (no overrides)."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "test-s01e08"
    show_id = "TEST"

    ensure_dirs(ep_id)
    manifests_dir = data_root / "manifests" / ep_id
    manifests_dir.mkdir(parents=True, exist_ok=True)

    faces_path = manifests_dir / "faces.jsonl"
    faces_path.write_text(json.dumps({"track_id": 1}) + "\n", encoding="utf-8")

    tracks_path = get_path(ep_id, "tracks")
    tracks_path.write_text(json.dumps({"track_id": 1}) + "\n", encoding="utf-8")

    identities_path = manifests_dir / "identities.json"
    identities_path.write_text(json.dumps({"identities": []}), encoding="utf-8")

    shows_dir = data_root / "shows" / show_id
    shows_dir.mkdir(parents=True, exist_ok=True)
    people_path = shows_dir / "people.json"
    people_path.write_text(json.dumps({"people": []}), encoding="utf-8")

    jobs_router.EPISODE_STORE = EpisodeStore()
    jobs_router.EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug="test", season=1, episode=8)

    service = JobService(data_root=data_root)

    def _fake_launch(**kwargs):
        return {
            "job_id": "minimal-test-job",
            "ep_id": ep_id,
            "state": "running",
            "started_at": "2025-01-01T00:00:00Z",
            "job_type": kwargs["job_type"],
        }

    monkeypatch.setattr(service, "_launch_job", _fake_launch)
    monkeypatch.setattr(jobs_router, "JOB_SERVICE", service)

    client = TestClient(app)
    resp = client.post(
        "/jobs/screen_time/analyze",
        json={"ep_id": ep_id}
    )

    assert resp.status_code == 200
    data = resp.json()

    assert data["ep_id"] == ep_id
    assert data["state"] == "running"
    assert "job_id" in data
    assert "started_at" in data
