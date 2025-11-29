"""Tests for execution mode (local vs redis) support in job APIs.

These tests verify:
1. Celery job endpoints accept execution_mode parameter
2. execution_mode="local" starts jobs in background and returns immediately with job_id
3. execution_mode="redis" (default) enqueues jobs via Celery
4. Grouping endpoints support execution_mode
5. Refresh similarity supports execution_mode
6. Local mode respects CPU thread limits for thermal safety
7. /celery_jobs/local endpoint returns running local jobs
"""

from __future__ import annotations

import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock


@pytest.fixture
def api_client():
    """Create a test client for the API."""
    from apps.api.main import app
    return TestClient(app)


class MockProcess:
    """Mock subprocess.Popen for testing local execution mode."""

    def __init__(self, returncode=0, stdout="Success\n", stderr=""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.pid = 12345
        self._poll_count = 0

    def poll(self):
        """Simulate process completion after first poll."""
        self._poll_count += 1
        if self._poll_count > 1:
            return self.returncode
        return None

    def communicate(self, timeout=None):
        return (self._stdout, self._stderr)

    def wait(self, timeout=None):
        return self.returncode


class TestExecutionModeParameter:
    """Tests for execution_mode parameter in job endpoints."""

    def test_detect_track_accepts_execution_mode_redis(self, api_client, monkeypatch):
        """POST /celery_jobs/detect_track accepts execution_mode='redis'."""
        captured: dict = {}

        def fake_delay(ep_id, options):
            captured["ep_id"] = ep_id
            captured["options"] = options
            return SimpleNamespace(id="job-redis-123")

        monkeypatch.setattr(
            "apps.api.routers.celery_jobs.run_detect_track_task",
            SimpleNamespace(delay=fake_delay),
        )
        monkeypatch.setattr("apps.api.routers.celery_jobs.check_active_job", lambda ep_id, op: None)

        payload = {
            "ep_id": "demo-s01e01",
            "stride": 6,
            "device": "cpu",
            "execution_mode": "redis",
        }

        resp = api_client.post("/celery_jobs/detect_track", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("job_id") == "job-redis-123"
        assert data.get("state") == "queued"
        assert data.get("execution_mode") == "redis"

    def test_detect_track_accepts_execution_mode_local(self, api_client, monkeypatch):
        """POST /celery_jobs/detect_track accepts execution_mode='local'.

        Local mode starts the job in background and returns immediately with job_id.
        The UI then polls /celery_jobs/local and /episodes/{ep_id}/progress for updates.
        """
        mock_process = MockProcess(returncode=0, stdout="Success\n")

        def fake_popen(*args, **kwargs):
            return mock_process

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: Path("/tmp"))

        payload = {
            "ep_id": "demo-s01e01",
            "stride": 6,
            "device": "cpu",
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/detect_track", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Local mode starts job and returns immediately
        assert data.get("state") == "started"
        assert data.get("execution_mode") == "local"
        assert "job_id" in data  # Has job_id for tracking
        assert data["job_id"].startswith("local-")
        assert "pid" in data  # Has process ID

    def test_faces_embed_accepts_execution_mode_local(self, api_client, monkeypatch):
        """POST /celery_jobs/faces_embed accepts execution_mode='local'."""
        mock_process = MockProcess(returncode=0, stdout="Success\n")

        def fake_popen(*args, **kwargs):
            return mock_process

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: Path("/tmp"))

        payload = {
            "ep_id": "demo-s01e01",
            "device": "cpu",
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/faces_embed", json=payload)
        data = resp.json()
        assert data.get("state") == "started"
        assert data.get("execution_mode") == "local"
        assert data["job_id"].startswith("local-")

    def test_cluster_accepts_execution_mode_local(self, api_client, monkeypatch):
        """POST /celery_jobs/cluster accepts execution_mode='local'."""
        mock_process = MockProcess(returncode=0, stdout="Success\n")

        def fake_popen(*args, **kwargs):
            return mock_process

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: Path("/tmp"))

        payload = {
            "ep_id": "demo-s01e01",
            "device": "cpu",
            "cluster_thresh": 0.7,
            "min_cluster_size": 2,
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/cluster", json=payload)
        data = resp.json()
        assert data.get("state") == "started"
        assert data.get("execution_mode") == "local"
        assert data["job_id"].startswith("local-")

    def test_execution_mode_defaults_to_redis(self, api_client, monkeypatch):
        """Execution mode should default to 'redis' when not specified."""
        captured: dict = {}

        def fake_delay(ep_id, options):
            captured["called"] = True
            return SimpleNamespace(id="job-default-123")

        monkeypatch.setattr(
            "apps.api.routers.celery_jobs.run_detect_track_task",
            SimpleNamespace(delay=fake_delay),
        )
        monkeypatch.setattr("apps.api.routers.celery_jobs.check_active_job", lambda ep_id, op: None)

        payload = {
            "ep_id": "demo-s01e01",
            "stride": 6,
            "device": "cpu",
            # No execution_mode specified - should default to redis
        }

        resp = api_client.post("/celery_jobs/detect_track", json=payload)
        assert resp.status_code == 200
        assert captured.get("called") is True  # Celery task was called (redis mode)


class TestGroupingExecutionMode:
    """Tests for execution_mode in grouping endpoints."""

    def test_batch_assign_async_local_mode(self, api_client, monkeypatch):
        """POST /episodes/{ep_id}/clusters/batch_assign_async with local mode."""
        # Mock grouping service
        def fake_batch_assign(ep_id, assignments):
            return {"assigned": 1, "failed": 0}

        monkeypatch.setattr(
            "apps.api.routers.grouping.grouping_service.batch_assign_clusters",
            fake_batch_assign,
        )

        payload = {
            "assignments": [{"cluster_id": "c1", "target_cast_id": "person1"}],
            "execution_mode": "local",
        }

        resp = api_client.post("/episodes/demo-s01e01/clusters/batch_assign_async", json=payload)
        data = resp.json()
        assert data.get("execution_mode") == "local"
        assert data.get("async") is False
        assert "local mode requested" in data.get("message", "")

    def test_group_clusters_async_local_mode(self, api_client, monkeypatch):
        """POST /episodes/{ep_id}/clusters/group_async with local mode."""
        # Mock grouping service
        def fake_group_auto(ep_id, progress_callback=None, protect_manual=True, facebank_first=True):
            if progress_callback:
                progress_callback("grouping", 1.0, "Done")
            return {"merged": 5, "total": 10}

        monkeypatch.setattr(
            "apps.api.routers.grouping.grouping_service.group_clusters_auto",
            fake_group_auto,
        )

        payload = {
            "strategy": "auto",
            "execution_mode": "local",
        }

        resp = api_client.post("/episodes/demo-s01e01/clusters/group_async", json=payload)
        data = resp.json()
        assert data.get("execution_mode") == "local"
        assert data.get("async") is False
        assert data.get("status") == "success"


class TestRefreshSimilarityExecutionMode:
    """Tests for execution_mode in refresh_similarity endpoint."""

    @pytest.mark.skip(reason="similarity_refresh module not yet implemented")
    def test_refresh_similarity_async_local_mode(self, api_client, monkeypatch):
        """POST /episodes/{ep_id}/refresh_similarity_async with local mode."""
        # Mock the refresh service
        def fake_refresh(ep_id):
            return {"refreshed": True}

        monkeypatch.setattr(
            "apps.api.services.similarity_refresh.refresh_similarity_indexes",
            fake_refresh,
        )

        resp = api_client.post(
            "/episodes/demo-s01e01/refresh_similarity_async",
            json={"execution_mode": "local"},
        )
        data = resp.json()
        assert data.get("status") == "completed"
        assert data.get("execution_mode") == "local"
        assert data.get("async") is False

    def test_refresh_similarity_async_redis_mode_no_celery(self, api_client, monkeypatch):
        """POST /episodes/{ep_id}/refresh_similarity_async with redis mode but no Celery."""
        # Force Celery to be unavailable
        monkeypatch.setattr(
            "apps.api.routers.episodes._check_celery_available",
            lambda: False,
        )

        resp = api_client.post(
            "/episodes/demo-s01e01/refresh_similarity_async",
            json={"execution_mode": "redis"},
        )
        data = resp.json()
        assert data.get("status") == "error"
        assert data.get("async") is False
        assert "unavailable" in data.get("message", "").lower()


class TestLocalJobsEndpoint:
    """Tests for /celery_jobs/local endpoint that lists running local jobs."""

    def test_list_local_jobs_empty(self, api_client, monkeypatch):
        """GET /celery_jobs/local returns empty list when no jobs running."""
        # Mock psutil to return no matching processes
        monkeypatch.setattr(
            "apps.api.routers.celery_jobs._detect_running_episode_processes",
            lambda: [],
        )

        resp = api_client.get("/celery_jobs/local")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("jobs") == []
        assert data.get("count") == 0

    def test_list_local_jobs_with_running_job(self, api_client, monkeypatch):
        """GET /celery_jobs/local returns running jobs."""
        mock_jobs = [
            {
                "job_id": "local-abc123",
                "ep_id": "test-s01e01",
                "operation": "detect_track",
                "pid": 12345,
                "state": "running",
            }
        ]

        # Mock the detection function
        monkeypatch.setattr(
            "apps.api.routers.celery_jobs._detect_running_episode_processes",
            lambda: mock_jobs,
        )
        # Clear registered jobs
        from apps.api.routers import celery_jobs
        celery_jobs._running_local_jobs.clear()

        resp = api_client.get("/celery_jobs/local")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("count") == 1
        assert len(data.get("jobs", [])) == 1
        assert data["jobs"][0]["ep_id"] == "test-s01e01"

    def test_list_local_jobs_filter_by_ep_id(self, api_client, monkeypatch):
        """GET /celery_jobs/local?ep_id=X filters by episode."""
        mock_jobs = [
            {"job_id": "local-1", "ep_id": "show-s01e01", "operation": "detect_track", "state": "running"},
            {"job_id": "local-2", "ep_id": "show-s01e02", "operation": "detect_track", "state": "running"},
        ]

        monkeypatch.setattr(
            "apps.api.routers.celery_jobs._detect_running_episode_processes",
            lambda: mock_jobs,
        )
        from apps.api.routers import celery_jobs
        celery_jobs._running_local_jobs.clear()

        resp = api_client.get("/celery_jobs/local?ep_id=show-s01e01")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("count") == 1
        assert data["jobs"][0]["ep_id"] == "show-s01e01"


class TestProfileAndSafetyInLocalMode:
    """Tests that performance profiles and safety are honored in local mode."""

    def test_local_mode_respects_cpu_threads(self, api_client, monkeypatch):
        """Local mode should respect cpu_threads setting."""
        captured_env: dict = {}
        mock_process = MockProcess(returncode=0, stdout="Success\n")

        def fake_popen(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return mock_process

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: Path("/tmp"))
        # Clear any existing registered jobs
        from apps.api.routers import celery_jobs
        celery_jobs._running_local_jobs.clear()

        payload = {
            "ep_id": "demo-s01e01",
            "stride": 6,
            "device": "cpu",
            "cpu_threads": 2,
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/detect_track", json=payload)
        assert resp.status_code == 200

        # Verify CPU thread limits were set in environment
        assert captured_env.get("SCREENALYTICS_MAX_CPU_THREADS") == "2"
        assert captured_env.get("OMP_NUM_THREADS") == "2"

        # Cleanup
        celery_jobs._running_local_jobs.clear()

    def test_local_mode_uses_process_group(self, api_client, monkeypatch):
        """Local mode should start subprocess with start_new_session=True for cleanup."""
        captured_kwargs: dict = {}
        mock_process = MockProcess(returncode=0, stdout="Success\n")

        def fake_popen(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return mock_process

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: Path("/tmp"))
        # Clear any existing registered jobs
        from apps.api.routers import celery_jobs
        celery_jobs._running_local_jobs.clear()

        payload = {
            "ep_id": "demo-s01e01",
            "stride": 6,
            "device": "cpu",
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/detect_track", json=payload)
        assert resp.status_code == 200

        # Verify process group is used (for killing child processes on cancel)
        assert captured_kwargs.get("start_new_session") is True

        # Cleanup
        celery_jobs._running_local_jobs.clear()


class TestLocalModeAlreadyRunning:
    """Tests for handling already-running local jobs."""

    def test_local_mode_rejects_duplicate_job(self, api_client, monkeypatch):
        """Local mode should reject request if same ep_id + operation already running."""
        mock_process = MockProcess(returncode=0, stdout="Success\n")

        def fake_popen(*args, **kwargs):
            return mock_process

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: Path("/tmp"))

        # Register a fake running job
        from apps.api.routers import celery_jobs
        celery_jobs._running_local_jobs["demo-s01e01:detect_track"] = {
            "job_id": "local-existing",
            "ep_id": "demo-s01e01",
            "operation": "detect_track",
            "pid": 99999,
            "state": "running",
        }

        # Mock psutil to say process is still running
        monkeypatch.setattr(
            "apps.api.routers.celery_jobs._get_running_local_job",
            lambda ep_id, op: celery_jobs._running_local_jobs.get(f"{ep_id}:{op}"),
        )

        payload = {
            "ep_id": "demo-s01e01",
            "stride": 6,
            "device": "cpu",
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/detect_track", json=payload)
        data = resp.json()

        # Should indicate job is already running
        assert data.get("state") == "already_running"
        assert data.get("job_id") == "local-existing"

        # Cleanup
        celery_jobs._running_local_jobs.clear()
