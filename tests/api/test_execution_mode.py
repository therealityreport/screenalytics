"""Tests for execution mode (local vs redis) support in job APIs.

These tests verify:
1. Celery job endpoints accept execution_mode parameter
2. execution_mode="local" runs jobs synchronously (blocking) and returns status directly
3. execution_mode="redis" (default) enqueues jobs via Celery
4. Local mode does NOT return job_id (truly synchronous)
5. Local mode respects CPU thread limits for thermal safety (hard cap at 2)
6. Local mode prevents duplicate concurrent runs
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

    def test_detect_track_local_mode_synchronous(self, api_client, monkeypatch):
        """POST /celery_jobs/detect_track with execution_mode='local' runs synchronously.

        Local mode should:
        - NOT return a job_id
        - Return status directly (completed/error)
        - Return logs from the subprocess
        - Include elapsed_seconds
        """
        mock_process = MockProcess(returncode=0, stdout="Line 1\nLine 2\nSuccess\n")

        def fake_popen(*args, **kwargs):
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
        data = resp.json()

        # Local mode returns synchronous response - no job_id
        assert "job_id" not in data
        assert data.get("status") == "completed"
        assert data.get("execution_mode") == "local"
        assert "logs" in data
        assert isinstance(data["logs"], list)
        assert "elapsed_seconds" in data

        # Cleanup
        celery_jobs._running_local_jobs.clear()

    def test_faces_embed_local_mode_synchronous(self, api_client, monkeypatch):
        """POST /celery_jobs/faces_embed with execution_mode='local' runs synchronously."""
        mock_process = MockProcess(returncode=0, stdout="Success\n")

        def fake_popen(*args, **kwargs):
            return mock_process

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: Path("/tmp"))
        from apps.api.routers import celery_jobs
        celery_jobs._running_local_jobs.clear()

        payload = {
            "ep_id": "demo-s01e01",
            "device": "cpu",
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/faces_embed", json=payload)
        data = resp.json()

        # Local mode returns synchronous response - no job_id
        assert "job_id" not in data
        assert data.get("status") == "completed"
        assert data.get("execution_mode") == "local"

        celery_jobs._running_local_jobs.clear()

    def test_cluster_local_mode_synchronous(self, api_client, monkeypatch):
        """POST /celery_jobs/cluster with execution_mode='local' runs synchronously."""
        mock_process = MockProcess(returncode=0, stdout="Success\n")

        def fake_popen(*args, **kwargs):
            return mock_process

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: Path("/tmp"))
        from apps.api.routers import celery_jobs
        celery_jobs._running_local_jobs.clear()

        payload = {
            "ep_id": "demo-s01e01",
            "device": "cpu",
            "cluster_thresh": 0.7,
            "min_cluster_size": 2,
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/cluster", json=payload)
        data = resp.json()

        # Local mode returns synchronous response - no job_id
        assert "job_id" not in data
        assert data.get("status") == "completed"
        assert data.get("execution_mode") == "local"

        celery_jobs._running_local_jobs.clear()

    def test_local_mode_returns_error_on_failure(self, api_client, monkeypatch):
        """Local mode should return status='error' when subprocess fails."""
        mock_process = MockProcess(returncode=1, stdout="", stderr="Something went wrong")

        def fake_popen(*args, **kwargs):
            return mock_process

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: Path("/tmp"))
        from apps.api.routers import celery_jobs
        celery_jobs._running_local_jobs.clear()

        payload = {
            "ep_id": "demo-s01e01",
            "stride": 6,
            "device": "cpu",
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/detect_track", json=payload)
        assert resp.status_code == 500
        data = resp.json()

        assert data.get("status") == "error"
        assert "error" in data
        assert data.get("return_code") == 1

        celery_jobs._running_local_jobs.clear()

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
        from apps.api.routers import celery_jobs
        celery_jobs._running_local_jobs.clear()

        resp = api_client.get("/celery_jobs/local")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("jobs") == []
        assert data.get("count") == 0

    def test_list_local_jobs_with_running_job(self, api_client, monkeypatch):
        """GET /celery_jobs/local returns running jobs."""
        mock_jobs = [
            {
                "job_id": "orphan-12345",
                "ep_id": "test-s01e01",
                "operation": "detect_track",
                "pid": 12345,
                "state": "running",
                "source": "detected",
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
            {"job_id": "orphan-1", "ep_id": "show-s01e01", "operation": "detect_track", "state": "running", "source": "detected"},
            {"job_id": "orphan-2", "ep_id": "show-s01e02", "operation": "detect_track", "state": "running", "source": "detected"},
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


class TestLocalModeThermalSafety:
    """Tests that local mode enforces thermal safety limits."""

    def test_local_mode_caps_cpu_threads_at_2(self, api_client, monkeypatch):
        """Local mode should hard-cap CPU threads at 2 regardless of profile."""
        captured_env: dict = {}
        mock_process = MockProcess(returncode=0, stdout="Success\n")

        def fake_popen(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return mock_process

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: Path("/tmp"))
        from apps.api.routers import celery_jobs
        celery_jobs._running_local_jobs.clear()

        # Request 8 threads - should be capped at 2
        payload = {
            "ep_id": "demo-s01e01",
            "stride": 6,
            "device": "cpu",
            "cpu_threads": 8,  # Request more threads
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/detect_track", json=payload)
        assert resp.status_code == 200

        # Verify CPU thread limits were capped at 2 (local mode hard cap)
        assert captured_env.get("SCREENALYTICS_MAX_CPU_THREADS") == "2"
        assert captured_env.get("OMP_NUM_THREADS") == "2"
        assert captured_env.get("MKL_NUM_THREADS") == "2"

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
        celery_jobs._running_local_jobs.clear()
        celery_jobs._running_local_jobs["demo-s01e01::detect_track"] = {
            "ep_id": "demo-s01e01",
            "operation": "detect_track",
            "pid": 99999,
        }

        payload = {
            "ep_id": "demo-s01e01",
            "stride": 6,
            "device": "cpu",
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/detect_track", json=payload)
        data = resp.json()

        # Should indicate job is already running with error status
        assert data.get("status") == "error"
        assert "already running" in data.get("error", "").lower()

        # Cleanup
        celery_jobs._running_local_jobs.clear()


class TestLocalModeResponseStructure:
    """Tests that verify the response structure for local mode."""

    def test_local_mode_includes_device_and_profile(self, api_client, monkeypatch):
        """Local mode response should include device and profile info."""
        mock_process = MockProcess(returncode=0, stdout="Success\n")

        def fake_popen(*args, **kwargs):
            return mock_process

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: Path("/tmp"))
        from apps.api.routers import celery_jobs
        celery_jobs._running_local_jobs.clear()

        payload = {
            "ep_id": "demo-s01e01",
            "stride": 6,
            "device": "coreml",
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/detect_track", json=payload)
        data = resp.json()

        assert data.get("status") == "completed"
        assert data.get("device") == "coreml"
        assert data.get("profile") is not None  # Should have resolved profile
        assert data.get("cpu_threads") == 2  # Capped at 2 for local mode

        celery_jobs._running_local_jobs.clear()

    def test_local_mode_logs_contain_context(self, api_client, monkeypatch):
        """Local mode logs should include execution context."""
        mock_process = MockProcess(returncode=0, stdout="Processing...\nDone.\n")

        def fake_popen(*args, **kwargs):
            return mock_process

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: Path("/tmp"))
        from apps.api.routers import celery_jobs
        celery_jobs._running_local_jobs.clear()

        payload = {
            "ep_id": "demo-s01e01",
            "stride": 6,
            "device": "cpu",
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/detect_track", json=payload)
        data = resp.json()

        logs = data.get("logs", [])
        assert len(logs) > 0

        # Should have [LOCAL MODE] markers and context
        log_text = "\n".join(logs)
        assert "[LOCAL MODE]" in log_text
        assert "CPU threads capped at 2" in log_text

        celery_jobs._running_local_jobs.clear()
