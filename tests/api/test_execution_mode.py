"""Tests for execution mode (local vs redis) support in job APIs.

These tests verify:
1. Celery job endpoints accept execution_mode parameter
2. execution_mode="local" runs jobs synchronously and returns completion status
3. execution_mode="redis" (default) enqueues jobs and returns job_id
4. Grouping endpoints support execution_mode
5. Refresh similarity supports execution_mode
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace


@pytest.fixture
def api_client():
    """Create a test client for the API."""
    from apps.api.main import app
    return TestClient(app)


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
        """POST /celery_jobs/detect_track accepts execution_mode='local'."""
        # Mock the subprocess run to avoid actually running a job
        import subprocess
        original_run = subprocess.run

        def fake_run(command, **kwargs):
            return SimpleNamespace(returncode=0, stdout="Success", stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: "/tmp")

        payload = {
            "ep_id": "demo-s01e01",
            "stride": 6,
            "device": "cpu",
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/detect_track", json=payload)
        # Should return completion status (not job_id for polling)
        data = resp.json()
        assert data.get("status") == "completed"
        assert data.get("execution_mode") == "local"
        assert "job_id" not in data

    def test_faces_embed_accepts_execution_mode_local(self, api_client, monkeypatch):
        """POST /celery_jobs/faces_embed accepts execution_mode='local'."""
        import subprocess

        def fake_run(command, **kwargs):
            return SimpleNamespace(returncode=0, stdout="Success", stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: "/tmp")

        payload = {
            "ep_id": "demo-s01e01",
            "device": "cpu",
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/faces_embed", json=payload)
        data = resp.json()
        assert data.get("status") == "completed"
        assert data.get("execution_mode") == "local"

    def test_cluster_accepts_execution_mode_local(self, api_client, monkeypatch):
        """POST /celery_jobs/cluster accepts execution_mode='local'."""
        import subprocess

        def fake_run(command, **kwargs):
            return SimpleNamespace(returncode=0, stdout="Success", stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: "/tmp")

        payload = {
            "ep_id": "demo-s01e01",
            "device": "cpu",
            "cluster_thresh": 0.7,
            "min_cluster_size": 2,
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/cluster", json=payload)
        data = resp.json()
        assert data.get("status") == "completed"
        assert data.get("execution_mode") == "local"

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


class TestLocalModeErrorHandling:
    """Tests for error handling in local execution mode."""

    def test_detect_track_local_error_returns_500(self, api_client, monkeypatch):
        """Local mode should return 500 on pipeline failure."""
        import subprocess

        def fake_run(command, **kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="Pipeline failed")

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: "/tmp")

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
        assert data.get("execution_mode") == "local"
        assert data.get("error") == "Pipeline failed"

    def test_detect_track_local_timeout_returns_error(self, api_client, monkeypatch):
        """Local mode should handle timeout gracefully."""
        import subprocess

        def fake_run(command, **kwargs):
            raise subprocess.TimeoutExpired(cmd=command, timeout=3600)

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: "/tmp")

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
        assert "timed out" in data.get("error", "").lower()


class TestLocalModeSynchronousBehavior:
    """Tests verifying local mode is truly synchronous with no job tracking."""

    def test_local_mode_returns_logs_in_response(self, api_client, monkeypatch):
        """Local mode should return logs directly in the response."""
        import subprocess

        def fake_run(command, **kwargs):
            return SimpleNamespace(
                returncode=0,
                stdout="Loading models...\nProcessing frames...\nCompleted!",
                stderr="",
            )

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: "/tmp")

        payload = {
            "ep_id": "demo-s01e01",
            "stride": 6,
            "device": "cpu",
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/detect_track", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Local mode returns logs directly - no polling needed
        assert "logs" in data
        assert isinstance(data["logs"], list)
        assert len(data["logs"]) > 0
        assert "Loading models..." in data["logs"]
        assert "Processing frames..." in data["logs"]

    def test_local_mode_returns_elapsed_time(self, api_client, monkeypatch):
        """Local mode should return elapsed time in the response."""
        import subprocess

        def fake_run(command, **kwargs):
            return SimpleNamespace(returncode=0, stdout="Done", stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: "/tmp")

        payload = {
            "ep_id": "demo-s01e01",
            "stride": 6,
            "device": "cpu",
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/detect_track", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Local mode includes elapsed time
        assert "elapsed_seconds" in data
        assert isinstance(data["elapsed_seconds"], (int, float))

    def test_local_mode_no_job_id(self, api_client, monkeypatch):
        """Local mode should not return a job_id - it's synchronous."""
        import subprocess

        def fake_run(command, **kwargs):
            return SimpleNamespace(returncode=0, stdout="Done", stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: "/tmp")

        payload = {
            "ep_id": "demo-s01e01",
            "stride": 6,
            "device": "cpu",
            "execution_mode": "local",
        }

        resp = api_client.post("/celery_jobs/detect_track", json=payload)
        data = resp.json()

        # Local mode is synchronous - no job ID needed
        assert "job_id" not in data
        assert data.get("status") == "completed"
        assert data.get("execution_mode") == "local"


class TestProfileAndSafetyInLocalMode:
    """Tests that performance profiles and safety are honored in local mode."""

    def test_local_mode_respects_cpu_threads(self, api_client, monkeypatch):
        """Local mode should respect cpu_threads setting."""
        import subprocess
        captured_env: dict = {}

        def fake_run(command, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return SimpleNamespace(returncode=0, stdout="Success", stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("apps.api.routers.celery_jobs._find_project_root", lambda: "/tmp")

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
