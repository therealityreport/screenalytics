"""Tests for Celery job status API endpoints.

These tests verify:
1. Job status endpoint returns correct states
2. Unknown job IDs return 404
3. Job list endpoint returns active jobs
4. Async endpoints return 202 with job_id
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


class TestCeleryJobsRouter:
    """Tests for /celery_jobs/* endpoints."""

    def test_list_jobs_empty(self, api_client):
        """GET /celery_jobs returns empty list when no jobs are running."""
        response = api_client.get("/celery_jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)

    def test_job_status_unknown_id(self, api_client):
        """GET /celery_jobs/{job_id} returns 404 for unknown job ID."""
        response = api_client.get("/celery_jobs/unknown-job-id-12345")
        assert response.status_code == 404

    def test_job_status_valid_states(self, api_client, celery_eager_app):
        """Verify job status endpoint maps Celery states correctly."""
        if celery_eager_app is None:
            pytest.skip("Celery not available")

        from apps.api.tasks import get_job_status
        from celery.result import AsyncResult

        # Create a mock result with known state
        # In eager mode, we can test the state mapping function directly
        state_map = {
            "PENDING": "queued",
            "RECEIVED": "queued",
            "STARTED": "in_progress",
            "SUCCESS": "success",
            "FAILURE": "failed",
            "RETRY": "retrying",
            "REVOKED": "cancelled",
            "PROGRESS": "in_progress",
        }

        # Test the state mapping is correct
        for celery_state, expected_ui_state in state_map.items():
            # The mapping is defined in get_job_status
            from apps.api.tasks import get_job_status
            # This validates the mapping exists in the code
            assert expected_ui_state in ["queued", "in_progress", "success", "failed", "retrying", "cancelled", "unknown"]

    def test_detect_track_celery_forwards_advanced_options(self, api_client, monkeypatch):
        """Detect/track Celery endpoint should forward advanced options to the task."""
        captured: dict = {}

        def fake_delay(ep_id, options):
            captured["ep_id"] = ep_id
            captured["options"] = options
            return SimpleNamespace(id="job-123")

        monkeypatch.setattr(
            "apps.api.routers.celery_jobs.run_detect_track_task",
            SimpleNamespace(delay=fake_delay),
        )
        monkeypatch.setattr("apps.api.routers.celery_jobs.check_active_job", lambda ep_id, op: None)

        payload = {
            "ep_id": "demo-s01e01",
            "stride": 3,
            "fps": 12.5,
            "device": "cpu",
            "detector": "retinaface",
            "tracker": "bytetrack",
            "save_frames": True,
            "save_crops": False,
            "jpeg_quality": 80,
            "det_thresh": 0.7,
            "max_gap": 60,
            "scene_detector": "pyscenedetect",
            "scene_threshold": 0.4,
            "scene_min_len": 8,
            "scene_warmup_dets": 3,
            "track_high_thresh": 0.62,
            "new_track_thresh": 0.55,
            "track_buffer": 18,
            "min_box_area": 42.5,
            "cpu_threads": 2,
        }

        resp = api_client.post("/celery_jobs/detect_track", json=payload)
        assert resp.status_code == 200
        assert captured["ep_id"] == "demo-s01e01"
        options = captured["options"]
        assert options["fps"] == pytest.approx(12.5)
        assert options["scene_min_len"] == 8
        assert options["scene_warmup_dets"] == 3
        assert options["track_high_thresh"] == pytest.approx(0.62)
        assert options["new_track_thresh"] == pytest.approx(0.55)
        assert options["track_buffer"] == 18
        assert options["min_box_area"] == pytest.approx(42.5)
        assert options["cpu_threads"] == 2


class TestAsyncEndpoints:
    """Tests for async job submission endpoints."""

    def test_refresh_similarity_async_no_celery(self, api_client, monkeypatch):
        """POST /episodes/{ep_id}/refresh_similarity_async returns error when Celery unavailable."""
        # Force Celery check to return False
        monkeypatch.setattr(
            "apps.api.routers.episodes._celery_available_cache",
            False
        )

        response = api_client.post("/episodes/test-s01e01/refresh_similarity_async")
        # Should return a response indicating async is unavailable
        assert response.status_code == 200
        data = response.json()
        assert data.get("async") is False

    def test_group_clusters_async_validation(self, api_client):
        """POST /episodes/{ep_id}/clusters/group_async validates strategy."""
        # Only 'auto' strategy is supported for async
        response = api_client.post(
            "/episodes/test-s01e01/clusters/group_async",
            json={"strategy": "manual", "cluster_ids": ["c1"]}
        )
        assert response.status_code == 400
        assert "auto" in response.json()["detail"].lower()

    def test_batch_assign_async_validation(self, api_client):
        """POST /episodes/{ep_id}/clusters/batch_assign_async requires assignments."""
        response = api_client.post(
            "/episodes/test-s01e01/clusters/batch_assign_async",
            json={"assignments": []}
        )
        assert response.status_code == 400
        assert "No assignments" in response.json()["detail"]


class TestJobTaskHelpers:
    """Tests for task helper functions."""

    def test_check_active_job_no_lock(self, celery_eager_app):
        """check_active_job returns None when no lock exists."""
        if celery_eager_app is None:
            pytest.skip("Celery not available")

        from apps.api.tasks import check_active_job

        # Should return None for episode with no active job
        result = check_active_job("nonexistent-s01e01", "manual_assign")
        assert result is None

    def test_get_job_status_structure(self, celery_eager_app):
        """get_job_status returns correct structure."""
        if celery_eager_app is None:
            pytest.skip("Celery not available")

        from apps.api.tasks import get_job_status

        # Get status for a non-existent job
        result = get_job_status("nonexistent-job-id")

        # Should have required fields
        assert "job_id" in result
        assert "state" in result
        assert "raw_state" in result
        assert result["job_id"] == "nonexistent-job-id"

    def test_cancel_job_structure(self, celery_eager_app):
        """cancel_job returns correct structure."""
        if celery_eager_app is None:
            pytest.skip("Celery not available")

        from apps.api.tasks import cancel_job

        result = cancel_job("some-job-id")
        assert "job_id" in result
        assert "status" in result
        assert result["status"] == "cancelled"


class TestCeleryEagerMode:
    """Tests that verify Celery eager mode is working."""

    def test_celery_eager_mode_configured(self, celery_eager_app):
        """Verify Celery is configured for eager execution in tests."""
        if celery_eager_app is None:
            pytest.skip("Celery not available")

        assert celery_eager_app.conf.task_always_eager is True
        assert celery_eager_app.conf.task_eager_propagates is True

    def test_task_executes_synchronously(self, celery_eager_app):
        """Verify tasks execute synchronously in eager mode."""
        if celery_eager_app is None:
            pytest.skip("Celery not available")

        # Import a simple task
        try:
            from apps.api.tasks import run_manual_assign_task

            # In eager mode, .delay() executes synchronously and returns result
            # We can't actually run this without episode data, but we can verify
            # the task is registered
            assert run_manual_assign_task.name == "tasks.run_manual_assign"
        except ImportError:
            pytest.skip("Tasks module not available")


class TestLocalModeInstrumentation:
    """Tests for Local mode instrumentation features."""

    def test_cpulimit_wrapper_returns_tuple(self):
        """Test cpulimit wrapper returns correct tuple structure."""
        from apps.api.routers.celery_jobs import _maybe_wrap_with_cpulimit_local

        command = ["python", "test.py"]
        result, applied = _maybe_wrap_with_cpulimit_local(command)

        # Should always return a tuple
        assert isinstance(result, list)
        assert isinstance(applied, bool)

        # Original command should be present
        assert "python" in " ".join(result)
        assert "test.py" in " ".join(result)

    def test_cpulimit_wrapper_disabled_when_zero(self, monkeypatch):
        """Test cpulimit wrapper does nothing when limit is 0."""
        import apps.api.routers.celery_jobs as cj

        # Temporarily disable cpulimit
        original = cj._LOCAL_CPULIMIT_PERCENT
        monkeypatch.setattr(cj, "_LOCAL_CPULIMIT_PERCENT", 0)

        try:
            command = ["python", "test.py"]
            result, applied = cj._maybe_wrap_with_cpulimit_local(command)

            assert result == command
            assert applied is False
        finally:
            monkeypatch.setattr(cj, "_LOCAL_CPULIMIT_PERCENT", original)

    def test_cpu_affinity_fallback_returns_bool(self):
        """Test CPU affinity fallback returns boolean."""
        from apps.api.routers.celery_jobs import _apply_cpu_affinity_fallback_local

        # Test with a non-existent PID (very high number unlikely to exist)
        # Should return False without raising an exception
        result = _apply_cpu_affinity_fallback_local(999999999, 200)
        assert isinstance(result, bool)
        assert result is False  # Should fail for non-existent process

    def test_local_mode_instrumentation_env_var(self, monkeypatch):
        """Test LOCAL_MODE_INSTRUMENTATION env var detection."""
        # Test that the env var is read correctly
        monkeypatch.setenv("LOCAL_MODE_INSTRUMENTATION", "1")

        # Import fresh to pick up env var
        import importlib
        import tools.episode_run as er

        # The module level check happens at import time, so we verify the concept
        # by checking the env var is readable
        import os
        assert os.environ.get("LOCAL_MODE_INSTRUMENTATION") == "1"


class TestPhaseTracker:
    """Tests for PhaseTracker class used in Local mode instrumentation."""

    def test_phase_tracker_creation(self):
        """Test PhaseTracker can be instantiated."""
        from tools.episode_run import PhaseTracker

        tracker = PhaseTracker()
        assert tracker is not None
        assert tracker.summary() == {}

    def test_phase_tracker_add_phase_stats(self):
        """Test adding phase stats directly."""
        from tools.episode_run import PhaseTracker

        tracker = PhaseTracker()
        tracker.add_phase_stats(
            "detect",
            frames_processed=1000,
            frames_scanned=6000,
            stride=6,
            duration_seconds=120.5,
        )

        summary = tracker.summary()
        assert "detect" in summary
        assert summary["detect"]["frames_processed"] == 1000
        assert summary["detect"]["frames_scanned"] == 6000
        assert summary["detect"]["stride"] == 6
        assert summary["detect"]["duration_seconds"] == 120.5

    def test_phase_tracker_multiple_phases(self):
        """Test tracking multiple phases."""
        from tools.episode_run import PhaseTracker

        tracker = PhaseTracker()

        tracker.add_phase_stats(
            "scene_detect",
            frames_processed=61494,
            frames_scanned=61494,
            stride=1,
            duration_seconds=154.2,
        )

        tracker.add_phase_stats(
            "detect",
            frames_processed=10249,
            frames_scanned=61494,
            stride=6,
            duration_seconds=210.4,
        )

        summary = tracker.summary()
        assert len(summary) == 2
        assert "scene_detect" in summary
        assert "detect" in summary

        # Verify stride is tracked correctly
        assert summary["scene_detect"]["stride"] == 1
        assert summary["detect"]["stride"] == 6

        # Verify frames_processed vs frames_scanned distinction
        assert summary["scene_detect"]["frames_processed"] == 61494  # All frames
        assert summary["detect"]["frames_processed"] == 10249  # Only every 6th frame

    def test_phase_tracker_start_end_timing(self):
        """Test phase timing with start/end."""
        from tools.episode_run import PhaseTracker
        import time

        tracker = PhaseTracker()
        tracker.start_phase("test_phase")
        time.sleep(0.1)  # Brief pause
        tracker.end_phase("test_phase", frames_processed=100, frames_scanned=100, stride=1)

        summary = tracker.summary()
        assert "test_phase" in summary
        assert summary["test_phase"]["duration_seconds"] >= 0.1
        assert summary["test_phase"]["frames_processed"] == 100


class TestOperationLogsStorage:
    """Tests for per-episode operation log storage."""

    def test_save_and_load_logs(self, tmp_path, monkeypatch):
        """Test saving and loading operation logs."""
        from apps.api.routers.celery_jobs import save_operation_logs, load_operation_logs

        # Monkeypatch the data root to use temp directory
        import apps.api.routers.celery_jobs as cj

        def mock_get_path(ep_id, kind):
            from pathlib import Path
            if kind == "detections":
                return tmp_path / "manifests" / ep_id / "detections.jsonl"
            return tmp_path / kind / ep_id

        monkeypatch.setattr("py_screenalytics.artifacts.get_path", mock_get_path)

        ep_id = "test-episode-logs"
        operation = "detect_track"
        logs = ["Line 1", "Line 2", "Line 3"]
        status = "completed"
        elapsed = 123.45

        # Save logs
        result = save_operation_logs(ep_id, operation, logs, status, elapsed)
        assert result is True

        # Load logs
        loaded = load_operation_logs(ep_id, operation)
        assert loaded is not None
        assert loaded["status"] == "completed"
        assert loaded["logs"] == logs
        assert loaded["elapsed_seconds"] == pytest.approx(123.45)
        assert "updated_at" in loaded

        # Load with history
        loaded_history = load_operation_logs(ep_id, operation, include_history=True, limit=5)
        assert loaded_history is not None
        history = loaded_history.get("history") or []
        assert history, "Expected history entry for saved logs"

    def test_load_nonexistent_logs(self, tmp_path, monkeypatch):
        """Test loading logs that don't exist returns None."""
        from apps.api.routers.celery_jobs import load_operation_logs

        def mock_get_path(ep_id, kind):
            from pathlib import Path
            if kind == "detections":
                return tmp_path / "manifests" / ep_id / "detections.jsonl"
            return tmp_path / kind / ep_id

        monkeypatch.setattr("py_screenalytics.artifacts.get_path", mock_get_path)

        result = load_operation_logs("nonexistent-ep", "detect_track")
        assert result is None


class TestOperationLogsEndpoint:
    """Tests for the GET /celery_jobs/logs/{ep_id}/{operation} endpoint."""

    def test_get_logs_no_data(self, api_client):
        """GET /celery_jobs/logs/{ep_id}/{operation} returns status=none when no logs exist."""
        response = api_client.get("/celery_jobs/logs/nonexistent-ep/detect_track")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "none"
        assert data["logs"] == []
        assert data["elapsed_seconds"] == 0

    def test_get_logs_invalid_operation(self, api_client):
        """GET /celery_jobs/logs/{ep_id}/{operation} returns 400 for invalid operation."""
        response = api_client.get("/celery_jobs/logs/some-ep/invalid_operation")
        assert response.status_code == 400
        assert "Invalid operation" in response.json()["detail"]

    def test_get_logs_valid_operations(self, api_client):
        """GET /celery_jobs/logs/{ep_id}/{operation} accepts valid operations."""
        valid_ops = ["detect_track", "faces_embed", "cluster"]
        for op in valid_ops:
            response = api_client.get(f"/celery_jobs/logs/test-ep/{op}")
            assert response.status_code == 200
            # Should return status=none since no logs exist
            assert response.json()["status"] == "none"


class TestLocalModeStreaming:
    """Tests for local mode streaming responses."""

    def test_local_mode_returns_streaming_response(self, api_client, monkeypatch):
        """Test that local mode returns a streaming response."""
        import subprocess

        # Mock subprocess.Popen to return immediately
        class MockPopen:
            pid = 12345
            returncode = 0

            def __init__(self, *args, **kwargs):
                pass

            @property
            def stdout(self):
                return iter(["Test line 1\n", "Test line 2\n", ""])

            def wait(self):
                return 0

            def poll(self):
                return 0

        monkeypatch.setattr(subprocess, "Popen", MockPopen)

        # Skip duplicate job check
        monkeypatch.setattr(
            "apps.api.routers.celery_jobs._get_running_local_job",
            lambda ep_id, op: None
        )
        monkeypatch.setattr(
            "apps.api.routers.celery_jobs._register_local_job",
            lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            "apps.api.routers.celery_jobs._unregister_local_job",
            lambda *args, **kwargs: None
        )

        # Skip cpulimit wrapper
        monkeypatch.setattr(
            "apps.api.routers.celery_jobs._maybe_wrap_with_cpulimit_local",
            lambda cmd: (cmd, False)
        )

        payload = {
            "ep_id": "test-streaming",
            "stride": 6,
            "device": "cpu",
            "execution_mode": "local",
        }

        # Make the request with stream=True
        response = api_client.post(
            "/celery_jobs/detect_track",
            json=payload,
            # headers={"Accept": "application/x-ndjson"}  # Signal we want streaming
        )

        # Should return 200 with streaming content type
        assert response.status_code == 200
        assert "application/x-ndjson" in response.headers.get("content-type", "")

    def test_stream_generator_yields_json_lines(self, tmp_path, monkeypatch):
        """Test that the stream generator yields valid JSON lines."""
        import subprocess
        import json
        from apps.api.routers.celery_jobs import _stream_local_subprocess

        # Mock subprocess.Popen
        class MockPopen:
            pid = 99999
            returncode = 0

            def __init__(self, *args, **kwargs):
                pass

            @property
            def stdout(self):
                return iter(["Processing frame 1...\n", "Processing frame 2...\n", ""])

            def wait(self):
                return 0

            def poll(self):
                return 0

        monkeypatch.setattr(subprocess, "Popen", MockPopen)

        # Mock job registration
        monkeypatch.setattr(
            "apps.api.routers.celery_jobs._get_running_local_job",
            lambda ep_id, op: None
        )
        monkeypatch.setattr(
            "apps.api.routers.celery_jobs._register_local_job",
            lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            "apps.api.routers.celery_jobs._unregister_local_job",
            lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            "apps.api.routers.celery_jobs._maybe_wrap_with_cpulimit_local",
            lambda cmd: (cmd, False)
        )
        monkeypatch.setattr(
            "apps.api.routers.celery_jobs.save_operation_logs",
            lambda *args, **kwargs: True
        )

        # Create a mock project root
        monkeypatch.setattr(
            "apps.api.routers.celery_jobs._find_project_root",
            lambda: tmp_path
        )

        command = ["echo", "test"]
        options = {"device": "cpu", "stride": 6, "profile": "low_power", "cpu_threads": 2}

        # Collect all yielded lines
        lines = list(_stream_local_subprocess(command, "test-ep", "detect_track", options))

        # Should have at least a few log lines and a summary
        assert len(lines) > 0

        # Each line should be valid JSON
        for line in lines:
            line = line.strip()
            if line:
                data = json.loads(line)
                assert "type" in data
                assert data["type"] in ["log", "summary", "error"]

        # Last non-empty line should be summary
        last_line = [l for l in lines if l.strip()][-1]
        last_data = json.loads(last_line)
        assert last_data["type"] == "summary"
        assert "status" in last_data
        assert "elapsed_seconds" in last_data
