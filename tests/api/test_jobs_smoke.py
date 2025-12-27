"""API smoke test for end-to-end pipeline via HTTP endpoints.

Tests the full pipeline workflow via API (not CLI):
- Submit detect_track → faces_embed → cluster jobs
- Poll until completion
- Verify all jobs succeed
- Verify metrics are returned in responses
- Verify artifacts exist on disk
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import jobs as jobs_router
from apps.api.services.jobs import JobService
from py_screenalytics import run_layout

RUN_ML_TESTS = os.environ.get("RUN_ML_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_ML_TESTS,
    reason="set RUN_ML_TESTS=1 to run ML integration tests"
)


def _create_test_video(target: Path, duration_sec: int = 10, fps: int = 24) -> Path:
    """Create synthetic test video."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        pytest.skip("opencv-python not installed")

    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_count = duration_sec * fps

    writer = cv2.VideoWriter(str(target), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video at {target}")

    for frame_idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Create 2 moving faces
        for face_id in range(2):
            x = 150 + face_id * 250
            y = 200 + int(30 * np.sin(frame_idx / 10.0))

            # Draw face-like rectangle
            cv2.rectangle(frame, (x, y), (x + 100, y + 120), (180, 160, 140), -1)
            cv2.circle(frame, (x + 30, y + 40), 8, (50, 50, 50), -1)
            cv2.circle(frame, (x + 70, y + 40), 8, (50, 50, 50), -1)

        writer.write(frame)

    writer.release()
    return target


@contextmanager
def _api_client(data_root: Path) -> TestClient:
    """Yield a TestClient bound to a fresh JobService rooted at data_root."""

    os.environ["SCREENALYTICS_DATA_ROOT"] = str(data_root)
    jobs_router.JOB_SERVICE = JobService(data_root=data_root)
    with TestClient(app) as client:
        yield client


def _api_request(client: TestClient, method: str, path: str, data: dict | None = None) -> dict:
    """Make API request using FastAPI TestClient."""
    if method == "GET":
        response = client.get(path)
    elif method == "POST":
        response = client.post(path, json=data or {})
    else:
        raise ValueError(f"Unsupported method: {method}")

    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise AssertionError(f"Expected dict response, got {type(payload)}")
    return payload


def _poll_job_until_done(
    client: TestClient,
    job_id: str,
    *,
    timeout_sec: int = 180,
    poll_interval: float = 2.0,
) -> dict:
    """Poll job status until it reaches terminal state, logging progress periodically."""
    start_time = time.time()
    last_log = -poll_interval
    last_progress: dict = {}
    status: dict = {"state": "unknown"}

    while time.time() - start_time < timeout_sec:
        status = _api_request(client, "GET", f"/jobs/{job_id}")
        state = status.get("state")

        if state in {"succeeded", "failed", "canceled"}:
            progress_payload = _api_request(client, "GET", f"/jobs/{job_id}/progress")
            status.setdefault("track_metrics", progress_payload.get("track_metrics"))
            status.setdefault("progress", progress_payload.get("progress"))
            return status

        elapsed = time.time() - start_time
        if elapsed - last_log >= 10:
            progress_payload = _api_request(client, "GET", f"/jobs/{job_id}/progress")
            last_progress = progress_payload.get("progress") or {}
            phase = last_progress.get("phase")
            frames_done = last_progress.get("frames_done")
            frames_total = last_progress.get("frames_total")
            print(
                f"    state={state} phase={phase} frames={frames_done}/{frames_total} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            last_log = elapsed

        time.sleep(poll_interval)

    progress_payload = _api_request(client, "GET", f"/jobs/{job_id}/progress")
    progress = progress_payload.get("progress") or last_progress
    raise TimeoutError(
        f"Job {job_id} did not complete within {timeout_sec}s "
        f"(state={status.get('state')} phase={progress.get('phase')} "
        f"frames={progress.get('frames_done')}/{progress.get('frames_total')})"
    )


@pytest.mark.timeout(900)
def test_api_full_pipeline_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test full pipeline via API: detect_track → faces_embed → cluster."""

    # Force CPU
    try:
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    except ImportError:
        pass

    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    # Create test video
    video_path = _create_test_video(tmp_path / "test.mp4", duration_sec=8, fps=24)

    # Copy video to data directory for API access
    video_dest = data_root / "videos" / "api-smoke-test.mp4"
    video_dest.parent.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.copy(video_path, video_dest)

    ep_id = "api-smoke-test"
    run_id = "run-smoke-1"
    manifest_root = run_layout.run_root(ep_id, run_id)
    embed_root = data_root / "embeds" / ep_id / "runs" / run_id

    with _api_client(data_root) as client:
        # Step 1: Submit detect_track job
        print("\n=== Step 1: Submitting detect_track job ===")
        detect_request = {
            "ep_id": ep_id,
            "run_id": run_id,
            "profile": "balanced",
            "stride": 6,
            "device": "cpu",
            "save_frames": False,
            "save_crops": False,
        }

        detect_response = _api_request(client, "POST", "/jobs/detect_track_async", detect_request)
        detect_job_id = detect_response["job_id"]

        print(f"  Job ID: {detect_job_id}")
        assert detect_response["state"] == "queued"

        # Poll until detect_track completes
        print("  Polling for completion...")
        detect_final = _poll_job_until_done(client, detect_job_id, timeout_sec=180)

        assert detect_final["state"] == "succeeded", \
            f"detect_track job failed: {detect_final.get('error', 'unknown error')}"

        # Verify metrics are included in response
        assert "track_metrics" in detect_final, "track_metrics missing from job response"
        track_metrics = detect_final["track_metrics"]

        assert "metrics" in track_metrics, "metrics missing from track_metrics"
        metrics = track_metrics["metrics"]

        # Verify key metric fields exist
        required_metrics = [
            "total_detections",
            "total_tracks",
            "tracks_per_minute",
            "short_track_fraction",
            "id_switch_rate",
        ]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

        print(f"  ✓ detect_track completed: {metrics['total_tracks']} tracks, {metrics['total_detections']} detections")

        # Verify artifacts exist
        assert (manifest_root / "detections.jsonl").exists(), "detections.jsonl not created"
        assert (manifest_root / "tracks.jsonl").exists(), "tracks.jsonl not created"
        assert (manifest_root / "track_metrics.json").exists(), "track_metrics.json not created"

        # Step 2: Submit faces_embed job
        print("\n=== Step 2: Submitting faces_embed job ===")
        faces_request = {
            "ep_id": ep_id,
            "run_id": run_id,
            "profile": "balanced",
            "device": "cpu",
            "save_frames": False,
            "save_crops": False,
        }

        faces_response = _api_request(client, "POST", "/jobs/faces_embed_async", faces_request)
        faces_job_id = faces_response["job_id"]

        print(f"  Job ID: {faces_job_id}")
        assert faces_response["state"] == "queued"

        # Poll until faces_embed completes
        print("  Polling for completion...")
        faces_final = _poll_job_until_done(client, faces_job_id, timeout_sec=150)

        assert faces_final["state"] == "succeeded", \
            f"faces_embed job failed: {faces_final.get('error', 'unknown error')}"

        print("  ✓ faces_embed completed")

        # Verify artifacts exist
        assert (manifest_root / "faces.jsonl").exists(), "faces.jsonl not created"
        assert (embed_root / "faces.npy").exists(), "faces.npy not created"

        # Step 3: Submit cluster job
        print("\n=== Step 3: Submitting cluster job ===")
        cluster_request = {
            "ep_id": ep_id,
            "run_id": run_id,
            "profile": "balanced",
            "device": "cpu",
            "cluster_thresh": 0.58,
            "min_cluster_size": 2,
            "min_identity_sim": 0.45,
        }

        cluster_response = _api_request(client, "POST", "/jobs/cluster_async", cluster_request)
        cluster_job_id = cluster_response["job_id"]

        print(f"  Job ID: {cluster_job_id}")
        assert cluster_response["state"] == "queued"

        # Poll until cluster completes
        print("  Polling for completion...")
        cluster_final = _poll_job_until_done(client, cluster_job_id, timeout_sec=120)

        assert cluster_final["state"] == "succeeded", \
            f"cluster job failed: {cluster_final.get('error', 'unknown error')}"

        # Verify cluster metrics are included
        assert "track_metrics" in cluster_final, "track_metrics missing after clustering"
        cluster_metrics = cluster_final["track_metrics"].get("cluster_metrics", {})

        required_cluster_metrics = [
            "singleton_count",
            "singleton_fraction",
            "largest_cluster_size",
            "largest_cluster_fraction",
            "total_clusters",
        ]
        for metric in required_cluster_metrics:
            assert metric in cluster_metrics, f"Missing cluster metric: {metric}"

        print(f"  ✓ cluster completed: {cluster_metrics['total_clusters']} clusters")

        # Verify artifacts exist
        assert (manifest_root / "identities.json").exists(), "identities.json not created"

        # Test GET /jobs/{job_id}/progress endpoint
        print("\n=== Testing progress endpoint ===")
        progress = _api_request(client, "GET", f"/jobs/{cluster_job_id}/progress")

        # Progress should include metrics after completion
        assert "track_metrics" in progress, "track_metrics missing from progress endpoint"

        print("  ✓ Progress endpoint returns metrics")

        print("\n=== API smoke test PASSED ===")
        print("  All 3 jobs completed successfully")
        print("  Metrics exposed via API")
        print("  All artifacts created")


@pytest.mark.timeout(300)
def test_api_job_not_found(tmp_path: Path) -> None:
    """Test that API returns 404 for non-existent job."""

    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    with _api_client(data_root) as client:
        response = client.get("/jobs/nonexistent-job-id")
        assert response.status_code == 404, \
            f"Expected 404 for non-existent job, got {response.status_code}"

        print("\n✓ API correctly returns 404 for non-existent job")


@pytest.mark.timeout(300)
def test_api_profile_validation(tmp_path: Path) -> None:
    """Test that API rejects invalid profile names."""

    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    with _api_client(data_root) as client:
        invalid_request = {
            "ep_id": "test",
            "profile": "invalid_profile_name",
            "device": "cpu",
        }

        response = client.post("/jobs/detect_track_async", json=invalid_request)

        # Should be rejected with 422 (validation error)
        assert response.status_code == 422, \
            f"Expected 422 for invalid profile, got {response.status_code}"

        print("\n✓ API correctly rejects invalid profile names")
