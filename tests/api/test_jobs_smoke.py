"""API smoke test for end-to-end pipeline via HTTP endpoints.

Tests the full pipeline workflow via API (not CLI):
- Submit detect_track → faces_embed → cluster jobs
- Poll until completion
- Verify all jobs succeed
- Verify metrics are returned in responses
- Verify artifacts exist on disk
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def _start_api_server(data_root: Path) -> tuple[subprocess.Popen, int]:
    """Start API server in background, return process and port."""
    port = 8765  # Use non-standard port to avoid conflicts

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "apps.api.main:app",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--log-level", "warning",
    ]

    env = os.environ.copy()
    env["SCREENALYTICS_DATA_ROOT"] = str(data_root)

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    time.sleep(3)

    # Check if server started successfully
    if proc.poll() is not None:
        stdout, stderr = proc.communicate()
        raise RuntimeError(f"API server failed to start:\nSTDOUT:\n{stdout.decode()}\n\nSTDERR:\n{stderr.decode()}")

    return proc, port


def _api_request(method: str, path: str, port: int, data: dict | None = None) -> dict:
    """Make API request using requests library."""
    try:
        import requests
    except ImportError:
        pytest.skip("requests library not installed")

    url = f"http://127.0.0.1:{port}{path}"

    if method == "GET":
        response = requests.get(url, timeout=10)
    elif method == "POST":
        response = requests.post(url, json=data, timeout=10)
    else:
        raise ValueError(f"Unsupported method: {method}")

    response.raise_for_status()
    return response.json()


def _poll_job_until_done(job_id: str, port: int, timeout_sec: int = 300) -> dict:
    """Poll job status until it reaches terminal state."""
    start_time = time.time()

    while time.time() - start_time < timeout_sec:
        status = _api_request("GET", f"/jobs/{job_id}", port)

        if status["state"] in ["succeeded", "failed", "canceled"]:
            return status

        time.sleep(2)  # Poll every 2 seconds

    raise TimeoutError(f"Job {job_id} did not complete within {timeout_sec}s")


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
    video_path = _create_test_video(tmp_path / "test.mp4", duration_sec=10, fps=24)

    # Copy video to data directory for API access
    video_dest = data_root / "videos" / "api-smoke-test.mp4"
    video_dest.parent.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.copy(video_path, video_dest)

    ep_id = "api-smoke-test"

    # Start API server
    server_proc, port = _start_api_server(data_root)

    try:
        # Step 1: Submit detect_track job
        print("\n=== Step 1: Submitting detect_track job ===")
        detect_request = {
            "ep_id": ep_id,
            "profile": "balanced",
            "stride": 6,
            "device": "cpu",
            "save_frames": False,
            "save_crops": False,
        }

        detect_response = _api_request("POST", "/jobs/detect_track_async", port, detect_request)
        detect_job_id = detect_response["job_id"]

        print(f"  Job ID: {detect_job_id}")
        assert detect_response["state"] == "running"

        # Poll until detect_track completes
        print("  Polling for completion...")
        detect_final = _poll_job_until_done(detect_job_id, port, timeout_sec=300)

        assert detect_final["state"] == "succeeded", \
            f"detect_track job failed: {detect_final.get('error', 'unknown error')}"

        # Verify metrics are included in response
        assert "track_metrics" in detect_final, "track_metrics missing from job response"
        track_metrics = detect_final["track_metrics"]

        assert "metrics" in track_metrics, "metrics missing from track_metrics"
        metrics = track_metrics["metrics"]

        # Verify key metric fields exist
        required_metrics = ["total_detections", "total_tracks", "tracks_per_minute", "short_track_fraction", "id_switch_rate"]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

        print(f"  ✓ detect_track completed: {metrics['total_tracks']} tracks, {metrics['total_detections']} detections")

        # Verify artifacts exist
        manifest_root = data_root / "manifests" / ep_id
        assert (manifest_root / "detections.jsonl").exists(), "detections.jsonl not created"
        assert (manifest_root / "tracks.jsonl").exists(), "tracks.jsonl not created"
        assert (manifest_root / "track_metrics.json").exists(), "track_metrics.json not created"

        # Step 2: Submit faces_embed job
        print("\n=== Step 2: Submitting faces_embed job ===")
        faces_request = {
            "ep_id": ep_id,
            "profile": "balanced",
            "device": "cpu",
            "save_frames": False,
            "save_crops": False,
        }

        faces_response = _api_request("POST", "/jobs/faces_embed_async", port, faces_request)
        faces_job_id = faces_response["job_id"]

        print(f"  Job ID: {faces_job_id}")
        assert faces_response["state"] == "running"

        # Poll until faces_embed completes
        print("  Polling for completion...")
        faces_final = _poll_job_until_done(faces_job_id, port, timeout_sec=240)

        assert faces_final["state"] == "succeeded", \
            f"faces_embed job failed: {faces_final.get('error', 'unknown error')}"

        print(f"  ✓ faces_embed completed")

        # Verify artifacts exist
        assert (manifest_root / "faces.jsonl").exists(), "faces.jsonl not created"
        assert (manifest_root / "faces.npy").exists(), "faces.npy not created"

        # Step 3: Submit cluster job
        print("\n=== Step 3: Submitting cluster job ===")
        cluster_request = {
            "ep_id": ep_id,
            "profile": "balanced",
            "device": "cpu",
            "cluster_thresh": 0.58,
            "min_cluster_size": 2,
            "min_identity_sim": 0.45,
        }

        cluster_response = _api_request("POST", "/jobs/cluster_async", port, cluster_request)
        cluster_job_id = cluster_response["job_id"]

        print(f"  Job ID: {cluster_job_id}")
        assert cluster_response["state"] == "running"

        # Poll until cluster completes
        print("  Polling for completion...")
        cluster_final = _poll_job_until_done(cluster_job_id, port, timeout_sec=120)

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
        progress = _api_request("GET", f"/jobs/{cluster_job_id}/progress", port)

        # Progress should include metrics after completion
        assert "track_metrics" in progress, "track_metrics missing from progress endpoint"

        print(f"  ✓ Progress endpoint returns metrics")

        print("\n=== API smoke test PASSED ===")
        print(f"  All 3 jobs completed successfully")
        print(f"  Metrics exposed via API")
        print(f"  All artifacts created")

    finally:
        # Clean up: terminate API server
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()


@pytest.mark.timeout(300)
def test_api_job_not_found(tmp_path: Path) -> None:
    """Test that API returns 404 for non-existent job."""

    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    server_proc, port = _start_api_server(data_root)

    try:
        import requests

        # Request non-existent job
        url = f"http://127.0.0.1:{port}/jobs/nonexistent-job-id"
        response = requests.get(url, timeout=10)

        assert response.status_code == 404, \
            f"Expected 404 for non-existent job, got {response.status_code}"

        print("\n✓ API correctly returns 404 for non-existent job")

    except ImportError:
        pytest.skip("requests library not installed")
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()


@pytest.mark.timeout(300)
def test_api_profile_validation(tmp_path: Path) -> None:
    """Test that API rejects invalid profile names."""

    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    server_proc, port = _start_api_server(data_root)

    try:
        import requests

        # Submit job with invalid profile
        url = f"http://127.0.0.1:{port}/jobs/detect_track_async"
        invalid_request = {
            "ep_id": "test",
            "profile": "invalid_profile_name",
            "device": "cpu",
        }

        response = requests.post(url, json=invalid_request, timeout=10)

        # Should be rejected with 422 (validation error)
        assert response.status_code == 422, \
            f"Expected 422 for invalid profile, got {response.status_code}"

        print("\n✓ API correctly rejects invalid profile names")

    except ImportError:
        pytest.skip("requests library not installed")
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
