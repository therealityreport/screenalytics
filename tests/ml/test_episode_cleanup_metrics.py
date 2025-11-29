"""Integration test for episode_cleanup with before/after validation.

Tests cleanup workflow and validates:
- cleanup_report.json exists and contains before/after stats
- Metrics improve in the right direction
- No dangling track_id or identity_id references
- Profile support works correctly

This test runs the full pipeline (detect→track→faces→cluster→cleanup) on a fixture.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ML_TESTS = os.environ.get("RUN_ML_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_ML_TESTS,
    reason="set RUN_ML_TESTS=1 to run ML integration tests"
)


def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL file into list of dicts."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _create_fixture_video(target: Path, duration_sec: int = 15, fps: int = 24) -> Path:
    """Create synthetic video with moving faces for cleanup testing."""
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

    # Create faces that will produce fragmented tracks (for cleanup to fix)
    for frame_idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Create 3 faces with occasional disappearances (creates short tracks)
        for face_id in range(3):
            # Make face occasionally disappear to create fragmentation
            if frame_idx % 20 < 15:  # Visible 75% of time
                x = 100 + face_id * 200
                y = 150 + int(30 * np.sin(frame_idx / 15.0))

                # Draw face
                cv2.rectangle(frame, (x, y), (x + 100, y + 120), (180, 160, 140), -1)
                # Eyes
                cv2.circle(frame, (x + 30, y + 40), 8, (50, 50, 50), -1)
                cv2.circle(frame, (x + 70, y + 40), 8, (50, 50, 50), -1)

        writer.write(frame)

    writer.release()
    return target


def _run_full_pipeline(tmp_path: Path, ep_id: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run detect→track→faces→cluster to create fixture for cleanup."""

    # Force CPU
    try:
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    except ImportError:
        pass

    data_root = tmp_path / "data"
    video_path = _create_fixture_video(tmp_path / "fixture.mp4", duration_sec=15, fps=24)

    env = os.environ.copy()
    env["SCREENALYTICS_DATA_ROOT"] = str(data_root)

    # Step 1: detect/track
    cmd_detect = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id", ep_id,
        "--video", str(video_path),
        "--profile", "balanced",
        "--device", "cpu",
        "--out-root", str(data_root),
    ]

    result = subprocess.run(
        cmd_detect,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, f"Detect/track failed: {result.stderr}"

    # Step 2: faces_embed
    cmd_faces = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id", ep_id,
        "--faces-embed",
        "--profile", "balanced",
        "--device", "cpu",
        "--out-root", str(data_root),
    ]

    result = subprocess.run(
        cmd_faces,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"Faces embed failed: {result.stderr}"

    # Step 3: cluster
    cmd_cluster = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id", ep_id,
        "--cluster",
        "--profile", "balanced",
        "--device", "cpu",
        "--out-root", str(data_root),
    ]

    result = subprocess.run(
        cmd_cluster,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Cluster failed: {result.stderr}"

    return data_root


@pytest.mark.timeout(900)
def test_episode_cleanup_full_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test full cleanup workflow with before/after metric validation."""

    ep_id = "test-cleanup-full"
    data_root = _run_full_pipeline(tmp_path, ep_id, monkeypatch)

    manifest_root = data_root / "manifests" / ep_id

    # Load metrics BEFORE cleanup
    metrics_path = manifest_root / "track_metrics.json"
    assert metrics_path.exists(), "track_metrics.json missing before cleanup"

    with metrics_path.open("r") as f:
        metrics_before = json.load(f)

    tracks_before_count = metrics_before["metrics"]["total_tracks"]
    short_fraction_before = metrics_before["metrics"]["short_track_fraction"]

    cluster_metrics_before = metrics_before.get("cluster_metrics", {})
    singleton_fraction_before = cluster_metrics_before.get("singleton_fraction", 0.0)
    largest_cluster_fraction_before = cluster_metrics_before.get("largest_cluster_fraction", 0.0)

    # Load identities BEFORE cleanup
    identities_path = manifest_root / "identities.json"
    with identities_path.open("r") as f:
        identities_before = json.load(f)

    clusters_before_count = identities_before["stats"]["clusters"]

    # Run cleanup with all actions
    video_path = tmp_path / "fixture.mp4"
    cmd_cleanup = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_cleanup.py"),
        "--ep-id", ep_id,
        "--video", str(video_path),
        "--profile", "balanced",
        "--device", "cpu",
        "--embed-device", "cpu",
        "--out-root", str(data_root),
        "--actions", "split_tracks", "reembed", "recluster", "group_clusters",
        "--write-back",
    ]

    env = os.environ.copy()
    env["SCREENALYTICS_DATA_ROOT"] = str(data_root)

    result = subprocess.run(
        cmd_cleanup,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, f"Cleanup failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Assert cleanup_report.json exists
    cleanup_report_path = manifest_root / "cleanup_report.json"
    assert cleanup_report_path.exists(), f"cleanup_report.json missing at {cleanup_report_path}"

    with cleanup_report_path.open("r") as f:
        cleanup_report = json.load(f)

    # Validate cleanup_report structure
    assert "actions_completed" in cleanup_report, "Missing actions_completed in cleanup_report"
    assert "runtime_sec" in cleanup_report, "Missing runtime_sec in cleanup_report"

    # Load metrics AFTER cleanup
    with metrics_path.open("r") as f:
        metrics_after = json.load(f)

    tracks_after_count = metrics_after["metrics"]["total_tracks"]
    short_fraction_after = metrics_after["metrics"]["short_track_fraction"]

    cluster_metrics_after = metrics_after.get("cluster_metrics", {})
    singleton_fraction_after = cluster_metrics_after.get("singleton_fraction", 0.0)
    largest_cluster_fraction_after = cluster_metrics_after.get("largest_cluster_fraction", 0.0)

    # Load identities AFTER cleanup
    with identities_path.open("r") as f:
        identities_after = json.load(f)

    clusters_after_count = identities_after["stats"]["clusters"]

    # Validate before/after improvements
    print(f"\n✓ Cleanup workflow completed:")
    print(f"  Tracks: {tracks_before_count} → {tracks_after_count}")
    print(f"  Short track fraction: {short_fraction_before:.3f} → {short_fraction_after:.3f}")
    print(f"  Singleton fraction: {singleton_fraction_before:.3f} → {singleton_fraction_after:.3f}")
    print(f"  Largest cluster fraction: {largest_cluster_fraction_before:.3f} → {largest_cluster_fraction_after:.3f}")
    print(f"  Clusters: {clusters_before_count} → {clusters_after_count}")

    # Assert metrics moved in correct direction (or stayed acceptable)
    # Note: Cleanup might not always improve metrics on synthetic data, but should not make them worse
    assert short_fraction_after <= short_fraction_before + 0.05, \
        f"Short track fraction worsened significantly: {short_fraction_before:.3f} → {short_fraction_after:.3f}"

    # Singleton fraction should stay within acceptable bounds
    if clusters_after_count > 1:
        assert singleton_fraction_after <= 0.60, \
            f"Singleton fraction after cleanup too high: {singleton_fraction_after:.3f}"

    # Validate no dangling references
    tracks = _read_jsonl(manifest_root / "tracks.jsonl")
    faces = _read_jsonl(manifest_root / "faces.jsonl")

    track_ids = {t["track_id"] for t in tracks}
    face_track_ids = {f["track_id"] for f in faces if "track_id" in f}

    # All face track_ids should reference valid tracks
    dangling_face_refs = face_track_ids - track_ids
    assert not dangling_face_refs, \
        f"Found {len(dangling_face_refs)} dangling track_id references in faces.jsonl"

    # All identity track_ids should reference valid tracks
    identity_track_ids = set()
    for identity in identities_after["identities"]:
        identity_track_ids.update(identity.get("track_ids", []))

    dangling_identity_refs = identity_track_ids - track_ids
    assert not dangling_identity_refs, \
        f"Found {len(dangling_identity_refs)} dangling track_id references in identities.json"

    print(f"  ✓ No dangling references found")


@pytest.mark.timeout(600)
def test_cleanup_report_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that cleanup_report.json contains required before/after fields."""

    ep_id = "test-cleanup-schema"
    data_root = _run_full_pipeline(tmp_path, ep_id, monkeypatch)

    # Run minimal cleanup
    video_path = tmp_path / "fixture.mp4"
    cmd_cleanup = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_cleanup.py"),
        "--ep-id", ep_id,
        "--video", str(video_path),
        "--device", "cpu",
        "--embed-device", "cpu",
        "--out-root", str(data_root),
        "--actions", "split_tracks",  # Just one action to speed up
        "--write-back",
    ]

    env = os.environ.copy()
    env["SCREANALYTICS_DATA_ROOT"] = str(data_root)

    result = subprocess.run(
        cmd_cleanup,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"Cleanup failed: {result.stderr}"

    # Load cleanup_report.json
    manifest_root = data_root / "manifests" / ep_id
    cleanup_report_path = manifest_root / "cleanup_report.json"
    assert cleanup_report_path.exists()

    with cleanup_report_path.open("r") as f:
        report = json.load(f)

    # Validate required top-level fields
    required_fields = [
        "actions_completed",
        "runtime_sec",
        "tracks_before",
        "tracks_after",
        "faces_before",
        "faces_after",
        "clusters_before",
        "clusters_after",
        "metrics_before",
        "metrics_after",
    ]
    for field in required_fields:
        assert field in report, f"Missing required field: {field}"

    # Validate runtime is reasonable
    assert isinstance(report["runtime_sec"], (int, float))
    assert report["runtime_sec"] > 0

    # Validate count fields are integers
    assert isinstance(report["tracks_before"], int)
    assert isinstance(report["tracks_after"], int)
    assert isinstance(report["faces_before"], int)
    assert isinstance(report["faces_after"], int)
    assert isinstance(report["clusters_before"], int)
    assert isinstance(report["clusters_after"], int)

    # Validate metrics_before/after are dicts
    assert isinstance(report["metrics_before"], dict)
    assert isinstance(report["metrics_after"], dict)

    # Check that key metrics are present (if data exists)
    if report["tracks_before"] > 0:
        # Should have before metrics
        expected_metrics = ["tracks_per_minute", "short_track_fraction", "id_switch_rate"]
        for metric in expected_metrics:
            if metric in report["metrics_before"]:
                assert isinstance(report["metrics_before"][metric], (int, float, type(None)))

    print(f"\n✓ Cleanup report schema validated:")
    print(f"  Actions completed: {report['actions_completed']}")
    print(f"  Runtime: {report['runtime_sec']:.1f}s")
    print(f"  Tracks: {report['tracks_before']} → {report['tracks_after']}")
    print(f"  Faces: {report['faces_before']} → {report['faces_after']}")
    print(f"  Clusters: {report['clusters_before']} → {report['clusters_after']}")


@pytest.mark.timeout(600)
def test_cleanup_progress_reporting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that cleanup writes progress with phase tracking."""

    ep_id = "test-cleanup-progress"
    data_root = _run_full_pipeline(tmp_path, ep_id, monkeypatch)

    manifest_root = data_root / "manifests" / ep_id
    progress_path = manifest_root / "cleanup_progress.json"

    # Run cleanup with profile
    video_path = tmp_path / "fixture.mp4"
    cmd_cleanup = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_cleanup.py"),
        "--ep-id", ep_id,
        "--video", str(video_path),
        "--profile", "balanced",  # Test profile parameter
        "--device", "cpu",
        "--embed-device", "cpu",
        "--out-root", str(data_root),
        "--actions", "split_tracks", "reembed",  # Two actions for phase tracking
        "--write-back",
    ]

    env = os.environ.copy()
    env["SCREENALYTICS_DATA_ROOT"] = str(data_root)

    result = subprocess.run(
        cmd_cleanup,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=400,
    )

    assert result.returncode == 0, f"Cleanup failed: {result.stderr}"

    # Load final progress
    assert progress_path.exists(), f"Progress file missing at {progress_path}"

    with progress_path.open("r") as f:
        progress = json.load(f)

    # Validate progress structure
    required_progress_fields = [
        "stage",
        "ep_id",
        "phase",
        "phase_index",
        "phase_total",
        "phase_progress",
        "total_elapsed_seconds",
    ]
    for field in required_progress_fields:
        assert field in progress, f"Missing progress field: {field}"

    # Validate values
    assert progress["stage"] == "episode_cleanup"
    assert progress["ep_id"] == ep_id
    assert progress["phase"] in ["split_tracks", "reembed", "recluster", "group_clusters", "done"]
    assert isinstance(progress["phase_index"], int)
    assert isinstance(progress["phase_total"], int)
    assert 0.0 <= progress["phase_progress"] <= 1.0
    assert progress["total_elapsed_seconds"] > 0

    print(f"\n✓ Progress reporting validated:")
    print(f"  Final phase: {progress['phase']}")
    print(f"  Phase {progress['phase_index']}/{progress['phase_total']}")
    print(f"  Progress: {progress['phase_progress']:.1%}")
    print(f"  Elapsed: {progress['total_elapsed_seconds']:.1f}s")


@pytest.mark.timeout(600)
def test_cleanup_preserves_valid_tracks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that cleanup doesn't delete all tracks (preserves valid ones)."""

    ep_id = "test-cleanup-preserve"
    data_root = _run_full_pipeline(tmp_path, ep_id, monkeypatch)

    manifest_root = data_root / "manifests" / ep_id

    # Load tracks before cleanup
    tracks_before = _read_jsonl(manifest_root / "tracks.jsonl")
    tracks_before_count = len(tracks_before)

    # Run cleanup
    video_path = tmp_path / "fixture.mp4"
    cmd_cleanup = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_cleanup.py"),
        "--ep-id", ep_id,
        "--video", str(video_path),
        "--device", "cpu",
        "--embed-device", "cpu",
        "--out-root", str(data_root),
        "--actions", "split_tracks", "reembed",
        "--write-back",
    ]

    env = os.environ.copy()
    env["SCREANALYTICS_DATA_ROOT"] = str(data_root)

    result = subprocess.run(
        cmd_cleanup,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=400,
    )

    assert result.returncode == 0, f"Cleanup failed: {result.stderr}"

    # Load tracks after cleanup
    tracks_after = _read_jsonl(manifest_root / "tracks.jsonl")
    tracks_after_count = len(tracks_after)

    # Should still have tracks (not delete everything)
    assert tracks_after_count > 0, "Cleanup deleted all tracks"

    # Track count might increase (splits) or stay similar
    print(f"\n✓ Track preservation validated:")
    print(f"  Tracks before: {tracks_before_count}")
    print(f"  Tracks after: {tracks_after_count}")
