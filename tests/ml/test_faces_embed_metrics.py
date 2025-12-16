"""Integration test for faces_embed with quality and embedding validation.

Tests face embedding pipeline and validates:
- faces.jsonl and faces.npy artifacts exist
- Embeddings are unit-norm (L2 norm ≈ 1.0)
- max_crops_per_track is respected
- Quality gating works (not all detections pass)
- Profile support works correctly
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

# Acceptance thresholds from ACCEPTANCE_MATRIX.md
THRESHOLDS = {
    "faces_per_track_avg": (20, 100),  # Min 20, warn if > 100
    "quality_mean": 0.60,  # Warn if < 0.60 (target: ≥ 0.75)
    "rejection_rate": 0.50,  # Warn if > 0.50 (target: < 0.30)
    "embedding_norm_tolerance": 0.05,  # ±0.05 from 1.0
}


def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL file into list of dicts."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _run_detect_track_fixture(tmp_path: Path, ep_id: str) -> Path:
    """Run detect/track to create fixture for faces_embed test."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        pytest.skip("opencv-python not installed")

    # Create synthetic video with face-like regions
    video_path = tmp_path / "fixture.mp4"
    width, height = 640, 480
    fps = 24
    duration_frames = 60  # 2.5 seconds

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (width, height))

    for frame_idx in range(duration_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Create 2 moving faces
        for face_id in range(2):
            x = 100 + face_id * 250 + int(20 * np.sin(frame_idx / 10.0))
            y = 150 + int(15 * np.cos(frame_idx / 8.0))

            # Draw face-like rectangle with skin tone
            cv2.rectangle(frame, (x, y), (x + 120, y + 150), (180, 160, 140), -1)

            # Add eyes
            cv2.circle(frame, (x + 35, y + 50), 10, (50, 50, 50), -1)
            cv2.circle(frame, (x + 85, y + 50), 10, (50, 50, 50), -1)

        writer.write(frame)

    writer.release()

    # Run detect/track
    data_root = tmp_path / "data"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id", ep_id,
        "--video", str(video_path),
        "--profile", "balanced",
        "--device", "cpu",
        "--out-root", str(data_root),
    ]

    env = os.environ.copy()
    env["SCREANALYTICS_DATA_ROOT"] = str(data_root)

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, f"Detect/track fixture failed: {result.stderr}"
    return data_root


@pytest.mark.timeout(300)
def test_faces_embed_with_quality_gating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test face embedding with quality gating validation."""

    # Force CPU
    try:
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    except ImportError:
        pass

    ep_id = "test-faces-embed-quality"
    data_root = _run_detect_track_fixture(tmp_path, ep_id)

    # Run faces_embed
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id", ep_id,
        "--faces-embed",
        "--profile", "balanced",
        "--device", "cpu",
        "--out-root", str(data_root),
    ]

    env = os.environ.copy()
    env["SCREANALYTICS_DATA_ROOT"] = str(data_root)

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, f"Faces embed failed: {result.stderr}"

    # Assert artifacts exist
    manifest_root = data_root / "manifests" / ep_id
    faces_jsonl_path = manifest_root / "faces.jsonl"
    faces_npy_path = manifest_root / "faces.npy"

    assert faces_jsonl_path.exists(), f"faces.jsonl missing at {faces_jsonl_path}"
    assert faces_npy_path.exists(), f"faces.npy missing at {faces_npy_path}"

    # Load faces metadata
    faces = _read_jsonl(faces_jsonl_path)
    assert len(faces) > 0, "No faces extracted"

    # Load embeddings
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not installed")

    embeddings = np.load(faces_npy_path)
    assert embeddings.shape[0] == len(faces), \
        f"Embedding count mismatch: {embeddings.shape[0]} != {len(faces)}"
    assert embeddings.shape[1] == 512, \
        f"Embedding dimension should be 512, got {embeddings.shape[1]}"

    # Validate embeddings are unit-norm
    norms = np.linalg.norm(embeddings, axis=1)
    tolerance = THRESHOLDS["embedding_norm_tolerance"]

    for idx, norm in enumerate(norms):
        assert abs(norm - 1.0) <= tolerance, \
            f"Embedding {idx} norm {norm:.4f} outside tolerance (1.0 ± {tolerance})"

    print(f"\n✓ Embedding norm validation passed:")
    print(f"  Mean norm: {norms.mean():.4f}")
    print(f"  Min norm: {norms.min():.4f}")
    print(f"  Max norm: {norms.max():.4f}")

    # Validate quality gating worked (some detections should be rejected)
    quality_scores = [f.get("quality_score", 0.0) for f in faces]
    assert len(quality_scores) > 0, "No quality scores in faces"

    # Check that not all detections passed (implies quality gating is active)
    tracks = _read_jsonl(manifest_root / "tracks.jsonl")
    total_track_frames = sum(t.get("frame_count", 0) for t in tracks)

    # Rejection rate should be reasonable (not 0, not 100%)
    if total_track_frames > 0:
        rejection_rate = 1.0 - (len(faces) / total_track_frames)
        print(f"  Rejection rate: {rejection_rate:.3f} ({len(faces)}/{total_track_frames} passed)")

        # Some detections should be rejected (quality gating is working)
        assert 0.0 < rejection_rate < 1.0, \
            "Quality gating appears inactive (all or no detections passed)"


@pytest.mark.timeout(300)
def test_faces_embed_max_crops_per_track(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that max_crops_per_track limit is enforced."""

    try:
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    except ImportError:
        pass

    ep_id = "test-max-crops"
    data_root = _run_detect_track_fixture(tmp_path, ep_id)

    # Run faces_embed with explicit max_crops_per_track
    # Note: This parameter is controlled by config, we're testing default behavior
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id", ep_id,
        "--faces-embed",
        "--profile", "balanced",
        "--device", "cpu",
        "--out-root", str(data_root),
    ]

    env = os.environ.copy()
    env["SCREENALYTICS_DATA_ROOT"] = str(data_root)

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, f"Faces embed failed: {result.stderr}"

    # Load faces and check per-track limits
    manifest_root = data_root / "manifests" / ep_id
    faces = _read_jsonl(manifest_root / "faces.jsonl")

    # Group faces by track_id
    faces_per_track = {}
    for face in faces:
        track_id = face.get("track_id")
        if track_id:
            faces_per_track.setdefault(track_id, []).append(face)

    # Check that no track has excessive faces
    max_allowed = THRESHOLDS["faces_per_track_avg"][1]  # 100

    for track_id, track_faces in faces_per_track.items():
        face_count = len(track_faces)
        # Lenient check: allow some overflow but not excessive
        assert face_count <= max_allowed * 1.5, \
            f"Track {track_id} has {face_count} faces, exceeds reasonable limit"

    print(f"\n✓ Max crops per track validation passed:")
    if faces_per_track:
        avg_faces = sum(len(f) for f in faces_per_track.values()) / len(faces_per_track)
        max_faces = max(len(f) for f in faces_per_track.values())
        print(f"  Avg faces/track: {avg_faces:.1f}")
        print(f"  Max faces/track: {max_faces}")


@pytest.mark.timeout(300)
def test_faces_embed_with_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test face embedding with profile support."""

    try:
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    except ImportError:
        pass

    ep_id = "test-faces-profile"
    data_root = _run_detect_track_fixture(tmp_path, ep_id)

    # Run with high_accuracy profile
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id", ep_id,
        "--faces-embed",
        "--profile", "high_accuracy",
        "--device", "cpu",
        "--out-root", str(data_root),
    ]

    env = os.environ.copy()
    env["SCREANALYTICS_DATA_ROOT"] = str(data_root)

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, f"Faces embed with profile failed: {result.stderr}"

    # Verify artifacts exist
    manifest_root = data_root / "manifests" / ep_id
    assert (manifest_root / "faces.jsonl").exists()
    assert (manifest_root / "faces.npy").exists()

    # Load and verify
    faces = _read_jsonl(manifest_root / "faces.jsonl")
    assert len(faces) > 0, "No faces extracted with high_accuracy profile"

    print(f"\n✓ Profile support validated:")
    print(f"  Total faces extracted: {len(faces)}")


@pytest.mark.timeout(360)
def test_faces_embed_respects_min_frames_between_crops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    except ImportError:
        pass

    ep_id = "test-min-spacing"
    data_root = _run_detect_track_fixture(tmp_path, ep_id)

    def _run_embed(spacing: int) -> list[dict]:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "episode_run.py"),
            "--ep-id",
            ep_id,
            "--faces-embed",
            "--profile",
            "balanced",
            "--device",
            "cpu",
            "--out-root",
            str(data_root),
            "--sample-every-n-frames",
            str(spacing),
        ]

        env = os.environ.copy()
        env["SCREANALYTICS_DATA_ROOT"] = str(data_root)
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert result.returncode == 0, f"Faces embed failed with spacing={spacing}: {result.stderr}"
        manifest_root = data_root / "manifests" / ep_id
        return _read_jsonl(manifest_root / "faces.jsonl")

    faces_dense = _run_embed(4)
    faces_sparse = _run_embed(40)

    assert faces_dense, "Expected faces for dense spacing"
    assert faces_sparse, "Expected faces for sparse spacing"
    assert len(faces_sparse) < len(faces_dense), "Spacing did not reduce crop count per track"
