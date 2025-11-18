"""Tests for automatic track splitting functionality."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Add tools to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

from split_mixed_tracks import split_high_spread_tracks


@pytest.fixture
def mock_episode_data(tmp_path):
    """Create mock episode data with mixed tracks."""
    ep_id = "test-s01e01"
    data_root = tmp_path / "data"
    manifests_dir = data_root / "manifests" / ep_id
    manifests_dir.mkdir(parents=True)

    # Create tracks with one high-spread track
    tracks = [
        {
            "track_id": 1,
            "ep_id": ep_id,
            "start_s": 0.0,
            "end_s": 1.0,
            "frame_span": [0, 30],
            "face_embedding_spread": 0.2,  # Low spread - good track
            "stats": {"detections": 10},
        },
        {
            "track_id": 2,
            "ep_id": ep_id,
            "start_s": 1.0,
            "end_s": 3.0,
            "frame_span": [30, 90],
            "face_embedding_spread": 0.8,  # High spread - mixed track
            "stats": {"detections": 20},
        },
        {
            "track_id": 3,
            "ep_id": ep_id,
            "start_s": 3.0,
            "end_s": 4.0,
            "frame_span": [90, 120],
            "face_embedding_spread": 0.15,  # Low spread - good track
            "stats": {"detections": 8},
        },
    ]

    tracks_path = manifests_dir / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as f:
        for track in tracks:
            f.write(json.dumps(track) + "\n")

    # Create faces with embeddings for track 2 (mixed track)
    # Simulate two different people in the same track
    np.random.seed(42)
    person1_base = np.random.rand(512).astype(np.float32)
    person1_base = person1_base / np.linalg.norm(person1_base)
    person2_base = np.random.rand(512).astype(np.float32)
    person2_base = person2_base / np.linalg.norm(person2_base)

    faces = []
    # Track 1 faces (not split)
    for i in range(5):
        faces.append(
            {
                "track_id": 1,
                "frame_idx": i,
                "embedding": (person1_base + np.random.randn(512) * 0.01).tolist(),
            }
        )

    # Track 2 faces (will be split) - 10 from person1, 10 from person2
    for i in range(10):
        faces.append(
            {
                "track_id": 2,
                "frame_idx": 30 + i,
                "embedding": (person1_base + np.random.randn(512) * 0.01).tolist(),
            }
        )
    for i in range(10, 20):
        faces.append(
            {
                "track_id": 2,
                "frame_idx": 30 + i,
                "embedding": (person2_base + np.random.randn(512) * 0.01).tolist(),
            }
        )

    # Track 3 faces (not split)
    for i in range(5):
        faces.append(
            {
                "track_id": 3,
                "frame_idx": 90 + i,
                "embedding": (person1_base + np.random.randn(512) * 0.01).tolist(),
            }
        )

    faces_path = manifests_dir / "faces.jsonl"
    with faces_path.open("w", encoding="utf-8") as f:
        for face in faces:
            f.write(json.dumps(face) + "\n")

    # Mock get_path to return our test directory
    import py_screenalytics.artifacts as artifacts_module

    original_get_path = artifacts_module.get_path

    def mock_get_path(ep_id_arg, artifact_type):
        if artifact_type == "detections":
            return manifests_dir / "detections_placeholder"
        return original_get_path(ep_id_arg, artifact_type)

    artifacts_module.get_path = mock_get_path

    yield ep_id, manifests_dir

    # Restore original
    artifacts_module.get_path = original_get_path


def test_split_high_spread_tracks_dry_run(mock_episode_data):
    """Test dry run mode identifies tracks to split without modifying files."""
    ep_id, manifests_dir = mock_episode_data

    result = split_high_spread_tracks(
        ep_id, spread_threshold=0.35, dry_run=True, manifests_dir=manifests_dir
    )

    assert result["dry_run"] is True
    assert result["split_count"] == 1  # Only track 2 should be flagged
    assert 2 in result["track_mapping"]  # Track 2 is in the mapping

    # Verify files were not modified
    tracks_path = manifests_dir / "tracks.jsonl"
    with tracks_path.open("r") as f:
        tracks = [json.loads(line) for line in f if line.strip()]
    assert len(tracks) == 3  # Original 3 tracks


def test_split_high_spread_tracks_real_split(mock_episode_data):
    """Test actual splitting of high-spread track."""
    ep_id, manifests_dir = mock_episode_data

    result = split_high_spread_tracks(
        ep_id, spread_threshold=0.35, dry_run=False, manifests_dir=manifests_dir
    )

    assert result["split_count"] == 1
    assert 2 in result["track_mapping"]
    assert (
        len(result["track_mapping"][2]) >= 2
    )  # Track 2 split into at least 2 sub-tracks

    # Verify tracks file was updated
    tracks_path = manifests_dir / "tracks.jsonl"
    with tracks_path.open("r") as f:
        tracks = [json.loads(line) for line in f if line.strip()]

    # Should have: tracks 1, 3 (kept) + new sub-tracks from track 2
    assert len(tracks) >= 4  # Original 2 + at least 2 from split

    # Track 2 should not exist anymore
    track_ids = [t["track_id"] for t in tracks]
    assert 2 not in track_ids

    # New tracks should have split_from metadata
    new_tracks = [t for t in tracks if t.get("stats", {}).get("split_from") == 2]
    assert len(new_tracks) >= 2


def test_split_updates_face_assignments(mock_episode_data):
    """Test that faces are reassigned to new track IDs."""
    ep_id, manifests_dir = mock_episode_data

    # Count faces for track 2 before split
    faces_path = manifests_dir / "faces.jsonl"
    with faces_path.open("r") as f:
        faces_before = [json.loads(line) for line in f if line.strip()]
    track2_faces_before = [f for f in faces_before if f["track_id"] == 2]
    assert len(track2_faces_before) == 20

    result = split_high_spread_tracks(
        ep_id, spread_threshold=0.35, dry_run=False, manifests_dir=manifests_dir
    )

    # After split, track 2 faces should be reassigned
    with faces_path.open("r") as f:
        faces_after = [json.loads(line) for line in f if line.strip()]

    track2_faces_after = [f for f in faces_after if f["track_id"] == 2]
    assert len(track2_faces_after) == 0  # No more faces assigned to track 2

    # Faces should be reassigned to new track IDs
    new_track_ids = result["track_mapping"][2]
    reassigned_faces = [f for f in faces_after if f["track_id"] in new_track_ids]
    assert len(reassigned_faces) == 20  # All 20 faces should be reassigned


def test_no_split_needed(mock_episode_data):
    """Test behavior when no tracks exceed threshold."""
    ep_id, manifests_dir = mock_episode_data

    result = split_high_spread_tracks(
        ep_id, spread_threshold=1.0, dry_run=False, manifests_dir=manifests_dir
    )

    assert result["split_count"] == 0
    assert len(result.get("flagged_tracks", [])) == 0


def test_split_preserves_non_flagged_tracks(mock_episode_data):
    """Test that non-flagged tracks are preserved unchanged."""
    ep_id, manifests_dir = mock_episode_data

    split_high_spread_tracks(
        ep_id, spread_threshold=0.35, dry_run=False, manifests_dir=manifests_dir
    )

    tracks_path = manifests_dir / "tracks.jsonl"
    with tracks_path.open("r") as f:
        tracks = [json.loads(line) for line in f if line.strip()]

    # Tracks 1 and 3 should still exist with original IDs
    track_ids = [t["track_id"] for t in tracks]
    assert 1 in track_ids
    assert 3 in track_ids

    # Their metadata should be unchanged
    track1 = next(t for t in tracks if t["track_id"] == 1)
    assert track1["face_embedding_spread"] == 0.2
    track3 = next(t for t in tracks if t["track_id"] == 3)
    assert track3["face_embedding_spread"] == 0.15
