"""Unit tests for ScreenTimeAnalyzer service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from apps.api.services.screentime import ScreenTimeAnalyzer, ScreenTimeConfig


@pytest.fixture
def temp_episode_data(tmp_path: Path, monkeypatch) -> tuple[str, Path]:
    """Create synthetic episode data for testing.

    Returns:
        (ep_id, data_root)
    """
    ep_id = "test-s01e01"
    show_id = "TEST"

    # Setup directory structure matching get_path() expectations
    # get_path() uses: data/manifests/ep_id/ and data/shows/show_id/
    data_root = tmp_path / "data"
    manifests_dir = data_root / "manifests" / ep_id
    shows_dir = data_root / "shows" / show_id

    manifests_dir.mkdir(parents=True, exist_ok=True)
    shows_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variable in fixture so all uses have correct path
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    # Create synthetic faces.jsonl
    # Cast member "alice" has:
    # - Track 1: faces at 1.0s, 1.5s, 2.0s (3 consecutive, gap < 0.5s)
    # - Track 2: faces at 5.0s, 6.0s (2 faces, gap = 1.0s > 0.5s tolerance)
    # Total expected: (2.0 - 1.0) + (6.0 - 5.0) = 1.0 + 1.0 = 2.0s
    #
    # Non-cast person "bob" should be ignored:
    # - Track 3: faces at 10.0s, 11.0s
    faces_data = [
        # Alice - Track 1 (continuous interval)
        {"track_id": 1, "frame_idx": 30, "ts": 1.0, "quality": 0.9},
        {"track_id": 1, "frame_idx": 45, "ts": 1.5, "quality": 0.85},
        {"track_id": 1, "frame_idx": 60, "ts": 2.0, "quality": 0.95},
        # Alice - Track 2 (two separate intervals due to gap)
        {"track_id": 2, "frame_idx": 150, "ts": 5.0, "quality": 0.8},
        {"track_id": 2, "frame_idx": 180, "ts": 6.0, "quality": 0.9},
        # Bob - Track 3 (should be ignored, no cast_id)
        {"track_id": 3, "frame_idx": 300, "ts": 10.0, "quality": 0.95},
        {"track_id": 3, "frame_idx": 330, "ts": 11.0, "quality": 0.9},
    ]

    faces_path = manifests_dir / "faces.jsonl"
    with faces_path.open("w", encoding="utf-8") as f:
        for face in faces_data:
            f.write(json.dumps(face) + "\n")

    # Create synthetic tracks.jsonl
    tracks_data = [
        {
            "track_id": 1,
            "start_frame": 30,
            "end_frame": 60,
            "first_frame_idx": 30,
            "last_frame_idx": 60,
            "frame_count": 3,
            "faces_count": 3,
            "first_ts": 1.0,
            "last_ts": 2.0,
        },
        {
            "track_id": 2,
            "start_frame": 150,
            "end_frame": 180,
            "first_frame_idx": 150,
            "last_frame_idx": 180,
            "frame_count": 2,
            "faces_count": 2,
            "first_ts": 5.0,
            "last_ts": 6.0,
        },
        {
            "track_id": 3,
            "start_frame": 300,
            "end_frame": 330,
            "first_frame_idx": 300,
            "last_frame_idx": 330,
            "frame_count": 2,
            "faces_count": 2,
            "first_ts": 10.0,
            "last_ts": 11.0,
        },
    ]

    tracks_path = manifests_dir / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as f:
        for track in tracks_data:
            f.write(json.dumps(track) + "\n")

    # Create synthetic identities.json
    # Identity "identity_alice" has tracks 1 and 2
    # Identity "identity_bob" has track 3
    identities_data = {
        "identities": [
            {
                "identity_id": "identity_alice",
                "track_ids": [1, 2],
            },
            {
                "identity_id": "identity_bob",
                "track_ids": [3],
            },
        ]
    }

    identities_path = manifests_dir / "identities.json"
    with identities_path.open("w", encoding="utf-8") as f:
        json.dump(identities_data, f, indent=2)

    # Create synthetic people.json
    # Alice has cast_id, Bob does not
    people_data = {
        "people": [
            {
                "person_id": "person_alice",
                "name": "Alice",
                "cast_id": "cast_alice",
                "cluster_ids": [f"{ep_id}:identity_alice"],
            },
            {
                "person_id": "person_bob",
                "name": "Bob",
                # No cast_id - should be ignored
                "cluster_ids": [f"{ep_id}:identity_bob"],
            },
        ]
    }

    people_path = shows_dir / "people.json"
    with people_path.open("w", encoding="utf-8") as f:
        json.dump(people_data, f, indent=2)

    return ep_id, data_root


def test_screentime_basic_analysis(temp_episode_data: tuple[str, Path]):
    """Test basic screen time analysis with synthetic data."""
    ep_id, data_root = temp_episode_data

    # Create analyzer with default config
    config = ScreenTimeConfig(
        quality_min=0.7,
        gap_tolerance_s=0.5,
        use_video_decode=True,
    )
    analyzer = ScreenTimeAnalyzer(config)

    # Run analysis
    result = analyzer.analyze_episode(ep_id)

    # Verify structure
    assert result["episode_id"] == ep_id
    assert "generated_at" in result
    assert "metrics" in result

    metrics = result["metrics"]

    # Should have exactly 1 cast member (Alice)
    assert len(metrics) == 1

    alice_metrics = metrics[0]
    assert alice_metrics["cast_id"] == "cast_alice"
    assert alice_metrics["person_id"] == "person_alice"

    # Verify screen time calculation
    # Track 1: 1.0s -> 1.5s -> 2.0s (continuous, gaps ≤ 0.5s)
    #   Duration: 2.0 - 1.0 = 1.0s
    # Track 2: 5.0s -> 6.0s (gap = 1.0s > 0.5s tolerance, so two separate intervals)
    #   Intervals: (5.0, 5.0) = 0.0s and (6.0, 6.0) = 0.0s
    # Total: 1.0s
    assert alice_metrics["visual_s"] == 1.0

    # Verify track and face counts
    assert alice_metrics["tracks_count"] == 2  # tracks 1 and 2
    assert alice_metrics["faces_count"] == 5  # 3 from track 1, 2 from track 2

    # Verify speaking_s and both_s are 0 (not implemented yet)
    assert alice_metrics["speaking_s"] == 0.0
    assert alice_metrics["both_s"] == 0.0

    # Verify confidence is calculated
    assert alice_metrics["confidence"] > 0.0
    assert alice_metrics["confidence"] <= 1.0


def test_screentime_ignores_non_cast(temp_episode_data: tuple[str, Path]):
    """Test that identities without cast_id are ignored."""
    ep_id, data_root = temp_episode_data

    analyzer = ScreenTimeAnalyzer()
    result = analyzer.analyze_episode(ep_id)

    metrics = result["metrics"]

    # Should only have Alice (with cast_id), not Bob (without cast_id)
    assert len(metrics) == 1
    assert metrics[0]["cast_id"] == "cast_alice"

    # Verify Bob's person_id is not in results
    person_ids = [m["person_id"] for m in metrics]
    assert "person_bob" not in person_ids


def test_screentime_quality_threshold(temp_episode_data: tuple[str, Path], monkeypatch):
    """Test that quality threshold filters faces correctly."""
    ep_id, data_root = temp_episode_data
    monkeypatch.setenv("SCREANALYTICS_DATA_ROOT", str(data_root))

    # Use high quality threshold to filter out some faces
    config = ScreenTimeConfig(
        quality_min=0.9,  # Will filter out faces with quality < 0.9
        gap_tolerance_s=0.5,
        use_video_decode=True,
    )
    analyzer = ScreenTimeAnalyzer(config)

    result = analyzer.analyze_episode(ep_id)
    metrics = result["metrics"]

    assert len(metrics) == 1
    alice_metrics = metrics[0]

    # With quality_min=0.9:
    # Track 1: keeps 1.0s (0.9), 2.0s (0.95); drops 1.5s (0.85)
    #   But 1.0s -> 2.0s gap is 1.0s > 0.5s tolerance, so two intervals
    #   Interval 1: 1.0s - 1.0s = 0.0s
    #   Interval 2: 2.0s - 2.0s = 0.0s
    #   Total: 0.0s
    # Track 2: keeps 6.0s (0.9); drops 5.0s (0.8)
    #   Single point: 6.0s - 6.0s = 0.0s
    # Total: 0.0s
    assert alice_metrics["visual_s"] == 0.0

    # Face count should be lower (only high-quality faces)
    # Track 1: 2 faces (quality >= 0.9)
    # Track 2: 1 face (quality >= 0.9)
    # Total: 3 faces
    assert alice_metrics["faces_count"] == 5  # Total faces processed, not filtered


def test_screentime_gap_tolerance(temp_episode_data: tuple[str, Path], monkeypatch):
    """Test that gap tolerance correctly groups intervals."""
    ep_id, data_root = temp_episode_data
    monkeypatch.setenv("SCREANALYTICS_DATA_ROOT", str(data_root))

    # Use large gap tolerance to merge all faces
    config = ScreenTimeConfig(
        quality_min=0.7,
        gap_tolerance_s=10.0,  # Large enough to merge everything
        use_video_decode=True,
    )
    analyzer = ScreenTimeAnalyzer(config)

    result = analyzer.analyze_episode(ep_id)
    metrics = result["metrics"]

    assert len(metrics) == 1
    alice_metrics = metrics[0]

    # With large gap tolerance (10.0s), both tracks merge (1.0s -> 6.0s)
    assert alice_metrics["visual_s"] == 5.0


def test_screentime_track_mode_uses_track_spans(temp_episode_data: tuple[str, Path]):
    """Track-based mode should use track spans plus padding."""
    ep_id, _ = temp_episode_data

    config = ScreenTimeConfig(
        quality_min=0.48,
        gap_tolerance_s=1.2,
        use_video_decode=True,
        screen_time_mode="tracks",
        edge_padding_s=0.2,
        track_coverage_min=0.0,
    )
    analyzer = ScreenTimeAnalyzer(config)

    result = analyzer.analyze_episode(ep_id)
    metrics = result["metrics"]

    assert len(metrics) == 1
    alice_metrics = metrics[0]

    # Track spans are 1.0s each, plus 0.2s padding on both ends => 1.4s per track.
    assert alice_metrics["visual_s"] == pytest.approx(2.8, rel=0.01)


def test_screentime_track_mode_honors_coverage_gate(
    temp_episode_data: tuple[str, Path],
):
    """Track-based mode should drop tracks with low coverage."""
    ep_id, data_root = temp_episode_data
    manifests_dir = data_root / "manifests" / ep_id
    tracks_path = manifests_dir / "tracks.jsonl"

    tracks = []
    with tracks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tracks.append(json.loads(line))

    for track in tracks:
        if track["track_id"] == 2:
            track["frame_count"] = (
                200  # artificially inflate duration to reduce coverage
            )
            track["last_frame_idx"] = track["first_frame_idx"] + 200

    with tracks_path.open("w", encoding="utf-8") as f:
        for track in tracks:
            f.write(json.dumps(track) + "\n")

    config = ScreenTimeConfig(
        quality_min=0.48,
        gap_tolerance_s=1.2,
        use_video_decode=True,
        screen_time_mode="tracks",
        edge_padding_s=0.0,
        track_coverage_min=0.2,
    )
    analyzer = ScreenTimeAnalyzer(config)

    result = analyzer.analyze_episode(ep_id)
    metrics = result["metrics"]

    assert len(metrics) == 1
    alice_metrics = metrics[0]

    # Track 2 fails the coverage gate, so only Track 1 contributes (~1.0s).
    assert alice_metrics["visual_s"] == pytest.approx(1.0, rel=0.01)


def test_screentime_write_outputs(temp_episode_data: tuple[str, Path], monkeypatch):
    """Test that outputs are written correctly to JSON and CSV."""
    ep_id, data_root = temp_episode_data
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    analyzer = ScreenTimeAnalyzer()
    metrics_data = analyzer.analyze_episode(ep_id)

    # Write outputs
    json_path, csv_path = analyzer.write_outputs(ep_id, metrics_data)

    # Verify paths exist
    assert json_path.exists()
    assert csv_path.exists()

    # Verify JSON content
    with json_path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)

    assert json_data["episode_id"] == ep_id
    assert "generated_at" in json_data
    assert len(json_data["metrics"]) == 1

    # Verify CSV content
    csv_content = csv_path.read_text(encoding="utf-8")
    lines = csv_content.strip().split("\n")

    # Should have header + 1 data row
    assert len(lines) == 2

    header = lines[0]
    assert "cast_id" in header
    assert "person_id" in header
    assert "visual_s" in header
    assert "confidence" in header

    data_row = lines[1]
    assert "cast_alice" in data_row
    assert "person_alice" in data_row


def test_screentime_missing_artifacts(monkeypatch, tmp_path: Path):
    """Test that missing artifacts raise appropriate errors."""
    ep_id = "test-s01e02"
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREANALYTICS_DATA_ROOT", str(data_root))

    analyzer = ScreenTimeAnalyzer()

    # Should raise FileNotFoundError for missing faces.jsonl
    with pytest.raises(FileNotFoundError, match="faces.jsonl not found"):
        analyzer.analyze_episode(ep_id)


def test_screentime_confidence_calculation(
    temp_episode_data: tuple[str, Path], monkeypatch
):
    """Test that confidence is calculated based on tracks and faces."""
    ep_id, data_root = temp_episode_data
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    analyzer = ScreenTimeAnalyzer()
    result = analyzer.analyze_episode(ep_id)

    metrics = result["metrics"]
    alice_metrics = metrics[0]

    # Confidence formula: min(1.0, 0.5 + (tracks_count * 0.05))
    # tracks_count = 2
    # confidence = 0.5 + (2 * 0.05) = 0.6
    expected_confidence = min(1.0, 0.5 + (2 * 0.05))
    assert alice_metrics["confidence"] == expected_confidence


def test_screentime_multiple_cast_members(tmp_path: Path, monkeypatch):
    """Test screen time analysis with multiple cast members."""
    ep_id = "test-s01e03"
    show_id = "TEST"

    # Setup directory structure matching get_path() expectations
    data_root = tmp_path / "data"
    manifests_dir = data_root / "manifests" / ep_id
    shows_dir = data_root / "shows" / show_id

    manifests_dir.mkdir(parents=True, exist_ok=True)
    shows_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variable early so get_path uses correct root
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    # Create faces for two cast members
    faces_data = [
        # Alice - Track 1 (gap = 0.5s, within tolerance)
        {"track_id": 1, "frame_idx": 30, "ts": 1.0, "quality": 0.9},
        {"track_id": 1, "frame_idx": 45, "ts": 1.5, "quality": 0.9},
        {"track_id": 1, "frame_idx": 60, "ts": 2.0, "quality": 0.9},
        # Bob - Track 2 (gap = 0.5s, within tolerance) - has more screen time
        {"track_id": 2, "frame_idx": 90, "ts": 3.0, "quality": 0.9},
        {"track_id": 2, "frame_idx": 105, "ts": 3.5, "quality": 0.9},
        {"track_id": 2, "frame_idx": 120, "ts": 4.0, "quality": 0.9},
        {"track_id": 2, "frame_idx": 135, "ts": 4.5, "quality": 0.9},
        {"track_id": 2, "frame_idx": 150, "ts": 5.0, "quality": 0.9},
    ]

    faces_path = manifests_dir / "faces.jsonl"
    with faces_path.open("w", encoding="utf-8") as f:
        for face in faces_data:
            f.write(json.dumps(face) + "\n")

    tracks_data = [
        {"track_id": 1, "start_frame": 30, "end_frame": 60},
        {"track_id": 2, "start_frame": 90, "end_frame": 150},
    ]

    tracks_path = manifests_dir / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as f:
        for track in tracks_data:
            f.write(json.dumps(track) + "\n")

    identities_data = {
        "identities": [
            {"identity_id": "identity_alice", "track_ids": [1]},
            {"identity_id": "identity_bob", "track_ids": [2]},
        ]
    }

    identities_path = manifests_dir / "identities.json"
    with identities_path.open("w", encoding="utf-8") as f:
        json.dump(identities_data, f, indent=2)

    people_data = {
        "people": [
            {
                "person_id": "person_alice",
                "name": "Alice",
                "cast_id": "cast_alice",
                "cluster_ids": [f"{ep_id}:identity_alice"],
            },
            {
                "person_id": "person_bob",
                "name": "Bob",
                "cast_id": "cast_bob",
                "cluster_ids": [f"{ep_id}:identity_bob"],
            },
        ]
    }

    people_path = shows_dir / "people.json"
    with people_path.open("w", encoding="utf-8") as f:
        json.dump(people_data, f, indent=2)

    monkeypatch.setenv("SCREANALYTICS_DATA_ROOT", str(data_root))

    analyzer = ScreenTimeAnalyzer()
    result = analyzer.analyze_episode(ep_id)

    metrics = result["metrics"]

    # Should have 2 cast members
    assert len(metrics) == 2

    # Metrics should be sorted by visual_s (descending)
    cast_ids = [m["cast_id"] for m in metrics]
    assert "cast_alice" in cast_ids
    assert "cast_bob" in cast_ids

    # Alice: 2.0 - 1.0 = 1.0s (faces at 1.0s, 1.5s, 2.0s - continuous)
    # Bob: 5.0 - 3.0 = 2.0s (faces at 3.0s, 3.5s, 4.0s, 4.5s, 5.0s - continuous)
    # Bob should be first (higher screen time)
    assert metrics[0]["cast_id"] == "cast_bob"
    assert metrics[0]["visual_s"] == 2.0
    assert metrics[1]["cast_id"] == "cast_alice"
    assert metrics[1]["visual_s"] == 1.0


def test_screentime_empty_results(tmp_path: Path, monkeypatch):
    """Test screen time analysis with no cast members."""
    ep_id = "test-s01e04"
    show_id = "TEST"

    # Setup directory structure matching get_path() expectations
    data_root = tmp_path / "data"
    manifests_dir = data_root / "manifests" / ep_id
    shows_dir = data_root / "shows" / show_id

    manifests_dir.mkdir(parents=True, exist_ok=True)
    shows_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variable early so get_path uses correct root
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    # Create minimal empty artifacts
    faces_data = []
    faces_path = manifests_dir / "faces.jsonl"
    with faces_path.open("w", encoding="utf-8") as f:
        for face in faces_data:
            f.write(json.dumps(face) + "\n")

    tracks_data = []
    tracks_path = manifests_dir / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as f:
        for track in tracks_data:
            f.write(json.dumps(track) + "\n")

    identities_data = {"identities": []}
    identities_path = manifests_dir / "identities.json"
    with identities_path.open("w", encoding="utf-8") as f:
        json.dump(identities_data, f, indent=2)

    people_data = {"people": []}
    people_path = shows_dir / "people.json"
    with people_path.open("w", encoding="utf-8") as f:
        json.dump(people_data, f, indent=2)

    analyzer = ScreenTimeAnalyzer()
    result = analyzer.analyze_episode(ep_id)

    # Should return empty metrics list
    assert result["episode_id"] == ep_id
    assert result["metrics"] == []
