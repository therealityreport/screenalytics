"""Tests for numeric track_id parsing across the system."""

from __future__ import annotations

import sys
from pathlib import Path


# Add FEATURES/tracking to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACKING_SRC = PROJECT_ROOT / "FEATURES" / "tracking" / "src"
if str(TRACKING_SRC) not in sys.path:
    sys.path.append(str(TRACKING_SRC))

from bytetrack_runner import Track, build_tracks


def test_track_to_record_emits_numeric_id():
    """Test that Track.to_record emits numeric track_id (not string)."""
    track = Track(
        track_id=42,
        ep_id="test-s01e01",
        start_s=0.0,
        end_s=1.0,
        start_frame=0,
        end_frame=30,
        bbox=[100, 100, 200, 200],
        last_frame_idx=30,
        hits=5,
        conf_sum=4.5,
    )

    record = track.to_record()

    # Verify track_id is numeric (int), not a formatted string
    assert isinstance(record["track_id"], int), f"track_id should be int, got {type(record['track_id'])}"
    assert record["track_id"] == 42

    # Verify it can be cast to int (redundant but explicit)
    assert int(record["track_id"]) == 42


def test_build_tracks_emits_parseable_track_ids():
    """Test that build_tracks produces records with int-parseable track_ids."""
    detections = [
        {
            "frame_idx": 0,
            "ts_s": 0.0,
            "bbox": [100, 100, 200, 200],
            "conf": 0.9,
            "ep_id": "test-s01e01",
        },
        {
            "frame_idx": 1,
            "ts_s": 0.033,
            "bbox": [102, 102, 202, 202],
            "conf": 0.85,
            "ep_id": "test-s01e01",
        },
        {
            "frame_idx": 50,
            "ts_s": 1.666,
            "bbox": [300, 300, 400, 400],
            "conf": 0.95,
            "ep_id": "test-s01e01",
        },
    ]

    cfg = {
        "track_thresh": 0.5,
        "match_thresh": 0.8,
        "track_buffer": 30,
    }

    tracks = list(build_tracks(detections, cfg))

    assert len(tracks) >= 1, "Should produce at least one track"

    for track in tracks:
        track_id = track["track_id"]

        # Verify track_id is numeric type
        assert isinstance(track_id, int), f"track_id should be int, got {type(track_id)}: {track_id}"

        # Verify it can be parsed with int() (critical for downstream)
        parsed = int(track_id)
        assert parsed > 0, f"track_id should be positive, got {parsed}"


def test_track_id_can_be_formatted_as_track_XXXX():
    """Test that numeric track_id can be formatted as track_XXXX."""
    track = Track(
        track_id=7,
        ep_id="test-s01e01",
        start_s=0.0,
        end_s=1.0,
        start_frame=0,
        end_frame=30,
        bbox=[100, 100, 200, 200],
        last_frame_idx=30,
    )

    record = track.to_record()
    track_id = record["track_id"]

    # Verify we can format it as expected by identities service
    formatted = f"track_{int(track_id):04d}"
    assert formatted == "track_0007"

    # Verify we can extract the numeric part
    assert int(track_id) == 7


def test_track_id_compatible_with_screentime_service():
    """Test that track_id format works with screentime mapping logic."""
    # Simulate screentime service behavior (apps/api/services/screentime.py:282)
    track_records = [
        {"track_id": 1},
        {"track_id": 2},
        {"track_id": 15},
    ]

    # Build mapping as screentime does: mapping[int(track_id)] = identity_id
    mapping = {}
    for track in track_records:
        track_id = track["track_id"]
        # This should not raise ValueError
        mapping[int(track_id)] = f"identity_{track_id}"

    assert len(mapping) == 3
    assert mapping[1] == "identity_1"
    assert mapping[2] == "identity_2"
    assert mapping[15] == "identity_15"


def test_track_id_compatible_with_identities_service():
    """Test that track_id format works with identities service formatting."""

    # Simulate identities service behavior (apps/api/services/identities.py:225)
    def format_track_id(track_id):
        """From identities.py: return f'track_{int(track_id):04d}'"""
        return f"track_{int(track_id):04d}"

    # Test with numeric track_ids from ByteTrackLite
    track_records = [
        {"track_id": 1},
        {"track_id": 42},
        {"track_id": 999},
    ]

    formatted_ids = [format_track_id(t["track_id"]) for t in track_records]

    assert formatted_ids[0] == "track_0001"
    assert formatted_ids[1] == "track_0042"
    assert formatted_ids[2] == "track_0999"


def test_track_id_compatible_with_storage_service():
    """Test that track_id format works with storage service path generation."""

    # Simulate storage service behavior (apps/api/services/storage.py:734)
    def get_track_prefix(track_id):
        """From storage.py: return f'track_{max(int(track_id), 0):04d}/'"""
        return f"track_{max(int(track_id), 0):04d}/"

    # Test with numeric track_ids
    track_records = [
        {"track_id": 5},
        {"track_id": 123},
    ]

    prefixes = [get_track_prefix(t["track_id"]) for t in track_records]

    assert prefixes[0] == "track_0005/"
    assert prefixes[1] == "track_0123/"
