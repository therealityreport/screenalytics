"""Regression test for legacy cluster_centroids.json format."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def test_build_cluster_track_reps_with_legacy_data():
    """Test that build_cluster_track_reps works with converted legacy centroids."""
    from apps.api.services.track_reps import build_cluster_track_reps

    ep_id = "rhobh-s01e02"

    # Create track_reps data directly
    track_reps_map = {
        "track_0001": {
            "track_id": "track_0001",
            "rep_frame": 100,
            "crop_key": "crops/track_0001/frame_000100.jpg",
            "embed": [0.1] * 512,
            "quality": {"det": 0.85, "std": 12.5},
        },
        "track_0002": {
            "track_id": "track_0002",
            "rep_frame": 200,
            "crop_key": "crops/track_0002/frame_000200.jpg",
            "embed": [0.12] * 512,
            "quality": {"det": 0.80, "std": 10.0},
        },
    }

    # Simulate converted legacy centroids (now in dict format with tracks)
    cluster_centroids = {
        "id_0001": {
            "centroid": [0.11] * 512,
            "tracks": ["track_0001", "track_0002"],  # Derived from identities.json
            "cohesion": 0.88,
        },
    }

    # Build cluster track reps
    result = build_cluster_track_reps(ep_id, "id_0001", track_reps_map, cluster_centroids)

    # Verify result structure
    assert result["cluster_id"] == "id_0001"
    assert result["cohesion"] == 0.88
    assert result["total_tracks"] == 2

    # Verify tracks are present
    tracks = result["tracks"]
    assert len(tracks) == 2, f"Expected 2 tracks, got {len(tracks)}"

    # Verify each track has similarity score
    for track in tracks:
        assert "track_id" in track
        assert "similarity" in track
        assert track["similarity"] is not None, "Track should have similarity score"

    print(f"✓ build_cluster_track_reps works with legacy data: {result['total_tracks']} tracks")


if __name__ == "__main__":
    test_build_cluster_track_reps_with_legacy_data()
    print("\n✓ Legacy centroid format tests passed!")
