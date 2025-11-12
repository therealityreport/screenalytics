"""Test that People Review renders one tile per track with similarity badges."""

def test_similarity_badge_rendering():
    """Test that similarity badge renders correctly with color coding."""
    import sys
    from pathlib import Path

    # Add UI pages to path
    workspace_ui = Path(__file__).resolve().parents[2] / "apps" / "workspace-ui"
    pages_dir = workspace_ui / "pages"
    if str(pages_dir) not in sys.path:
        sys.path.insert(0, str(pages_dir))

    # Import the render function (need to handle streamlit mock)
    try:
        # Mock streamlit before import
        import unittest.mock as mock
        sys.modules['streamlit'] = mock.MagicMock()

        # Now we can import
        from importlib import import_module
        faces_review = import_module("3_Faces_Review")

        # Test high similarity (>= 0.75) - should be green
        badge_high = faces_review._render_similarity_badge(0.85)
        assert "green" in badge_high.lower()
        assert "85%" in badge_high

        # Test medium similarity (0.60 - 0.75) - should be orange
        badge_medium = faces_review._render_similarity_badge(0.67)
        assert "orange" in badge_medium.lower()
        assert "67%" in badge_medium

        # Test low similarity (< 0.60) - should be red
        badge_low = faces_review._render_similarity_badge(0.45)
        assert "red" in badge_low.lower()
        assert "45%" in badge_low

        # Test None - should return empty
        badge_none = faces_review._render_similarity_badge(None)
        assert badge_none == ""

    except ImportError as e:
        # Skip test if streamlit or other dependencies not available in test environment
        import pytest
        pytest.skip(f"Streamlit mock failed: {e}")


def test_people_clusters_endpoint_integration():
    """Integration test: verify endpoint returns all tracks without limit."""
    import json
    from pathlib import Path
    import tempfile

    from apps.api.services.track_reps import (
        write_track_reps,
        write_cluster_centroids,
        build_cluster_track_reps,
    )

    ep_id = "test-s01e01"

    # Create 20 track representatives (more than old limit of 5)
    track_reps_list = []
    for i in range(1, 21):
        track_reps_list.append({
            "track_id": f"track_{i:04d}",
            "rep_frame": 100 + i,
            "crop_key": f"crops/track_{i:04d}/frame_{100+i:06d}.jpg",
            "embed": [0.1] * 512,
            "quality": {"det": 0.8, "std": 10.0},
        })

    # Create cluster with all 20 tracks
    cluster_centroids = {
        "id_0001": {
            "centroid": [0.1] * 512,
            "tracks": [f"track_{i:04d}" for i in range(1, 21)],
            "cohesion": 0.82,
        }
    }

    track_reps_map = {rep["track_id"]: rep for rep in track_reps_list}
    result = build_cluster_track_reps(ep_id, "id_0001", track_reps_map, cluster_centroids)

    # Verify all 20 tracks are returned (no limit to 5)
    assert result["total_tracks"] == 20, f"Should return all 20 tracks, got {result['total_tracks']}"
    assert len(result["tracks"]) == 20, f"Should have 20 track items, got {len(result['tracks'])}"

    # Verify each track has similarity badge data
    for track in result["tracks"]:
        assert "similarity" in track, f"Track {track['track_id']} missing similarity"
        assert "crop_key" in track, f"Track {track['track_id']} missing crop_key"
        assert track["similarity"] is not None, f"Track {track['track_id']} has None similarity"

    # Verify no track was silently dropped
    returned_track_ids = {t["track_id"] for t in result["tracks"]}
    expected_track_ids = {f"track_{i:04d}" for i in range(1, 21)}
    assert returned_track_ids == expected_track_ids, "Some tracks were not returned"
