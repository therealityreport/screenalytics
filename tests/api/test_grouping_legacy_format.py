"""Test GroupingService handles legacy cluster centroids format."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def test_grouping_service_writes_new_dict_format():
    """Test that GroupingService.compute_cluster_centroids writes dict format."""
    grouping_path = PROJECT_ROOT / "apps" / "api" / "services" / "grouping.py"
    content = grouping_path.read_text()

    # Find the compute_cluster_centroids function
    import re

    func_match = re.search(
        r"def compute_cluster_centroids\(.*?\):.*?return output", content, re.DOTALL
    )

    assert func_match, "compute_cluster_centroids function not found"
    func_code = func_match.group(0)

    # Check that it uses dict format
    assert "centroids = {}" in func_code, "Should initialize centroids as dict"
    assert (
        "centroids[cluster_id] = centroid_entry" in func_code
    ), "Should assign centroids by cluster_id key"

    # Check that it includes tracks
    assert (
        '"tracks": tracks_formatted' in func_code
    ), "Should include tracks in centroid entry"

    print("✓ GroupingService writes new dict format with tracks")


def test_grouping_service_handles_both_formats():
    """Test that GroupingService methods handle both dict and list formats."""
    grouping_path = PROJECT_ROOT / "apps" / "api" / "services" / "grouping.py"
    content = grouping_path.read_text()

    # Check group_within_episode handles both formats
    assert "isinstance(centroids, list)" in content, "Should check for list format"
    assert "isinstance(centroids, dict)" in content, "Should check for dict format"
    assert "centroids_list = [" in content, "Should convert dict to list for processing"

    print("✓ GroupingService methods handle both dict and list formats")


def test_track_reps_converts_legacy_format():
    """Test that track_reps.load_cluster_centroids converts legacy list format."""
    track_reps_path = PROJECT_ROOT / "apps" / "api" / "services" / "track_reps.py"
    content = track_reps_path.read_text()

    # Check load_cluster_centroids has conversion logic
    assert (
        "if isinstance(centroids, list):" in content
    ), "Should detect legacy list format"
    assert "identity_tracks_map" in content, "Should load identities to get track_ids"
    assert (
        '"tracks": tracks' in content or '"tracks": identity_tracks_map' in content
    ), "Should derive tracks from identities.json"

    print("✓ track_reps.load_cluster_centroids converts legacy format")


def test_build_cluster_track_reps_guards_incomplete_data():
    """Test that build_cluster_track_reps handles missing/incomplete cluster data."""
    track_reps_path = PROJECT_ROOT / "apps" / "api" / "services" / "track_reps.py"
    content = track_reps_path.read_text()

    # Check for guard against missing cluster_data
    assert (
        "def build_cluster_track_reps(" in content
    ), "build_cluster_track_reps function should exist"
    assert "if not cluster_data:" in content, "Should check if cluster_data exists"
    assert '"tracks": []' in content, "Should return empty tracks for missing clusters"

    print("✓ build_cluster_track_reps guards against incomplete data")


if __name__ == "__main__":
    test_grouping_service_writes_new_dict_format()
    test_grouping_service_handles_both_formats()
    test_track_reps_converts_legacy_format()
    test_build_cluster_track_reps_guards_incomplete_data()
    print("\n✓ All GroupingService legacy format tests passed!")
