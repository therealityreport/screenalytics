"""Test cluster cleanup progress logging functionality."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def test_compute_centroids_has_progress_callback():
    """Test that compute_cluster_centroids accepts progress_callback parameter."""
    grouping_path = PROJECT_ROOT / "apps" / "api" / "services" / "grouping.py"
    content = grouping_path.read_text()

    # Find the compute_cluster_centroids function definition (multi-line)
    import re

    func_pattern = r"def compute_cluster_centroids\(.*?\):"
    match = re.search(func_pattern, content, re.DOTALL)

    assert match, "compute_cluster_centroids function not found"
    func_def = match.group(0)

    # Check for progress_callback parameter
    assert (
        "progress_callback" in func_def
    ), "compute_cluster_centroids should have progress_callback parameter"

    print("✓ compute_cluster_centroids has progress_callback parameter")


def test_group_within_episode_has_progress_callback():
    """Test that group_within_episode accepts progress_callback parameter."""
    grouping_path = PROJECT_ROOT / "apps" / "api" / "services" / "grouping.py"
    content = grouping_path.read_text()

    # Find the group_within_episode function definition (multi-line)
    import re

    func_pattern = r"def group_within_episode\(.*?\):"
    match = re.search(func_pattern, content, re.DOTALL)

    assert match, "group_within_episode function not found"
    func_def = match.group(0)

    # Check for progress_callback parameter
    assert (
        "progress_callback" in func_def
    ), "group_within_episode should have progress_callback parameter"

    print("✓ group_within_episode has progress_callback parameter")


def test_grouping_service_has_progress_logging():
    """Test that GroupingService uses logger for progress messages."""
    grouping_path = PROJECT_ROOT / "apps" / "api" / "services" / "grouping.py"
    content = grouping_path.read_text()

    # Check for progress logging patterns
    assert "[cluster_cleanup]" in content, "Should have cluster_cleanup log tags"
    assert "LOGGER.info" in content, "Should use logger for progress updates"
    assert (
        "Computing centroids" in content or "computing centroids" in content.lower()
    ), "Should log centroid computation"
    assert (
        "agglomerative clustering" in content.lower()
    ), "Should log agglomerative clustering step"

    print("✓ GroupingService has progress logging")


def test_episode_cleanup_uses_progress_callbacks():
    """Test that episode_cleanup.py uses progress callbacks for grouping."""
    cleanup_path = PROJECT_ROOT / "tools" / "episode_cleanup.py"
    content = cleanup_path.read_text()

    # Check that episode_cleanup calls grouping with progress callbacks
    assert (
        "progress_callback=" in content
    ), "Should pass progress_callback to grouping methods"
    assert (
        "log_progress" in content or "log_within_progress" in content
    ), "Should define progress callback functions"

    # Check for progress logging
    assert "[cleanup]" in content, "Should have cleanup log tags"
    assert (
        "centroid progress" in content or "grouping" in content
    ), "Should log centroid/grouping progress"

    print("✓ episode_cleanup uses progress callbacks")


def test_mps_device_support_enabled():
    """Test that MPS device is supported for clustering."""
    cleanup_path = PROJECT_ROOT / "tools" / "episode_cleanup.py"
    content = cleanup_path.read_text()

    # Check that device choices include mps
    assert "mps" in content.lower(), "Should support MPS device option"

    # Check device argument
    import re

    device_pattern = r"--device.*choices.*\[.*mps"
    assert re.search(
        device_pattern, content, re.IGNORECASE | re.DOTALL
    ), "Should have MPS in device choices"

    print("✓ MPS device support is enabled")


if __name__ == "__main__":
    test_compute_centroids_has_progress_callback()
    test_group_within_episode_has_progress_callback()
    test_grouping_service_has_progress_logging()
    test_episode_cleanup_uses_progress_callbacks()
    test_mps_device_support_enabled()
    print("\n✓ All cluster cleanup progress tests passed!")
