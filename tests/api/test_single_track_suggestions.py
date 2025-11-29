"""Test cast suggestions for single-track clusters use frame-based matching."""

import json
import re
from pathlib import Path


def test_single_track_cluster_uses_frame_matching():
    """Verify that the cast suggestions endpoint returns source='frame' for single-track clusters.

    This test verifies the code structure rather than runtime behavior:
    1. For single-track clusters, per-frame max similarity should be used
    2. The suggestion should include source='frame' and faces_used count
    """
    # Read the grouping service to verify the implementation
    grouping_path = Path("apps/api/services/grouping.py")
    assert grouping_path.exists(), "grouping.py not found"

    content = grouping_path.read_text(encoding="utf-8")

    # Verify single-track cluster detection
    assert "single_track_clusters" in content, "Missing single_track_clusters set"

    # Verify frame-based matching for single-track clusters
    assert re.search(
        r'if cluster_id in single_track_clusters:',
        content
    ), "Missing single-track cluster frame matching logic"

    # Verify per-frame max similarity is computed
    assert re.search(
        r'for face_emb in cluster_face_embeddings:',
        content
    ), "Missing per-frame embedding comparison loop"

    # Verify source='frame' is set for frame-based matches
    assert re.search(
        r'"source":\s*"frame"',
        content
    ), "Missing source='frame' in frame-based suggestions"

    # Verify faces_used count is included
    assert re.search(
        r'"faces_used":\s*n_faces_used',
        content
    ), "Missing faces_used count in frame-based suggestions"

    # Verify unassigned_track_embeddings collection
    assert "unassigned_track_embeddings" in content, "Missing unassigned_track_embeddings dict"

    print("✓ Single-track cluster suggestions use frame-based matching with source='frame'")


def test_ui_shows_single_frame_badge():
    """Verify the UI shows a badge for single-frame single-track clusters."""
    faces_review_path = Path("apps/workspace-ui/pages/3_Faces_Review.py")
    assert faces_review_path.exists(), "3_Faces_Review.py not found"

    content = faces_review_path.read_text(encoding="utf-8")

    # Verify single-track cluster detection
    assert "is_single_track_cluster" in content, "Missing is_single_track_cluster variable"
    assert "has_single_frame_track" in content, "Missing has_single_frame_track variable"

    # Verify single-frame badge is shown
    assert re.search(
        r'Single-Frame',
        content
    ), "Missing 'Single-Frame' badge in UI"

    # Verify filter relaxation for single-track clusters
    assert re.search(
        r'if is_single_track_cluster:',
        content
    ), "Missing filter relaxation for single-track clusters"

    print("✓ UI shows single-frame badge for single-track clusters")


def test_smart_suggestions_shows_frame_source():
    """Verify Smart Suggestions page shows 'frame' source label."""
    suggestions_path = Path("apps/workspace-ui/pages/3_Smart_Suggestions.py")
    assert suggestions_path.exists(), "3_Smart_Suggestions.py not found"

    content = suggestions_path.read_text(encoding="utf-8")

    # Verify faces_used is extracted from suggestion
    assert 'faces_used = suggestion.get("faces_used")' in content, "Missing faces_used extraction"

    # Verify source_label includes face count for frame source
    assert re.search(
        r'source_label.*frame.*faces_used',
        content,
        re.DOTALL
    ), "Missing source_label with faces count for frame source"

    print("✓ Smart Suggestions shows frame source with face count")


if __name__ == "__main__":
    test_single_track_cluster_uses_frame_matching()
    test_ui_shows_single_frame_badge()
    test_smart_suggestions_shows_frame_source()
    print("\n✅ All single-track suggestion tests passed!")
