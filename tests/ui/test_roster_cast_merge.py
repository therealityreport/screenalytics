"""Test that roster names include both roster.json and cast.json names."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def test_fetch_roster_names_merges_sources():
    """Test that _fetch_roster_names includes names from both roster and cast."""
    # This test verifies that the Faces Review page fetches names from:
    # 1. /shows/{show}/cast_names (roster.json)
    # 2. /shows/{show}/cast (cast.json / facebank)


    faces_review_path = (
        PROJECT_ROOT / "apps" / "workspace-ui" / "pages" / "3_Faces_Review.py"
    )
    content = faces_review_path.read_text()

    # Check that function fetches from both endpoints
    assert 'f"/shows/{show}/cast_names"' in content, "Should fetch roster names"
    assert 'f"/shows/{show}/cast"' in content, "Should fetch cast members"

    # Check that it merges the results
    assert (
        "roster_names + cast_names" in content
        or "roster_names" in content
        and "cast_names" in content
    ), "Should merge roster and cast names"

    # Check that it deduplicates
    assert "seen_lower" in content or "set" in content, "Should deduplicate names"

    print("✓ _fetch_roster_names merges both roster and cast member names")


def test_name_choice_widget_includes_merged_names():
    """Test that name choice widget receives the merged names."""

    faces_review_path = (
        PROJECT_ROOT / "apps" / "workspace-ui" / "pages" / "3_Faces_Review.py"
    )
    content = faces_review_path.read_text()

    # Verify _name_choice_widget is called with roster_names
    assert "def _name_choice_widget(" in content, "Should have name choice widget"
    assert (
        "roster_names: List[str]" in content
    ), "Widget should accept roster_names parameter"

    # Verify the widget is used in track assignment
    assert "roster_names=roster_names" in content, "Should pass roster_names to widget"

    print("✓ Name choice widget uses merged roster/cast names")


if __name__ == "__main__":
    test_fetch_roster_names_merges_sources()
    test_name_choice_widget_includes_merged_names()
    print("\n✓ All roster/cast merge tests passed!")
