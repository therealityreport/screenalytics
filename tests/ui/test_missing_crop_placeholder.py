"""Test that missing crops show placeholder instead of error message."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def test_cluster_view_shows_placeholder_for_missing_crops():
    """Test that track representatives without crops show placeholder image."""
    faces_review_path = (
        PROJECT_ROOT / "apps" / "workspace-ui" / "pages" / "3_Faces_Review.py"
    )
    content = faces_review_path.read_text()

    # Check that hide_if_missing=False is used for track representatives
    assert (
        "hide_if_missing=False" in content
    ), "Should use hide_if_missing=False to show placeholder"

    # Check that "Missing crop" caption is not shown (removed)
    # Count occurrences - should be minimal or none in track rep views
    missing_crop_count = content.count('"Missing crop"')
    assert (
        missing_crop_count == 0
    ), f"Found {missing_crop_count} 'Missing crop' captions, should show placeholder instead"

    print("✓ Track representatives show placeholder for missing crops")


def test_set_view_no_rerun_in_callback():
    """Test that _set_view doesn't call st.rerun() to avoid callback warning."""
    faces_review_path = (
        PROJECT_ROOT / "apps" / "workspace-ui" / "pages" / "3_Faces_Review.py"
    )
    content = faces_review_path.read_text()

    # Find the _set_view function definition (simple approach)
    lines = content.split("\n")
    in_set_view = False
    set_view_lines = []

    for i, line in enumerate(lines):
        if "def _set_view(" in line:
            in_set_view = True
        elif in_set_view:
            # Function ends at next def or at dedent
            if line.startswith("def ") and not line.startswith("    "):
                break
            set_view_lines.append(line)

    set_view_code = "\n".join(set_view_lines)

    assert set_view_code, "_set_view function not found"

    # Verify it doesn't call st.rerun()
    assert (
        "st.rerun()" not in set_view_code
    ), "_set_view should not call st.rerun() - causes no-op warning when used as callback"

    # Verify it sets session state
    assert "st.session_state[" in set_view_code, "_set_view should set session state"

    print("✓ _set_view doesn't call st.rerun() (avoids callback warning)")


def test_direct_set_view_calls_have_rerun():
    """Test that direct (non-callback) _set_view calls are followed by st.rerun()."""
    faces_review_path = (
        PROJECT_ROOT / "apps" / "workspace-ui" / "pages" / "3_Faces_Review.py"
    )
    content = faces_review_path.read_text()

    # Find direct _set_view calls that are not in lambda/on_click contexts
    import re

    # Look for pattern: _set_view(...) followed by return without st.rerun()
    # This pattern should have st.rerun() before return
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "_set_view(" in line and "lambda" not in line and "on_click" not in line:
            # Check if followed by return within next 2 lines
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line == "return":
                    # This is a direct call followed by return - should have st.rerun() before return
                    prev_line = lines[i].strip() if i > 0 else ""
                    assert (
                        "st.rerun()" in next_line
                        or i + 2 < len(lines)
                        and "st.rerun()" in lines[i + 2]
                    ), f"Line {i+1}: _set_view followed by return should call st.rerun()"

    print("✓ Direct _set_view calls followed by return have st.rerun()")


if __name__ == "__main__":
    test_cluster_view_shows_placeholder_for_missing_crops()
    test_set_view_no_rerun_in_callback()
    test_direct_set_view_calls_have_rerun()
    print("\n✓ All missing crop placeholder tests passed!")
