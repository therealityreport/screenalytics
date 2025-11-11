from __future__ import annotations

from pathlib import Path


def test_facebank_page_contains_name_controls() -> None:
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "apps" / "workspace-ui" / "pages" / "3_Faces_Review.py"
    source = path.read_text(encoding="utf-8")
    assert "_identity_name_controls" in source
    assert "_name_choice_widget" in source
