from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAGE_FILES = [
    PROJECT_ROOT / "apps" / "workspace-ui" / "streamlit_app.py",
    PROJECT_ROOT / "apps" / "workspace-ui" / "pages" / "1_Episodes.py",
    PROJECT_ROOT / "apps" / "workspace-ui" / "pages" / "2_Episode_Detail.py",
    PROJECT_ROOT / "apps" / "workspace-ui" / "pages" / "3_Faces_Review.py",
    PROJECT_ROOT / "apps" / "workspace-ui" / "pages" / "4_Screentime.py",
    PROJECT_ROOT / "apps" / "workspace-ui" / "pages" / "5_Health.py",
]


@pytest.mark.parametrize("path", PAGE_FILES)
def test_streamlit_page_compiles(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    assert (
        "import ui_helpers" in source
    ), f"{path.name} should import ui_helpers for shared state"
