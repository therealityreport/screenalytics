from __future__ import annotations

from pathlib import Path


def test_facebank_page_compiles() -> None:
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "apps" / "workspace-ui" / "pages" / "3_Faces_Review.py"
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    assert "_render_identity_grid" in source
