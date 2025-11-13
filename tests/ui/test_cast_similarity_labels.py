from __future__ import annotations

from pathlib import Path


def test_cast_page_mentions_similarity_labels() -> None:
    project_root = Path(__file__).resolve().parents[2]
    cast_page = project_root / "apps" / "workspace-ui" / "pages" / "4_Cast.py"
    source = cast_page.read_text(encoding="utf-8")
    assert "Avg Similarity" in source
    assert "avg similarity" in source.lower()
