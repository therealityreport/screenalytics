import ast
from pathlib import Path
from typing import Any, Dict, List


def _load_track_row_func():
    project_root = Path(__file__).resolve().parents[2]
    helpers_path = project_root / "apps" / "workspace-ui" / "ui_helpers.py"
    source = helpers_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(helpers_path))
    func_node = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "track_row_html"
        ),
        None,
    )
    assert func_node is not None, "track_row_html not found"
    func_source = ast.get_source_segment(source, func_node)
    namespace = {"html": __import__("html"), "List": List, "Dict": Dict, "Any": Any}
    exec(func_source, namespace)
    return namespace["track_row_html"]


def test_track_row_html_generates_markup() -> None:
    track_row_html = _load_track_row_func()
    items = [
        {"url": "https://example.com/frame_000001.jpg", "frame_idx": 1},
        {"url": "https://example.com/frame_000002.jpg", "frame_idx": 2},
    ]
    html_block = track_row_html(7, items, thumb_height=150)
    assert "rail-7" in html_block
    assert "thumb" in html_block
