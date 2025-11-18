import ast
from pathlib import Path


def test_fetch_track_media_uses_cursor_and_returns_tuple() -> None:
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "apps" / "workspace-ui" / "pages" / "3_Faces_Review.py"
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(path))
    fetch_func = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_fetch_track_media"
    )
    arg_names = [arg.arg for arg in fetch_func.args.args]
    kwarg_names = [arg.arg for arg in fetch_func.args.kwonlyargs]
    all_args = set(arg_names + kwarg_names)
    assert "cursor" in all_args, "_fetch_track_media must accept a cursor parameter"
    has_tuple_return = any(
        isinstance(node.value, ast.Tuple)
        for node in ast.walk(fetch_func)
        if isinstance(node, ast.Return)
    )
    assert has_tuple_return, "_fetch_track_media should return (items, next_cursor)"
    assert "next_start_after" in source
