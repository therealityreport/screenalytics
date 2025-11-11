from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = PROJECT_ROOT / "apps" / "workspace-ui" / "ui_helpers.py"


def test_ui_helpers_compiles() -> None:
    source = HELPER_PATH.read_text(encoding="utf-8")
    compile(source, str(HELPER_PATH), "exec")
