from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_layout_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "apps" / "workspace-ui" / "episode_detail_layout.py"
    spec = importlib.util.spec_from_file_location("episode_detail_layout", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load episode_detail_layout module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def test_resolve_stage_key_prefers_aliases() -> None:
    layout = _load_layout_module()
    stages = {"detect_track": {"status": "success"}}
    resolved = layout.resolve_stage_key("detect", stages.keys())
    assert resolved == "detect_track"
