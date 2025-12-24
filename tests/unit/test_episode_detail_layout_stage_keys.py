from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from py_screenalytics import run_layout

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


def test_canonical_status_from_entry_derived_success() -> None:
    layout = _load_layout_module()
    entry = {
        "status": "not_started",
        "derived": True,
        "derived_from": ["manifests/detect.json"],
    }
    assert layout.canonical_status_from_entry(entry) == "success"


def test_resolve_run_debug_pdf(tmp_path, monkeypatch) -> None:
    layout = _load_layout_module()
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))

    ep_id = "ep-debug"
    run_id = "Attempt1_2025-12-23_000000EST"
    run_root = run_layout.run_root(ep_id, run_id)
    pdf_path = run_root / "exports" / "run_debug.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4 test")

    info = layout.resolve_run_debug_pdf(ep_id, run_id, stage_entry=None)
    assert info.exists
    assert info.local_path == pdf_path

    missing_info = layout.resolve_run_debug_pdf(ep_id, "Attempt2_2025-12-23_000000EST", stage_entry=None)
    assert not missing_info.exists

    export_index = run_root / "exports" / "export_index.json"
    export_index.write_text(
        '{"export_s3_key": "runs/demo/s01/e01/Attempt1_2025-12-23_000000EST/exports/debug_report.pdf"}',
        encoding="utf-8",
    )
    (run_root / "exports" / "run_debug.pdf").unlink()
    remote_info = layout.resolve_run_debug_pdf(ep_id, run_id, stage_entry=None)
    assert not remote_info.exists
    assert remote_info.s3_key
