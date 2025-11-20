from __future__ import annotations

from pathlib import Path

from py_screenalytics.artifacts import ensure_dirs, get_path
from tools.episode_run import _write_jsonl


def test_failed_detect_run_keeps_previous_manifests(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "atomic-failure"
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    detections_path = manifests_dir / "detections.jsonl"
    tracks_path = manifests_dir / "tracks.jsonl"
    detections_path.write_text("SENTINEL_DETECTIONS\n", encoding="utf-8")
    tracks_path.write_text("SENTINEL_TRACKS\n", encoding="utf-8")

    # Simulate a failed rerun that wrote temp files but never replaced live manifests
    detections_tmp = detections_path.with_suffix(".jsonl.tmp")
    tracks_tmp = tracks_path.with_suffix(".jsonl.tmp")
    detections_tmp.write_text("BROKEN\n", encoding="utf-8")
    tracks_tmp.write_text("BROKEN\n", encoding="utf-8")

    assert detections_path.read_text(encoding="utf-8") == "SENTINEL_DETECTIONS\n"
    assert tracks_path.read_text(encoding="utf-8") == "SENTINEL_TRACKS\n"
    # Temp artifacts should not be treated as live manifests
    assert detections_tmp.exists()
    assert tracks_tmp.exists()


def test_detect_manifest_swap_removes_temp_files(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "atomic-success"
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    detections_path = manifests_dir / "detections.jsonl"
    tracks_path = manifests_dir / "tracks.jsonl"
    detections_path.write_text("OLD\n", encoding="utf-8")
    tracks_path.write_text("OLD\n", encoding="utf-8")

    _write_jsonl(detections_path, [{"d": 1}])
    _write_jsonl(tracks_path, [{"t": 1}])

    assert not detections_path.with_suffix(".jsonl.tmp").exists()
    assert not tracks_path.with_suffix(".jsonl.tmp").exists()
    assert '"d": 1' in detections_path.read_text(encoding="utf-8")
    assert '"t": 1' in tracks_path.read_text(encoding="utf-8")
