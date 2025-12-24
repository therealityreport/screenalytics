from __future__ import annotations

from apps.api.services.run_artifact_store import delete_run
from py_screenalytics import run_layout
from py_screenalytics.artifacts import get_path
from py_screenalytics.episode_status import write_stage_started


def test_delete_run_local_wipes_tree(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "demo-s01e01"
    run_id = "Attempt1_2025-12-23_000000EST"

    run_root = run_layout.run_root(ep_id, run_id)
    (run_root / "exports").mkdir(parents=True, exist_ok=True)
    (run_root / "exports" / "run_debug.pdf").write_bytes(b"%PDF-1.4 test")
    (run_root / "episode_status.json").write_text("{}", encoding="utf-8")
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    (run_root / "logs" / "detect.jsonl").write_text("{}", encoding="utf-8")

    frames_run_root = get_path(ep_id, "frames_root") / "runs" / run_id
    (frames_run_root / "crops" / "track_0001").mkdir(parents=True, exist_ok=True)
    (frames_run_root / "crops" / "track_0001" / "frame_000001.jpg").write_bytes(b"x")

    result = delete_run(ep_id, run_id, delete_remote=False)

    assert not run_root.exists()
    assert not frames_run_root.exists()
    assert str(run_root) in result.deleted_local_paths
    assert str(frames_run_root) in result.deleted_local_paths


def test_delete_run_skips_episode_scoped_frames(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "demo-s01e01"
    run_id = "Attempt2_2025-12-23_000000EST"

    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    frames_root = get_path(ep_id, "frames_root")
    legacy_frames_dir = frames_root / "frames"
    legacy_frames_dir.mkdir(parents=True, exist_ok=True)
    (legacy_frames_dir / "frame_000001.jpg").write_bytes(b"x")

    result = delete_run(ep_id, run_id, delete_remote=False)

    assert legacy_frames_dir.exists()
    assert any(
        entry.get("path") == str(legacy_frames_dir)
        and entry.get("reason") == "episode_scoped_shared"
        for entry in result.skipped
    )


def test_delete_run_refuses_active_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    ep_id = "demo-s01e01"
    run_id = "Attempt3_2025-12-23_000000EST"

    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    write_stage_started(ep_id, run_id, "detect")

    try:
        delete_run(ep_id, run_id, delete_remote=False)
    except RuntimeError as exc:
        assert "Refusing to delete active run" in str(exc)
    else:
        raise AssertionError("Expected active run deletion to be refused")
