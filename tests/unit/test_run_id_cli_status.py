from __future__ import annotations

import json
from pathlib import Path

import tools.episode_run as episode_run
from py_screenalytics import run_layout


def test_cli_missing_run_id_generates_status(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"\x00\x01fakevideo")
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    def _fake_require_valid_video(*_args, **_kwargs) -> dict:
        return {
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "frame_count": 120,
            "duration_sec": 4.0,
        }

    def _fake_probe_video(*_args, **_kwargs) -> tuple[float, int]:
        return 30.0, 120

    def _fake_run_full_pipeline(*_args, **_kwargs):
        return (
            5,  # det_count
            2,  # track_count
            10,  # frames_sampled
            "cpu",  # pipeline_device
            "cpu",  # detector_device
            30.0,  # analyzed_fps
            {},  # track_metrics
            {"count": 0},  # scene_summary
            {},  # detection_conf_hist
            None,  # detect_track_stats
            None,  # tracker_config_summary
        )

    def _fake_sync_artifacts(*_args, **_kwargs) -> episode_run.S3SyncResult:
        return episode_run.S3SyncResult(success=True, stats={}, errors=[], partial=False)

    monkeypatch.setattr(episode_run, "require_valid_video", _fake_require_valid_video)
    monkeypatch.setattr(episode_run, "_probe_video", _fake_probe_video)
    monkeypatch.setattr(episode_run, "_run_full_pipeline", _fake_run_full_pipeline)
    monkeypatch.setattr(episode_run, "_sync_artifacts_to_s3", _fake_sync_artifacts)
    monkeypatch.setattr(episode_run, "_maybe_run_body_tracking", lambda *args, **kwargs: None)
    monkeypatch.setattr(episode_run, "_storage_context", lambda *args, **kwargs: (None, None, None))

    ep_id = "demo-s01e01"
    exit_code = episode_run.main(
        [
            "--ep-id",
            ep_id,
            "--video",
            str(video_path),
            "--device",
            "cpu",
            "--out-root",
            str(data_root),
        ]
    )
    assert exit_code == 0

    run_ids = run_layout.list_run_ids(ep_id)
    assert len(run_ids) == 1
    run_id = run_ids[0]

    status_path = run_layout.run_root(ep_id, run_id) / "episode_status.json"
    assert status_path.exists()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload.get("run_id") == run_id
    stages = payload.get("stages") or {}
    assert stages.get("detect", {}).get("status") == "success"
