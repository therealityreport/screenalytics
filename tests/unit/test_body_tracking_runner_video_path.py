from __future__ import annotations

from pathlib import Path

from py_screenalytics.artifacts import get_path


def test_body_tracking_runner_finds_screenalytics_video_layout(tmp_path, monkeypatch) -> None:
    """BodyTrackingRunner should discover data/videos/{ep_id}/episode.mp4 via get_path()."""
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path / "data"))

    ep_id = "test-episode-1"
    video_path = get_path(ep_id, "video")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.touch()

    from FEATURES.body_tracking.src.body_tracking_runner import BodyTrackingRunner

    runner = BodyTrackingRunner(
        episode_id=ep_id,
        config_path=None,
        fusion_config_path=None,
        video_path=None,
        output_dir=tmp_path / "out",
        skip_existing=True,
    )

    assert isinstance(runner.video_path, Path)
    assert runner.video_path == video_path

