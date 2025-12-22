from __future__ import annotations

from datetime import datetime, timezone

import tools.episode_run as episode_run
from tests.ui._helpers_loader import load_ui_helpers_module


def test_stage_progress_finalizing_updates_and_stall_message(monkeypatch) -> None:
    updates: list[dict] = []

    def _capture_update(_ep_id, _run_id, *, stage_key=None, stage_update=None, **_kwargs):
        updates.append(stage_update or {})
        return {}

    monkeypatch.setattr(episode_run, "update_episode_status", _capture_update)
    times = iter(["2024-01-01T00:00:00Z", "2024-01-01T00:00:05Z"])
    monkeypatch.setattr(episode_run, "_utcnow_iso", lambda: next(times))
    monkeypatch.setattr(episode_run.time, "time", lambda: 0.0)

    heartbeat = episode_run.StageStatusHeartbeat(
        ep_id="ep-demo",
        run_id="run-1",
        stage_key="faces",
        frames_total=10,
        started_at="2024-01-01T00:00:00Z",
        heartbeat_interval=0.0,
    )

    heartbeat.update(
        done=10,
        phase="finalizing",
        message="Finalizing",
        mark_frames_done=True,
        mark_finalize_start=True,
        force=True,
    )

    first = updates[-1]
    assert first["progress"]["phase"] == "finalizing"
    assert first["timestamps"].get("frames_done_at")
    assert first["timestamps"].get("ended_at") is None

    heartbeat.update(
        done=10,
        phase="finalizing",
        message="Finalizing",
        force=True,
    )
    second = updates[-1]
    assert second["progress"]["last_update_at"] != first["progress"]["last_update_at"]

    helpers = load_ui_helpers_module()
    stall_msg = helpers.stage_progress_stall_message(
        second["progress"],
        now=datetime(2024, 1, 1, 0, 0, 40, tzinfo=timezone.utc),
        threshold_seconds=30,
    )
    assert stall_msg and "Stalled" in stall_msg
