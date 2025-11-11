import os

import pytest

from tools.episode_run import TrackRecorder


@pytest.mark.skipif(os.getenv("RUN_ML_TESTS") != "1", reason="Set RUN_ML_TESTS=1 to enable slow tracking tests")
def test_track_recorder_splits_on_gap() -> None:
    recorder = TrackRecorder(max_gap=5, remap_ids=True)
    ts = 0.0
    # Two tracker IDs alternating frames, no gap > max_gap
    for frame_idx in range(0, 30, 2):
        recorder.record(tracker_track_id=1, frame_idx=frame_idx, ts=ts, bbox=[0, 0, 10, 10], class_label="face")
        recorder.record(tracker_track_id=2, frame_idx=frame_idx, ts=ts, bbox=[20, 20, 30, 30], class_label="face")
        ts += 0.1

    # Large gap for tracker 1 should spawn a new export id and count as a lost track + id switch
    recorder.record(tracker_track_id=1, frame_idx=50, ts=ts, bbox=[0, 0, 10, 10], class_label="face")
    recorder.finalize()
    metrics = recorder.metrics
    assert metrics["tracks_born"] == 3
    assert metrics["tracks_lost"] == 3
    assert metrics["id_switches"] == 1
    longest = recorder.top_long_tracks()
    assert longest[0]["frame_count"] >= 15
