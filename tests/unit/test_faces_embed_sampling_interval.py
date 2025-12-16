from __future__ import annotations


from tools import episode_run


def _track_samples(count: int) -> list[dict]:
    return [
        {
            "track_id": 1,
            "frame_idx": idx,
            "ts": float(idx) / 24.0,
            "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
        }
        for idx in range(count)
    ]


def test_sample_track_uniformly_respects_sample_interval_and_keeps_last_frame() -> None:
    samples = _track_samples(100)
    out = episode_run._sample_track_uniformly(samples, max_samples=999, min_samples=1, sample_interval=12)
    frame_idxs = [row["frame_idx"] for row in out]
    assert frame_idxs[0] == 0
    assert frame_idxs[-1] == 99
    # Enforced spacing for the main run (last frame is always appended as an anchor).
    assert all((b - a) >= 12 for a, b in zip(frame_idxs, frame_idxs[1:-1]))


def test_sample_track_uniformly_enforces_min_samples_when_interval_is_large() -> None:
    samples = _track_samples(100)
    out = episode_run._sample_track_uniformly(samples, max_samples=999, min_samples=5, sample_interval=80)
    assert len(out) >= 5

