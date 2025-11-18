from __future__ import annotations

import argparse
import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from py_screenalytics.artifacts import ensure_dirs
from tools import episode_run


class _FakeDetector:
    model_name = "test-detector"

    def __init__(self) -> None:
        self._resolved_device = "cpu"

    @property
    def resolved_device(self) -> str:
        return self._resolved_device

    def ensure_ready(self) -> None:  # pragma: no cover - no-op
        return None

    def detect(self, frame) -> list[episode_run.DetectionSample]:  # noqa: ANN201
        bbox = np.array([4.0, 4.0, 28.0, 28.0], dtype=np.float32)
        return [
            episode_run.DetectionSample(
                bbox=bbox,
                conf=0.99,
                class_idx=0,
                class_label=episode_run.FACE_CLASS_LABEL,
            )
        ]


class _FakeTracker:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.update_calls: list[int] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def update(self, detections, frame_idx: int, frame):  # noqa: ANN201
        self.update_calls.append(frame_idx)
        tracked: list[episode_run.TrackedObject] = []
        for idx, det in enumerate(detections):
            tracked.append(
                episode_run.TrackedObject(
                    track_id=idx + 1,
                    bbox=det.bbox,
                    conf=det.conf,
                    class_idx=det.class_idx,
                    class_label=det.class_label,
                    det_index=idx,
                )
            )
        return tracked


def test_tracker_resets_on_scene_cuts(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "demo-s01e01"
    ensure_dirs(ep_id)

    video_path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (32, 32),
    )
    try:
        for _ in range(10):
            frame = np.full((32, 32, 3), 128, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    monkeypatch.setattr(episode_run, "detect_scene_cuts", lambda *a, **k: [2, 5])
    tracker_holder: dict[str, _FakeTracker] = {}

    def _build_tracker_sim(name: str, frame_rate: float):  # noqa: ANN001
        tracker = _FakeTracker()
        tracker_holder["instance"] = tracker
        return tracker

    monkeypatch.setattr(episode_run, "_build_tracker_adapter", _build_tracker_sim)
    monkeypatch.setattr(
        episode_run, "_build_face_detector", lambda *a, **k: _FakeDetector()
    )

    args = argparse.Namespace(
        ep_id=ep_id,
        stride=1,
        fps=None,
        device="cpu",
        detector="retinaface",
        tracker="bytetrack",
        det_thresh=0.5,
        max_gap=30,
        save_frames=False,
        save_crops=False,
        jpeg_quality=85,
        scene_detector="internal",
        scene_threshold=0.3,
        scene_min_len=1,
        scene_warmup_dets=2,
    )

    result = episode_run._run_full_pipeline(
        args,
        video_path,
        source_fps=10.0,
        progress=None,
        target_fps=None,
        frame_exporter=None,
    )
    (*_, scene_summary) = result

    tracker = tracker_holder.get("instance")
    assert tracker is not None
    assert tracker.reset_calls == 2
    assert scene_summary["count"] == 2
