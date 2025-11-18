from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("scenedetect")

from tools import episode_run


def _write_color_bars(
    path: Path, colors: list[tuple[int, int, int]], frames_per_color: int
) -> None:
    width, height = 64, 64
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (width, height),
    )
    try:
        for color in colors:
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            for _ in range(frames_per_color):
                writer.write(frame)
    finally:
        writer.release()


def test_detect_scene_cuts_pyscenedetect(tmp_path) -> None:
    video_path = tmp_path / "scene_cuts.mp4"
    frames_per_color = 12
    _write_color_bars(
        video_path,
        colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        frames_per_color=frames_per_color,
    )

    cuts = episode_run.detect_scene_cuts_pyscenedetect(
        str(video_path),
        threshold=15.0,
        min_len=5,
    )

    assert len(cuts) == 2
    assert any(abs(cut - frames_per_color) <= 1 for cut in cuts)
    assert any(abs(cut - frames_per_color * 2) <= 1 for cut in cuts)
