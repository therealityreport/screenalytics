from __future__ import annotations

from tests.ui._helpers_loader import load_ui_helpers_module


def test_compose_progress_text_formats_lines():
    helpers = load_ui_helpers_module()
    progress = {
        "phase": "faces_embed",
        "frames_done": 225,
        "frames_total": 450,
        "secs_done": 12.5,
        "secs_total": 25.0,
        "detector": "retinaface",
        "tracker": "bytetrack",
        "device": "mps",
        "fps_infer": 962.07,
    }
    status_line, frames_line = helpers.compose_progress_text(
        progress,
        requested_device="cpu",
        requested_detector="retinaface",
        requested_tracker="bytetrack",
    )
    assert "phase=faces_embed" in status_line
    assert "detector=RetinaFace" in status_line
    assert "tracker=ByteTrack" in status_line
    assert "device=mps" in status_line
    assert "fps=962.07 fps" in status_line
    assert frames_line == "Frames 225 / 450"
