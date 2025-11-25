import importlib.util
from pathlib import Path


def _load_module():
    module_path = Path("apps/workspace-ui/track_frame_utils.py").resolve()
    spec = importlib.util.spec_from_file_location("track_frame_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_scope_track_frames_filters_other_tracks():
    track_frame_utils = _load_module()
    frames = [
        {
            "frame_idx": 49504,
            "track_id": 889,
            "media_url": "correct.jpg",
            "faces": [
                {"track_id": 889, "face_id": "face_889", "quality": {"score": 0.4}, "media_url": "correct.jpg"},
                {"track_id": 890, "face_id": "face_890", "quality": {"score": 0.99}, "media_url": "wrong.jpg"},
            ],
        }
    ]

    scoped, missing = track_frame_utils.scope_track_frames(frames, 889)
    assert not missing
    assert len(scoped) == 1
    frame = scoped[0]
    assert frame["track_id"] == 889
    assert frame["faces"]
    assert all(face["track_id"] == 889 for face in frame["faces"])
    # Quality should prefer the track-local best (score 0.4 here) and ignore higher score on other track
    assert frame["faces"][0]["face_id"] == "face_889"


def test_best_track_frame_idx_uses_track_only_faces():
    track_frame_utils = _load_module()
    frames = [
        {
            "frame_idx": 1,
            "faces": [
                {"track_id": 889, "quality": {"score": 0.5}},
                {"track_id": 890, "quality": {"score": 0.99}},
            ],
        },
        {
            "frame_idx": 2,
            "faces": [{"track_id": 889, "quality": {"score": 0.9}}],
        },
    ]
    best_idx = track_frame_utils.best_track_frame_idx(frames, 889, None)
    assert best_idx == 2
