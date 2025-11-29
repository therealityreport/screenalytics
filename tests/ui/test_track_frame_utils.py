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


def test_track_frame_prefers_track_crop_when_other_track_in_frame():
    track_frame_utils = _load_module()
    frames = [
        {
            "frame_idx": 49504,
            "faces": [
                {"track_id": 1497, "face_id": "face_1497", "quality": {"score": 0.55}},
                {"track_id": 1498, "face_id": "face_1498", "quality": {"score": 0.99}},
            ],
        }
    ]

    scoped, missing = track_frame_utils.scope_track_frames(frames, 1497)
    assert not missing
    assert len(scoped) == 1
    frame = scoped[0]
    assert frame.get("frame_idx") == 49504
    assert all(face.get("track_id") == 1497 for face in frame.get("faces", []))
    assert frame["faces"][0]["face_id"] == "face_1497"
    assert track_frame_utils.best_track_frame_idx(scoped, 1497, None) == 49504


def test_two_tracks_one_frame_never_mixes_urls():
    """Regression test for bug where track detail showed wrong person's thumbnail.

    Scenario: Frame 49504 has two people (blonde in track 1497, brunette in track 1498).
    The brunette has higher quality score (0.99 vs 0.55).
    When viewing track 1497 detail, we must ONLY show the blonde face (lower quality),
    never the brunette face from track 1498 (even though it's higher quality).
    """
    track_frame_utils = _load_module()

    # Simulate API response for a frame with two tracks
    # Track 1497: lower quality (0.55) blonde face - correct for this track
    # Track 1498: higher quality (0.99) brunette face - MUST NOT appear for track 1497
    frames = [
        {
            "frame_idx": 49504,
            "track_id": 1497,  # Frame-level track_id (should be ignored)
            "media_url": "crops/track_1497/frame_049504.jpg",
            "faces": [
                {
                    "track_id": 1497,
                    "face_id": "face_1497_049504",
                    "quality": {"score": 0.55},
                    "media_url": "crops/track_1497/frame_049504.jpg",  # Blonde (correct)
                },
                {
                    "track_id": 1498,
                    "face_id": "face_1498_049504",
                    "quality": {"score": 0.99},
                    "media_url": "crops/track_1498/frame_049504.jpg",  # Brunette (WRONG for 1497!)
                },
            ],
            "other_tracks": [1498],
        }
    ]

    # Scope to track 1497
    scoped, missing = track_frame_utils.scope_track_frames(frames, 1497)

    # Must return exactly one frame
    assert len(scoped) == 1, "Should return one scoped frame"
    assert not missing, "Should have no missing faces for track 1497"

    frame = scoped[0]

    # Frame must be for track 1497
    assert frame["frame_idx"] == 49504
    assert frame["track_id"] == 1497

    # Faces list must contain ONLY track 1497 faces
    faces = frame.get("faces", [])
    assert len(faces) == 1, "Should have exactly one face for track 1497"
    assert all(f["track_id"] == 1497 for f in faces), "All faces must belong to track 1497"

    # The best (and only) face for this track is the blonde (lower quality)
    best_face = faces[0]
    assert best_face["face_id"] == "face_1497_049504", "Must use track 1497's face"
    assert best_face["media_url"] == "crops/track_1497/frame_049504.jpg", "Must use track 1497's crop URL"

    # Verify the brunette face (track 1498) is completely absent
    brunette_in_faces = any(f.get("face_id") == "face_1498_049504" for f in faces)
    assert not brunette_in_faces, "Track 1498 face must not appear in track 1497's faces list"

    # Best frame logic should also only consider track 1497 faces
    best_idx = track_frame_utils.best_track_frame_idx(scoped, 1497, None)
    assert best_idx == 49504, "Should identify frame 49504 as best for track 1497"
