from __future__ import annotations

from py_screenalytics.face_alignment.fan2d import compute_alignment_quality


def test_compute_alignment_quality_is_deterministic_and_bounded() -> None:
    bbox = [0, 0, 100, 120]
    landmarks = [[float(i), float(i % 20)] for i in range(68)]

    score1 = compute_alignment_quality(bbox, landmarks, min_face_size=20)
    score2 = compute_alignment_quality(bbox, landmarks, min_face_size=20)

    assert 0.0 <= score1 <= 1.0
    assert score1 == score2


def test_compute_alignment_quality_prefers_larger_faces() -> None:
    landmarks = [[float(i), float(i % 20)] for i in range(68)]

    small = compute_alignment_quality([0, 0, 12, 12], landmarks, min_face_size=20)
    large = compute_alignment_quality([0, 0, 120, 120], landmarks, min_face_size=20)

    assert 0.0 <= small <= 1.0
    assert 0.0 <= large <= 1.0
    assert large > small

