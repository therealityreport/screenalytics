"""Unit tests for interval math helpers used by screentime."""

from __future__ import annotations

import pytest

from py_screenalytics.intervals import (
    compute_self_overlap,
    compute_union_duration,
    merge_intervals,
)


def test_merge_intervals_non_overlapping() -> None:
    assert merge_intervals([(0.0, 1.0), (2.0, 3.0)]) == [(0.0, 1.0), (2.0, 3.0)]


def test_merge_intervals_overlapping() -> None:
    assert merge_intervals([(0.0, 1.0), (0.5, 2.0)]) == [(0.0, 2.0)]


def test_merge_intervals_nested() -> None:
    assert merge_intervals([(0.0, 3.0), (1.0, 2.0)]) == [(0.0, 3.0)]


def test_merge_intervals_gap_tolerance_bridges_small_gaps() -> None:
    intervals = [(0.0, 1.0), (1.4, 2.0)]
    assert merge_intervals(intervals, gap_tolerance_s=0.0) == intervals
    assert merge_intervals(intervals, gap_tolerance_s=0.5) == [(0.0, 2.0)]


def test_compute_union_duration_overlapping() -> None:
    union = compute_union_duration([(0.0, 1.0), (0.5, 1.5)])
    assert union == pytest.approx(1.5)


def test_compute_self_overlap_overlapping() -> None:
    overlap = compute_self_overlap([(0.0, 1.0), (0.5, 1.5)])
    # sum = 1.0 + 1.0 = 2.0, union = 1.5, overlap = 0.5
    assert overlap == pytest.approx(0.5)


def test_compute_self_overlap_identical_intervals() -> None:
    overlap = compute_self_overlap([(0.0, 1.0), (0.0, 1.0)])
    assert overlap == pytest.approx(1.0)


def test_compute_self_overlap_point_intervals_respects_min_duration() -> None:
    overlap = compute_self_overlap([(1.0, 1.0), (1.0, 1.0)], min_duration_s=0.033)
    assert overlap == pytest.approx(0.033)

