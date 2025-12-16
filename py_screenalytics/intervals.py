"""Interval math utilities.

These helpers are intentionally dependency-free so they can be reused by
pipeline code, services, and reports.
"""

from __future__ import annotations

from typing import Iterable, TypeAlias

Interval: TypeAlias = tuple[float, float]


def merge_intervals(intervals: Iterable[Interval], *, gap_tolerance_s: float = 0.0) -> list[Interval]:
    """Merge overlapping time intervals.

    Args:
        intervals: Iterable of (start_s, end_s) tuples.
        gap_tolerance_s: When > 0, also merges intervals that are separated by
            a gap <= gap_tolerance_s (useful for "gap bridging" policies).

    Returns:
        A list of merged intervals sorted by start time.
    """
    gap = float(gap_tolerance_s or 0.0)
    if gap < 0:
        gap = 0.0

    normalized: list[Interval] = []
    for start, end in intervals:
        try:
            s = float(start)
            e = float(end)
        except (TypeError, ValueError):
            continue
        if e < s:
            s, e = e, s
        normalized.append((s, e))

    if not normalized:
        return []

    normalized.sort(key=lambda x: x[0])
    merged: list[list[float]] = [[normalized[0][0], normalized[0][1]]]
    for start, end in normalized[1:]:
        prev = merged[-1]
        if start - prev[1] <= gap:
            prev[1] = max(prev[1], end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def interval_duration(interval: Interval, *, min_duration_s: float = 0.0) -> float:
    """Return the duration of an interval in seconds.

    Args:
        interval: (start_s, end_s) timestamps in seconds.
        min_duration_s: When > 0, any interval shorter than this is treated as
            having duration min_duration_s (useful for single-sample "point"
            intervals).
    """
    try:
        start, end = interval
        s = float(start)
        e = float(end)
    except (TypeError, ValueError):
        return 0.0

    if e < s:
        s, e = e, s

    duration = max(0.0, e - s)
    minimum = float(min_duration_s or 0.0)
    if minimum > 0.0 and duration < minimum:
        return minimum
    return duration


def sum_interval_durations(intervals: Iterable[Interval], *, min_duration_s: float = 0.0) -> float:
    """Sum durations of intervals without merging overlaps."""
    return sum(interval_duration(interval, min_duration_s=min_duration_s) for interval in intervals)


def compute_union_duration(
    intervals: Iterable[Interval],
    *,
    gap_tolerance_s: float = 0.0,
    min_duration_s: float = 0.0,
) -> float:
    """Compute union duration of intervals in seconds.

    Note: gap_tolerance_s controls whether small gaps are bridged prior to
    measuring union duration.
    """
    merged = merge_intervals(intervals, gap_tolerance_s=gap_tolerance_s)
    return sum_interval_durations(merged, min_duration_s=min_duration_s)


def compute_self_overlap(intervals: Iterable[Interval], *, min_duration_s: float = 0.0) -> float:
    """Compute self-overlap (double-counted time) for a set of intervals.

    self_overlap = sum(interval_durations) - union(interval_durations)

    This is useful for detecting duplicate/overlapping tracks assigned to the
    same entity: if multiple track intervals overlap in time, naive summation
    inflates totals and self_overlap becomes > 0.
    """
    summed = sum_interval_durations(intervals, min_duration_s=min_duration_s)
    union = compute_union_duration(intervals, gap_tolerance_s=0.0, min_duration_s=min_duration_s)
    return max(0.0, summed - union)


__all__ = [
    "Interval",
    "merge_intervals",
    "interval_duration",
    "sum_interval_durations",
    "compute_union_duration",
    "compute_self_overlap",
]

