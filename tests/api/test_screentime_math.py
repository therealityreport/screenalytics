"""Integration tests for screentime math outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.api.services.screentime import ScreenTimeAnalyzer, ScreenTimeConfig


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_screentime_math_reports_union_vs_sum_and_multi_cast_overlap(tmp_path: Path, monkeypatch) -> None:
    """Union coverage is bounded, while cast sum time can exceed due to overlap."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "test-s01e10"
    show_id = "TEST"

    manifests_dir = data_root / "manifests" / ep_id
    manifests_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        manifests_dir / "faces.jsonl",
        [
            {"track_id": 1, "ts": 0.0, "quality": 0.99},
            {"track_id": 2, "ts": 0.0, "quality": 0.99},
        ],
    )
    _write_jsonl(
        manifests_dir / "tracks.jsonl",
        [
            {"track_id": 1, "first_ts": 0.0, "last_ts": 10.0},
            {"track_id": 2, "first_ts": 5.0, "last_ts": 15.0},
        ],
    )
    (manifests_dir / "identities.json").write_text(
        json.dumps(
            {
                "identities": [
                    {"identity_id": "id_a", "track_ids": [1], "person_id": "person_a"},
                    {"identity_id": "id_b", "track_ids": [2], "person_id": "person_b"},
                ]
            }
        ),
        encoding="utf-8",
    )

    shows_dir = data_root / "shows" / show_id
    shows_dir.mkdir(parents=True, exist_ok=True)
    (shows_dir / "people.json").write_text(
        json.dumps(
            {
                "people": [
                    {"person_id": "person_a", "name": "A", "cast_id": "cast_a"},
                    {"person_id": "person_b", "name": "B", "cast_id": "cast_b"},
                ]
            }
        ),
        encoding="utf-8",
    )

    analyzer = ScreenTimeAnalyzer(
        ScreenTimeConfig(
            screen_time_mode="tracks",
            use_video_decode=True,
            gap_tolerance_s=0.0,
            edge_padding_s=0.0,
            quality_min=0.0,
        )
    )
    result = analyzer.analyze_episode(ep_id)

    math_block = result.get("math") or {}
    assert math_block.get("cast_union_coverage_s") == pytest.approx(15.0)
    assert math_block.get("cast_sum_time_s") == pytest.approx(20.0)
    assert math_block.get("multi_cast_overlap_s") == pytest.approx(5.0)


def test_screentime_math_reports_self_overlap_for_duplicate_tracks(tmp_path: Path, monkeypatch) -> None:
    """Self-overlap diagnostics reveal within-cast duplicate track overlap."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "test-s01e11"
    show_id = "TEST"

    manifests_dir = data_root / "manifests" / ep_id
    manifests_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        manifests_dir / "faces.jsonl",
        [
            {"track_id": 1, "ts": 0.0, "quality": 0.99},
            {"track_id": 2, "ts": 0.0, "quality": 0.99},
        ],
    )
    _write_jsonl(
        manifests_dir / "tracks.jsonl",
        [
            {"track_id": 1, "first_ts": 0.0, "last_ts": 10.0},
            {"track_id": 2, "first_ts": 5.0, "last_ts": 15.0},
        ],
    )
    (manifests_dir / "identities.json").write_text(
        json.dumps(
            {
                "identities": [
                    {"identity_id": "id_a", "track_ids": [1, 2], "person_id": "person_a"},
                ]
            }
        ),
        encoding="utf-8",
    )

    shows_dir = data_root / "shows" / show_id
    shows_dir.mkdir(parents=True, exist_ok=True)
    (shows_dir / "people.json").write_text(
        json.dumps(
            {
                "people": [
                    {"person_id": "person_a", "name": "A", "cast_id": "cast_a"},
                ]
            }
        ),
        encoding="utf-8",
    )

    analyzer = ScreenTimeAnalyzer(
        ScreenTimeConfig(
            screen_time_mode="tracks",
            use_video_decode=True,
            gap_tolerance_s=0.0,
            edge_padding_s=0.0,
            quality_min=0.0,
        )
    )
    result = analyzer.analyze_episode(ep_id)

    # Per-cast visual time uses union-of-intervals, so duplicates do not inflate.
    metrics = result.get("metrics") or []
    assert len(metrics) == 1
    assert metrics[0]["cast_id"] == "cast_a"
    assert metrics[0]["visual_s"] == pytest.approx(15.0)

    math_block = result.get("math") or {}
    top = math_block.get("self_overlap_top_casts") or []
    assert top, "expected self-overlap diagnostics"
    assert top[0]["cast_id"] == "cast_a"
    assert top[0]["self_overlap_s"] == pytest.approx(5.0)

