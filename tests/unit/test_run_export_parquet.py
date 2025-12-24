"""Unit tests for segments.parquet export."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_comparison(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_export_segments_parquet_schema_and_determinism(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    sys.path.insert(0, str(PROJECT_ROOT))
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))

    from py_screenalytics import run_layout
    from py_screenalytics.run_manifests import write_stage_manifest
    from apps.api.services.run_export import export_segments_parquet

    ep_id = "ep-parquet"
    run_id = "run123"
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    comparison_path = run_root / "body_tracking" / "screentime_comparison.json"
    _write_comparison(
        comparison_path,
        {
            "summary": {"total_identities": 2},
            "breakdowns": [
                {
                    "identity_id": "id1",
                    "breakdown": {"body_only_duration": 2.0},
                    "body_only_segments": [
                        {"start_time": 0.0, "end_time": 1.0, "duration": 1.0, "segment_type": "body"},
                        {"start_time": 1.0, "end_time": 2.0, "duration": 1.0, "segment_type": "body"},
                    ],
                },
                {
                    "identity_id": "id2",
                    "breakdown": {"body_only_duration": 1.5},
                    "body_only_segments": [
                        {"start_time": 0.0, "end_time": 1.5, "duration": 1.5, "segment_type": "body"},
                    ],
                },
            ],
        },
    )

    now = datetime.now(timezone.utc)
    write_stage_manifest(
        ep_id,
        run_id,
        "track_fusion",
        "SUCCESS",
        started_at=now,
        finished_at=now,
        duration_s=0.1,
        model_versions={"fusion": "v1"},
        thresholds={"min_iou": 0.5},
    )

    path1 = export_segments_parquet(ep_id, run_id)
    df1 = pandas.read_parquet(path1)

    required = {
        "run_id",
        "episode_id",
        "model_versions",
        "identity",
        "identity_id",
        "track_id",
        "segment_start",
        "segment_end",
        "duration_s",
        "confidence",
        "source",
        "thresholds_snapshot_hash",
    }
    assert required.issubset(set(df1.columns))

    path2 = export_segments_parquet(ep_id, run_id, output_path=run_root / "exports" / "segments_copy.parquet")
    df2 = pandas.read_parquet(path2)
    pandas.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))


def test_run_segments_export_blocks_when_missing_segments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    sys.path.insert(0, str(PROJECT_ROOT))
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))

    from py_screenalytics import run_layout
    from py_screenalytics.episode_status import Stage, StageStatus, read_episode_status
    from py_screenalytics.run_manifests import read_stage_manifest
    from apps.api.services.run_export import run_segments_export

    ep_id = "ep-blocked"
    run_id = "run456"
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    result = run_segments_export(ep_id=ep_id, run_id=run_id)
    assert result is None
    assert not (run_root / "exports" / "segments.parquet").exists()

    status = read_episode_status(ep_id, run_id)
    state = status.stages.get(Stage.SEGMENTS)
    assert state is not None
    assert state.status == StageStatus.BLOCKED

    manifest = read_stage_manifest(ep_id, run_id, "segments")
    assert manifest is not None
    assert str(manifest.get("status", "")).lower() == "blocked"


def test_run_segments_export_hydrates_from_s3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    sys.path.insert(0, str(PROJECT_ROOT))
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))

    from py_screenalytics import run_layout
    from py_screenalytics.episode_status import Stage, StageStatus, read_episode_status
    from apps.api.services import storage as storage_module
    from apps.api.services.run_export import run_segments_export

    payload = {
        "summary": {"total_identities": 1},
        "breakdowns": [
            {
                "identity_id": "id1",
                "breakdown": {"body_only_duration": 1.0},
                "body_only_segments": [
                    {"start_time": 0.0, "end_time": 1.0, "duration": 1.0, "segment_type": "body"},
                ],
            },
        ],
    }
    payload_bytes = json.dumps(payload).encode("utf-8")

    class FakeStorage:
        def s3_enabled(self) -> bool:
            return True

        def download_bytes(self, key: str) -> bytes | None:
            if "screentime_comparison.json" in key:
                return payload_bytes
            return None

    monkeypatch.setattr(storage_module, "StorageService", FakeStorage)

    ep_id = "ep-hydrate"
    run_id = "run-hydrate"
    run_layout.run_root(ep_id, run_id).mkdir(parents=True, exist_ok=True)

    output_path = run_segments_export(ep_id=ep_id, run_id=run_id)
    assert output_path is not None
    assert output_path.exists()

    status = read_episode_status(ep_id, run_id)
    state = status.stages.get(Stage.SEGMENTS)
    assert state is not None
    assert state.status == StageStatus.SUCCESS
    assert pandas.read_parquet(output_path).shape[0] > 0


def test_run_segments_export_reconciliation_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    sys.path.insert(0, str(PROJECT_ROOT))
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))

    from py_screenalytics import run_layout
    from py_screenalytics.episode_status import Stage, StageStatus, read_episode_status
    from py_screenalytics.run_manifests import read_stage_manifest
    from apps.api.services.run_export import run_segments_export

    ep_id = "ep-mismatch"
    run_id = "run789"
    run_root = run_layout.run_root(ep_id, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    comparison_path = run_root / "body_tracking" / "screentime_comparison.json"
    _write_comparison(
        comparison_path,
        {
            "summary": {"total_identities": 1},
            "breakdowns": [
                {
                    "identity_id": "id1",
                    "breakdown": {"body_only_duration": 4.0},
                    "body_only_segments": [
                        {"start_time": 0.0, "end_time": 1.0, "duration": 1.0, "segment_type": "body"},
                    ],
                },
            ],
        },
    )

    with pytest.raises(ValueError):
        run_segments_export(ep_id=ep_id, run_id=run_id)

    status = read_episode_status(ep_id, run_id)
    state = status.stages.get(Stage.SEGMENTS)
    assert state is not None
    assert state.status == StageStatus.FAILED

    manifest = read_stage_manifest(ep_id, run_id, "segments")
    assert manifest is not None
    assert str(manifest.get("status", "")).lower() == "failed"
    assert not (run_root / "exports" / "segments.parquet").exists()
