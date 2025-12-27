from __future__ import annotations

import json
from pathlib import Path

from apps.api.services.run_validator import validate_run_integrity
from py_screenalytics import run_layout


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_validator_catches_mismatches(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))

    ep_id = "demo-s01e01"
    run_id = "Attempt1_2025-01-01_000000EST"
    run_root = run_layout.run_root(ep_id, run_id)

    _write_jsonl(run_root / "tracks.jsonl", [{"track_id": 1}])
    _write_jsonl(run_root / "faces.jsonl", [{"face_id": "f1", "track_id": 999}])
    _write_jsonl(
        run_root / "track_reps.jsonl",
        [{"track_id": 999, "best_crop_rel_path": "crops/track_0999/frame_000001.jpg"}],
    )
    _write_json(
        run_root / "identities.json",
        {
            "identities": [
                {"identity_id": "c1", "track_ids": [2]},
            ],
            "manual_assignments": {"missing_cluster": {"cast_id": "cast_1"}},
            "track_overrides": {"999": {"cast_id": "cast_1"}},
            "face_exclusions": {"missing_face": {"excluded": True}},
        },
    )

    report = validate_run_integrity(ep_id, run_id)
    codes = {entry.get("code") for entry in report.get("errors", [])}
    assert "faces_missing_tracks" in codes
    assert "identities_missing_tracks" in codes
    assert "track_reps_missing_tracks" in codes
    assert "assignment_missing_cluster" in codes
    assert "assignment_missing_track" in codes
    assert "exclusion_missing_face" in codes


def test_validator_allows_missing_faces_manifest_with_track_counts(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))

    ep_id = "demo-s01e02"
    run_id = "Attempt1_2025-01-02_000000EST"
    run_root = run_layout.run_root(ep_id, run_id)

    _write_jsonl(run_root / "tracks.jsonl", [{"track_id": 1, "faces_count": 5}])
    _write_json(run_root / "identities.json", {"identities": [{"identity_id": "c1", "track_ids": [1]}]})

    report = validate_run_integrity(ep_id, run_id)
    error_codes = {entry.get("code") for entry in report.get("errors", [])}
    warning_codes = {entry.get("code") for entry in report.get("warnings", [])}
    assert "missing_faces" not in error_codes
    assert "faces_manifest_missing" in warning_codes


def test_validator_tracks_without_clusters_excludes_clustered_tracks(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))

    ep_id = "demo-s01e03"
    run_id = "Attempt1_2025-01-03_000000EST"
    run_root = run_layout.run_root(ep_id, run_id)

    _write_jsonl(run_root / "tracks.jsonl", [{"track_id": 1}, {"track_id": 2}])
    _write_json(run_root / "identities.json", {"identities": [{"identity_id": "c1", "track_ids": [1]}]})

    report = validate_run_integrity(ep_id, run_id)
    warning = next(
        (entry for entry in report.get("warnings", []) if entry.get("code") == "tracks_without_clusters"),
        None,
    )
    assert warning is not None
    assert warning["details"]["count"] == 1
    summary = report.get("summary", {})
    assert summary.get("unclustered_tracks") == 1
