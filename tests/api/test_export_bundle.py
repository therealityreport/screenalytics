"""Tests for the run debug bundle export endpoint.

Verifies that GET /episodes/{ep_id}/runs/{run_id}/export returns a ZIP archive
containing the expected files.
"""

import io
import json
import os
import zipfile
from pathlib import Path

os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ.setdefault("SCREENALYTICS_FAKE_DB", "1")

from fastapi.testclient import TestClient

from apps.api.main import app
from py_screenalytics.artifacts import ensure_dirs, get_path


def _write_json(path: Path, payload) -> None:
    """Write JSON payload to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    """Write JSONL rows to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_export_bundle_contains_expected_files(tmp_path, monkeypatch):
    """Test that the export bundle contains all expected JSON files."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    ep_id = "demo-s01e01"
    run_id = "test-run-001"
    ensure_dirs(ep_id)

    # Set up run directory structure
    manifests_dir = get_path(ep_id, "detections").parent
    run_dir = manifests_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write minimal artifacts
    _write_jsonl(
        run_dir / "tracks.jsonl",
        [
            {"track_id": 1, "class": "face", "frame_count": 10},
            {"track_id": 2, "class": "face", "frame_count": 5},
        ],
    )
    _write_jsonl(
        run_dir / "faces.jsonl",
        [
            {"track_id": 1, "frame_idx": 0, "ts": 0.0},
            {"track_id": 2, "frame_idx": 1, "ts": 0.5},
        ],
    )
    _write_json(
        run_dir / "identities.json",
        {
            "ep_id": ep_id,
            "identities": [
                {"identity_id": "id_0001", "label": "Person One", "track_ids": [1]},
                {"identity_id": "id_0002", "label": "Person Two", "track_ids": [2]},
            ],
            "stats": {},
        },
    )
    _write_json(run_dir / "track_metrics.json", {"ep_id": ep_id, "metrics": {}})

    client = TestClient(app)

    # Call export endpoint
    resp = client.get(
        f"/episodes/{ep_id}/runs/{run_id}/export",
        params={"include_artifacts": True, "include_images": False, "include_logs": False},
    )
    assert resp.status_code == 200, f"Export failed: {resp.text}"
    assert resp.headers.get("content-type") == "application/zip"

    # Verify ZIP contents
    zip_data = io.BytesIO(resp.content)
    with zipfile.ZipFile(zip_data, "r") as zf:
        file_list = zf.namelist()

        # Required metadata files (always present)
        assert "run_summary.json" in file_list, "Missing run_summary.json"
        assert "jobs.json" in file_list, "Missing jobs.json"
        assert "identity_assignments.json" in file_list, "Missing identity_assignments.json"
        assert "identity_locks.json" in file_list, "Missing identity_locks.json"
        assert "smart_suggestion_batches.json" in file_list, "Missing smart_suggestion_batches.json"
        assert "smart_suggestions.json" in file_list, "Missing smart_suggestions.json"
        assert "smart_suggestions_applied.json" in file_list, "Missing smart_suggestions_applied.json"

        # Verify run_summary.json structure
        run_summary = json.loads(zf.read("run_summary.json"))
        assert run_summary["ep_id"] == ep_id
        assert run_summary["run_id"] == run_id
        assert "schema_version" in run_summary

        # Verify identity_locks.json structure
        locks = json.loads(zf.read("identity_locks.json"))
        assert locks["ep_id"] == ep_id
        assert locks["run_id"] == run_id
        assert "locks" in locks

        # Verify smart_suggestion_batches.json structure
        batches = json.loads(zf.read("smart_suggestion_batches.json"))
        assert batches["ep_id"] == ep_id
        assert batches["run_id"] == run_id
        assert "batches" in batches

        # Verify smart_suggestions.json structure
        suggestions = json.loads(zf.read("smart_suggestions.json"))
        assert suggestions["ep_id"] == ep_id
        assert suggestions["run_id"] == run_id
        assert "suggestions" in suggestions

        # Verify smart_suggestions_applied.json structure
        applied = json.loads(zf.read("smart_suggestions_applied.json"))
        assert applied["ep_id"] == ep_id
        assert applied["run_id"] == run_id
        assert "applies" in applied

        # Artifacts should be present (include_artifacts=True)
        assert "tracks.jsonl" in file_list, "Missing tracks.jsonl artifact"
        assert "faces.jsonl" in file_list, "Missing faces.jsonl artifact"
        assert "identities.json" in file_list, "Missing identities.json artifact"


def test_export_bundle_404_for_missing_run(tmp_path, monkeypatch):
    """Test that export returns 404 for non-existent run."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    ep_id = "demo-s01e01"
    ensure_dirs(ep_id)

    client = TestClient(app)

    # Call export for non-existent run
    resp = client.get(f"/episodes/{ep_id}/runs/nonexistent-run/export")
    assert resp.status_code == 404, f"Expected 404 but got {resp.status_code}"


def test_export_bundle_excludes_artifacts_when_disabled(tmp_path, monkeypatch):
    """Test that include_artifacts=False excludes artifact files."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    ep_id = "demo-s01e02"
    run_id = "test-run-002"
    ensure_dirs(ep_id)

    # Set up run directory structure
    manifests_dir = get_path(ep_id, "detections").parent
    run_dir = manifests_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write minimal artifacts
    _write_jsonl(run_dir / "tracks.jsonl", [{"track_id": 1}])
    _write_jsonl(run_dir / "faces.jsonl", [{"track_id": 1, "frame_idx": 0}])
    _write_json(run_dir / "identities.json", {"ep_id": ep_id, "identities": []})

    client = TestClient(app)

    # Call export with include_artifacts=False
    resp = client.get(
        f"/episodes/{ep_id}/runs/{run_id}/export",
        params={"include_artifacts": False, "include_images": False, "include_logs": False},
    )
    assert resp.status_code == 200

    # Verify ZIP contents
    zip_data = io.BytesIO(resp.content)
    with zipfile.ZipFile(zip_data, "r") as zf:
        file_list = zf.namelist()

        # Metadata files should still be present
        assert "run_summary.json" in file_list
        assert "jobs.json" in file_list
        assert "identity_locks.json" in file_list

        # Artifacts should NOT be present
        assert "tracks.jsonl" not in file_list, "tracks.jsonl should be excluded"
        assert "faces.jsonl" not in file_list, "faces.jsonl should be excluded"
