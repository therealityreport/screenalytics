"""Tests for the run debug bundle export endpoint.

Verifies that GET /episodes/{ep_id}/runs/{run_id}/export returns a PDF report
containing the expected sections.
"""

import json
import os
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


def test_export_bundle_returns_pdf(tmp_path, monkeypatch):
    """Test that the export endpoint returns a valid PDF."""
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
        run_dir / "detections.jsonl",
        [
            {"frame_idx": 0, "bbox": [100, 100, 200, 200], "score": 0.95},
            {"frame_idx": 1, "bbox": [110, 110, 210, 210], "score": 0.92},
        ],
    )
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
            "stats": {"faces": 2, "clusters": 2},
        },
    )
    _write_json(
        run_dir / "track_metrics.json",
        {
            "ep_id": ep_id,
            "metrics": {"tracks_born": 2, "tracks_lost": 2, "id_switches": 0},
            "scene_cuts": {"count": 1},
        },
    )

    client = TestClient(app)

    # Call export endpoint
    resp = client.get(f"/episodes/{ep_id}/runs/{run_id}/export")
    assert resp.status_code == 200, f"Export failed: {resp.text}"
    assert resp.headers.get("content-type") == "application/pdf"

    # Verify PDF magic bytes
    pdf_content = resp.content
    assert pdf_content[:5] == b"%PDF-", "Response is not a valid PDF"

    # Verify content-disposition header
    content_disp = resp.headers.get("content-disposition", "")
    assert "debug_report.pdf" in content_disp, f"Unexpected filename: {content_disp}"


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


def test_export_pdf_with_body_tracking_data(tmp_path, monkeypatch):
    """Test that PDF export works with body tracking artifacts.

    This test ensures the PDF generator handles all artifact types correctly,
    including the body tracking subdirectory.
    """
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
    _write_json(run_dir / "identities.json", {"ep_id": ep_id, "identities": [], "stats": {}})

    # Add body tracking artifacts
    body_dir = run_dir / "body_tracking"
    body_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(body_dir / "body_detections.jsonl", [{"frame_idx": 0, "bbox": [0, 0, 100, 200]}])
    _write_jsonl(body_dir / "body_tracks.jsonl", [{"track_id": 100001, "frame_count": 5}])
    _write_json(
        body_dir / "track_fusion.json",
        {"num_face_tracks": 1, "num_body_tracks": 1, "num_fused_identities": 2, "identities": {}},
    )
    _write_json(
        body_dir / "screentime_comparison.json",
        {
            "summary": {
                "total_identities": 2,
                "identities_with_gain": 1,
                "total_duration_gain": 5.0,
            },
            "breakdowns": [],
        },
    )

    client = TestClient(app)

    # Call export
    resp = client.get(f"/episodes/{ep_id}/runs/{run_id}/export")
    assert resp.status_code == 200, f"Export failed: {resp.text}"

    # Verify it's a valid PDF
    pdf_content = resp.content
    assert pdf_content[:5] == b"%PDF-", "Response is not a valid PDF"

    # Verify PDF has reasonable size (should be at least a few KB with all sections)
    assert len(pdf_content) > 2000, f"PDF too small ({len(pdf_content)} bytes), may be incomplete"

    # Verify content-disposition header
    content_disp = resp.headers.get("content-disposition", "")
    assert "debug_report.pdf" in content_disp


def test_export_returns_s3_upload_headers(tmp_path, monkeypatch):
    """Test that export response includes S3 upload status headers."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")  # Local backend, no actual S3 upload
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    ep_id = "demo-s01e03"
    run_id = "test-run-s3-headers"
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

    # Call export with upload_to_s3=True (but local backend, so no actual upload)
    resp = client.get(
        f"/episodes/{ep_id}/runs/{run_id}/export",
        params={"upload_to_s3": True},
    )
    assert resp.status_code == 200, f"Export failed: {resp.text}"

    # With local backend, S3 upload returns success but nothing uploaded
    s3_success = resp.headers.get("X-S3-Upload-Success")
    assert s3_success == "true", f"Expected S3 upload success header, got: {s3_success}"

    # PDF should still be valid
    assert resp.content[:4] == b"%PDF"


def test_export_pdf_is_valid_with_complete_artifacts(tmp_path, monkeypatch):
    """Test that exported PDF is valid when all expected artifacts exist."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    ep_id = "demo-s01e04"
    run_id = "test-run-full"
    ensure_dirs(ep_id)

    # Set up run directory structure
    manifests_dir = get_path(ep_id, "detections").parent
    run_dir = manifests_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write comprehensive artifacts
    _write_jsonl(run_dir / "detections.jsonl", [{"frame_idx": 0, "bbox": [0, 0, 100, 100]}])
    _write_jsonl(run_dir / "tracks.jsonl", [{"track_id": 1, "frame_count": 10}])
    _write_jsonl(run_dir / "faces.jsonl", [{"track_id": 1, "frame_idx": 0}])
    _write_json(run_dir / "identities.json", {"ep_id": ep_id, "identities": []})
    _write_json(run_dir / "track_metrics.json", {"metrics": {}})
    _write_json(run_dir / "detect_track.json", {"run_id": run_id, "frames_total": 1000})

    # Add body tracking artifacts
    body_dir = run_dir / "body_tracking"
    body_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(body_dir / "body_detections.jsonl", [{"frame_idx": 0}])
    _write_jsonl(body_dir / "body_tracks.jsonl", [{"track_id": 100001}])
    _write_json(body_dir / "track_fusion.json", {"num_face_tracks": 1, "num_body_tracks": 1})
    _write_json(body_dir / "screentime_comparison.json", {"summary": {"total_identities": 1}})

    client = TestClient(app)

    resp = client.get(f"/episodes/{ep_id}/runs/{run_id}/export")
    assert resp.status_code == 200
    assert resp.content[:4] == b"%PDF"
    # PDF with all artifacts should be larger than minimal
    assert len(resp.content) > 5000, f"PDF too small: {len(resp.content)} bytes"
