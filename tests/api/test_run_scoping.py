"""Tests for run-scoped isolation.

Verifies that:
1. Assignments in run A don't appear in run B
2. Dismissed suggestions in run A don't hide run B suggestions
3. "Apply All" requires batch_id
4. "Apply All" skips locked identities
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


def _setup_run(data_root: Path, ep_id: str, run_id: str) -> Path:
    """Set up a run with minimal artifacts."""
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    run_dir = manifests_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        run_dir / "tracks.jsonl",
        [
            {"track_id": 1, "class": "face", "frame_count": 10},
            {"track_id": 2, "class": "face", "frame_count": 5},
            {"track_id": 3, "class": "face", "frame_count": 8},
        ],
    )
    _write_jsonl(
        run_dir / "faces.jsonl",
        [
            {"track_id": 1, "frame_idx": 0, "ts": 0.0},
            {"track_id": 2, "frame_idx": 1, "ts": 0.5},
            {"track_id": 3, "frame_idx": 2, "ts": 1.0},
        ],
    )
    _write_json(
        run_dir / "identities.json",
        {
            "ep_id": ep_id,
            "identities": [
                {"identity_id": "id_0001", "label": "Cluster 1", "track_ids": [1]},
                {"identity_id": "id_0002", "label": "Cluster 2", "track_ids": [2]},
                {"identity_id": "id_0003", "label": "Cluster 3", "track_ids": [3]},
            ],
            "stats": {},
        },
    )
    _write_json(
        run_dir / "cluster_centroids.json",
        {
            "ep_id": ep_id,
            "centroids": {
                "id_0001": [0.1] * 512,
                "id_0002": [0.2] * 512,
                "id_0003": [0.3] * 512,
            },
        },
    )
    return run_dir


def test_apply_all_requires_batch_id(tmp_path, monkeypatch):
    """Test that apply_all returns error without batch_id."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    ep_id = "demo-s01e01"
    run_id = "run-scope-test-1"
    _setup_run(data_root, ep_id, run_id)

    client = TestClient(app)

    # Try apply_all without batch_id in body (should fail validation)
    resp = client.post(
        f"/episodes/{ep_id}/smart_suggestions/apply_all",
        params={"run_id": run_id},
        json={},  # Missing batch_id
    )
    assert resp.status_code == 422, f"Expected 422 validation error, got {resp.status_code}: {resp.text}"


def test_apply_all_requires_run_id(tmp_path, monkeypatch):
    """Test that apply_all requires run_id query param."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    ep_id = "demo-s01e01"
    run_id = "run-scope-test-2"
    _setup_run(data_root, ep_id, run_id)

    client = TestClient(app)

    # Try apply_all without run_id param (should fail validation)
    resp = client.post(
        f"/episodes/{ep_id}/smart_suggestions/apply_all",
        # No run_id param
        json={"batch_id": "some-batch-id"},
    )
    assert resp.status_code == 422, f"Expected 422 validation error, got {resp.status_code}: {resp.text}"


def test_identity_locks_are_run_scoped(tmp_path, monkeypatch):
    """Test that identity locks are scoped per run."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    ep_id = "demo-s01e01"
    run_a = "run-scope-a"
    run_b = "run-scope-b"
    _setup_run(data_root, ep_id, run_a)
    _setup_run(data_root, ep_id, run_b)

    client = TestClient(app)

    # Lock an identity in run A
    resp = client.post(
        f"/episodes/{ep_id}/identities/id_0001/lock",
        params={"run_id": run_a},
        json={"reason": "Testing run isolation"},
    )
    assert resp.status_code == 200, f"Lock failed: {resp.text}"

    # Verify lock exists in run A
    resp = client.get(f"/episodes/{ep_id}/identity_locks", params={"run_id": run_a})
    assert resp.status_code == 200
    locks_a = resp.json().get("locks", [])
    locked_ids_a = [l.get("identity_id") for l in locks_a if l.get("locked")]
    assert "id_0001" in locked_ids_a, "Identity should be locked in run A"

    # Verify lock does NOT exist in run B (run isolation)
    resp = client.get(f"/episodes/{ep_id}/identity_locks", params={"run_id": run_b})
    assert resp.status_code == 200
    locks_b = resp.json().get("locks", [])
    locked_ids_b = [l.get("identity_id") for l in locks_b if l.get("locked")]
    assert "id_0001" not in locked_ids_b, "Identity should NOT be locked in run B"


def test_suggestion_batches_are_run_scoped(tmp_path, monkeypatch):
    """Test that suggestion batches are scoped per run."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    ep_id = "demo-s01e01"
    run_a = "run-batch-a"
    run_b = "run-batch-b"
    _setup_run(data_root, ep_id, run_a)
    _setup_run(data_root, ep_id, run_b)

    # Import run_persistence_service to create batches directly
    from apps.api.services.run_persistence import run_persistence_service

    # Create a suggestion batch in run A
    batch_a_id = run_persistence_service.create_suggestion_batch(
        ep_id=ep_id,
        run_id=run_a,
        generator_version="test_v1",
        generator_config_json={},
    )

    client = TestClient(app)

    # List batches in run A - should see the batch
    resp = client.get(f"/episodes/{ep_id}/smart_suggestions/batches", params={"run_id": run_a})
    assert resp.status_code == 200
    batches_a = resp.json().get("batches", [])
    batch_ids_a = [b.get("batch_id") for b in batches_a]
    assert batch_a_id in batch_ids_a, f"Batch {batch_a_id} should be in run A"

    # List batches in run B - should NOT see the batch
    resp = client.get(f"/episodes/{ep_id}/smart_suggestions/batches", params={"run_id": run_b})
    assert resp.status_code == 200
    batches_b = resp.json().get("batches", [])
    batch_ids_b = [b.get("batch_id") for b in batches_b]
    assert batch_a_id not in batch_ids_b, f"Batch {batch_a_id} should NOT be in run B"


def test_dismissed_suggestions_are_run_scoped(tmp_path, monkeypatch):
    """Test that dismissed suggestions in run A don't affect run B."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    ep_id = "demo-s01e01"
    run_a = "run-dismiss-a"
    run_b = "run-dismiss-b"
    _setup_run(data_root, ep_id, run_a)
    _setup_run(data_root, ep_id, run_b)

    # Import run_persistence_service to create batches and suggestions
    from apps.api.services.run_persistence import run_persistence_service

    # Create batches in both runs
    batch_a_id = run_persistence_service.create_suggestion_batch(
        ep_id=ep_id, run_id=run_a, generator_version="test_v1", generator_config_json={}
    )
    batch_b_id = run_persistence_service.create_suggestion_batch(
        ep_id=ep_id, run_id=run_b, generator_version="test_v1", generator_config_json={}
    )

    # Insert same suggestion in both batches
    inserted_a = run_persistence_service.insert_suggestions(
        ep_id=ep_id,
        run_id=run_a,
        batch_id=batch_a_id,
        rows=[{
            "type": "cast_match",
            "target_identity_id": "id_0001",
            "suggested_person_id": "cast_001",
            "confidence": 0.85,
            "evidence_json": {},
        }],
    )
    suggestion_ids_a = [s.get("suggestion_id") for s in inserted_a]
    inserted_b = run_persistence_service.insert_suggestions(
        ep_id=ep_id,
        run_id=run_b,
        batch_id=batch_b_id,
        rows=[{
            "type": "cast_match",
            "target_identity_id": "id_0001",
            "suggested_person_id": "cast_001",
            "confidence": 0.85,
            "evidence_json": {},
        }],
    )

    client = TestClient(app)

    # Dismiss the suggestion in run A
    resp = client.post(
        f"/episodes/{ep_id}/smart_suggestions/dismiss",
        params={"run_id": run_a},
        json={"batch_id": batch_a_id, "suggestion_ids": suggestion_ids_a, "dismissed": True},
    )
    assert resp.status_code == 200, f"Dismiss failed: {resp.text}"

    # Verify suggestion is dismissed in run A
    suggestions_a = run_persistence_service.list_suggestions(
        ep_id=ep_id, run_id=run_a, batch_id=batch_a_id, include_dismissed=True
    )
    assert len(suggestions_a) == 1
    assert suggestions_a[0].get("dismissed") is True, "Suggestion should be dismissed in run A"

    # Verify suggestion is NOT dismissed in run B (run isolation)
    suggestions_b = run_persistence_service.list_suggestions(
        ep_id=ep_id, run_id=run_b, batch_id=batch_b_id, include_dismissed=True
    )
    assert len(suggestions_b) == 1
    assert suggestions_b[0].get("dismissed") is not True, "Suggestion should NOT be dismissed in run B"


def test_apply_all_skips_locked_identities(tmp_path, monkeypatch):
    """Test that apply_all skips locked identities and returns count.

    This test creates suggestions ONLY for locked identities, so apply_all
    should skip all of them without attempting batch_assign_clusters.
    """
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")

    ep_id = "demo-s01e01"
    run_id = "run-lock-test"
    _setup_run(data_root, ep_id, run_id)

    # Import run_persistence_service
    from apps.api.services.run_persistence import run_persistence_service

    # Create a batch with suggestions
    batch_id = run_persistence_service.create_suggestion_batch(
        ep_id=ep_id, run_id=run_id, generator_version="test_v1", generator_config_json={}
    )

    # Insert suggestions only for identities that will be locked
    run_persistence_service.insert_suggestions(
        ep_id=ep_id,
        run_id=run_id,
        batch_id=batch_id,
        rows=[
            {
                "type": "cast_match",
                "target_identity_id": "id_0001",
                "suggested_person_id": "cast_001",
                "confidence": 0.85,
                "evidence_json": {},
            },
            {
                "type": "cast_match",
                "target_identity_id": "id_0002",
                "suggested_person_id": "cast_002",
                "confidence": 0.80,
                "evidence_json": {},
            },
        ],
    )

    # Lock BOTH identities - so all suggestions will be skipped
    run_persistence_service.set_identity_lock(
        ep_id=ep_id,
        run_id=run_id,
        identity_id="id_0001",
        locked=True,
        reason="Testing lock enforcement",
    )
    run_persistence_service.set_identity_lock(
        ep_id=ep_id,
        run_id=run_id,
        identity_id="id_0002",
        locked=True,
        reason="Testing lock enforcement",
    )

    client = TestClient(app)

    # Call apply_all - all suggestions should be skipped because all are locked
    resp = client.post(
        f"/episodes/{ep_id}/smart_suggestions/apply_all",
        params={"run_id": run_id},
        json={"batch_id": batch_id},
    )
    assert resp.status_code == 200, f"Apply all failed: {resp.text}"

    data = resp.json()
    counts = data.get("counts", data)  # Handle both old and new response formats
    skipped_locked = counts.get("skipped_locked", counts.get("skipped_locked_count", 0))
    # Should have skipped both locked identities
    assert skipped_locked >= 2, f"Expected skipped_locked >= 2, got {counts}"
