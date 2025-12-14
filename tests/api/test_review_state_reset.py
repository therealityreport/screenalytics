from __future__ import annotations

import json

from fastapi.testclient import TestClient

from apps.api.main import app
from py_screenalytics.artifacts import get_path


def test_reset_face_review_state_archives_and_clears(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "review-reset-demo"

    manifests_dir = get_path(ep_id, "detections").parent
    manifests_dir.mkdir(parents=True, exist_ok=True)
    state_path = manifests_dir / "face_review_state.json"
    original_state = {
        "initial_unassigned_pass_done": True,
        "decisions": [
            {
                "pair_type": "unassigned_unassigned",
                "cluster_a": "id_0001",
                "cluster_b": "id_0002",
                "person_id": None,
                "decision": "no",
                "timestamp": "2025-01-01T00:00:00Z",
            }
        ],
        "updated_at": "2025-01-01T00:00:00Z",
    }
    state_path.write_text(json.dumps(original_state), encoding="utf-8")

    client = TestClient(app)
    resp = client.post(f"/episodes/{ep_id}/face_review/reset_state", json={"archive_existing": True})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("status") == "success"

    archived_name = payload.get("archived")
    assert isinstance(archived_name, str) and archived_name
    assert (manifests_dir / archived_name).exists()

    new_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert new_state.get("initial_unassigned_pass_done") is False
    assert new_state.get("decisions") == []


def test_reset_dismissed_suggestions_archives_and_clears(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    ep_id = "dismissed-reset-demo"

    manifests_dir = get_path(ep_id, "detections").parent
    manifests_dir.mkdir(parents=True, exist_ok=True)
    dismissed_path = manifests_dir / "dismissed_suggestions.json"
    dismissed_path.write_text(
        json.dumps({"dismissed": ["cluster_001", "person:p_0001"], "updated_at": "2025-01-01T00:00:00Z"}),
        encoding="utf-8",
    )

    client = TestClient(app)
    resp = client.post(
        f"/episodes/{ep_id}/dismissed_suggestions/reset_state",
        json={"archive_existing": True},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("status") == "success"

    archived_name = payload.get("archived")
    assert isinstance(archived_name, str) and archived_name
    assert (manifests_dir / archived_name).exists()

    new_state = json.loads(dismissed_path.read_text(encoding="utf-8"))
    assert new_state.get("dismissed") == []
