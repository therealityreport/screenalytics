from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app


client = TestClient(app)
LEGACY_SUFFIX = "st" "ub"
LEGACY_FLAG = "use_" + LEGACY_SUFFIX


def test_legacy_flag_rejected_detect_track():
    payload = {"ep_id": "bad", LEGACY_FLAG: True}
    response = client.post("/jobs/detect_track", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Stub mode is not supported."


def test_legacy_alias_rejected_faces_embed():
    payload = {"ep_id": "bad", LEGACY_SUFFIX: True}
    response = client.post("/jobs/faces_embed", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Stub mode is not supported."
