import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from py_screenalytics.artifacts import ensure_dirs, get_path


@pytest.fixture(autouse=True)
def _setup_env(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    return data_root


def test_roster_add_and_get(monkeypatch):
    client = TestClient(app)
    show = "demo"
    resp = client.get(f"/shows/{show}/cast_names")
    assert resp.status_code == 200
    assert resp.json()["names"] == []

    resp = client.post(f"/shows/{show}/cast_names", json={"name": "Kyle Richards"})
    assert resp.status_code == 200
    assert resp.json()["names"] == ["Kyle Richards"]

    # Duplicate (different casing) should not create a second entry.
    resp = client.post(f"/shows/{show}/cast_names", json={"name": "kyle richards"})
    assert resp.status_code == 200
    assert resp.json()["names"] == ["Kyle Richards"]

    resp = client.get(f"/shows/{show}/cast_names")
    assert resp.status_code == 200
    assert resp.json()["names"] == ["Kyle Richards"]


def test_identity_name_assign_adds_to_roster(tmp_path, monkeypatch):
    client = TestClient(app)
    ep_id = "demo-s01e01"
    ensure_dirs(ep_id)
    manifests_dir = get_path(ep_id, "detections").parent
    identities_path = manifests_dir / "identities.json"
    identities_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ep_id": ep_id,
        "identities": [{"identity_id": "id_0001", "label": None, "track_ids": [1]}],
        "stats": {},
    }
    identities_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    resp = client.post(
        f"/episodes/{ep_id}/identities/id_0001/name",
        json={"name": "Lisa Vanderpump", "show": "demo"},
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "Lisa Vanderpump"

    updated = json.loads(identities_path.read_text(encoding="utf-8"))
    assert updated["identities"][0]["name"] == "Lisa Vanderpump"

    roster_path = Path(tmp_path) / "data" / "rosters" / "demo.json"
    assert roster_path.exists()
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    assert roster["names"] == ["Lisa Vanderpump"]
