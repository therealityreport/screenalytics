import os

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import people as people_router
from apps.api.services.people import PeopleService


@pytest.fixture()
def temp_services(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    # Reset people service to use the temp data root (cast uses helper with env)
    people_router.people_service = PeopleService(data_root=data_root)
    return data_root


def test_cast_creation_creates_person_with_cast_id(temp_services):
    client = TestClient(app)
    show_id = "rhoslc"

    payload = {
        "name": "Heather Gay",
        "role": "main",
        "status": "active",
        "aliases": ["Heather"],
    }
    resp = client.post(f"/shows/{show_id}/cast", json=payload)
    assert resp.status_code == 200
    cast_id = resp.json()["cast_id"]

    people = client.get(f"/shows/{show_id}/people").json().get("people", [])
    target = next((p for p in people if p.get("name") == "Heather Gay"), None)
    assert target is not None
    assert target.get("cast_id") == cast_id
    assert "Heather" in (target.get("aliases") or [])


def test_cast_creation_updates_existing_person(temp_services):
    client = TestClient(app)
    show_id = "rhoslc"

    # Seed an existing person without cast_id
    seed_resp = client.post(f"/shows/{show_id}/people", json={"name": "Lisa Barlow", "aliases": ["Lisa"]})
    assert seed_resp.status_code == 200
    person_id = seed_resp.json()["person_id"]

    # Create cast member with same name; should update the existing person with cast_id
    payload = {
        "name": "Lisa Barlow",
        "role": "main",
        "status": "active",
    }
    cast_resp = client.post(f"/shows/{show_id}/cast", json=payload)
    assert cast_resp.status_code == 200
    cast_id = cast_resp.json()["cast_id"]

    people = client.get(f"/shows/{show_id}/people").json().get("people", [])
    target = next(p for p in people if p.get("person_id") == person_id)
    assert target.get("cast_id") == cast_id
