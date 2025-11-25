import os

os.environ.setdefault("STORAGE_BACKEND", "local")

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import people as people_router
from apps.api.services.people import PeopleService
from apps.api.services.storage import StorageService


def test_create_person_with_cast_id(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    people_router.people_service = PeopleService(data_root=data_root)
    people_router.storage_service = StorageService()
    client = TestClient(app)

    show_id = "demo"
    payload = {"name": "Alice", "cast_id": "cast_001"}
    resp = client.post(f"/shows/{show_id}/people", json=payload)
    assert resp.status_code == 200
    person = resp.json()
    assert person["cast_id"] == "cast_001"

    # Listing should include cast_id
    resp = client.get(f"/shows/{show_id}/people")
    assert resp.status_code == 200
    people = resp.json().get("people", [])
    assert people and people[0]["cast_id"] == "cast_001"


def test_update_person_cast_id(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    people_router.people_service = PeopleService(data_root=data_root)
    people_router.storage_service = StorageService()
    client = TestClient(app)

    show_id = "demo"
    resp = client.post(f"/shows/{show_id}/people", json={"name": "Bob"})
    assert resp.status_code == 200
    person_id = resp.json()["person_id"]

    update = {"cast_id": "cast_002"}
    resp = client.patch(f"/shows/{show_id}/people/{person_id}", json=update)
    assert resp.status_code == 200
    updated = resp.json()
    assert updated["cast_id"] == "cast_002"

    # Persisted value remains on list read
    resp = client.get(f"/shows/{show_id}/people")
    assert resp.status_code == 200
    people = resp.json().get("people", [])
    stored = next(p for p in people if p.get("person_id") == person_id)
    assert stored["cast_id"] == "cast_002"
