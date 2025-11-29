import os

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routers import grouping as grouping_router
from apps.api.routers import people as people_router
from apps.api.services.grouping import GroupingService
from apps.api.services.people import PeopleService
from apps.api.services.storage import StorageService


@pytest.fixture()
def isolated_services(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    # Reset router-level services to the temp data root
    people_router.people_service = PeopleService(data_root=data_root)
    people_router.storage_service = StorageService()

    grouping_router.grouping_service = GroupingService(data_root=data_root)
    grouping_router.grouping_service.people_service = PeopleService(data_root=data_root)
    # Avoid filesystem requirement for centroids/identities
    grouping_router.grouping_service.load_cluster_centroids = (
        lambda ep_id: {"centroids": {"id_0001": {"centroid": [0.0, 0.0, 0.0, 0.0]}}}
    )
    grouping_router.grouping_service._update_identities_with_people = lambda *a, **k: None

    return data_root


def test_manual_assign_sets_cast_id(tmp_path, isolated_services):
    client = TestClient(app)
    show_id = "rhoslc"
    ep_id = "rhoslc-s06e02"

    # Create a person without cast_id
    resp = client.post(f"/shows/{show_id}/people", json={"name": "Heather"})
    assert resp.status_code == 200
    person_id = resp.json()["person_id"]

    payload = {
        "strategy": "manual",
        "cluster_ids": ["id_0001"],
        "target_person_id": person_id,
        "cast_id": "cast_heather",
    }
    resp = client.post(f"/episodes/{ep_id}/clusters/group", json=payload)
    assert resp.status_code == 200

    people = client.get(f"/shows/{show_id}/people").json().get("people", [])
    target = next(p for p in people if p.get("person_id") == person_id)
    assert target.get("cast_id") == "cast_heather"
    assert f"{ep_id}:id_0001" in target.get("cluster_ids", [])
