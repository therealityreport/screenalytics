"""Test cast CRUD endpoints and bulk import."""

import json
import os
from pathlib import Path

os.environ.setdefault("STORAGE_BACKEND", "local")

from fastapi.testclient import TestClient

from apps.api.main import app


def test_cast_crud(tmp_path, monkeypatch):
    """Test create, read, update, delete cast members."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    client = TestClient(app)
    show_id = "rhobh"

    # List cast (empty initially)
    resp = client.get(f"/shows/{show_id}/cast")
    assert resp.status_code == 200
    data = resp.json()
    assert data["show_id"] == show_id
    assert data["count"] == 0
    assert data["cast"] == []

    # Create cast member
    payload = {
        "name": "Kyle Richards",
        "role": "main",
        "status": "active",
        "aliases": ["Kyle", "Kyle R"],
        "seasons": ["S01", "S02", "S03"],
        "social": {"instagram": "@kylerichards18"},
    }
    resp = client.post(f"/shows/{show_id}/cast", json=payload)
    assert resp.status_code == 200
    member = resp.json()
    assert member["name"] == "Kyle Richards"
    assert member["role"] == "main"
    assert member["status"] == "active"
    assert member["aliases"] == ["Kyle", "Kyle R"]
    assert member["seasons"] == ["S01", "S02", "S03"]
    cast_id = member["cast_id"]

    # Get cast member
    resp = client.get(f"/shows/{show_id}/cast/{cast_id}")
    assert resp.status_code == 200
    member = resp.json()
    assert member["name"] == "Kyle Richards"
    assert member["cast_id"] == cast_id

    # Update cast member
    update = {"role": "friend", "status": "past", "seasons": ["S01"]}
    resp = client.patch(f"/shows/{show_id}/cast/{cast_id}", json=update)
    assert resp.status_code == 200
    member = resp.json()
    assert member["role"] == "friend"
    assert member["status"] == "past"
    assert member["seasons"] == ["S01"]

    # List cast (should have 1 member now)
    resp = client.get(f"/shows/{show_id}/cast")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert len(data["cast"]) == 1

    # Delete cast member
    resp = client.delete(f"/shows/{show_id}/cast/{cast_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # Verify deletion
    resp = client.get(f"/shows/{show_id}/cast/{cast_id}")
    assert resp.status_code == 404


def test_cast_bulk_import(tmp_path, monkeypatch):
    """Test bulk import with merge by name."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    client = TestClient(app)
    show_id = "rhobh"

    # Create initial cast member
    payload = {
        "name": "Kyle Richards",
        "role": "main",
        "status": "active",
        "seasons": ["S01"],
    }
    resp = client.post(f"/shows/{show_id}/cast", json=payload)
    assert resp.status_code == 200
    kyle_id = resp.json()["cast_id"]

    # Bulk import (should merge Kyle and create Lisa)
    import_payload = {
        "members": [
            {
                "name": "Kyle Richards",  # Should merge with existing
                "role": "main",
                "status": "active",
                "seasons": ["S01", "S02", "S03"],
            },
            {
                "name": "Lisa Vanderpump",  # Should create new
                "role": "main",
                "status": "past",
                "seasons": ["S01", "S02"],
            },
        ],
        "force_new": False,
    }
    resp = client.post(f"/shows/{show_id}/cast/import", json=import_payload)
    assert resp.status_code == 200
    result = resp.json()
    assert result["total"] == 2
    assert result["created_count"] == 1  # Lisa
    assert result["updated_count"] == 1  # Kyle
    assert result["skipped_count"] == 0

    # Verify Kyle was updated
    resp = client.get(f"/shows/{show_id}/cast/{kyle_id}")
    assert resp.status_code == 200
    kyle = resp.json()
    assert kyle["seasons"] == ["S01", "S02", "S03"]

    # Verify Lisa was created
    resp = client.get(f"/shows/{show_id}/cast")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    lisa = next((m for m in data["cast"] if m["name"] == "Lisa Vanderpump"), None)
    assert lisa is not None
    assert lisa["status"] == "past"


def test_cast_bulk_import_force_new(tmp_path, monkeypatch):
    """Test bulk import with force_new=True."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    client = TestClient(app)
    show_id = "rhobh"

    # Create initial cast member
    payload = {
        "name": "Kyle Richards",
        "role": "main",
        "status": "active",
    }
    resp = client.post(f"/shows/{show_id}/cast", json=payload)
    assert resp.status_code == 200

    # Bulk import with force_new=True
    import_payload = {
        "members": [
            {"name": "Kyle Richards", "role": "friend"}  # Should create duplicate
        ],
        "force_new": True,
    }
    resp = client.post(f"/shows/{show_id}/cast/import", json=import_payload)
    assert resp.status_code == 200
    result = resp.json()
    assert result["created_count"] == 1  # New Kyle
    assert result["updated_count"] == 0

    # Verify there are now 2 Kyles
    resp = client.get(f"/shows/{show_id}/cast")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    kyles = [m for m in data["cast"] if m["name"] == "Kyle Richards"]
    assert len(kyles) == 2


def test_cast_season_filter(tmp_path, monkeypatch):
    """Test filtering cast by season."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    client = TestClient(app)
    show_id = "rhobh"

    # Create cast members with different seasons
    members = [
        {"name": "Kyle Richards", "seasons": ["S01", "S02", "S03"]},
        {"name": "Lisa Vanderpump", "seasons": ["S01", "S02"]},
        {"name": "Erika Jayne", "seasons": ["S06", "S07"]},
    ]
    for member in members:
        resp = client.post(f"/shows/{show_id}/cast", json=member)
        assert resp.status_code == 200

    # Filter by S01 (Kyle and Lisa)
    resp = client.get(f"/shows/{show_id}/cast?season=S01")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    names = {m["name"] for m in data["cast"]}
    assert names == {"Kyle Richards", "Lisa Vanderpump"}

    # Filter by S06 (Erika only)
    resp = client.get(f"/shows/{show_id}/cast?season=S06")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["cast"][0]["name"] == "Erika Jayne"
