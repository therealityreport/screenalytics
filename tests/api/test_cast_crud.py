"""Test cast CRUD endpoints and bulk import."""

import json
import os

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


def test_cast_list_include_featured_thumbnail(tmp_path, monkeypatch):
    """Ensure include_featured returns featured thumbnail URLs."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    client = TestClient(app)
    show_id = "rhobh"

    # Create a cast member
    payload = {
        "name": "Brandi Glanville",
        "role": "main",
        "status": "active",
    }
    resp = client.post(f"/shows/{show_id}/cast", json=payload)
    assert resp.status_code == 200
    cast_id = resp.json()["cast_id"]

    # Create facebank payload with a featured seed
    facebank_dir = data_root / "facebank" / show_id / cast_id
    facebank_dir.mkdir(parents=True, exist_ok=True)
    featured_seed_id = "seed-123"
    facebank_payload = {
        "show_id": show_id,
        "cast_id": cast_id,
        "seeds": [
            {
                "fb_id": featured_seed_id,
                "display_uri": "local/thumbs/featured.png",
                "image_uri": "local/thumbs/featured.png",
            }
        ],
        "exemplars": [],
        "featured_seed_id": featured_seed_id,
        "updated_at": "2024-01-01T00:00:00Z",
    }
    (facebank_dir / "facebank.json").write_text(json.dumps(facebank_payload))

    resp = client.get(f"/shows/{show_id}/cast", params={"include_featured": "1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    entry = data["cast"][0]
    assert entry["featured_thumbnail_url"] == "local/thumbs/featured.png"


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
        "members": [{"name": "Kyle Richards", "role": "friend"}],  # Should create duplicate
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


def test_show_registry(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    client = TestClient(app)

    resp = client.post(
        "/shows",
        json={
            "show_id": "newshow",
            "title": "RHOBH",
            "full_name": "The Real Housewives of Beverly Hills",
            "imdb_series_id": "tt1720601",
        },
    )
    assert resp.status_code == 200
    created = resp.json()
    assert created["show_id"] == "NEWSHOW"
    assert created["title"] == "RHOBH"
    assert created["full_name"] == "The Real Housewives of Beverly Hills"
    assert created["imdb_series_id"] == "tt1720601"
    assert created["created"] is True
    assert created["cast_count"] == 0

    resp = client.get("/shows")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 1
    assert payload["shows"][0]["show_id"] == "NEWSHOW"
    assert payload["shows"][0]["full_name"] == "The Real Housewives of Beverly Hills"
    assert payload["shows"][0]["imdb_series_id"] == "tt1720601"

    resp = client.post(
        "/shows",
        json={"show_id": "newshow", "title": "Updated", "imdb_series_id": "tt9999999"},
    )
    assert resp.status_code == 200
    updated = resp.json()
    assert updated["created"] is False
    assert updated["title"] == "Updated"
    assert updated["full_name"] == "The Real Housewives of Beverly Hills"
    assert updated["imdb_series_id"] == "tt9999999"
