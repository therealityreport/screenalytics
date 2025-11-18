from __future__ import annotations

import json

from apps.api.services import identities as identity_service


def _write_identities(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_identities(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_assign_track_name_splits_cluster(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "testshow-s01e01"
    manifests_dir = data_root / "manifests" / ep_id
    identities_path = manifests_dir / "identities.json"
    payload = {
        "ep_id": ep_id,
        "identities": [
            {
                "identity_id": "id_0001",
                "track_ids": [1, 2],
                "name": None,
            }
        ],
        "stats": {},
    }
    _write_identities(identities_path, payload)

    result = identity_service.assign_track_name(ep_id, 1, "Kyle", show="testshow")

    assert result["split"] is True
    assert result["track_id"] == 1
    assert result["identity_id"] != "id_0001"

    updated = _read_identities(identities_path)
    assert updated["stats"]["clusters"] == 2
    original = next(item for item in updated["identities"] if item["identity_id"] == "id_0001")
    assert original["track_ids"] == [2]
    new_identity = next(item for item in updated["identities"] if item["identity_id"] == result["identity_id"])
    assert new_identity["track_ids"] == [1]
    assert new_identity["name"] == "Kyle"


def test_assign_track_name_reuses_identity_for_single_track(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "demo-s02e03"
    manifests_dir = data_root / "manifests" / ep_id
    identities_path = manifests_dir / "identities.json"
    payload = {
        "ep_id": ep_id,
        "identities": [
            {
                "identity_id": "id_0005",
                "track_ids": [7],
                "name": None,
            }
        ],
        "stats": {},
    }
    _write_identities(identities_path, payload)

    result = identity_service.assign_track_name(ep_id, 7, "Lisa", show="demo")

    assert result["split"] is False
    assert result["identity_id"] == "id_0005"
    updated = _read_identities(identities_path)
    assert len(updated["identities"]) == 1
    assert updated["identities"][0]["name"] == "Lisa"
    assert updated["identities"][0]["track_ids"] == [7]
