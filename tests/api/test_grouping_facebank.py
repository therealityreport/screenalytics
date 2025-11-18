from __future__ import annotations

import json

import numpy as np

from apps.api.services.cast import CastService
from apps.api.services.facebank import FacebankService
from apps.api.services.grouping import GroupingService


def test_group_using_facebank_assigns_clusters(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    show_id = "TEST"
    cast_service = CastService(data_root)
    cast_member = cast_service.create_cast_member(show_id, name="Tester")

    facebank_service = FacebankService(data_root)
    image_path = data_root / "seed.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"seed")
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    facebank_service.add_seed(
        show_id, cast_member["cast_id"], str(image_path), embedding
    )

    ep_id = "test-s01e01"
    manifests_dir = data_root / "manifests" / ep_id
    manifests_dir.mkdir(parents=True, exist_ok=True)

    centroids_payload = {
        "ep_id": ep_id,
        "centroids": [
            {
                "cluster_id": "id_0001",
                "centroid": embedding.tolist(),
                "num_faces": 3,
            }
        ],
    }
    (manifests_dir / "cluster_centroids.json").write_text(
        json.dumps(centroids_payload), encoding="utf-8"
    )

    identities_payload = {
        "ep_id": ep_id,
        "identities": [
            {
                "identity_id": "id_0001",
                "label": None,
                "track_ids": [1],
                "size": 1,
            }
        ],
        "stats": {},
    }
    (manifests_dir / "identities.json").write_text(
        json.dumps(identities_payload), encoding="utf-8"
    )

    service = GroupingService(data_root)
    result = service.group_using_facebank(ep_id)

    assert result["matched_clusters"] == 1
    assert not result["unmatched_clusters"]
    assert result["assigned"][0]["cast_id"] == cast_member["cast_id"]

    updated_identities = json.loads(
        (manifests_dir / "identities.json").read_text(encoding="utf-8")
    )
    assert updated_identities["identities"][0]["person_id"]
