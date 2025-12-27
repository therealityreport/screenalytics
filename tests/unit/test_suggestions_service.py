from __future__ import annotations

import numpy as np

from apps.api.services import suggestions as sugg


def _norm(vec: list[float]) -> np.ndarray:
    arr = np.array(vec, dtype=np.float32)
    normalized = sugg._l2_normalize(arr)
    assert normalized is not None
    return normalized


def test_suggestions_exclude_faces_from_centroid() -> None:
    faces = [
        {"face_id": "f1", "track_id": 1, "embedding": [1.0, 0.0]},
        {"face_id": "f2", "track_id": 1, "embedding": [0.0, 1.0]},
    ]
    track_embeddings, _ = sugg._track_embeddings_from_faces(faces, {"f2"})
    cluster_embeddings = sugg._cluster_embeddings_from_tracks({"c1": [1]}, track_embeddings, {})

    cast_embeddings = {
        "cast_a": _norm([1.0, 0.0]),
        "cast_b": _norm([0.0, 1.0]),
    }
    suggestions = sugg.compute_cast_suggestions(
        cluster_embeddings,
        cast_embeddings,
        top_k=2,
        min_similarity=0.0,
    )
    assert suggestions["c1"][0]["cast_id"] == "cast_a"


def test_suggestions_order_stable_on_ties() -> None:
    cluster_embeddings = {"c1": _norm([1.0, 0.0])}
    cast_embeddings = {
        "cast_b": _norm([1.0, 0.0]),
        "cast_a": _norm([1.0, 0.0]),
    }
    suggestions = sugg.compute_cast_suggestions(
        cluster_embeddings,
        cast_embeddings,
        top_k=2,
        min_similarity=0.0,
    )
    assert [entry["cast_id"] for entry in suggestions["c1"]] == ["cast_a", "cast_b"]
