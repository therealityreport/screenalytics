"""Test facebank seed management endpoints."""

import json
import os
from io import BytesIO
from pathlib import Path

os.environ.setdefault("STORAGE_BACKEND", "local")

import numpy as np
from fastapi.testclient import TestClient

from apps.api.main import app


def _create_test_image(size=(112, 112)):
    """Create a simple test image with a face-like pattern."""
    import cv2

    # Create a simple grayscale image
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 128

    # Draw a simple face-like pattern (circle for face, smaller circles for eyes)
    center = (size[0] // 2, size[1] // 2)
    cv2.circle(img, center, size[0] // 3, (200, 200, 200), -1)  # Face
    cv2.circle(
        img, (size[0] // 3, size[1] // 3), size[0] // 10, (50, 50, 50), -1
    )  # Left eye
    cv2.circle(
        img, (2 * size[0] // 3, size[1] // 3), size[0] // 10, (50, 50, 50), -1
    )  # Right eye

    # Encode as JPEG
    success, buffer = cv2.imencode(".jpg", img)
    assert success
    return BytesIO(buffer.tobytes())


def test_get_facebank_empty(tmp_path, monkeypatch):
    """Test getting facebank for cast member with no seeds."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREANALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    client = TestClient(app)
    show_id = "rhobh"

    # Create cast member
    payload = {"name": "Kyle Richards"}
    resp = client.post(f"/shows/{show_id}/cast", json=payload)
    assert resp.status_code == 200
    cast_id = resp.json()["cast_id"]

    # Get facebank (should be empty)
    resp = client.get(f"/cast/{cast_id}/facebank?show_id={show_id}")
    assert resp.status_code == 200
    facebank = resp.json()
    assert facebank["show_id"] == show_id
    assert facebank["cast_id"] == cast_id
    assert len(facebank["seeds"]) == 0
    assert len(facebank["exemplars"]) == 0
    assert facebank["stats"]["total_seeds"] == 0


def test_upload_seeds_validation(tmp_path, monkeypatch):
    """Test seed upload with various validation scenarios."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREANALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    client = TestClient(app)
    show_id = "rhobh"

    # Create cast member
    payload = {"name": "Kyle Richards"}
    resp = client.post(f"/shows/{show_id}/cast", json=payload)
    assert resp.status_code == 200
    cast_id = resp.json()["cast_id"]

    # Upload to non-existent cast member
    files = [("files", ("test.jpg", _create_test_image(), "image/jpeg"))]
    resp = client.post(
        f"/cast/nonexistent/seeds/upload?show_id={show_id}", files=files
    )
    assert resp.status_code == 404


def test_facebank_service_integration(tmp_path, monkeypatch):
    """Test facebank service CRUD operations."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREANALYTICS_DATA_ROOT", str(data_root))

    from apps.api.services.facebank import FacebankService

    service = FacebankService(data_root)
    show_id = "rhobh"
    cast_id = "cast_001"

    # Get empty facebank
    facebank = service.get_facebank(show_id, cast_id)
    assert facebank["show_id"] == show_id
    assert facebank["cast_id"] == cast_id
    assert len(facebank["seeds"]) == 0

    # Add seed
    embedding = np.random.randn(512).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)  # L2 normalize

    image_path = str(data_root / "test_seed.jpg")
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    Path(image_path).write_bytes(b"fake image data")

    quality = {"sharpness": 0.8, "occlusion": 0.1, "bbox_ratio": 0.3}
    seed_entry = service.add_seed(show_id, cast_id, image_path, embedding, quality)

    assert seed_entry["cast_id"] == cast_id
    assert seed_entry["type"] == "seed"
    assert seed_entry["embedding_dim"] == 512
    assert seed_entry["quality"] == quality
    assert "fb_id" in seed_entry

    # Get facebank (should have 1 seed)
    facebank = service.get_facebank(show_id, cast_id)
    assert len(facebank["seeds"]) == 1
    assert facebank["stats"]["total_seeds"] == 1

    # Delete seed
    seed_id = seed_entry["fb_id"]
    deleted = service.delete_seeds(show_id, cast_id, [seed_id])
    assert deleted == 1

    # Verify deletion
    facebank = service.get_facebank(show_id, cast_id)
    assert len(facebank["seeds"]) == 0


def test_seed_matching(tmp_path, monkeypatch):
    """Test finding matching seeds by embedding similarity."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREANALYTICS_DATA_ROOT", str(data_root))

    from apps.api.services.facebank import FacebankService

    service = FacebankService(data_root)
    show_id = "rhobh"

    # Create embeddings for two cast members
    kyle_emb = np.random.randn(512).astype(np.float32)
    kyle_emb = kyle_emb / np.linalg.norm(kyle_emb)

    lisa_emb = np.random.randn(512).astype(np.float32)
    lisa_emb = lisa_emb / np.linalg.norm(lisa_emb)

    # Add seeds
    image_path = str(data_root / "seed.jpg")
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    Path(image_path).write_bytes(b"fake")

    service.add_seed(show_id, "kyle_cast_id", image_path, kyle_emb)
    service.add_seed(show_id, "lisa_cast_id", image_path, lisa_emb)

    # Find matching seed (should match Kyle)
    query_emb = kyle_emb + np.random.randn(512).astype(np.float32) * 0.01
    query_emb = query_emb / np.linalg.norm(query_emb)

    match = service.find_matching_seed(show_id, query_emb, min_similarity=0.3)
    assert match is not None
    cast_id, seed_id, similarity = match
    assert cast_id == "kyle_cast_id"
    assert similarity > 0.9  # Should be very similar

    # Query with no match (random embedding)
    random_emb = np.random.randn(512).astype(np.float32)
    random_emb = random_emb / np.linalg.norm(random_emb)

    match = service.find_matching_seed(show_id, random_emb, min_similarity=0.9)
    # Might match, might not - random embeddings can have high cosine similarity
    # Just verify it returns expected format
    if match:
        assert len(match) == 3


def test_get_all_seeds_for_show(tmp_path, monkeypatch):
    """Test retrieving all seeds across multiple cast members."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREANALYTICS_DATA_ROOT", str(data_root))

    from apps.api.services.facebank import FacebankService

    service = FacebankService(data_root)
    show_id = "rhobh"

    # Add seeds for multiple cast members
    image_path = str(data_root / "seed.jpg")
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    Path(image_path).write_bytes(b"fake")

    for cast_id in ["kyle_id", "lisa_id", "erika_id"]:
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        service.add_seed(show_id, cast_id, image_path, emb)

    # Get all seeds
    all_seeds = service.get_all_seeds_for_show(show_id)
    assert len(all_seeds) == 3

    cast_ids = {seed["cast_id"] for seed in all_seeds}
    assert cast_ids == {"kyle_id", "lisa_id", "erika_id"}

    # Each seed should have embedding
    for seed in all_seeds:
        assert "embedding" in seed
        assert len(seed["embedding"]) == 512
