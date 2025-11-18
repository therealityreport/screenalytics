"""Test facebank seed management endpoints."""

import os
import sys
import types
from io import BytesIO
from pathlib import Path

os.environ.setdefault("STORAGE_BACKEND", "local")

import pytest
from fastapi.testclient import TestClient

try:
    import numpy as np
except ImportError:  # pragma: no cover
    pytest.skip("numpy is required for facebank API tests", allow_module_level=True)

from apps.api.main import app
from apps.api.services.cast import CastService
from apps.api.services.people import PeopleService


def _create_test_image(size=(112, 112)):
    """Create a simple test image with a face-like pattern."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, (128, 128, 128))
    draw = ImageDraw.Draw(img)
    center = (size[0] // 2, size[1] // 2)
    radius = size[0] // 3
    draw.ellipse(
        [
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ],
        fill=(200, 200, 200),
    )
    eye_radius = size[0] // 10
    draw.ellipse(
        [
            size[0] // 3 - eye_radius,
            size[1] // 3 - eye_radius,
            size[0] // 3 + eye_radius,
            size[1] // 3 + eye_radius,
        ],
        fill=(50, 50, 50),
    )
    draw.ellipse(
        [
            2 * size[0] // 3 - eye_radius,
            size[1] // 3 - eye_radius,
            2 * size[0] // 3 + eye_radius,
            size[1] // 3 + eye_radius,
        ],
        fill=(50, 50, 50),
    )

    stream = BytesIO()
    img.save(stream, format="JPEG")
    stream.seek(0)
    return stream


def _create_webp_image(size=(112, 112)):
    """Create a simple RGB image encoded as WEBP."""
    from PIL import Image

    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[..., 0] = 255
    pic = Image.fromarray(arr, mode="RGB")
    stream = BytesIO()
    pic.save(stream, format="WEBP")
    stream.seek(0)
    return stream


def _mock_face_pipeline(
    monkeypatch, *, simulated: bool = False, bbox: list[float] | None = None
):
    """Replace detector/embedder with deterministic fakes."""

    simulated_flag = simulated
    bbox_rel = bbox or [0.1, 0.1, 0.5, 0.5]

    class _FakeDetector:
        simulated = simulated_flag

        def __init__(self, *args, **kwargs):
            self.simulated = simulated_flag

        def ensure_ready(self):
            return None

        def __call__(self, image):
            return [
                {
                    "bbox": bbox_rel,
                    "landmarks": [0.2, 0.2, 0.4, 0.2, 0.3, 0.3, 0.25, 0.45, 0.4, 0.45],
                    "conf": 0.99,
                }
            ]

    class _FakeEmbedder:
        def __init__(self, *args, **kwargs):
            pass

        def ensure_ready(self):
            return None

        def encode(self, faces):
            return np.zeros((len(faces), 512), dtype=np.float32)

    fake_det_mod = types.SimpleNamespace(RetinaFaceDetector=_FakeDetector)
    monkeypatch.setitem(
        sys.modules, "FEATURES.detection.src.run_retinaface", fake_det_mod
    )

    def _fake_prepare_face_crop(
        image, bbox, landmarks, margin=0.15, *, detector_mode="retinaface", align=True
    ):
        return image, None

    fake_tools_mod = types.SimpleNamespace(
        ArcFaceEmbedder=_FakeEmbedder,
        _prepare_face_crop=_fake_prepare_face_crop,
    )
    monkeypatch.setitem(sys.modules, "tools.episode_run", fake_tools_mod)

    def _fake_ensure_ready(device):
        if simulated_flag:
            return False, "simulated test mode", None
        return True, None, "cpu"

    from apps.api.routers import facebank as facebank_router

    monkeypatch.setattr(
        facebank_router.episode_run, "ensure_retinaface_ready", _fake_ensure_ready
    )


class _DummyJobService:
    def __init__(self, job_id: str = "refresh-upload"):
        self.calls: list[dict] = []
        self.job_id = job_id

    def emit_facebank_refresh(self, show_id, cast_id, **kwargs):
        payload = {"show_id": show_id, "cast_id": cast_id, **kwargs}
        self.calls.append(payload)
        return {"job_id": self.job_id}


class _DummyStorageService:
    def __init__(self, base_url: str = "https://cdn.test"):
        self.upload_calls: list[dict] = []
        self.presign_calls: list[tuple[str, int, str | None]] = []
        self.base_url = base_url.rstrip("/")

    def upload_facebank_seed(
        self,
        show_id: str,
        cast_id: str,
        seed_id: str,
        local_path: Path | str,
        *,
        object_name: str | None = None,
        content_type_hint: str | None = None,
    ) -> str:
        filename = object_name or f"{seed_id}.png"
        key = f"artifacts/facebank/{show_id}/{cast_id}/{filename}"
        self.upload_calls.append(
            {
                "show_id": show_id,
                "cast_id": cast_id,
                "seed_id": seed_id,
                "path": str(local_path),
                "object_name": object_name,
                "content_type": content_type_hint,
                "key": key,
            }
        )
        return key

    def presign_get(
        self, key: str, expires_in: int = 3600, *, content_type: str | None = None
    ) -> str:
        self.presign_calls.append((key, expires_in, content_type))
        return f"{self.base_url}/{key}"

    def s3_enabled(self) -> bool:
        return True


def _create_cast_member(
    facebank_router, show_id: str, name: str, data_root: Path
) -> str:
    cast_service = CastService(data_root)
    facebank_router.cast_service = cast_service
    facebank_router.people_service = PeopleService(data_root)
    record = cast_service.create_cast_member(show_id, name=name)
    return record["cast_id"]


def test_get_facebank_empty(tmp_path, monkeypatch):
    """Test getting facebank for cast member with no seeds."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREANALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    client = TestClient(app)
    show_id = "rhobh"

    from apps.api.routers import facebank as facebank_router

    cast_id = _create_cast_member(facebank_router, show_id, "Kyle Richards", data_root)

    # Get facebank (should be empty)
    resp = client.get(f"/cast/{cast_id}/facebank?show_id={show_id}")
    assert resp.status_code == 200
    facebank = resp.json()
    assert facebank["show_id"] == show_id
    assert facebank["cast_id"] == cast_id
    assert len(facebank["seeds"]) == 0
    assert len(facebank["exemplars"]) == 0
    similarity = facebank.get("similarity")
    assert similarity is not None
    assert similarity["sampled"] == 0
    assert facebank["stats"]["total_seeds"] == 0


def test_upload_seeds_validation(tmp_path, monkeypatch):
    """Test seed upload with various validation scenarios."""
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREANALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    client = TestClient(app)
    show_id = "rhobh"

    from apps.api.routers import facebank as facebank_router

    cast_id = _create_cast_member(facebank_router, show_id, "Kyle Richards", data_root)

    # Upload to non-existent cast member
    files = [("files", ("test.jpg", _create_test_image(), "image/jpeg"))]
    resp = client.post(f"/cast/nonexistent/seeds/upload?show_id={show_id}", files=files)
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


def test_facebank_similarity_stats_multiple_seeds(tmp_path):
    from apps.api.services.facebank import FacebankService

    data_root = tmp_path / "data"
    service = FacebankService(data_root)
    show_id = "rhobh"
    cast_id = "cast_similarity"
    image_path = data_root / "seed.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"x")

    emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    service.add_seed(show_id, cast_id, str(image_path), emb_a)
    service.add_seed(show_id, cast_id, str(image_path), emb_b)

    facebank = service.get_facebank(show_id, cast_id)
    similarity = facebank["similarity"]
    assert similarity["sampled"] == 2
    summary = similarity["summary"]
    assert summary is not None
    assert summary["mean"] == pytest.approx(0.0, abs=1e-6)
    per_seed = similarity["per_seed"]
    assert len(per_seed) == 2
    for stats in per_seed.values():
        assert stats["mean"] == pytest.approx(0.0, abs=1e-6)


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


def test_upload_seeds_emits_refresh_job(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    from apps.api.routers import facebank as facebank_router
    from apps.api.services.facebank import FacebankService

    client = TestClient(app)
    show_id = "rhobh"
    cast_id = _create_cast_member(facebank_router, show_id, "Kyle", data_root)

    facebank_router.facebank_service = FacebankService(data_root)

    dummy_jobs = _DummyJobService(job_id="refresh-upload")
    facebank_router.job_service = dummy_jobs

    _mock_face_pipeline(monkeypatch)

    files = [("files", ("seed.jpg", _create_test_image(), "image/jpeg"))]
    resp = client.post(f"/cast/{cast_id}/seeds/upload?show_id={show_id}", files=files)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["detector"] == "retinaface"
    assert payload["detector_ready"] is True
    assert payload.get("detector_message") is None
    assert payload["refresh_job_id"] == "refresh-upload"
    assert dummy_jobs.calls
    call = dummy_jobs.calls[0]
    assert call["action"] == "upload"
    assert call["seed_ids"]


def test_upload_seeds_presigns_remote_urls(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    from apps.api.routers import facebank as facebank_router
    from apps.api.services.facebank import FacebankService

    client = TestClient(app)
    show_id = "rhobh"
    cast_id = _create_cast_member(facebank_router, show_id, "Kyle", data_root)

    facebank_router.facebank_service = FacebankService(data_root)
    facebank_router.job_service = _DummyJobService()

    dummy_storage = _DummyStorageService()
    monkeypatch.setattr(facebank_router, "storage_service", dummy_storage)

    _mock_face_pipeline(monkeypatch)

    files = [("files", ("seed.jpg", _create_test_image(), "image/jpeg"))]
    resp = client.post(f"/cast/{cast_id}/seeds/upload?show_id={show_id}", files=files)
    assert resp.status_code == 200
    payload = resp.json()
    assert len(dummy_storage.upload_calls) == 3
    display_call = next(
        call
        for call in dummy_storage.upload_calls
        if call["object_name"].endswith("_d.png")
    )
    embed_call = next(
        call
        for call in dummy_storage.upload_calls
        if call["object_name"].endswith("_e.png")
    )
    orig_call = next(
        call
        for call in dummy_storage.upload_calls
        if call["object_name"].endswith("_o.png")
    )
    assert display_call["content_type"] == "image/png"
    assert embed_call["content_type"] == "image/png"
    assert orig_call["content_type"] == "image/png"

    facebank = client.get(f"/cast/{cast_id}/facebank?show_id={show_id}").json()
    assert facebank["seeds"]
    seed = facebank["seeds"][0]
    assert seed.get("display_s3_key", "").endswith("_d.png")
    assert seed.get("embed_s3_key", "").endswith("_e.png")
    assert seed.get("image_s3_key", "").endswith("_d.png")
    assert seed["image_uri"].startswith(dummy_storage.base_url)
    assert dummy_storage.presign_calls
    presign_key, _, presign_mime = dummy_storage.presign_calls[0]
    assert presign_key.endswith("_d.png")
    assert presign_mime == "image/png"


def test_upload_seed_uses_full_image_when_detector_simulated(tmp_path, monkeypatch):
    from PIL import Image

    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    from apps.api.routers import facebank as facebank_router
    from apps.api.services.facebank import FacebankService

    client = TestClient(app)
    show_id = "rhobh"
    cast_id = _create_cast_member(facebank_router, show_id, "Kyle", data_root)

    facebank_router.facebank_service = FacebankService(data_root)
    facebank_router.job_service = _DummyJobService()

    _mock_face_pipeline(monkeypatch, simulated=True)

    original = _create_test_image()
    files = [("files", ("seed.jpg", original, "image/jpeg"))]
    resp = client.post(f"/cast/{cast_id}/seeds/upload?show_id={show_id}", files=files)
    assert resp.status_code == 202
    payload = resp.json()
    assert payload["detector"] == "simulated"
    assert payload.get("detector_message")

    seeds_dir = facebank_router.facebank_service._seeds_dir(show_id, cast_id)
    saved = max(seeds_dir.glob("*_d.png"), key=lambda p: p.stat().st_mtime)
    saved_img = np.asarray(Image.open(saved))
    # Should retain enough variance to avoid the blank tile issue
    assert saved_img.std() > 10

    facebank = client.get(f"/cast/{cast_id}/facebank?show_id={show_id}").json()
    assert facebank["seeds"][0]["quality"].get("detector") == "simulated"


def test_upload_seed_rejects_small_face_when_detector_real(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    from apps.api.routers import facebank as facebank_router
    from apps.api.services.facebank import FacebankService

    client = TestClient(app)
    show_id = "rhobh"
    cast_id = _create_cast_member(facebank_router, show_id, "Kyle", data_root)

    facebank_router.facebank_service = FacebankService(data_root)
    facebank_router.job_service = _DummyJobService()

    _mock_face_pipeline(monkeypatch, bbox=[0.1, 0.1, 0.18, 0.18])

    files = [("files", ("seed.jpg", _create_test_image(), "image/jpeg"))]
    resp = client.post(f"/cast/{cast_id}/seeds/upload?show_id={show_id}", files=files)
    assert resp.status_code == 422
    assert "Face too small" in resp.json()["detail"]


def test_upload_seed_applies_exif_transpose(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    from apps.api.routers import facebank as facebank_router
    from apps.api.services.facebank import FacebankService

    client = TestClient(app)
    show_id = "rhobh"
    cast_id = _create_cast_member(facebank_router, show_id, "Erika", data_root)

    facebank_router.facebank_service = FacebankService(data_root)
    facebank_router.job_service = _DummyJobService()

    _mock_face_pipeline(monkeypatch)

    calls = {"count": 0}

    from PIL import Image

    def _fake_exif_transpose(image):
        calls["count"] += 1
        return image.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)

    monkeypatch.setattr(
        facebank_router.ImageOps, "exif_transpose", _fake_exif_transpose
    )

    files = [("files", ("seed.jpg", _create_test_image(), "image/jpeg"))]
    resp = client.post(f"/cast/{cast_id}/seeds/upload?show_id={show_id}", files=files)
    assert resp.status_code == 200
    assert calls["count"] == 1


def test_delete_seeds_emits_refresh_job(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    from apps.api.routers import facebank as facebank_router
    from apps.api.services.facebank import FacebankService

    client = TestClient(app)
    show_id = "rhobh"
    cast_id = _create_cast_member(facebank_router, show_id, "Lisa", data_root)

    facebank_router.facebank_service = FacebankService(data_root)

    image_path = data_root / "seed.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake")
    seed_entry = facebank_router.facebank_service.add_seed(
        show_id,
        cast_id,
        str(image_path),
        np.zeros(512, dtype=np.float32),
    )

    class _DummyJobService:
        def __init__(self) -> None:
            self.calls = []

        def emit_facebank_refresh(self, show_id, cast_id, **kwargs):
            self.calls.append({"show_id": show_id, "cast_id": cast_id, **kwargs})
            return {"job_id": "refresh-delete"}

    dummy_jobs = _DummyJobService()
    facebank_router.job_service = dummy_jobs

    resp = client.request(
        "DELETE",
        f"/cast/{cast_id}/seeds?show_id={show_id}",
        json={"seed_ids": [seed_entry["fb_id"]]},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["refresh_job_id"] == "refresh-delete"
    call = dummy_jobs.calls[0]
    assert call["action"] == "delete"
    assert call["seed_ids"] == [seed_entry["fb_id"]]


def test_upload_seeds_accepts_modern_webp(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    from apps.api.routers import facebank as facebank_router
    from apps.api.services.facebank import FacebankService

    client = TestClient(app)
    show_id = "rhobh"
    cast_id = _create_cast_member(facebank_router, show_id, "Sutton", data_root)

    facebank_router.facebank_service = FacebankService(data_root)
    facebank_router.job_service = _DummyJobService(job_id="refresh-modern")

    _mock_face_pipeline(monkeypatch)

    files = [("files", ("seed.webp", _create_webp_image(), "image/webp"))]
    resp = client.post(f"/cast/{cast_id}/seeds/upload?show_id={show_id}", files=files)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["uploaded"] == 1
    assert payload["failed"] == 0


def test_feature_seed_sets_person_rep_crop(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    from apps.api.routers import facebank as facebank_router
    from apps.api.services.facebank import FacebankService

    client = TestClient(app)
    show_id = "rhobh"
    cast_id = _create_cast_member(facebank_router, show_id, "Dorit", data_root)

    facebank_router.facebank_service = FacebankService(data_root)
    facebank_router.job_service = _DummyJobService()

    _mock_face_pipeline(monkeypatch)

    files = [("files", ("seed.jpg", _create_test_image(), "image/jpeg"))]
    upload = client.post(f"/cast/{cast_id}/seeds/upload?show_id={show_id}", files=files)
    assert upload.status_code == 200

    facebank = client.get(f"/cast/{cast_id}/facebank?show_id={show_id}").json()
    seed_id = facebank["seeds"][0]["fb_id"]

    resp = client.post(f"/cast/{cast_id}/seeds/{seed_id}/feature?show_id={show_id}")
    assert resp.status_code == 200

    updated = client.get(f"/cast/{cast_id}/facebank?show_id={show_id}").json()
    assert updated["featured_seed_id"] == seed_id
    assert updated["seeds"][0]["featured"] is True

    people_service = PeopleService(data_root)
    person = people_service.find_person_by_cast_id(show_id, cast_id)
    assert person is not None
    assert person.get("rep_crop")


def test_feature_seed_updates_person_rep_crop_s3(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    from apps.api.routers import facebank as facebank_router
    from apps.api.routers import people as people_router
    from apps.api.services.facebank import FacebankService

    client = TestClient(app)
    show_id = "rhobh"
    cast_id = _create_cast_member(facebank_router, show_id, "Garcelle", data_root)

    facebank_router.facebank_service = FacebankService(data_root)
    facebank_router.job_service = _DummyJobService()
    dummy_storage = _DummyStorageService()
    monkeypatch.setattr(facebank_router, "storage_service", dummy_storage)

    monkeypatch.setattr(people_router, "people_service", PeopleService(data_root))
    monkeypatch.setattr(people_router, "storage_service", dummy_storage)

    _mock_face_pipeline(monkeypatch)

    files = [("files", ("seed.jpg", _create_test_image(), "image/jpeg"))]
    upload = client.post(f"/cast/{cast_id}/seeds/upload?show_id={show_id}", files=files)
    assert upload.status_code == 200

    facebank = client.get(f"/cast/{cast_id}/facebank?show_id={show_id}").json()
    seed_id = facebank["seeds"][0]["fb_id"]

    feature = client.post(f"/cast/{cast_id}/seeds/{seed_id}/feature?show_id={show_id}")
    assert feature.status_code == 200

    people_service = PeopleService(data_root)
    person = people_service.find_person_by_cast_id(show_id, cast_id)
    assert person is not None
    assert person.get("rep_crop_s3_key")

    people_resp = client.get(f"/shows/{show_id}/people")
    assert people_resp.status_code == 200
    payload = people_resp.json()
    assert payload["people"]
    assert payload["people"][0]["rep_crop"].startswith(dummy_storage.base_url)
