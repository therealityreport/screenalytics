from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from apps.api.main import app
from apps.api.services.facebank import FacebankService

try:
    from tests.api import test_facebank as _facebank_tests  # noqa: F401
except pytest.skip.Exception as exc:  # pragma: no cover
    pytest.skip(str(exc), allow_module_level=True)

from tests.api.test_facebank import (  # noqa: E402
    _DummyJobService,
    _DummyStorageService,
    _create_cast_member,
    _create_test_image,
    _mock_face_pipeline,
)


def _prepare_facebank_env(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    from apps.api.routers import facebank as facebank_router

    facebank_router.facebank_service = FacebankService(data_root)
    facebank_router.job_service = _DummyJobService()
    dummy_storage = _DummyStorageService()
    monkeypatch.setattr(facebank_router, "storage_service", dummy_storage)
    return facebank_router, dummy_storage


def _upload_seed(monkeypatch, facebank_router, show_id: str, cast_id: str, image, filename: str = "seed.jpg"):
    _mock_face_pipeline(monkeypatch)
    client = TestClient(app)
    files = [("files", (filename, image, "image/jpeg"))]
    resp = client.post(f"/cast/{cast_id}/seeds/upload?show_id={show_id}", files=files)
    assert resp.status_code == 200
    return resp.json()


def _latest_display_path(facebank_router, show_id: str, cast_id: str):
    seeds_dir = facebank_router.facebank_service._seeds_dir(show_id, cast_id)
    return max(seeds_dir.glob("*_d.*"), key=lambda p: p.stat().st_mtime)


def test_seed_derivative_retains_large_resolution(monkeypatch, tmp_path):
    facebank_router, _ = _prepare_facebank_env(monkeypatch, tmp_path)
    show_id = "rhobh"
    cast_id = _create_cast_member(facebank_router, show_id, "Kyle", tmp_path / "data")

    payload = _upload_seed(
        monkeypatch,
        facebank_router,
        show_id,
        cast_id,
        _create_test_image(size=(900, 600)),
        filename="wide.jpg",
    )
    assert payload["uploaded"] == 1
    display_path = _latest_display_path(facebank_router, show_id, cast_id)
    with Image.open(display_path) as img:
        size = img.size
    assert size == (900, 600)

    facebank = facebank_router.facebank_service.get_facebank(show_id, cast_id)
    seed = facebank["seeds"][0]
    assert seed["display_dims"] == [900, 600]
    assert not seed.get("display_low_res")


def test_seed_tiny_no_upscale(monkeypatch, tmp_path):
    facebank_router, _ = _prepare_facebank_env(monkeypatch, tmp_path)
    show_id = "rhobh"
    cast_id = _create_cast_member(facebank_router, show_id, "Lisa", tmp_path / "data")

    payload = _upload_seed(
        monkeypatch,
        facebank_router,
        show_id,
        cast_id,
        _create_test_image(size=(120, 120)),
        filename="tiny.jpg",
    )
    assert payload["uploaded"] == 1
    display_path = _latest_display_path(facebank_router, show_id, cast_id)
    with Image.open(display_path) as img:
        size = img.size
    assert size == (120, 120)

    facebank = facebank_router.facebank_service.get_facebank(show_id, cast_id)
    seed = facebank["seeds"][0]
    assert seed["display_dims"] == [120, 120]
    assert seed.get("display_low_res") is True
