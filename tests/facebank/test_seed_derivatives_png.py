from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.services.facebank import FacebankService

try:  # Ensure we honor the numpy skip used by the shared facebank helpers
    from tests.api import test_facebank as _facebank_tests  # noqa: F401
except (
    pytest.skip.Exception
) as exc:  # pragma: no cover - mirrors upstream skip handling
    pytest.skip(str(exc), allow_module_level=True)

from tests.api.test_facebank import (  # noqa: E402
    _DummyJobService,
    _DummyStorageService,
    _create_cast_member,
    _create_test_image,
    _mock_face_pipeline,
)


def test_seed_upload_writes_png_derivatives(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("STORAGE_BACKEND", "local")

    from apps.api.routers import facebank as facebank_router

    show_id = "rhobh"
    cast_id = _create_cast_member(facebank_router, show_id, "Kyle", data_root)

    facebank_router.facebank_service = FacebankService(data_root)
    facebank_router.job_service = _DummyJobService()
    dummy_storage = _DummyStorageService()
    monkeypatch.setattr(facebank_router, "storage_service", dummy_storage)

    _mock_face_pipeline(monkeypatch)

    client = TestClient(app)
    files = [("files", ("seed.jpg", _create_test_image(size=(640, 640)), "image/jpeg"))]
    resp = client.post(f"/cast/{cast_id}/seeds/upload?show_id={show_id}", files=files)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["uploaded"] == 1
    uploaded = payload["seeds"][0]
    assert uploaded["display_key"].endswith("_d.png")
    assert uploaded["orig_key"].endswith("_o.png")
    assert all(call[2] == "image/png" for call in dummy_storage.presign_calls)
    assert len(dummy_storage.upload_calls) == 3

    facebank = facebank_router.facebank_service.get_facebank(show_id, cast_id)
    assert facebank["seeds"]
    seed = facebank["seeds"][0]
    assert seed["display_s3_key"].endswith("_d.png")
    assert seed["embed_s3_key"].endswith("_e.png")
    assert seed["orig_s3_key"].endswith("_o.png")
    assert seed["display_dims"]
    assert not seed.get("display_low_res")
