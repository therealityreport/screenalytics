from fastapi.testclient import TestClient

from apps.api.main import app
from tests.api.test_facebank import (
    _DummyJobService,
    _DummyStorageService,
    _create_cast_member,
    _create_test_image,
    _mock_face_pipeline,
)


def test_seed_upload_response_includes_simulated_detector_flag(tmp_path, monkeypatch):
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

    _mock_face_pipeline(monkeypatch, simulated=True)

    def _fake_ensure_ready(device):
        return False, "weights missing", None

    monkeypatch.setattr(facebank_router.episode_run, "ensure_retinaface_ready", _fake_ensure_ready)

    files = [("files", ("seed.jpg", _create_test_image(), "image/jpeg"))]
    resp = client.post(f"/cast/{cast_id}/seeds/upload?show_id={show_id}", files=files)
    assert resp.status_code == 202
    payload = resp.json()
    assert payload["detector"] == "simulated"
    assert payload["detector_message"]
