from __future__ import annotations

import sys
import types

from fastapi.testclient import TestClient

from apps.api.main import app


def test_run_stage_job_idempotent(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SCREENALYTICS_FAKE_DB", "1")
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "redis",
        types.SimpleNamespace(Redis=object, from_url=lambda *args, **kwargs: None),
    )
    class _DummyConf(dict):
        def __getattr__(self, name: str) -> object:  # noqa: D401
            return self.get(name)

        def __setattr__(self, name: str, value: object) -> None:
            self[name] = value

    class _DummyCelery:
        def __init__(self, *_args, **_kwargs) -> None:
            self.conf = _DummyConf()

        def autodiscover_tasks(self, *_args, **_kwargs) -> None:
            return None

        def task(self, *args, **kwargs):
            def _decorator(func):
                if not hasattr(func, "delay"):
                    func.delay = lambda *d_args, **d_kwargs: None  # type: ignore[attr-defined]
                return func
            return _decorator

    celery_stub = types.SimpleNamespace(
        Task=object,
        states=types.SimpleNamespace(PROGRESS="PROGRESS"),
        group=lambda *args, **kwargs: None,
        chord=lambda *args, **kwargs: None,
        Celery=_DummyCelery,
    )
    celery_result_stub = types.SimpleNamespace(AsyncResult=object, GroupResult=object)
    kombu_stub = types.SimpleNamespace(Queue=lambda *args, **kwargs: object())
    monkeypatch.setitem(sys.modules, "celery", celery_stub)
    monkeypatch.setitem(sys.modules, "celery.result", celery_result_stub)
    monkeypatch.setitem(sys.modules, "kombu", kombu_stub)

    calls: list[tuple[str, dict]] = []

    class _Result:
        def __init__(self, job_id: str) -> None:
            self.id = job_id

    def _fake_delay(ep_id: str, options: dict) -> _Result:
        calls.append((ep_id, options))
        return _Result(f"job-{len(calls)}")

    monkeypatch.setattr("apps.api.tasks.run_detect_track_task.delay", _fake_delay)

    client = TestClient(app)
    ep_id = "demo-s01e01"
    run_id = "Attempt1_2025-01-01_000000EST"

    resp1 = client.post(
        f"/episodes/{ep_id}/runs/{run_id}/jobs/detect_track",
        json={"params": {"stride": 5}},
    )
    assert resp1.status_code == 200
    body1 = resp1.json()
    assert body1.get("status") == "queued"
    assert body1.get("job_id") == "job-1"

    resp2 = client.post(
        f"/episodes/{ep_id}/runs/{run_id}/jobs/detect_track",
        json={"params": {"stride": 5}},
    )
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2.get("status") == "existing"
    assert len(calls) == 1
