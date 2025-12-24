from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from py_screenalytics import run_layout


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "apps" / "workspace-ui" / "faces_review_artifacts.py"
    spec = importlib.util.spec_from_file_location("faces_review_artifacts", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load faces_review_artifacts module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def test_resolve_run_artifact_paths(tmp_path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))

    ep_id = "ep-test"
    run_id = "Attempt1_2025-12-24_000000EST"
    paths = module.resolve_run_artifact_paths(ep_id, run_id, ["identities.json"])

    expected = run_layout.run_root(ep_id, run_id) / "identities.json"
    assert paths["identities.json"] == expected


def test_ensure_run_artifacts_local_hydrates_required(tmp_path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(tmp_path))

    ep_id = "ep-test"
    run_id = "Attempt1_2025-12-24_000000EST"

    def _fake_keys(_ep_id: str, _run_id: str, rel_path: str) -> list[str]:
        return [f"runs/{_ep_id}/{_run_id}/{rel_path}"]

    monkeypatch.setattr(module.run_layout, "run_artifact_s3_keys_for_read", _fake_keys)

    class FakeStorage:
        def __init__(self) -> None:
            self.downloaded: list[str] = []

        def s3_enabled(self) -> bool:
            return True

        def object_exists(self, key: str) -> bool:
            return True

        def download_bytes(self, key: str) -> bytes | None:
            self.downloaded.append(key)
            return b"{}"

    storage = FakeStorage()
    result = module.ensure_run_artifacts_local(
        ep_id,
        run_id,
        ["identities.json"],
        storage=storage,
    )

    assert not result.missing_required
    assert result.hydrated == ["identities.json"]
    expected_path = run_layout.run_root(ep_id, run_id) / "identities.json"
    assert expected_path.exists()
