from __future__ import annotations

from py_screenalytics.artifacts import ensure_dirs
from tools.episode_run import _body_tracking_dir_for_run


def test_body_tracking_dir_for_run_is_run_scoped(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "ep123"
    ensure_dirs(ep_id)

    run_id = "runABC"
    expected = data_root / "manifests" / ep_id / "runs" / run_id / "body_tracking"
    assert _body_tracking_dir_for_run(ep_id, run_id) == expected


def test_body_tracking_dir_for_run_falls_back_to_legacy(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("SCREENALYTICS_DATA_ROOT", str(data_root))

    ep_id = "ep123"
    ensure_dirs(ep_id)

    expected = data_root / "manifests" / ep_id / "body_tracking"
    assert _body_tracking_dir_for_run(ep_id, None) == expected

