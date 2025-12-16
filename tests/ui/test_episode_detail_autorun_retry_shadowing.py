from __future__ import annotations

import runpy
import sys
import time
import types
from pathlib import Path


class _DummyCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyColumn(_DummyCtx):
    def button(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return False

    def code(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None


class _RerunException(BaseException):
    pass


class _StopException(BaseException):
    pass


class _StreamlitStub(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.session_state: dict = {}
        self.query_params: dict = {}
        self.errors = types.SimpleNamespace(StreamlitAPIException=Exception)

    def cache_data(self, *args, **kwargs):  # noqa: ANN002, ANN003
        def deco(fn):  # noqa: ANN001
            fn.clear = lambda: None  # type: ignore[attr-defined]
            return fn

        return deco

    def dialog(self, *args, **kwargs):  # noqa: ANN002, ANN003
        def deco(fn):  # noqa: ANN001
            return fn

        return deco

    def set_page_config(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def title(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def markdown(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def subheader(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def caption(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def divider(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def progress(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def toast(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def info(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def warning(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def error(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def success(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def write(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def code(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def expander(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return _DummyCtx()

    def container(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return _DummyCtx()

    def spinner(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return _DummyCtx()

    def columns(self, spec, *args, **kwargs):  # noqa: ANN002, ANN003
        count = spec if isinstance(spec, int) else len(spec)
        return tuple(_DummyColumn() for _ in range(count))

    def tabs(self, labels, *args, **kwargs):  # noqa: ANN002, ANN003
        return [_DummyCtx() for _ in labels]

    def selectbox(self, label, options, index=0, key=None, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if key is not None and key in self.session_state:
            return self.session_state[key]
        return options[index] if options else None

    def number_input(self, *args, key=None, value=None, **kwargs):  # noqa: ANN002, ANN003
        if key is not None and key in self.session_state:
            return self.session_state[key]
        return value if value is not None else kwargs.get("min_value", 0)

    def slider(self, *args, key=None, value=None, **kwargs):  # noqa: ANN002, ANN003
        if key is not None and key in self.session_state:
            return self.session_state[key]
        return value if value is not None else kwargs.get("min_value", 0)

    def checkbox(self, *args, key=None, value=False, **kwargs):  # noqa: ANN002, ANN003
        if key is not None and key in self.session_state:
            return bool(self.session_state[key])
        return bool(value)

    def button(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return False

    def stop(self):  # noqa: D401
        raise _StopException()

    def rerun(self):  # noqa: D401
        raise _RerunException()


def test_episode_detail_autorun_retry_does_not_shadow_manifest_helper(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)

    project_root = Path(__file__).resolve().parents[2]
    workspace_dir = project_root / "apps" / "workspace-ui"
    page_path = workspace_dir / "pages" / "2_Episode_Detail.py"

    st_mod = _StreamlitStub()
    ep_id = "show_s01e01"
    st_mod.session_state["ep_id"] = ep_id
    st_mod.session_state[f"{ep_id}::autorun_pipeline"] = True
    st_mod.session_state[f"{ep_id}::autorun_faces_trigger"] = True
    st_mod.session_state[f"{ep_id}::autorun_faces_retry"] = 30

    monkeypatch.setitem(sys.modules, "streamlit", st_mod)
    monkeypatch.setitem(sys.modules, "streamlit.runtime", types.ModuleType("streamlit.runtime"))

    scriptrunner_mod = types.ModuleType("streamlit.runtime.scriptrunner")
    scriptrunner_mod.RerunException = _RerunException  # type: ignore[attr-defined]
    scriptrunner_mod.StopException = _StopException  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "streamlit.runtime.scriptrunner", scriptrunner_mod)

    components_pkg = types.ModuleType("streamlit.components")
    components_v1 = types.ModuleType("streamlit.components.v1")
    components_v1.html = lambda *args, **kwargs: None  # noqa: ANN002, ANN003
    components_pkg.v1 = components_v1  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "streamlit.components", components_pkg)
    monkeypatch.setitem(sys.modules, "streamlit.components.v1", components_v1)

    # Stub `py_screenalytics.artifacts.get_path` so the page doesn't need real media artifacts.
    py_screenalytics_pkg = types.ModuleType("py_screenalytics")
    artifacts_mod = types.ModuleType("py_screenalytics.artifacts")
    run_layout_mod = types.ModuleType("py_screenalytics.run_layout")

    def _get_path(ep_id_value: str, kind: str) -> Path:
        base = tmp_path / "manifests" / ep_id_value
        base.mkdir(parents=True, exist_ok=True)
        mapping = {
            "video": base / "video.mp4",
            "detections": base / "detections.jsonl",
            "tracks": base / "tracks.jsonl",
        }
        return mapping.get(kind, base / f"{kind}.dat")

    artifacts_mod.get_path = _get_path  # type: ignore[attr-defined]
    py_screenalytics_pkg.artifacts = artifacts_mod  # type: ignore[attr-defined]

    def _read_active_run_id(_ep_id: str):  # noqa: ANN001
        return None

    def _normalize_run_id(run_id_value: str) -> str:
        return run_id_value.strip()

    def _run_root(ep_id_value: str, run_id_value: str) -> Path:
        return tmp_path / "manifests" / ep_id_value / "runs" / run_id_value

    def _list_run_ids(_ep_id: str):  # noqa: ANN001
        return []

    run_layout_mod.read_active_run_id = _read_active_run_id  # type: ignore[attr-defined]
    run_layout_mod.normalize_run_id = _normalize_run_id  # type: ignore[attr-defined]
    run_layout_mod.run_root = _run_root  # type: ignore[attr-defined]
    run_layout_mod.list_run_ids = _list_run_ids  # type: ignore[attr-defined]
    py_screenalytics_pkg.run_layout = run_layout_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "py_screenalytics", py_screenalytics_pkg)
    monkeypatch.setitem(sys.modules, "py_screenalytics.artifacts", artifacts_mod)
    monkeypatch.setitem(sys.modules, "py_screenalytics.run_layout", run_layout_mod)

    monkeypatch.syspath_prepend(str(workspace_dir))
    import ui_helpers as helpers  # noqa: E402

    # Make the page fully offline / fast.
    monkeypatch.setattr(helpers, "DATA_ROOT", tmp_path, raising=False)
    (tmp_path / "manifests" / ep_id / "runs").mkdir(parents=True, exist_ok=True)
    for name in ["detections.jsonl", "tracks.jsonl", "faces.jsonl", "identities.json", "track_metrics.json"]:
        (tmp_path / "manifests" / ep_id / name).write_text("", encoding="utf-8")

    monkeypatch.setattr(
        helpers,
        "init_page",
        lambda title="Episode Detail": {"api_base": "http://stub", "backend": "local", "bucket": None, "ep_id": ep_id},
        raising=False,
    )
    monkeypatch.setattr(helpers, "inject_log_container_css", lambda: None, raising=False)
    monkeypatch.setattr(helpers, "hydrate_logs_for_episode", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(helpers, "invalidate_running_jobs_cache", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(helpers, "render_previous_logs", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(helpers, "pipeline_combo_supported", lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(helpers, "detector_is_face_only", lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(helpers, "get_execution_mode", lambda *_a, **_k: "local", raising=False)
    monkeypatch.setattr(helpers, "render_execution_mode_selector", lambda *_a, **_k: "local", raising=False)
    monkeypatch.setattr(helpers, "get_running_job_for_episode", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(helpers, "get_all_running_jobs_for_episode", lambda *_a, **_k: {}, raising=False)
    monkeypatch.setattr(helpers, "get_episode_progress", lambda *_a, **_k: {}, raising=False)
    monkeypatch.setattr(helpers, "api_post", lambda *_a, **_k: {}, raising=False)

    def _api_get(path: str, params=None, timeout=None):  # noqa: ANN001
        if path == "/config/storage":
            return {"status": "success", "backend_type": "local", "validation": {"warnings": []}}
        if path == f"/episodes/{ep_id}":
            return {
                "ep_id": ep_id,
                "show_slug": "show",
                "season_number": 1,
                "episode_number": 1,
                "s3": {"v2_key": None, "v2_exists": False, "v1_key": None, "v1_exists": False},
                "local": {"path": str(tmp_path / "video.mp4"), "exists": True},
            }
        if path == f"/episodes/{ep_id}/status":
            return {
                "detect_track": {"status": "missing"},
                "faces_embed": {"status": "missing"},
                "cluster": {"status": "missing"},
                "tracks_ready": False,
            }
        if path.startswith("/jobs"):
            return {"jobs": []}
        if path.startswith("/celery_jobs"):
            return {"jobs": []}
        return {}

    monkeypatch.setattr(helpers, "api_get", _api_get, raising=False)
    monkeypatch.setattr(
        helpers,
        "get_episode_status",
        lambda _ep, *, run_id=None: _api_get(f"/episodes/{_ep}/status"),
        raising=False,
    )

    original_sys_path = list(sys.path)
    try:
        runpy.run_path(str(page_path), run_name="__main__")
    except (_RerunException, _StopException):
        # Control-flow exceptions are expected in Streamlit apps; this test only
        # guards against crashes from helper shadowing in the auto-run retry path.
        pass
    finally:
        sys.path[:] = original_sys_path
