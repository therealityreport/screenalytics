from __future__ import annotations

import json
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
        self._cache_data_store: dict[tuple[str, str], dict[object, object]] = {}

    def _cache_key(self, args, kwargs):  # noqa: ANN001, D401
        """Best-effort hashable cache key for stubbed st.cache_data."""
        try:
            key = (args, tuple(sorted(kwargs.items())))
            hash(key)
            return key
        except Exception:
            return (repr(args), repr(sorted(kwargs.items(), key=lambda item: str(item[0]))))

    def cache_data(self, *args, **kwargs):  # noqa: ANN002, ANN003
        def deco(fn):  # noqa: ANN001
            cache_key = (fn.__module__, fn.__qualname__)
            cache = self._cache_data_store.setdefault(cache_key, {})

            def wrapper(*f_args, **f_kwargs):  # noqa: ANN001
                key = self._cache_key(f_args, f_kwargs)
                if key in cache:
                    return cache[key]
                value = fn(*f_args, **f_kwargs)
                cache[key] = value
                return value

            def clear():  # noqa: ANN001
                cache.clear()

            wrapper.clear = clear  # type: ignore[attr-defined]
            return wrapper

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

    def metric(self, *args, **kwargs):  # noqa: ANN002, ANN003
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

    def dataframe(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

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

    # Patch `py_screenalytics.artifacts.get_path` so the page doesn't need real media artifacts.
    def _get_path(ep_id_value: str, kind: str) -> Path:
        base = tmp_path / "manifests" / ep_id_value
        base.mkdir(parents=True, exist_ok=True)
        mapping = {
            "video": base / "video.mp4",
            "detections": base / "detections.jsonl",
            "tracks": base / "tracks.jsonl",
        }
        return mapping.get(kind, base / f"{kind}.dat")

    import py_screenalytics.artifacts as artifacts_mod  # noqa: E402

    monkeypatch.setattr(artifacts_mod, "get_path", _get_path, raising=False)

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
        lambda _ep, run_id=None: _api_get(f"/episodes/{_ep}/status"),  # noqa: ARG005
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


def test_episode_detail_faces_completion_promotes_cluster_trigger(tmp_path, monkeypatch) -> None:
    """Regression: faces completion flag must not be dropped when phase already == cluster."""
    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)

    project_root = Path(__file__).resolve().parents[2]
    workspace_dir = project_root / "apps" / "workspace-ui"
    page_path = workspace_dir / "pages" / "2_Episode_Detail.py"

    st_mod = _StreamlitStub()
    ep_id = "show_s01e01"
    st_mod.session_state["ep_id"] = ep_id
    st_mod.session_state[f"{ep_id}::autorun_pipeline"] = True
    st_mod.session_state[f"{ep_id}::autorun_phase"] = "cluster"
    st_mod.session_state[f"{ep_id}::autorun_cluster_trigger"] = False
    st_mod.session_state[f"{ep_id}::faces_embed_just_completed"] = True
    st_mod.session_state[f"{ep_id}::faces_embed_completed_at"] = time.time()
    st_mod.session_state[f"{ep_id}::faces_embed_summary"] = {"status": "completed", "faces": 5}

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

    def _get_path(ep_id_value: str, kind: str) -> Path:
        base = tmp_path / "manifests" / ep_id_value
        base.mkdir(parents=True, exist_ok=True)
        mapping = {
            "video": base / "video.mp4",
            "detections": base / "detections.jsonl",
            "tracks": base / "tracks.jsonl",
        }
        return mapping.get(kind, base / f"{kind}.dat")

    import py_screenalytics.artifacts as artifacts_mod  # noqa: E402

    monkeypatch.setattr(artifacts_mod, "get_path", _get_path, raising=False)

    monkeypatch.syspath_prepend(str(workspace_dir))
    import ui_helpers as helpers  # noqa: E402

    monkeypatch.setattr(helpers, "DATA_ROOT", tmp_path, raising=False)
    (tmp_path / "manifests" / ep_id / "runs").mkdir(parents=True, exist_ok=True)

    # Minimal manifests needed for the page to compute readiness without IO errors.
    (tmp_path / "manifests" / ep_id / "faces.jsonl").write_text("{\"frame_idx\":0}\n" * 5, encoding="utf-8")
    for name in ["detections.jsonl", "tracks.jsonl", "identities.json", "track_metrics.json"]:
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
                "detect_track": {"status": "success"},
                "faces_embed": {"status": "missing"},
                "cluster": {"status": "missing"},
                "tracks_ready": True,
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
        lambda _ep, run_id=None: _api_get(f"/episodes/{_ep}/status"),  # noqa: ARG005
        raising=False,
    )

    original_sys_path = list(sys.path)
    try:
        runpy.run_path(str(page_path), run_name="__main__")
    except _RerunException:
        # The page should rerun after promoting the completion signal.
        pass
    finally:
        sys.path[:] = original_sys_path

    assert st_mod.session_state.get(f"{ep_id}::autorun_phase") == "cluster"
    assert st_mod.session_state.get(f"{ep_id}::autorun_cluster_trigger") is True


def test_episode_detail_autorun_hydrate_clears_presence_cache(tmp_path, monkeypatch) -> None:
    """Regression: hydrate should clear cached presence so auto-run doesn't stop on rerun."""
    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)

    project_root = Path(__file__).resolve().parents[2]
    workspace_dir = project_root / "apps" / "workspace-ui"
    page_path = workspace_dir / "pages" / "2_Episode_Detail.py"

    st_mod = _StreamlitStub()
    ep_id = "show_s01e01"
    run_id = "Attempt1_2024-01-01_000000EST"
    st_mod.session_state["ep_id"] = ep_id
    st_mod.session_state[f"{ep_id}::active_run_id"] = run_id
    st_mod.session_state[f"{ep_id}::autorun_pipeline"] = True
    st_mod.session_state[f"{ep_id}::autorun_phase"] = "track_fusion"

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

    def _get_path(ep_id_value: str, kind: str) -> Path:
        base = tmp_path / "manifests" / ep_id_value
        base.mkdir(parents=True, exist_ok=True)
        mapping = {
            "video": base / "video.mp4",
            "detections": base / "detections.jsonl",
            "tracks": base / "tracks.jsonl",
            "frames_root": base / "frames",
        }
        return mapping.get(kind, base / f"{kind}.dat")

    import py_screenalytics.artifacts as artifacts_mod  # noqa: E402

    monkeypatch.setattr(artifacts_mod, "get_path", _get_path, raising=False)

    monkeypatch.syspath_prepend(str(workspace_dir))
    import ui_helpers as helpers  # noqa: E402

    monkeypatch.setattr(helpers, "DATA_ROOT", tmp_path, raising=False)
    run_root = tmp_path / "manifests" / ep_id / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (tmp_path / "manifests" / ep_id / "video.mp4").write_text("", encoding="utf-8")
    (run_root / "detections.jsonl").write_text("", encoding="utf-8")
    (run_root / "tracks.jsonl").write_text("", encoding="utf-8")

    class _DummyClient:
        def download_file(self, bucket, key, filename):  # noqa: ANN001
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    class _DummyStorage:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.bucket = "stub-bucket"
            self._client = _DummyClient()

        def s3_enabled(self) -> bool:
            return True

        def object_exists(self, key):  # noqa: ANN001
            return True

    import apps.api.services.storage as storage_mod  # noqa: E402

    monkeypatch.setattr(storage_mod, "StorageService", _DummyStorage, raising=False)

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
                "detect_track": {"status": "success"},
                "faces_embed": {"status": "success"},
                "cluster": {"status": "success"},
                "tracks_ready": True,
            }
        if path == f"/episodes/{ep_id}/video_meta":
            return {}
        if path.startswith("/jobs"):
            return {"jobs": []}
        if path.startswith("/celery_jobs"):
            return {"jobs": []}
        return {}

    monkeypatch.setattr(helpers, "api_get", _api_get, raising=False)
    monkeypatch.setattr(
        helpers,
        "get_episode_status",
        lambda _ep, run_id=None: _api_get(f"/episodes/{_ep}/status"),  # noqa: ARG005
        raising=False,
    )

    original_sys_path = list(sys.path)
    try:
        for _ in range(2):
            try:
                runpy.run_path(str(page_path), run_name="__main__")
            except (_RerunException, _StopException):
                pass
    finally:
        sys.path[:] = original_sys_path

    assert st_mod.session_state.get(f"{ep_id}::autorun_error") != "local_artifacts_missing_after_hydrate"


def test_episode_detail_autorun_track_fusion_legacy_marker_advances(tmp_path, monkeypatch) -> None:
    """Regression: legacy track fusion markers should allow auto-run to advance."""
    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)

    project_root = Path(__file__).resolve().parents[2]
    workspace_dir = project_root / "apps" / "workspace-ui"
    page_path = workspace_dir / "pages" / "2_Episode_Detail.py"

    st_mod = _StreamlitStub()
    ep_id = "show_s01e01"
    run_id = "Attempt1_2024-01-01_000000EST"
    st_mod.session_state["ep_id"] = ep_id
    st_mod.session_state[f"{ep_id}::active_run_id"] = run_id
    st_mod.session_state[f"{ep_id}::autorun_pipeline"] = True
    st_mod.session_state[f"{ep_id}::autorun_phase"] = "track_fusion"

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

    def _get_path(ep_id_value: str, kind: str) -> Path:
        base = tmp_path / "manifests" / ep_id_value
        base.mkdir(parents=True, exist_ok=True)
        mapping = {
            "video": base / "video.mp4",
            "detections": base / "detections.jsonl",
            "tracks": base / "tracks.jsonl",
        }
        return mapping.get(kind, base / f"{kind}.dat")

    import py_screenalytics.artifacts as artifacts_mod  # noqa: E402

    monkeypatch.setattr(artifacts_mod, "get_path", _get_path, raising=False)

    monkeypatch.syspath_prepend(str(workspace_dir))
    import ui_helpers as helpers  # noqa: E402

    monkeypatch.setattr(helpers, "DATA_ROOT", tmp_path, raising=False)
    manifests_root = tmp_path / "manifests" / ep_id
    run_root = manifests_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (manifests_root / "runs").mkdir(parents=True, exist_ok=True)
    (manifests_root / "body_tracking").mkdir(parents=True, exist_ok=True)
    for name in ["detections.jsonl", "tracks.jsonl", "faces.jsonl", "identities.json", "track_metrics.json"]:
        (manifests_root / name).write_text("", encoding="utf-8")

    # Legacy marker + artifacts for track fusion (run-scoped marker missing on purpose).
    (manifests_root / "runs" / "body_tracking_fusion.json").write_text(
        json.dumps(
            {
                "phase": "body_tracking_fusion",
                "status": "success",
                "run_id": run_id,
                "started_at": "2024-01-01T00:00:00Z",
                "finished_at": "2024-01-01T00:01:00Z",
            }
        ),
        encoding="utf-8",
    )
    (manifests_root / "body_tracking" / "track_fusion.json").write_text("{}", encoding="utf-8")

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
                "detect_track": {"status": "success"},
                "faces_embed": {"status": "success"},
                "cluster": {"status": "success"},
                "tracks_ready": True,
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
        lambda _ep, run_id=None: _api_get(f"/episodes/{_ep}/status"),  # noqa: ARG005
        raising=False,
    )

    original_sys_path = list(sys.path)
    try:
        runpy.run_path(str(page_path), run_name="__main__")
    except _RerunException:
        pass
    finally:
        sys.path[:] = original_sys_path

    assert st_mod.session_state.get(f"{ep_id}::autorun_phase") == "pdf"
