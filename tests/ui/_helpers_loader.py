from __future__ import annotations

import importlib.util
import sys
import types
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPERS_PATH = PROJECT_ROOT / "apps" / "workspace-ui" / "ui_helpers.py"


def _ensure_streamlit_shims() -> None:
    if "streamlit" not in sys.modules:
        streamlit_mod = types.ModuleType("streamlit")
        streamlit_mod.session_state = {}
        streamlit_mod.sidebar = types.SimpleNamespace(
            header=lambda *args, **kwargs: None,
            code=lambda *args, **kwargs: None,
            success=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
            caption=lambda *args, **kwargs: None,
        )
        streamlit_mod.set_page_config = lambda *args, **kwargs: None
        streamlit_mod.title = lambda *args, **kwargs: None
        streamlit_mod.caption = lambda *args, **kwargs: None
        streamlit_mod.query_params = {}
        streamlit_mod.experimental_set_query_params = lambda **kwargs: None
        streamlit_mod.sidebar = types.SimpleNamespace(
            header=lambda *args, **kwargs: None,
            code=lambda *args, **kwargs: None,
            success=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
            caption=lambda *args, **kwargs: None,
        )
        streamlit_mod.session_state = {}
        streamlit_mod.columns = lambda *args, **kwargs: [types.SimpleNamespace()] * (
            args[0] if args else kwargs.get("spec", 1)
        )
        streamlit_mod.container = lambda *args, **kwargs: types.SimpleNamespace(
            __enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: False
        )
        streamlit_mod.empty = lambda: types.SimpleNamespace(
            write=lambda *a, **k: None,
            caption=lambda *a, **k: None,
            code=lambda *a, **k: None,
            info=lambda *a, **k: None,
        )
        streamlit_mod.progress = lambda *args, **kwargs: types.SimpleNamespace(progress=lambda *a, **k: None)
        streamlit_mod.error = lambda *args, **kwargs: None
        streamlit_mod.success = lambda *args, **kwargs: None
        streamlit_mod.warning = lambda *args, **kwargs: None
        streamlit_mod.info = lambda *args, **kwargs: None
        streamlit_mod.write = lambda *args, **kwargs: None
        streamlit_mod.code = lambda *args, **kwargs: None
        streamlit_mod.button = lambda *args, **kwargs: False
        streamlit_mod.checkbox = lambda *args, **kwargs: False
        streamlit_mod.selectbox = lambda *args, **kwargs: kwargs.get("index", 0)
        streamlit_mod.number_input = lambda *args, **kwargs: kwargs.get("value", 0)
        streamlit_mod.text_input = lambda *args, **kwargs: kwargs.get("value", "")
        streamlit_mod.tabs = lambda labels: [
            types.SimpleNamespace(
                __enter__=lambda self: None,
                __exit__=lambda self, exc_type, exc, tb: False,
            )
            for _ in labels
        ]
        streamlit_mod.stop = lambda: None
        streamlit_mod.rerun = lambda: None
        streamlit_mod.session_state = {}
        sys.modules["streamlit"] = streamlit_mod

    sys.modules.setdefault("streamlit.runtime", types.SimpleNamespace())

    components_pkg = types.ModuleType("streamlit.components")
    sys.modules["streamlit.components"] = components_pkg
    components_v1 = types.ModuleType("streamlit.components.v1")
    components_v1.html = lambda *args, **kwargs: None
    sys.modules["streamlit.components.v1"] = components_v1
    components_pkg.v1 = components_v1


def _ensure_requests_shim() -> None:
    if "requests" not in sys.modules:
        sys.modules["requests"] = types.SimpleNamespace(RequestException=Exception, HTTPError=Exception)


@lru_cache(maxsize=1)
def load_ui_helpers_module():
    _ensure_requests_shim()
    _ensure_streamlit_shims()
    spec = importlib.util.spec_from_file_location("workspace_ui_helpers_test", HELPERS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module
