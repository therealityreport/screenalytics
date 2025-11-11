from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPERS_PATH = PROJECT_ROOT / "apps" / "workspace-ui" / "ui_helpers.py"


def _load_helpers():
    sys.modules.setdefault("requests", types.SimpleNamespace(RequestException=Exception, HTTPError=Exception))

    streamlit_mod = types.ModuleType("streamlit")
    streamlit_mod.session_state = {}
    sys.modules["streamlit"] = streamlit_mod

    components_pkg = types.ModuleType("streamlit.components")
    sys.modules["streamlit.components"] = components_pkg
    streamlit_mod.components = components_pkg
    components_v1 = types.ModuleType("streamlit.components.v1")
    components_v1.html = lambda *args, **kwargs: None
    sys.modules["streamlit.components.v1"] = components_v1
    components_pkg.v1 = components_v1

    spec = importlib.util.spec_from_file_location("workspace_ui_helpers_media", HELPERS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_ensure_media_url_local_file(tmp_path):
    helpers = _load_helpers()
    sample = tmp_path / "frame.jpg"
    sample.write_bytes(b"\xff\xd8\xff\xdbtestjpegbytes")
    data_url = helpers.ensure_media_url(sample)
    assert isinstance(data_url, str)
    assert data_url.startswith("data:image/jpeg;base64,")
    https_url = "https://example.com/frame.jpg"
    assert helpers.ensure_media_url(https_url) == https_url
    assert helpers.ensure_media_url(None) is None
