"""Workspace Upload page wrapper for sidebar navigation."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure workspace modules are importable when this page is loaded directly
PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

# When opened via sidebar, always start in NEW EPISODE mode (no ep_id).
params = st.query_params
if "ep_id" in params:
    params.pop("ep_id", None)
    st.query_params = params
# Clear any lingering episode state so the upload page does not reuse it.
st.session_state.pop("ep_id", None)
st.session_state.pop("_ep_id_query_origin", None)
st.session_state.pop("upload_ep_params_cleaned", None)

# Import the canonical upload page (top-level Upload_Video.py).
# All Streamlit rendering happens inside that module.
import Upload_Video  # noqa: F401  # type: ignore

# Run the upload page.
Upload_Video.main()
