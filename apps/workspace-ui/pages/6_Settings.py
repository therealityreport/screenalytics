"""Settings page for configuring storage mode and other runtime options."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402

cfg = helpers.init_page("Settings")

st.title("Settings")

st.markdown("""
Configure runtime settings for the Screenalytics workspace. Changes here are
persisted in your session and apply to all pages.
""")

# -----------------------------------------------------------------------------
# Storage Mode Settings
# -----------------------------------------------------------------------------

st.header("Storage Mode")

st.markdown("""
Storage mode determines where pipeline artifacts (frames, crops, thumbnails,
manifests) are written and read from.
""")

STORAGE_MODES = {
    "s3": {
        "label": "S3 / MinIO (Cloud)",
        "description": "Write directly to S3 or MinIO object storage. Best for production and distributed access.",
        "icon": "cloud",
    },
    "local": {
        "label": "Local Filesystem",
        "description": "Write only to local disk. Fastest for single-machine development, but no cloud sync.",
        "icon": "hdd",
    },
    "hybrid": {
        "label": "Hybrid (Local + S3 Sync)",
        "description": "Write locally first, then sync to S3 in background. Good balance of speed and durability.",
        "icon": "sync",
    },
}

# Current values from session state
current_backend = st.session_state.get("backend", "s3")
current_bucket = st.session_state.get("bucket", "screenalytics")

# Show current mode
col1, col2 = st.columns([2, 1])
with col1:
    mode_info = STORAGE_MODES.get(current_backend, STORAGE_MODES["s3"])
    st.info(f"**Current Mode:** {mode_info['label']}")
    st.caption(mode_info["description"])

with col2:
    st.metric("Bucket", current_bucket)

# Mode selector
st.subheader("Change Storage Mode")

new_mode = st.radio(
    "Select storage mode:",
    options=list(STORAGE_MODES.keys()),
    index=list(STORAGE_MODES.keys()).index(current_backend) if current_backend in STORAGE_MODES else 0,
    format_func=lambda x: STORAGE_MODES[x]["label"],
    horizontal=True,
)

# Bucket configuration (only for S3/hybrid modes)
if new_mode in ("s3", "hybrid"):
    st.subheader("S3 Configuration")

    # Try to get S3 status from API
    try:
        s3_status = helpers.api_get("/storage/status")
        s3_enabled = s3_status.get("s3_enabled", False)
        s3_bucket = s3_status.get("bucket", "")
        s3_region = s3_status.get("region", "")
        s3_endpoint = s3_status.get("endpoint", "")

        if s3_enabled:
            st.success("S3 is enabled and configured")
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Bucket", value=s3_bucket, disabled=True, key="s3_bucket_display")
            with col2:
                if s3_endpoint:
                    st.text_input("Endpoint", value=s3_endpoint, disabled=True, key="s3_endpoint_display")
                elif s3_region:
                    st.text_input("Region", value=s3_region, disabled=True, key="s3_region_display")
        else:
            st.warning("S3 is not enabled. Set AWS credentials or MINIO_* environment variables.")

    except requests.RequestException:
        st.warning("Could not fetch S3 status from API. S3 may not be configured.")

    new_bucket = st.text_input(
        "Override bucket name (session only):",
        value=current_bucket,
        help="Change the bucket name for this session. Does not affect .env configuration.",
    )
else:
    new_bucket = "local"
    st.info("Local mode does not use S3 storage.")

# Apply button
st.divider()

if st.button("Apply Changes", type="primary", use_container_width=True):
    changed = False

    if new_mode != current_backend:
        st.session_state["backend"] = new_mode
        # Also set env var for any subprocess
        os.environ["STORAGE_BACKEND"] = new_mode
        changed = True

    if new_bucket != current_bucket:
        st.session_state["bucket"] = new_bucket
        if new_mode != "local":
            os.environ["AWS_S3_BUCKET"] = new_bucket
        changed = True

    if changed:
        st.success("Settings updated! Changes will apply to new operations.")
        st.rerun()
    else:
        st.info("No changes to apply.")

# -----------------------------------------------------------------------------
# Pipeline Defaults
# -----------------------------------------------------------------------------

st.header("Pipeline Defaults")

st.markdown("Configure default settings for pipeline jobs.")

col1, col2 = st.columns(2)

with col1:
    device_label = st.session_state.get("device_default_label", "auto")
    new_device = st.selectbox(
        "Default device:",
        options=["auto", "cpu", "cuda:0", "mps"],
        index=["auto", "cpu", "cuda:0", "mps"].index(device_label) if device_label in ["auto", "cpu", "cuda:0", "mps"] else 0,
        help="Device to use for ML operations. 'auto' picks GPU if available.",
    )
    if new_device != device_label:
        st.session_state["device_default_label"] = new_device
        st.info("Device preference updated.")

with col2:
    detector_choice = st.session_state.get("detector_choice", "retinaface")
    new_detector = st.selectbox(
        "Default face detector:",
        options=["retinaface", "yolov8n-face", "scrfd"],
        index=["retinaface", "yolov8n-face", "scrfd"].index(detector_choice) if detector_choice in ["retinaface", "yolov8n-face", "scrfd"] else 0,
        help="Face detection model for the pipeline.",
    )
    if new_detector != detector_choice:
        st.session_state["detector_choice"] = new_detector
        st.info("Detector preference updated.")

# Scene detector
scene_detector = st.session_state.get("scene_detector_choice", "content")
new_scene_detector = st.selectbox(
    "Scene detector:",
    options=["content", "adaptive", "threshold", "none"],
    index=["content", "adaptive", "threshold", "none"].index(scene_detector) if scene_detector in ["content", "adaptive", "threshold", "none"] else 0,
    help="Algorithm for detecting scene changes in video.",
)
if new_scene_detector != scene_detector:
    st.session_state["scene_detector_choice"] = new_scene_detector
    st.info("Scene detector preference updated.")

# -----------------------------------------------------------------------------
# Environment Info
# -----------------------------------------------------------------------------

st.header("Environment")

with st.expander("Environment Variables (Read-only)", expanded=False):
    env_vars = [
        "STORAGE_BACKEND",
        "AWS_S3_BUCKET",
        "SCREENALYTICS_OBJECT_STORE_BUCKET",
        "MINIO_ENDPOINT",
        "SCREENALYTICS_API_URL",
        "SCREENALYTICS_DATA_ROOT",
    ]

    for var in env_vars:
        value = os.environ.get(var)
        if value:
            st.code(f"{var}={value}")
        else:
            st.caption(f"{var} (not set)")

# Session state debug
with st.expander("Session State (Debug)", expanded=False):
    debug_keys = ["backend", "bucket", "api_base", "ep_id", "device_default_label", "detector_choice"]
    for key in debug_keys:
        value = st.session_state.get(key, "(not set)")
        st.text(f"{key}: {value}")
