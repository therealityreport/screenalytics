from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple
from zoneinfo import ZoneInfo

import requests
import streamlit as st
from streamlit.runtime.scriptrunner import RerunException, StopException

LOGGER = logging.getLogger("episode_detail")

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402

from py_screenalytics.artifacts import get_path  # noqa: E402
from py_screenalytics import run_layout  # noqa: E402

FRAME_JPEG_SIZE_EST_BYTES = 220_000
CROP_JPEG_SIZE_EST_BYTES = 40_000
AVG_FACES_PER_FRAME = 1.5
JPEG_DEFAULT = int(os.environ.get("SCREENALYTICS_JPEG_QUALITY", "72"))
MIN_FRAMES_BETWEEN_CROPS_DEFAULT = int(os.environ.get("SCREENALYTICS_MIN_FRAMES_BETWEEN_CROPS", "32"))
EST_TZ = ZoneInfo("America/New_York")

# =============================================================================
# Pipeline Defaults (from config/pipeline/*.yaml)
# These are the canonical defaults used when no prior run exists.
# Session-only: changes reset on app restart.
# =============================================================================

# Detection defaults (config/pipeline/detection.yaml)
DETECT_CONFIDENCE_DEFAULT = 0.50  # detection.yaml: confidence_th
DETECT_MIN_SIZE_DEFAULT = 16  # detection.yaml: min_size
DETECT_IOU_DEFAULT = 0.5  # detection.yaml: iou_th

# Tracking defaults (config/pipeline/tracking.yaml)
TRACK_THRESH_DEFAULT = 0.55  # tracking.yaml: track_thresh
TRACK_MATCH_THRESH_DEFAULT = 0.65  # tracking.yaml: match_thresh
TRACK_NEW_THRESH_DEFAULT = 0.60  # tracking.yaml: new_track_thresh
TRACK_BUFFER_DEFAULT = 90  # tracking.yaml: track_buffer

# Clustering defaults (config/pipeline/clustering.yaml)
CLUSTER_THRESH_DEFAULT = 0.52  # clustering.yaml: cluster_thresh
CLUSTER_MIN_SIZE_DEFAULT = 1  # clustering.yaml: min_cluster_size
CLUSTER_MIN_IDENTITY_SIM_DEFAULT = 0.45  # clustering.yaml: min_identity_sim

# Faces harvest defaults
FACES_THUMB_SIZE_DEFAULT = 256


def _default_faces_min_frames_between_crops(video_meta: Dict[str, Any] | None) -> int:
    """Default crop sampling interval for Faces Harvest.

    Target: ~1/2 the detected FPS, rounded to an even integer.
    Examples:
      - 23.98 fps → 12
      - 29.97/30 fps → 14
    """
    meta = video_meta or {}
    fps = helpers.coerce_float(meta.get("fps_detected")) or helpers.coerce_float(meta.get("fps"))
    if fps is None or fps <= 0:
        return max(1, int(MIN_FRAMES_BETWEEN_CROPS_DEFAULT))

    interval = int(round(fps / 2.0))
    if interval < 1:
        interval = 1
    # Prefer even intervals (e.g., 30fps -> 14 instead of 15).
    if interval % 2 == 1 and interval > 1:
        interval -= 1
    return max(1, min(interval, 600))


def _get_pipeline_settings_key(ep_id: str, category: str, field: str) -> str:
    """Generate a session state key for pipeline settings."""
    return f"pipeline_settings::{ep_id}::{category}::{field}"


def _generate_attempt_run_id(ep_id: str) -> str:
    """Generate a human-readable run_id for an episode attempt.

    Backward-compatible: if the currently loaded `py_screenalytics.run_layout` module
    predates `generate_attempt_run_id`, compute it here so Streamlit hot reloads
    don't crash on a stale import.
    """
    generator = getattr(run_layout, "generate_attempt_run_id", None)
    if callable(generator):
        return str(generator(ep_id))

    existing = run_layout.list_run_ids(ep_id)
    max_attempt = 0
    for candidate in existing:
        if not candidate.startswith("Attempt"):
            continue
        suffix = candidate[len("Attempt") :]
        num_str = suffix.split("_", 1)[0]
        try:
            max_attempt = max(max_attempt, int(num_str))
        except (TypeError, ValueError):
            continue

    attempt_num = (max_attempt + 1) if max_attempt > 0 else (len(existing) + 1)
    timestamp = datetime.now(EST_TZ).strftime("%Y-%m-%d_%H%M%S")
    run_id = f"Attempt{attempt_num}_{timestamp}EST"
    while run_id in existing:
        attempt_num += 1
        run_id = f"Attempt{attempt_num}_{timestamp}EST"
    return run_layout.normalize_run_id(run_id)


def _trigger_pdf_export_if_needed(
    ep_id: str,
    run_id: str,
    cfg: Dict[str, Any],
    *,
    force: bool = False,
) -> Tuple[bool, str]:
    """Trigger PDF export if not already exported for this run.

    Idempotency guard: checks export_index.json to avoid re-exporting.

    Args:
        ep_id: Episode ID.
        run_id: Run ID.
        cfg: Config dict with api_base.
        force: If True, skip idempotency check and always export.

    Returns:
        Tuple of (success, message).
    """
    from apps.api.services.run_artifact_store import read_export_index

    # Idempotency check: skip if PDF already exported for this run
    if not force:
        try:
            export_index = read_export_index(ep_id, run_id)
            if export_index:
                export_type = export_index.get("export_type")
                export_upload = export_index.get("export_upload", {})
                if export_type == "pdf" and export_upload.get("success"):
                    s3_key = export_upload.get("s3_key") or export_index.get("export_s3_key")
                    LOGGER.info(
                        "[PDF-EXPORT] Skipping export - PDF already exists for %s/%s (key=%s)",
                        ep_id, run_id, s3_key,
                    )
                    return True, f"PDF already exported: {s3_key or 'local'}"
        except Exception as exc:
            LOGGER.warning("[PDF-EXPORT] Failed to check export index: %s", exc)

    # Trigger PDF export via API
    export_url = f"{cfg['api_base']}/episodes/{ep_id}/runs/{run_id}/export"
    LOGGER.info("[PDF-EXPORT] Triggering export for %s/%s via %s", ep_id, run_id, export_url)

    try:
        resp = requests.get(export_url, timeout=300)
        resp.raise_for_status()
    except requests.RequestException as exc:
        error_msg = helpers.describe_error(export_url, exc)
        LOGGER.error("[PDF-EXPORT] Export failed: %s", error_msg)
        return False, f"Export failed: {error_msg}"

    # Extract S3 upload status from response headers
    s3_attempted = resp.headers.get("X-S3-Upload-Attempted", "").lower() == "true"
    s3_success = resp.headers.get("X-S3-Upload-Success", "").lower() == "true"
    s3_key = resp.headers.get("X-S3-Upload-Key", "")
    s3_error = resp.headers.get("X-S3-Upload-Error", "")

    if s3_attempted and s3_success and s3_key:
        LOGGER.info("[PDF-EXPORT] Export successful - uploaded to S3: %s", s3_key)
        return True, f"PDF exported to S3: {s3_key}"
    elif s3_attempted and s3_error:
        LOGGER.warning("[PDF-EXPORT] Export generated but S3 upload failed: %s", s3_error)
        return True, f"PDF generated (S3 upload failed: {s3_error})"
    else:
        LOGGER.info("[PDF-EXPORT] Export generated (local storage)")
        return True, "PDF exported (local storage)"


def _render_pipeline_settings_dialog(ep_id: str, video_meta: Dict[str, Any] | None) -> None:
    """Render the unified pipeline settings dialog with all phase settings."""

    # Keys for settings dialog state
    dialog_key = f"{ep_id}::pipeline_settings_dialog"

    @st.dialog("Pipeline Settings", width="large")
    def settings_dialog():
        st.markdown("Configure settings for all pipeline phases. Changes apply to this session only.")

        # Create tabs for each phase
        tab_detect, tab_harvest, tab_cluster = st.tabs([
            "Detect/Track",
            "Faces Harvest",
            "Cluster Identities"
        ])

        # ─── DETECT/TRACK SETTINGS ───────────────────────────────────────────
        with tab_detect:
            st.markdown("#### Detection & Tracking Settings")

            col1, col2 = st.columns(2)

            with col1:
                # Device selection
                supported_devices = helpers.list_supported_devices()
                device_key = _get_pipeline_settings_key(ep_id, "detect", "device")
                device_default = helpers.DEFAULT_DEVICE_LABEL
                if device_key not in st.session_state:
                    st.session_state[device_key] = device_default
                # Fix stale session state (e.g. old "CoreML (Apple Silicon)" format)
                if st.session_state[device_key] not in supported_devices:
                    st.session_state[device_key] = device_default if device_default in supported_devices else supported_devices[0]
                device_idx = supported_devices.index(st.session_state[device_key])
                st.selectbox(
                    "Device",
                    supported_devices,
                    index=device_idx,
                    key=device_key,
                    help="Compute device for face detection and tracking",
                )

                # Detector selection
                detector_key = _get_pipeline_settings_key(ep_id, "detect", "detector")
                detector_options = list(helpers.DETECTOR_VALUE_MAP.keys())
                if detector_key not in st.session_state:
                    st.session_state[detector_key] = "RetinaFace (recommended)"
                detector_idx = detector_options.index(st.session_state[detector_key]) if st.session_state[detector_key] in detector_options else 0
                st.selectbox(
                    "Detector",
                    detector_options,
                    index=detector_idx,
                    key=detector_key,
                    help="Face detection model to use",
                )

                # Tracker selection
                tracker_key = _get_pipeline_settings_key(ep_id, "detect", "tracker")
                tracker_options = list(helpers.TRACKER_VALUE_MAP.keys())
                if tracker_key not in st.session_state:
                    st.session_state[tracker_key] = "ByteTrack (default)"
                tracker_idx = tracker_options.index(st.session_state[tracker_key]) if st.session_state[tracker_key] in tracker_options else 0
                st.selectbox(
                    "Tracker",
                    tracker_options,
                    index=tracker_idx,
                    key=tracker_key,
                    help="Multi-object tracker for linking faces across frames",
                )

            with col2:
                # Stride
                stride_key = _get_pipeline_settings_key(ep_id, "detect", "stride")
                if stride_key not in st.session_state:
                    st.session_state[stride_key] = helpers.DEFAULT_STRIDE
                st.number_input(
                    "Stride",
                    min_value=1,
                    max_value=50,
                    step=1,
                    key=stride_key,
                    help="Process every Nth frame. Higher = faster but may miss faces.",
                )

                # FPS
                fps_key = _get_pipeline_settings_key(ep_id, "detect", "fps")
                fps_default = 0.0
                if video_meta and video_meta.get("fps_detected"):
                    fps_default = float(video_meta["fps_detected"])
                if fps_key not in st.session_state:
                    st.session_state[fps_key] = fps_default
                st.number_input(
                    "FPS",
                    min_value=0.0,
                    max_value=120.0,
                    step=1.0,
                    key=fps_key,
                    help="Frames per second (0 = auto-detect from video)",
                )

                # Detection threshold
                det_thresh_key = _get_pipeline_settings_key(ep_id, "detect", "det_thresh")
                if det_thresh_key not in st.session_state:
                    st.session_state[det_thresh_key] = helpers.DEFAULT_DET_THRESH
                st.slider(
                    "Detection threshold",
                    min_value=0.1,
                    max_value=0.9,
                    step=0.01,
                    key=det_thresh_key,
                    help="Minimum confidence for face detections (higher = fewer false positives)",
                )

            st.divider()

            col3, col4 = st.columns(2)

            with col3:
                # Save frames
                save_frames_key = _get_pipeline_settings_key(ep_id, "detect", "save_frames")
                if save_frames_key not in st.session_state:
                    st.session_state[save_frames_key] = True
                st.checkbox(
                    "Save sampled frames",
                    key=save_frames_key,
                    help="Store full frames for QA and future crops",
                )

                # Save crops
                save_crops_key = _get_pipeline_settings_key(ep_id, "detect", "save_crops")
                if save_crops_key not in st.session_state:
                    st.session_state[save_crops_key] = True
                st.checkbox(
                    "Save face crops",
                    key=save_crops_key,
                    help="Export aligned face crops during detection",
                )

            with col4:
                # CPU threads - default to 4 for balanced profile
                cpu_threads_key = _get_pipeline_settings_key(ep_id, "detect", "cpu_threads")
                cpu_thread_options = [2, 4, 6, 8]
                if cpu_threads_key not in st.session_state:
                    st.session_state[cpu_threads_key] = 4
                if st.session_state[cpu_threads_key] not in cpu_thread_options:
                    st.session_state[cpu_threads_key] = 4
                st.selectbox(
                    "CPU threads (cap)",
                    options=cpu_thread_options,
                    index=cpu_thread_options.index(st.session_state[cpu_threads_key]),  # Default to 4
                    key=cpu_threads_key,
                    help="CPU thread limit (low_power=2, balanced=4, performance=8)",
                )

                # Max gap
                max_gap_key = _get_pipeline_settings_key(ep_id, "detect", "max_gap")
                if max_gap_key not in st.session_state:
                    st.session_state[max_gap_key] = helpers.DEFAULT_MAX_GAP
                st.number_input(
                    "Max gap (frames)",
                    min_value=1,
                    max_value=240,
                    step=1,
                    key=max_gap_key,
                    help="Max frames a face can be missing before track ends",
                )

            # Advanced: ByteTrack thresholds
            with st.expander("Advanced Tracking", expanded=False):
                adv_col1, adv_col2 = st.columns(2)
                with adv_col1:
                    track_high_key = _get_pipeline_settings_key(ep_id, "detect", "track_high_thresh")
                    if track_high_key not in st.session_state:
                        st.session_state[track_high_key] = TRACK_THRESH_DEFAULT
                    st.slider(
                        "Track high threshold",
                        min_value=0.1,
                        max_value=0.9,
                        step=0.01,
                        key=track_high_key,
                        help="Min confidence to continue tracking (tracking.yaml: track_thresh)",
                    )

                with adv_col2:
                    track_new_key = _get_pipeline_settings_key(ep_id, "detect", "new_track_thresh")
                    if track_new_key not in st.session_state:
                        st.session_state[track_new_key] = TRACK_NEW_THRESH_DEFAULT
                    st.slider(
                        "New track threshold",
                        min_value=0.1,
                        max_value=0.9,
                        step=0.01,
                        key=track_new_key,
                        help="Min confidence to start new track (tracking.yaml: new_track_thresh)",
                    )

                adv_col3, adv_col4 = st.columns(2)
                with adv_col3:
                    jpeg_key = _get_pipeline_settings_key(ep_id, "detect", "jpeg_quality")
                    if jpeg_key not in st.session_state:
                        st.session_state[jpeg_key] = JPEG_DEFAULT
                    st.number_input(
                        "JPEG quality",
                        min_value=50,
                        max_value=100,
                        step=5,
                        key=jpeg_key,
                        help="Compression quality for saved images",
                    )

                with adv_col4:
                    match_thresh_key = _get_pipeline_settings_key(ep_id, "detect", "match_thresh")
                    if match_thresh_key not in st.session_state:
                        st.session_state[match_thresh_key] = TRACK_MATCH_THRESH_DEFAULT
                    st.slider(
                        "Match threshold (IoU)",
                        min_value=0.3,
                        max_value=0.95,
                        step=0.01,
                        key=match_thresh_key,
                        help="IoU threshold for bbox matching (tracking.yaml: match_thresh)",
                    )

            # Scene Detection Settings - dedicated section
            with st.expander("Scene Detection (PySceneDetect)", expanded=True):
                st.markdown("""
                **Scene detection** identifies hard cuts (camera changes) in the video.
                When enabled, the tracker resets at each cut to prevent incorrectly linking
                faces across scene changes. **Recommended ON for reality TV content.**
                """)

                scene_col1, scene_col2 = st.columns(2)

                with scene_col1:
                    # Scene detector selector
                    scene_detector_key = _get_pipeline_settings_key(ep_id, "detect", "scene_detector")
                    if scene_detector_key not in st.session_state:
                        st.session_state[scene_detector_key] = "PySceneDetect (recommended)"
                    scene_detector_options = helpers.SCENE_DETECTOR_LABELS
                    scene_detector_idx = 0
                    current_detector = st.session_state[scene_detector_key]
                    if current_detector in scene_detector_options:
                        scene_detector_idx = scene_detector_options.index(current_detector)
                    st.selectbox(
                        "Scene Detector",
                        scene_detector_options,
                        index=scene_detector_idx,
                        key=scene_detector_key,
                        help="Method for detecting scene cuts. PySceneDetect uses content-based detection; HSV is a lightweight fallback.",
                    )

                    # Scene threshold
                    scene_thresh_key = _get_pipeline_settings_key(ep_id, "detect", "scene_threshold")
                    if scene_thresh_key not in st.session_state:
                        st.session_state[scene_thresh_key] = helpers.SCENE_THRESHOLD_DEFAULT
                    st.number_input(
                        "Scene Threshold",
                        min_value=5.0,
                        max_value=60.0,
                        step=1.0,
                        key=scene_thresh_key,
                        help="Sensitivity for detecting cuts. Lower = more sensitive (default: 27.0)",
                    )

                with scene_col2:
                    # Min scene length
                    scene_min_key = _get_pipeline_settings_key(ep_id, "detect", "scene_min_len")
                    if scene_min_key not in st.session_state:
                        st.session_state[scene_min_key] = helpers.SCENE_MIN_LEN_DEFAULT
                    st.number_input(
                        "Min Scene Length",
                        min_value=1,
                        max_value=60,
                        step=1,
                        key=scene_min_key,
                        help="Minimum frames between cuts to prevent rapid-fire detection (default: 12)",
                    )

                    # Warmup detections
                    scene_warmup_key = _get_pipeline_settings_key(ep_id, "detect", "scene_warmup_dets")
                    if scene_warmup_key not in st.session_state:
                        st.session_state[scene_warmup_key] = helpers.SCENE_WARMUP_DETS_DEFAULT
                    st.number_input(
                        "Warmup Detections",
                        min_value=0,
                        max_value=10,
                        step=1,
                        key=scene_warmup_key,
                        help="Forced detections after each cut to re-establish tracks (default: 3)",
                    )

        # ─── FACES HARVEST SETTINGS ──────────────────────────────────────────
        with tab_harvest:
            st.markdown("#### Faces Harvest Settings")

            col1, col2 = st.columns(2)

            with col1:
                # Device selection for embeddings
                faces_device_key = _get_pipeline_settings_key(ep_id, "harvest", "device")
                if faces_device_key not in st.session_state:
                    st.session_state[faces_device_key] = helpers.DEFAULT_DEVICE_LABEL
                # Fix stale session state (e.g. old "CoreML (Apple Silicon)" format)
                if st.session_state[faces_device_key] not in supported_devices:
                    st.session_state[faces_device_key] = helpers.DEFAULT_DEVICE_LABEL if helpers.DEFAULT_DEVICE_LABEL in supported_devices else supported_devices[0]
                faces_device_idx = supported_devices.index(st.session_state[faces_device_key])
                st.selectbox(
                    "Device (embeddings)",
                    supported_devices,
                    index=faces_device_idx,
                    key=faces_device_key,
                    help="Device for ArcFace embeddings. CoreML/GPU recommended.",
                )

                # Min frames between crops
                min_frames_key = _get_pipeline_settings_key(ep_id, "harvest", "min_frames_between_crops")
                computed_min_frames = _default_faces_min_frames_between_crops(video_meta)
                current_min_frames = helpers.coerce_int(st.session_state.get(min_frames_key))
                if current_min_frames is None or (
                    current_min_frames == MIN_FRAMES_BETWEEN_CROPS_DEFAULT
                    and computed_min_frames != MIN_FRAMES_BETWEEN_CROPS_DEFAULT
                ):
                    st.session_state[min_frames_key] = computed_min_frames
                st.number_input(
                    "Min frames between crops",
                    min_value=1,
                    max_value=600,
                    step=1,
                    key=min_frames_key,
                    help="Spacing between face crops to reduce duplicates",
                )

            with col2:
                # Save frames
                faces_save_frames_key = _get_pipeline_settings_key(ep_id, "harvest", "save_frames")
                if faces_save_frames_key not in st.session_state:
                    st.session_state[faces_save_frames_key] = False
                st.checkbox(
                    "Save full frames",
                    key=faces_save_frames_key,
                    help="Store full frames during harvest (increases storage)",
                )

                # Save crops
                faces_save_crops_key = _get_pipeline_settings_key(ep_id, "harvest", "save_crops")
                if faces_save_crops_key not in st.session_state:
                    st.session_state[faces_save_crops_key] = True
                st.checkbox(
                    "Save face crops",
                    key=faces_save_crops_key,
                    help="Export aligned face crops for review",
                )

            st.divider()

            col3, col4 = st.columns(2)
            with col3:
                # Thumbnail size
                thumb_size_key = _get_pipeline_settings_key(ep_id, "harvest", "thumb_size")
                if thumb_size_key not in st.session_state:
                    st.session_state[thumb_size_key] = FACES_THUMB_SIZE_DEFAULT
                st.number_input(
                    "Thumbnail size",
                    min_value=64,
                    max_value=512,
                    step=32,
                    key=thumb_size_key,
                    help="Size of face thumbnails in pixels",
                )

            with col4:
                # JPEG quality
                faces_jpeg_key = _get_pipeline_settings_key(ep_id, "harvest", "jpeg_quality")
                if faces_jpeg_key not in st.session_state:
                    st.session_state[faces_jpeg_key] = JPEG_DEFAULT
                st.select_slider(
                    "JPEG quality",
                    options=[60, 70, 80, 90],
                    value=st.session_state[faces_jpeg_key] if st.session_state[faces_jpeg_key] in [60, 70, 80, 90] else 70,
                    key=faces_jpeg_key,
                    help="Image quality for saved crops (lower = smaller files)",
                )

        # ─── CLUSTER SETTINGS ────────────────────────────────────────────────
        with tab_cluster:
            st.markdown("#### Clustering Settings")

            col1, col2 = st.columns(2)

            with col1:
                # Device selection
                cluster_device_key = _get_pipeline_settings_key(ep_id, "cluster", "device")
                if cluster_device_key not in st.session_state:
                    st.session_state[cluster_device_key] = helpers.DEFAULT_DEVICE_LABEL
                # Fix stale session state (e.g. old "CoreML (Apple Silicon)" format)
                if st.session_state[cluster_device_key] not in supported_devices:
                    st.session_state[cluster_device_key] = helpers.DEFAULT_DEVICE_LABEL if helpers.DEFAULT_DEVICE_LABEL in supported_devices else supported_devices[0]
                cluster_device_idx = supported_devices.index(st.session_state[cluster_device_key])
                st.selectbox(
                    "Device",
                    supported_devices,
                    index=cluster_device_idx,
                    key=cluster_device_key,
                    help="Device for similarity computations",
                )

                # Cluster similarity threshold
                cluster_thresh_key = _get_pipeline_settings_key(ep_id, "cluster", "cluster_thresh")
                if cluster_thresh_key not in st.session_state:
                    st.session_state[cluster_thresh_key] = CLUSTER_THRESH_DEFAULT
                thresh_val = st.slider(
                    "Similarity threshold",
                    min_value=0.4,
                    max_value=0.9,
                    step=0.01,
                    key=cluster_thresh_key,
                    help="Higher = stricter clustering (clustering.yaml: cluster_thresh)",
                )
                # Threshold guidance
                if thresh_val >= 0.80:
                    st.caption("Very strict: May over-split same person")
                elif thresh_val >= 0.70:
                    st.caption("Strict: Good for similar-looking people")
                elif thresh_val >= 0.55:
                    st.caption("Balanced: Recommended for most content")
                else:
                    st.caption("Lenient: May merge different people")

            with col2:
                # Min cluster size
                min_size_key = _get_pipeline_settings_key(ep_id, "cluster", "min_cluster_size")
                if min_size_key not in st.session_state:
                    st.session_state[min_size_key] = CLUSTER_MIN_SIZE_DEFAULT
                st.number_input(
                    "Min tracks per identity",
                    min_value=1,
                    max_value=50,
                    step=1,
                    key=min_size_key,
                    help="Clusters smaller than this are discarded (clustering.yaml: min_cluster_size)",
                )

                # Min identity similarity
                min_id_sim_key = _get_pipeline_settings_key(ep_id, "cluster", "min_identity_sim")
                if min_id_sim_key not in st.session_state:
                    st.session_state[min_id_sim_key] = CLUSTER_MIN_IDENTITY_SIM_DEFAULT
                st.slider(
                    "Min identity similarity",
                    min_value=0.3,
                    max_value=0.8,
                    step=0.01,
                    key=min_id_sim_key,
                    help="Tracks below this similarity to centroid are outliers (clustering.yaml: min_identity_sim)",
                )

        st.divider()
        if st.button("Close", use_container_width=True, type="primary"):
            st.rerun()

    # Button to open dialog (gear icon)
    if st.button("⚙️", key=dialog_key, help="Open pipeline settings"):
        settings_dialog()


def _get_detect_settings(ep_id: str) -> Dict[str, Any]:
    """Get current detect/track settings from session state with defaults."""
    def _get(field: str, default: Any) -> Any:
        key = _get_pipeline_settings_key(ep_id, "detect", field)
        return st.session_state.get(key, default)

    device_label = _get("device", helpers.DEFAULT_DEVICE_LABEL)
    detector_label = _get("detector", "RetinaFace (recommended)")
    tracker_label = _get("tracker", "ByteTrack (default)")

    # Scene detector: convert label to value
    scene_detector_label = _get("scene_detector", "PySceneDetect (recommended)")
    scene_detector_value = helpers.SCENE_DETECTOR_VALUE_MAP.get(
        scene_detector_label, helpers.SCENE_DETECTOR_DEFAULT
    )

    return {
        "device": helpers.DEVICE_VALUE_MAP.get(device_label, "auto"),
        "device_label": device_label,
        "detector": helpers.DETECTOR_VALUE_MAP.get(detector_label, "retinaface"),
        "detector_label": detector_label,
        "tracker": helpers.TRACKER_VALUE_MAP.get(tracker_label, "bytetrack"),
        "tracker_label": tracker_label,
        "stride": int(_get("stride", helpers.DEFAULT_STRIDE)),
        "fps": float(_get("fps", 0.0)),
        "det_thresh": float(_get("det_thresh", helpers.DEFAULT_DET_THRESH)),
        "save_frames": bool(_get("save_frames", True)),
        "save_crops": bool(_get("save_crops", True)),
        "cpu_threads": int(_get("cpu_threads", 4)),
        "max_gap": int(_get("max_gap", helpers.DEFAULT_MAX_GAP)),
        "track_high_thresh": float(_get("track_high_thresh", TRACK_THRESH_DEFAULT)),
        "new_track_thresh": float(_get("new_track_thresh", TRACK_NEW_THRESH_DEFAULT)),
        "match_thresh": float(_get("match_thresh", TRACK_MATCH_THRESH_DEFAULT)),
        "jpeg_quality": int(_get("jpeg_quality", JPEG_DEFAULT)),
        # Scene detection settings
        "scene_detector": scene_detector_value,
        "scene_detector_label": scene_detector_label,
        "scene_threshold": float(_get("scene_threshold", helpers.SCENE_THRESHOLD_DEFAULT)),
        "scene_min_len": int(_get("scene_min_len", helpers.SCENE_MIN_LEN_DEFAULT)),
        "scene_warmup_dets": int(_get("scene_warmup_dets", helpers.SCENE_WARMUP_DETS_DEFAULT)),
    }


def _get_harvest_settings(ep_id: str, video_meta: Dict[str, Any] | None) -> Dict[str, Any]:
    """Get current faces harvest settings from session state with defaults."""
    def _get(field: str, default: Any) -> Any:
        key = _get_pipeline_settings_key(ep_id, "harvest", field)
        return st.session_state.get(key, default)

    device_label = _get("device", helpers.DEFAULT_DEVICE_LABEL)
    # Promote fps-derived default when the session still holds the legacy global default.
    computed_min_frames = _default_faces_min_frames_between_crops(video_meta)
    min_frames_key = _get_pipeline_settings_key(ep_id, "harvest", "min_frames_between_crops")
    current_min_frames = helpers.coerce_int(st.session_state.get(min_frames_key))
    if current_min_frames is None or (
        current_min_frames == MIN_FRAMES_BETWEEN_CROPS_DEFAULT
        and computed_min_frames != MIN_FRAMES_BETWEEN_CROPS_DEFAULT
    ):
        st.session_state[min_frames_key] = computed_min_frames

    return {
        "device": helpers.DEVICE_VALUE_MAP.get(device_label, "auto"),
        "device_label": device_label,
        "save_frames": bool(_get("save_frames", False)),
        "save_crops": bool(_get("save_crops", True)),
        "min_frames_between_crops": int(_get("min_frames_between_crops", computed_min_frames)),
        "thumb_size": int(_get("thumb_size", FACES_THUMB_SIZE_DEFAULT)),
        "jpeg_quality": int(_get("jpeg_quality", JPEG_DEFAULT)),
    }


def _get_cluster_settings(ep_id: str) -> Dict[str, Any]:
    """Get current cluster settings from session state with defaults."""
    def _get(field: str, default: Any) -> Any:
        key = _get_pipeline_settings_key(ep_id, "cluster", field)
        return st.session_state.get(key, default)

    device_label = _get("device", helpers.DEFAULT_DEVICE_LABEL)

    return {
        "device": helpers.DEVICE_VALUE_MAP.get(device_label, "auto"),
        "device_label": device_label,
        "cluster_thresh": float(_get("cluster_thresh", CLUSTER_THRESH_DEFAULT)),
        "min_cluster_size": int(_get("min_cluster_size", CLUSTER_MIN_SIZE_DEFAULT)),
        "min_identity_sim": float(_get("min_identity_sim", CLUSTER_MIN_IDENTITY_SIM_DEFAULT)),
    }


@st.cache_data(ttl=30, show_spinner=False)
def _cached_job_defaults(ep_id: str, job_type: str) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    """Cached fetch of job defaults to reduce repeated API calls on page load."""
    try:
        resp = helpers.api_get(f"/jobs?ep_id={ep_id}&job_type={job_type}&limit=1")
    except requests.RequestException:
        return {}, None
    jobs = resp.get("jobs") or []
    if not jobs:
        return {}, None
    job = jobs[0]
    requested = job.get("requested")
    if isinstance(requested, dict):
        return dict(requested), job
    return {}, job


def _load_job_defaults(ep_id: str, job_type: str) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    """Load job defaults with caching."""
    return _cached_job_defaults(ep_id, job_type)


def _format_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return value
    try:
        est = dt.astimezone(EST_TZ)
    except Exception:
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return est.strftime("%Y-%m-%d %H:%M:%S ET")


def _format_runtime(runtime_sec: Any) -> str | None:
    try:
        total = float(runtime_sec)
    except (TypeError, ValueError):
        return None
    if total < 0:
        return None
    seconds = int(round(total))
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 3600:
        minutes, rem = divmod(seconds, 60)
        return f"{minutes}m {rem:02d}s"
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    return f"{hours}h {minutes:02d}m"


def _format_video_duration(duration_sec: Any) -> str | None:
    """Format video duration in human-readable form (e.g., '45m 30s' or '1h 23m')."""
    try:
        total = float(duration_sec)
    except (TypeError, ValueError):
        return None
    if total <= 0:
        return None
    seconds = int(round(total))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        minutes, rem = divmod(seconds, 60)
        return f"{minutes}m {rem:02d}s"
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    return f"{hours}h {minutes:02d}m"


def _runtime_from_iso(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return None
    delta = (end_dt - start_dt).total_seconds()
    return delta if delta >= 0 else None


def _estimate_storage_bytes(
    duration_sec: float | None,
    fps: float,
    save_frames: bool,
    save_crops: bool,
    avg_faces_per_frame: float = AVG_FACES_PER_FRAME,
) -> Dict[str, int]:
    """Estimate storage requirements for a detect/track job.

    Returns dict with 'frames_bytes', 'crops_bytes', 'total_bytes', and 'sampled_frames'.
    """
    result = {
        "frames_bytes": 0,
        "crops_bytes": 0,
        "thumbs_bytes": 0,
        "total_bytes": 0,
        "sampled_frames": 0,
    }

    if not duration_sec or duration_sec <= 0 or fps <= 0:
        return result

    # Calculate sampled frame count
    sampled_frames = int(duration_sec * fps)
    result["sampled_frames"] = sampled_frames

    if save_frames:
        result["frames_bytes"] = sampled_frames * FRAME_JPEG_SIZE_EST_BYTES

    if save_crops:
        # Estimate faces detected across all frames
        total_faces = int(sampled_frames * avg_faces_per_frame)
        result["crops_bytes"] = total_faces * CROP_JPEG_SIZE_EST_BYTES
        # Thumbnails are generated per unique track (roughly 1 thumb per face)
        result["thumbs_bytes"] = total_faces * 20_000  # ~20KB per thumb

    result["total_bytes"] = (
        result["frames_bytes"] + result["crops_bytes"] + result["thumbs_bytes"]
    )

    return result


def _format_storage_size(bytes_val: int) -> str:
    """Format bytes as human-readable string (e.g., '1.5 GB', '234 MB')."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    if bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.1f} KB"
    if bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.1f} MB"
    return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"


def _render_storage_estimate(
    duration_sec: float | None,
    fps: float,
    save_frames: bool,
    save_crops: bool,
) -> None:
    """Render storage impact estimate in the UI."""
    estimate = _estimate_storage_bytes(duration_sec, fps, save_frames, save_crops)

    if estimate["total_bytes"] == 0:
        return

    # Build estimate string
    parts = []
    if estimate["frames_bytes"] > 0:
        parts.append(f"Frames: {_format_storage_size(estimate['frames_bytes'])}")
    if estimate["crops_bytes"] > 0:
        parts.append(f"Crops: {_format_storage_size(estimate['crops_bytes'])}")
    if estimate["thumbs_bytes"] > 0:
        parts.append(f"Thumbs: {_format_storage_size(estimate['thumbs_bytes'])}")

    total_str = _format_storage_size(estimate["total_bytes"])
    parts_str = " + ".join(parts) if parts else ""

    # Display in a subtle info box
    st.caption(
        f"**Est. Storage:** {total_str} ({parts_str}) • "
        f"{estimate['sampled_frames']:,} frames @ {fps:.1f} FPS"
    )


def _fetch_artifact_status(ep_id: str) -> Dict[str, Any] | None:
    """Fetch artifact sync status from API."""
    try:
        return helpers.api_get(f"/episodes/{ep_id}/artifact_status")
    except requests.RequestException:
        return None


def _render_sync_status_badge(status: str) -> str:
    """Return a styled badge for sync status."""
    badges = {
        "synced": "✅ Synced",
        "partial": "⚠️ Partial",
        "pending": "🔄 Pending",
        "empty": "📭 Empty",
        "s3_disabled": "⚙️ S3 Disabled",
        "unknown": "❓ Unknown",
    }
    return badges.get(status, status)


def _render_artifact_counts(local: Dict[str, int], s3: Dict[str, int]) -> str:
    """Format artifact counts as a summary string."""
    parts = []
    for key in ["frames", "crops", "thumbs_tracks", "manifests"]:
        local_count = local.get(key, 0)
        s3_count = s3.get(key, 0)
        label = key.replace("_", " ").title()
        if local_count > 0 or s3_count > 0:
            parts.append(f"{label}: {local_count} local / {s3_count} S3")
    return " | ".join(parts) if parts else "No artifacts"


def _choose_value(*candidates: Any, fallback: str) -> str:
    for candidate in candidates:
        if isinstance(candidate, str):
            cleaned = candidate.strip()
            if cleaned:
                return cleaned.lower()
    return fallback


def _resolved_device_label(label: str | None) -> str:
    normalized = label if label in helpers.DEVICE_LABELS else None
    if not normalized or normalized == "Auto":
        return helpers._guess_device_label()
    return normalized


def _detect_setting_key(ep_id: str, field: str) -> str:
    return f"episode_detail_detect::{ep_id}::{field}"


def _job_activity_key(ep_id: str) -> str:
    return f"{ep_id}::job_active"


def _detect_job_state_key(ep_id: str) -> str:
    return f"{ep_id}::detect_job_running"


def _set_job_active(ep_id: str, active: bool) -> None:
    st.session_state[_job_activity_key(ep_id)] = bool(active)


def _job_active(ep_id: str) -> bool:
    return bool(st.session_state.get(_job_activity_key(ep_id), False))


# Stale job detection constants
JOB_STALE_TIMEOUT_SECONDS = 300  # 5 minutes without progress update = stale


def _resolve_session_run_id(ep_id: str) -> str | None:
    """Best-effort run_id for run-scoped artifacts in this UI session."""
    # Prefer the auto-run run_id when present, otherwise use the attempt selector value.
    for key in (f"{ep_id}::autorun_run_id", f"{ep_id}::active_run_id"):
        value = st.session_state.get(key)
        if not isinstance(value, str):
            continue
        candidate = value.strip()
        if not candidate:
            continue
        try:
            return run_layout.normalize_run_id(candidate)
        except ValueError:
            continue
    try:
        qp_value = st.query_params.get("run_id")
    except Exception:
        qp_value = None
    candidate = None
    if isinstance(qp_value, str):
        candidate = qp_value.strip()
    elif isinstance(qp_value, list) and qp_value:
        candidate = str(qp_value[0]).strip()
    if candidate:
        try:
            return run_layout.normalize_run_id(candidate)
        except ValueError:
            pass
    return None


def _get_progress_file_age(ep_id: str) -> float | None:
    """Get the age of the progress file in seconds, or None if not found."""
    run_id = _resolve_session_run_id(ep_id)
    legacy_dir = get_path(ep_id, "detections").parent
    candidates: list[Path] = []
    if run_id:
        try:
            candidates.append(run_layout.run_root(ep_id, run_id) / "progress.json")
        except ValueError:
            pass
    candidates.append(legacy_dir / "progress.json")
    try:
        for progress_path in candidates:
            if progress_path.exists():
                return time.time() - progress_path.stat().st_mtime
    except OSError:
        pass
    return None


def _get_most_recent_run_marker_age(ep_id: str) -> float | None:
    """Get the age of the most recently updated run marker in seconds.

    Checks all phase markers (detect_track, faces_embed, cluster) and returns
    the age of whichever was most recently modified. Returns None if no markers found.
    """
    run_id = _resolve_session_run_id(ep_id)
    legacy_runs_dir = get_path(ep_id, "detections").parent / "runs"
    scoped_runs_dir: Path | None = None
    if run_id:
        try:
            scoped_runs_dir = run_layout.run_root(ep_id, run_id)
        except ValueError:
            scoped_runs_dir = None

    phases = ["detect_track.json", "faces_embed.json", "cluster.json"]
    most_recent_mtime = None

    for phase_file in phases:
        for runs_dir in (scoped_runs_dir, legacy_runs_dir):
            if runs_dir is None:
                continue
            marker_path = runs_dir / phase_file
            try:
                if marker_path.exists():
                    mtime = marker_path.stat().st_mtime
                    if most_recent_mtime is None or mtime > most_recent_mtime:
                        most_recent_mtime = mtime
            except OSError:
                continue

    if most_recent_mtime is not None:
        return time.time() - most_recent_mtime
    return None


# Suggestion 8: Mtime-based retry backoff constants
_MAX_RETRY_ATTEMPTS = 30  # Max 30 attempts (~30 seconds)
_ARTIFACT_FRESHNESS_WINDOW = 60  # Consider artifacts "fresh" if modified within 60s


def _get_manifest_mtime(ep_id: str, phase: str) -> float | None:
    """Get the mtime of a manifest file for a specific phase."""
    manifest_map = {
        "faces": "faces.jsonl",
        "detect": "tracks.jsonl",
        "cluster": "identities.json",
    }
    filename = manifest_map.get(phase)
    if not filename:
        return None
    run_id = _resolve_session_run_id(ep_id)
    legacy_dir = get_path(ep_id, "detections").parent
    candidates: list[Path] = []
    if run_id:
        try:
            candidates.append(run_layout.run_root(ep_id, run_id) / filename)
        except ValueError:
            pass
    candidates.append(legacy_dir / filename)
    try:
        for manifest_path in candidates:
            if manifest_path.exists():
                return manifest_path.stat().st_mtime
    except OSError:
        pass
    return None


def _should_retry_phase_trigger(ep_id: str, phase: str, retry_count: int) -> tuple[bool, str]:
    """Suggestion 8: Mtime-based retry backoff instead of fixed 10 attempts.

    Check if we should retry waiting for a phase to be ready based on:
    1. Artifact freshness (if manifest was recently modified, keep waiting)
    2. Retry count with increased limit (30 attempts vs 10)

    Returns: (should_retry, status_message)
    """
    # Check if artifacts are freshly written (within 60s)
    manifest_mtime = _get_manifest_mtime(ep_id, phase)
    if manifest_mtime is not None:
        age_seconds = time.time() - manifest_mtime
        if age_seconds < _ARTIFACT_FRESHNESS_WINDOW:
            # Artifacts are fresh, definitely keep retrying
            return True, f"Artifacts fresh ({int(age_seconds)}s old), retrying..."

    # Check retry count with longer cap
    if retry_count < _MAX_RETRY_ATTEMPTS:
        return True, f"Waiting for {phase} phase... ({retry_count}/{_MAX_RETRY_ATTEMPTS})"

    return False, f"Timeout waiting for {phase} prerequisites after {_MAX_RETRY_ATTEMPTS} attempts"


def _sync_job_state_with_api(
    ep_id: str,
    running_job_key: str,
    running_detect_job: dict | None,
    running_faces_job: dict | None,
    running_cluster_job: dict | None,
    running_audio_job: dict | None,
) -> tuple[bool, str | None]:
    """Synchronize session state with API-based job status.

    This is the single source of truth for job status. If the API says no job
    is running but session state says one is, this clears the session state
    and returns information about the stale job.

    Returns:
        Tuple of (any_job_running, stale_job_warning)
    """
    api_says_running = any([
        running_detect_job,
        running_faces_job,
        running_cluster_job,
        running_audio_job,
    ])
    session_says_running = (
        st.session_state.get(running_job_key, False) or
        _job_active(ep_id)
    )

    stale_warning = None

    if api_says_running:
        # API confirms job is running - trust it
        return True, None

    if session_says_running and not api_says_running:
        # Session thinks a job is running but API disagrees
        # This could mean the job crashed, was cancelled externally, or completed
        progress_age = _get_progress_file_age(ep_id)

        if progress_age is not None and progress_age < 30:
            # Progress file was updated recently - job may have just finished
            # Give it a moment before declaring it stale
            pass
        else:
            # Clear the stale session state
            st.session_state[running_job_key] = False
            _set_job_active(ep_id, False)

            # Generate a warning if progress is old
            if progress_age is not None and progress_age > JOB_STALE_TIMEOUT_SECONDS:
                stale_warning = (
                    f"A previous job appears to have stalled or crashed "
                    f"(no progress update for {int(progress_age // 60)} minutes). "
                    f"Controls have been re-enabled."
                )
            else:
                LOGGER.debug(
                    "Cleared stale session state for %s - API says no job running",
                    ep_id
                )

    return api_says_running, stale_warning


def _status_cache_key(ep_id: str) -> str:
    return f"{ep_id}::status_payload"


def _status_timestamp_key(ep_id: str) -> str:
    return f"{ep_id}::status_fetched_at"


def _status_fetch_token_key(ep_id: str) -> str:
    return f"{ep_id}::status_fetch_token"


def _status_force_refresh_key(ep_id: str) -> str:
    return f"{ep_id}::status_force_refresh"


def _refresh_click_key(ep_id: str) -> str:
    return f"{ep_id}::status_refresh_clicked_at"


def _status_mtimes_key(ep_id: str) -> str:
    return f"{ep_id}::status_mtimes"


def _navigate_to_upload(ep_id: str) -> None:
    helpers.set_ep_id(ep_id, rerun=False, origin="replace")
    params = st.query_params
    params["ep_id"] = ep_id
    st.query_params = params
    helpers.try_switch_page("pages/0_Upload_Video.py")


def _render_device_summary(requested: str | None, resolved: str | None) -> None:
    req_label = helpers.device_label_from_value(requested) if requested else None
    resolved_label = helpers.device_label_from_value(resolved or requested)
    if not (req_label or resolved_label):
        return
    if req_label and resolved_label and req_label != resolved_label:
        caption = f"Device: requested {req_label} → resolved {resolved_label}"
        if req_label in {"CUDA", "CoreML", "MPS"} and resolved_label == "CPU":
            st.caption(f"⚠️ {caption}")
        else:
            st.caption(caption)
    else:
        st.caption(f"Device: {resolved_label or req_label}")


def _estimate_runtime_seconds(frames: int, device_value: str) -> float:
    per_device = {
        "cpu": 45.0,
        "cuda": 110.0,
        "coreml": 90.0,
        "mps": 70.0,
    }
    rate = per_device.get((device_value or "cpu").lower(), 40.0)
    if frames <= 0 or rate <= 0:
        return 0.0
    return frames / rate


# =============================================================================
# Improve Faces Modal (Episode Detail version)
# =============================================================================

def _improve_faces_state_key(ep_id: str, run_id: str | None, suffix: str) -> str:
    scope = run_id or "legacy"
    return f"{ep_id}::{scope}::improve_faces::{suffix}"


def _session_improve_faces_state_key(ep_id: str, suffix: str) -> str:
    return _improve_faces_state_key(ep_id, _resolve_session_run_id(ep_id), suffix)


def _start_improve_faces_ep_detail(ep_id: str, *, force: bool = False) -> bool:
    """Fetch initial suggestions and activate the Improve Faces modal on Episode Detail."""
    run_id = _resolve_session_run_id(ep_id)
    # Ensure any one-shot trigger only fires once (prevents re-opening on each rerender).
    st.session_state.pop(_improve_faces_state_key(ep_id, run_id, "trigger"), None)
    if not run_id:
        st.warning("Improve Faces requires a run-scoped attempt (run_id). Select an attempt above and retry.")
        return False
    try:
        resp = helpers.api_get(
            f"/episodes/{ep_id}/face_review/initial_unassigned_suggestions",
            params={"run_id": run_id},
        )
    except Exception as exc:
        st.error(f"Failed to load Improve Faces suggestions: {exc}")
        return False

    suggestions = resp.get("suggestions", []) if isinstance(resp, dict) else []
    initial_done = bool(resp.get("initial_pass_done")) if isinstance(resp, dict) else False
    active_key = _improve_faces_state_key(ep_id, run_id, "active")
    suggestions_key = _improve_faces_state_key(ep_id, run_id, "suggestions")
    index_key = _improve_faces_state_key(ep_id, run_id, "index")
    empty_reason_key = _improve_faces_state_key(ep_id, run_id, "empty_reason")
    complete_key = _improve_faces_state_key(ep_id, run_id, "complete")

    if not suggestions or initial_done:
        reason = "initial_done" if initial_done else "no_suggestions"
        if force:
            st.session_state[active_key] = True
            st.session_state[suggestions_key] = []
            st.session_state[index_key] = 0
            st.session_state[empty_reason_key] = reason
            st.session_state[complete_key] = True
            st.rerun()
            return True

        st.session_state.pop(active_key, None)
        st.session_state.pop(suggestions_key, None)
        st.session_state.pop(index_key, None)
        st.session_state.pop(empty_reason_key, None)
        st.session_state[complete_key] = True
        return False

    st.session_state[active_key] = True
    st.session_state[suggestions_key] = suggestions
    st.session_state[index_key] = 0
    st.session_state.pop(complete_key, None)
    st.session_state.pop(empty_reason_key, None)
    st.rerun()
    return True


def _render_improve_faces_modal_ep_detail(ep_id: str) -> None:
    """Render Improve Faces dialog on Episode Detail page if active."""
    run_id = _resolve_session_run_id(ep_id)
    active_key = _improve_faces_state_key(ep_id, run_id, "active")
    if not st.session_state.get(active_key):
        return

    suggestions_key = _improve_faces_state_key(ep_id, run_id, "suggestions")
    index_key = _improve_faces_state_key(ep_id, run_id, "index")
    empty_reason_key = _improve_faces_state_key(ep_id, run_id, "empty_reason")
    complete_key = _improve_faces_state_key(ep_id, run_id, "complete")

    suggestions = st.session_state.get(suggestions_key, []) or []
    idx = st.session_state.get(index_key, 0) or 0

    @st.dialog("Improve Face Clustering", width="large")
    def _dialog():
        suggestions_local = st.session_state.get(suggestions_key, []) or []
        current_idx = st.session_state.get(index_key, 0) or 0

        def _render_thumb(url: str | None) -> None:
            """Render face crop filling the column width."""
            if not url:
                st.markdown("*No image available*")
                return
            st.image(url, use_container_width=True)

        if not suggestions_local or current_idx >= len(suggestions_local):
            empty_reason = st.session_state.get(empty_reason_key)
            if empty_reason == "initial_done":
                st.info("Improve Faces initial pass already completed for this episode.")
            elif empty_reason == "no_suggestions":
                st.info("No Improve Faces suggestions right now.")
            st.success("All suggestions reviewed!")
            st.markdown("Click **Faces Review** to continue assigning faces to cast members.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Faces Review", type="primary", use_container_width=True, key="improve_go_faces_review"):
                    st.session_state.pop(active_key, None)
                    st.session_state.pop(suggestions_key, None)
                    st.session_state.pop(index_key, None)
                    st.session_state.pop(empty_reason_key, None)
                    st.session_state[complete_key] = True
                    try:
                        qp = st.query_params
                        qp["ep_id"] = ep_id
                        if run_id:
                            qp["run_id"] = run_id
                        else:
                            try:
                                del qp["run_id"]
                            except Exception:
                                pass
                        st.query_params = qp
                    except Exception:
                        pass
                    st.switch_page("pages/3_Faces_Review.py")
            with col2:
                if st.button("Close", use_container_width=True, key="improve_close"):
                    st.session_state.pop(active_key, None)
                    st.session_state.pop(suggestions_key, None)
                    st.session_state.pop(index_key, None)
                    st.session_state.pop(empty_reason_key, None)
                    st.session_state[complete_key] = True
                    st.rerun()
            return

        suggestion = suggestions_local[current_idx]
        cluster_a = suggestion.get("cluster_a", {}) if isinstance(suggestion, dict) else {}
        cluster_b = suggestion.get("cluster_b", {}) if isinstance(suggestion, dict) else {}
        similarity = suggestion.get("similarity", 0)

        st.markdown(f"**Are they the same person?** — {current_idx + 1} of {len(suggestions_local)}")
        st.progress((current_idx + 1) / len(suggestions_local))

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            crop_url_a = cluster_a.get("crop_url")
            resolved_a = helpers.resolve_thumb(crop_url_a) if crop_url_a else None
            _render_thumb(resolved_a)
            st.caption(f"Cluster: {cluster_a.get('id', '?')}")
            st.caption(f"Tracks: {cluster_a.get('tracks', 0)} · Faces: {cluster_a.get('faces', 0)}")

        with img_col2:
            crop_url_b = cluster_b.get("crop_url")
            resolved_b = helpers.resolve_thumb(crop_url_b) if crop_url_b else None
            _render_thumb(resolved_b)
            st.caption(f"Cluster: {cluster_b.get('id', '?')}")
            st.caption(f"Tracks: {cluster_b.get('tracks', 0)} · Faces: {cluster_b.get('faces', 0)}")

        st.caption(f"Similarity: {similarity:.1%}")

        btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 1])

        def _advance():
            st.session_state[index_key] = current_idx + 1

        with btn_col1:
            if st.button("Yes", type="primary", use_container_width=True, key=f"ep_improve_yes_{current_idx}"):
                exec_mode = helpers.get_execution_mode(ep_id)
                payload = {
                    "pair_type": "unassigned_unassigned",
                    "cluster_a_id": cluster_a.get("id"),
                    "cluster_b_id": cluster_b.get("id"),
                    "decision": "merge",
                    "execution_mode": "redis" if exec_mode != "local" else "local",
                }
                try:
                    helpers.api_post(
                        f"/episodes/{ep_id}/face_review/decision/start",
                        json=payload,
                        params={"run_id": run_id},
                    )
                    _advance()
                except Exception as exc:
                    st.error(f"Failed to save merge decision: {exc}")
                    LOGGER.error("[FACE_REVIEW] Merge decision failed: %s", exc)

        with btn_col2:
            if st.button("No", use_container_width=True, key=f"ep_improve_no_{current_idx}"):
                exec_mode = helpers.get_execution_mode(ep_id)
                payload = {
                    "pair_type": "unassigned_unassigned",
                    "cluster_a_id": cluster_a.get("id"),
                    "cluster_b_id": cluster_b.get("id"),
                    "decision": "reject",
                    "execution_mode": "redis" if exec_mode != "local" else "local",
                }
                try:
                    helpers.api_post(
                        f"/episodes/{ep_id}/face_review/decision/start",
                        json=payload,
                        params={"run_id": run_id},
                    )
                    _advance()
                except Exception as exc:
                    st.error(f"Failed to save reject decision: {exc}")
                    LOGGER.error("[FACE_REVIEW] Reject decision failed: %s", exc)

        with btn_col3:
            if st.button("Skip All", use_container_width=True, key=f"ep_improve_skip_{current_idx}"):
                st.session_state.pop(active_key, None)
                st.session_state.pop(suggestions_key, None)
                st.session_state.pop(index_key, None)
                st.session_state[complete_key] = True
                st.rerun()

    _dialog()



@st.cache_data(ttl=30, show_spinner=False)
def _cached_count_manifest_rows(path_str: str, mtime: float) -> int | None:
    """Cache manifest row counts using path+mtime as cache key."""
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def _cached_manifest_has_rows(path_str: str, mtime: float) -> bool:
    """Cache manifest existence check using path+mtime as cache key."""
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return True
    except OSError:
        return False
    return False


def _count_manifest_rows(path: Path) -> int | None:
    """Count rows in manifest (cached by path+mtime)."""
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
        return _cached_count_manifest_rows(str(path), mtime)
    except OSError:
        return None


def _manifest_has_rows(path: Path) -> bool:
    """Check if manifest has rows (cached by path+mtime)."""
    if not path.exists() or not path.is_file():
        return False
    try:
        mtime = path.stat().st_mtime
        return _cached_manifest_has_rows(str(path), mtime)
    except OSError:
        return False


@st.cache_data(ttl=10, show_spinner=False)
def _cached_episode_details(ep_id: str, cache_key: float) -> Dict[str, Any]:
    """Cache episode details API response with 10s TTL."""
    return helpers.api_get(f"/episodes/{ep_id}")


@st.cache_data(ttl=10, show_spinner=False)
def _cached_episode_status(
    ep_id: str,
    cache_key: float,
    marker_mtimes: tuple,
    run_id: str | None = None,
) -> Dict[str, Any] | None:
    """Cache episode status API response with 10s TTL.

    Args:
        ep_id: Episode ID
        cache_key: Fetch token for manual cache busting
        marker_mtimes: Tuple of artifact mtimes to auto-invalidate cache (runs + manifests)
        run_id: Optional attempt/run_id to scope status.
    """
    return helpers.get_episode_status(ep_id, run_id=run_id)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_storage_status() -> Dict[str, Any] | None:
    """Cache storage status API response with 60s TTL (rarely changes)."""
    try:
        return helpers.api_get("/config/storage")
    except requests.RequestException:
        return None


# Suggestion 2: Adaptive cache TTL based on job state
# During active jobs: 10s TTL (human can't perceive <10s delay in progress updates)
# When idle: 3s TTL for responsive manual refresh
_CACHE_TTL_ACTIVE = 10  # Longer TTL during job execution
_CACHE_TTL_IDLE = 3  # Shorter TTL when idle


def _any_job_active(ep_id: str) -> bool:
    """Check if any job is active for this episode (from session state)."""
    _autorun_key = f"{ep_id}::autorun_active"
    if st.session_state.get(_autorun_key):
        return True
    running_key = f"episode_detail_running_job:{ep_id}"
    if st.session_state.get(running_key):
        return True
    return False


def _get_adaptive_ttl(ep_id: str) -> int:
    """Get cache TTL based on job activity - longer during jobs, shorter when idle."""
    return _CACHE_TTL_ACTIVE if _any_job_active(ep_id) else _CACHE_TTL_IDLE


@st.cache_data(ttl=_CACHE_TTL_ACTIVE, show_spinner=False)
def _cached_celery_jobs() -> Dict[str, Any] | None:
    """Cache celery jobs API response (adaptive TTL handled at call site)."""
    try:
        return helpers.api_get("/celery_jobs")
    except requests.RequestException:
        return None


@st.cache_data(ttl=_CACHE_TTL_ACTIVE, show_spinner=False)
def _cached_episode_jobs(ep_id: str) -> Dict[str, Any] | None:
    """Cache episode jobs list API response."""
    try:
        return helpers.api_get(f"/jobs?ep_id={ep_id}&limit=20")
    except requests.RequestException:
        return None


@st.cache_data(ttl=_CACHE_TTL_ACTIVE, show_spinner=False)
def _cached_local_jobs(ep_id: str | None = None) -> Dict[str, Any] | None:
    """Cache local jobs list API response."""
    try:
        params = f"?ep_id={ep_id}" if ep_id else ""
        return helpers.api_get(f"/celery_jobs/local{params}")
    except requests.RequestException:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def _cached_video_meta(ep_id: str) -> Dict[str, Any] | None:
    """Cache video metadata API response with 60s TTL (static for an episode)."""
    try:
        return helpers.api_get(f"/episodes/{ep_id}/video_meta")
    except requests.RequestException:
        return None


def _detect_track_manifests_ready(detections_path: Path, tracks_path: Path) -> dict:
    detections_ready = _manifest_has_rows(detections_path)
    tracks_ready = _manifest_has_rows(tracks_path)
    tracks_only = bool(tracks_ready and not detections_ready)
    manifest_ready = bool(detections_ready and tracks_ready)
    return {
        "detections_ready": detections_ready,
        "tracks_ready": tracks_ready,
        "manifest_ready": manifest_ready,
        "tracks_only_fallback": tracks_only,
    }


def _compute_detect_track_effective_status(
    detect_status: Dict[str, Any],
    *,
    manifest_ready: bool,
    tracks_only_fallback: bool,
    tracks_ready_flag: bool,
    job_state: str | None = None,
) -> tuple[str, bool, bool, bool]:
    normalized_job_state = str(job_state or "").strip().lower()
    if normalized_job_state == "running":
        return "running", False, False, tracks_only_fallback
    if tracks_ready_flag:
        return "success", True, False, tracks_only_fallback
    normalized_status = str(detect_status.get("status") or "missing").strip().lower()
    if not normalized_status:
        normalized_status = "missing"
    manifest_tracks_ready = bool(manifest_ready)
    if normalized_status == "success":
        if manifest_tracks_ready:
            return "success", True, False, tracks_only_fallback
        return "stale", False, False, tracks_only_fallback
    if manifest_tracks_ready:
        return "success", True, True, tracks_only_fallback
    return normalized_status, False, False, tracks_only_fallback


def _estimated_sampled_frames(meta: Dict[str, Any] | None, stride: int) -> int | None:
    if not meta:
        return None
    frames_val = meta.get("frames") if isinstance(meta, dict) else None
    fps_detected = meta.get("fps_detected") if isinstance(meta, dict) else None
    duration_sec = meta.get("duration_sec") if isinstance(meta, dict) else None
    frames = None
    try:
        if frames_val is not None:
            frames = float(frames_val)
        elif duration_sec and (fps_detected or fps_detected == 0):
            frames = float(duration_sec) * float(fps_detected or 0)
        elif duration_sec:
            frames = float(duration_sec) * 24.0
    except (TypeError, ValueError):
        frames = None
    if not frames or frames <= 0:
        return None
    stride_val = max(int(stride or 1), 1)
    return max(int(frames // stride_val), 0)


cfg = helpers.init_page("Episode Detail")
helpers.render_page_header("workspace-ui:2_Episode_Detail", "Episode Detail")
helpers.inject_log_container_css()  # Limit log container height with scrolling
flash_error = st.session_state.pop("episode_detail_flash_error", None)
flash_message = st.session_state.pop("episode_detail_flash", None)
if flash_error:
    st.error(flash_error)
if flash_message:
    st.success(flash_message)

# Bug #8: Remove legacy unnamespaced keys that may exist from older sessions
# Current code uses namespaced keys via _get_pipeline_settings_key(ep_id, "detect", "detector")
if "detector" in st.session_state:
    del st.session_state["detector"]
if "tracker" in st.session_state:
    del st.session_state["tracker"]


def _handle_missing_episode(ep_id: str) -> None:
    st.warning("Episode not tracked yet.")
    parsed = helpers.parse_ep_id(ep_id)
    if not parsed:
        st.info("Unable to parse show/season/episode. Use the S3 browser to create it.")
        st.stop()
    payload = {
        "ep_id": ep_id,
        "show_slug": str(parsed["show"]).lower(),
        "season": int(parsed["season"]),
        "episode": int(parsed["episode"]),
    }
    if st.button("Create episode in store", key="episode_detail_create"):
        try:
            helpers.api_post("/episodes/upsert_by_id", payload)
        except requests.RequestException as exc:
            st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/upsert_by_id", exc))
        else:
            st.success("Episode tracked. Reloading…")
            helpers.set_ep_id(ep_id)
            st.rerun()
    st.stop()


def _prompt_for_episode() -> None:
    st.subheader("Select Episode from S3")

    # Fetch shows from S3
    try:
        shows_payload = helpers.api_get("/episodes/s3_shows")
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/s3_shows", exc))
        st.stop()

    shows = shows_payload.get("shows", [])
    if not shows:
        st.info("No shows found in S3. Upload an episode first.")
        st.stop()

    # Show dropdown
    show_options = {show["show"]: show for show in shows}
    selected_show = st.selectbox(
        "Show",
        list(show_options.keys()),
        format_func=lambda s: f"{s} ({show_options[s]['episode_count']} episodes)",
        key="episode_detail_show_select",
    )

    if not selected_show:
        st.stop()

    # Fetch episodes for selected show
    try:
        episodes_payload = helpers.api_get(f"/episodes/s3_shows/{selected_show}/episodes")
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/s3_shows/{selected_show}/episodes", exc))
        st.stop()

    episodes = episodes_payload.get("episodes", [])
    if not episodes:
        st.warning(f"No episodes found for show '{selected_show}'")
        st.stop()

    tracked_eps = [ep for ep in episodes if ep.get("exists_in_store")]
    orphan_eps = [ep for ep in episodes if not ep.get("exists_in_store")]
    show_orphans = True
    if orphan_eps:
        default_show = not bool(tracked_eps)
        show_orphans = st.checkbox(
            f"Show {len(orphan_eps)} orphan uploads (⚠)",
            value=default_show,
            key="episode_detail_show_orphans",
            help="Orphan uploads are raw S3 videos that were removed from the EpisodeStore. "
            "Delete them permanently with `python tools/prune_orphan_episodes.py --apply`.",
        )
        if not show_orphans and tracked_eps:
            st.info(
                f"Hiding {len(orphan_eps)} orphan uploads. Run `python tools/prune_orphan_episodes.py --apply` to remove them."
            )
    filtered_episodes = [ep for ep in episodes if show_orphans or ep.get("exists_in_store")]
    if not filtered_episodes:
        st.warning("No tracked episodes available. Upload a video or enable orphan view above.")
        st.stop()

    # Episode dropdown
    episode_options = {ep["ep_id"]: ep for ep in filtered_episodes}
    selected_ep_id = st.selectbox(
        "Episode",
        list(episode_options.keys()),
        format_func=lambda eid: f"S{episode_options[eid]['season']:02d}E{episode_options[eid]['episode']:02d} ({eid}) {'✓' if episode_options[eid]['exists_in_store'] else '⚠'}",
        key="episode_detail_ep_select",
    )

    if not selected_ep_id:
        st.stop()

    selected_episode = episode_options[selected_ep_id]

    # Show episode info
    st.caption(f"S3 key: `{selected_episode['key']}`")
    if selected_episode["exists_in_store"]:
        st.caption("✓ Tracked in episode store")
    else:
        st.warning("⚠ Not tracked in episode store yet. Click 'Load Episode' to create it.")

    if st.button("Load Episode", use_container_width=True, type="primary"):
        # If not in store, create it first
        if not selected_episode["exists_in_store"]:
            parsed = helpers.parse_ep_id(selected_ep_id)
            if parsed:
                payload = {
                    "ep_id": selected_ep_id,
                    "show_slug": str(parsed["show"]).lower(),
                    "season": int(parsed["season"]),
                    "episode": int(parsed["episode"]),
                }
                try:
                    helpers.api_post("/episodes/upsert_by_id", payload)
                    st.success(f"Episode `{selected_ep_id}` created in store.")
                except requests.RequestException as exc:
                    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/upsert_by_id", exc))
                    st.stop()

        helpers.set_ep_id(selected_ep_id)
        st.rerun()

    st.stop()


def _format_phase_status(label: str, status: Dict[str, Any], count_key: str) -> str:
    status_value = str(status.get("status") or "missing").lower()
    if status_value == "success":
        count_val = status.get(count_key)
        parts = [f"{label}: Complete"]
        if isinstance(count_val, int):
            parts.append(f"({count_val:,})")
        finished = _format_timestamp(status.get("finished_at"))
        if finished:
            parts.append(f"• finished {finished}")
        return " ".join(parts)
    if status_value == "missing":
        base = f"{label}: Not started"
    else:
        base = f"{label}: {status_value.title()}"
    finished = _format_timestamp(status.get("finished_at"))
    if finished:
        base += f" • last run {finished}"
    if status.get("error"):
        base += f" • {status['error']}"
    return base


def _ensure_local_artifacts(ep_id: str, details: Dict[str, Any]) -> bool:
    local_block = details.setdefault("local", {})
    video_path = get_path(ep_id, "video")
    if video_path.exists():
        local_block["path"] = str(video_path)
        local_block["exists"] = True
        return True
    s3_meta = details.get("s3") or {}
    if not (s3_meta.get("v2_exists") or s3_meta.get("v1_exists")):
        st.error("Episode is not mirrored in S3; mirror/upload the video before running this job.")
        return False
    mirror_path = f"/episodes/{ep_id}/mirror"
    with st.spinner("Mirroring video from S3 (this may take several minutes for large files)…"):
        try:
            # Use longer timeout for S3 downloads (10 minutes)
            resp = helpers.api_post(mirror_path, timeout=600)
        except requests.RequestException as exc:
            st.error(helpers.describe_error(f"{cfg['api_base']}{mirror_path}", exc))
            return False
        st.success(
            f"Mirrored to {helpers.link_local(resp['local_video_path'])} " f"({helpers.human_size(resp.get('bytes'))})"
        )
        local_block["path"] = resp.get("local_video_path") or str(video_path)
        local_block["exists"] = True
        return True


def _launch_detect_job(
    local_exists: bool,
    ep_id: str,
    details: Dict[str, Any],
    job_payload: Dict[str, Any],
    device_value: str,
    detector_value: str,
    tracker_value: str,
    mode_label: str,
    device_label: str,
    running_state_key: str | None = None,
    *,
    active_job_key: str | None = None,
    detect_flag_key: str | None = None,
):
    current_local = local_exists
    if not current_local:
        if not _ensure_local_artifacts(ep_id, details):
            return current_local, None, "mirror_failed"
        current_local = True
    # Clear completion marker when starting new job
    st.session_state.pop(f"{ep_id}::detect_job_complete", None)
    if running_state_key:
        st.session_state[running_state_key] = True
    if active_job_key:
        st.session_state[active_job_key] = True
    if detect_flag_key:
        st.session_state[detect_flag_key] = True
    try:
        # Use execution mode from UI settings (respects local/redis toggle)
        execution_mode = helpers.get_execution_mode(ep_id)
        mode_desc = "local" if execution_mode == "local" else "Celery"
        runner = helpers.run_pipeline_job_with_mode
        if execution_mode == "local":
            summary, error_message = runner(
                ep_id,
                "detect_track",
                job_payload,
                requested_device=device_value,
                requested_detector=detector_value,
                requested_tracker=tracker_value,
            )
        else:
            with st.spinner(f"Running detect/track via {mode_desc} ({mode_label} on {device_label})…"):
                summary, error_message = runner(
                    ep_id,
                    "detect_track",
                    job_payload,
                    requested_device=device_value,
                    requested_detector=detector_value,
                    requested_tracker=tracker_value,
                )
    finally:
        if running_state_key:
            st.session_state[running_state_key] = False
        if active_job_key:
            st.session_state[active_job_key] = False
        if detect_flag_key:
            st.session_state[detect_flag_key] = False
    return current_local, summary, error_message


ep_id = helpers.get_ep_id()
if not ep_id:
    _prompt_for_episode()
ep_id = ep_id.strip()
canonical_ep_id = ep_id.lower()
if canonical_ep_id != ep_id:
    helpers.set_ep_id(canonical_ep_id)
    st.rerun()
ep_id = canonical_ep_id

# Legacy: Smart Suggestions navigation (now replaced by Improve Faces modal)
# Clear any stale navigation flags from previous sessions
_autorun_navigate_key = f"{ep_id}::autorun_navigate_to_suggestions"
st.session_state.pop(_autorun_navigate_key, None)

# Trigger Improve Faces modal if flag is set (after cluster completion)
if st.session_state.get(_session_improve_faces_state_key(ep_id, "trigger")):
    LOGGER.info("[IMPROVE_FACES] Trigger flag detected, starting Improve Faces modal")
    _start_improve_faces_ep_detail(ep_id, force=True)

# Render Improve Faces modal if active
_render_improve_faces_modal_ep_detail(ep_id)
# Skip heavy page rendering while modal is open to speed up YES/NO flows
if st.session_state.get(_session_improve_faces_state_key(ep_id, "active")):
    st.stop()

running_job_key = f"{ep_id}::pipeline_job_running"
if running_job_key not in st.session_state:
    st.session_state[running_job_key] = False
detect_running_key = _detect_job_state_key(ep_id)
if detect_running_key not in st.session_state:
    st.session_state[detect_running_key] = False
if _job_activity_key(ep_id) not in st.session_state:
    st.session_state[_job_activity_key(ep_id)] = False
job_running = bool(st.session_state.get(running_job_key))

# Hydrate logs for this episode on page load (local mode log persistence)
# This fetches any previously saved logs so they can be displayed without re-running jobs
helpers.hydrate_logs_for_episode(ep_id)

# Cache API responses with 10s TTL to reduce repeated requests
cache_key = time.time() // 10

try:
    details = _cached_episode_details(ep_id, cache_key)
except requests.HTTPError as exc:
    if exc.response is not None and exc.response.status_code == 404:
        _handle_missing_episode(ep_id)
    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/{ep_id}", exc))
    st.stop()
except requests.RequestException as exc:
    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/{ep_id}", exc))
    st.stop()

status_cache_key = _status_cache_key(ep_id)
status_ts_key = _status_timestamp_key(ep_id)
fetch_token_key = _status_fetch_token_key(ep_id)
mtimes_key = _status_mtimes_key(ep_id)
force_refresh_key = _status_force_refresh_key(ep_id)
force_refresh = bool(st.session_state.pop(force_refresh_key, False))
fetch_token = st.session_state.get(fetch_token_key, 0)
status_payload = st.session_state.get(status_cache_key)

# Attempt selection (run_id-scoped pipeline artifacts).
# Store selected attempt in session state; empty string means legacy/unscoped.
_active_run_id_key = f"{ep_id}::active_run_id"
_active_run_id_pending_key = f"{ep_id}::active_run_id_pending"
_autorun_run_id_key = f"{ep_id}::autorun_run_id"
_new_attempt_requested_key = f"{ep_id}::new_attempt_requested"
_attempt_init_key = f"{ep_id}::attempt_selector_initialized"
if _active_run_id_key not in st.session_state:
    st.session_state[_active_run_id_key] = ""
selected_attempt_raw = st.session_state.get(_active_run_id_key)
selected_attempt = selected_attempt_raw.strip() if isinstance(selected_attempt_raw, str) else ""
attempt_locked = False

# Streamlit doesn't allow mutating the session_state for an instantiated widget key.
# When other UI actions (e.g., Auto-Run) need to programmatically change the selected
# attempt, they write the desired run_id to this pending key and trigger a rerun.
_pending_attempt = st.session_state.pop(_active_run_id_pending_key, None)
if isinstance(_pending_attempt, str):
    selected_attempt = _pending_attempt.strip()
    st.session_state[_active_run_id_key] = selected_attempt
    st.session_state[_status_force_refresh_key(ep_id)] = True
    st.session_state[_attempt_init_key] = True
    attempt_locked = True

if st.session_state.pop(_new_attempt_requested_key, False):
    selected_attempt = _generate_attempt_run_id(ep_id)
    try:
        run_layout.run_root(ep_id, selected_attempt).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    st.session_state[_active_run_id_key] = selected_attempt
    st.session_state[_status_force_refresh_key(ep_id)] = True
    st.session_state[_attempt_init_key] = True
    attempt_locked = True

if not attempt_locked:
    try:
        qp_run_id_value = st.query_params.get("run_id")
    except Exception:
        qp_run_id_value = None
    qp_candidate = None
    if isinstance(qp_run_id_value, str):
        qp_candidate = qp_run_id_value.strip()
    elif isinstance(qp_run_id_value, list) and qp_run_id_value:
        qp_candidate = str(qp_run_id_value[0]).strip()
    if qp_candidate:
        try:
            normalized_qp_run_id = run_layout.normalize_run_id(qp_candidate)
        except ValueError:
            normalized_qp_run_id = None
        if normalized_qp_run_id and normalized_qp_run_id != selected_attempt:
            selected_attempt = normalized_qp_run_id
            st.session_state[_active_run_id_key] = selected_attempt
            st.session_state[_status_force_refresh_key(ep_id)] = True
            st.session_state[_attempt_init_key] = True

if not st.session_state.get(_attempt_init_key) and not selected_attempt:
    try:
        persisted_run_id = run_layout.read_active_run_id(ep_id)
    except Exception:
        persisted_run_id = None
    if isinstance(persisted_run_id, str) and persisted_run_id.strip():
        selected_attempt = persisted_run_id.strip()
        st.session_state[_active_run_id_key] = selected_attempt
        st.session_state[_attempt_init_key] = True
    else:
        # If no active_run.json exists yet, fall back to the most recently modified run directory.
        try:
            candidate_run_ids = run_layout.list_run_ids(ep_id)
        except Exception:
            candidate_run_ids = []
        latest_run_id: str | None = None
        latest_mtime = -1.0
        for candidate in candidate_run_ids:
            try:
                mtime = run_layout.run_root(ep_id, candidate).stat().st_mtime
            except (FileNotFoundError, OSError, ValueError):
                continue
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_run_id = candidate
        if latest_run_id:
            selected_attempt = latest_run_id
            st.session_state[_active_run_id_key] = selected_attempt
            st.session_state[_attempt_init_key] = True

# Normalize/validate selected attempt (invalid values fall back to legacy).
selected_attempt_run_id: str | None = None
if selected_attempt:
    try:
        selected_attempt_run_id = run_layout.normalize_run_id(selected_attempt)
    except ValueError:
        st.session_state[_active_run_id_key] = ""
        selected_attempt_run_id = None

# Keep run_id in the URL so cross-page navigation stays run-scoped.
try:
    current_run_qp = st.query_params.get("run_id", "") or ""
    desired_run_qp = selected_attempt_run_id or ""
    if current_run_qp != desired_run_qp:
        qp = st.query_params
        if desired_run_qp:
            qp["run_id"] = desired_run_qp
        else:
            try:
                del qp["run_id"]
            except Exception:
                pass
        st.query_params = qp
except Exception:
    # Query param sync is best-effort; do not block page rendering.
    pass

_manifests_root = get_path(ep_id, "detections").parent
_runs_root = _manifests_root / "runs"
_scoped_manifests_dir = (
    run_layout.run_root(ep_id, selected_attempt_run_id)
    if selected_attempt_run_id
    else _manifests_root
)
_scoped_markers_dir = _scoped_manifests_dir if selected_attempt_run_id else _runs_root
_track_metrics_path = _scoped_manifests_dir / "track_metrics.json"


# Batch file stat operations for efficiency - use try/except to avoid separate exists() calls
def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except (FileNotFoundError, OSError):
        return 0


current_mtimes = (
    selected_attempt_run_id or "legacy",
    _safe_mtime(_scoped_markers_dir / "detect_track.json"),
    _safe_mtime(_scoped_markers_dir / "faces_embed.json"),
    _safe_mtime(_scoped_markers_dir / "cluster.json"),
    _safe_mtime(_scoped_manifests_dir / "detections.jsonl"),
    _safe_mtime(_scoped_manifests_dir / "tracks.jsonl"),
    _safe_mtime(_scoped_manifests_dir / "faces.jsonl"),
    _safe_mtime(_track_metrics_path),
    _safe_mtime(_scoped_manifests_dir / "identities.json"),
)
cached_mtimes = st.session_state.get(mtimes_key)
should_refresh_status = force_refresh or _job_active(ep_id) or status_payload is None or cached_mtimes != current_mtimes
if should_refresh_status:
    fetch_token += 1
    st.session_state[fetch_token_key] = fetch_token
    # Include manifests in the cache key so status refreshes when identities/faces/tracks change.
    status_payload = _cached_episode_status(ep_id, fetch_token, current_mtimes, selected_attempt_run_id)
    st.session_state[status_cache_key] = status_payload
    st.session_state[status_ts_key] = time.time()
    st.session_state[mtimes_key] = current_mtimes
status_refreshed_at = st.session_state.get(status_ts_key)

if status_payload is None:
    detect_phase_status: Dict[str, Any] = {}
    faces_phase_status: Dict[str, Any] = {"status": "unknown"}
    cluster_phase_status: Dict[str, Any] = {"status": "unknown"}
else:
    detect_phase_status = status_payload.get("detect_track") or {}
    faces_phase_status = status_payload.get("faces_embed") or {}
    cluster_phase_status = status_payload.get("cluster") or {}

api_active_run_id = (status_payload or {}).get("active_run_id")
if not st.session_state.get(_attempt_init_key) and not selected_attempt:
    if isinstance(api_active_run_id, str) and api_active_run_id.strip():
        st.session_state[_active_run_id_key] = api_active_run_id.strip()
        st.session_state[_attempt_init_key] = True
        st.session_state[_status_force_refresh_key(ep_id)] = True
        st.rerun()
    st.session_state[_attempt_init_key] = True

prefixes = helpers.episode_artifact_prefixes(ep_id)
bucket_name = cfg.get("bucket")
manifests_dir = _scoped_manifests_dir
tracks_path = manifests_dir / "tracks.jsonl"
detections_path = manifests_dir / "detections.jsonl"
faces_path = manifests_dir / "faces.jsonl"
identities_path = manifests_dir / "identities.json"
analytics_dir = (
    manifests_dir / "analytics"
    if selected_attempt_run_id
    else helpers.DATA_ROOT / "analytics" / ep_id
)
screentime_json_path = analytics_dir / "screentime.json"
detect_job_defaults, detect_job_record = _load_job_defaults(ep_id, "detect_track")
faces_job_defaults, faces_job_record = _load_job_defaults(ep_id, "faces_embed")
cluster_job_defaults, cluster_job_record = _load_job_defaults(ep_id, "cluster")
_, screentime_job_record = _load_job_defaults(ep_id, "screen_time_analyze")
local_video_exists = bool(details["local"].get("exists"))
# Use cached video_meta (60s TTL) - no need for session state caching
video_meta = _cached_video_meta(ep_id) if local_video_exists else None


# =============================================================================
# System Status Check (A16-A17, A20: Storage backend and device validation)
# =============================================================================
def _render_system_status():
    """Show system configuration warnings at page load."""
    try:
        # Check storage backend status (cached for 60s)
        storage_status = _cached_storage_status()
        if storage_status and storage_status.get("status") == "success":
            validation = storage_status.get("validation")
            if validation:
                # Show warning if using fallback backend
                if validation.get("is_fallback"):
                    original = validation.get("original_backend", "unknown")
                    current = validation.get("backend", "local")
                    st.warning(
                        f"⚠️ **Storage Fallback Active**: STORAGE_BACKEND='{original}' is invalid. "
                        f"Using '{current}' instead. Fix configuration to avoid data loss."
                    )
                # Show any validation warnings
                for warning in validation.get("warnings") or []:
                    st.warning(f"⚠️ {warning}")

            # Check S3 credentials if using S3-based backend
            backend_type = storage_status.get("backend_type")
            if backend_type in ("s3", "minio", "hybrid"):
                s3_preflight = storage_status.get("s3_preflight")
                if s3_preflight and not s3_preflight.get("success"):
                    error = s3_preflight.get("error", "Unknown error")
                    st.error(f"🔴 **S3 Credentials Invalid**: {error}")

    except Exception as exc:
        LOGGER.debug("[system-status] Failed to fetch storage status: %s", exc)


# Show system status warnings at top of page
_render_system_status()


# =============================================================================
# Execution Mode Selector
# =============================================================================
# Store execution mode globally for this episode so all actions respect it
with st.expander("🔧 Execution Settings", expanded=False):
    exec_mode_col1, exec_mode_col2 = st.columns([2, 3])
    with exec_mode_col1:
        execution_mode = helpers.render_execution_mode_selector(ep_id, key_suffix="episode_detail")
    with exec_mode_col2:
        if execution_mode == "local":
            st.info("**Local Mode**: Jobs run synchronously in-process. No Redis/Celery needed.")
        else:
            st.info("**Redis Mode**: Jobs are queued via Celery for background processing.")


# =============================================================================
# Current Jobs Panel (Celery + subprocess + local background jobs)
# =============================================================================
with st.expander("⚙️ Current Jobs", expanded=False):
    try:
        all_jobs: list[dict] = []
        seen_job_ids: set[str] = set()  # Avoid duplicates across sources

        # Fetch LOCAL jobs FIRST (most common for local mode, highest priority)
        local_response = _cached_local_jobs(ep_id)
        local_jobs = local_response.get("jobs", []) if local_response else []
        for job in local_jobs:
            job_id = job.get("job_id", "unknown")
            if job_id in seen_job_ids:
                continue
            seen_job_ids.add(job_id)
            state = job.get("state", "unknown")
            # Local jobs in registry are always running/in_progress
            op = job.get("operation", "Pipeline Job")
            op_label = op.replace("_", " ").title() if op else "Pipeline Job"
            pid = job.get("pid")
            all_jobs.append({
                "job_id": job_id,
                "name": f"{op_label} (PID {pid})" if pid else op_label,
                "state": state,
                "worker": f"PID {pid}" if pid else "",
                "ep_id": job.get("ep_id"),
                "source": "local",
                "pid": pid,
            })

        # Fetch Celery jobs (cached for 5s)
        celery_response = _cached_celery_jobs()
        celery_jobs = celery_response.get("jobs", []) if celery_response else []
        for job in celery_jobs:
            job_id = job.get("job_id", "unknown")
            if job_id in seen_job_ids:
                continue
            # Only show jobs for current episode
            if job.get("ep_id") != ep_id:
                continue
            seen_job_ids.add(job_id)
            all_jobs.append({
                "job_id": job_id,
                "name": job.get("name", "Celery Task"),
                "state": job.get("state", "unknown"),
                "worker": job.get("worker", ""),
                "ep_id": job.get("ep_id"),
                "source": "celery",
            })

        # Fetch subprocess-based jobs (legacy, cached for 5s, filtered to current episode)
        jobs_response = _cached_episode_jobs(ep_id)
        subprocess_jobs = jobs_response.get("jobs", []) if jobs_response else []
        for job in subprocess_jobs:
            job_id = job.get("job_id", "unknown")
            if job_id in seen_job_ids:
                continue
            # Only show running/queued jobs, not completed ones
            state = job.get("state", "unknown")
            if state in ("running", "queued", "in_progress"):
                seen_job_ids.add(job_id)
                all_jobs.append({
                    "job_id": job_id,
                    "name": job.get("job_type", "Pipeline Job"),
                    "state": state,
                    "worker": "",
                    "ep_id": job.get("ep_id"),
                    "source": "subprocess",
                })

        if not all_jobs:
            st.info("No background jobs currently running.")
        else:
            st.caption(f"Found {len(all_jobs)} active job(s) for this episode")
            for job in all_jobs:
                job_id = job.get("job_id", "unknown")
                job_name = job.get("name", "unknown")
                job_state = job.get("state", "unknown")
                worker = job.get("worker", "")
                source = job.get("source", "unknown")

                # State badge
                if job_state in ("in_progress", "running"):
                    badge = "🔄"
                elif job_state == "queued":
                    badge = "⏳"
                elif job_state == "scheduled":
                    badge = "📅"
                else:
                    badge = "❓"

                # Display job card with cancel button
                col1, col2, col3 = st.columns([2.5, 1, 0.5])
                with col1:
                    st.markdown(f"**{badge} {job_name}**")
                    short_id = f"{job_id[:12]}..." if len(job_id) > 12 else job_id
                    st.caption(f"ID: `{short_id}` ({source})")
                with col2:
                    st.caption(f"State: {job_state}")
                    if worker:
                        st.caption(f"Worker: {worker.split('@')[-1]}")
                with col3:
                    # Cancel button
                    cancel_key = f"cancel_{job_id}"
                    if st.button("❌", key=cancel_key, help="Cancel this job"):
                        try:
                            if source == "local":
                                # Local jobs use the celery_jobs cancel endpoint
                                cancel_resp = helpers.api_post(f"/celery_jobs/{job_id}/cancel")
                            elif source == "celery":
                                cancel_resp = helpers.api_post(f"/celery_jobs/{job_id}/cancel")
                            else:
                                cancel_resp = helpers.api_post(f"/jobs/{job_id}/cancel")
                            if cancel_resp:
                                st.success(f"Cancelled job {job_id[:8]}...")
                                # Clear caches to reflect cancellation
                                _cached_local_jobs.clear()
                                st.rerun()
                            else:
                                st.error("Failed to cancel job")
                        except Exception as cancel_err:
                            st.error(f"Cancel failed: {cancel_err}")
                st.divider()
    except Exception as e:
        st.warning(f"Could not fetch job status: {e}")


with st.expander(f"Episode {ep_id}", expanded=False):
    st.write(f"Show `{details['show_slug']}` · Season {details['season_number']} Episode {details['episode_number']}")
    st.write(f"S3 v2 → `{details['s3']['v2_key']}` (exists={details['s3']['v2_exists']})")
    st.write(f"S3 v1 → `{details['s3']['v1_key']}` (exists={details['s3']['v1_exists']})")
    if not details["s3"]["v2_exists"] and details["s3"]["v1_exists"]:
        st.warning("Legacy v1 object detected; mirroring will use it until the v2 path is populated.")
    st.write(f"Local → {helpers.link_local(details['local']['path'])} (exists={details['local']['exists']})")
    if prefixes:
        st.caption(
            "S3 artifacts → "
            f"Frames {helpers.s3_uri(prefixes['frames'], bucket_name)} | "
            f"Crops {helpers.s3_uri(prefixes['crops'], bucket_name)} | "
            f"Manifests {helpers.s3_uri(prefixes['manifests'], bucket_name)}"
        )
    if tracks_path.exists():
        st.caption(f"Latest detector: {helpers.tracks_detector_label(ep_id)}")
        st.caption(f"Latest tracker: {helpers.tracks_tracker_label(ep_id)}")

    # S3 Sync Status Section
    artifact_status = _fetch_artifact_status(ep_id)
    if artifact_status:
        sync_status = artifact_status.get("sync_status", "unknown")
        st.markdown("---")
        st.markdown(f"**Artifact Sync Status:** {_render_sync_status_badge(sync_status)}")
        local_counts = artifact_status.get("local", {})
        s3_counts = artifact_status.get("s3", {})
        st.caption(_render_artifact_counts(local_counts, s3_counts))

        # Show sync button if artifacts need syncing
        if sync_status in ["pending", "partial"]:
            if st.button("🔄 Sync to S3", key=f"sync_artifacts_{ep_id}", help="Upload local artifacts to S3"):
                with st.spinner("Syncing artifacts to S3..."):
                    try:
                        sync_resp = helpers.api_post(f"/episodes/{ep_id}/sync_thumbnails_to_s3", timeout=300)
                        uploaded = sync_resp.get("uploaded_thumbs", 0) + sync_resp.get("uploaded_crops", 0)
                        if uploaded > 0:
                            st.success(f"Uploaded {uploaded} artifacts to S3")
                        else:
                            st.info("No new artifacts to upload")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sync failed: {e}")

manifest_state = _detect_track_manifests_ready(detections_path, tracks_path)

# Get status values from API
faces_status_value = str(faces_phase_status.get("status") or "missing").lower()
cluster_status_value = str(cluster_phase_status.get("status") or "missing").lower()
tracks_ready_flag = bool((status_payload or {}).get("tracks_ready"))

# FIX: Override stale "running" status if job just completed (completion flag set)
# This ensures the UI updates immediately when a job completes via log stream,
# without waiting for cache invalidation or API polling
if faces_status_value == "running" and st.session_state.get(f"{ep_id}::faces_embed_just_completed"):
    faces_status_value = "success"  # Override to success since job completed
    LOGGER.info("[STATUS-FIX] Faces: Override stale 'running' -> 'success' due to just_completed flag")
    helpers.invalidate_running_jobs_cache(ep_id)
if cluster_status_value == "running" and st.session_state.get(f"{ep_id}::cluster_just_completed"):
    cluster_status_value = "success"  # Override to success since job completed
    LOGGER.info("[STATUS-FIX] Cluster: Override stale 'running' -> 'success' due to just_completed flag")
    helpers.invalidate_running_jobs_cache(ep_id)
detect_job_state = (detect_job_record or {}).get("state")
# FIX: Override stale "running" state if job just completed (completion flag set)
# This ensures the UI updates immediately when a job completes via log stream,
# without waiting for cache invalidation or API polling
if detect_job_state == "running" and st.session_state.get(f"{ep_id}::detect_track_just_completed"):
    detect_job_state = None  # Clear running state to allow status to reflect completion
    LOGGER.info("[STATUS-FIX] Detected stale 'running' state with just_completed flag, clearing job_state")
    # Invalidate cache to ensure next check uses fresh data
    helpers.invalidate_running_jobs_cache(ep_id)
detect_status_value, tracks_ready, using_manifest_fallback, tracks_only_fallback = _compute_detect_track_effective_status(
    detect_phase_status,
    manifest_ready=manifest_state["manifest_ready"],
    tracks_only_fallback=manifest_state["tracks_only_fallback"],
    tracks_ready_flag=tracks_ready_flag,
    job_state=detect_job_state,
)
if cluster_status_value in {"missing", "unknown"}:
    identities_count_manifest = None
    cluster_metrics_block: dict[str, Any] | None = None
    artifact_mtime = 0.0
    if identities_path.exists():
        try:
            payload = json.loads(identities_path.read_text(encoding="utf-8"))
            identities_list = payload.get("identities") if isinstance(payload, dict) else None
            if isinstance(identities_list, list):
                identities_count_manifest = len(identities_list)
        except (json.JSONDecodeError, OSError, KeyError):
            # File may be corrupted or in unexpected format - silently skip
            pass
        try:
            artifact_mtime = identities_path.stat().st_mtime
        except OSError:
            artifact_mtime = 0.0
    if _track_metrics_path.exists():
        try:
            metrics_data = json.loads(_track_metrics_path.read_text(encoding="utf-8"))
            if isinstance(metrics_data, dict):
                block = metrics_data.get("cluster_metrics")
                cluster_metrics_block = block if isinstance(block, dict) else None
        except (json.JSONDecodeError, OSError, KeyError):
            # File may be corrupted or in unexpected format - silently skip
            cluster_metrics_block = None
        try:
            artifact_mtime = max(artifact_mtime, _track_metrics_path.stat().st_mtime)
        except OSError:
            pass
    if identities_count_manifest is None and isinstance(cluster_metrics_block, dict):
        identities_count_manifest = helpers.coerce_int(
            cluster_metrics_block.get("total_clusters_after") or cluster_metrics_block.get("total_clusters")
        )
    if identities_count_manifest is not None or cluster_metrics_block:
        cluster_phase_status = dict(cluster_phase_status)
        if isinstance(cluster_metrics_block, dict):
            cluster_phase_status.setdefault("singleton_stats", cluster_metrics_block.get("singleton_stats"))
            cluster_phase_status.setdefault("singleton_merge", cluster_metrics_block.get("singleton_merge"))
            cluster_phase_status.setdefault("singleton_fraction_before", cluster_metrics_block.get("singleton_fraction_before"))
            cluster_phase_status.setdefault("singleton_fraction_after", cluster_metrics_block.get("singleton_fraction_after"))
            cluster_phase_status.setdefault("total_clusters_before", cluster_metrics_block.get("total_clusters_before"))
            cluster_phase_status.setdefault("total_clusters_after", cluster_metrics_block.get("total_clusters_after"))
        cluster_phase_status["status"] = "success"
        cluster_phase_status["identities"] = identities_count_manifest
        cluster_phase_status["source"] = cluster_phase_status.get("source") or "manifest_fallback"
        # Read marker file for timestamps and device info FIRST (most authoritative source)
        _cluster_marker_path = _scoped_markers_dir / "cluster.json"
        if _cluster_marker_path.exists():
            try:
                _marker_data = json.loads(_cluster_marker_path.read_text(encoding="utf-8"))
                if isinstance(_marker_data, dict):
                    if not cluster_phase_status.get("started_at"):
                        cluster_phase_status["started_at"] = _marker_data.get("started_at")
                    if not cluster_phase_status.get("finished_at"):
                        cluster_phase_status["finished_at"] = _marker_data.get("finished_at")
                    if not cluster_phase_status.get("device"):
                        cluster_phase_status["device"] = _marker_data.get("device")
            except (json.JSONDecodeError, OSError):
                pass
        # Fallback to artifact mtime for finished_at if marker didn't have it
        if not cluster_phase_status.get("finished_at") and artifact_mtime:
            cluster_phase_status["finished_at"] = (
                datetime.fromtimestamp(artifact_mtime, tz=timezone.utc).replace(microsecond=0).isoformat() + "Z"
            )
        # Compute runtime_sec if timestamps are available
        if not cluster_phase_status.get("runtime_sec"):
            _runtime = _runtime_from_iso(
                cluster_phase_status.get("started_at"),
                cluster_phase_status.get("finished_at"),
            )
            if _runtime is not None:
                cluster_phase_status["runtime_sec"] = _runtime
        cluster_status_value = "success"
jpeg_state = helpers.coerce_int(detect_phase_status.get("jpeg_quality"))
device_state = detect_phase_status.get("device")
requested_device_state = detect_phase_status.get("requested_device")
resolved_device_state = detect_phase_status.get("resolved_device")
screentime_status_value = "missing"
screentime_error = None
screentime_started_at = None
screentime_finished_at = None
if screentime_job_record:
    job_state = str(screentime_job_record.get("state") or "").lower()
    screentime_error = screentime_job_record.get("error")
    screentime_started_at = screentime_job_record.get("started_at")
    screentime_finished_at = screentime_job_record.get("ended_at")
    if job_state == "running":
        screentime_status_value = "running"
    elif job_state in {"failed", "error"}:
        screentime_status_value = "error"
    elif job_state == "succeeded":
        screentime_status_value = "success"
if screentime_status_value == "missing" and screentime_json_path.exists():
    screentime_status_value = "success"
    if screentime_finished_at is None:
        screentime_finished_at = (
            datetime.fromtimestamp(screentime_json_path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat() + "Z"
        )
screentime_runtime = _format_runtime(_runtime_from_iso(screentime_started_at, screentime_finished_at))
status_running = (
    detect_status_value == "running"
    or faces_status_value == "running"
    or cluster_status_value == "running"
    or str(detect_job_state or "").lower() == "running"
    or screentime_status_value == "running"
)
if status_running:
    _set_job_active(ep_id, True)
elif not job_running:
    _set_job_active(ep_id, False)

# Other status values
faces_count_value = helpers.coerce_int(faces_phase_status.get("faces"))
identities_count_value = helpers.coerce_int(cluster_phase_status.get("identities"))
faces_manifest_count = None
faces_ready_state = False
faces_manifest_fallback = bool(faces_phase_status.get("faces_manifest_fallback"))
faces_manifest_exists = faces_path.exists()
if faces_manifest_exists:
    faces_manifest_count = _count_manifest_rows(faces_path) or 0
if faces_status_value == "success":
    faces_ready_state = True
elif faces_status_value in {"missing", "unknown"} and faces_manifest_exists:
    # Manifest exists but API reports missing/unknown - use manifest fallback
    faces_ready_state = True
    faces_manifest_fallback = True
# Note: "stale" status is NOT treated as ready - it needs to be re-run
if faces_count_value is None and faces_manifest_count is not None:
    faces_count_value = faces_manifest_count

# If detect status is missing but manifests are present, synthesize a summary so the UI still shows completion.
if not detect_phase_status and manifest_state["manifest_ready"]:
    detect_phase_status = {
        "status": "success",
        "detections": _count_manifest_rows(detections_path) or 0,
        "tracks": _count_manifest_rows(tracks_path) or 0,
        "finished_at": None,
    }
    detect_status_value = "success"
    using_manifest_fallback = True

# Add pipeline state indicators (even if status API is temporarily unavailable)
with st.expander("Pipeline Status", expanded=False):
    if st.button("Refresh status", key="episode_status_refresh", use_container_width=True):
        now = time.time()
        last_click = float(st.session_state.get(_refresh_click_key(ep_id), 0.0))
        if now - last_click < 1.0:
            st.caption("Please wait ≥1s between refreshes.")
        else:
            st.session_state[_refresh_click_key(ep_id)] = now
            st.session_state[_status_force_refresh_key(ep_id)] = True
            st.rerun()
    if status_refreshed_at:
        refreshed_dt = datetime.fromtimestamp(status_refreshed_at, tz=timezone.utc).astimezone(EST_TZ)
        refreshed_label = refreshed_dt.strftime("%Y-%m-%d %H:%M:%S ET")
        st.caption(f"Status refreshed at {refreshed_label}")
    else:
        st.caption("Status will refresh when a job starts or you press refresh.")

    attempt_col1, attempt_col2 = st.columns([3, 1])
    with attempt_col1:
        available_run_ids = run_layout.list_run_ids(ep_id)
        current_value = st.session_state.get(_active_run_id_key)
        current_str = current_value.strip() if isinstance(current_value, str) else ""
        attempt_options = [""] + available_run_ids
        if current_str and current_str not in attempt_options:
            attempt_options.append(current_str)

        def _format_attempt(value: str) -> str:
            return "Legacy (no run_id)" if not value else value

        st.selectbox(
            "Attempt (run_id)",
            attempt_options,
            key=_active_run_id_key,
            format_func=_format_attempt,
            help="Scopes status/artifacts and any new Detect/Faces/Cluster runs to this attempt.",
            disabled=job_running,
        )
    with attempt_col2:
        if st.button(
            "New attempt",
            key=f"{ep_id}::new_attempt_btn",
            use_container_width=True,
            disabled=job_running,
        ):
            st.session_state[_new_attempt_requested_key] = True
            st.rerun()

    selected_attempt_label = st.session_state.get(_active_run_id_key)
    selected_attempt_label = selected_attempt_label.strip() if isinstance(selected_attempt_label, str) else ""
    if selected_attempt_label:
        st.caption(f"Selected attempt: `{selected_attempt_label}`")
    else:
        st.caption("Selected attempt: legacy (no run_id)")
    if isinstance(api_active_run_id, str) and api_active_run_id.strip():
        if api_active_run_id.strip() != selected_attempt_label:
            st.caption(f"API active_run_id: `{api_active_run_id.strip()}`")

    if ep_id == "rhoslc-s06e11":
        st.divider()
        st.caption("Debug tools (rhoslc-s06e11)")
        clear_confirm_key = f"{ep_id}::clear_attempts_confirm"
        clear_confirmed = st.checkbox(
            "Confirm: delete ALL run-scoped attempts for this episode (manifests + frames).",
            key=clear_confirm_key,
            disabled=job_running,
        )
        if st.button(
            "🧹 Clear all previous attempts",
            key=f"{ep_id}::clear_attempts_btn",
            use_container_width=True,
            disabled=job_running or not clear_confirmed,
        ):
            with st.spinner("Deleting run-scoped attempt artifacts..."):
                try:
                    runs_dir = run_layout.runs_root(ep_id)
                    frames_runs_dir = get_path(ep_id, "frames_root") / "runs"

                    removed_dirs = 0
                    removed_files = 0

                    if runs_dir.exists():
                        for child in list(runs_dir.iterdir()):
                            try:
                                if child.is_dir():
                                    shutil.rmtree(child)
                                    removed_dirs += 1
                                else:
                                    child.unlink()
                                    removed_files += 1
                            except Exception as exc:
                                LOGGER.warning("Failed to remove %s: %s", child, exc)

                    if frames_runs_dir.exists():
                        shutil.rmtree(frames_runs_dir)

                    runs_dir.mkdir(parents=True, exist_ok=True)

                    # Reset attempt selection + status caches
                    st.session_state[_active_run_id_pending_key] = ""
                    st.session_state.pop(_autorun_run_id_key, None)
                    st.session_state[_attempt_init_key] = True
                    st.session_state[_status_force_refresh_key(ep_id)] = True
                    st.session_state.pop(status_cache_key, None)
                    st.session_state.pop(mtimes_key, None)

                    try:
                        qp = st.query_params
                        try:
                            del qp["run_id"]
                        except Exception:
                            pass
                        st.query_params = qp
                    except Exception:
                        pass

                    helpers.invalidate_running_jobs_cache(ep_id)
                    st.success(
                        f"Cleared {removed_dirs} run directories and {removed_files} run marker files for `{ep_id}`."
                    )
                    time.sleep(0.25)
                    st.rerun()
                except (RerunException, StopException):
                    # st.rerun()/st.stop() raise control-flow exceptions; allow Streamlit to handle them.
                    raise
                except Exception as exc:
                    st.error("Failed to clear previous attempts.")
                    st.exception(exc)

    nav_disabled = not bool(selected_attempt_run_id)
    nav_help = "Select a run-scoped attempt (run_id) above." if nav_disabled else None
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button(
            "Faces Review",
            key=f"{ep_id}::{selected_attempt_run_id or 'legacy'}::nav_faces_review",
            use_container_width=True,
            disabled=nav_disabled,
            help=nav_help,
        ):
            try:
                qp = st.query_params
                qp["ep_id"] = ep_id
                if selected_attempt_run_id:
                    qp["run_id"] = selected_attempt_run_id
                else:
                    try:
                        del qp["run_id"]
                    except Exception:
                        pass
                st.query_params = qp
            except Exception:
                pass
            st.switch_page("pages/3_Faces_Review.py")
    with nav_col2:
        if st.button(
            "Smart Suggestions",
            key=f"{ep_id}::{selected_attempt_run_id or 'legacy'}::nav_smart_suggestions",
            use_container_width=True,
            disabled=nav_disabled,
            help=nav_help,
        ):
            try:
                qp = st.query_params
                qp["ep_id"] = ep_id
                if selected_attempt_run_id:
                    qp["run_id"] = selected_attempt_run_id
                else:
                    try:
                        del qp["run_id"]
                    except Exception:
                        pass
                st.query_params = qp
            except Exception:
                pass
            st.switch_page("pages/3_Smart_Suggestions.py")
    coreml_available = status_payload.get("coreml_available") if status_payload else None
    if coreml_available is False and helpers.is_apple_silicon():
        st.warning(
            "⚠️ CoreML acceleration isn't available on this host. Install `onnxruntime-coreml` to avoid CPU-only runs."
        )
    col1, col2, col3 = st.columns(3)

    with col1:
        detect_params: list[str] = []
        stride_state = helpers.coerce_int(detect_phase_status.get("stride"))
        if stride_state:
            detect_params.append(f"stride={stride_state}")
        det_thresh_state = helpers.coerce_float(detect_phase_status.get("det_thresh"))
        if det_thresh_state is not None:
            detect_params.append(f"det_thresh={det_thresh_state:.2f}")
        max_gap_state = helpers.coerce_int(detect_phase_status.get("max_gap"))
        if max_gap_state is not None:
            detect_params.append(f"max_gap={max_gap_state}")
        scene_thresh_state = helpers.coerce_float(detect_phase_status.get("scene_threshold"))
        if scene_thresh_state is not None:
            detect_params.append(f"scene={scene_thresh_state:.2f}")
        track_high_state = helpers.coerce_float(detect_phase_status.get("track_high_thresh"))
        if track_high_state is not None:
            detect_params.append(f"track_high={track_high_state:.2f}")
        new_track_state = helpers.coerce_float(detect_phase_status.get("new_track_thresh"))
        if new_track_state is not None:
            detect_params.append(f"new_track={new_track_state:.2f}")
        save_frames_state = detect_phase_status.get("save_frames")
        if save_frames_state is not None:
            detect_params.append(f"save_frames={'on' if save_frames_state else 'off'}")
        save_crops_state = detect_phase_status.get("save_crops")
        if save_crops_state is not None:
            detect_params.append(f"save_crops={'on' if save_crops_state else 'off'}")
        if jpeg_state:
            detect_params.append(f"jpeg={jpeg_state}")
        detect_runtime = _format_runtime(detect_phase_status.get("runtime_sec"))
        if requested_device_state and requested_device_state != device_state:
            detect_params.append(f"requested={helpers.device_label_from_value(requested_device_state)}")
        device_label = helpers.device_label_from_value(
            resolved_device_state or device_state or requested_device_state or helpers.DEFAULT_DEVICE
        )
        if device_label:
            detect_params.append(f"device={device_label}")
        if detect_status_value == "success":
            runtime_label = detect_runtime or "n/a"
            st.success(f"✅ **Detect/Track**: Complete (Runtime: {runtime_label})")
            det = detect_phase_status.get("detector") or "--"
            trk = detect_phase_status.get("tracker") or "--"
            st.caption(f"{det} + {trk}")
            detections = detect_phase_status.get("detections")
            tracks = detect_phase_status.get("tracks")
            st.caption(f"{(detections or 0):,} detections, {(tracks or 0):,} tracks")
            ratio_value = helpers.coerce_float(
                detect_phase_status.get("track_to_detection_ratio") or detect_phase_status.get("track_ratio")
            )
            if ratio_value is not None:
                st.caption(f"Tracks / detections: {ratio_value:.2f}")
                if ratio_value < 0.1:
                    st.caption(
                        "⚠️ Track-to-detection ratio < 0.10. Consider lowering ByteTrack thresholds or rerunning detect/track."
                    )
            # Show manifest-fallback caption when status was inferred from manifests
            if using_manifest_fallback or detect_phase_status.get("metadata_missing"):
                st.warning(
                    "⚠️ Detect/Track details inferred from manifests (metadata missing). "
                    "Detector/tracker and runtime may be inaccurate; rerun detect/track for fresh metadata."
                )
            if tracks_only_fallback:
                st.warning("⚠️ Tracks exist but detections are missing. Rerun detect/track to regenerate detections.")
        elif detect_status_value == "running":
            st.info("⏳ **Detect/Track**: Running")
            if detect_job_record and detect_job_record.get("started_at"):
                st.caption(f"Started at {detect_job_record['started_at']}")
            st.caption("Live progress appears in the log panel below.")
        elif detect_status_value == "stale":
            st.warning("⚠️ **Detect/Track**: Status stale (manifests missing)")
            st.caption("Rerun Detect/Track Faces to rebuild detections/tracks for this episode.")
        elif detect_status_value == "partial":
            st.warning("⚠️ **Detect/Track**: Detections present but tracks missing")
            st.caption("Rerun detect/track to rebuild tracks.")
        elif detect_status_value == "missing":
            st.info("⏳ **Detect/Track**: Not started")
            st.caption("Run detect/track first.")
        else:
            st.error(f"⚠️ **Detect/Track**: {detect_status_value.title()}")
            if detect_phase_status.get("error"):
                st.caption(detect_phase_status["error"])
        if detect_params:
            st.caption("Params: " + ", ".join(detect_params))
        if tracks_only_fallback:
            st.warning(
                "⚠️ Tracks manifest is present but detections are missing. Rerun Detect/Track to regenerate detections "
                "before continuing."
            )
        if jpeg_state:
            st.caption(f"JPEG quality: {jpeg_state}")
        _render_device_summary(requested_device_state, resolved_device_state or device_state)
        finished = _format_timestamp(detect_phase_status.get("finished_at"))
        if finished:
            st.caption(f"Last run: {finished}")
        # Show video duration and run duration on separate lines
        video_duration = _format_video_duration(detect_phase_status.get("video_duration_sec"))
        if video_duration:
            st.caption(f"Video Duration: {video_duration}")
        if detect_runtime:
            st.caption(f"Run Duration: {detect_runtime}")
        elif detect_status_value == "success":
            st.caption("Run Duration: n/a")

    with col2:
        faces_params: list[str] = []
        faces_device_state = faces_phase_status.get("device")
        faces_device_request = faces_phase_status.get("requested_device")
        faces_resolved_state = faces_phase_status.get("resolved_device")
        faces_runtime = _format_runtime(faces_phase_status.get("runtime_sec"))
        faces_job_state = str((faces_job_record or {}).get("state") or "").lower()
        faces_error_msg = faces_phase_status.get("error") or (faces_job_record or {}).get("error")
        if faces_device_request and faces_device_request != faces_device_state:
            faces_params.append(f"requested={helpers.device_label_from_value(faces_device_request)}")
        if faces_device_state:
            faces_params.append(f"device={helpers.device_label_from_value(faces_device_state)}")
        save_frames_state = faces_phase_status.get("save_frames")
        if save_frames_state is not None:
            faces_params.append(f"save_frames={'on' if save_frames_state else 'off'}")
        save_crops_state = faces_phase_status.get("save_crops")
        if save_crops_state is not None:
            faces_params.append(f"save_crops={'on' if save_crops_state else 'off'}")
        spacing_state = helpers.coerce_int(faces_phase_status.get("min_frames_between_crops"))
        if spacing_state:
            faces_params.append(f"spacing={spacing_state}")
        thumb_size_state = helpers.coerce_int(faces_phase_status.get("thumb_size"))
        if thumb_size_state:
            faces_params.append(f"thumb={thumb_size_state}px")
        faces_jpeg_state = helpers.coerce_int(faces_phase_status.get("jpeg_quality"))
        if faces_jpeg_state:
            faces_params.append(f"jpeg={faces_jpeg_state}")
        if faces_status_value == "stale":
            # Stale: detect/track was rerun after this faces harvest
            face_count_label = helpers.format_count(faces_count_value) or "0"
            st.warning(f"⚠️ **Faces Harvest**: Outdated ({face_count_label} faces)")
            st.caption("Detect/Track was rerun. Rerun **Faces Harvest** to rebuild embeddings for the new tracks.")
        elif faces_ready_state:
            runtime_label = faces_runtime or "n/a"
            st.success(f"✅ **Faces Harvest**: Complete (Runtime: {runtime_label})")
            face_count_label = helpers.format_count(faces_count_value) or "0"
            st.caption(f"Faces: {face_count_label} (harvest completed)")
            if faces_manifest_fallback:
                st.caption("ℹ️ Using manifest fallback; status may be stale.")
        elif faces_status_value == "success":
            st.warning("⚠️ **Faces Harvest**: Manifest unavailable locally")
            st.caption("Faces completed on the backend, but faces.jsonl has not been mirrored locally yet.")
        elif faces_job_state == "failed":
            st.error("⚠️ **Faces Harvest**: Failed")
            if faces_error_msg:
                st.caption(faces_error_msg)
        elif faces_status_value not in {"missing", "unknown"}:
            st.warning(f"⚠️ **Faces Harvest**: {faces_status_value.title()}")
            if faces_error_msg:
                st.caption(faces_error_msg)
        elif tracks_ready:
            st.info("⏳ **Faces Harvest**: Ready to run")
            st.caption("Click 'Run Faces Harvest' below.")
        else:
            st.info("⏳ **Faces Harvest**: Waiting for tracks")
            st.caption("Complete detect/track first.")
        if faces_params:
            st.caption("Params: " + ", ".join(faces_params))
        _render_device_summary(faces_device_request, faces_resolved_state or faces_device_state)
        finished = _format_timestamp(faces_phase_status.get("finished_at"))
        if finished:
            st.caption(f"Last run: {finished}")
        if faces_runtime:
            st.caption(f"Run Duration: {faces_runtime}")
        elif faces_status_value == "success":
            st.caption("Run Duration: n/a")

    with col3:
        cluster_params: list[str] = []
        cluster_device_state = cluster_phase_status.get("device")
        cluster_device_request = cluster_phase_status.get("requested_device")
        cluster_resolved_state = cluster_phase_status.get("resolved_device")
        cluster_runtime = _format_runtime(cluster_phase_status.get("runtime_sec"))
        cluster_job_state = str((cluster_job_record or {}).get("state") or "").lower()
        cluster_error_msg = cluster_phase_status.get("error") or (cluster_job_record or {}).get("error")
        if cluster_device_request and cluster_device_request != cluster_device_state:
            cluster_params.append(f"requested={helpers.device_label_from_value(cluster_device_request)}")
        if cluster_device_state:
            cluster_params.append(f"device={helpers.device_label_from_value(cluster_device_state)}")
        cluster_thresh_state = helpers.coerce_float(cluster_phase_status.get("cluster_thresh"))
        if cluster_thresh_state is not None:
            cluster_params.append(f"thresh={cluster_thresh_state:.2f}")
        min_cluster_state = helpers.coerce_int(cluster_phase_status.get("min_cluster_size"))
        if min_cluster_state is not None:
            cluster_params.append(f"min_cluster={min_cluster_state}")
        identities_label = helpers.format_count(identities_count_value) or "0"
        if cluster_status_value == "stale":
            # Stale: detect/track or faces was rerun after this clustering
            st.warning(f"⚠️ **Cluster**: Outdated ({identities_label} identities)")
            st.caption("Detect/Track was rerun. Rerun **Faces Harvest** first, then **Cluster** to rebuild identities.")
        elif cluster_status_value == "success":
            runtime_label = cluster_runtime or "n/a"
            st.success(f"✅ **Cluster**: Complete (Runtime: {runtime_label})")
            st.caption(f"Identities: {identities_label}")
            if identities_count_value == 0:
                st.warning("Cluster finished but found 0 identities. Rerun after checking detect/track and faces outputs.")
        elif cluster_status_value == "running":
            st.info("⏳ **Cluster**: Running")
            started = _format_timestamp(cluster_phase_status.get("started_at"))
            if started:
                st.caption(f"Started at {started}")
            st.caption("Live progress appears in the log panel below.")
        elif cluster_job_state == "failed":
            st.error("⚠️ **Cluster**: Failed")
            if cluster_error_msg:
                st.caption(cluster_error_msg)
        elif cluster_status_value not in {"missing", "unknown"}:
            st.warning(f"⚠️ **Cluster**: {cluster_status_value.title()}")
            if cluster_error_msg:
                st.caption(cluster_error_msg)
        elif faces_ready_state:
            if (faces_count_value or 0) == 0:
                st.info("ℹ️ **Cluster**: No faces to cluster")
                st.caption("Faces harvest finished with 0 faces → expect 0 identities.")
            else:
                st.info("⏳ **Cluster**: Ready to run")
                st.caption("Click 'Run Cluster' below.")
        else:
            st.info("⏳ **Cluster**: Waiting for faces")
            st.caption("Complete faces harvest first.")
        if cluster_params:
            st.caption("Params: " + ", ".join(cluster_params))
        merge_block = cluster_phase_status.get("singleton_merge") or {}
        singleton_stats = cluster_phase_status.get("singleton_stats") or merge_block.get("singleton_stats") or {}
        if not isinstance(singleton_stats, dict):
            singleton_stats = {}
        before_block = singleton_stats.get("before") or {}
        after_block = singleton_stats.get("after") or {}
        if not isinstance(before_block, dict):
            before_block = {}
        if not isinstance(after_block, dict):
            after_block = {}
        before_frac = helpers.coerce_float(before_block.get("singleton_fraction"))
        if before_frac is None:
            before_frac = helpers.coerce_float(cluster_phase_status.get("singleton_fraction_before"))
        after_frac = helpers.coerce_float(after_block.get("singleton_fraction"))
        if after_frac is None:
            after_frac = helpers.coerce_float(cluster_phase_status.get("singleton_fraction_after"))
        threshold = helpers.coerce_float(singleton_stats.get("threshold"))
        if threshold is None:
            threshold = helpers.coerce_float(cluster_phase_status.get("singleton_merge_threshold"))
        clusters_before = helpers.coerce_int(before_block.get("cluster_count"))
        if clusters_before is None:
            clusters_before = helpers.coerce_int(cluster_phase_status.get("total_clusters_before"))
        clusters_after = helpers.coerce_int(after_block.get("cluster_count"))
        if clusters_after is None:
            clusters_after = helpers.coerce_int(cluster_phase_status.get("total_clusters_after"))
        merge_count = helpers.coerce_int(after_block.get("merge_count"))
        if merge_count is None:
            merge_count = helpers.coerce_int(merge_block.get("num_singleton_merges"))
        sim_thresh = merge_block.get("similarity_thresh") or merge_block.get("secondary_cluster_thresh")
        neighbor_top_k = merge_block.get("neighbor_top_k") or merge_block.get("max_pairs_per_track")
        merge_enabled = merge_block.get("enabled") if merge_block else False
        primary_frac = after_frac if after_frac is not None else before_frac
        has_metrics = any(val is not None for val in [before_frac, after_frac, clusters_before, clusters_after, merge_count])
        use_after_merge_label = bool(merge_enabled and after_frac is not None)
        if has_metrics:
            lines: list[str] = []
            if before_frac is not None and after_frac is not None:
                thresh_label = f"{threshold:.2f}" if threshold is not None else "?"
                lines.append(f"Singletons: {before_frac:.2f} → {after_frac:.2f} (threshold {thresh_label})")
            elif before_frac is not None:
                thresh_label = f" (threshold {threshold:.2f})" if threshold is not None else ""
                lines.append(f"Singletons: {before_frac:.2f}{thresh_label}")
            elif after_frac is not None:
                thresh_label = f" (threshold {threshold:.2f})" if threshold is not None else ""
                lines.append(f"Singletons (post-merge): {after_frac:.2f}{thresh_label}")
            if clusters_before is not None and clusters_after is not None:
                before_clusters = helpers.format_count(clusters_before) or str(clusters_before)
                after_clusters = helpers.format_count(clusters_after) or str(clusters_after)
                lines.append(f"Clusters: {before_clusters} → {after_clusters}")
            elif clusters_after is not None:
                after_clusters = helpers.format_count(clusters_after) or str(clusters_after)
                lines.append(f"Clusters: {after_clusters}")
            elif clusters_before is not None:
                before_clusters = helpers.format_count(clusters_before) or str(clusters_before)
                lines.append(f"Clusters: {before_clusters}")
            if merge_enabled and merge_count is not None:
                sim_label = f"{sim_thresh:.2f}" if isinstance(sim_thresh, (int, float)) else sim_thresh or "?"
                neighbor_label = neighbor_top_k if neighbor_top_k is not None else "?"
                lines.append(f"Merge: {merge_count} pairs (sim ≥ {sim_label}, top_k={neighbor_label})")
            for entry in lines:
                st.caption(entry)
            if threshold is not None and primary_frac is not None:
                high_label = "🚧 High singleton fraction after merge" if use_after_merge_label else "🚧 High singleton fraction"
                ok_label = (
                    "✅ Singletons reduced below threshold after merge"
                    if use_after_merge_label
                    else "✅ Singleton fraction below threshold"
                )
                if primary_frac > threshold:
                    st.warning(high_label)
                elif merge_enabled:
                    st.success(ok_label)
        else:
            if cluster_status_value == "success":
                st.caption("Singleton metrics unavailable for this run.")
            else:
                st.caption("Singleton metrics not available until clustering completes.")
        _render_device_summary(cluster_device_request, cluster_resolved_state or cluster_device_state)
        finished = _format_timestamp(cluster_phase_status.get("finished_at"))
        if finished:
            st.caption(f"Last run: {finished}")
        if cluster_runtime:
            st.caption(f"Run Duration: {cluster_runtime}")
        elif cluster_status_value == "success":
            st.caption("Run Duration: n/a")


detector_override = st.session_state.pop("episode_detail_detector_override", None)
tracker_override = st.session_state.pop("episode_detail_tracker_override", None)
device_override = st.session_state.pop("episode_detail_device_override", None)
autorun_detect = st.session_state.pop("episode_detail_detect_autorun_flag", False)

detect_detector_value = _choose_value(
    detector_override,
    detect_job_defaults.get("detector"),
    detect_phase_status.get("detector"),
    helpers.tracks_detector_value(ep_id),
    fallback=helpers.DEFAULT_DETECTOR,
)
detect_tracker_value = _choose_value(
    tracker_override,
    detect_job_defaults.get("tracker"),
    detect_phase_status.get("tracker"),
    helpers.tracks_tracker_value(ep_id),
    fallback=helpers.DEFAULT_TRACKER,
)
detect_device_default_value = _choose_value(
    device_override,
    detect_job_defaults.get("requested_device"),
    detect_phase_status.get("requested_device"),
    detect_job_defaults.get("device"),
    detect_phase_status.get("device"),
    fallback=helpers.DEFAULT_DEVICE,
)
detect_device_label_default = helpers.device_label_from_value(detect_device_default_value)
detect_device_label_default = _resolved_device_label(detect_device_label_default)
detect_detector_label = helpers.detector_label_from_value(detect_detector_value)
detect_tracker_label = helpers.tracker_label_from_value(detect_tracker_value)

faces_device_default_value = _choose_value(
    faces_job_defaults.get("requested_device"),
    faces_job_defaults.get("device"),
    faces_job_defaults.get("embed_device"),
    faces_phase_status.get("requested_device"),
    faces_phase_status.get("device"),
    faces_phase_status.get("embed_device"),
    fallback=detect_device_default_value,
)
faces_device_label_default = helpers.device_label_from_value(faces_device_default_value)
faces_device_label_default = _resolved_device_label(faces_device_label_default)
faces_save_frames_default = faces_job_defaults.get("save_frames")
if faces_save_frames_default is None:
    faces_save_frames_default = False
faces_save_crops_default = faces_job_defaults.get("save_crops")
if faces_save_crops_default is None:
    faces_save_crops_default = True
faces_jpeg_quality_default = helpers.coerce_int(faces_job_defaults.get("jpeg_quality")) or JPEG_DEFAULT
faces_min_frames_between_crops_default = helpers.coerce_int(
    faces_job_defaults.get("min_frames_between_crops")
)
if faces_min_frames_between_crops_default is None:
    faces_min_frames_between_crops_default = helpers.coerce_int(
        faces_phase_status.get("min_frames_between_crops")
    )
if faces_min_frames_between_crops_default is None:
    faces_min_frames_between_crops_default = _default_faces_min_frames_between_crops(video_meta)

cluster_device_default_value = _choose_value(
    cluster_phase_status.get("requested_device"),
    cluster_job_defaults.get("device"),
    cluster_phase_status.get("device"),
    fallback=faces_device_default_value,
)
cluster_device_label_default = helpers.device_label_from_value(cluster_device_default_value)
cluster_device_label_default = _resolved_device_label(cluster_device_label_default)
# Always use the configured default (0.58) - ignore cached values from previous runs
cluster_thresh_default = helpers.DEFAULT_CLUSTER_SIMILARITY
min_cluster_size_default = helpers.coerce_int(cluster_job_defaults.get("min_cluster_size"))
if min_cluster_size_default is None:
    min_cluster_size_default = helpers.coerce_int(cluster_phase_status.get("min_cluster_size"))
if min_cluster_size_default is None:
    min_cluster_size_default = 2

detect_inflight = bool(st.session_state.get(detect_running_key))
faces_ready = faces_ready_state
detector_manifest_value = helpers.tracks_detector_value(ep_id)
tracker_manifest_value = helpers.tracks_tracker_value(ep_id)
detector_face_only = helpers.detector_is_face_only(ep_id, detect_phase_status)
combo_detector, combo_tracker = helpers.detect_tracker_combo(ep_id, detect_phase_status)
combo_supported_harvest = helpers.pipeline_combo_supported("harvest", combo_detector, combo_tracker)
combo_supported_cluster = helpers.pipeline_combo_supported("cluster", combo_detector, combo_tracker)

# =============================================================================
# Pipeline Settings Header with Settings Dialog Button
# =============================================================================
settings_col1, settings_col2 = st.columns([10, 1])
with settings_col1:
    st.markdown("### Pipeline Settings")
with settings_col2:
    # Render the settings dialog and its trigger button
    _render_pipeline_settings_dialog(ep_id, video_meta)

_profile_session_prefix = f"episode_detail::{ep_id}"
_profile_widget_key = f"{_profile_session_prefix}::global_profile"

# Determine default profile: from previous job, phase status, or device-based default
_profile_default_value = (
    detect_job_defaults.get("profile")
    or faces_job_defaults.get("profile")
    or cluster_job_defaults.get("profile")
    or detect_phase_status.get("profile")
    or faces_phase_status.get("profile")
)
if not _profile_default_value:
    _profile_default_value = "balanced"  # Default to Balanced

_profile_seed_value = helpers.profile_value_from_state(
    st.session_state.get(_profile_widget_key, _profile_default_value)
)

# Sanitize the selectbox session state to use label format
if _profile_widget_key in st.session_state:
    _stored_value = st.session_state[_profile_widget_key]
    if _stored_value not in helpers.PROFILE_LABELS:
        _sanitized = helpers.PROFILE_LABEL_MAP.get(str(_stored_value).lower())
        if _sanitized:
            st.session_state[_profile_widget_key] = _sanitized
        else:
            del st.session_state[_profile_widget_key]

profile_label = st.selectbox(
    "Performance Profile",
    helpers.PROFILE_LABELS,
    index=helpers.profile_label_index(_profile_seed_value),
    key=_profile_widget_key,
    help="Controls stride, export settings, and resource usage for all pipeline jobs. "
         "**Balanced** (default) is recommended for most use cases.",
)
profile_value = helpers.PROFILE_VALUE_MAP.get(profile_label, _profile_seed_value)
profile_changed = profile_value != _profile_seed_value
profile_defaults = helpers.profile_defaults(profile_value)

# ── Auto-Run Pipeline ────────────────────────────────────────────────────────
# Session state keys for auto-run pipeline
_autorun_key = f"{ep_id}::autorun_pipeline"
_autorun_phase_key = f"{ep_id}::autorun_phase"  # "detect", "faces", "cluster", or None

# Check if auto-run is active and a phase just completed
autorun_active = st.session_state.get(_autorun_key, False)
autorun_phase = st.session_state.get(_autorun_phase_key)

# Auto-run container for the button
with st.container():
    auto_col1, auto_col2 = st.columns([3, 1])
    with auto_col1:
        if autorun_active:
            phase_labels = {"detect": "Detect/Track", "faces": "Faces Harvest", "cluster": "Clustering"}
            current_phase_label = phase_labels.get(autorun_phase, "Unknown")
            # Get completed stages log
            completed_stages = st.session_state.get(f"{ep_id}::autorun_completed_stages", [])
            # Build status display
            status_lines = []
            for stage_info in completed_stages:
                status_lines.append(f"✅ {stage_info}")
            status_lines.append(f"🔄 Currently running: {current_phase_label}")
            st.info("**Auto-Run Pipeline Active**\n\n" + "\n\n".join(status_lines))

            # Suggestion 4: Pipeline Progress Bar with ETA
            # Weight phases by typical duration: detect=50%, faces=30%, cluster=20%
            phase_weights = {"detect": 0.50, "faces": 0.30, "cluster": 0.20}
            phase_order = ["detect", "faces", "cluster"]

            # Calculate completed phase weight
            completed_weight = 0.0
            for phase in phase_order:
                if phase in [s.split(" ")[0].lower() for s in completed_stages]:
                    completed_weight += phase_weights.get(phase, 0)
                elif phase == "detect" and any("Detect" in s for s in completed_stages):
                    completed_weight += phase_weights["detect"]
                elif phase == "faces" and any("Faces" in s for s in completed_stages):
                    completed_weight += phase_weights["faces"]

            # Get current phase progress (from running jobs or 0)
            current_phase_weight = phase_weights.get(autorun_phase, 0)
            current_phase_pct = 0.0
            # Fetch running jobs for progress (uses cached batch API)
            _running_jobs = helpers.get_all_running_jobs_for_episode(ep_id)
            if autorun_phase == "detect" and _running_jobs.get("detect_track"):
                current_phase_pct = _running_jobs["detect_track"].get("progress_pct", 0) / 100
            elif autorun_phase == "faces" and _running_jobs.get("faces_embed"):
                current_phase_pct = _running_jobs["faces_embed"].get("progress_pct", 0) / 100
            elif autorun_phase == "cluster" and _running_jobs.get("cluster"):
                current_phase_pct = _running_jobs["cluster"].get("progress_pct", 0) / 100

            total_progress = completed_weight + (current_phase_pct * current_phase_weight)
            st.progress(min(total_progress, 1.0))
            st.caption(f"Pipeline: {int(total_progress * 100)}% complete")

            # Timing metrics display (expandable)
            _autorun_run_id = st.session_state.get(_autorun_run_id_key)
            timing_state = helpers.get_timing_state(ep_id, _autorun_run_id or selected_attempt_run_id)
            if timing_state and timing_state.jobs:
                with st.expander("Stage Timing", expanded=False):
                    # Build timing table
                    stage_ops = ["detect_track", "faces_embed", "cluster"]
                    stage_labels = {"detect_track": "Detect/Track", "faces_embed": "Faces Harvest", "cluster": "Cluster"}
                    timing_rows = []
                    for op in stage_ops:
                        metrics = helpers.format_timing_for_display(timing_state, op)
                        timing_rows.append({
                            "Stage": stage_labels.get(op, op),
                            "First Log": metrics["first_line"],
                            "Complete": metrics["completed"],
                            "Runtime": metrics["runtime"],
                            "Stall→Next": metrics["stall_to_next"],
                        })
                    # Display as markdown table
                    header = "| Stage | First Log | Complete | Runtime | Stall→Next |"
                    divider = "|-------|-----------|----------|---------|------------|"
                    rows = [f"| {r['Stage']} | {r['First Log']} | {r['Complete']} | {r['Runtime']} | {r['Stall→Next']} |" for r in timing_rows]
                    st.markdown("\n".join([header, divider] + rows))

                    # Total stall time summary
                    total_stall = timing_state.total_stall_time()
                    if total_stall > 0:
                        from autorun_timing import format_timing_duration
                        st.caption(f"Total inter-stage stall: {format_timing_duration(total_stall)}")
        else:
            st.caption("Run all pipeline stages in sequence: Detect/Track → Faces Harvest → Cluster")
    with auto_col2:
        if autorun_active:
            if st.button("⏹️ Stop Auto-Run", key="stop_autorun", use_container_width=True, type="secondary"):
                st.session_state[_autorun_key] = False
                st.session_state[_autorun_phase_key] = None
                st.session_state.pop(_autorun_run_id_key, None)
                st.toast("Auto-run stopped")
                st.rerun()
        else:
            # Disable if no video or job already running
            autorun_disabled = not local_video_exists or st.session_state.get(running_job_key, False)
            autorun_start_requested_key = f"{ep_id}::autorun_start_requested"

            if st.button(
                "🚀 Auto-Run Pipeline",
                key="start_autorun",
                use_container_width=True,
                type="primary",
                disabled=autorun_disabled,
                help="Run Detect/Track → Faces Harvest → Cluster automatically (resets review state first)",
            ):
                st.session_state[autorun_start_requested_key] = True
                st.rerun()

            if st.session_state.pop(autorun_start_requested_key, False):
                if autorun_disabled:
                    st.error("Auto-run is disabled (missing video or another job is running).")
                    st.stop()

                new_run_id = _generate_attempt_run_id(ep_id)
                try:
                    run_layout.run_root(ep_id, new_run_id).mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

                # Always reset review state when starting auto-run (best-effort; don't block pipeline start).
                with st.spinner("Resetting review state..."):
                    try:
                        helpers.api_post(
                            f"/episodes/{ep_id}/dismissed_suggestions/reset_state",
                            json={"archive_existing": True},
                            params={"run_id": new_run_id},
                            timeout=15,
                        )
                        helpers.api_post(
                            f"/episodes/{ep_id}/face_review/reset_state",
                            json={"archive_existing": True},
                            params={"run_id": new_run_id},
                            timeout=15,
                        )
                    except Exception as exc:
                        LOGGER.warning("[AUTORUN] Reset review state failed (continuing anyway): %s", exc)

                # Clear local UI session state so the next cluster run can re-open Improve Faces cleanly.
                for suffix in ("complete", "active", "suggestions", "index", "empty_reason", "trigger"):
                    st.session_state.pop(_improve_faces_state_key(ep_id, new_run_id, suffix), None)
                for key in (
                    f"{ep_id}::improve_faces_complete",
                    f"{ep_id}::improve_faces_active",
                    f"{ep_id}::improve_faces_suggestions",
                    f"{ep_id}::improve_faces_index",
                    f"{ep_id}::improve_faces_empty_reason",
                    f"{ep_id}::trigger_improve_faces",
                ):
                    st.session_state.pop(key, None)

                st.session_state[_autorun_run_id_key] = new_run_id
                # Do not modify the attempt selector widget key after it is created.
                # Instead, set a pending value that is applied at the top of the script
                # on the next rerun.
                st.session_state[_active_run_id_pending_key] = new_run_id
                LOGGER.info("[AUTORUN] Starting new pipeline run_id=%s", new_run_id)

                # Clear old manifest data to prevent stale data confusion
                # Archive old run markers and clear status cache
                _manifests_dir = helpers.DATA_ROOT / "manifests" / ep_id
                _runs_dir = _manifests_dir / "runs"
                archived_count = 0
                manifest_archived = 0
                if _runs_dir.exists():
                    archive_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    for marker_file in ["detect_track.json", "faces_embed.json", "cluster.json"]:
                        marker_path = _runs_dir / marker_file
                        if marker_path.exists():
                            try:
                                archive_path = _runs_dir / f"{marker_file}.{archive_time}.bak"
                                marker_path.rename(archive_path)
                                archived_count += 1
                                LOGGER.info("[AUTORUN] Archived old marker: %s -> %s", marker_file, archive_path.name)
                            except OSError as e:
                                LOGGER.warning("[AUTORUN] Failed to archive %s: %s", marker_file, e)
                    # Archive manifest files as well to force a clean pipeline run
                    manifest_files = [
                        "detections.jsonl",
                        "tracks.jsonl",
                        "faces.jsonl",
                        "identities.json",
                        "track_metrics.json",
                    ]
                    for manifest_file in manifest_files:
                        manifest_path = _manifests_dir / manifest_file
                        if manifest_path.exists():
                            try:
                                archive_path = _manifests_dir / f"{manifest_file}.{archive_time}.bak"
                                manifest_path.rename(archive_path)
                                manifest_archived += 1
                                LOGGER.info("[AUTORUN] Archived old manifest: %s -> %s", manifest_file, archive_path.name)
                            except OSError as e:
                                LOGGER.warning("[AUTORUN] Failed to archive manifest %s: %s", manifest_file, e)
                if archived_count > 0:
                    LOGGER.info("[AUTORUN] Archived %d old run markers", archived_count)
                if manifest_archived > 0:
                    LOGGER.info("[AUTORUN] Archived %d manifest files for clean auto-run start", manifest_archived)

                # Record baseline mtimes to avoid promoting stale artifacts from prior runs
                def _safe_mtime(path: Path) -> float:
                    try:
                        return path.stat().st_mtime
                    except (FileNotFoundError, OSError):
                        return 0.0

                st.session_state[f"{ep_id}::autorun_detect_baseline_mtime"] = max(
                    _safe_mtime(_manifests_dir / "tracks.jsonl"),
                    _safe_mtime(_runs_dir / "detect_track.json"),
                )
                st.session_state[f"{ep_id}::autorun_faces_baseline_mtime"] = max(
                    _safe_mtime(_manifests_dir / "faces.jsonl"),
                    _safe_mtime(_runs_dir / "faces_embed.json"),
                )
                st.session_state[f"{ep_id}::autorun_cluster_baseline_mtime"] = max(
                    _safe_mtime(_manifests_dir / "identities.json"),
                    _safe_mtime(_runs_dir / "cluster.json"),
                )

                # Clear all status caches to force fresh fetch
                st.session_state.pop(_status_mtimes_key(ep_id), None)
                st.session_state.pop(_status_cache_key(ep_id), None)
                st.session_state[_status_force_refresh_key(ep_id)] = True
                helpers.invalidate_running_jobs_cache(ep_id)
                _cached_local_jobs.clear()

                # Clear any stale promotion keys from previous runs
                st.session_state.pop(f"{ep_id}::autorun_detect_promoted_mtime", None)
                st.session_state.pop(f"{ep_id}::autorun_faces_promoted_mtime", None)

                st.session_state[_autorun_key] = True
                st.session_state[_autorun_phase_key] = "detect"
                st.session_state[f"{ep_id}::autorun_completed_stages"] = []  # Clear completed stages log
                st.session_state[f"{ep_id}::autorun_started_at"] = time.time()  # Track when auto-run started
                st.session_state["episode_detail_detect_autorun_flag"] = True  # Trigger detect job
                st.toast("Starting auto-run pipeline...")
                st.rerun()

# Debug: Show auto-run state (collapsible)
if autorun_active:
    with st.expander("🔍 Auto-Run Debug Info", expanded=False):
        # Get current settings for display
        _debug_detect_settings = _get_detect_settings(ep_id)
        debug_cols = st.columns(2)
        with debug_cols[0]:
            st.markdown("**Session State (Auto-Run)**")
            st.code(f"""
autorun_key: {st.session_state.get(_autorun_key)}
autorun_phase: {st.session_state.get(_autorun_phase_key)}
completed_stages: {st.session_state.get(f"{ep_id}::autorun_completed_stages", [])}
detect_autorun_flag: {st.session_state.get("episode_detail_detect_autorun_flag", False)}
faces_trigger: {st.session_state.get(f"{ep_id}::autorun_faces_trigger", False)}
cluster_trigger: {st.session_state.get(f"{ep_id}::autorun_cluster_trigger", False)}
            """, language="yaml")
        with debug_cols[1]:
            st.markdown("**Job State**")
            st.code(f"""
running_job_key: {st.session_state.get(running_job_key, False)}
detect_running: {st.session_state.get(_detect_job_state_key(ep_id), False)}
job_activity: {st.session_state.get(_job_activity_key(ep_id), False)}
tracks_just_completed: {st.session_state.get(f"{ep_id}::tracks_just_completed", False)}
faces_just_completed: {st.session_state.get(f"{ep_id}::faces_just_completed", False)}
            """, language="yaml")
            st.markdown("**Completion Flags (from ui_helpers)**")
            st.code(f"""
detect_track_just_completed: {st.session_state.get(f"{ep_id}::detect_track_just_completed", False)}
faces_embed_just_completed: {st.session_state.get(f"{ep_id}::faces_embed_just_completed", False)}
cluster_just_completed: {st.session_state.get(f"{ep_id}::cluster_just_completed", False)}
            """, language="yaml")
        st.markdown("**Phase Status (from API)**")
        st.code(f"""
detect_status: {detect_phase_status.get("status", "?")}
faces_status: {faces_phase_status.get("status", "?")}
cluster_status: {cluster_phase_status.get("status", "?")}
        """, language="yaml")
        st.markdown("**Timing (for fallback checks)**")
        _autorun_started = st.session_state.get(f"{ep_id}::autorun_started_at", 0)
        _autorun_started_str = datetime.fromtimestamp(_autorun_started).isoformat() if _autorun_started else "N/A"
        st.code(f"""
autorun_started_at: {_autorun_started_str}
faces_completed_at: {st.session_state.get(f"{ep_id}::faces_completed_at", "N/A")}
tracks_completed_at: {st.session_state.get(f"{ep_id}::tracks_completed_at", "N/A")}
        """, language="yaml")
        st.markdown("**Pipeline Settings (Current)**")
        st.code(f"""
scene_detector: {_debug_detect_settings.get("scene_detector", "?")}
scene_threshold: {_debug_detect_settings.get("scene_threshold", "?")}
scene_min_len: {_debug_detect_settings.get("scene_min_len", "?")}
scene_warmup_dets: {_debug_detect_settings.get("scene_warmup_dets", "?")}
stride: {_debug_detect_settings.get("stride", "?")}
device: {_debug_detect_settings.get("device", "?")}
detector: {_debug_detect_settings.get("detector", "?")}
tracker: {_debug_detect_settings.get("tracker", "?")}
        """, language="yaml")

st.divider()

# =============================================================================
# AUTO-RUN FALLBACK PROMOTION
# When auto-run is active but session flags were lost (e.g., page refresh during
# local mode job), check API status and manifest mtimes to recover state.
# =============================================================================
if autorun_active:
    autorun_phase = st.session_state.get(_autorun_phase_key)
    _fallback_triggered = False

    # Helper: Get run marker mtime for a phase
    def _get_run_marker_mtime(phase: str) -> float | None:
        marker_path = _scoped_markers_dir / f"{phase}.json"
        try:
            if marker_path.exists():
                return marker_path.stat().st_mtime
        except OSError:
            pass
        return None

    # Helper: Get manifest file mtime (returns 0 if doesn't exist).
    # NOTE: Avoid shadowing the global `_get_manifest_mtime(ep_id, phase)` helper used by
    # `_should_retry_phase_trigger()` later in this file (auto-run retry path).
    def _get_manifest_file_mtime(filename: str) -> float:
        manifest_path = manifests_dir / filename
        try:
            if manifest_path.exists():
                return manifest_path.stat().st_mtime
        except OSError:
            pass
        return 0.0

    # Helper: Check if manifest was updated recently (within N seconds)
    def _is_manifest_fresh(filename: str, max_age_sec: float = 120.0) -> bool:
        mtime = _get_manifest_file_mtime(filename)
        return mtime > 0 and (time.time() - mtime) < max_age_sec

    def _get_progress_mtime() -> float:
        progress_path = manifests_dir / "progress.json"
        try:
            if progress_path.exists():
                return progress_path.stat().st_mtime
        except OSError:
            pass
        return 0.0

    def _read_progress_payload() -> Dict[str, Any]:
        progress_path = manifests_dir / "progress.json"
        if not progress_path.exists():
            return {}
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    # Fallback 1: detect → faces
    # If auto-run is stuck on "detect" but API says detect is success and no job running
    if autorun_phase == "detect":
        detect_api_status = detect_phase_status.get("status")
        detect_job_running = helpers.get_running_job_for_episode(ep_id, "detect_track") is not None

        # Get when auto-run started - manifest must be newer than this
        autorun_started_at = st.session_state.get(f"{ep_id}::autorun_started_at", 0.0)
        detect_baseline = st.session_state.get(f"{ep_id}::autorun_detect_baseline_mtime", autorun_started_at)

        # FIX: Clamp future/stale autorun_started_at to prevent blocking
        # This handles cases where session state has a stale timestamp from a prior session
        # or clock skew causes the timestamp to be ahead of artifact mtimes
        current_time = time.time()
        if autorun_started_at > current_time + 60:
            LOGGER.warning(
                "[AUTORUN FALLBACK] autorun_started_at (%.1f) is in the future (current=%.1f), clamping",
                autorun_started_at, current_time
            )
            autorun_started_at = current_time
            st.session_state[f"{ep_id}::autorun_started_at"] = autorun_started_at

        # Check if tracks manifest is from current auto-run session
        tracks_manifest_mtime = _get_manifest_file_mtime("tracks.jsonl")
        tracks_manifest_from_current_run = tracks_manifest_mtime > detect_baseline

        # Also check run marker mtime as additional signal
        detect_marker_mtime = _get_run_marker_mtime("detect_track") or 0.0
        detect_marker_from_current_run = detect_marker_mtime > detect_baseline
        # Also allow a fresh progress.json “done” as a completion signal
        progress_mtime = _get_progress_mtime()
        progress_from_current_run = progress_mtime > detect_baseline
        progress_payload = _read_progress_payload()
        progress_done = str(progress_payload.get("step") or "").lower() == "done" or str(progress_payload.get("phase") or "").lower() == "done"
        if progress_done:
            progress_from_current_run = True

        # Debug logging for fallback conditions
        LOGGER.info(
            "[AUTORUN FALLBACK] detect check: api_status=%s, job_running=%s, "
            "tracks_mtime=%.1f, marker_mtime=%.1f, autorun_started=%.1f, "
            "manifest_current=%s, marker_current=%s",
            detect_api_status, detect_job_running,
            tracks_manifest_mtime, detect_marker_mtime, autorun_started_at,
            tracks_manifest_from_current_run, detect_marker_from_current_run
        )

        # Accept either manifest or marker being current (marker is more reliable for local mode)
        detect_from_current_run = (
            tracks_manifest_from_current_run
            or detect_marker_from_current_run
            or progress_from_current_run
        )

        # FIX: Safety fallback - if API=success and manifests exist AND are fresh, promote
        # This handles cases where autorun_started_at is stale/future compared to artifact mtimes
        # But we require manifests to be reasonably fresh (within 10 min) to avoid using stale data
        if detect_api_status == "success" and not detect_job_running and not detect_from_current_run:
            det_path = detections_path
            trk_path = tracks_path
            if det_path.exists() and trk_path.exists():
                # Check freshness - manifests must be modified within last 10 minutes
                manifest_freshness_window = 600  # 10 minutes in seconds
                manifest_age = current_time - tracks_manifest_mtime
                if manifest_age < manifest_freshness_window:
                    LOGGER.warning(
                        "[AUTORUN FALLBACK] mtime check failed (tracks=%.1f, marker=%.1f, started=%.1f) "
                        "but API=success and manifests are fresh (age=%.1fs) - promoting",
                        tracks_manifest_mtime, detect_marker_mtime, autorun_started_at, manifest_age
                    )
                    detect_from_current_run = True  # Allow the existing block to handle promotion
                else:
                    LOGGER.warning(
                        "[AUTORUN FALLBACK] mtime check failed and manifests are stale (age=%.1fs > %ds) - NOT promoting",
                        manifest_age, manifest_freshness_window
                    )

        if detect_api_status == "success" and not detect_job_running and detect_from_current_run:
            # Check if we already processed this (use run marker OR manifest mtime to avoid double-firing)
            _detect_promoted_key = f"{ep_id}::autorun_detect_promoted_mtime"
            marker_mtime = _get_run_marker_mtime("detect_track")
            # FALLBACK: If marker doesn't exist (e.g., subprocess interrupted), use manifest mtime
            effective_mtime = marker_mtime if marker_mtime else tracks_manifest_mtime
            last_promoted_mtime = st.session_state.get(_detect_promoted_key)

            LOGGER.info(
                "[AUTORUN FALLBACK] detect promotion check: marker_mtime=%s, manifest_mtime=%.1f, "
                "effective_mtime=%.1f, last_promoted=%.1f",
                marker_mtime, tracks_manifest_mtime, effective_mtime,
                last_promoted_mtime or 0.0
            )

            if effective_mtime > 0 and (last_promoted_mtime is None or effective_mtime > last_promoted_mtime):
                # Read counts from manifest for the completed stage log
                det_count = 0
                trk_count = 0
                det_path = detections_path
                trk_path = tracks_path
                try:
                    if det_path.exists():
                        with det_path.open("r", encoding="utf-8") as fh:
                            det_count = sum(1 for line in fh if line.strip())
                    if trk_path.exists():
                        with trk_path.open("r", encoding="utf-8") as fh:
                            trk_count = sum(1 for line in fh if line.strip())
                except OSError:
                    pass

                # Best-effort: write a run marker if missing
                marker_path = _scoped_markers_dir / "detect_track.json"
                if not marker_path.exists():
                    try:
                        marker_path.parent.mkdir(parents=True, exist_ok=True)
                        marker_payload = {
                            "phase": "detect_track",
                            "status": "success",
                            "ep_id": ep_id,
                            "detections": det_count,
                            "tracks": trk_count,
                            "finished_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                        }
                        if selected_attempt_run_id:
                            marker_payload["run_id"] = selected_attempt_run_id
                        marker_path.write_text(json.dumps(marker_payload, indent=2), encoding="utf-8")
                    except OSError:
                        pass

                # Guard: If zero detections/tracks, stop auto-run (scene-only or failed run)
                if det_count == 0 or trk_count == 0:
                    LOGGER.warning("[AUTORUN FALLBACK] Stopping: zero detections/tracks (det=%s, trk=%s)", det_count, trk_count)
                    st.session_state[_autorun_key] = False
                    st.session_state[_autorun_phase_key] = None
                    st.session_state[f"{ep_id}::autorun_error"] = f"Zero detections ({det_count}) or tracks ({trk_count})"
                    st.error(f"❌ Auto-run stopped: No faces detected (detections={det_count}, tracks={trk_count}).")
                else:
                    LOGGER.info("[AUTORUN FALLBACK] Detect complete (API status=success), promoting to faces phase")
                    # Mark as promoted to avoid double-firing (use effective_mtime which falls back to manifest)
                    st.session_state[_detect_promoted_key] = effective_mtime
                    # Set completion flags
                    st.session_state[f"{ep_id}::tracks_just_completed"] = True
                    st.session_state[f"{ep_id}::detect_track_just_completed"] = True
                    st.session_state[f"{ep_id}::tracks_completed_at"] = time.time()
                    # Log completed stage
                    completed = st.session_state.get(f"{ep_id}::autorun_completed_stages", [])
                    if not any("Detect" in s for s in completed):
                        completed.append(f"Detect/Track ({det_count:,} detections, {trk_count:,} tracks)")
                        st.session_state[f"{ep_id}::autorun_completed_stages"] = completed
                    # Advance to faces phase
                    st.session_state[_autorun_phase_key] = "faces"
                    st.session_state[f"{ep_id}::autorun_faces_trigger"] = True
                    st.toast("✅ Detect/Track complete - advancing to Faces Harvest...")
                    _fallback_triggered = True

    # Fallback 2: faces → cluster
    # If auto-run is stuck on "faces" but faces job is done and no job running
    # Accept "success" or "stale" status (stale happens when detect is newer than faces)
    # Also check manifest/marker freshness as additional signal
    elif autorun_phase == "faces":
        faces_api_status = faces_phase_status.get("status")
        faces_job_running = helpers.get_running_job_for_episode(ep_id, "faces_embed") is not None

        # Accept "success" or "stale" - both mean faces harvest has run at some point
        # "stale" occurs when detect_track marker is newer than faces marker (timestamp comparison)
        faces_status_ok = faces_api_status in ("success", "stale")

        # Get when auto-run started - manifest/marker must be newer than this
        autorun_started_at = st.session_state.get(f"{ep_id}::autorun_started_at", 0.0)
        faces_baseline = st.session_state.get(f"{ep_id}::autorun_faces_baseline_mtime", autorun_started_at)

        # Check manifest mtime - must be newer than when auto-run started
        faces_manifest_mtime = _get_manifest_file_mtime("faces.jsonl")
        faces_manifest_from_current_run = faces_manifest_mtime > faces_baseline

        # Also check run marker mtime - this is written when faces_embed completes
        faces_marker_mtime = _get_run_marker_mtime("faces_embed") or 0.0
        faces_marker_from_current_run = faces_marker_mtime > faces_baseline
        # Also allow a fresh progress.json “done” as a completion signal
        progress_mtime = _get_progress_mtime()
        progress_from_current_run = progress_mtime > faces_baseline

        # Check if faces marker is newer than detect marker (proves faces ran after detect)
        detect_marker_mtime = _get_run_marker_mtime("detect_track") or 0.0
        faces_ran_after_detect = faces_marker_mtime > detect_marker_mtime

        LOGGER.info(
            "[AUTORUN FALLBACK] faces check: status=%s, job_running=%s, status_ok=%s, "
            "manifest_mtime=%.1f, marker_mtime=%.1f, detect_marker=%.1f, autorun_started=%.1f, "
            "manifest_current=%s, marker_current=%s, faces_after_detect=%s",
            faces_api_status, faces_job_running, faces_status_ok,
            faces_manifest_mtime, faces_marker_mtime, detect_marker_mtime, autorun_started_at,
            faces_manifest_from_current_run, faces_marker_from_current_run, faces_ran_after_detect
        )

        # Proceed if:
        # 1. No job currently running
        # 2. Either manifest or marker is from current auto-run session
        # 3. Either API status is OK, or faces ran after detect (marker comparison)
        faces_from_current_run = (
            faces_manifest_from_current_run
            or faces_marker_from_current_run
            or progress_from_current_run
        )
        faces_valid = faces_status_ok or faces_ran_after_detect

        if not faces_job_running and faces_from_current_run and faces_valid:
            # Check if we already processed this
            _faces_promoted_key = f"{ep_id}::autorun_faces_promoted_mtime"
            # Use manifest mtime as source of truth (more reliable than run marker in edge cases)
            manifest_mtime = faces_manifest_mtime
            marker_mtime = _get_run_marker_mtime("faces_embed")
            # Use the most recent of manifest or marker mtime
            effective_mtime = max(manifest_mtime, marker_mtime or 0.0)
            last_promoted_mtime = st.session_state.get(_faces_promoted_key)

            if effective_mtime > 0 and (last_promoted_mtime is None or effective_mtime > last_promoted_mtime):
                # Read faces count from manifest
                faces_count = 0
                faces_manifest_path = faces_path
                try:
                    if faces_manifest_path.exists():
                        with faces_manifest_path.open("r", encoding="utf-8") as fh:
                            faces_count = sum(1 for line in fh if line.strip())
                except OSError:
                    pass

                # Guard: If zero faces, stop auto-run
                if faces_count == 0:
                    LOGGER.warning("[AUTORUN FALLBACK] Stopping: zero faces harvested")
                    st.session_state[_autorun_key] = False
                    st.session_state[_autorun_phase_key] = None
                    st.session_state[f"{ep_id}::autorun_error"] = "Zero faces harvested - cannot cluster"
                    st.error(f"❌ Auto-run stopped: No faces harvested (faces={faces_count}).")
                else:
                    LOGGER.info("[AUTORUN FALLBACK] Faces complete (status=%s, from_current_run=%s), promoting to cluster", faces_api_status, faces_from_current_run)
                    # Mark as promoted using effective_mtime
                    st.session_state[_faces_promoted_key] = effective_mtime
                    # Set completion flags
                    st.session_state[f"{ep_id}::faces_just_completed"] = True
                    st.session_state[f"{ep_id}::faces_completed_at"] = time.time()
                    # Log completed stage
                    completed = st.session_state.get(f"{ep_id}::autorun_completed_stages", [])
                    if not any("Faces" in s for s in completed):
                        completed.append(f"Faces Harvest ({faces_count:,} faces)")
                        st.session_state[f"{ep_id}::autorun_completed_stages"] = completed
                    # Advance to cluster phase
                    st.session_state[_autorun_phase_key] = "cluster"
                    st.session_state[f"{ep_id}::autorun_cluster_trigger"] = True
                    st.toast("✅ Faces Harvest complete - advancing to Cluster...")
                    _fallback_triggered = True

    if _fallback_triggered:
        st.rerun()

col_detect, col_faces, col_cluster = st.columns(3)

# Check for running jobs for each phase
running_detect_job = helpers.get_running_job_for_episode(ep_id, "detect_track")
running_faces_job = helpers.get_running_job_for_episode(ep_id, "faces_embed")
running_cluster_job = helpers.get_running_job_for_episode(ep_id, "cluster")
running_audio_job = helpers.get_running_job_for_episode(ep_id, "audio_pipeline")

# Synchronize session state with API-based job status
# This is the single source of truth - clears stale session flags if API says no job running
job_running, stale_job_warning = _sync_job_state_with_api(
    ep_id,
    running_job_key,
    running_detect_job,
    running_faces_job,
    running_cluster_job,
    running_audio_job,
)
if stale_job_warning:
    st.warning(f"⚠️ {stale_job_warning}")

# Session state keys for cancel confirmation dialogs
confirm_cancel_detect_key = f"{ep_id}::confirm_cancel_detect"
confirm_cancel_faces_key = f"{ep_id}::confirm_cancel_faces"
confirm_cancel_cluster_key = f"{ep_id}::confirm_cancel_cluster"

with col_detect:
    st.markdown("### Detect/Track Faces")
    session_prefix = f"episode_detail_detect::{ep_id}"

    # Show running job progress if a job is active
    # Skip if we already marked this job as complete (prevents infinite refresh loop)
    detect_job_complete_key = f"{ep_id}::detect_job_complete"
    if running_detect_job and not st.session_state.get(detect_job_complete_key):
        # Bug 6 fix: Generate unique key even if job_id is missing
        job_id = running_detect_job.get("job_id") or f"detect_{hash(str(running_detect_job)) % 10000}"
        progress_pct = running_detect_job.get("progress_pct", 0)
        frames_done = running_detect_job.get("frames_done", 0)
        frames_total = running_detect_job.get("frames_total", 0)
        state = running_detect_job.get("state", "running")

        # Auto-refresh when job hits 100% or state indicates completion
        job_complete = progress_pct >= 99.5 or state in ("done", "success", "completed")
        if job_complete:
            st.success(f"✅ **Detect/Track complete!** ({frames_done:,} / {frames_total:,} frames)")
            # Mark job as complete to prevent infinite refresh loop
            st.session_state[detect_job_complete_key] = True
            # Force status refresh to pick up new data
            st.session_state[_status_force_refresh_key(ep_id)] = True
            # Mark tracks as fresh to bypass stale status check (Celery jobs may have status update delays)
            st.session_state[f"{ep_id}::tracks_completed_at"] = time.time()  # Bug 4 fix: Use timestamp

            # Check if auto-run is active - trigger next phase
            if st.session_state.get(_autorun_key) and st.session_state.get(_autorun_phase_key) == "detect":
                st.caption("Auto-run: Starting Faces Harvest...")
                st.session_state[_autorun_phase_key] = "faces"
                st.session_state[f"{ep_id}::autorun_faces_trigger"] = True
                # Suggestion 3: No delay for auto-run transitions
                st.rerun()
            else:
                st.caption("Refreshing to show results...")
                # Brief delay for manual runs to show success message
                time.sleep(0.5)
                st.rerun()

        st.info(f"🔄 **Detect/Track job running** ({state})")
        if frames_total > 0:
            st.progress(min(progress_pct / 100, 1.0))
            st.caption(f"Progress: {frames_done:,} / {frames_total:,} frames ({progress_pct:.1f}%)")
        else:
            st.caption(f"Progress: {progress_pct:.1f}%")

        # Refresh and Cancel buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🔄 Refresh", key=f"refresh_detect_{job_id}", use_container_width=True):
                st.rerun()
        with btn_col2:
            if st.button("❌ Cancel", key=f"cancel_detect_{job_id}", use_container_width=True):
                success, msg = helpers.cancel_running_job(job_id)
                if success:
                    st.success(msg)
                    time.sleep(0.3)  # Suggestion 3: Reduced delay
                    st.rerun()
                else:
                    st.error(msg)

        st.divider()

    # ─── GET SETTINGS FROM PIPELINE SETTINGS DIALOG ──────────────────────────
    # All settings are now managed in the unified Pipeline Settings dialog (gear icon)
    # We just read the values here to build the job payload
    detect_settings = _get_detect_settings(ep_id)

    # Extract values from settings
    stride_value = detect_settings["stride"]
    fps_value = detect_settings["fps"]
    det_thresh_value = detect_settings["det_thresh"]
    save_frames = detect_settings["save_frames"]
    save_crops = detect_settings["save_crops"]
    cpu_threads_value = detect_settings["cpu_threads"]
    max_gap_value = detect_settings["max_gap"]
    jpeg_quality = detect_settings["jpeg_quality"]
    track_high_value = detect_settings["track_high_thresh"]
    track_new_value = detect_settings["new_track_thresh"]
    scene_detector_value = detect_settings["scene_detector"]
    scene_threshold_value = detect_settings["scene_threshold"]
    scene_min_len_value = detect_settings["scene_min_len"]
    scene_warmup_value = detect_settings["scene_warmup_dets"]
    detect_device_value = detect_settings["device"]
    detect_device_label = detect_settings["device_label"]

    # Build summary for display
    stride_hint = "every frame" if stride_value == 1 else f"every {stride_value}th frame"
    export_bits: list[str] = []
    if save_frames:
        export_bits.append("frames")
    if save_crops:
        export_bits.append("crops")
    export_text = "saving " + " & ".join(export_bits) if export_bits else "no frame/crop exports"

    # Show compact settings summary
    scene_label = helpers.scene_detector_label(scene_detector_value)
    st.info(
        f"**Detect/Track** → {detect_detector_label} + {detect_tracker_label} on {detect_device_label} "
        f"· stride {stride_value} ({stride_hint}), {export_text}\n\n"
        f"Scene detection: **{scene_label}** (threshold {scene_threshold_value:.1f})"
    )
    if job_running:
        st.caption("Another job is running; Detect/Track controls will re-enable once it completes.")

    job_payload = helpers.default_detect_track_payload(
        ep_id,
        stride=int(stride_value),
        det_thresh=float(det_thresh_value),
        device=detect_device_value,
    )
    job_payload.update(
        {
            "profile": profile_value,
            "save_frames": bool(save_frames),
            "save_crops": bool(save_crops),
            "jpeg_quality": int(jpeg_quality),
            "max_gap": int(max_gap_value),
            "scene_detector": scene_detector_value,
            "scene_threshold": float(scene_threshold_value),
            "scene_min_len": int(scene_min_len_value),
            "scene_warmup_dets": int(scene_warmup_value),
            "cpu_threads": int(cpu_threads_value),
        }
    )
    job_payload["detector"] = detect_detector_value
    job_payload["tracker"] = detect_tracker_value
    if detect_tracker_value == "bytetrack" and track_high_value is not None and track_new_value is not None:
        job_payload["track_high_thresh"] = float(track_high_value)
        job_payload["new_track_thresh"] = float(track_new_value)
    if fps_value > 0:
        job_payload["fps"] = fps_value
    mode_label = f"{detect_detector_label} + {detect_tracker_label}"

    if selected_attempt_run_id:
        job_payload["run_id"] = selected_attempt_run_id
    elif autorun_active:
        autorun_run_id = st.session_state.get(_autorun_run_id_key)
        if isinstance(autorun_run_id, str) and autorun_run_id.strip():
            job_payload["run_id"] = autorun_run_id.strip()

    def _process_detect_result(summary: Dict[str, Any] | None, error_message: str | None) -> None:
        # DEBUG: Log entry to help diagnose auto-run issues
        autorun_val = st.session_state.get(_autorun_key)
        phase_val = st.session_state.get(_autorun_phase_key)
        # FIX 5: Enhanced diagnostic logging
        LOGGER.info(
            "[AUTORUN] _process_detect_result: summary_truthy=%s, summary_status=%s, error=%s, autorun=%s, phase=%s",
            bool(summary),
            summary.get("status") if isinstance(summary, dict) else None,
            error_message,
            autorun_val,
            phase_val
        )
        # Extra debug: show what summary contains
        if summary:
            LOGGER.info("[AUTORUN] Summary type=%s, keys=%s", type(summary).__name__, list(summary.keys())[:10])

        # FIX 3: Synthesize summary if empty but no error
        # This handles cases where streaming succeeded but didn't return a proper summary
        if not summary and not error_message:
            LOGGER.warning("[AUTORUN] No summary received, attempting to synthesize from manifests")
            det_path = detections_path
            trk_path = tracks_path
            if det_path.exists() and trk_path.exists():
                LOGGER.info("[AUTORUN] Manifests exist, synthesizing summary")
                summary = {"status": "completed", "fallback": True}

        # IMPORTANT: Set auto-run progression FIRST, before any early returns
        # This ensures the next phase is triggered even if we can't display counts
        if summary and not error_message:
            # Mark job complete flags for single runs and refresh helpers
            detect_job_complete_key = f"{ep_id}::detect_job_complete"
            st.session_state[detect_job_complete_key] = True
            st.session_state[f"{ep_id}::detect_track_just_completed"] = True
            st.session_state[_status_force_refresh_key(ep_id)] = True
            st.session_state[f"{ep_id}::tracks_completed_at"] = time.time()  # Bug 4 fix: Use timestamp
            # Invalidate ALL caches to force fresh status on next page load
            helpers.invalidate_running_jobs_cache(ep_id)
            _cached_local_jobs.clear()
            # Clear status cache to force refetch
            st.session_state.pop(_status_mtimes_key(ep_id), None)
            st.session_state.pop(_status_cache_key(ep_id), None)
            if st.session_state.get(_autorun_key) and st.session_state.get(_autorun_phase_key) == "detect":
                # Log completed stage with counts from summary
                LOGGER.info("[AUTORUN] Detect complete, checking if we can advance to faces phase")
                LOGGER.info("[AUTORUN] Raw summary keys: %s", list(summary.keys()) if summary else "None")
                normalized = helpers.normalize_summary(ep_id, summary)
                LOGGER.info("[AUTORUN] Normalized summary: detections=%s, tracks=%s",
                           normalized.get("detections"), normalized.get("tracks"))
                det_count = helpers.coerce_int(normalized.get("detections"))
                trk_count = helpers.coerce_int(normalized.get("tracks"))
                LOGGER.info("[AUTORUN] Coerced counts: det_count=%s, trk_count=%s", det_count, trk_count)

                # FALLBACK: If normalize_summary didn't get counts, read directly from manifest
                # This is critical for local mode where streaming response may not include counts
                if det_count is None or det_count == 0:
                    det_path = detections_path
                    if det_path.exists():
                        try:
                            with det_path.open("r", encoding="utf-8") as fh:
                                det_count = sum(1 for line in fh if line.strip())
                            LOGGER.info("[AUTORUN] Fallback det_count from manifest: %s", det_count)
                        except OSError:
                            pass
                if trk_count is None or trk_count == 0:
                    trk_path = tracks_path
                    if trk_path.exists():
                        try:
                            with trk_path.open("r", encoding="utf-8") as fh:
                                trk_count = sum(1 for line in fh if line.strip())
                            LOGGER.info("[AUTORUN] Fallback trk_count from manifest: %s", trk_count)
                        except OSError:
                            pass

                # Bug D fix: Validate detection/track counts before advancing
                if det_count is None or det_count == 0 or trk_count is None or trk_count == 0:
                    LOGGER.warning("[AUTORUN] Stopping: zero detections/tracks (det=%s, trk=%s)", det_count, trk_count)
                    st.session_state[_autorun_key] = False
                    st.session_state[_autorun_phase_key] = None
                    st.session_state[f"{ep_id}::autorun_error"] = f"Zero detections ({det_count or 0}) or tracks ({trk_count or 0}) - cannot continue"
                    st.error(f"❌ Auto-run stopped: No faces detected (detections={det_count or 0}, tracks={trk_count or 0}).")
                    st.rerun()
                    return  # Unreachable but explicit - st.rerun() raises exception

                completed = st.session_state.get(f"{ep_id}::autorun_completed_stages", [])
                completed.append(f"Detect/Track ({det_count:,} detections, {trk_count:,} tracks)")
                st.session_state[f"{ep_id}::autorun_completed_stages"] = completed
                st.session_state[_autorun_phase_key] = "faces"
                st.session_state[f"{ep_id}::autorun_faces_trigger"] = True
                # Verify state was set before rerun
                verify_phase = st.session_state.get(_autorun_phase_key)
                verify_trigger = st.session_state.get(f"{ep_id}::autorun_faces_trigger")
                LOGGER.info("[AUTORUN] Set phase=faces, trigger=True. Verify: phase=%s, trigger=%s", verify_phase, verify_trigger)

                # CRITICAL: Verify state was actually set correctly before rerun
                if verify_phase != "faces" or not verify_trigger:
                    LOGGER.error("[AUTORUN] FAILED to set faces trigger state! phase=%s, trigger=%s", verify_phase, verify_trigger)
                    st.error("⚠️ Auto-run state error: Could not set faces trigger - please try again")
                    st.session_state[_autorun_key] = False
                    st.session_state[_autorun_phase_key] = None
                    return

                st.toast("✅ Detect/Track complete - advancing to Faces Harvest...")
                LOGGER.info("[AUTORUN] About to st.rerun() to trigger faces phase")
                # CRITICAL: Rerun immediately to trigger faces phase
                # Don't wait for end of function which may have early returns
                st.rerun()

        if error_message:
            # Bug 2 fix: Stop auto-run on error to prevent getting stuck
            if st.session_state.get(_autorun_key):
                st.session_state[_autorun_key] = False
                st.session_state[_autorun_phase_key] = None
                st.session_state[f"{ep_id}::autorun_error"] = error_message
                st.error(f"❌ Auto-run stopped due to error in Detect/Track phase.")

            if error_message == "mirror_failed":
                st.error("Failed to mirror video from S3. Check that the video exists in S3 and you have network connectivity.")
                return
            if "RetinaFace weights missing or could not initialize" in error_message:
                st.error(error_message)
                st.caption("Run `python scripts/fetch_models.py` then retry.")
            else:
                st.error(error_message)
            return
        if not summary:
            return
        normalized = helpers.normalize_summary(ep_id, summary)
        detections = helpers.coerce_int(normalized.get("detections"))
        tracks = helpers.coerce_int(normalized.get("tracks"))
        frames_exported = helpers.coerce_int(normalized.get("frames_exported"))
        crops_exported = helpers.coerce_int(normalized.get("crops_exported"))
        detector_summary = normalized.get("detector")
        tracker_summary = normalized.get("tracker")
        track_ratio_value = helpers.coerce_float(
            normalized.get("track_to_detection_ratio") or normalized.get("track_ratio")
        )
        detector_is_scene = isinstance(detector_summary, str) and detector_summary in helpers.SCENE_DETECTOR_LABEL_MAP
        has_detections = detections is not None and detections > 0
        has_tracks = tracks is not None and tracks > 0
        issue_messages: list[str] = []
        if detector_is_scene:
            detector_label = helpers.SCENE_DETECTOR_LABEL_MAP.get(detector_summary, detector_summary)
            issue_messages.append(f"Pipeline stopped after scene detection ({detector_label}); detect/track never ran.")
        if not has_detections or not has_tracks:
            det_label = helpers.format_count(detections) or "0"
            track_label = helpers.format_count(tracks) or "0"
            issue_messages.append(f"No detections/tracks were created (detections={det_label}, tracks={track_label}).")
        if issue_messages:
            st.error(" ".join(issue_messages) + " Please rerun **Detect/Track Faces** to generate the manifests.")
            return
        if track_ratio_value is not None and track_ratio_value < 0.1:
            st.warning(
                "⚠️ Track-to-detection ratio is below 0.10. Consider lowering ByteTrack thresholds or inspecting the episode."
            )
        details_line = [
            (f"detections: {helpers.format_count(detections)}" if detections is not None else "detections: ?"),
            (f"tracks: {helpers.format_count(tracks)}" if tracks is not None else "tracks: ?"),
        ]
        if track_ratio_value is not None:
            details_line.append(f"tracks/detections: {track_ratio_value:.2f}")
        if frames_exported:
            details_line.append(f"frames exported: {helpers.format_count(frames_exported)}")
        if crops_exported:
            details_line.append(f"crops exported: {helpers.format_count(crops_exported)}")
        if detector_summary:
            details_line.append(f"detector: {helpers.detector_label_from_value(detector_summary)}")
        if tracker_summary:
            details_line.append(f"tracker: {helpers.tracker_label_from_value(tracker_summary)}")
        st.session_state["episode_detail_flash"] = "Detect/track complete · " + " · ".join(details_line)
        # Note: status refresh and auto-run progression are now set at the TOP of this function
        # before any early returns, to ensure they always happen on success
        st.rerun()

    if autorun_detect:
        local_video_exists, summary, error_message = _launch_detect_job(
            local_video_exists,
            ep_id,
            details,
            job_payload,
            detect_device_value,
            detect_detector_value,
            detect_tracker_value,
            mode_label,
            detect_device_label,
            running_state_key=running_job_key,
            active_job_key=_job_activity_key(ep_id),
            detect_flag_key=detect_running_key,
        )
        try:
            _process_detect_result(summary, error_message)
        except (RerunException, StopException):
            # st.rerun() and st.stop() work by raising exceptions - let them propagate
            raise
        except Exception as e:
            LOGGER.exception("[AUTORUN] Exception in _process_detect_result (auto-run trigger): %s", e)
            st.error(f"⚠️ Auto-run error: {e}")
            # Clear auto-run to prevent stuck state
            st.session_state[_autorun_key] = False
            st.session_state[_autorun_phase_key] = None
        # Note: _process_detect_result() handles auto-run progression and calls st.rerun()
        # Code after this point is unreachable when job completes successfully

    if not local_video_exists:
        s3_meta = details.get("s3") or {}
        s3_exists = s3_meta.get("v2_exists") or s3_meta.get("v1_exists")
        if s3_exists:
            st.info("Local mirror missing; Detect/Track will mirror automatically from S3 before starting.")
        else:
            st.warning("Video not found locally or in S3. Upload the video first via the Upload page.")

    # Display total frames from video metadata
    total_frames = None
    if video_meta:
        frames_val = video_meta.get("frames")
        fps_detected = video_meta.get("fps_detected")
        duration_sec = video_meta.get("duration_sec")
        try:
            if frames_val is not None:
                total_frames = int(frames_val)
            elif duration_sec and fps_detected:
                total_frames = int(float(duration_sec) * float(fps_detected))
        except (TypeError, ValueError):
            pass

    if total_frames:
        st.markdown(f"**Total Frames:** {total_frames:,}")

    run_label = "Detect/Track (auto-mirrors from S3)"
    # Disable button if any job is running (ours or detected from API)
    detect_button_disabled = job_running or detect_status_value == "running" or running_detect_job is not None

    if running_detect_job:
        # Show warning that a job is already running
        st.warning(f"⚠️ A detect/track job is already running ({running_detect_job.get('progress_pct', 0):.1f}% complete). Cancel it above to start a new one.")

    if st.button(run_label, use_container_width=True, disabled=detect_button_disabled):
        # Keep runtime logs anchored just below the button for local mode runs.
        detect_log_container = st.container()
        with detect_log_container:
            local_video_exists, summary, error_message = _launch_detect_job(
                local_video_exists,
                ep_id,
                details,
                job_payload,
                detect_device_value,
                detect_detector_value,
                detect_tracker_value,
                mode_label,
                detect_device_label,
                running_state_key=running_job_key,
                active_job_key=_job_activity_key(ep_id),
                detect_flag_key=detect_running_key,
            )
            try:
                _process_detect_result(summary, error_message)
            except (RerunException, StopException):
                # st.rerun() and st.stop() work by raising exceptions - let them propagate
                raise
            except Exception as e:
                LOGGER.exception("[AUTORUN] Exception in _process_detect_result (button click): %s", e)
                st.error(f"⚠️ Auto-run error: {e}")
                # Clear auto-run to prevent stuck state
                st.session_state[_autorun_key] = False
                st.session_state[_autorun_phase_key] = None
    st.caption("Mirrors required video artifacts automatically before detect/track starts.")

    # Show previous run logs (only in local mode, collapsed by default)
    if helpers.get_execution_mode(ep_id) == "local":
        helpers.render_previous_logs(ep_id, "detect_track", expanded=False)

with col_faces:
    # CRITICAL: Check for detect_track completion flag set by ui_helpers
    # This catches cases where the return value didn't reach the caller
    # (Streamlit may restart script on widget updates during streaming)
    _detect_just_completed_key = f"{ep_id}::detect_track_just_completed"
    if st.session_state.get(_detect_just_completed_key):
        LOGGER.info("[AUTORUN] Detected detect_track_just_completed flag, advancing to faces phase")
        # Get the stored completion context (do not clear until we successfully handle it).
        stored_summary = st.session_state.get(f"{ep_id}::detect_track_summary")
        completed_at = st.session_state.get(f"{ep_id}::detect_track_completed_at")

        # Set tracks_completed_at if not already set
        if not st.session_state.get(f"{ep_id}::tracks_completed_at"):
            st.session_state[f"{ep_id}::tracks_completed_at"] = completed_at or time.time()
            LOGGER.info("[AUTORUN] Set tracks_completed_at from detect_track completion flag")

        # If auto-run is active, ALWAYS ensure the faces trigger is set after detect completes.
        autorun_active_now = bool(st.session_state.get(_autorun_key))
        phase_now = st.session_state.get(_autorun_phase_key)
        faces_trigger_key = f"{ep_id}::autorun_faces_trigger"
        faces_trigger_now = bool(st.session_state.get(faces_trigger_key))

        if autorun_active_now and not faces_trigger_now and phase_now in (None, "detect", "faces"):
            det_count: int | None = None
            track_count: int | None = None
            if isinstance(stored_summary, dict) and stored_summary:
                normalized = helpers.normalize_summary(ep_id, stored_summary)
                det_count = helpers.coerce_int(normalized.get("detections"))
                track_count = helpers.coerce_int(normalized.get("tracks"))
            if det_count is None or track_count is None:
                try:
                    if detections_path.exists():
                        det_count = sum(1 for line in detections_path.open("r", encoding="utf-8") if line.strip())
                    if tracks_path.exists():
                        track_count = sum(1 for line in tracks_path.open("r", encoding="utf-8") if line.strip())
                except OSError:
                    pass

            completed = st.session_state.get(f"{ep_id}::autorun_completed_stages", [])
            if not any("Detect/Track" in s for s in completed):
                det_display = f"{det_count:,}" if isinstance(det_count, int) else "?"
                track_display = f"{track_count:,}" if isinstance(track_count, int) else "?"
                completed.append(f"Detect/Track ({det_display} detections, {track_display} tracks)")
                st.session_state[f"{ep_id}::autorun_completed_stages"] = completed

            st.session_state[_autorun_phase_key] = "faces"
            st.session_state[faces_trigger_key] = True
            st.session_state[_status_force_refresh_key(ep_id)] = True
            LOGGER.info(
                "[AUTORUN] Promoted detect completion -> faces trigger (phase was %s, det=%s, tracks=%s)",
                phase_now,
                det_count,
                track_count,
            )

            # Clear completion signal now that we've promoted it.
            st.session_state.pop(_detect_just_completed_key, None)
            st.session_state.pop(f"{ep_id}::detect_track_summary", None)

            st.toast("✅ Detect/Track complete - starting Faces Harvest...")
            st.rerun()

        # No promotion needed (manual run, or faces already queued); clear the completion signal.
        st.session_state.pop(_detect_just_completed_key, None)
        st.session_state.pop(f"{ep_id}::detect_track_summary", None)

    st.markdown("### Faces Harvest")
    st.caption(_format_phase_status("Faces Harvest", faces_phase_status, "faces"))

    # Show running job progress if a job is active
    # Skip if we already marked this job as complete (prevents infinite refresh loop)
    faces_job_complete_key = f"{ep_id}::faces_job_complete"
    if running_faces_job and not st.session_state.get(faces_job_complete_key):
        # Bug 6 fix: Generate unique key even if job_id is missing
        job_id = running_faces_job.get("job_id") or f"faces_{hash(str(running_faces_job)) % 10000}"
        progress_pct = running_faces_job.get("progress_pct", 0)
        state = running_faces_job.get("state", "running")

        # Auto-refresh when job hits 100% or state indicates completion
        job_complete = progress_pct >= 99.5 or state in ("done", "success", "completed")
        if job_complete:
            st.success("✅ **Faces Harvest complete!**")
            # Mark job as complete to prevent infinite refresh loop
            st.session_state[faces_job_complete_key] = True
            # Force status refresh to pick up new data
            st.session_state[_status_force_refresh_key(ep_id)] = True
            # Mark faces as fresh to bypass stale status check (Celery jobs may have status update delays)
            st.session_state[f"{ep_id}::faces_completed_at"] = time.time()  # Bug 4 fix: Use timestamp

            # Check if auto-run is active - trigger next phase
            if st.session_state.get(_autorun_key) and st.session_state.get(_autorun_phase_key) == "faces":
                st.caption("Auto-run: Starting Clustering...")
                st.session_state[_autorun_phase_key] = "cluster"
                st.session_state[f"{ep_id}::autorun_cluster_trigger"] = True
                # Suggestion 3: No delay for auto-run transitions
                st.rerun()
            else:
                st.caption("Refreshing to show results...")
                # Brief delay for manual runs to show success message
                time.sleep(0.5)
                st.rerun()

        st.info(f"🔄 **Faces Harvest job running** ({state})")
        st.progress(min(progress_pct / 100, 1.0))
        st.caption(f"Progress: {progress_pct:.1f}%")

        # Refresh and Cancel buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🔄 Refresh", key=f"refresh_faces_{job_id}", use_container_width=True):
                st.rerun()
        with btn_col2:
            if st.button("❌ Cancel", key=f"cancel_faces_{job_id}", use_container_width=True):
                success, msg = helpers.cancel_running_job(job_id)
                if success:
                    st.success(msg)
                    time.sleep(0.3)  # Suggestion 3: Reduced delay
                    st.rerun()
                else:
                    st.error(msg)

        st.divider()

    # ─── GET SETTINGS FROM PIPELINE SETTINGS DIALOG ──────────────────────────
    # All settings are now managed in the unified Pipeline Settings dialog (gear icon)
    harvest_settings = _get_harvest_settings(ep_id, video_meta)

    faces_device_value = harvest_settings["device"]
    faces_device_label = harvest_settings["device_label"]
    faces_save_frames = harvest_settings["save_frames"]
    faces_save_crops = harvest_settings["save_crops"]
    faces_min_frames_between_crops = harvest_settings["min_frames_between_crops"]
    faces_thumb_size = harvest_settings["thumb_size"]
    faces_jpeg_quality = harvest_settings["jpeg_quality"]

    # Show compact settings summary
    export_bits: list[str] = []
    if faces_save_frames:
        export_bits.append("frames")
    if faces_save_crops:
        export_bits.append("crops")
    export_text = "saving " + " & ".join(export_bits) if export_bits else "crops only"
    interval_note = ""
    if video_meta:
        fps_detected = helpers.coerce_float(video_meta.get("fps_detected")) or helpers.coerce_float(video_meta.get("fps"))
        if fps_detected and fps_detected > 0:
            interval_note = f" (~{faces_min_frames_between_crops / fps_detected:.2f}s @ {fps_detected:.2f}fps)"
    st.info(
        f"**Faces Harvest** → {faces_device_label} · {export_text}, "
        f"crop interval {faces_min_frames_between_crops} frames{interval_note}"
    )

    # Estimate counts
    harvest_frame_est = (
        helpers.coerce_int(detect_phase_status.get("frames_exported"))
        or helpers.coerce_int(detect_phase_status.get("sampled_frames"))
    )
    harvest_faces_est = _count_manifest_rows(detections_path)
    if harvest_faces_est is None:
        harvest_faces_est = helpers.coerce_int(detect_phase_status.get("detections"))
    if harvest_faces_est is None and harvest_frame_est:
        harvest_faces_est = int(harvest_frame_est * AVG_FACES_PER_FRAME)
    harvest_estimates: list[str] = []
    if faces_save_frames and harvest_frame_est:
        frame_bytes = int(harvest_frame_est * FRAME_JPEG_SIZE_EST_BYTES)
        harvest_estimates.append(f"frames ≈{helpers.human_size(frame_bytes)}")
    if faces_save_crops and harvest_faces_est:
        crop_bytes = int(harvest_faces_est * CROP_JPEG_SIZE_EST_BYTES)
        harvest_estimates.append(f"crops ≈{helpers.human_size(crop_bytes)}")
    if harvest_estimates:
        st.caption("Estimated output: " + " + ".join(harvest_estimates))

    # Improved messaging for when Harvest Faces is disabled
    if not local_video_exists:
        s3_meta = details.get("s3") or {}
        s3_exists = s3_meta.get("v2_exists") or s3_meta.get("v1_exists")
        if s3_exists:
            st.info("Local mirror missing; video will be mirrored from S3 automatically when Faces Harvest starts.")
        else:
            st.warning("Video not found locally or in S3. Upload the video first.")
    elif faces_status_value == "stale":
        st.warning(
            "**Harvest Faces is outdated**: Detect/Track was rerun after the last faces harvest.\n\n"
            "Track IDs have changed. Rerun **Faces Harvest** to rebuild embeddings for the new tracks."
        )
    elif not tracks_ready:
        message = (
            "**Harvest Faces is unavailable**: Face detection/tracking has not run yet.\n\n"
            "Run **Detect/Track Faces** first to generate `detections.jsonl` and `tracks.jsonl` for this episode."
        )
        if tracks_only_fallback:
            message = (
                "**Harvest Faces is unavailable**: Tracks exist but detections are missing.\n\n"
                "Run **Detect/Track Faces** again to regenerate detections before harvesting faces."
            )
        st.warning(message)
        if detect_phase_status and detect_phase_status.get("detector") == "pyscenedetect":
            st.error(
                "⚠️ **Scene detection only**: Your last run only executed scene detection (PySceneDetect), "
                "not full face detection + tracking. Please run **Detect/Track Faces** again to generate tracks."
            )
    elif faces_status_value == "running":
        st.info("Faces harvest is running. Progress will update automatically; clustering remains disabled until completion.")
    elif not detector_face_only:
        if detector_manifest_value is None:
            st.warning(
                "Unable to determine which detector produced the current tracks. Rerun Detect/Track Faces "
                "with RetinaFace + ByteTrack before harvesting."
            )
        else:
            st.warning(
                f"Current tracks were generated with unsupported detector "
                f"{helpers.detector_label_from_value(detector_manifest_value)}. Rerun Detect/Track Faces "
                "with a supported detector/tracker before harvesting."
            )
    elif not combo_supported_harvest:
        current_combo = f"{helpers.detector_label_from_value(combo_detector)} + {helpers.tracker_label_from_value(combo_tracker)}"
        st.error(
            f"Harvest requires a supported detector/tracker combo. Last detect run used **{current_combo}**. "
            "Select a supported combo (e.g., RetinaFace + ByteTrack/StrongSORT) and rerun detect/track."
        )

    # Check if tracks just completed (bypasses stale status check for auto-run progression)
    # Bug 4 fix: Use timestamp instead of boolean to survive multiple reruns
    tracks_completed_at = st.session_state.get(f"{ep_id}::tracks_completed_at")
    tracks_just_completed = tracks_completed_at is not None and (time.time() - tracks_completed_at < 30)
    # If tracks just completed, treat as ready even if API hasn't updated
    effective_tracks_ready = tracks_ready or tracks_just_completed

    faces_disabled = (
        (not effective_tracks_ready)
        or (not detector_face_only)
        or job_running
        or faces_status_value == "running"
        or (not combo_supported_harvest)
        or tracks_only_fallback
        or running_faces_job is not None
    )

    if running_faces_job:
        st.warning(f"⚠️ A faces harvest job is already running ({running_faces_job.get('progress_pct', 0):.1f}% complete). Cancel it above to start a new one.")

    # Check for auto-run trigger from pipeline automation
    # Bug 1 fix: Use .get() instead of .pop() - only clear after job successfully starts
    autorun_faces_trigger = st.session_state.get(f"{ep_id}::autorun_faces_trigger", False)

    # Debug: Show why faces might be disabled (only when auto-run is active)
    if autorun_active and autorun_faces_trigger:
        with st.expander("🔍 Faces Phase Debug", expanded=True):
            st.markdown(f"**Trigger received**: `autorun_faces_trigger={autorun_faces_trigger}`")
            st.markdown(f"**faces_disabled**: `{faces_disabled}`")
            if faces_disabled:
                st.markdown("**Disabled because:**")
                reasons = []
                if not effective_tracks_ready:
                    reasons.append(f"- tracks not ready (tracks_ready={tracks_ready}, tracks_just_completed={tracks_just_completed})")
                if not detector_face_only:
                    reasons.append(f"- detector not face-only ({detector_face_only=})")
                if job_running:
                    reasons.append(f"- job already running ({job_running=})")
                if faces_status_value == "running":
                    reasons.append(f"- faces status is 'running' ({faces_status_value=})")
                if not combo_supported_harvest:
                    reasons.append(f"- combo not supported ({combo_supported_harvest=})")
                if tracks_only_fallback:
                    reasons.append(f"- tracks only fallback ({tracks_only_fallback=})")
                if running_faces_job is not None:
                    reasons.append(f"- faces job already running ({running_faces_job=})")
                st.code("\n".join(reasons) if reasons else "No specific reason found", language="text")

    should_run_faces = st.button("Run Faces Harvest", use_container_width=True, disabled=faces_disabled)

    # Auto-run trigger: simulate button click if auto-run is active and not disabled
    if autorun_faces_trigger:
        if not faces_disabled:
            should_run_faces = True
            # Bug 1 fix: Clear trigger AFTER we confirm job will start
            st.session_state.pop(f"{ep_id}::autorun_faces_trigger", None)
            # Bug 3 fix: Reset retry counter on success
            st.session_state.pop(f"{ep_id}::autorun_faces_retry", None)
            st.info("🤖 Auto-Run: Starting Faces Harvest...")
        else:
            # Suggestion 8: Mtime-based retry with longer limit
            # FIX: Reduced retry delay from 1s to 0.25s for faster phase advancement
            retry_count = st.session_state.get(f"{ep_id}::autorun_faces_retry", 0) + 1
            should_retry, status_msg = _should_retry_phase_trigger(ep_id, "faces", retry_count)
            if should_retry:
                st.session_state[f"{ep_id}::autorun_faces_retry"] = retry_count
                st.caption(f"⏳ {status_msg}")
                time.sleep(0.25)  # Reduced from 1s for faster advancement
                st.rerun()
            else:
                # Give up after max retries
                st.session_state.pop(f"{ep_id}::autorun_faces_trigger", None)
                st.session_state.pop(f"{ep_id}::autorun_faces_retry", None)
                st.session_state[_autorun_key] = False
                st.session_state[_autorun_phase_key] = None
                st.error(f"❌ Auto-run stopped: {status_msg}")

    if should_run_faces:
        can_run_faces = True
        if not local_video_exists:
            can_run_faces = _ensure_local_artifacts(ep_id, details)
            if can_run_faces:
                local_video_exists = True
        if can_run_faces:
            payload = {
                "ep_id": ep_id,
                "device": faces_device_value,
                "profile": profile_value,
                "save_frames": bool(faces_save_frames),
                "save_crops": bool(faces_save_crops),
                "min_frames_between_crops": int(faces_min_frames_between_crops),
                "jpeg_quality": int(faces_jpeg_quality),
                "thumb_size": int(faces_thumb_size),
            }
            if selected_attempt_run_id:
                payload["run_id"] = selected_attempt_run_id
            elif autorun_active:
                autorun_run_id = st.session_state.get(_autorun_run_id_key)
                if isinstance(autorun_run_id, str) and autorun_run_id.strip():
                    payload["run_id"] = autorun_run_id.strip()
            st.session_state[running_job_key] = True
            # Clear completion marker when starting new job
            st.session_state.pop(f"{ep_id}::faces_job_complete", None)
            _set_job_active(ep_id, True)
            try:
                # Use execution mode from UI settings (respects local/redis toggle)
                execution_mode = helpers.get_execution_mode(ep_id)
                mode_desc = "local" if execution_mode == "local" else "Celery"
                if execution_mode == "local":
                    # Local mode handles its own UI - no spinner needed
                    summary, error_message = helpers.run_pipeline_job_with_mode(
                        ep_id,
                        "faces_embed",
                        payload,
                        requested_device=faces_device_value,
                        requested_detector=helpers.tracks_detector_value(ep_id),
                        requested_tracker=helpers.tracks_tracker_value(ep_id),
                    )
                else:
                    with st.spinner(f"Running faces harvest via {mode_desc}…"):
                        summary, error_message = helpers.run_pipeline_job_with_mode(
                            ep_id,
                            "faces_embed",
                            payload,
                            requested_device=faces_device_value,
                            requested_detector=helpers.tracks_detector_value(ep_id),
                            requested_tracker=helpers.tracks_tracker_value(ep_id),
                        )
            finally:
                st.session_state[running_job_key] = False
                _set_job_active(ep_id, False)

            # IMPORTANT: Set auto-run progression FIRST, before any UI display
            # This ensures the next phase is triggered even if UI display has issues
            LOGGER.info("[AUTORUN] Faces job returned: summary=%s, error=%s", bool(summary), error_message)
            # Synthesize summary if empty but no error (parity with detect path)
            if not summary and not error_message:
                LOGGER.warning("[AUTORUN] Faces summary missing, synthesizing from manifests")
                summary = {"status": "completed", "fallback": True}
            # If summary exists but status missing, set completed to allow progression
            if summary and not summary.get("status"):
                summary["status"] = "completed"
            if summary and not error_message:
                st.session_state[_status_force_refresh_key(ep_id)] = True
                st.session_state[f"{ep_id}::faces_completed_at"] = time.time()  # Bug 4 fix: Use timestamp
                # Invalidate ALL caches to force fresh status on next page load
                helpers.invalidate_running_jobs_cache(ep_id)
                _cached_local_jobs.clear()
                # Clear status cache mtimes to force refetch (these are recalculated on page load)
                st.session_state.pop(_status_mtimes_key(ep_id), None)
                st.session_state.pop(_status_cache_key(ep_id), None)
                autorun_val = st.session_state.get(_autorun_key)
                phase_val = st.session_state.get(_autorun_phase_key)
                LOGGER.info("[AUTORUN] After faces: autorun=%s, phase=%s", autorun_val, phase_val)
                if autorun_val and phase_val == "faces":
                    # Log completed stage with counts from summary
                    normalized = helpers.normalize_summary(ep_id, summary)
                    faces_count = helpers.coerce_int(normalized.get("faces"))

                    # FALLBACK: If normalize_summary didn't get faces count, read directly from manifest
                    if faces_count is None or faces_count == 0:
                        faces_manifest_path = faces_path
                        if faces_manifest_path.exists():
                            try:
                                with faces_manifest_path.open("r", encoding="utf-8") as fh:
                                    faces_count = sum(1 for line in fh if line.strip())
                                LOGGER.info("[AUTORUN] Fallback faces_count from manifest: %s", faces_count)
                            except OSError:
                                pass

                    # Bug E fix: Validate faces count before advancing to cluster
                    if faces_count is None or faces_count == 0:
                        LOGGER.warning("[AUTORUN] Stopping: zero faces harvested (faces=%s)", faces_count)
                        st.session_state[_autorun_key] = False
                        st.session_state[_autorun_phase_key] = None
                        st.session_state[f"{ep_id}::autorun_error"] = f"Zero faces harvested - cannot cluster"
                        st.error(f"❌ Auto-run stopped: No faces harvested (faces={faces_count or 0}). Nothing to cluster.")
                        st.rerun()

                    completed = st.session_state.get(f"{ep_id}::autorun_completed_stages", [])
                    completed.append(f"Faces Harvest ({faces_count:,} faces)")
                    st.session_state[f"{ep_id}::autorun_completed_stages"] = completed
                    st.session_state[_autorun_phase_key] = "cluster"
                    st.session_state[f"{ep_id}::autorun_cluster_trigger"] = True

                    # Verify state was set before rerun
                    verify_phase = st.session_state.get(_autorun_phase_key)
                    verify_trigger = st.session_state.get(f"{ep_id}::autorun_cluster_trigger")
                    LOGGER.info("[AUTORUN] Set phase=cluster, trigger=True. Verify: phase=%s, trigger=%s", verify_phase, verify_trigger)

                    if verify_phase != "cluster" or not verify_trigger:
                        LOGGER.error("[AUTORUN] FAILED to set cluster trigger state!")
                        st.error("⚠️ Auto-run state error: Could not set cluster trigger")
                        st.session_state[_autorun_key] = False
                        st.session_state[_autorun_phase_key] = None
                        # Don't call rerun - let the page render normally
                    else:
                        st.toast("✅ Faces Harvest complete - advancing to Cluster...")
                        LOGGER.info("[AUTORUN] About to st.rerun() to trigger cluster phase")
                        # CRITICAL: Rerun immediately to trigger cluster phase
                        # Don't wait for end of function which may have early returns
                        st.rerun()

            if error_message:
                # Bug 2 fix: Stop auto-run on error to prevent getting stuck
                if st.session_state.get(_autorun_key):
                    st.session_state[_autorun_key] = False
                    st.session_state[_autorun_phase_key] = None
                    st.session_state[f"{ep_id}::autorun_error"] = error_message
                    st.error(f"❌ Auto-run stopped due to error in Faces Harvest phase.")

                if "tracks.jsonl" in error_message.lower():
                    st.error("Run detect/track first.")
                else:
                    st.error(error_message)
            else:
                normalized = helpers.normalize_summary(ep_id, summary)
                faces_count = normalized.get("faces")
                crops_exported = normalized.get("crops_exported")
                flash_parts = []
                if isinstance(faces_count, int):
                    flash_parts.append(f"faces: {faces_count:,}")
                if crops_exported:
                    flash_parts.append(f"crops exported: {crops_exported:,}")
                flash_parts.append(f"thumb size: {int(faces_thumb_size)}px")
                flash_msg = "Faces harvest complete" + (" · " + ", ".join(flash_parts) if flash_parts else "")
                st.session_state["episode_detail_flash"] = flash_msg
                # Note: status refresh and auto-run progression are now set BEFORE this block
                st.rerun()

    # Show previous run logs (only in local mode, collapsed by default)
    if helpers.get_execution_mode(ep_id) == "local":
        helpers.render_previous_logs(ep_id, "faces_embed", expanded=False)

with col_cluster:
    # CRITICAL: Check for faces_embed completion flag set by ui_helpers
    # This catches cases where the return value didn't reach the caller
    # (Streamlit may restart script on widget updates during streaming)
    _faces_just_completed_key = f"{ep_id}::faces_embed_just_completed"
    if st.session_state.get(_faces_just_completed_key):
        LOGGER.info("[AUTORUN] Detected faces_embed_just_completed flag, advancing to cluster phase")
        # Get the stored completion context (do not clear until we successfully handle it).
        stored_summary = st.session_state.get(f"{ep_id}::faces_embed_summary")
        completed_at = st.session_state.get(f"{ep_id}::faces_embed_completed_at")

        # Set faces_completed_at if not already set
        if not st.session_state.get(f"{ep_id}::faces_completed_at"):
            st.session_state[f"{ep_id}::faces_completed_at"] = completed_at or time.time()
            LOGGER.info("[AUTORUN] Set faces_completed_at from faces_embed completion flag")

        # If auto-run is active, ALWAYS ensure the cluster trigger is set after faces complete.
        # This is intentionally tolerant of phase desync (e.g., phase already moved to "cluster"
        # but the trigger was missed, or the script was restarted mid-stream).
        autorun_active_now = bool(st.session_state.get(_autorun_key))
        phase_now = st.session_state.get(_autorun_phase_key)
        cluster_trigger_key = f"{ep_id}::autorun_cluster_trigger"
        cluster_trigger_now = bool(st.session_state.get(cluster_trigger_key))

        if autorun_active_now and not cluster_trigger_now:
            faces_count: int | None = None
            if isinstance(stored_summary, dict) and stored_summary:
                normalized = helpers.normalize_summary(ep_id, stored_summary)
                faces_count = helpers.coerce_int(normalized.get("faces"))
            if faces_count is None:
                try:
                    if faces_path.exists():
                        faces_count = sum(1 for line in faces_path.open("r", encoding="utf-8") if line.strip())
                except OSError:
                    faces_count = None

            completed = st.session_state.get(f"{ep_id}::autorun_completed_stages", [])
            if not any("Faces Harvest" in s for s in completed):
                faces_count_display = f"{faces_count:,}" if isinstance(faces_count, int) else "?"
                completed.append(f"Faces Harvest ({faces_count_display} faces)")
                st.session_state[f"{ep_id}::autorun_completed_stages"] = completed

            # Advance to cluster phase and trigger the next stage.
            st.session_state[_autorun_phase_key] = "cluster"
            st.session_state[cluster_trigger_key] = True
            # Prevent faces from being repeatedly re-triggered after completion.
            st.session_state.pop(f"{ep_id}::autorun_faces_trigger", None)
            st.session_state[_status_force_refresh_key(ep_id)] = True
            LOGGER.info(
                "[AUTORUN] Promoted faces completion -> cluster trigger (phase was %s, faces=%s)",
                phase_now,
                faces_count,
            )

            # Clear completion signal now that we've promoted it.
            st.session_state.pop(_faces_just_completed_key, None)
            st.session_state.pop(f"{ep_id}::faces_embed_summary", None)

            st.toast("✅ Faces Harvest complete - starting Clustering...")
            st.rerun()

        # No promotion needed (manual run, or cluster already queued); clear the completion signal.
        st.session_state.pop(_faces_just_completed_key, None)
        st.session_state.pop(f"{ep_id}::faces_embed_summary", None)

    # CRITICAL: Check for cluster completion flag set by ui_helpers
    # This handles the final phase where cluster completes and marks auto-run done
    _cluster_just_completed_key = f"{ep_id}::cluster_just_completed"
    if st.session_state.get(_cluster_just_completed_key):
        LOGGER.info("[AUTORUN] Detected cluster_just_completed flag, finalizing auto-run")
        # Clear the flag to prevent re-processing
        st.session_state.pop(_cluster_just_completed_key, None)
        # Get the summary that was stored
        stored_summary = st.session_state.pop(f"{ep_id}::cluster_summary", None)

        # Check if auto-run is active and waiting on cluster phase
        if st.session_state.get(_autorun_key) and st.session_state.get(_autorun_phase_key) == "cluster":
            # Build completed stage entry
            if stored_summary:
                normalized = helpers.normalize_summary(ep_id, stored_summary)
                identities_count = helpers.coerce_int(normalized.get("identities"))
            else:
                identities_count = None
            # Fallback to manifest count
            if identities_count is None:
                try:
                    if identities_path.exists():
                        import json
                        data = json.loads(identities_path.read_text(encoding="utf-8"))
                        if isinstance(data, dict) and isinstance(data.get("identities"), list):
                            identities_count = len(data["identities"])
                except (OSError, json.JSONDecodeError):
                    pass
            # Log completed stage
            completed = st.session_state.get(f"{ep_id}::autorun_completed_stages", [])
            stage_label = f"Cluster ({identities_count or '?':,} identities)"
            if stage_label not in completed:
                completed.append(stage_label)
                st.session_state[f"{ep_id}::autorun_completed_stages"] = completed

            # Trigger PDF export before marking auto-run complete (idempotent)
            if selected_attempt_run_id:
                with st.spinner("Generating PDF report..."):
                    pdf_success, pdf_msg = _trigger_pdf_export_if_needed(
                        ep_id, selected_attempt_run_id, cfg
                    )
                if pdf_success:
                    st.success(f"📄 {pdf_msg}")
                    # Add PDF stage to completed stages
                    pdf_stage = "PDF Export"
                    if pdf_stage not in completed:
                        completed.append(pdf_stage)
                        st.session_state[f"{ep_id}::autorun_completed_stages"] = completed
                else:
                    st.warning(f"⚠️ {pdf_msg}")
            else:
                LOGGER.warning("[AUTORUN] No run_id available for PDF export")

            # Mark auto-run complete
            st.session_state[_autorun_key] = False
            st.session_state[_autorun_phase_key] = None
            st.session_state[_status_force_refresh_key(ep_id)] = True
            LOGGER.info("[AUTORUN] Auto-run pipeline complete via completion flag")
            st.success("🎉 **Auto-Run Pipeline Complete!** All phases finished successfully.")
            st.rerun()

    st.markdown("### Cluster Identities")
    st.caption(_format_phase_status("Cluster Identities", cluster_phase_status, "identities"))

    # Show running job progress if a job is active
    # Skip if we already marked this job as complete (prevents infinite refresh loop)
    cluster_job_complete_key = f"{ep_id}::cluster_job_complete"
    if running_cluster_job and not st.session_state.get(cluster_job_complete_key):
        # Bug 6 fix: Generate unique key even if job_id is missing
        job_id = running_cluster_job.get("job_id") or f"cluster_{hash(str(running_cluster_job)) % 10000}"
        progress_pct = running_cluster_job.get("progress_pct", 0)
        state = running_cluster_job.get("state", "running")

        # Auto-refresh when job hits 100% or state indicates completion
        job_complete = progress_pct >= 99.5 or state in ("done", "success", "completed")
        if job_complete:
            st.success("✅ **Cluster complete!**")
            # Mark job as complete to prevent infinite refresh loop
            st.session_state[cluster_job_complete_key] = True
            # Force status refresh to pick up new data
            st.session_state[_status_force_refresh_key(ep_id)] = True
            # Trigger Improve Faces when cluster finishes (run-scoped).
            if selected_attempt_run_id:
                st.session_state[_improve_faces_state_key(ep_id, selected_attempt_run_id, "trigger")] = True

            # Check if auto-run is active - this is the final phase, trigger PDF export and mark complete
            if st.session_state.get(_autorun_key) and st.session_state.get(_autorun_phase_key) == "cluster":
                # Trigger PDF export before marking auto-run complete (idempotent)
                if selected_attempt_run_id:
                    with st.spinner("Generating PDF report..."):
                        pdf_success, pdf_msg = _trigger_pdf_export_if_needed(
                            ep_id, selected_attempt_run_id, cfg
                        )
                    if pdf_success:
                        st.success(f"📄 {pdf_msg}")
                    else:
                        st.warning(f"⚠️ {pdf_msg}")
                st.success("🎉 **Auto-Run Pipeline Complete!** All phases finished successfully.")
                st.session_state[_autorun_key] = False
                st.session_state[_autorun_phase_key] = None
            # Trigger Improve Faces modal on this page (not redirect)
            if selected_attempt_run_id:
                st.caption("Opening Improve Faces...")
            else:
                st.caption("Cluster complete (legacy run). Select a run-scoped attempt (run_id) to use Improve Faces.")
            time.sleep(0.3)
            st.rerun()

        st.info(f"🔄 **Cluster job running** ({state})")
        st.progress(min(progress_pct / 100, 1.0))
        st.caption(f"Progress: {progress_pct:.1f}%")

        # Refresh and Cancel buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🔄 Refresh", key=f"refresh_cluster_{job_id}", use_container_width=True):
                st.rerun()
        with btn_col2:
            if st.button("❌ Cancel", key=f"cancel_cluster_{job_id}", use_container_width=True):
                success, msg = helpers.cancel_running_job(job_id)
                if success:
                    st.success(msg)
                    time.sleep(0.3)  # Suggestion 3: Reduced delay
                    st.rerun()
                else:
                    st.error(msg)

        st.divider()

    # ─── GET SETTINGS FROM PIPELINE SETTINGS DIALOG ──────────────────────────
    # All settings are now managed in the unified Pipeline Settings dialog (gear icon)
    cluster_settings = _get_cluster_settings(ep_id)

    cluster_device_value = cluster_settings["device"]
    cluster_device_label = cluster_settings["device_label"]
    cluster_thresh_value = cluster_settings["cluster_thresh"]
    min_cluster_size_value = cluster_settings["min_cluster_size"]

    # Threshold guidance
    if cluster_thresh_value >= 0.80:
        thresh_hint = "🔴 Very strict"
    elif cluster_thresh_value >= 0.70:
        thresh_hint = "🟡 Strict"
    elif cluster_thresh_value >= 0.55:
        thresh_hint = "🟢 Balanced"
    else:
        thresh_hint = "🟠 Lenient"

    # Show compact settings summary
    st.info(
        f"**Clustering** → {cluster_device_label} · threshold {cluster_thresh_value:.2f} ({thresh_hint}), "
        f"min tracks {min_cluster_size_value}"
    )

    if not local_video_exists:
        s3_meta = details.get("s3") or {}
        s3_exists = s3_meta.get("v2_exists") or s3_meta.get("v1_exists")
        if s3_exists:
            st.info("Local mirror missing; artifacts will be mirrored automatically when clustering starts.")
        else:
            st.warning("Video not found locally or in S3. Upload the video first.")
    elif not tracks_ready:
        st.caption("Run detect/track first; clustering requires fresh tracks and faces.")

    # Check if faces harvest succeeded with zero faces
    zero_faces_success = faces_status_value == "success" and (
        (faces_count_value is not None and faces_count_value == 0)
        or (faces_manifest_count == 0)
    )

    if zero_faces_success:
        st.info("Faces harvest completed with 0 faces. Clustering is disabled until faces are available.")
    elif cluster_status_value == "stale":
        st.warning(
            "**Cluster is outdated**: Detect/Track was rerun after the last clustering.\n\n"
            "Track IDs have changed. Rerun **Faces Harvest** and then **Cluster** to rebuild identities."
        )
    elif faces_status_value == "stale":
        st.warning(
            "**Faces are outdated**: Detect/Track was rerun after the last faces harvest.\n\n"
            "Rerun **Faces Harvest** first, then cluster."
        )
    elif not faces_ready:
        if faces_status_value == "running":
            st.caption("Faces harvest is running — wait for it to finish before clustering.")
        elif faces_status_value == "error":
            st.error("Faces harvest failed. Rerun harvest to generate embeddings before clustering.")
        elif faces_status_value == "success":
            st.warning("Faces manifest not mirrored locally. Mirror artifacts before clustering.")
        else:
            st.caption("Run faces harvest first.")
    elif not detector_face_only:
        st.warning("Current tracks were generated with a legacy detector. Rerun detect/track first.")
    elif not combo_supported_cluster:
        combo_label = f"{helpers.detector_label_from_value(combo_detector)} + {helpers.tracker_label_from_value(combo_tracker)}"
        st.error(
            f"Cluster requires RetinaFace + ByteTrack tracks. Last detect run used **{combo_label}**. "
            "Rerun detect/track with the supported combo before clustering."
        )
    elif cluster_status_value == "running":
        st.info("Clustering is currently running. Wait for it to complete before starting another run.")

    # Check if faces just completed (bypasses stale status check for auto-run progression)
    # Bug 4 fix: Use timestamp instead of boolean to survive multiple reruns
    # Extended timeout from 30s to 120s to survive double-rerun and slow status refresh
    faces_completed_at = st.session_state.get(f"{ep_id}::faces_completed_at")
    faces_just_completed = faces_completed_at is not None and (time.time() - faces_completed_at < 120)
    # If faces just completed, treat the status as fresh even if API says stale
    effective_faces_stale = faces_status_value == "stale" and not faces_just_completed
    # Similarly, if faces just completed, assume faces are ready
    effective_faces_ready = faces_ready or faces_just_completed

    # CRITICAL FIX: If cluster trigger is active from auto-run, faces JUST completed
    # The trigger was set immediately after faces completed successfully, so trust it
    # This bypasses status caching issues that can cause cluster to never start
    cluster_trigger_active = st.session_state.get(f"{ep_id}::autorun_cluster_trigger", False)
    if cluster_trigger_active:
        LOGGER.info(
            "[AUTORUN CLUSTER] Trigger active - bypassing status checks. "
            "Original: faces_ready=%s, faces_just_completed=%s, effective_faces_ready=%s, effective_faces_stale=%s",
            faces_ready, faces_just_completed, effective_faces_ready, effective_faces_stale
        )
        # Force these values since we KNOW faces just completed (trigger was set right after)
        faces_just_completed = True
        effective_faces_ready = True
        effective_faces_stale = False

    # Cluster relies on tracks.jsonl; use the run-scoped manifest as the most direct readiness signal.
    tracks_manifest_ready = _manifest_has_rows(tracks_path)
    effective_tracks_ready_for_cluster = tracks_ready or tracks_manifest_ready

    cluster_disabled = (
        (not effective_faces_ready)
        or (not detector_face_only)
        or (not effective_tracks_ready_for_cluster)
        or job_running
        or zero_faces_success
        or (not combo_supported_cluster)
        or effective_faces_stale
        or cluster_status_value == "running"
        or running_cluster_job is not None
    )

    # Debug logging for cluster_disabled calculation
    if autorun_active:
        LOGGER.info(
            "[AUTORUN CLUSTER] cluster_disabled=%s: effective_faces_ready=%s, detector_face_only=%s, "
            "tracks_ready=%s (manifest=%s, effective=%s), job_running=%s, zero_faces_success=%s, combo_supported=%s, "
            "effective_faces_stale=%s, cluster_status=%s, running_cluster_job=%s, trigger_active=%s",
            cluster_disabled, effective_faces_ready, detector_face_only, tracks_ready, tracks_manifest_ready,
            effective_tracks_ready_for_cluster, job_running, zero_faces_success, combo_supported_cluster,
            effective_faces_stale, cluster_status_value,
            running_cluster_job is not None, cluster_trigger_active
        )

    if running_cluster_job:
        st.warning(f"⚠️ A cluster job is already running ({running_cluster_job.get('progress_pct', 0):.1f}% complete). Cancel it above to start a new one.")

    def _auto_group_clusters(ep_id: str) -> Tuple[Dict[str, Any] | None, str | None]:
        run_id = selected_attempt_run_id or _resolve_session_run_id(ep_id)
        if not run_id:
            return None, "Select a non-legacy attempt (run_id) before grouping clusters."
        payload = {
            "strategy": "auto",
            "protect_manual": True,
            "facebank_first": True,
            "skip_cast_assignment": False,  # Auto-assign clusters to cast members
        }
        try:
            resp = helpers.api_post(
                f"/episodes/{ep_id}/clusters/group",
                json=payload,
                params={"run_id": run_id},
                timeout=300,
            )
        except requests.RequestException as exc:
            return None, helpers.describe_error(f"{cfg['api_base']}/episodes/{ep_id}/clusters/group", exc)
        if not resp:
            return None, "Grouping API returned no response"
        if isinstance(resp, dict):
            err_msg = resp.get("error") or resp.get("detail")
            status_value = str(resp.get("status") or "").lower()
            if status_value and status_value not in {"success", "ok"} and not err_msg:
                err_msg = f"Unexpected status: {status_value}"
            if err_msg:
                return None, str(err_msg)
        return resp, None

    def _group_flash_text(group_response: Dict[str, Any]) -> str | None:
        group_result = group_response.get("result") if isinstance(group_response, dict) else None
        if not isinstance(group_result, dict):
            return "Auto-group complete"
        within = group_result.get("within_episode") or {}
        across = group_result.get("across_episodes") or {}
        merged_groups = helpers.coerce_int(within.get("merged_count"))
        assignments = group_result.get("assignments")
        if isinstance(assignments, dict):
            assignments = assignments.get("assigned") or assignments.get("assignments")
        if assignments is None:
            assignments = across.get("assigned")
        assigned_count = len(assignments or []) if isinstance(assignments, list) else 0
        new_people = helpers.coerce_int(across.get("new_people_count"))
        facebank_assigned = helpers.coerce_int(group_result.get("facebank_assigned"))
        parts = []
        if merged_groups:
            parts.append(f"merged {merged_groups} group(s)")
        if assigned_count:
            parts.append(f"assigned {assigned_count} cluster(s)")
        if new_people:
            parts.append(f"{new_people} new people")
        if facebank_assigned:
            parts.append(f"{facebank_assigned} facebank matches")
        if not parts:
            return "Auto-group complete (draft people stay in Needs Cast Assignment)"
        return "Auto-grouped " + ", ".join(parts) + " (draft people stay in Needs Cast Assignment)"

    # Check for auto-run trigger from pipeline automation
    # Bug 1 fix: Use .get() instead of .pop() - only clear after job successfully starts
    autorun_cluster_trigger = st.session_state.get(f"{ep_id}::autorun_cluster_trigger", False)

    # Debug: Show why cluster might be disabled (only when auto-run is active)
    if autorun_active and autorun_cluster_trigger:
        with st.expander("🔍 Cluster Phase Debug", expanded=True):
            st.markdown(f"**Trigger received**: `autorun_cluster_trigger={autorun_cluster_trigger}`")
            st.markdown(f"**cluster_disabled**: `{cluster_disabled}`")
            if cluster_disabled:
                st.markdown("**Disabled because:**")
                reasons = []
                if not effective_faces_ready:
                    reasons.append(f"- faces not ready (faces_ready={faces_ready}, faces_just_completed={faces_just_completed})")
                if not detector_face_only:
                    reasons.append(f"- detector not face-only ({detector_face_only=})")
                if not effective_tracks_ready_for_cluster:
                    reasons.append(
                        f"- tracks not ready (tracks_ready={tracks_ready}, tracks_manifest_ready={tracks_manifest_ready})"
                    )
                if job_running:
                    reasons.append(f"- job already running ({job_running=})")
                if zero_faces_success:
                    reasons.append(f"- zero faces success ({zero_faces_success=})")
                if not combo_supported_cluster:
                    reasons.append(f"- combo not supported ({combo_supported_cluster=})")
                if effective_faces_stale:
                    reasons.append(f"- faces stale (faces_status_value={faces_status_value}, faces_just_completed={faces_just_completed})")
                if cluster_status_value == "running":
                    reasons.append(f"- cluster status is 'running' ({cluster_status_value=})")
                if running_cluster_job is not None:
                    reasons.append(f"- cluster job already running ({running_cluster_job=})")
                st.code("\n".join(reasons) if reasons else "No specific reason found", language="text")

    should_run_cluster = st.button("Run Cluster", use_container_width=True, disabled=cluster_disabled)

    # Auto-run trigger: simulate button click if auto-run is active and not disabled
    if autorun_cluster_trigger:
        if not cluster_disabled:
            should_run_cluster = True
            # Bug 1 fix: Clear trigger AFTER we confirm job will start
            st.session_state.pop(f"{ep_id}::autorun_cluster_trigger", None)
            # Bug 3 fix: Reset retry counter on success
            st.session_state.pop(f"{ep_id}::autorun_cluster_retry", None)
            st.info("🤖 Auto-Run: Starting Cluster...")
        else:
            # Suggestion 8: Mtime-based retry with longer limit
            # FIX: Reduced retry delay from 1s to 0.25s for faster phase advancement
            retry_count = st.session_state.get(f"{ep_id}::autorun_cluster_retry", 0) + 1
            should_retry, status_msg = _should_retry_phase_trigger(ep_id, "cluster", retry_count)
            if should_retry:
                st.session_state[f"{ep_id}::autorun_cluster_retry"] = retry_count
                st.caption(f"⏳ {status_msg}")
                time.sleep(0.25)  # Reduced from 1s for faster advancement
                st.rerun()
            else:
                # Give up after max retries
                st.session_state.pop(f"{ep_id}::autorun_cluster_trigger", None)
                st.session_state.pop(f"{ep_id}::autorun_cluster_retry", None)
                st.session_state[_autorun_key] = False
                st.session_state[_autorun_phase_key] = None
                st.error(f"❌ Auto-run stopped: {status_msg}")

    if should_run_cluster:
        can_run_cluster = True
        if not local_video_exists:
            can_run_cluster = _ensure_local_artifacts(ep_id, details)
            if can_run_cluster:
                local_video_exists = True
        # Ensure faces manifest is mirrored locally before clustering
        if can_run_cluster and not faces_path.exists():
            with st.spinner("Mirroring faces artifacts from S3…"):
                try:
                    # Use the new mirror_artifacts endpoint that actually mirrors faces/identities
                    mirror_resp = helpers.api_post(
                        f"/episodes/{ep_id}/mirror_artifacts",
                        json={"artifacts": ["faces", "identities"]},
                    )
                    if mirror_resp.get("faces_manifest_exists"):
                        st.success("Faces manifest mirrored successfully.")
                    else:
                        errors = mirror_resp.get("errors", {})
                        error_msg = errors.get("faces", "Faces manifest not found in S3")
                        st.error(f"Failed to mirror faces: {error_msg}")
                        can_run_cluster = False
                except requests.RequestException as exc:
                    st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/{ep_id}/mirror_artifacts", exc))
                    can_run_cluster = False
        if can_run_cluster:
            payload = {
                "ep_id": ep_id,
                "device": cluster_device_value,
                "cluster_thresh": float(cluster_thresh_value),
                "min_cluster_size": int(min_cluster_size_value),
                "profile": profile_value,
            }
            if selected_attempt_run_id:
                payload["run_id"] = selected_attempt_run_id
            elif autorun_active:
                autorun_run_id = st.session_state.get(_autorun_run_id_key)
                if isinstance(autorun_run_id, str) and autorun_run_id.strip():
                    payload["run_id"] = autorun_run_id.strip()
            st.session_state[running_job_key] = True
            # Clear completion marker when starting new job
            st.session_state.pop(f"{ep_id}::cluster_job_complete", None)
            _set_job_active(ep_id, True)
            try:
                # Use execution mode from UI settings (respects local/redis toggle)
                execution_mode = helpers.get_execution_mode(ep_id)
                mode_desc = "local" if execution_mode == "local" else "Celery"
                if execution_mode == "local":
                    # Local mode handles its own UI - no spinner needed
                    summary, error_message = helpers.run_pipeline_job_with_mode(
                        ep_id,
                        "cluster",
                        payload,
                        requested_device=cluster_device_value,
                        requested_detector=helpers.tracks_detector_value(ep_id),
                        requested_tracker=helpers.tracks_tracker_value(ep_id),
                    )
                else:
                    with st.spinner(f"Clustering faces via {mode_desc}…"):
                        summary, error_message = helpers.run_pipeline_job_with_mode(
                            ep_id,
                            "cluster",
                            payload,
                            requested_device=cluster_device_value,
                            requested_detector=helpers.tracks_detector_value(ep_id),
                            requested_tracker=helpers.tracks_tracker_value(ep_id),
                        )
            finally:
                st.session_state[running_job_key] = False
                _set_job_active(ep_id, False)

            # IMPORTANT: Set auto-run completion FIRST, before any UI display
            # This ensures the pipeline is marked complete even if UI display has issues
            if summary and not error_message:
                st.session_state[_status_force_refresh_key(ep_id)] = True
                # Invalidate running jobs cache to ensure fresh state
                helpers.invalidate_running_jobs_cache(ep_id)
                _cached_local_jobs.clear()
                if st.session_state.get(_autorun_key) and st.session_state.get(_autorun_phase_key) == "cluster":
                    # Log completed stage with counts from summary
                    normalized = helpers.normalize_summary(ep_id, summary)
                    identities_count = normalized.get("identities")
                    faces_count = normalized.get("faces")
                    completed = st.session_state.get(f"{ep_id}::autorun_completed_stages", [])
                    id_count = identities_count if isinstance(identities_count, int) else "?"
                    fc_count = faces_count if isinstance(faces_count, int) else "?"
                    completed.append(f"Clustering ({id_count} identities, {fc_count} faces)")
                    st.session_state[f"{ep_id}::autorun_completed_stages"] = completed
                    st.session_state[_autorun_key] = False
                    st.session_state[_autorun_phase_key] = None
                    # Don't navigate away - let Improve Faces modal show on this page
                    st.toast("🎉 Auto-Run Pipeline Complete! Opening Improve Faces...")

            if error_message:
                # Bug 2 fix: Stop auto-run on error to prevent getting stuck
                if st.session_state.get(_autorun_key):
                    st.session_state[_autorun_key] = False
                    st.session_state[_autorun_phase_key] = None
                    st.session_state[f"{ep_id}::autorun_error"] = error_message
                    st.error(f"❌ Auto-run stopped due to error in Cluster phase.")

                if "faces.jsonl" in error_message.lower():
                    st.error("Run faces harvest first.")
                else:
                    st.error(error_message)
            else:
                normalized = helpers.normalize_summary(ep_id, summary)
                identities_count = normalized.get("identities")
                faces_count = normalized.get("faces")
                cluster_flash_parts = []
                if isinstance(identities_count, int):
                    cluster_flash_parts.append(f"identities: {identities_count:,}")
                if isinstance(faces_count, int):
                    cluster_flash_parts.append(f"faces: {faces_count:,}")
                flash_msg = f"Clustered (thresh {cluster_thresh_value:.2f}, min {int(min_cluster_size_value)})" + (
                    " · " + ", ".join(cluster_flash_parts) if cluster_flash_parts else ""
                )
                group_flash = None
                group_error = None
                with st.spinner("Auto-grouping clusters…"):
                    group_response, group_error = _auto_group_clusters(ep_id)
                if group_response:
                    group_flash = _group_flash_text(group_response)
                elif group_error:
                    st.session_state["episode_detail_flash_error"] = f"Auto-group failed: {group_error}"
                if group_flash:
                    flash_msg = flash_msg + " · " + group_flash

                # Build completion flash message
                completed = st.session_state.get(f"{ep_id}::autorun_completed_stages", [])
                if completed:
                    completion_summary = " → ".join([s.split(" (")[0] for s in completed])
                    flash_msg = f"Auto-Run Pipeline Complete! ({completion_summary}) " + flash_msg

                st.session_state["episode_detail_flash"] = flash_msg

                # Trigger Improve Faces modal on this page (not redirect)
                if selected_attempt_run_id:
                    st.session_state[_improve_faces_state_key(ep_id, selected_attempt_run_id, "trigger")] = True
                    LOGGER.info(
                        "[CLUSTER_COMPLETE] Set trigger flag for Improve Faces modal, ep_id=%s run_id=%s",
                        ep_id,
                        selected_attempt_run_id,
                    )
                    st.toast("🎯 Cluster complete! Opening Improve Faces...")
                else:
                    st.toast("🎯 Cluster complete! Select a run-scoped attempt (run_id) to use Improve Faces.")
                st.rerun()

    # Keep latest cluster log handy for copy/paste
    helpers.render_previous_logs(ep_id, "cluster", expanded=False)

    # Show appropriate button based on state
    if cluster_status_value == "success":
        # If Improve Faces was completed, show "FACES REVIEW" button
        improve_faces_complete = bool(
            st.session_state.get(_improve_faces_state_key(ep_id, selected_attempt_run_id, "complete"))
        )
        if improve_faces_complete:
            faces_review_disabled = not bool(selected_attempt_run_id)
            if st.button(
                "📋 FACES REVIEW",
                key=f"faces_review_cta_{ep_id}",
                use_container_width=True,
                type="primary",
                disabled=faces_review_disabled,
                help="Select a run-scoped attempt (run_id) to review this run." if faces_review_disabled else None,
            ):
                try:
                    qp = st.query_params
                    qp["ep_id"] = ep_id
                    if selected_attempt_run_id:
                        qp["run_id"] = selected_attempt_run_id
                    else:
                        try:
                            del qp["run_id"]
                        except Exception:
                            pass
                    st.query_params = qp
                except Exception:
                    pass
                st.switch_page("pages/3_Faces_Review.py")
        else:
            # Manual Improve Faces launcher
            improve_faces_disabled = not bool(selected_attempt_run_id)
            if st.button(
                "🎯 Improve Faces",
                key=f"improve_faces_cta_{ep_id}",
                use_container_width=True,
                disabled=improve_faces_disabled,
                help="Select a run-scoped attempt (run_id) to enable Improve Faces." if improve_faces_disabled else None,
            ):
                st.session_state[_improve_faces_state_key(ep_id, selected_attempt_run_id, "trigger")] = True
                st.rerun()

    st.divider()

st.subheader("Debug / Export")

# Display artifact store backend
try:
    from apps.api.services.run_artifact_store import get_artifact_store_display, get_artifact_store_status
    _artifact_store_display = get_artifact_store_display()
    _artifact_store_status = get_artifact_store_status()
    _s3_enabled = _artifact_store_status.get("config", {}).get("s3_enabled", False)
    st.caption(f"**Artifact Store:** {_artifact_store_display}")
except Exception:
    _s3_enabled = False
    st.caption("**Artifact Store:** Local filesystem (status unavailable)")

if not selected_attempt_run_id:
    st.info("Select a non-legacy attempt (run_id) to export a run debug report.")
else:
    st.caption(f"Exporting run_id: `{selected_attempt_run_id}`")
    opt_key_prefix = f"{ep_id}::{selected_attempt_run_id}::export_bundle"

    export_state_key = f"{opt_key_prefix}::payload"
    export_clicked = st.button(
        "Generate PDF Report",
        key=f"{opt_key_prefix}::export",
        type="primary",
        use_container_width=False,
        help="Generate a Screen Time Run Debug Report PDF with pipeline stats and artifact manifest",
    )
    if export_clicked:
        with st.spinner("Generating PDF report…"):
            url = f"{cfg['api_base']}/episodes/{ep_id}/runs/{selected_attempt_run_id}/export"
            try:
                resp = requests.get(url, timeout=300)
                resp.raise_for_status()
            except requests.RequestException as exc:
                st.error(helpers.describe_error(url, exc))
            else:
                content_disp = resp.headers.get("Content-Disposition", "") or ""
                filename = None
                if "filename=" in content_disp:
                    filename = content_disp.split("filename=", 1)[1].strip().strip('"')
                if not filename:
                    filename = f"screenalytics_{ep_id}_{selected_attempt_run_id}_debug_report.pdf"

                # Check S3 upload status from response headers
                s3_upload_attempted = resp.headers.get("X-S3-Upload-Attempted", "").lower() == "true"
                s3_upload_success = resp.headers.get("X-S3-Upload-Success", "").lower() == "true"
                s3_upload_key = resp.headers.get("X-S3-Upload-Key")
                s3_upload_error = resp.headers.get("X-S3-Upload-Error")

                st.session_state[export_state_key] = {
                    "filename": filename,
                    "bytes": resp.content,
                    "s3_upload_attempted": s3_upload_attempted,
                    "s3_upload_success": s3_upload_success,
                    "s3_upload_key": s3_upload_key,
                    "s3_upload_error": s3_upload_error,
                }

    bundle = st.session_state.get(export_state_key)
    if isinstance(bundle, dict) and bundle.get("bytes"):
        file_bytes = bundle.get("bytes") or b""
        filename = bundle.get("filename") or f"screenalytics_{ep_id}_{selected_attempt_run_id}_debug_report.pdf"
        try:
            st.caption(f"Report ready: {len(file_bytes) / 1024:.1f} KB")
        except Exception:
            pass

        # Display S3 upload status
        s3_upload_attempted = bundle.get("s3_upload_attempted", False)
        s3_upload_success = bundle.get("s3_upload_success", False)
        s3_upload_key = bundle.get("s3_upload_key")
        s3_upload_error = bundle.get("s3_upload_error")
        if s3_upload_attempted:
            if s3_upload_success and s3_upload_key:
                st.success(f"✅ Saved to S3: `{s3_upload_key}`")
            elif s3_upload_error:
                st.warning(f"⚠️ S3 upload attempted but failed: {s3_upload_error}")
            else:
                st.warning("⚠️ S3 upload attempted but status unknown")
        else:
            st.info("S3 upload not attempted (local backend or disabled)")
        st.download_button(
            "Download PDF",
            data=file_bytes,
            file_name=filename,
            mime="application/pdf",
            key=f"{opt_key_prefix}::download",
            use_container_width=False,
        )
        if st.button("Clear", key=f"{opt_key_prefix}::clear_bundle"):
            st.session_state.pop(export_state_key, None)
            st.rerun()

st.subheader("Artifacts")


def _render_artifact_entry(label: str, local_path: Path, key_suffix: str, s3_key: str | None = None) -> None:
    st.write(f"{label} → {helpers.link_local(local_path)}")
    if not s3_key:
        return
    uri_col, button_col = st.columns([4, 1])
    uri_col.code(helpers.s3_uri(s3_key, bucket_name))
    if button_col.button("Presign", key=f"presign_{key_suffix}"):
        try:
            presign_resp = helpers.api_get("/files/presign", params={"key": s3_key})
        except requests.RequestException as exc:
            st.error(helpers.describe_error(f"{cfg['api_base']}/files/presign", exc))
        else:
            url_value = presign_resp.get("url")
            if url_value:
                st.code(url_value)
                ttl_val = presign_resp.get("expires_in")
                if ttl_val:
                    st.caption(f"Expires in {ttl_val}s")
            else:
                st.warning("Presign unavailable for this key.")


manifests_prefix = (prefixes or {}).get("manifests") if prefixes else None
_render_artifact_entry(
    "Video",
    get_path(ep_id, "video"),
    "video",
    details["s3"]["v2_key"] or details["s3"]["v1_key"],
)
detections_key = f"{manifests_prefix}detections.jsonl" if manifests_prefix else None
tracks_key = f"{manifests_prefix}tracks.jsonl" if manifests_prefix else None
faces_key = f"{manifests_prefix}faces.jsonl" if manifests_prefix else None
identities_key = f"{manifests_prefix}identities.json" if manifests_prefix else None
_render_artifact_entry("Detections", detections_path, "detections", detections_key)
_render_artifact_entry("Tracks", tracks_path, "tracks", tracks_key)
_render_artifact_entry("Faces", faces_path, "faces", faces_key)
_render_artifact_entry("Identities", identities_path, "identities", identities_key)
_render_artifact_entry("Screentime (json)", analytics_dir / "screentime.json", "screentime_json")
_render_artifact_entry("Screentime (csv)", analytics_dir / "screentime.csv", "screentime_csv")


def _read_json_artifact(path: Path, max_lines: int = 2000) -> tuple[str | None, str | None]:
    """Return (content, error) for a JSON/JSONL artifact with defensive limits."""
    if not path.exists():
        return None, f"{path.name} does not exist."
    try:
        if path.suffix.lower() == ".jsonl":
            lines = []
            with path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle, start=1):
                    if idx > max_lines:
                        lines.append(f"... truncated after {max_lines} lines ...")
                        break
                    lines.append(line.rstrip("\n"))
            return "\n".join(lines), None
        if path.suffix.lower() == ".json":
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            return json.dumps(payload, indent=2, ensure_ascii=False), None
    except Exception as exc:
        return None, f"Failed to load {path.name}: {exc}"
    return None, f"Unsupported file type for {path.name}"


st.subheader("Debug: Raw JSON artifacts")
artifact_groups = {
    "Detect / Faces / Tracks": [
        detections_path,
        tracks_path,
        faces_path,
    ],
    "Cluster": [
        identities_path,
        _track_metrics_path,
    ],
    "Screentime": [
        analytics_dir / "screentime.json",
    ],
}
for group, paths in artifact_groups.items():
    existing = [p for p in paths if p.exists()]
    with st.expander(group, expanded=False):
        if not existing:
            st.caption("No artifacts found for this stage.")
            continue
        labels = [p.name for p in existing]
        selected = st.selectbox(
            "Choose artifact",
            labels,
            key=f"{group}::artifact_selector",
        )
        chosen_path = next((p for p in existing if p.name == selected), None)
        if not chosen_path:
            st.caption("Select a file to view its contents.")
            continue
        st.caption(f"Path: {helpers.link_local(chosen_path)}")
        content, err = _read_json_artifact(chosen_path)
        if err:
            st.error(err)
            continue
        st.code(content or "", language="json")


# =============================================================================
# Auto-refresh when jobs are running (CELERY MODE ONLY)
# =============================================================================
# In Celery mode: auto-refresh every 3 seconds to poll for updates.
# In Local mode: DO NOT auto-refresh while job is running - logs stream via SSE
# and auto-refresh would disconnect the stream, killing the subprocess.
# BUT: refresh once when job completes to update the UI status.

_any_job_running = running_detect_job or running_faces_job or running_cluster_job or running_audio_job
_execution_mode = helpers.get_execution_mode(ep_id)

# Check if a local mode job JUST completed
# Two detection methods:
# 1. progress.json shows "done" and was recently updated (for Detect/Track, Faces Harvest)
# 2. A run marker was recently updated (for Cluster, which cleans up its progress file)
_local_job_just_completed = False
if _execution_mode == "local" and not _any_job_running:
    # Method 1: Check progress.json for "done" status
    _progress_data = helpers.get_episode_progress(ep_id)
    if _progress_data and _progress_data.get("step") == "done":
        _progress_age = _get_progress_file_age(ep_id)
        # If progress file was updated in last 10 seconds and shows "done", job just finished
        if _progress_age is not None and _progress_age < 10:
            _local_job_just_completed = True

    # Method 2: Check if any run marker was recently updated (catches Cluster completion)
    if not _local_job_just_completed:
        _run_marker_age = _get_most_recent_run_marker_age(ep_id)
        # If a run marker was updated in last 10 seconds, a job just finished
        if _run_marker_age is not None and _run_marker_age < 10:
            _local_job_just_completed = True

if _local_job_just_completed:
    # Local mode job just completed - refresh once to show updated status
    if st.session_state.get(_session_improve_faces_state_key(ep_id, "trigger")):
        st.caption("✅ Cluster completed! Opening Improve Faces...")
        import time as _time
        _time.sleep(0.5)
        st.rerun()  # Stay on this page, modal will open
    else:
        st.caption("✅ Job completed! Refreshing to show results...")
        import time as _time
        _time.sleep(1.5)
        st.rerun()
elif _any_job_running and _execution_mode != "local":
    # Celery mode: poll for updates since jobs run in background
    import time as _time
    _running_ops = []
    if running_detect_job:
        _running_ops.append("Detect/Track")
    if running_faces_job:
        _running_ops.append("Faces Harvest")
    if running_cluster_job:
        _running_ops.append("Cluster")
    if running_audio_job:
        _running_ops.append("Audio Pipeline")
    st.caption(f"⏳ Auto-refreshing for running job(s): {', '.join(_running_ops)}...")
    _time.sleep(3)
    st.rerun()
elif _any_job_running and _execution_mode == "local":
    # Local mode: logs stream via SSE, no auto-refresh needed
    st.caption("📡 Streaming logs from local subprocess... (do not refresh page)")
