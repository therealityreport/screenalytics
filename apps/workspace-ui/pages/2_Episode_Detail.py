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
PROJECT_ROOT = PAGE_PATH.parents[3]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402
import episode_detail_layout as stage_layout  # noqa: E402

from py_screenalytics.artifacts import get_path  # noqa: E402
from py_screenalytics import run_layout  # noqa: E402
from py_screenalytics.episode_status import stage_artifacts  # noqa: E402

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


_SUCCESS_JOB_STATUSES = {"completed", "success", "succeeded"}


def _summary_status_ok(summary: Dict[str, Any] | None) -> bool:
    if not isinstance(summary, dict) or not summary:
        return False
    status = str(summary.get("status") or "").strip().lower()
    if status not in _SUCCESS_JOB_STATUSES:
        return False
    if summary.get("error"):
        return False
    return True


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
    export_params = {"include_screentime": "false"}
    LOGGER.info(
        "[PDF-EXPORT] Triggering export for %s/%s via %s (include_screentime=%s)",
        ep_id,
        run_id,
        export_url,
        export_params["include_screentime"],
    )

    try:
        resp = requests.get(export_url, params=export_params, timeout=300)
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


def _read_json_payload(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_marker_payload_with_fallback(
    primary_path: Path,
    fallback_path: Path | None,
    *,
    run_id: str | None = None,
    require_run_match: bool = True,
) -> tuple[dict[str, Any] | None, float | None, bool]:
    """Load a run marker payload, falling back to legacy markers when run-scoped is missing."""
    payload = _read_json_payload(primary_path) if primary_path.exists() else None
    if isinstance(payload, dict):
        try:
            return payload, primary_path.stat().st_mtime, False
        except OSError:
            return payload, None, False
    if fallback_path and fallback_path.exists():
        payload = _read_json_payload(fallback_path)
        if isinstance(payload, dict):
            if run_id and require_run_match:
                marker_run_id = payload.get("run_id")
                if not (isinstance(marker_run_id, str) and marker_run_id.strip() == run_id):
                    return None, None, False
            try:
                return payload, fallback_path.stat().st_mtime, True
            except OSError:
                return payload, None, True
    return None, None, False


def _progress_pct_from_payload(payload: dict[str, Any] | None) -> float | None:
    if not isinstance(payload, dict):
        return None
    for key in ("progress", "progress_pct", "pct"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            pct = float(value)
            if pct > 1.0:
                pct = pct / 100.0
            return max(0.0, min(1.0, pct))
    status = str(payload.get("status") or "").strip().lower()
    if status in {"completed", "success"}:
        return 1.0
    return None


@st.cache_data(ttl=30, show_spinner=False)
def _cached_run_artifact_presence(
    ep_id: str,
    run_id: str,
    rel_paths: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    try:
        run_id_norm = run_layout.normalize_run_id(run_id)
    except ValueError:
        run_id_norm = None
    if not run_id_norm:
        for rel_path in rel_paths:
            result[rel_path] = {"local": False, "remote": False, "s3_key": None, "path": None}
        return result

    storage = None
    try:
        from apps.api.services.storage import StorageService

        storage = StorageService()
    except Exception as exc:
        LOGGER.warning("[RUN_ARTIFACT] Storage init failed: %s", exc)

    for rel_path in rel_paths:
        local_path = run_layout.run_root(ep_id, run_id_norm) / rel_path
        local_exists = local_path.exists()
        remote_exists = False
        remote_key = None
        canonical_key = None
        legacy_key = None
        canonical_remote = False
        legacy_remote = False
        try:
            keys = run_layout.run_artifact_s3_keys_for_read(ep_id, run_id_norm, rel_path)
        except Exception:
            keys = []
        if keys:
            canonical_key = keys[0]
            if len(keys) > 1 and keys[1] != canonical_key:
                legacy_key = keys[1]
        if not local_exists and storage and storage.s3_enabled():
            try:
                if canonical_key:
                    canonical_remote = storage.object_exists(canonical_key)
                if legacy_key:
                    legacy_remote = storage.object_exists(legacy_key)
            except Exception as exc:
                LOGGER.warning("[RUN_ARTIFACT] S3 check failed for %s/%s: %s", ep_id, rel_path, exc)
        remote_exists = canonical_remote or legacy_remote
        if canonical_remote:
            remote_key = canonical_key
        elif legacy_remote:
            remote_key = legacy_key
        result[rel_path] = {
            "local": local_exists,
            "remote": remote_exists,
            "s3_key": remote_key,
            "path": str(local_path),
            "canonical_key": canonical_key,
            "canonical_remote": canonical_remote,
            "legacy_key": legacy_key,
            "legacy_remote": legacy_remote,
        }
    return result


def _presence_from_cache(
    entry: dict[str, Any] | None,
    local_path: Path,
) -> stage_layout.ArtifactPresence:
    local_exists = bool(entry.get("local")) if isinstance(entry, dict) else local_path.exists()
    remote_exists = bool(entry.get("remote")) if isinstance(entry, dict) else False
    s3_key = entry.get("s3_key") if isinstance(entry, dict) else None
    return stage_layout.ArtifactPresence(
        local=local_exists,
        remote=remote_exists,
        path=str(local_path),
        s3_key=s3_key if isinstance(s3_key, str) else None,
    )


def _ensure_run_artifacts_local(
    ep_id: str,
    run_id: str,
    artifacts: dict[str, stage_layout.ArtifactPresence],
) -> tuple[bool, str | None]:
    try:
        from apps.api.services.storage import StorageService

        storage = StorageService()
    except Exception as exc:
        return False, f"storage_init_failed: {exc}"

    if not storage.s3_enabled() or not storage._client:
        return False, "s3_disabled"

    for rel_path, presence in artifacts.items():
        if presence.local:
            continue
        if not presence.remote or not presence.s3_key:
            return False, f"missing_remote: {rel_path}"
        local_path = run_layout.run_root(ep_id, run_id) / rel_path
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            storage._client.download_file(storage.bucket, presence.s3_key, str(local_path))
            LOGGER.info(
                "[RUN_ARTIFACT] Mirrored %s from s3://%s/%s",
                rel_path,
                storage.bucket,
                presence.s3_key,
            )
        except Exception as exc:
            return False, f"download_failed: {rel_path}: {exc}"
    return True, None


def _render_downstream_progress(
    progress_payload: dict[str, Any] | None,
    *,
    running: bool,
) -> None:
    progress_pct = _progress_pct_from_payload(progress_payload)
    if progress_pct is None and running:
        progress_pct = 0.05
    if progress_pct is not None:
        st.progress(min(progress_pct, 1.0))
    if not progress_payload:
        return
    progress_line = helpers.stage_progress_line(progress_payload)
    if progress_line:
        st.caption(progress_line)
    stall_msg = helpers.stage_progress_stall_message(progress_payload) if running else None
    if stall_msg:
        st.warning(stall_msg)


def _render_downstream_log_expander(
    label: str,
    *,
    marker_payload: dict[str, Any] | None,
    progress_payload: dict[str, Any] | None,
    artifacts_hint: str | None = None,
) -> None:
    with st.expander(f"{label} Detailed Log", expanded=False):
        if progress_payload:
            st.code(json.dumps(progress_payload, indent=2), language="json")
            return
        if marker_payload:
            st.code(json.dumps(marker_payload, indent=2), language="json")
            return
        if artifacts_hint:
            st.caption(artifacts_hint)
            return
        st.caption("No logs available yet.")


def _render_stage_artifacts_expander(
    label: str,
    artifacts: list[tuple[str, Path, str]],
    *,
    hint: str | None = None,
) -> None:
    with st.expander(f"{label} Artifacts", expanded=False):
        if hint:
            st.caption(hint)
        if not artifacts:
            st.caption("No artifacts available yet.")
            return
        for artifact_label, artifact_path, scope in artifacts:
            status = "local" if artifact_path.exists() else "missing"
            st.caption(f"{artifact_label} ({scope} · {status})")
            st.write(helpers.link_local(artifact_path))


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
    running_body_tracking_job: dict | None = None,
    running_body_fusion_job: dict | None = None,
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
        running_body_tracking_job,
        running_body_fusion_job,
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


def _clear_improve_faces_state(ep_id: str, run_id: str | None) -> None:
    for suffix in ("complete", "active", "suggestions", "index", "empty_reason", "trigger"):
        st.session_state.pop(_improve_faces_state_key(ep_id, run_id, suffix), None)


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


@st.cache_data(ttl=10, show_spinner=False)
def _cached_episode_status_file(
    ep_id: str,
    run_id: str | None,
    status_mtime: float,
) -> Dict[str, Any] | None:
    if not run_id:
        return None
    try:
        run_root = run_layout.run_root(ep_id, run_id)
    except ValueError:
        return None
    return _read_json_payload(run_root / "episode_status.json")


@st.cache_data(ttl=60, show_spinner=False)
def _cached_storage_status() -> Dict[str, Any] | None:
    """Cache storage status API response with 60s TTL (rarely changes)."""
    try:
        return helpers.api_get("/config/storage")
    except requests.RequestException:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def _cached_db_health() -> Dict[str, Any] | None:
    """Cache DB health API response with 30s TTL."""
    try:
        return helpers.api_get("/config/db_health")
    except requests.RequestException:
        return None


# Suggestion 2: Adaptive cache TTL based on job state
# During active jobs: 10s TTL (human can't perceive <10s delay in progress updates)
# When idle: 3s TTL for responsive manual refresh
_CACHE_TTL_ACTIVE = 10  # Longer TTL during job execution
_CACHE_TTL_IDLE = 3  # Shorter TTL when idle


def _any_job_active(ep_id: str) -> bool:
    """Check if any job is active for this episode (from session state)."""
    if st.session_state.get(f"{ep_id}::autorun_pipeline"):
        return True
    if _job_active(ep_id):
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


def _find_running_subprocess_job(
    jobs_payload: Dict[str, Any] | None,
    *,
    job_type: str,
    run_id: str | None = None,
) -> Dict[str, Any] | None:
    if not isinstance(jobs_payload, dict):
        return None
    jobs = jobs_payload.get("jobs")
    if not isinstance(jobs, list):
        return None
    running_states = {"running", "queued", "in_progress"}
    for job in jobs:
        if not isinstance(job, dict):
            continue
        if str(job.get("job_type") or "") != job_type:
            continue
        state = str(job.get("state") or "").strip().lower()
        if state not in running_states:
            continue
        if run_id:
            requested = job.get("requested") if isinstance(job.get("requested"), dict) else {}
            requested_run_id = requested.get("run_id") if isinstance(requested.get("run_id"), str) else None
            if requested_run_id and requested_run_id.strip() != run_id:
                continue
        return job
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
    normalized_status = str(detect_status.get("status") or "missing").strip().lower()
    if not normalized_status:
        normalized_status = "missing"
    if normalized_status in {"error", "failed"}:
        return "error", False, False, tracks_only_fallback
    if normalized_status == "stale":
        return "stale", False, False, tracks_only_fallback
    if tracks_ready_flag and normalized_status == "success" and not detect_status.get("error"):
        return "success", True, False, tracks_only_fallback
    if normalized_status == "success":
        if manifest_ready:
            return "success", True, False, tracks_only_fallback
        return "stale", False, False, tracks_only_fallback
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


@st.cache_data(ttl=60, show_spinner=False)
def _cached_yaml(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _body_tracking_enabled_config() -> bool:
    cfg = _cached_yaml(str(PROJECT_ROOT / "config" / "pipeline" / "body_detection.yaml"))
    enabled = bool((cfg.get("body_tracking") or {}).get("enabled", False))
    env_override = os.environ.get("AUTO_RUN_BODY_TRACKING", "").strip().lower()
    if env_override in ("0", "false", "no", "off"):
        enabled = False
    elif env_override in ("1", "true", "yes", "on"):
        enabled = True
    return enabled


def _track_fusion_enabled_config() -> bool:
    cfg = _cached_yaml(str(PROJECT_ROOT / "config" / "pipeline" / "track_fusion.yaml"))
    return bool((cfg.get("track_fusion") or {}).get("enabled", False))


def _track_fusion_reid_enabled_config() -> bool:
    cfg = _cached_yaml(str(PROJECT_ROOT / "config" / "pipeline" / "track_fusion.yaml"))
    return bool((cfg.get("reid_handoff") or {}).get("enabled", False))


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

# DB-dependent features (suggestions/locks) should be disabled when DB is unavailable.
db_health_snapshot = _cached_db_health()
db_available = False
if isinstance(db_health_snapshot, dict):
    configured = bool(db_health_snapshot.get("configured"))
    ok = db_health_snapshot.get("ok")
    migrations_ok = db_health_snapshot.get("migrations_ok")
    db_available = configured and ok is not False and migrations_ok is not False

# Legacy: Smart Suggestions navigation (now replaced by Improve Faces modal)
# Clear any stale navigation flags from previous sessions
_autorun_navigate_key = f"{ep_id}::autorun_navigate_to_suggestions"
st.session_state.pop(_autorun_navigate_key, None)


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
_autorun_key = f"{ep_id}::autorun_pipeline"
_autorun_phase_key = f"{ep_id}::autorun_phase"
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


episode_status_mtime = _safe_mtime(_scoped_manifests_dir / "episode_status.json") if selected_attempt_run_id else 0

current_mtimes = (
    selected_attempt_run_id or "legacy",
    _safe_mtime(_scoped_markers_dir / "detect_track.json"),
    _safe_mtime(_scoped_markers_dir / "faces_embed.json"),
    _safe_mtime(_scoped_markers_dir / "cluster.json"),
    _safe_mtime(_scoped_markers_dir / "body_tracking.json"),
    _safe_mtime(_scoped_markers_dir / "body_tracking_fusion.json"),
    _safe_mtime(_runs_root / "detect_track.json") if selected_attempt_run_id else 0,
    _safe_mtime(_runs_root / "faces_embed.json") if selected_attempt_run_id else 0,
    _safe_mtime(_runs_root / "cluster.json") if selected_attempt_run_id else 0,
    _safe_mtime(_runs_root / "body_tracking.json") if selected_attempt_run_id else 0,
    _safe_mtime(_runs_root / "body_tracking_fusion.json") if selected_attempt_run_id else 0,
    episode_status_mtime,
    _safe_mtime(_scoped_manifests_dir / "detections.jsonl"),
    _safe_mtime(_scoped_manifests_dir / "tracks.jsonl"),
    _safe_mtime(_scoped_manifests_dir / "faces.jsonl"),
    _safe_mtime(_track_metrics_path),
    _safe_mtime(_scoped_manifests_dir / "identities.json"),
    _safe_mtime(_scoped_manifests_dir / "body_tracking" / "body_tracks.jsonl"),
    _safe_mtime(_scoped_manifests_dir / "body_tracking" / "track_fusion.json"),
    _safe_mtime(_scoped_manifests_dir / "body_tracking" / "screentime_comparison.json"),
    _safe_mtime(_scoped_manifests_dir / "exports" / "export_index.json") if selected_attempt_run_id else 0,
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
episode_status_payload = _cached_episode_status_file(ep_id, selected_attempt_run_id, episode_status_mtime)

def _phase_from_episode_status(
    status_payload: dict[str, Any] | None,
    stage_key: str,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(status_payload, dict):
        return fallback
    stages = status_payload.get("stages")
    if not isinstance(stages, dict):
        return fallback
    stage_entry = stages.get(stage_key)
    if not isinstance(stage_entry, dict):
        return fallback
    merged = dict(fallback or {})
    status_value = stage_entry.get("status")
    if status_value:
        merged["status"] = status_value
    timestamps = stage_entry.get("timestamps") if isinstance(stage_entry.get("timestamps"), dict) else {}
    progress_payload = stage_entry.get("progress")
    if isinstance(progress_payload, dict):
        merged["progress"] = progress_payload
    started_at = timestamps.get("started_at") or stage_entry.get("started_at")
    if started_at:
        merged["started_at"] = started_at
    ended_at = timestamps.get("ended_at") or stage_entry.get("ended_at")
    if ended_at:
        merged["finished_at"] = ended_at
    if started_at and ended_at:
        merged["runtime_sec"] = _runtime_from_iso(started_at, ended_at)
    else:
        duration = stage_entry.get("duration_s")
        if duration is not None:
            merged["runtime_sec"] = duration
    error_reason = stage_entry.get("error_reason")
    if error_reason:
        merged["error"] = error_reason
        merged["error_reason"] = error_reason
    metrics = stage_entry.get("metrics") if isinstance(stage_entry.get("metrics"), dict) else {}
    if stage_key == "detect":
        for key in ("detections", "tracks", "rtf", "effective_fps_processing", "scene_cut_count", "forced_scene_warmup_ratio"):
            if metrics.get(key) is not None:
                merged[key] = metrics.get(key)
    elif stage_key == "faces":
        for key in ("faces", "embedding_backend_actual", "embedding_model_name", "embedding_backend_fallback_reason"):
            if metrics.get(key) is not None:
                merged[key] = metrics.get(key)
    elif stage_key == "cluster":
        for key in ("identities", "faces", "singleton_fraction_before", "singleton_fraction_after", "cluster_thresh", "min_cluster_size", "min_identity_sim"):
            if metrics.get(key) is not None:
                merged[key] = metrics.get(key)
    return merged

if status_payload is None:
    detect_phase_status: Dict[str, Any] = {}
    faces_phase_status: Dict[str, Any] = {"status": "unknown"}
    cluster_phase_status: Dict[str, Any] = {"status": "unknown"}
    body_tracking_phase_status: Dict[str, Any] = {"status": "unknown"}
    track_fusion_phase_status: Dict[str, Any] = {"status": "unknown"}
    pdf_phase_status: Dict[str, Any] = {"status": "unknown"}
else:
    detect_phase_status = status_payload.get("detect_track") or {}
    faces_phase_status = status_payload.get("faces_embed") or {}
    cluster_phase_status = status_payload.get("cluster") or {}
    body_tracking_phase_status = {}
    track_fusion_phase_status = {}
    pdf_phase_status = {}

if isinstance(episode_status_payload, dict):
    detect_phase_status = _phase_from_episode_status(episode_status_payload, "detect", detect_phase_status)
    faces_phase_status = _phase_from_episode_status(episode_status_payload, "faces", faces_phase_status)
    cluster_phase_status = _phase_from_episode_status(episode_status_payload, "cluster", cluster_phase_status)
    body_tracking_phase_status = _phase_from_episode_status(
        episode_status_payload, "body_tracking", body_tracking_phase_status
    )
    track_fusion_phase_status = _phase_from_episode_status(
        episode_status_payload, "track_fusion", track_fusion_phase_status
    )
    pdf_phase_status = _phase_from_episode_status(episode_status_payload, "pdf", pdf_phase_status)

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
detect_job_defaults, detect_job_record = _load_job_defaults(ep_id, "detect_track")
faces_job_defaults, faces_job_record = _load_job_defaults(ep_id, "faces_embed")
cluster_job_defaults, cluster_job_record = _load_job_defaults(ep_id, "cluster")
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

        # Check DB health (cached for 30s)
        db_health = _cached_db_health()
        if isinstance(db_health, dict):
            configured = bool(db_health.get("configured"))
            ok = db_health.get("ok")
            migrations_ok = db_health.get("migrations_ok")
            err = db_health.get("error")
            if not configured:
                st.warning(
                    "⚠️ **DB Not Configured**: `DB_URL` is not set. DB-backed features are disabled "
                    "(identity locks, suggestion history/audit, some review actions). "
                    "Set `DB_URL` (Postgres) or `SCREENALYTICS_FAKE_DB=1` for local-only dev."
                )
            elif ok is False:
                st.error(f"🔴 **DB Unhealthy**: {err or 'unknown error'}")
            elif migrations_ok is False:
                st.warning("⚠️ **DB Schema Incomplete**: required tables/migrations are missing.")

    except Exception as exc:
        LOGGER.debug("[system-status] Failed to fetch storage status: %s", exc)


# Show system status warnings at top of page
_render_system_status()

with st.expander("🗄️ DB Health", expanded=False):
    db_health = _cached_db_health()
    if not isinstance(db_health, dict):
        st.caption("DB health unavailable.")
    else:
        configured = bool(db_health.get("configured"))
        ok = db_health.get("ok")
        fake_db = bool(db_health.get("fake_db"))
        psycopg2_available = db_health.get("psycopg2_available")
        latency_ms = db_health.get("latency_ms")
        migrations_ok = db_health.get("migrations_ok")
        err = db_health.get("error")

        st.caption(f"configured={configured} fake_db={fake_db} psycopg2={psycopg2_available}")
        if isinstance(latency_ms, (int, float)):
            st.caption(f"latency_ms={latency_ms}")
        if err:
            st.caption(f"error={err}")
        if migrations_ok is not None:
            st.caption(f"migrations_ok={migrations_ok}")
        tables = db_health.get("tables")
        if isinstance(tables, dict) and tables and not fake_db:
            missing = [name for name, present in tables.items() if present is False]
            if missing:
                st.warning(f"Missing tables: {', '.join(missing)}")
            else:
                st.success("All required tables present.")


# =============================================================================
# Run Performance / Quality (UI-only diagnostics)
# =============================================================================
with st.expander("⚡ Performance & Quality", expanded=False):
    if not selected_attempt_run_id:
        st.caption("Select a run-scoped attempt (run_id) to view run-specific performance + quality diagnostics.")
    else:
        _run_perf_prefix = f"{ep_id}::{selected_attempt_run_id}"

        def _read_json_best_effort(path: Path) -> dict[str, Any] | None:
            path = Path(path)
            if not path.exists():
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            return payload if isinstance(payload, dict) else None

        def _iter_jsonl_best_effort(path: Path):
            path = Path(path)
            if not path.exists():
                return
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(row, dict):
                            yield row
            except OSError:
                return

        detect_marker = _read_json_best_effort(_scoped_markers_dir / "detect_track.json") or {}
        faces_marker = _read_json_best_effort(_scoped_markers_dir / "faces_embed.json") or {}
        cluster_marker = _read_json_best_effort(_scoped_markers_dir / "cluster.json") or {}
        body_marker = _read_json_best_effort(_scoped_markers_dir / "body_tracking.json") or {}
        fusion_marker = _read_json_best_effort(_scoped_markers_dir / "body_tracking_fusion.json") or {}

        detect_rtf = helpers.coerce_float(detect_marker.get("rtf"))
        detect_eff_fps = helpers.coerce_float(detect_marker.get("effective_fps_processing"))
        detect_frames_processed = helpers.coerce_int(detect_marker.get("face_detect_frames_processed"))
        detect_frames_total = helpers.coerce_int(detect_marker.get("frames_total"))
        detect_stride_eff = helpers.coerce_int(detect_marker.get("stride_effective")) or helpers.coerce_int(detect_marker.get("stride"))
        detect_onnx_provider = detect_marker.get("onnx_provider_resolved") or detect_marker.get("resolved_device")
        forced_scene_warmup_ratio = helpers.coerce_float(detect_marker.get("forced_scene_warmup_ratio"))
        scene_cuts_total = (
            helpers.coerce_int(detect_marker.get("scene_cut_count"))
            or helpers.coerce_int(detect_marker.get("scene_cuts_total"))
            or helpers.coerce_int(detect_marker.get("scene_cuts"))
        )

        baseline_rtf = helpers.coerce_float(st.session_state.get(f"{_run_perf_prefix}::baseline_detect_rtf"))
        baseline_eff_fps = helpers.coerce_float(st.session_state.get(f"{_run_perf_prefix}::baseline_detect_eff_fps"))
        delta_rtf = None
        delta_eff_fps = None
        if detect_rtf is not None and baseline_rtf is not None:
            delta_rtf = f"{(detect_rtf - baseline_rtf):+.2f}x"
        if detect_eff_fps is not None and baseline_eff_fps is not None:
            delta_eff_fps = f"{(detect_eff_fps - baseline_eff_fps):+.2f} fps"

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Detect RTF", f"{detect_rtf:.2f}x" if detect_rtf is not None else "n/a", delta=delta_rtf)
        with metric_col2:
            st.metric(
                "Detect Effective FPS",
                f"{detect_eff_fps:.2f} fps" if detect_eff_fps is not None else "n/a",
                delta=delta_eff_fps,
            )
        with metric_col3:
            if detect_frames_processed is not None:
                note = f"/ {detect_frames_total:,}" if isinstance(detect_frames_total, int) and detect_frames_total > 0 else ""
                st.metric("Frames Processed", f"{detect_frames_processed:,}{note}")
            else:
                st.metric("Frames Processed", "n/a")

        perf_lines: list[str] = []
        if detect_stride_eff is not None:
            perf_lines.append(f"stride_effective={detect_stride_eff}")
        if detect_onnx_provider:
            perf_lines.append(f"onnx_provider={detect_onnx_provider}")
        if forced_scene_warmup_ratio is not None:
            perf_lines.append(f"forced_scene_warmup_ratio={forced_scene_warmup_ratio:.3f}")
        if scene_cuts_total is not None:
            perf_lines.append(f"scene_cuts={scene_cuts_total}")
        if perf_lines:
            st.caption("Detect/Track: " + " · ".join(perf_lines))

        # Tracking fragmentation stats from run-scoped track_metrics.json (if present).
        forced_splits = None
        id_switches = None
        track_metrics_payload: dict[str, Any] | None = None
        cluster_metrics_block: dict[str, Any] | None = None
        try:
            if _track_metrics_path.exists():
                _tm = json.loads(_track_metrics_path.read_text(encoding="utf-8"))
                if isinstance(_tm, dict):
                    track_metrics_payload = _tm
                    cluster_metrics_block = _tm.get("cluster_metrics") if isinstance(_tm.get("cluster_metrics"), dict) else None
                metrics_block = _tm.get("metrics") if isinstance(_tm, dict) else None
                if isinstance(metrics_block, dict):
                    forced_splits = helpers.coerce_int(metrics_block.get("forced_splits"))
                    id_switches = helpers.coerce_int(metrics_block.get("id_switches"))
        except (OSError, json.JSONDecodeError):
            pass
        frag_bits: list[str] = []
        if forced_splits is not None:
            frag_bits.append(f"forced_splits={forced_splits:,}")
        if id_switches is not None:
            frag_bits.append(f"id_switches={id_switches:,}")
        if frag_bits:
            st.caption("Tracking: " + " · ".join(frag_bits))

        # Track embedding coherence (spread) - computed from tracks.jsonl (best effort).
        coherence_threshold = 0.30
        try:
            coherence_threshold = float(os.environ.get("TRACK_COHERENCE_WARN", "0.30"))
        except (TypeError, ValueError):
            coherence_threshold = 0.30
        spreads: list[float] = []
        flagged = 0
        try:
            if tracks_path.exists():
                for row in _iter_jsonl_best_effort(tracks_path):
                    spread = helpers.coerce_float(row.get("face_embedding_spread"))
                    if spread is None:
                        continue
                    spreads.append(float(spread))
                    if spread >= coherence_threshold:
                        flagged += 1
        except Exception:
            spreads = []
            flagged = 0
        if spreads:
            avg_spread = sum(spreads) / max(len(spreads), 1)
            st.caption(
                f"Track coherence: tracks_with_spread={len(spreads):,} · mixed_tracks={flagged:,} "
                f"(spread≥{coherence_threshold:.2f}) · avg_spread={avg_spread:.3f} · max_spread={max(spreads):.3f}"
            )

        # Embedding backend (faces) and fusion mode (body/fusion).
        embed_backend_actual = faces_marker.get("embedding_backend_actual") or faces_phase_status.get("embedding_backend_actual")
        embed_fallback = faces_marker.get("embedding_backend_fallback_reason") or faces_phase_status.get(
            "embedding_backend_fallback_reason"
        )
        if embed_backend_actual:
            msg = f"Faces embed backend: {embed_backend_actual}"
            if isinstance(embed_fallback, str) and embed_fallback.strip():
                msg = msg + f" (fallback_reason={embed_fallback.strip()})"
            st.caption(msg)

        body_reid = body_marker.get("body_reid")
        if isinstance(body_reid, dict):
            cfg_enabled = body_reid.get("enabled_config")
            effective = body_reid.get("enabled_effective")
            skip_reason = body_reid.get("reid_skip_reason")
            st.caption(
                "Body Re-ID: "
                + " · ".join(
                    [
                        f"enabled_config={cfg_enabled}",
                        f"enabled_effective={effective}",
                        f"skip_reason={skip_reason}" if skip_reason else "skip_reason=None",
                    ]
                )
            )

        if fusion_marker:
            fusion_status = fusion_marker.get("status")
            if fusion_status:
                st.caption(f"Track fusion marker: status={fusion_status}")

        st.divider()
        st.markdown("#### Regression Flags")

        def _safe_ratio(numer: int | None, denom: int | None) -> float | None:
            if numer is None or denom is None or denom <= 0:
                return None
            return float(numer) / float(denom)

        def _singleton_rate_from_metrics(metrics_payload: dict[str, Any] | None) -> float | None:
            if not isinstance(metrics_payload, dict):
                return None
            block = metrics_payload.get("cluster_metrics")
            if not isinstance(block, dict):
                return None
            rate = helpers.coerce_float(block.get("singleton_fraction_after") or block.get("singleton_fraction"))
            if rate is not None:
                return rate
            singles = helpers.coerce_int(block.get("singleton_count"))
            total = helpers.coerce_int(block.get("total_clusters"))
            return _safe_ratio(singles, total)

        def _detect_rtf_for_run(run_root: Path) -> float | None:
            status_payload = _read_json_best_effort(run_root / "episode_status.json")
            if isinstance(status_payload, dict):
                detect_block = status_payload.get("stages", {}).get("detect")
                if isinstance(detect_block, dict):
                    metrics = detect_block.get("metrics")
                    if isinstance(metrics, dict):
                        rtf_val = helpers.coerce_float(metrics.get("rtf"))
                        if rtf_val is not None:
                            return rtf_val
            marker = _read_json_best_effort(run_root / "detect_track.json") or {}
            return helpers.coerce_float(marker.get("rtf"))

        def _track_count_for_run(run_root: Path) -> int | None:
            status_payload = _read_json_best_effort(run_root / "episode_status.json")
            if isinstance(status_payload, dict):
                detect_block = status_payload.get("stages", {}).get("detect")
                if isinstance(detect_block, dict):
                    metrics = detect_block.get("metrics")
                    if isinstance(metrics, dict):
                        count = helpers.coerce_int(metrics.get("tracks"))
                        if count is not None:
                            return count
            marker = _read_json_best_effort(run_root / "detect_track.json") or {}
            return helpers.coerce_int(marker.get("tracks"))

        def _forced_splits_share_for_run(run_root: Path) -> float | None:
            metrics_payload = _read_json_best_effort(run_root / "track_metrics.json") or {}
            metrics_block = metrics_payload.get("metrics") if isinstance(metrics_payload, dict) else None
            forced = helpers.coerce_int(metrics_block.get("forced_splits") if isinstance(metrics_block, dict) else None)
            tracks_total = helpers.coerce_int(metrics_block.get("tracks_born") if isinstance(metrics_block, dict) else None)
            if tracks_total is None:
                tracks_total = _track_count_for_run(run_root)
            return _safe_ratio(forced, tracks_total)

        def _fused_pairs_for_run(run_root: Path) -> tuple[int | None, int | None, dict[str, Any] | None]:
            payload = _read_json_best_effort(run_root / "body_tracking" / "track_fusion.json") or {}
            if not isinstance(payload, dict):
                return None, None, None
            diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
            pairs = helpers.coerce_int(diagnostics.get("final_pairs"))
            if pairs is None:
                pairs = helpers.coerce_int(payload.get("num_fused_identities"))
            comparisons = helpers.coerce_int(diagnostics.get("reid_comparisons"))
            return pairs, comparisons, payload

        def _prev_successful_run_id() -> str | None:
            if not selected_attempt_run_id:
                return None
            candidates: list[tuple[float, str]] = []
            for run_id in run_layout.list_run_ids(ep_id):
                if run_id == selected_attempt_run_id:
                    continue
                try:
                    mtime = run_layout.run_root(ep_id, run_id).stat().st_mtime
                except (OSError, ValueError):
                    mtime = 0.0
                candidates.append((mtime, run_id))
            for _, run_id in sorted(candidates, reverse=True):
                run_root = run_layout.run_root(ep_id, run_id)
                status_payload = _read_json_best_effort(run_root / "episode_status.json")
                if isinstance(status_payload, dict):
                    detect_block = status_payload.get("stages", {}).get("detect")
                    if isinstance(detect_block, dict) and detect_block.get("status") == "success":
                        return run_id
                marker = _read_json_best_effort(run_root / "detect_track.json")
                if isinstance(marker, dict) and str(marker.get("status") or "").lower() == "success":
                    return run_id
            return None

        baseline_run_id = _prev_successful_run_id()
        baseline_root = run_layout.run_root(ep_id, baseline_run_id) if baseline_run_id else None

        current_tracks = helpers.coerce_int(detect_marker.get("tracks")) if isinstance(detect_marker, dict) else None
        current_forced_share = _safe_ratio(forced_splits, current_tracks)
        current_singleton_rate = _singleton_rate_from_metrics(track_metrics_payload)
        current_fused_pairs, current_reid_comparisons, _ = _fused_pairs_for_run(_scoped_manifests_dir)

        baseline_rtf = _detect_rtf_for_run(baseline_root) if baseline_root else None
        baseline_forced_share = _forced_splits_share_for_run(baseline_root) if baseline_root else None
        baseline_singleton_rate = _singleton_rate_from_metrics(
            _read_json_best_effort(baseline_root / "track_metrics.json") if baseline_root else None
        )
        baseline_fused_pairs = _fused_pairs_for_run(baseline_root)[0] if baseline_root else None

        warn_msgs: list[str] = []
        fail_msgs: list[str] = []

        if detect_rtf is not None:
            warn_thresh = max((baseline_rtf or 0) * 1.3, 2.5)
            fail_thresh = max((baseline_rtf or 0) * 2.0, 4.0)
            if detect_rtf > fail_thresh:
                fail_msgs.append(f"Detect RTF {detect_rtf:.2f}x > {fail_thresh:.2f}x (quality gate fail)")
            elif detect_rtf > warn_thresh:
                warn_msgs.append(f"Detect RTF {detect_rtf:.2f}x > {warn_thresh:.2f}x")

        if current_forced_share is not None:
            if baseline_forced_share is None:
                warn_limit = 0.85
                fail_limit = 0.95
            else:
                warn_limit = baseline_forced_share + 0.05
                fail_limit = baseline_forced_share + 0.10
            if current_forced_share > fail_limit:
                fail_msgs.append(
                    f"Forced splits share {current_forced_share:.2%} > {fail_limit:.2%} (quality gate fail)"
                )
            elif current_forced_share > warn_limit:
                warn_msgs.append(f"Forced splits share {current_forced_share:.2%} > {warn_limit:.2%}")

        if current_singleton_rate is not None:
            warn_limit = (baseline_singleton_rate + 0.05) if baseline_singleton_rate is not None else 0.65
            fail_limit = 0.75
            if current_singleton_rate > fail_limit:
                fail_msgs.append(
                    f"Singleton rate {current_singleton_rate:.2%} > {fail_limit:.2%} (quality gate fail)"
                )
            elif current_singleton_rate > warn_limit:
                warn_msgs.append(f"Singleton rate {current_singleton_rate:.2%} > {warn_limit:.2%}")

        if current_fused_pairs is not None:
            if baseline_fused_pairs is None:
                warn_limit = 10
                fail_limit = 0
            else:
                warn_limit = int(baseline_fused_pairs * 0.8)
                fail_limit = int(baseline_fused_pairs * 0.5)
            if current_fused_pairs == 0 or (baseline_fused_pairs is not None and current_fused_pairs < fail_limit):
                fail_msgs.append(
                    f"Fused pairs {current_fused_pairs} below baseline ({baseline_fused_pairs}) (quality gate fail)"
                )
            elif baseline_fused_pairs is None and current_fused_pairs < warn_limit:
                warn_msgs.append(f"Fused pairs {current_fused_pairs} < {warn_limit}")
            elif baseline_fused_pairs is not None and current_fused_pairs < warn_limit:
                warn_msgs.append(f"Fused pairs {current_fused_pairs} < {warn_limit}")

        reid_enabled = _track_fusion_reid_enabled_config()
        if reid_enabled and (current_reid_comparisons is not None and current_reid_comparisons == 0):
            reid_skip_reason = None
            body_reid = body_marker.get("body_reid") if isinstance(body_marker, dict) else None
            if isinstance(body_reid, dict):
                reid_skip_reason = body_reid.get("reid_skip_reason")
            warn_msgs.append(
                "Re-ID handoff enabled but 0 comparisons performed"
                + (f" (skip_reason={reid_skip_reason})" if reid_skip_reason else "")
                + ". Verify torchreid install or disable reid_handoff."
            )

        if baseline_run_id:
            st.caption(f"Baseline run_id: `{baseline_run_id}`")
        else:
            st.caption("Baseline run_id: none (using absolute thresholds)")

        for msg in fail_msgs:
            st.error(f"🚨 {msg}")
        for msg in warn_msgs:
            st.warning(f"⚠️ {msg}")
        if not fail_msgs and not warn_msgs:
            st.success("No regression flags triggered.")

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
            "Episode-level S3 prefixes (legacy, not run-scoped) → "
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
# Setup pipeline config + run-scoped artifacts (body tracking / fusion / PDF).
body_tracking_required = bool(selected_attempt_run_id)
body_tracking_config_enabled = _body_tracking_enabled_config()
body_tracking_enabled = bool(body_tracking_required and body_tracking_config_enabled)
track_fusion_required = bool(selected_attempt_run_id)
track_fusion_config_enabled = _track_fusion_enabled_config()
track_fusion_enabled = bool(track_fusion_required and body_tracking_enabled and track_fusion_config_enabled)
track_fusion_reid_enabled = bool(selected_attempt_run_id) and track_fusion_config_enabled and _track_fusion_reid_enabled_config()

body_tracking_dir = manifests_dir / "body_tracking"
body_detections_path = body_tracking_dir / "body_detections.jsonl"
body_tracks_path = body_tracking_dir / "body_tracks.jsonl"
track_fusion_path = body_tracking_dir / "track_fusion.json"
screentime_comparison_path = body_tracking_dir / "screentime_comparison.json"
legacy_body_tracking_dir = _manifests_root / "body_tracking"
legacy_body_detections_path = legacy_body_tracking_dir / "body_detections.jsonl"
legacy_body_tracks_path = legacy_body_tracking_dir / "body_tracks.jsonl"
legacy_track_fusion_path = legacy_body_tracking_dir / "track_fusion.json"

faces_presence = stage_layout.ArtifactPresence(local=faces_path.exists(), remote=False, path=str(faces_path))
body_detections_presence = stage_layout.ArtifactPresence(local=body_detections_path.exists(), remote=False, path=str(body_detections_path))
body_tracks_presence = stage_layout.ArtifactPresence(local=body_tracks_path.exists(), remote=False, path=str(body_tracks_path))
track_fusion_presence = stage_layout.ArtifactPresence(local=track_fusion_path.exists(), remote=False, path=str(track_fusion_path))

if selected_attempt_run_id:
    _presence_map = _cached_run_artifact_presence(
        ep_id,
        selected_attempt_run_id,
        (
            "faces.jsonl",
            "body_tracking/body_detections.jsonl",
            "body_tracking/body_tracks.jsonl",
            "body_tracking/track_fusion.json",
        ),
    )
    faces_presence = _presence_from_cache(_presence_map.get("faces.jsonl"), faces_path)
    body_detections_presence = _presence_from_cache(
        _presence_map.get("body_tracking/body_detections.jsonl"),
        body_detections_path,
    )
    body_tracks_presence = _presence_from_cache(
        _presence_map.get("body_tracking/body_tracks.jsonl"),
        body_tracks_path,
    )
    track_fusion_presence = _presence_from_cache(
        _presence_map.get("body_tracking/track_fusion.json"),
        track_fusion_path,
    )

body_tracking_marker_path = _scoped_markers_dir / "body_tracking.json"
body_fusion_marker_path = _scoped_markers_dir / "body_tracking_fusion.json"
legacy_body_tracking_marker_path = _runs_root / "body_tracking.json"
legacy_body_fusion_marker_path = _runs_root / "body_tracking_fusion.json"

body_tracking_status_value = "missing"
body_tracking_error: str | None = None
body_tracking_marker_payload: dict[str, Any] | None = None
body_tracking_manifest_fallback = False
body_tracking_legacy_available = False

body_fusion_status_value = "missing"
body_fusion_error: str | None = None
body_fusion_marker_payload: dict[str, Any] | None = None
body_fusion_manifest_fallback = False
body_fusion_legacy_available = False

pdf_export_status_value = "missing"
pdf_export_detail: str | None = None
fusion_mode_label: str | None = None
fusion_mode_detail: str | None = None
fusion_mode_hint: str | None = None
_running_body_tracking_job: dict[str, Any] | None = None
_running_body_fusion_job: dict[str, Any] | None = None
export_index: dict[str, Any] | None = None

if selected_attempt_run_id:
    _jobs_payload_for_downstream = _cached_episode_jobs(ep_id)
    _running_body_tracking_job = _find_running_subprocess_job(
        _jobs_payload_for_downstream, job_type="body_tracking", run_id=selected_attempt_run_id
    )
    _running_body_fusion_job = _find_running_subprocess_job(
        _jobs_payload_for_downstream, job_type="body_tracking_fusion", run_id=selected_attempt_run_id
    )
    body_tracking_running = bool(_running_body_tracking_job)
    body_fusion_running = bool(_running_body_fusion_job)
    if body_tracking_running:
        body_tracking_status_value = "running"
    if body_fusion_running:
        body_fusion_status_value = "running"

    if not body_tracking_running:
        body_tracking_marker_payload, _marker_mtime, _ = _load_marker_payload_with_fallback(
            body_tracking_marker_path,
            legacy_body_tracking_marker_path,
            run_id=selected_attempt_run_id,
        )
    if body_tracking_marker_payload and not body_tracking_running:
        marker_status = str((body_tracking_marker_payload or {}).get("status") or "").strip().lower()
        marker_error = (body_tracking_marker_payload or {}).get("error")
        marker_run_id = (body_tracking_marker_payload or {}).get("run_id")
        marker_run_matches = isinstance(marker_run_id, str) and marker_run_id.strip() == selected_attempt_run_id
        run_scoped_ok = (
            stage_layout.artifact_available(body_detections_presence)
            and stage_layout.artifact_available(body_tracks_presence)
        )
        legacy_ok = bool(legacy_body_detections_path.exists() and legacy_body_tracks_path.exists())
        body_tracking_legacy_available = bool(legacy_ok and marker_run_matches)
        if marker_status in {"error", "failed"} or marker_error:
            body_tracking_status_value = "error"
            body_tracking_error = str(marker_error) if marker_error else f"marker_status={marker_status}"
        elif marker_status == "success" and marker_run_matches and run_scoped_ok:
            body_tracking_status_value = "success"
        elif marker_status == "success":
            body_tracking_status_value = "stale"
            if not marker_run_matches:
                body_tracking_error = "run_id_mismatch"
            elif not run_scoped_ok:
                body_tracking_error = "missing_artifacts"
    elif not body_tracking_running:
        artifacts_ok = (
            stage_layout.artifact_available(body_detections_presence)
            and stage_layout.artifact_available(body_tracks_presence)
        )
        if artifacts_ok:
            body_tracking_status_value = "success"
            body_tracking_manifest_fallback = True

    if not body_fusion_running:
        body_fusion_marker_payload, _marker_mtime, _ = _load_marker_payload_with_fallback(
            body_fusion_marker_path,
            legacy_body_fusion_marker_path,
            run_id=selected_attempt_run_id,
        )
    if body_fusion_marker_payload and not body_fusion_running:
        marker_status = str((body_fusion_marker_payload or {}).get("status") or "").strip().lower()
        marker_error = (body_fusion_marker_payload or {}).get("error")
        marker_run_id = (body_fusion_marker_payload or {}).get("run_id")
        marker_run_matches = isinstance(marker_run_id, str) and marker_run_id.strip() == selected_attempt_run_id
        run_scoped_ok = stage_layout.artifact_available(track_fusion_presence)
        legacy_ok = bool(legacy_track_fusion_path.exists())
        body_fusion_legacy_available = bool(legacy_ok and marker_run_matches)
        if marker_status in {"error", "failed"} or marker_error:
            body_fusion_status_value = "error"
            body_fusion_error = str(marker_error) if marker_error else f"marker_status={marker_status}"
        elif marker_status == "success" and marker_run_matches and run_scoped_ok:
            body_fusion_status_value = "success"
        elif marker_status == "success":
            body_fusion_status_value = "stale"
            if not marker_run_matches:
                body_fusion_error = "run_id_mismatch"
            elif not run_scoped_ok:
                body_fusion_error = "missing_artifacts"
    elif not body_fusion_running:
        artifacts_ok = stage_layout.artifact_available(track_fusion_presence)
        if artifacts_ok:
            body_fusion_status_value = "success"
            body_fusion_manifest_fallback = True

    if not body_tracking_enabled and body_tracking_status_value in {"missing", "unknown"}:
        body_tracking_status_value = "error"
        body_tracking_error = body_tracking_error or "disabled_by_config"
    if not track_fusion_enabled and body_fusion_status_value in {"missing", "unknown"}:
        body_fusion_status_value = "error"
        if not body_tracking_enabled:
            body_fusion_error = body_fusion_error or "body_tracking_disabled"
        else:
            body_fusion_error = body_fusion_error or "disabled_by_config"

    # Prefer episode_status signals when local markers are stale/missing.
    body_tracking_phase_value = str(body_tracking_phase_status.get("status") or "").strip().lower()
    if body_tracking_phase_value and body_tracking_phase_value not in {"missing", "unknown", "stale"}:
        if body_tracking_status_value in {"missing", "unknown", "stale"}:
            body_tracking_status_value = body_tracking_phase_value
            if body_tracking_phase_value in {"error", "failed"}:
                body_tracking_error = (
                    body_tracking_phase_status.get("error")
                    or body_tracking_phase_status.get("error_reason")
                    or body_tracking_error
                )

    track_fusion_phase_value = str(track_fusion_phase_status.get("status") or "").strip().lower()
    if track_fusion_phase_value and track_fusion_phase_value not in {"missing", "unknown", "stale"}:
        if body_fusion_status_value in {"missing", "unknown", "stale"}:
            body_fusion_status_value = track_fusion_phase_value
            if track_fusion_phase_value in {"error", "failed"}:
                body_fusion_error = (
                    track_fusion_phase_status.get("error")
                    or track_fusion_phase_status.get("error_reason")
                    or body_fusion_error
                )

    # PDF export is run-scoped via exports/export_index.json.
    try:
        from apps.api.services.run_artifact_store import read_export_index

        export_index = read_export_index(ep_id, selected_attempt_run_id)
    except Exception:
        export_index = None
    if isinstance(export_index, dict):
        export_type = export_index.get("export_type")
        export_bytes = export_index.get("export_bytes")
        export_upload = export_index.get("export_upload") if isinstance(export_index.get("export_upload"), dict) else {}
        export_success = export_type == "pdf" and isinstance(export_bytes, int) and export_bytes > 0
        if export_success:
            pdf_export_status_value = "success"
            if export_upload.get("attempted") and not export_upload.get("success"):
                pdf_export_detail = "PDF generated (S3 upload failed)"
            elif export_upload.get("success"):
                pdf_export_detail = "PDF uploaded"
            else:
                pdf_export_detail = "PDF generated"

    pdf_phase_value = str(pdf_phase_status.get("status") or "").strip().lower()
    if pdf_phase_value and pdf_phase_value not in {"missing", "unknown", "stale"}:
        if pdf_export_status_value in {"missing", "unknown", "stale"}:
            pdf_export_status_value = pdf_phase_value
        if pdf_phase_value in {"error", "failed"}:
            pdf_export_detail = (
                pdf_phase_status.get("error")
                or pdf_phase_status.get("error_reason")
                or pdf_export_detail
            )

    if track_fusion_enabled:
        body_reid = body_tracking_marker_payload.get("body_reid") if isinstance(body_tracking_marker_payload, dict) else None
        enabled_effective = False
        skip_reason = None
        runtime_error = None
        if isinstance(body_reid, dict):
            enabled_effective = bool(body_reid.get("enabled_effective"))
            skip_reason = body_reid.get("reid_skip_reason")
            runtime_error = body_reid.get("torchreid_runtime_error")

        if not track_fusion_reid_enabled:
            fusion_mode_label = "IoU-only"
            fusion_mode_detail = "reid_handoff.enabled=false"
        elif enabled_effective:
            fusion_mode_label = "Hybrid (IoU + Re-ID)"
            fusion_mode_detail = None
        else:
            fusion_mode_label = "IoU-only"
            detail_bits: list[str] = []
            runtime_clean = runtime_error.strip() if isinstance(runtime_error, str) else ""
            if "missing torchreid.utils" in runtime_clean:
                fusion_mode_detail = "Re-ID unavailable: missing torchreid.utils"
                fusion_mode_hint = (
                    "Install deep-person-reid (pip install -r requirements-ml.txt); "
                    "pip torchreid==0.2.x is incompatible"
                )
            else:
                if isinstance(skip_reason, str) and skip_reason.strip():
                    detail_bits.append(f"skip_reason={skip_reason.strip()}")
                if runtime_clean:
                    detail_bits.append(f"torchreid_error={runtime_clean}")
                fusion_mode_detail = "; ".join(detail_bits) if detail_bits else "Re-ID unavailable"

body_tracking_progress_payload = None
track_fusion_progress_payload = None
if selected_attempt_run_id:
    body_tracking_progress_payload = _read_json_payload(manifests_dir / "progress_body_tracking.json")
    track_fusion_progress_payload = _read_json_payload(manifests_dir / "progress_body_tracking_fusion.json")
body_tracking_status_progress = (
    body_tracking_phase_status.get("progress") if isinstance(body_tracking_phase_status, dict) else None
)
track_fusion_status_progress = (
    track_fusion_phase_status.get("progress") if isinstance(track_fusion_phase_status, dict) else None
)
pdf_status_progress = pdf_phase_status.get("progress") if isinstance(pdf_phase_status, dict) else None

status_running = (
    detect_status_value == "running"
    or faces_status_value == "running"
    or cluster_status_value == "running"
    or str(detect_job_state or "").lower() == "running"
    or body_tracking_status_value == "running"
    or body_fusion_status_value == "running"
    or pdf_export_status_value == "running"
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
faces_manifest_exists = faces_presence.local
faces_manifest_available = stage_layout.artifact_available(faces_presence)
if faces_manifest_exists:
    faces_manifest_count = _count_manifest_rows(faces_path) or 0
if faces_status_value == "success":
    faces_ready_state = True
elif faces_status_value in {"missing", "unknown", "stale"} and faces_manifest_available:
    # Manifest exists but API reports missing/unknown/stale - use manifest fallback
    faces_ready_state = True
    faces_manifest_fallback = True
if faces_count_value is None and faces_manifest_count is not None:
    faces_count_value = faces_manifest_count

# If status API is missing but run-scoped artifacts exist, only synthesize a "success"
# when we can also validate a matching run marker (prevents stale/other-run promotion).
if not detect_phase_status and manifest_state["manifest_ready"]:
    marker_ok = False
    marker_payload: dict[str, Any] | None = None
    marker_path = _scoped_markers_dir / "detect_track.json"
    if marker_path.exists():
        try:
            marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            marker_payload = None
    if isinstance(marker_payload, dict):
        marker_status = str(marker_payload.get("status") or "").strip().lower()
        marker_error = marker_payload.get("error")
        marker_run_id = marker_payload.get("run_id")
        marker_run_matches = True
        if selected_attempt_run_id:
            marker_run_matches = isinstance(marker_run_id, str) and marker_run_id.strip() == selected_attempt_run_id
        marker_ok = marker_status == "success" and not marker_error and marker_run_matches

    if marker_ok:
        detect_phase_status = {
            "status": "success",
            "run_id": selected_attempt_run_id,
            "detections": _count_manifest_rows(detections_path) or 0,
            "tracks": _count_manifest_rows(tracks_path) or 0,
            "finished_at": (marker_payload or {}).get("finished_at"),
            "source": "marker_manifest_fallback",
            "metadata_missing": True,
        }
        detect_status_value = "success"
        tracks_ready = True
        using_manifest_fallback = True

autorun_phase_raw = st.session_state.get(f"{ep_id}::autorun_phase")
body_tracking_started_at = (
    body_tracking_phase_status.get("started_at")
    if isinstance(body_tracking_phase_status, dict) and body_tracking_phase_status.get("started_at")
    else (body_tracking_marker_payload or {}).get("started_at")
)
body_tracking_finished_at = (
    body_tracking_phase_status.get("finished_at")
    if isinstance(body_tracking_phase_status, dict) and body_tracking_phase_status.get("finished_at")
    else (body_tracking_marker_payload or {}).get("finished_at")
)
body_tracking_runtime = _format_runtime(_runtime_from_iso(body_tracking_started_at, body_tracking_finished_at))
body_detections_count = _count_manifest_rows(body_detections_path)
body_tracks_count = _count_manifest_rows(body_tracks_path)

track_fusion_started_at = (
    track_fusion_phase_status.get("started_at")
    if isinstance(track_fusion_phase_status, dict) and track_fusion_phase_status.get("started_at")
    else (body_fusion_marker_payload or {}).get("started_at")
)
track_fusion_finished_at = (
    track_fusion_phase_status.get("finished_at")
    if isinstance(track_fusion_phase_status, dict) and track_fusion_phase_status.get("finished_at")
    else (body_fusion_marker_payload or {}).get("finished_at")
)
track_fusion_runtime = _format_runtime(_runtime_from_iso(track_fusion_started_at, track_fusion_finished_at))
track_fusion_faces_ready = stage_layout.artifact_available(faces_presence)
track_fusion_body_tracks_ready = stage_layout.artifact_available(body_tracks_presence)

pdf_export_started_at = (
    pdf_phase_status.get("started_at") if isinstance(pdf_phase_status, dict) else None
)
pdf_export_finished_at = (
    pdf_phase_status.get("finished_at") if isinstance(pdf_phase_status, dict) else None
)
pdf_export_runtime = _format_runtime(_runtime_from_iso(pdf_export_started_at, pdf_export_finished_at))

# Add pipeline state indicators (even if status API is temporarily unavailable)
st.markdown("## Setup Pipeline")

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

attempt_col1, = st.columns([1])
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
        "Selected Attempt (run_id)",
        attempt_options,
        key=_active_run_id_key,
        format_func=_format_attempt,
        help="Scopes status/artifacts and per-stage runs to this attempt.",
        disabled=job_running,
    )

selected_attempt_label = st.session_state.get(_active_run_id_key)
selected_attempt_label = selected_attempt_label.strip() if isinstance(selected_attempt_label, str) else ""
if selected_attempt_label:
    st.caption(f"Selected attempt: `{selected_attempt_label}`")
else:
    st.caption("Selected attempt: legacy (no run_id)")
if isinstance(api_active_run_id, str) and api_active_run_id.strip():
    if api_active_run_id.strip() != selected_attempt_label:
        st.caption(f"API active_run_id: `{api_active_run_id.strip()}`")

with st.expander("Recent Attempts", expanded=False):
    recent_runs: list[dict[str, Any]] = []
    for run_id in run_layout.list_run_ids(ep_id):
        if selected_attempt_run_id and run_id == selected_attempt_run_id:
            continue
        run_root = run_layout.run_root(ep_id, run_id)
        status_path = run_root / "episode_status.json"
        status_payload = _read_json_payload(status_path) if status_path.exists() else None
        stages = status_payload.get("stages") if isinstance(status_payload, dict) else {}
        completed_stages = []
        if isinstance(stages, dict):
            for stage_key, entry in stages.items():
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("status") or "").strip().lower() == "success":
                    completed_stages.append(stage_layout.stage_label(stage_key))
        sort_mtime = _safe_mtime(status_path) or _safe_mtime(run_root)
        if not sort_mtime and isinstance(status_payload, dict):
            updated_at = status_payload.get("updated_at")
            if isinstance(updated_at, str):
                sort_mtime = _safe_mtime(run_root)
        recent_runs.append(
            {
                "run_id": run_id,
                "completed": ", ".join(completed_stages) if completed_stages else "No stages completed",
                "mtime": sort_mtime,
            }
        )

    recent_runs.sort(key=lambda item: item.get("mtime") or 0.0, reverse=True)
    recent_runs = recent_runs[:5]

    if not recent_runs:
        st.caption("No previous attempts found.")
    else:
        header_cols = st.columns([3, 4, 2, 1])
        header_cols[0].caption("Run ID")
        header_cols[1].caption("Completed stages")
        header_cols[2].caption("Last update")
        header_cols[3].caption("")
        for entry in recent_runs:
            run_id = entry["run_id"]
            run_root = run_layout.run_root(ep_id, run_id)
            status_path = run_root / "episode_status.json"
            updated_iso = None
            if status_path.exists():
                mtime = _safe_mtime(status_path)
                if mtime:
                    updated_iso = (
                        datetime.fromtimestamp(mtime, tz=timezone.utc)
                        .replace(microsecond=0)
                        .isoformat()
                        .replace("+00:00", "Z")
                    )
            updated_label = _format_timestamp(updated_iso) or "—"
            row_cols = st.columns([3, 4, 2, 1])
            row_cols[0].code(run_id)
            row_cols[1].caption(entry["completed"])
            row_cols[2].caption(updated_label)
            if row_cols[3].button(
                "Select",
                key=f"{ep_id}::select_recent_attempt::{run_id}",
                use_container_width=True,
            ):
                st.session_state[_active_run_id_pending_key] = run_id
                st.session_state[_status_force_refresh_key(ep_id)] = True
                st.rerun()

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

_SETUP_STAGE_PLAN = ("detect", "faces", "cluster", "body_tracking", "track_fusion", "pdf")


def _setup_stage_plan() -> list[str]:
    return list(_SETUP_STAGE_PLAN)


def _normalize_stage_status(value: str | None) -> str:
    return str(value or "").strip().lower()


def _setup_stage_statuses() -> dict[str, str]:
    return {
        "detect": _normalize_stage_status(detect_status_value),
        "faces": "success" if faces_ready_state else _normalize_stage_status(faces_status_value),
        "cluster": _normalize_stage_status(cluster_status_value),
        "body_tracking": _normalize_stage_status(body_tracking_status_value),
        "track_fusion": _normalize_stage_status(body_fusion_status_value),
        "pdf": _normalize_stage_status(pdf_export_status_value),
    }


def _is_setup_stage_done(status: str) -> bool:
    return status in {"success", "completed", "done"}


def is_setup_pipeline_complete(stage_statuses: dict[str, str] | None = None) -> bool:
    statuses = stage_statuses or _setup_stage_statuses()
    return all(_is_setup_stage_done(statuses.get(stage, "")) for stage in _setup_stage_plan())


setup_stage_plan = _setup_stage_plan()
setup_stage_statuses = _setup_stage_statuses()

autorun_active_flag = bool(st.session_state.get(_autorun_key, False))
autorun_phase_value = st.session_state.get(_autorun_phase_key)
autorun_phase_label = None
if autorun_active_flag and autorun_phase_value:
    autorun_phase_key = stage_layout.normalize_stage_key(str(autorun_phase_value))
    if autorun_phase_key:
        autorun_phase_label = stage_layout.stage_label(autorun_phase_key)

setup_complete = bool(selected_attempt_run_id and is_setup_pipeline_complete(setup_stage_statuses))
running_stage_label = next(
    (
        stage_layout.stage_label(stage)
        for stage in setup_stage_plan
        if setup_stage_statuses.get(stage) in {"running", "finalizing", "syncing"}
    ),
    None,
)
error_stage_label = next(
    (
        stage_layout.stage_label(stage)
        for stage in setup_stage_plan
        if setup_stage_statuses.get(stage) in {"error", "failed"}
    ),
    None,
)
if not selected_attempt_run_id:
    pipeline_state_label = "Select run_id"
elif error_stage_label:
    pipeline_state_label = f"Error ({error_stage_label})"
elif setup_complete:
    pipeline_state_label = "Complete"
elif autorun_phase_label:
    pipeline_state_label = f"Running ({autorun_phase_label})"
elif running_stage_label:
    pipeline_state_label = f"Running ({running_stage_label})"
else:
    pipeline_state_label = "Ready"

if not setup_complete:
    _clear_improve_faces_state(ep_id, _resolve_session_run_id(ep_id))

header_col1, header_col2, header_col3 = st.columns(3)
with header_col1:
    st.metric("Episode ID", ep_id)
with header_col2:
    st.metric("Run ID", selected_attempt_run_id or "legacy")
with header_col3:
    st.metric("Pipeline State", pipeline_state_label)

action_col1, action_col2, action_col3 = st.columns(3)
with action_col1:
    autorun_active = bool(st.session_state.get(_autorun_key, False))
    autorun_start_requested_key = f"{ep_id}::autorun_start_requested"
    autorun_rerun_requested_key = f"{ep_id}::autorun_rerun_requested"
    if autorun_active:
        if st.button("⏹️ Stop Auto-Run", key=f"{ep_id}::stop_autorun_header", use_container_width=True):
            st.session_state[_autorun_key] = False
            st.session_state[_autorun_phase_key] = None
            st.session_state.pop(_autorun_run_id_key, None)
            st.toast("Auto-run stopped")
            st.rerun()
    else:
        autorun_disabled = not local_video_exists or _job_active(ep_id) or st.session_state.get(running_job_key, False)
        if st.button(
            "🚀 Run New Attempt (Setup Pipeline)",
            key=f"{ep_id}::start_autorun_header",
            use_container_width=True,
            type="primary",
            disabled=autorun_disabled,
        ):
            st.session_state[autorun_start_requested_key] = True
            st.rerun()
        rerun_disabled = autorun_disabled or not selected_attempt_run_id
        rerun_help = "Select a run-scoped attempt (run_id) above." if not selected_attempt_run_id else None
        if st.button(
            "↻ Re-run selected attempt",
            key=f"{ep_id}::rerun_autorun_header",
            use_container_width=True,
            type="secondary",
            disabled=rerun_disabled,
            help=rerun_help,
        ):
            st.session_state[autorun_rerun_requested_key] = True
            st.rerun()
with action_col2:
    pdf_header_disabled = (
        job_running
        or not selected_attempt_run_id
        or pdf_export_status_value == "running"
    )
    if st.button(
        "📄 Export Debug PDF",
        key=f"{ep_id}::export_pdf_header",
        use_container_width=True,
        disabled=pdf_header_disabled,
    ):
        with st.spinner("Generating PDF report..."):
            ok, msg = _trigger_pdf_export_if_needed(ep_id, selected_attempt_run_id, cfg)
        if not ok:
            st.error(msg)
        else:
            st.success(msg)
            st.session_state[_status_force_refresh_key(ep_id)] = True
            st.rerun()
with action_col3:
    if setup_complete:
        if st.button(
            "Continue to Faces Review",
            key=f"{ep_id}::{selected_attempt_run_id or 'legacy'}::nav_faces_review",
            use_container_width=True,
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
        st.caption("Complete the setup pipeline to continue.")

st.markdown("### Selected Attempt Status")
st.caption("Pipeline dependencies: detect → faces → cluster → body_tracking → track_fusion → pdf")
coreml_available = status_payload.get("coreml_available") if status_payload else None
if coreml_available is False and helpers.is_apple_silicon():
    st.warning(
        "⚠️ CoreML acceleration isn't available on this host. Install `onnxruntime-coreml` to avoid CPU-only runs."
    )
running_jobs_snapshot = helpers.get_all_running_jobs_for_episode(ep_id)
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
        running_detect_snapshot = running_jobs_snapshot.get("detect_track") if running_jobs_snapshot else None
        if running_detect_snapshot and running_detect_snapshot.get("progress_pct") is not None:
            st.progress(min(running_detect_snapshot["progress_pct"] / 100, 1.0))
            if running_detect_snapshot.get("message"):
                st.caption(running_detect_snapshot["message"])
        else:
            st.progress(0.05)
        detect_progress = detect_phase_status.get("progress")
        progress_line = helpers.stage_progress_line(detect_progress) if detect_progress else None
        if progress_line:
            st.caption(progress_line)
        stall_msg = helpers.stage_progress_stall_message(detect_progress) if detect_progress else None
        if stall_msg:
            st.warning(stall_msg)
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
    st.markdown('<a name="faces-harvest-card"></a>', unsafe_allow_html=True)
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
    elif faces_status_value == "running":
        st.info("⏳ **Faces Harvest**: Running")
        started = _format_timestamp(faces_phase_status.get("started_at"))
        if started:
            st.caption(f"Started at {started}")
        running_faces_snapshot = running_jobs_snapshot.get("faces_embed") if running_jobs_snapshot else None
        if running_faces_snapshot and running_faces_snapshot.get("progress_pct") is not None:
            st.progress(min(running_faces_snapshot["progress_pct"] / 100, 1.0))
            if running_faces_snapshot.get("message"):
                st.caption(running_faces_snapshot["message"])
        else:
            st.progress(0.05)
        faces_progress = faces_phase_status.get("progress")
        progress_line = helpers.stage_progress_line(faces_progress) if faces_progress else None
        if progress_line:
            st.caption(progress_line)
        stall_msg = helpers.stage_progress_stall_message(faces_progress) if faces_progress else None
        if stall_msg:
            st.warning(stall_msg)
        st.caption("Live progress appears in the log panel below.")
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
        running_cluster_snapshot = running_jobs_snapshot.get("cluster") if running_jobs_snapshot else None
        if running_cluster_snapshot and running_cluster_snapshot.get("progress_pct") is not None:
            st.progress(min(running_cluster_snapshot["progress_pct"] / 100, 1.0))
            if running_cluster_snapshot.get("message"):
                st.caption(running_cluster_snapshot["message"])
        else:
            st.progress(0.05)
        cluster_progress = cluster_phase_status.get("progress")
        progress_line = helpers.stage_progress_line(cluster_progress) if cluster_progress else None
        if progress_line:
            st.caption(progress_line)
        stall_msg = helpers.stage_progress_stall_message(cluster_progress) if cluster_progress else None
        if stall_msg:
            st.warning(stall_msg)
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

row2_col1, row2_col2, row2_col3 = st.columns(3)

with row2_col1:
    st.markdown('<a name="body-tracking-card"></a>', unsafe_allow_html=True)
    body_tracking_progress_display = body_tracking_status_progress or body_tracking_progress_payload
    if not selected_attempt_run_id:
        st.info("⏳ **Body Tracking**: Select a run_id")
        st.caption("Select a run-scoped attempt (run_id) to enable body tracking.")
    elif not body_tracking_config_enabled:
        st.error("⚠️ **Body Tracking**: Disabled (required)")
        st.caption("Enable body tracking (AUTO_RUN_BODY_TRACKING or body_detection.yaml).")
    elif body_tracking_status_value == "success":
        runtime_label = body_tracking_runtime or "n/a"
        st.success(f"✅ **Body Tracking**: Complete (Runtime: {runtime_label})")
        if body_detections_count is not None:
            st.caption(f"Body detections: {body_detections_count:,}")
        if body_tracks_count is not None:
            st.caption(f"Body tracks: {body_tracks_count:,}")
    elif body_tracking_status_value == "running":
        st.info("⏳ **Body Tracking**: Running")
        started = _format_timestamp(body_tracking_started_at)
        if started:
            st.caption(f"Started at {started}")
        _render_downstream_progress(body_tracking_progress_display, running=True)
    elif body_tracking_status_value in {"error", "failed"}:
        st.error("⚠️ **Body Tracking**: Failed")
        if body_tracking_error:
            st.caption(body_tracking_error)
    elif body_tracking_status_value not in {"missing", "unknown"}:
        st.warning(f"⚠️ **Body Tracking**: {body_tracking_status_value.title()}")
    else:
        st.info("⏳ **Body Tracking**: Ready to run")
        st.caption("Run body tracking to generate body tracks.")
    if body_tracking_manifest_fallback:
        st.caption("ℹ️ Using manifest fallback; run-scoped marker/artifacts may be missing.")
    finished = _format_timestamp(body_tracking_finished_at)
    if finished:
        st.caption(f"Last run: {finished}")
    if body_tracking_runtime:
        st.caption(f"Run Duration: {body_tracking_runtime}")
    elif body_tracking_status_value == "success":
        st.caption("Run Duration: n/a")

    body_tracking_btn_label = (
        "Re-run Body Tracking" if body_tracking_status_value == "success" else "Run Body Tracking"
    )
    body_tracking_btn_disabled = (
        job_running
        or _running_body_tracking_job is not None
        or not selected_attempt_run_id
        or not body_tracking_enabled
        or not local_video_exists
    )
    if st.button(
        body_tracking_btn_label,
        key=f"{ep_id}::{selected_attempt_run_id or 'legacy'}::run_body_tracking",
        use_container_width=True,
        disabled=body_tracking_btn_disabled,
    ):
        with st.spinner("Starting body tracking..."):
            try:
                helpers.api_post(
                    "/jobs/body_tracking/run",
                    json={"ep_id": ep_id, "run_id": selected_attempt_run_id},
                    timeout=15,
                )
            except Exception as exc:
                st.error(f"Failed to start body tracking: {exc}")
            else:
                st.success("Body tracking started.")
                st.session_state[_status_force_refresh_key(ep_id)] = True
                st.rerun()

    _render_downstream_log_expander(
        "Body Tracking",
        marker_payload=body_tracking_marker_payload,
        progress_payload=body_tracking_progress_display,
        artifacts_hint="Body tracking artifacts found via manifest fallback."
        if body_tracking_manifest_fallback
        else None,
    )
    body_tracking_artifacts = [
        ("body_tracking/body_detections.jsonl", body_detections_path, "run"),
        ("body_tracking/body_tracks.jsonl", body_tracks_path, "run"),
    ]
    show_legacy_body_tracking = body_tracking_legacy_available and not stage_layout.artifact_available(
        body_tracks_presence
    )
    if show_legacy_body_tracking:
        body_tracking_artifacts.extend(
            [
                ("legacy/body_tracking/body_detections.jsonl", legacy_body_detections_path, "legacy"),
                ("legacy/body_tracking/body_tracks.jsonl", legacy_body_tracks_path, "legacy"),
            ]
        )
    body_tracking_hint = (
        "⚠️ Legacy artifacts detected; run-scoped artifacts are missing for this attempt."
        if show_legacy_body_tracking
        else None
    )
    _render_stage_artifacts_expander(
        "Body Tracking",
        body_tracking_artifacts,
        hint=body_tracking_hint,
    )

with row2_col2:
    track_fusion_progress_display = track_fusion_status_progress or track_fusion_progress_payload
    if not selected_attempt_run_id:
        st.info("⏳ **Track Fusion**: Select a run_id")
        st.caption("Select a run-scoped attempt (run_id) to enable track fusion.")
    elif not track_fusion_config_enabled or not body_tracking_enabled:
        st.error("⚠️ **Track Fusion**: Disabled (required)")
        st.caption("Enable track fusion (track_fusion.enabled) and body tracking.")
    elif body_fusion_status_value == "success":
        runtime_label = track_fusion_runtime or "n/a"
        st.success(f"✅ **Track Fusion**: Complete (Runtime: {runtime_label})")
        if track_fusion_path.exists():
            st.caption(f"Fusion output: {helpers.link_local(track_fusion_path)}")
        if screentime_comparison_path.exists():
            st.caption(f"Fusion comparison: {helpers.link_local(screentime_comparison_path)}")
    elif body_fusion_status_value == "running":
        st.info("⏳ **Track Fusion**: Running")
        started = _format_timestamp(track_fusion_started_at)
        if started:
            st.caption(f"Started at {started}")
        _render_downstream_progress(track_fusion_progress_display, running=True)
    elif body_fusion_status_value in {"error", "failed"}:
        st.error("⚠️ **Track Fusion**: Failed")
        if body_fusion_error:
            st.caption(body_fusion_error)
        else:
            st.warning(f"⚠️ **Track Fusion**: {body_fusion_status_value.title()}")
    elif body_fusion_status_value in {"missing", "unknown"}:
        missing = []
        if not track_fusion_body_tracks_ready:
            missing.append("body_tracking/body_tracks.jsonl")
        if not track_fusion_faces_ready:
            missing.append("faces.jsonl")
        upstream_complete = body_tracking_status_value == "success" and faces_ready_state
        prereq_state, prereq_message = helpers.describe_prereq_state(
            missing,
            upstream_complete=upstream_complete,
        )
        if missing:
            run_label = selected_attempt_run_id or "legacy"
            if prereq_state == "waiting":
                st.info("⏳ **Track Fusion**: Waiting for prerequisites")
            else:
                st.warning("⚠️ **Track Fusion**: Missing prerequisites")
            st.caption(prereq_message)
            if not track_fusion_body_tracks_ready:
                st.markdown(
                    f"Run [Body Tracking](#body-tracking-card) for this run_id (`{run_label}`) to generate "
                    "`body_tracking/body_tracks.jsonl`.",
                    unsafe_allow_html=True,
                )
                st.caption(f"Missing: {helpers.link_local(body_tracks_path)}")
            if not track_fusion_faces_ready:
                st.markdown(
                    f"Run [Faces Harvest](#faces-harvest-card) for this run_id (`{run_label}`) "
                    "to generate `faces.jsonl`.",
                    unsafe_allow_html=True,
                )
                st.caption(f"Missing: {helpers.link_local(faces_path)}")
        else:
            st.info("⏳ **Track Fusion**: Ready to run")
            st.caption("Body tracks and faces are available.")
            if body_tracks_presence.remote and not body_tracks_presence.local:
                st.caption("Body tracks are available in S3; local mirror will be pulled on run.")
            if faces_presence.remote and not faces_presence.local:
                st.caption("Faces manifest is available in S3; local mirror will be pulled on run.")
    if fusion_mode_label:
        detail = f" ({fusion_mode_detail})" if fusion_mode_detail else ""
        st.caption(f"Fusion mode: {fusion_mode_label}{detail}")
    if fusion_mode_hint:
        st.caption(f"Install hint: {fusion_mode_hint}")
    finished = _format_timestamp(track_fusion_finished_at)
    if finished:
        st.caption(f"Last run: {finished}")
    if track_fusion_runtime:
        st.caption(f"Run Duration: {track_fusion_runtime}")
    elif body_fusion_status_value == "success":
        st.caption("Run Duration: n/a")
    if body_fusion_manifest_fallback:
        st.caption("ℹ️ Using manifest fallback; run-scoped marker/artifacts may be missing.")

    track_fusion_btn_label = (
        "Re-run Track Fusion" if body_fusion_status_value == "success" else "Run Track Fusion"
    )
    track_fusion_btn_disabled = (
        job_running
        or _running_body_fusion_job is not None
        or not selected_attempt_run_id
        or not track_fusion_enabled
        or not track_fusion_body_tracks_ready
        or not track_fusion_faces_ready
    )
    if st.button(
        track_fusion_btn_label,
        key=f"{ep_id}::{selected_attempt_run_id or 'legacy'}::run_track_fusion",
        use_container_width=True,
        disabled=track_fusion_btn_disabled,
    ):
        with st.spinner("Starting track fusion..."):
            try:
                helpers.api_post(
                    "/jobs/body_tracking/fusion",
                    json={"ep_id": ep_id, "run_id": selected_attempt_run_id},
                    timeout=15,
                )
            except Exception as exc:
                st.error(f"Failed to start track fusion: {exc}")
            else:
                st.success("Track fusion started.")
                st.session_state[_status_force_refresh_key(ep_id)] = True
                st.rerun()

    _render_downstream_log_expander(
        "Track Fusion",
        marker_payload=body_fusion_marker_payload,
        progress_payload=track_fusion_progress_display,
        artifacts_hint="Track fusion output found via manifest fallback."
        if body_fusion_manifest_fallback
        else None,
    )
    track_fusion_artifacts = [
        ("body_tracking/track_fusion.json", track_fusion_path, "run"),
        ("body_tracking/screentime_comparison.json", screentime_comparison_path, "run"),
    ]
    show_legacy_track_fusion = body_fusion_legacy_available and not stage_layout.artifact_available(
        track_fusion_presence
    )
    if show_legacy_track_fusion:
        track_fusion_artifacts.append(
            ("legacy/body_tracking/track_fusion.json", legacy_track_fusion_path, "legacy")
        )
    track_fusion_hint = (
        "⚠️ Legacy artifacts detected; run-scoped artifacts are missing for this attempt."
        if show_legacy_track_fusion
        else None
    )
    _render_stage_artifacts_expander(
        "Track Fusion",
        track_fusion_artifacts,
        hint=track_fusion_hint,
    )

with row2_col3:
    pdf_progress_display = pdf_status_progress
    if not selected_attempt_run_id:
        st.info("⏳ **PDF Export**: Select a run_id")
        st.caption("Select a run-scoped attempt (run_id) to export a debug report.")
    elif pdf_export_status_value == "success":
        runtime_label = pdf_export_runtime or "n/a"
        detail = f" ({pdf_export_detail})" if pdf_export_detail else ""
        st.success(f"✅ **PDF Export**: Complete (Runtime: {runtime_label}){detail}")
    elif pdf_export_status_value == "running":
        st.info("⏳ **PDF Export**: Running")
        started = _format_timestamp(pdf_export_started_at)
        if started:
            st.caption(f"Started at {started}")
        _render_downstream_progress(pdf_progress_display, running=True)
    elif pdf_export_status_value in {"error", "failed"}:
        st.error("⚠️ **PDF Export**: Failed")
        if pdf_export_detail:
            st.caption(pdf_export_detail)
    else:
        st.info("⏳ **PDF Export**: Ready to run")
        st.caption("Generate the Episode Details debug report.")
    if export_index and isinstance(export_index, dict):
        export_bytes = export_index.get("export_bytes")
        export_upload = export_index.get("export_upload") if isinstance(export_index.get("export_upload"), dict) else {}
        export_key = export_upload.get("s3_key") or export_index.get("export_s3_key")
        if export_bytes:
            st.caption(f"Export size: {export_bytes / 1024:.1f} KB")
        if export_key:
            st.caption(f"S3 key: `{export_key}`")
        if export_upload.get("attempted") and not export_upload.get("success"):
            st.caption("S3 upload failed; export is local-only.")
    finished = _format_timestamp(pdf_export_finished_at)
    if finished:
        st.caption(f"Last run: {finished}")
    if pdf_export_runtime:
        st.caption(f"Run Duration: {pdf_export_runtime}")
    elif pdf_export_status_value == "success":
        st.caption("Run Duration: n/a")

    pdf_btn_label = "Re-export PDF" if pdf_export_status_value == "success" else "Export PDF"
    pdf_btn_disabled = (
        job_running
        or not selected_attempt_run_id
        or pdf_export_status_value == "running"
    )
    if st.button(
        pdf_btn_label,
        key=f"{ep_id}::{selected_attempt_run_id or 'legacy'}::run_pdf_export",
        use_container_width=True,
        disabled=pdf_btn_disabled,
    ):
        with st.spinner("Generating PDF report..."):
            ok, msg = _trigger_pdf_export_if_needed(
                ep_id,
                selected_attempt_run_id,
                cfg,
                force=pdf_export_status_value == "success",
            )
        if not ok:
            st.error(msg)
        else:
            st.success(msg)
            st.session_state[_status_force_refresh_key(ep_id)] = True
            st.rerun()

    _render_downstream_log_expander(
        "PDF Export",
        marker_payload=export_index if isinstance(export_index, dict) else None,
        progress_payload=pdf_progress_display,
    )
    pdf_artifacts = [
        ("exports/export_index.json", _scoped_manifests_dir / "exports" / "export_index.json", "run"),
    ]
    _render_stage_artifacts_expander("PDF Export", pdf_artifacts)

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

with st.expander("Advanced: Stage Summary", expanded=False):
    summary_rows: list[dict[str, Any]] = []
    if not selected_attempt_run_id:
        st.info("Select a run-scoped attempt (run_id) to view per-stage status and artifacts.")
    else:
        stage_plan = list(setup_stage_plan)
        if episode_status_payload:
            st.caption("Source: episode_status.json (run-scoped)")
        else:
            st.caption("Source: stage markers + manifests (episode_status.json missing)")

        run_root = run_layout.run_root(ep_id, selected_attempt_run_id)
        stage_artifact_map: dict[str, list[dict[str, Any]]] = {}
        artifact_rel_paths: set[str] = set()

        def _artifact_rel(path: Path) -> str | None:
            try:
                return str(path.relative_to(run_root))
            except Exception:
                return None

        for stage_key in stage_plan:
            stage_entry = (
                (episode_status_payload or {}).get("stages", {}).get(stage_key)
                if isinstance(episode_status_payload, dict)
                else None
            )
            artifacts = stage_entry.get("artifacts") if isinstance(stage_entry, dict) else None
            if not isinstance(artifacts, list):
                artifacts = stage_artifacts(ep_id, selected_attempt_run_id, stage_key)
            stage_artifact_map[stage_key] = artifacts
            for artifact in artifacts:
                path_raw = artifact.get("path") if isinstance(artifact, dict) else None
                if not path_raw:
                    continue
                rel = _artifact_rel(Path(path_raw))
                if rel:
                    artifact_rel_paths.add(rel)

        presence_map: dict[str, dict[str, Any]] = {}
        if artifact_rel_paths:
            presence_map = _cached_run_artifact_presence(
                ep_id,
                selected_attempt_run_id,
                tuple(sorted(artifact_rel_paths)),
            )

        def _artifact_display(artifact: dict[str, Any]) -> str:
            path_raw = artifact.get("path")
            label = artifact.get("label") or (Path(path_raw).name if path_raw else "artifact")
            scope = artifact.get("scope") or ("run" if "runs" in str(path_raw or "") else "legacy")
            source = "missing"
            if path_raw:
                path_obj = Path(path_raw)
                if path_obj.exists():
                    source = "local"
                else:
                    rel = _artifact_rel(path_obj)
                    presence = presence_map.get(rel or "")
                    if isinstance(presence, dict) and presence.get("remote"):
                        source = "s3"
            return f"{label} ({scope} · {source})"

        def _fallback_stage_entry(stage_key: str) -> dict[str, Any]:
            if stage_key == "detect":
                return {
                    "status": detect_status_value,
                    "started_at": detect_phase_status.get("started_at"),
                    "ended_at": detect_phase_status.get("finished_at"),
                    "duration_s": detect_phase_status.get("runtime_sec"),
                    "error_reason": detect_phase_status.get("error") or detect_phase_status.get("error_reason"),
                }
            if stage_key == "faces":
                return {
                    "status": faces_status_value,
                    "started_at": faces_phase_status.get("started_at"),
                    "ended_at": faces_phase_status.get("finished_at"),
                    "duration_s": faces_phase_status.get("runtime_sec"),
                    "error_reason": faces_phase_status.get("error") or faces_phase_status.get("error_reason"),
                }
            if stage_key == "cluster":
                return {
                    "status": cluster_status_value,
                    "started_at": cluster_phase_status.get("started_at"),
                    "ended_at": cluster_phase_status.get("finished_at"),
                    "duration_s": cluster_phase_status.get("runtime_sec"),
                    "error_reason": cluster_phase_status.get("error") or cluster_phase_status.get("error_reason"),
                }
            if stage_key == "body_tracking":
                return {
                    "status": body_tracking_status_value,
                    "started_at": body_tracking_started_at,
                    "ended_at": body_tracking_finished_at,
                    "duration_s": _runtime_from_iso(body_tracking_started_at, body_tracking_finished_at),
                    "error_reason": body_tracking_error,
                }
            if stage_key == "track_fusion":
                return {
                    "status": body_fusion_status_value,
                    "started_at": track_fusion_started_at,
                    "ended_at": track_fusion_finished_at,
                    "duration_s": _runtime_from_iso(track_fusion_started_at, track_fusion_finished_at),
                    "error_reason": body_fusion_error,
                }
            if stage_key == "pdf":
                return {
                    "status": pdf_export_status_value,
                    "started_at": pdf_export_started_at,
                    "ended_at": pdf_export_finished_at,
                    "duration_s": _runtime_from_iso(pdf_export_started_at, pdf_export_finished_at),
                    "error_reason": pdf_export_detail if pdf_export_status_value == "error" else None,
                }
            return {"status": "unknown"}

        def _merge_stage_entry(stage_key: str, stage_entry: dict[str, Any] | None) -> dict[str, Any]:
            fallback_entry = _fallback_stage_entry(stage_key)
            if not isinstance(stage_entry, dict):
                return fallback_entry
            stage_status = str(stage_entry.get("status") or "").strip().lower()
            fallback_status = str(fallback_entry.get("status") or "").strip().lower()
            if stage_status in {"missing", "unknown", "stale"} and fallback_status not in {"missing", "unknown", "stale", ""}:
                return fallback_entry
            merged = dict(stage_entry)
            for key in ("started_at", "ended_at", "duration_s", "error_reason"):
                if not merged.get(key) and fallback_entry.get(key) is not None:
                    merged[key] = fallback_entry.get(key)
            return merged

        stages_payload = episode_status_payload.get("stages", {}) if isinstance(episode_status_payload, dict) else {}
        for stage_key in stage_plan:
            stage_entry = _merge_stage_entry(stage_key, stages_payload.get(stage_key))
            started_at = stage_entry.get("started_at")
            ended_at = stage_entry.get("ended_at")
            duration_val = stage_entry.get("duration_s")
            if duration_val is None:
                duration_val = _runtime_from_iso(started_at, ended_at)
            duration_label = _format_runtime(duration_val) or "n/a"
            artifacts = stage_artifact_map.get(stage_key) or []
            artifacts_label = ", ".join(_artifact_display(a) for a in artifacts if isinstance(a, dict)) or "n/a"
            status_value = setup_stage_statuses.get(stage_key) or stage_entry.get("status") or "unknown"
            summary_rows.append(
                {
                    "Stage": stage_layout.stage_label(stage_key),
                    "Status": status_value,
                    "Started": started_at or "—",
                    "Ended": ended_at or "—",
                    "Duration": duration_label,
                    "Artifacts": artifacts_label,
                    "Error": stage_entry.get("error_reason") or "",
                }
            )

    if summary_rows:
        st.dataframe(summary_rows, hide_index=True, use_container_width=True)
    else:
        st.caption("No stage summary available.")

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
# detect|faces|cluster|body_tracking|track_fusion|pdf|None

# Check if auto-run is active and a phase just completed
autorun_active = st.session_state.get(_autorun_key, False)
autorun_phase = st.session_state.get(_autorun_phase_key)

# Auto-run container for the button
with st.container():
    auto_col1, auto_col2 = st.columns([3, 1])
    with auto_col1:
        if autorun_active:
            current_phase_key = stage_layout.normalize_stage_key(autorun_phase)
            current_phase_label = stage_layout.stage_label(current_phase_key)
            completed_labels = [
                stage_layout.stage_label(stage)
                for stage in setup_stage_plan
                if _is_setup_stage_done(setup_stage_statuses.get(stage, ""))
            ]
            # Build status display
            status_lines = [f"✅ {label}" for label in completed_labels]
            if not status_lines:
                status_lines.append("No stages completed yet.")
            st.info("**Setup Pipeline Active**\n\n" + "\n\n".join(status_lines))
            active_run_id = st.session_state.get(_autorun_run_id_key) or selected_attempt_run_id
            if active_run_id:
                st.caption(f"Active attempt: `{active_run_id}`")

            # Progress: count completed stages from the merged setup status model.
            stage_plan = setup_stage_plan
            completed_count = len(completed_labels)
            total_count = len(stage_plan)
            completed_keys = {
                stage for stage in stage_plan if _is_setup_stage_done(setup_stage_statuses.get(stage, ""))
            }

            current_phase_pct = 0.0
            _running_jobs = helpers.get_all_running_jobs_for_episode(ep_id)
            if current_phase_key == "detect" and _running_jobs.get("detect_track"):
                current_phase_pct = _running_jobs["detect_track"].get("progress_pct", 0) / 100
            elif current_phase_key == "faces" and _running_jobs.get("faces_embed"):
                current_phase_pct = _running_jobs["faces_embed"].get("progress_pct", 0) / 100
            elif current_phase_key == "cluster" and _running_jobs.get("cluster"):
                current_phase_pct = _running_jobs["cluster"].get("progress_pct", 0) / 100
            elif current_phase_key == "body_tracking":
                pct = _progress_pct_from_payload(body_tracking_status_progress or body_tracking_progress_payload)
                if pct is None and _running_body_tracking_job:
                    pct = 0.05
                current_phase_pct = pct or 0.0
            elif current_phase_key == "track_fusion":
                pct = _progress_pct_from_payload(track_fusion_status_progress or track_fusion_progress_payload)
                if pct is None and _running_body_fusion_job:
                    pct = 0.05
                current_phase_pct = pct or 0.0
            elif current_phase_key == "pdf":
                pct = _progress_pct_from_payload(pdf_status_progress)
                if pct is None and pdf_export_status_value == "running":
                    pct = 0.05
                current_phase_pct = pct or 0.0

            if total_count:
                if current_phase_key in stage_plan and current_phase_key not in completed_keys:
                    total_progress = (completed_count + current_phase_pct) / total_count
                else:
                    total_progress = completed_count / total_count
            else:
                total_progress = 0.0

            st.progress(min(total_progress, 1.0))
            progress_label = f"Pipeline: {int(total_progress * 100)}% complete"
            count_label = f"{completed_count}/{total_count} stages complete" if total_count else "Stages: n/a"
            running_label = f"Currently running: {current_phase_label}"
            st.caption(f"{progress_label} · {count_label} · {running_label}")

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
            st.caption(
                "Run the setup pipeline in sequence: Detect/Track → Faces → Cluster → "
                "Body Tracking → Track Fusion → PDF"
            )
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
            autorun_disabled = not local_video_exists or _job_active(ep_id) or st.session_state.get(running_job_key, False)
            autorun_start_requested_key = f"{ep_id}::autorun_start_requested"
            autorun_rerun_requested_key = f"{ep_id}::autorun_rerun_requested"

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
                if db_available:
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
                st.session_state.pop(f"{ep_id}::autorun_cluster_promoted_mtime", None)

                st.session_state[_autorun_key] = True
                st.session_state[_autorun_phase_key] = "detect"
                st.session_state[f"{ep_id}::autorun_completed_stages"] = []  # Clear completed stages log
                st.session_state[f"{ep_id}::autorun_started_at"] = time.time()  # Track when auto-run started
                st.session_state["episode_detail_detect_autorun_flag"] = True  # Trigger detect job
                st.toast("Starting setup pipeline (new attempt)...")
                st.rerun()
            elif st.session_state.pop(autorun_rerun_requested_key, False):
                if autorun_disabled:
                    st.error("Auto-run is disabled (missing video or another job is running).")
                    st.stop()
                if not selected_attempt_run_id:
                    st.error("Select a run-scoped attempt before re-running the setup pipeline.")
                    st.stop()

                run_id = selected_attempt_run_id
                run_root = run_layout.run_root(ep_id, run_id)

                def _safe_mtime(path: Path) -> float:
                    try:
                        return path.stat().st_mtime
                    except (FileNotFoundError, OSError):
                        return 0.0

                st.session_state[f"{ep_id}::autorun_detect_baseline_mtime"] = max(
                    _safe_mtime(run_root / "tracks.jsonl"),
                    _safe_mtime(run_root / "detect_track.json"),
                )
                st.session_state[f"{ep_id}::autorun_faces_baseline_mtime"] = max(
                    _safe_mtime(run_root / "faces.jsonl"),
                    _safe_mtime(run_root / "faces_embed.json"),
                )
                st.session_state[f"{ep_id}::autorun_cluster_baseline_mtime"] = max(
                    _safe_mtime(run_root / "identities.json"),
                    _safe_mtime(run_root / "cluster.json"),
                )

                st.session_state.pop(_status_mtimes_key(ep_id), None)
                st.session_state.pop(_status_cache_key(ep_id), None)
                st.session_state[_status_force_refresh_key(ep_id)] = True
                helpers.invalidate_running_jobs_cache(ep_id)
                _cached_local_jobs.clear()

                st.session_state.pop(f"{ep_id}::autorun_detect_promoted_mtime", None)
                st.session_state.pop(f"{ep_id}::autorun_faces_promoted_mtime", None)
                st.session_state.pop(f"{ep_id}::autorun_cluster_promoted_mtime", None)

                st.session_state[_autorun_run_id_key] = run_id
                st.session_state[_active_run_id_pending_key] = run_id
                st.session_state[_autorun_key] = True
                st.session_state[_autorun_phase_key] = "detect"
                st.session_state[f"{ep_id}::autorun_completed_stages"] = []
                st.session_state[f"{ep_id}::autorun_started_at"] = time.time()
                st.session_state["episode_detail_detect_autorun_flag"] = True
                st.toast("Starting setup pipeline (selected attempt)...")
                st.rerun()

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

        # Only treat explicit success as promotable; "stale" must re-run before clustering.
        faces_status_ok = str(faces_api_status or "").strip().lower() == "success"

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

    # Fallback 3: cluster → body tracking
    elif autorun_phase == "cluster":
        cluster_job_running = helpers.get_running_job_for_episode(ep_id, "cluster") is not None
        cluster_baseline = st.session_state.get(f"{ep_id}::autorun_cluster_baseline_mtime", 0.0)
        cluster_marker_path = _scoped_markers_dir / "cluster.json"
        legacy_cluster_marker_path = _runs_root / "cluster.json"
        marker_payload, marker_mtime, _ = _load_marker_payload_with_fallback(
            cluster_marker_path,
            legacy_cluster_marker_path,
            run_id=selected_attempt_run_id,
        )
        marker_mtime = float(marker_mtime or 0.0)
        marker_status = str((marker_payload or {}).get("status") or "").strip().lower()
        marker_error = (marker_payload or {}).get("error")
        marker_run_id = (marker_payload or {}).get("run_id")
        marker_run_matches = isinstance(marker_run_id, str) and marker_run_id.strip() == (selected_attempt_run_id or "")
        cluster_from_current_run = marker_mtime > float(cluster_baseline or 0.0)

        if not cluster_job_running and marker_status == "success" and not marker_error and marker_run_matches and cluster_from_current_run:
            _cluster_promoted_key = f"{ep_id}::autorun_cluster_promoted_mtime"
            last_promoted = st.session_state.get(_cluster_promoted_key)
            if last_promoted is None or marker_mtime > float(last_promoted or 0.0):
                st.session_state[_cluster_promoted_key] = marker_mtime
                completed = st.session_state.get(f"{ep_id}::autorun_completed_stages", [])
                if not any("Cluster" in s or "Clustering" in s for s in completed):
                    completed.append("Cluster")
                    st.session_state[f"{ep_id}::autorun_completed_stages"] = completed
                if not body_tracking_enabled:
                    st.session_state[_autorun_key] = False
                    st.session_state[_autorun_phase_key] = None
                    st.session_state[f"{ep_id}::autorun_error"] = "body_tracking_disabled"
                    st.error("❌ Auto-run stopped (Body Tracking): disabled_by_config")
                else:
                    st.session_state[_autorun_phase_key] = "body_tracking"
                    st.toast("✅ Cluster complete - advancing to Body Tracking...")
                _fallback_triggered = True

    if _fallback_triggered:
        st.rerun()

col_detect, col_faces, col_cluster = st.columns(3)

# Check for running jobs for each phase
running_detect_job = helpers.get_running_job_for_episode(ep_id, "detect_track")
running_faces_job = helpers.get_running_job_for_episode(ep_id, "faces_embed")
running_cluster_job = helpers.get_running_job_for_episode(ep_id, "cluster")
running_audio_job = helpers.get_running_job_for_episode(ep_id, "audio_pipeline")
_subprocess_jobs_payload = _cached_episode_jobs(ep_id)
running_body_tracking_job = _find_running_subprocess_job(
    _subprocess_jobs_payload, job_type="body_tracking", run_id=selected_attempt_run_id
)
running_body_fusion_job = _find_running_subprocess_job(
    _subprocess_jobs_payload, job_type="body_tracking_fusion", run_id=selected_attempt_run_id
)

# Synchronize session state with API-based job status
# This is the single source of truth - clears stale session flags if API says no job running
job_running, stale_job_warning = _sync_job_state_with_api(
    ep_id,
    running_job_key,
    running_detect_job,
    running_faces_job,
    running_cluster_job,
    running_audio_job,
    running_body_tracking_job,
    running_body_fusion_job,
)
if stale_job_warning:
    st.warning(f"⚠️ {stale_job_warning}")

# =============================================================================
# AUTO-RUN SETUP ORCHESTRATION (body tracking -> fusion -> PDF)
# =============================================================================
def _autorun_stop(stage_label: str, message: str) -> None:
    LOGGER.error("[AUTORUN] Stopping at %s: %s", stage_label, message)
    st.session_state[_autorun_key] = False
    st.session_state[_autorun_phase_key] = None
    st.session_state[f"{ep_id}::autorun_error"] = message
    st.error(f"❌ Auto-run stopped ({stage_label}): {message}")


def _autorun_append_completed(label: str) -> None:
    completed = st.session_state.get(f"{ep_id}::autorun_completed_stages", [])
    if not isinstance(completed, list):
        completed = []
    if label not in completed:
        completed.append(label)
        st.session_state[f"{ep_id}::autorun_completed_stages"] = completed


def _autorun_drive_downstream(run_id: str) -> None:
    phase_now = st.session_state.get(_autorun_phase_key)
    if phase_now not in {"body_tracking", "track_fusion", "pdf"}:
        return

    if phase_now == "body_tracking":
        if not body_tracking_enabled:
            _autorun_stop("Body Tracking", "disabled_by_config")
            return
        if body_tracking_status_value == "stale" and body_tracking_legacy_available:
            _autorun_append_completed("Body Tracking (legacy)")
            st.session_state[_autorun_phase_key] = "track_fusion"
            st.toast("⚠️ Body Tracking complete (legacy artifacts).")
            st.rerun()
        if stage_layout.downstream_stage_allows_advance(body_tracking_status_value, body_tracking_error):
            _autorun_append_completed("Body Tracking")
            st.session_state[_autorun_phase_key] = "track_fusion"
            st.rerun()
        if body_tracking_status_value in {"error", "failed"}:
            _autorun_stop("Body Tracking", body_tracking_error or body_tracking_status_value)
            return
        if body_tracking_status_value == "running":
            return
        try:
            helpers.api_post(
                "/jobs/body_tracking/run",
                json={"ep_id": ep_id, "run_id": run_id},
                timeout=15,
            )
        except Exception as exc:
            _autorun_stop("Body Tracking", f"enqueue_failed: {exc}")
            return
        st.session_state[_status_force_refresh_key(ep_id)] = True
        st.toast("🏃 Starting Body Tracking...")
        st.rerun()

    if phase_now == "track_fusion":
        if not track_fusion_enabled:
            _autorun_stop("Track Fusion", "disabled_by_config")
            return
        if body_fusion_status_value == "stale" and body_fusion_legacy_available:
            stage_label = "Track Fusion (legacy)"
            if fusion_mode_label:
                stage_label = f"{stage_label} ({fusion_mode_label})"
            _autorun_append_completed(stage_label)
            st.session_state[_autorun_phase_key] = "pdf"
            st.toast("⚠️ Track Fusion complete (legacy artifacts).")
            st.rerun()
        if stage_layout.downstream_stage_allows_advance(body_fusion_status_value, body_fusion_error):
            stage_label = "Track Fusion"
            if fusion_mode_label:
                stage_label = f"{stage_label} ({fusion_mode_label})"
            _autorun_append_completed(stage_label)
            st.session_state[_autorun_phase_key] = "pdf"
            st.rerun()
        prereq_ok, missing = stage_layout.track_fusion_prereq_state(faces_presence, body_tracks_presence)
        if not prereq_ok:
            _autorun_stop("Track Fusion", f"missing_prereqs: {', '.join(missing)}")
            return
        if not (faces_presence.local and body_tracks_presence.local):
            hydrate_key = f"{ep_id}::{run_id}::track_fusion_hydrate_attempted"
            if not st.session_state.get(hydrate_key):
                st.session_state[hydrate_key] = True
                with st.spinner("Mirroring run artifacts from S3…"):
                    ok, err = _ensure_run_artifacts_local(
                        ep_id,
                        run_id,
                        {
                            "faces.jsonl": faces_presence,
                            "body_tracking/body_tracks.jsonl": body_tracks_presence,
                        },
                    )
                if not ok:
                    _autorun_stop("Track Fusion", f"hydrate_failed: {err}")
                    return
                # Clear cached presence so the rerun re-checks freshly hydrated artifacts.
                try:
                    _cached_run_artifact_presence.clear()
                except Exception as exc:
                    LOGGER.debug("[RUN_ARTIFACT] Cache clear failed after hydrate: %s", exc)
                st.session_state[_status_force_refresh_key(ep_id)] = True
                st.rerun()
            else:
                _autorun_stop("Track Fusion", "local_artifacts_missing_after_hydrate")
                return
        if body_fusion_status_value in {"error", "failed"}:
            _autorun_stop("Track Fusion", body_fusion_error or body_fusion_status_value)
            return
        if body_fusion_status_value == "running":
            return
        try:
            helpers.api_post(
                "/jobs/body_tracking/fusion",
                json={"ep_id": ep_id, "run_id": run_id},
                timeout=15,
            )
        except Exception as exc:
            _autorun_stop("Track Fusion", f"enqueue_failed: {exc}")
            return
        st.session_state[_status_force_refresh_key(ep_id)] = True
        st.toast("🔗 Starting Track Fusion...")
        st.rerun()

    if phase_now == "pdf":
        if pdf_export_status_value == "success":
            _autorun_append_completed("PDF Export")
            # Finalize auto-run only when all enabled stages are complete for this run_id.
            st.session_state[_autorun_key] = False
            st.session_state[_autorun_phase_key] = None
            st.session_state[_status_force_refresh_key(ep_id)] = True
            completion_bits = [
                "Detect/Track",
                "Faces",
                "Cluster",
                "Body Tracking",
                "Track Fusion",
                "PDF",
            ]
            completion_label = " · ".join(completion_bits)
            st.success(f"🎉 **Setup Pipeline Complete!** ({completion_label})")
            st.rerun()

        with st.spinner("Generating PDF report..."):
            ok, msg = _trigger_pdf_export_if_needed(ep_id, run_id, cfg)
        if not ok:
            _autorun_stop("PDF Export", msg)
            return
        st.success(msg)
        st.session_state[_status_force_refresh_key(ep_id)] = True
        st.rerun()


_autorun_active_now = bool(st.session_state.get(_autorun_key))
_autorun_phase_now = st.session_state.get(_autorun_phase_key)
_autorun_run_id_now = st.session_state.get(_autorun_run_id_key) or selected_attempt_run_id
if _autorun_active_now and isinstance(_autorun_run_id_now, str) and _autorun_run_id_now.strip():
    _autorun_run_id_now = _autorun_run_id_now.strip()
    if selected_attempt_run_id and _autorun_run_id_now != selected_attempt_run_id:
        st.session_state[_active_run_id_pending_key] = _autorun_run_id_now
        st.session_state[_status_force_refresh_key(ep_id)] = True
        st.rerun()
    if _autorun_phase_now in {"body_tracking", "track_fusion", "pdf"} and selected_attempt_run_id:
        _autorun_drive_downstream(selected_attempt_run_id)

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

        # Best-effort fallback: only synthesize a "completed" summary when we can also
        # validate a successful run marker for the selected run_id (prevents stale promotion).
        if not summary and not error_message:
            LOGGER.warning("[AUTORUN] No summary received, attempting to synthesize from run marker + manifests")
            det_path = detections_path
            trk_path = tracks_path
            marker_path = _scoped_markers_dir / "detect_track.json"
            marker_payload: dict[str, Any] | None = None
            if marker_path.exists():
                try:
                    marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    marker_payload = None
            from py_screenalytics.run_scoping import should_synthesize_detect_track_success

            if should_synthesize_detect_track_success(
                run_id=selected_attempt_run_id,
                marker_payload=marker_payload,
                detections_exists=det_path.exists(),
                tracks_exists=trk_path.exists(),
            ):
                LOGGER.info("[AUTORUN] Marker+manifests validated; synthesizing summary")
                summary = {"status": "completed", "fallback": True}

        # IMPORTANT: Set auto-run progression FIRST, before any early returns.
        # Never treat a truthy dict as success; require a success status and no error.
        if _summary_status_ok(summary) and not error_message:
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
        # Capture current detect RTF/FPS so the Performance panel can show before/after deltas after reruns.
        if selected_attempt_run_id:
            perf_prefix = f"{ep_id}::{selected_attempt_run_id}"
            marker_payload: dict[str, Any] | None = None
            marker_path = _scoped_markers_dir / "detect_track.json"
            if marker_path.exists():
                try:
                    marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    marker_payload = None
            if isinstance(marker_payload, dict) and str(marker_payload.get("status") or "").strip().lower() == "success":
                rtf = helpers.coerce_float(marker_payload.get("rtf"))
                eff_fps = helpers.coerce_float(marker_payload.get("effective_fps_processing"))
                if rtf is not None:
                    st.session_state[f"{perf_prefix}::baseline_detect_rtf"] = float(rtf)
                else:
                    st.session_state.pop(f"{perf_prefix}::baseline_detect_rtf", None)
                if eff_fps is not None:
                    st.session_state[f"{perf_prefix}::baseline_detect_eff_fps"] = float(eff_fps)
                else:
                    st.session_state.pop(f"{perf_prefix}::baseline_detect_eff_fps", None)
            else:
                st.session_state.pop(f"{perf_prefix}::baseline_detect_rtf", None)
                st.session_state.pop(f"{perf_prefix}::baseline_detect_eff_fps", None)

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
        with st.expander("🔍 Faces Phase Debug", expanded=False):
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
            LOGGER.info("[AUTORUN] Faces job returned: summary=%s, status=%s, error=%s", bool(summary), getattr(summary, "get", lambda *_: None)("status") if isinstance(summary, dict) else None, error_message)
            # Best-effort fallback: only synthesize a "completed" summary when we can also
            # validate a successful run marker for the selected run_id (prevents stale promotion).
            if not summary and not error_message:
                LOGGER.warning("[AUTORUN] Faces summary missing, attempting to synthesize from run marker + manifests")
                marker_path = _scoped_markers_dir / "faces_embed.json"
                marker_payload: dict[str, Any] | None = None
                if marker_path.exists():
                    try:
                        marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        marker_payload = None
                marker_status = str((marker_payload or {}).get("status") or "").strip().lower()
                marker_error = (marker_payload or {}).get("error")
                marker_run_id = (marker_payload or {}).get("run_id")
                marker_run_matches = True
                if selected_attempt_run_id:
                    marker_run_matches = isinstance(marker_run_id, str) and marker_run_id.strip() == selected_attempt_run_id
                if faces_path.exists() and marker_status == "success" and not marker_error and marker_run_matches:
                    LOGGER.info("[AUTORUN] Marker+faces.jsonl validated; synthesizing summary")
                    summary = {"status": "completed", "fallback": True}

            if _summary_status_ok(summary) and not error_message:
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
        LOGGER.info("[AUTORUN] Detected cluster_just_completed flag")
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

            if not body_tracking_enabled:
                _autorun_stop("Body Tracking", "disabled_by_config")
            else:
                next_phase = "body_tracking"
                st.session_state[_autorun_phase_key] = next_phase
                st.session_state[_status_force_refresh_key(ep_id)] = True
                LOGGER.info("[AUTORUN] Cluster complete -> advancing to %s", next_phase)
                st.toast("✅ Cluster complete - advancing to Body Tracking...")
                st.rerun()

    st.markdown("### Cluster Identities")
    st.caption(_format_phase_status("Cluster Identities", cluster_phase_status, "identities"))

    # If we just ran a recluster sweep, surface deltas against the captured baseline.
    _recluster_prefix = f"{ep_id}::{selected_attempt_run_id}::recluster"
    recluster_baseline = st.session_state.get(f"{_recluster_prefix}::baseline")
    if isinstance(recluster_baseline, dict) and cluster_status_value == "success":
        before_frac = helpers.coerce_float(recluster_baseline.get("singleton_fraction_after"))
        after_frac = helpers.coerce_float(cluster_phase_status.get("singleton_fraction_after"))
        before_thresh = helpers.coerce_float(recluster_baseline.get("cluster_thresh"))
        after_thresh = helpers.coerce_float(cluster_phase_status.get("cluster_thresh"))
        if before_frac is not None and after_frac is not None:
            delta = after_frac - before_frac
            st.caption(
                f"Recluster delta: singleton_fraction_after {before_frac:.1%} → {after_frac:.1%} ({delta:+.1%})"
            )
        if before_thresh is not None and after_thresh is not None and before_thresh != after_thresh:
            st.caption(f"Recluster delta: cluster_thresh {before_thresh:.2f} → {after_thresh:.2f}")

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
            autorun_cluster_active = bool(
                st.session_state.get(_autorun_key) and st.session_state.get(_autorun_phase_key) == "cluster"
            )

            # Auto-run: advance to setup stages after clustering finishes.
            if autorun_cluster_active:
                if not body_tracking_enabled:
                    _autorun_stop("Body Tracking", "disabled_by_config")
                else:
                    st.session_state[_autorun_phase_key] = "body_tracking"
                    st.toast("✅ Cluster complete - advancing to Body Tracking...")
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
        with st.expander("🔍 Cluster Phase Debug", expanded=False):
            st.markdown(f"**Trigger received**: `autorun_cluster_trigger={autorun_cluster_trigger}`")
            st.markdown(f"**cluster_disabled**: `{cluster_disabled}`")
            if cluster_disabled:
                st.markdown("**Disabled because:**")
                reasons = []
                if not effective_faces_ready:
                    reasons.append(
                        f"- faces not ready (faces_ready={faces_ready}, faces_just_completed={faces_just_completed})"
                    )
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
                    reasons.append(
                        f"- faces stale (faces_status_value={faces_status_value}, faces_just_completed={faces_just_completed})"
                    )
                if cluster_status_value == "running":
                    reasons.append(f"- cluster status is 'running' ({cluster_status_value=})")
                if running_cluster_job is not None:
                    reasons.append(f"- cluster job already running ({running_cluster_job=})")
                st.code("\n".join(reasons) if reasons else "No specific reason found", language="text")

    # Quick recluster control (cluster_thresh sweep entrypoint).
    quick_recluster_disabled = cluster_disabled or (not bool(selected_attempt_run_id))
    quick_thresh_options = [0.45, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70]
    try:
        current_thresh = float(cluster_thresh_value)
    except (TypeError, ValueError):
        current_thresh = None
    if current_thresh is not None:
        quick_thresh_options.append(round(current_thresh, 2))
    quick_thresh_options = sorted({round(float(v), 2) for v in quick_thresh_options})
    quick_thresh_default = round(current_thresh, 2) if current_thresh is not None else quick_thresh_options[0]
    quick_thresh = st.selectbox(
        "Quick recluster: cluster_thresh override",
        quick_thresh_options,
        index=quick_thresh_options.index(quick_thresh_default) if quick_thresh_default in quick_thresh_options else 0,
        key=f"{_recluster_prefix}::cluster_thresh_override",
        disabled=quick_recluster_disabled,
    )
    quick_recluster_clicked = st.button(
        f"Recluster with cluster_thresh={float(quick_thresh):.2f}",
        use_container_width=True,
        disabled=quick_recluster_disabled,
        type="secondary",
        help="Select a run-scoped attempt (run_id) to use quick recluster." if not selected_attempt_run_id else None,
        key=f"{_recluster_prefix}::quick_recluster",
    )
    run_cluster_clicked = st.button("Run Cluster", use_container_width=True, disabled=cluster_disabled)

    if quick_recluster_clicked:
        # Capture baseline metrics for delta display.
        st.session_state[f"{_recluster_prefix}::baseline"] = {
            "cluster_thresh": helpers.coerce_float(cluster_phase_status.get("cluster_thresh")),
            "singleton_fraction_after": helpers.coerce_float(cluster_phase_status.get("singleton_fraction_after")),
            "captured_at": time.time(),
        }
        # Apply override to session-scoped cluster settings so the payload uses it.
        st.session_state[_get_pipeline_settings_key(ep_id, "cluster", "cluster_thresh")] = float(quick_thresh)
        cluster_thresh_value = float(quick_thresh)

    should_run_cluster = bool(quick_recluster_clicked or run_cluster_clicked)

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
                autorun_cluster_active = bool(
                    st.session_state.get(_autorun_key) and st.session_state.get(_autorun_phase_key) == "cluster"
                )
                if autorun_cluster_active:
                    # Log completed stage with counts from summary
                    normalized = helpers.normalize_summary(ep_id, summary)
                    identities_count = normalized.get("identities")
                    faces_count = normalized.get("faces")
                    completed = st.session_state.get(f"{ep_id}::autorun_completed_stages", [])
                    id_count = identities_count if isinstance(identities_count, int) else "?"
                    fc_count = faces_count if isinstance(faces_count, int) else "?"
                    completed.append(f"Clustering ({id_count} identities, {fc_count} faces)")
                    st.session_state[f"{ep_id}::autorun_completed_stages"] = completed
                    if not body_tracking_enabled:
                        _autorun_stop("Body Tracking", "disabled_by_config")
                    else:
                        st.session_state[_autorun_phase_key] = "body_tracking"
                        st.toast("✅ Cluster complete - advancing to Body Tracking...")

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
                # Setup stages run after cluster; do not mark Auto-Run complete here.

                st.session_state["episode_detail_flash"] = flash_msg

                autorun_cluster_active = bool(
                    st.session_state.get(_autorun_key) and st.session_state.get(_autorun_phase_key) == "cluster"
                )
                if autorun_cluster_active:
                    st.toast("✅ Cluster complete (Auto-Run). Continuing setup stages…")
                st.rerun()


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
                resp = requests.get(url, params={"include_screentime": "false"}, timeout=300)
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

def _render_artifact_entry(
    label: str,
    local_path: Path,
    key_suffix: str,
    s3_keys: list[tuple[str, str]] | None = None,
) -> None:
    st.write(f"{label} → {helpers.link_local(local_path)}")
    if not s3_keys:
        return
    for scope_label, s3_key in s3_keys:
        st.caption(scope_label)
        uri_col, button_col = st.columns([4, 1])
        uri_col.code(helpers.s3_uri(s3_key, bucket_name))
        safe_label = (
            scope_label.replace("⚠️", "legacy")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace(" ", "_")
        )
        if button_col.button("Presign", key=f"presign_{key_suffix}_{safe_label}"):
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


with st.expander("Artifacts", expanded=False):
    manifests_prefix = (prefixes or {}).get("manifests") if prefixes else None
    video_keys: list[tuple[str, str]] = []
    if details["s3"]["v2_key"]:
        video_keys.append(("episode v2", details["s3"]["v2_key"]))
    elif details["s3"]["v1_key"]:
        video_keys.append(("episode v1 (legacy)", details["s3"]["v1_key"]))

    _render_artifact_entry(
        "Video",
        get_path(ep_id, "video"),
        "video",
        video_keys or None,
    )

    run_presence_map: dict[str, dict[str, Any]] = {}
    run_rel_paths = ("detections.jsonl", "tracks.jsonl", "faces.jsonl", "identities.json")
    if selected_attempt_run_id:
        run_presence_map = _cached_run_artifact_presence(
            ep_id,
            selected_attempt_run_id,
            run_rel_paths,
        )

    def _run_s3_entries(rel_path: str, *, local_exists: bool) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        if selected_attempt_run_id:
            presence = run_presence_map.get(rel_path) or {}
            canonical_key = presence.get("canonical_key")
            legacy_key = presence.get("legacy_key")
            canonical_remote = bool(presence.get("canonical_remote"))
            legacy_remote = bool(presence.get("legacy_remote"))
            status = "local" if local_exists else ("present" if canonical_remote else "missing")
            if canonical_key:
                entries.append((f"run ({status})", canonical_key))
            if legacy_key and (legacy_remote or (not local_exists and not canonical_remote)):
                legacy_status = "present" if legacy_remote else "missing"
                prefix = "⚠️ legacy" if legacy_remote and not canonical_remote else "legacy"
                entries.append((f"{prefix} ({legacy_status})", legacy_key))
            return entries

        if manifests_prefix:
            entries.append(("legacy (episode-level)", f"{manifests_prefix}{rel_path}"))
        return entries

    _render_artifact_entry(
        "Detections",
        detections_path,
        "detections",
        _run_s3_entries("detections.jsonl", local_exists=detections_path.exists()) or None,
    )
    _render_artifact_entry(
        "Tracks",
        tracks_path,
        "tracks",
        _run_s3_entries("tracks.jsonl", local_exists=tracks_path.exists()) or None,
    )
    _render_artifact_entry(
        "Faces",
        faces_path,
        "faces",
        _run_s3_entries("faces.jsonl", local_exists=faces_path.exists()) or None,
    )
    _render_artifact_entry(
        "Identities",
        identities_path,
        "identities",
        _run_s3_entries("identities.json", local_exists=identities_path.exists()) or None,
    )
    
    out_of_scope_artifacts: list[tuple[str, Path]] = []
    if selected_attempt_run_id:
        run_analytics_dir = manifests_dir / "analytics"
        for filename in ("screentime.json", "screentime.csv"):
            candidate = run_analytics_dir / filename
            if candidate.exists():
                out_of_scope_artifacts.append((f"run analytics/{filename}", candidate))
    legacy_analytics_dir = helpers.DATA_ROOT / "analytics" / ep_id
    for filename in ("screentime.json", "screentime.csv"):
        candidate = legacy_analytics_dir / filename
        if candidate.exists():
            out_of_scope_artifacts.append((f"legacy analytics/{filename}", candidate))
    if out_of_scope_artifacts:
        st.caption("Out-of-scope artifacts present (not used for setup completion):")
        for label, path in out_of_scope_artifacts:
            st.caption(f"{label} → {helpers.link_local(path)}")
    
    
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


with st.expander("Debug: Raw JSON artifacts", expanded=False):
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
    }
    group_labels = list(artifact_groups.keys())
    selected_group = st.selectbox(
        "Artifact group",
        group_labels,
        index=0,
        key="debug_artifact_group",
    )
    paths = artifact_groups.get(selected_group, [])
    existing = [p for p in paths if p.exists()]
    if not existing:
        st.caption("No artifacts found for this group.")
    else:
        labels = [p.name for p in existing]
        selected = st.selectbox(
            "Choose artifact",
            labels,
            key=f"{selected_group}::artifact_selector",
        )
        chosen_path = next((p for p in existing if p.name == selected), None)
        if not chosen_path:
            st.caption("Select a file to view its contents.")
        else:
            st.caption(f"Path: {helpers.link_local(chosen_path)}")
            content, err = _read_json_artifact(chosen_path)
            if err:
                st.error(err)
            else:
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
