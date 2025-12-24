#!/usr/bin/env python
"""Dev-only CLI to run detection → tracking for a single episode."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import platform
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Protocol, Set, Tuple
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import queue
import threading

import logging

# Add project root to path for imports BEFORE applying CPU limits
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Apply global CPU limits BEFORE importing any ML libraries
# Uses centralized configuration from apps.common.cpu_limits (default: 3 threads = ~300% CPU)
# Override with env var: SCREENALYTICS_MAX_CPU_THREADS=N
from apps.common.cpu_limits import apply_global_cpu_limits
apply_global_cpu_limits()

import cv2
import numpy as np

from apps.api.services.storage import (
    EpisodeContext,
    StorageService,
    artifact_prefixes,
    episode_context_from_id,
)
from py_screenalytics.artifacts import ensure_dirs, get_path
from py_screenalytics import run_layout
from py_screenalytics.episode_status import (
    BlockedReason,
    blocked_update_needed,
    collect_git_state,
    normalize_stage_key,
    stage_artifacts,
    stage_update_from_marker,
    update_episode_status,
    write_stage_blocked,
    write_stage_failed,
    write_stage_finished,
    write_stage_started,
)
from py_screenalytics.run_gates import GateReason, GateResult, check_prereqs
from py_screenalytics.run_manifests import StageBlockedInfo, StageErrorInfo, write_stage_manifest
from py_screenalytics.run_logs import append_log
from tools._img_utils import safe_crop, safe_imwrite, to_u8_bgr
from tools.debug_thumbs import (
    init_debug_logger,
    debug_thumbs_enabled,
    NullLogger,
    JsonlLogger,
)


def _load_tracking_config_yaml() -> dict[str, Any]:
    """Load tracking configuration from YAML file if available."""
    config_path = REPO_ROOT / "config" / "pipeline" / "tracking.yaml"
    if not config_path.exists():
        LOGGER.debug("Tracking config YAML not found at %s, using defaults", config_path)
        return {}

    try:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if config:
            LOGGER.info("Loaded tracking config from %s", config_path)
            return config
    except Exception as exc:
        LOGGER.warning("Failed to load tracking config YAML: %s", exc)

    return {}


def _load_detection_config_yaml() -> dict[str, Any]:
    """Load detection configuration from YAML file if available."""
    config_path = REPO_ROOT / "config" / "pipeline" / "detection.yaml"
    if not config_path.exists():
        LOGGER.debug("Detection config YAML not found at %s, using defaults", config_path)
        return {}

    try:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if config:
            LOGGER.info("Loaded detection config from %s", config_path)
            return config
    except Exception as exc:
        LOGGER.warning("Failed to load detection config YAML: %s", exc)

    return {}


def _load_embedding_config() -> dict[str, Any]:
    """Load embedding configuration from YAML file if available."""
    config_path = REPO_ROOT / "config" / "pipeline" / "embedding.yaml"
    if not config_path.exists():
        return {"embedding": {"backend": "pytorch"}}

    try:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if config:
            return config
    except Exception as exc:
        LOGGER.warning("Failed to load embedding config YAML: %s", exc)

    return {"embedding": {"backend": "pytorch"}}


def _load_alignment_config() -> dict[str, Any]:
    """Load face alignment configuration from YAML file if available."""
    config_path = REPO_ROOT / "config" / "pipeline" / "face_alignment.yaml"
    if not config_path.exists():
        return {}

    try:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if config:
            return config
    except Exception as exc:
        LOGGER.warning("Failed to load alignment config YAML: %s", exc)

    return {}


def _load_body_tracking_config() -> dict[str, Any]:
    """Load body tracking configuration from YAML file if available.

    Environment variable overrides:
        AUTO_RUN_BODY_TRACKING: If set to "0" or "false", disables body tracking
            regardless of YAML config. If set to "1" or "true", enables body tracking.
            If unset, uses the YAML config value (default: enabled).
    """
    config_path = REPO_ROOT / "config" / "pipeline" / "body_detection.yaml"
    config: dict[str, Any] = {}

    if config_path.exists():
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
        except Exception as exc:
            LOGGER.warning("Failed to load body tracking config YAML: %s", exc)

    # Environment variable override for body tracking enabled state
    env_override = os.environ.get("AUTO_RUN_BODY_TRACKING", "").strip().lower()
    if env_override in ("0", "false", "no", "off"):
        if "body_tracking" not in config:
            config["body_tracking"] = {}
        config["body_tracking"]["enabled"] = False
        LOGGER.debug("[body_tracking] Disabled via AUTO_RUN_BODY_TRACKING env var")
    elif env_override in ("1", "true", "yes", "on"):
        if "body_tracking" not in config:
            config["body_tracking"] = {}
        config["body_tracking"]["enabled"] = True
        LOGGER.debug("[body_tracking] Enabled via AUTO_RUN_BODY_TRACKING env var")

    return config


PIPELINE_VERSION = os.environ.get("SCREENALYTICS_PIPELINE_VERSION", "2025-11-11")
APP_VERSION = os.environ.get("SCREENALYTICS_APP_VERSION", PIPELINE_VERSION)
TRACKER_CONFIG = os.environ.get("SCREENALYTICS_TRACKER_CONFIG", "bytetrack.yaml")
TRACKER_NAME = Path(TRACKER_CONFIG).stem if TRACKER_CONFIG else "bytetrack"
PROGRESS_FRAME_STEP = int(os.environ.get("SCREENALYTICS_PROGRESS_FRAME_STEP", 10))  # Smoother progress (was 25)
PROGRESS_TIME_INTERVAL = float(os.environ.get("SCREENALYTICS_PROGRESS_TIME_INTERVAL", 2.0))  # Emit at least every N seconds
STAGE_HEARTBEAT_INTERVAL = float(
    os.environ.get("SCREENALYTICS_STAGE_HEARTBEAT_INTERVAL", 5.0)
)  # Episode_status heartbeat cadence
TRACKING_DIAG_INTERVAL = max(int(os.environ.get("SCREENALYTICS_TRACK_DIAG_INTERVAL", "100")), 1)
LOGGER = logging.getLogger("episode_run")
DATA_ROOT = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
DETECTOR_CHOICES = ("retinaface",)
DEFAULT_DETECTOR = DETECTOR_CHOICES[0]
TRACKER_CHOICES = ("bytetrack", "strongsort")
DEFAULT_TRACKER = TRACKER_CHOICES[0]
ARC_FACE_MODEL_NAME = os.environ.get("ARCFACE_MODEL", "arcface_r100_v1")
RETINAFACE_MODEL_NAME = os.environ.get("RETINAFACE_MODEL", "retinaface_r50_v1")
FACE_CLASS_LABEL = "face"
# Minimum face bounding box area in pixels
# Lower values detect smaller faces but may increase false positives
# Original: 20.0, Now: 10.0 to capture smaller background faces
MIN_FACE_AREA = 10.0
FACE_RATIO_BOUNDS = (0.5, 2.0)
# Detection confidence threshold - lower values catch more faces but may increase false positives
# Original: 0.65, Now: 0.50 to capture smaller/distant faces
RETINAFACE_SCORE_THRESHOLD = 0.50
RETINAFACE_NMS = 0.45

RUN_MARKERS_SUBDIR = "runs"

# Local mode instrumentation - enabled via env var for verbose phase-level logging
LOCAL_MODE_INSTRUMENTATION = os.environ.get("LOCAL_MODE_INSTRUMENTATION", "").lower() in ("1", "true", "yes")

_APPLE_SILICON = sys.platform == "darwin" and platform.machine().lower().startswith(("arm", "aarch64"))
APPLE_SILICON_HOST = _APPLE_SILICON

# Performance profile defaults used when --profile is provided and a value is missing
PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "low_power": {
        "frame_stride": 8,
        "detection_fps_limit": 8.0,
        "coreml_input_size": 384,
        "save_frames": False,
        "save_crops": False,
    },
    "balanced": {
        "frame_stride": 5,
        "detection_fps_limit": 24.0,
        "coreml_input_size": 480,
        "save_frames": False,
        "save_crops": False,
    },
    "high_accuracy": {
        "frame_stride": 1,
        "detection_fps_limit": 30.0,
        "coreml_input_size": 640,
        "save_frames": True,
        "save_crops": True,
    },
}
PROFILE_CHOICES = ("fast_cpu", "low_power", "balanced", "high_accuracy")


class PhaseTracker:
    """Track timing and frame counts for each pipeline phase (for local mode diagnostics)."""

    def __init__(self) -> None:
        self._phases: Dict[str, Dict[str, Any]] = {}
        self._current_phase: str | None = None
        self._current_start: float | None = None

    def start_phase(self, phase: str) -> None:
        """Begin timing a new phase."""
        self._current_phase = phase
        self._current_start = time.time()
        if phase not in self._phases:
            self._phases[phase] = {
                "frames_processed": 0,
                "frames_scanned": 0,
                "stride": 1,
                "duration_seconds": 0.0,
            }

    def end_phase(
        self,
        phase: str | None = None,
        *,
        frames_processed: int = 0,
        frames_scanned: int = 0,
        stride: int = 1,
    ) -> None:
        """End timing for a phase and record stats."""
        target_phase = phase or self._current_phase
        if not target_phase or target_phase not in self._phases:
            return
        if self._current_start is not None:
            elapsed = time.time() - self._current_start
            self._phases[target_phase]["duration_seconds"] += elapsed
        self._phases[target_phase]["frames_processed"] = frames_processed
        self._phases[target_phase]["frames_scanned"] = frames_scanned
        self._phases[target_phase]["stride"] = stride
        self._current_phase = None
        self._current_start = None

    def add_phase_stats(
        self,
        phase: str,
        *,
        frames_processed: int,
        frames_scanned: int,
        stride: int,
        duration_seconds: float,
    ) -> None:
        """Directly add stats for a phase (for phases timed externally)."""
        self._phases[phase] = {
            "frames_processed": frames_processed,
            "frames_scanned": frames_scanned,
            "stride": stride,
            "duration_seconds": round(duration_seconds, 2),
        }

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Return summary of all phases."""
        return dict(self._phases)

    def log_summary(self, ep_id: str) -> None:
        """Log a formatted summary of all phases for local mode diagnostics."""
        if not LOCAL_MODE_INSTRUMENTATION:
            return
        lines = [f"[LOCAL MODE] detect_track phase summary for {ep_id}:"]
        total_duration = 0.0
        for phase_name, stats in self._phases.items():
            duration = stats.get("duration_seconds", 0.0)
            total_duration += duration
            frames_proc = stats.get("frames_processed", 0)
            frames_scan = stats.get("frames_scanned", 0)
            stride = stats.get("stride", 1)
            lines.append(
                f"  {phase_name}: frames_processed={frames_proc} frames_scanned={frames_scan} "
                f"stride={stride} wall_time_s={duration:.1f}s"
            )
        lines.append(f"  TOTAL wall_time_s={total_duration:.1f}s ({total_duration/60:.1f}m)")
        for line in lines:
            LOGGER.info(line)
            # Also print to stdout for local mode visibility
            print(line)


def _coreml_provider_available() -> bool:
    try:
        import onnxruntime as ort  # type: ignore

        providers = ort.get_available_providers()
    except Exception:
        return False
    return any(provider.lower().startswith("coreml") for provider in providers)


COREML_PROVIDER_AVAILABLE = _coreml_provider_available()
if _APPLE_SILICON and not COREML_PROVIDER_AVAILABLE:
    LOGGER.warning(
        "CoreMLExecutionProvider unavailable on Apple Silicon. Install onnxruntime-coreml to enable CoreML acceleration."
    )


def _load_performance_profile(profile_name: str | None) -> Dict[str, Any]:
    """Load performance profile config from YAML with sensible fallbacks."""
    if not profile_name:
        return {}
    normalized = str(profile_name).strip().lower()
    if normalized == "fast_cpu":
        normalized = "low_power"

    config_path = REPO_ROOT / "config" / "pipeline" / "performance_profiles.yaml"
    loaded_profiles: Dict[str, Any] = {}
    if config_path.exists():
        try:
            import yaml

            with config_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
                if isinstance(data, dict):
                    loaded_profiles = data
        except Exception as exc:  # pragma: no cover - optional dependency
            LOGGER.warning("Failed to load performance profiles: %s", exc)

    if normalized in loaded_profiles and isinstance(loaded_profiles[normalized], dict):
        return loaded_profiles[normalized]

    return PROFILE_DEFAULTS.get(normalized, {})


def _flag_present(raw_argv: List[str], *flags: str) -> bool:
    """Check if any of the given flags appear in the raw argv list."""
    for token in raw_argv:
        for flag in flags:
            if token == flag or token.startswith(f"{flag}="):
                return True
    return False


def _parse_retinaface_det_size(value: str | None) -> tuple[int, int] | None:
    if not value:
        return 640, 640
    tokens: list[str] = []
    buf = value.replace("x", ",").replace("X", ",")
    for part in buf.split(","):
        part = part.strip()
        if part:
            tokens.append(part)
    if len(tokens) != 2:
        return 640, 640
    try:
        width = max(int(float(tokens[0])), 1)
        height = max(int(float(tokens[1])), 1)
        return width, height
    except ValueError:
        return 640, 640


RETINAFACE_DET_SIZE = _parse_retinaface_det_size(os.environ.get("RETINAFACE_DET_SIZE"))
# Default to 480x480 for CoreML to reduce thermal load on Apple Silicon
RETINAFACE_COREML_DET_SIZE = _parse_retinaface_det_size(os.environ.get("RETINAFACE_COREML_DET_SIZE") or "480x480")


def _normalize_det_thresh(value: float | str | None) -> float:
    try:
        numeric = float(value) if value is not None else RETINAFACE_SCORE_THRESHOLD
    except (TypeError, ValueError):
        numeric = RETINAFACE_SCORE_THRESHOLD
    return min(max(numeric, 0.0), 1.0)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "off", "no"}:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


# Minimum bounding box area for ByteTrack to consider a detection
# Lower values allow tracking smaller faces. Original: 20.0, Now: 10.0
BYTE_TRACK_MIN_BOX_AREA_DEFAULT = max(
    _env_float("SCREENALYTICS_MIN_BOX_AREA", _env_float("BYTE_TRACK_MIN_BOX_AREA", 10.0)),
    0.0,
)
DEFAULT_GMC_METHOD = os.environ.get("SCREENALYTICS_GMC_METHOD", "sparseOptFlow")
DEFAULT_REID_MODEL = os.environ.get("SCREENALYTICS_REID_MODEL", "yolov8n-cls.pt")
DEFAULT_REID_ENABLED = os.environ.get("SCREENALYTICS_REID_ENABLED", "1").lower() in {
    "1",
    "true",
    "yes",
}
RETINAFACE_HELP = (
    "RetinaFace weights missing or could not initialize. See README 'Models' or run scripts/fetch_models.py."
)
ARC_FACE_HELP = "ArcFace weights missing or could not initialize. See README 'Models' or run scripts/fetch_models.py."
# Strict tracking defaults (matching config/pipeline/tracking.yaml)
# Tightened after multi-person track bug analysis (track 57 in rhoslc-s06e88)
GATE_APPEAR_T_HARD_DEFAULT = float(os.environ.get("TRACK_GATE_APPEAR_HARD", "0.70"))  # Was 0.75, lowered to catch more switches
GATE_APPEAR_T_SOFT_DEFAULT = float(os.environ.get("TRACK_GATE_APPEAR_SOFT", "0.78"))  # Was 0.82, lowered for faster streak detection
GATE_APPEAR_STREAK_DEFAULT = max(int(os.environ.get("TRACK_GATE_APPEAR_STREAK", "2")), 1)
GATE_IOU_THRESHOLD_DEFAULT = float(os.environ.get("TRACK_GATE_IOU", "0.55"))  # Was 0.50, raised to catch near-misses like 0.5013
GATE_PROTO_MOMENTUM_DEFAULT = min(max(float(os.environ.get("TRACK_GATE_PROTO_MOM", "0.85")), 0.0), 1.0)  # Was 0.90, reduced for faster adaptation
GATE_EMB_EVERY_DEFAULT = max(
    int(os.environ.get("TRACK_GATE_EMB_EVERY", "3")), 0
)  # Was 10, reduced to catch more person switches (critical fix)

# Adaptive rerun: disable appearance gate when it is clearly over-splitting.
GATE_AUTO_RERUN_ENABLED = str(os.environ.get("TRACK_GATE_AUTO_RERUN", "1")).strip().lower() in {"1", "true", "yes", "on"}
GATE_AUTO_RERUN_FORCED_SPLITS_THRESHOLD = max(int(os.environ.get("TRACK_GATE_AUTO_RERUN_FORCED_SPLITS", "250")), 0)
GATE_AUTO_RERUN_MAX_ID_SWITCHES = max(int(os.environ.get("TRACK_GATE_AUTO_RERUN_MAX_ID_SWITCHES", "10")), 0)
# Require that a meaningful share of forced_splits came from the gate (vs scene cuts) when available.
GATE_AUTO_RERUN_MIN_GATE_SPLITS_SHARE = max(
    min(float(os.environ.get("TRACK_GATE_AUTO_RERUN_MIN_GATE_SHARE", "0.5")), 1.0),
    0.0,
)
# ByteTrack spatial matching - strict defaults
TRACK_BUFFER_BASE_DEFAULT = max(
    _env_int("SCREANALYTICS_TRACK_BUFFER", _env_int("BYTE_TRACK_BUFFER", 15)),
    1,
)
BYTE_TRACK_MATCH_THRESH_DEFAULT = _env_float("BYTE_TRACK_MATCH_THRESH", 0.85)
TRACK_HIGH_THRESH_DEFAULT = _env_float(
    "SCREENALYTICS_TRACK_HIGH_THRESH",
    _env_float("BYTE_TRACK_HIGH_THRESH", 0.45),
)
TRACK_NEW_THRESH_DEFAULT = _env_float(
    "SCREENALYTICS_NEW_TRACK_THRESH",
    _env_float("BYTE_TRACK_NEW_TRACK_THRESH", 0.70),
)
TRACK_MAX_GAP_SEC = float(os.environ.get("TRACK_MAX_GAP_SEC", "0.5"))
TRACK_PROTO_MAX_SAMPLES = max(int(os.environ.get("TRACK_PROTO_MAX_SAMPLES", "12")), 2)
TRACK_PROTO_SIM_DELTA = float(os.environ.get("TRACK_PROTO_SIM_DELTA", "0.08"))
TRACK_PROTO_SIM_MIN = float(os.environ.get("TRACK_PROTO_SIM_MIN", "0.6"))
DEFAULT_CLUSTER_SIMILARITY = float(
    os.environ.get("SCREANALYTICS_CLUSTER_SIM", "0.68")
)  # Stricter threshold (was 0.58) to prevent grouping different people

# Load and apply YAML config overrides if available (only if env vars not set)
_YAML_CONFIG = _load_tracking_config_yaml()
if _YAML_CONFIG and "BYTE_TRACK_BUFFER" not in os.environ and "SCREANALYTICS_TRACK_BUFFER" not in os.environ:
    if "track_buffer" in _YAML_CONFIG:
        TRACK_BUFFER_BASE_DEFAULT = max(int(_YAML_CONFIG["track_buffer"]), 1)
        LOGGER.info("Applied track_buffer=%s from YAML config", TRACK_BUFFER_BASE_DEFAULT)
if _YAML_CONFIG and "BYTE_TRACK_MATCH_THRESH" not in os.environ:
    if "match_thresh" in _YAML_CONFIG:
        BYTE_TRACK_MATCH_THRESH_DEFAULT = float(_YAML_CONFIG["match_thresh"])
        LOGGER.info(
            "Applied match_thresh=%.2f from YAML config",
            BYTE_TRACK_MATCH_THRESH_DEFAULT,
        )
if _YAML_CONFIG and "BYTE_TRACK_HIGH_THRESH" not in os.environ and "SCREENALYTICS_TRACK_HIGH_THRESH" not in os.environ:
    if "track_thresh" in _YAML_CONFIG:
        TRACK_HIGH_THRESH_DEFAULT = float(_YAML_CONFIG["track_thresh"])
        TRACK_NEW_THRESH_DEFAULT = TRACK_HIGH_THRESH_DEFAULT
        LOGGER.info("Applied track_thresh=%.2f from YAML config", TRACK_HIGH_THRESH_DEFAULT)

# Appearance gate enable/disable (default from tracking.yaml; override via env or CLI)
GATE_ENABLED_DEFAULT = True
if "TRACK_GATE_ENABLED" in os.environ:
    cleaned = str(os.environ.get("TRACK_GATE_ENABLED", "")).strip().lower()
    if cleaned in {"0", "false", "no", "off"}:
        GATE_ENABLED_DEFAULT = False
    elif cleaned in {"1", "true", "yes", "on"}:
        GATE_ENABLED_DEFAULT = True
elif _YAML_CONFIG and "gate_enabled" in _YAML_CONFIG:
    try:
        GATE_ENABLED_DEFAULT = bool(_YAML_CONFIG.get("gate_enabled"))
        LOGGER.info("Applied gate_enabled=%s from YAML config", GATE_ENABLED_DEFAULT)
    except Exception:
        GATE_ENABLED_DEFAULT = True

# Load detection config for wide shot mode and small face settings
_DETECTION_CONFIG = _load_detection_config_yaml()
WIDE_SHOT_MODE_ENABLED = _DETECTION_CONFIG.get("wide_shot_mode", False)
WIDE_SHOT_INPUT_SIZE = int(_DETECTION_CONFIG.get("wide_shot_input_size", 960))
WIDE_SHOT_MIN_FACE_SIZE = int(_DETECTION_CONFIG.get("wide_shot_min_face_size", 12))
WIDE_SHOT_CONFIDENCE_TH = float(_DETECTION_CONFIG.get("wide_shot_confidence_th", 0.40))
DETECTION_MIN_SIZE = int(_DETECTION_CONFIG.get("min_size", 16))
DETECTION_CONFIDENCE_TH = float(_DETECTION_CONFIG.get("confidence_th", 0.50))
# Person fallback settings
PERSON_FALLBACK_ENABLED = _DETECTION_CONFIG.get("enable_person_fallback", False)
PERSON_FALLBACK_MIN_BODY_HEIGHT = int(_DETECTION_CONFIG.get("person_fallback_min_body_height", 100))
PERSON_FALLBACK_FACE_REGION_RATIO = float(_DETECTION_CONFIG.get("person_fallback_face_region_ratio", 0.25))
PERSON_FALLBACK_CONFIDENCE_TH = float(_DETECTION_CONFIG.get("person_fallback_confidence_th", 0.50))
PERSON_FALLBACK_MAX_PER_FRAME = int(_DETECTION_CONFIG.get("person_fallback_max_per_frame", 10))

if WIDE_SHOT_MODE_ENABLED:
    LOGGER.info(
        "Wide shot mode ENABLED: input_size=%d, min_face=%d, conf=%.2f",
        WIDE_SHOT_INPUT_SIZE,
        WIDE_SHOT_MIN_FACE_SIZE,
        WIDE_SHOT_CONFIDENCE_TH,
    )
if PERSON_FALLBACK_ENABLED:
    LOGGER.info(
        "Person fallback ENABLED: min_body_height=%d, face_ratio=%.2f",
        PERSON_FALLBACK_MIN_BODY_HEIGHT,
        PERSON_FALLBACK_FACE_REGION_RATIO,
    )

MIN_IDENTITY_SIMILARITY = float(os.environ.get("SCREENALYTICS_MIN_IDENTITY_SIM", "0.55"))  # Stricter (was 0.50)
# Minimum cohesion for a cluster to be kept together; below this, split into singletons
MIN_CLUSTER_COHESION = float(os.environ.get("SCREENALYTICS_MIN_CLUSTER_COHESION", "0.55"))
FACE_MIN_CONFIDENCE = float(os.environ.get("FACES_MIN_CONF", "0.60"))
FACE_MIN_BLUR = float(os.environ.get("FACES_MIN_BLUR", "18.0"))  # TUNED: 35.0 -> 18.0 to allow more blurry faces
FACE_MIN_STD = float(os.environ.get("FACES_MIN_STD", "1.0"))

# Lower thresholds for single-face tracks (more permissive to avoid orphaned clusters)
# Single-face tracks have no redundancy, so we accept lower quality to get at least one embedding
FACE_MIN_CONFIDENCE_SINGLE = float(os.environ.get("FACES_MIN_CONF_SINGLE", "0.45"))
FACE_MIN_BLUR_SINGLE = float(os.environ.get("FACES_MIN_BLUR_SINGLE", "15.0"))
FACE_MIN_STD_SINGLE = float(os.environ.get("FACES_MIN_STD_SINGLE", "0.5"))
BYTE_TRACK_BUFFER_DEFAULT = TRACK_BUFFER_BASE_DEFAULT
BYTE_TRACK_HIGH_THRESH_DEFAULT = TRACK_HIGH_THRESH_DEFAULT
BYTE_TRACK_NEW_TRACK_THRESH_DEFAULT = TRACK_NEW_THRESH_DEFAULT
BYTE_TRACK_MIN_BOX_AREA = BYTE_TRACK_MIN_BOX_AREA_DEFAULT


def _resolve_track_sample_limit(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "unlimited", "all", "off", "disable"}:
            return None
        try:
            numeric = int(float(text))
        except ValueError:
            return None
    else:
        numeric = int(value)
    return numeric if numeric > 0 else None


def _set_track_sample_limit(value: int | None) -> None:
    global TRACK_SAMPLE_LIMIT
    TRACK_SAMPLE_LIMIT = _resolve_track_sample_limit(value)


TRACK_SAMPLE_LIMIT = _resolve_track_sample_limit(os.environ.get("SCREENALYTICS_TRACK_SAMPLE_LIMIT"))


# Seed-based detection boosting configuration
SEED_BOOST_ENABLED = _env_flag("SEED_BOOST_ENABLED", False)
SEED_BOOST_SCORE_DELTA = float(os.environ.get("SEED_BOOST_SCORE_DELTA", "0.15"))
SEED_BOOST_MIN_SIM = float(os.environ.get("SEED_BOOST_MIN_SIM", "0.42"))

SCENE_DETECTOR_CHOICES = ("pyscenedetect", "internal", "off")
_RAW_SCENE_DETECTOR = os.environ.get("SCENE_DETECTOR", "pyscenedetect").strip().lower()
SCENE_DETECTOR_DEFAULT = _RAW_SCENE_DETECTOR if _RAW_SCENE_DETECTOR in SCENE_DETECTOR_CHOICES else "pyscenedetect"
SCENE_DETECT_DEFAULT = SCENE_DETECTOR_DEFAULT != "off"
SCENE_THRESHOLD_DEFAULT = _env_float("SCENE_THRESHOLD", 27.0)
SCENE_MIN_LEN_DEFAULT = max(_env_int("SCENE_MIN_LEN", 12), 1)
_SCENE_WARMUP_FRAMES_PER_CUT_DEFAULT = max(int(_DETECTION_CONFIG.get("scene_warmup_frames_per_cut", 1) or 0), 0)
SCENE_WARMUP_DETS_DEFAULT = max(_env_int("SCENE_WARMUP_DETS", _SCENE_WARMUP_FRAMES_PER_CUT_DEFAULT), 0)
SCENE_WARMUP_CAP_RATIO_DEFAULT = float(
    os.environ.get("SCENE_WARMUP_CAP_RATIO", _DETECTION_CONFIG.get("scene_warmup_cap_ratio", 0.25))
)
SCENE_WARMUP_CAP_RATIO_DEFAULT = max(min(SCENE_WARMUP_CAP_RATIO_DEFAULT, 10.0), 0.0)


def _normalize_device_label(device: str | None) -> str:
    normalized = (device or "cpu").lower()
    if normalized in {"0", "cuda", "gpu"}:
        return "cuda"
    return normalized


def _filter_providers(requested: list[str], available: list[str], allow_cpu_fallback: bool = True) -> list[str]:
    """
    Filter ONNX providers to only those available, optionally including CPU fallback.

    Args:
        requested: Desired providers in priority order
        available: Providers available in this ONNX Runtime build
        allow_cpu_fallback: If True, append CPUExecutionProvider as fallback (default: True)

    Returns:
        Filtered list with optional CPU fallback appended
    """
    filtered = [p for p in requested if p in available]
    if allow_cpu_fallback and "CPUExecutionProvider" not in filtered:
        filtered.append("CPUExecutionProvider")
    return filtered


class AcceleratorUnavailableError(RuntimeError):
    """Raised when a requested accelerator (CoreML/CUDA) is not available and CPU fallback is disabled."""
    pass


def _onnx_providers_for(
    device: str | None,
    allow_cpu_fallback: bool = False,
    cpu_threads: int | None = None,
) -> tuple[list[str], str]:
    """
    Select ONNX Runtime execution providers based on device preference.

    IMPORTANT: This function now FAILS FAST by default when an accelerator is
    explicitly requested but unavailable. To allow silent fallback to CPU, you
    must explicitly pass allow_cpu_fallback=True.

    Order of preference for device="auto":
    - CUDA (NVIDIA GPUs on Linux/Windows)
    - CoreML (Apple Silicon M1/M2/M3 on macOS)
    - CPU (fallback, only if allowed)

    Args:
        device: Device preference ("auto", "cuda", "coreml", "cpu", etc.)
        allow_cpu_fallback: If False (default) and the requested accelerator is
                           unavailable, raises AcceleratorUnavailableError instead
                           of silently falling back to CPU. This prevents thermal
                           issues on laptops.
        cpu_threads: If provided, limit CPU provider threads (intra/inter-op).
                    Ignored if CPU provider is not used.

    Returns:
        (providers, resolved_device) tuple where providers is a list of
        ONNX execution providers in priority order, and resolved_device
        is a string label for logging ("cuda", "coreml", or "cpu").

    Raises:
        AcceleratorUnavailableError: If accelerator is requested but unavailable
                                      and allow_cpu_fallback=False.
    """
    normalized = (device or "auto").lower()
    auto_requested = normalized == "auto"
    explicit_cpu = normalized == "cpu"
    providers: list[str] = ["CPUExecutionProvider"]
    resolved = "cpu"

    # Get available ONNX providers
    try:
        import onnxruntime as ort  # type: ignore

        available = ort.get_available_providers()
    except Exception:
        available = []

    def _log_providers(selected: list[str], resolved_device: str) -> None:
        """Log provider selection when local mode instrumentation is enabled."""
        if LOCAL_MODE_INSTRUMENTATION:
            msg = f"[LOCAL MODE] ONNX providers: {selected} (resolved_device={resolved_device}, available={available})"
            LOGGER.info(msg)
            print(msg)

    # Explicit CPU request - always allowed
    if explicit_cpu:
        _log_providers(providers, resolved)
        return providers, resolved

    # Explicit CUDA request (NVIDIA GPUs)
    if normalized in {"cuda", "0", "gpu"}:
        requested = ["CUDAExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            providers = _filter_providers(requested, available, allow_cpu_fallback=allow_cpu_fallback)
            resolved = "cuda"
            _log_providers(providers, resolved)
            return providers, resolved

        # CUDA unavailable - fail fast or warn
        if not allow_cpu_fallback:
            error_msg = (
                "CUDA requested (device=cuda) but CUDAExecutionProvider is not available. "
                "Either install CUDA-enabled onnxruntime, select a different device (coreml/cpu), "
                "or set --allow-cpu-fallback to enable CPU fallback."
            )
            LOGGER.error(error_msg)
            raise AcceleratorUnavailableError(error_msg)

        # Fallback allowed - warn and continue
        LOGGER.warning(
            "[FALLBACK] CUDA requested but CUDAExecutionProvider unavailable. "
            "Falling back to CPU (this will be slower and may cause thermal issues)."
        )
        print("[WARN] CUDA unavailable; running on CPU (fallback enabled). This will be slower and hotter.")
        _log_providers(providers, resolved)
        return providers, resolved

    # Explicit MPS/CoreML request (Apple Silicon)
    if normalized in {"mps", "metal", "apple", "coreml"}:
        requested = ["CoreMLExecutionProvider"]
        if "CoreMLExecutionProvider" in available:
            providers = _filter_providers(requested, available, allow_cpu_fallback=allow_cpu_fallback)
            resolved = "coreml"
            if not allow_cpu_fallback:
                LOGGER.info(
                    "CoreML-only mode enabled: CPU fallback disabled to enforce <300%% CPU budget."
                )
            _log_providers(providers, resolved)
            return providers, resolved

        # CoreML unavailable - fail fast or warn
        if not allow_cpu_fallback:
            error_msg = (
                f"CoreML requested (device={normalized}) but CoreMLExecutionProvider is not available. "
                "Either install onnxruntime-coreml, select a different device (cpu), "
                "or set --allow-cpu-fallback to enable CPU fallback."
            )
            LOGGER.error(error_msg)
            raise AcceleratorUnavailableError(error_msg)

        # Fallback allowed - warn and continue
        LOGGER.warning(
            "[FALLBACK] CoreML requested but CoreMLExecutionProvider unavailable. "
            "Falling back to CPU (this will be slower and may cause thermal issues)."
        )
        print("[WARN] CoreML unavailable; running on CPU (fallback enabled). This will be slower and hotter.")
        _log_providers(providers, resolved)
        return providers, resolved

    # Auto-detect best available provider
    if normalized == "auto":
        # Prefer CUDA on Linux/Windows
        if "CUDAExecutionProvider" in available:
            providers = _filter_providers(["CUDAExecutionProvider"], available, allow_cpu_fallback=allow_cpu_fallback)
            resolved = "cuda"
            _log_providers(providers, resolved)
            return providers, resolved

        # Prefer CoreML on macOS (Apple Silicon) when CUDA is unavailable
        if "CoreMLExecutionProvider" in available:
            # On macOS with Apple Silicon, prefer CoreML over CPU
            providers = _filter_providers(["CoreMLExecutionProvider"], available, allow_cpu_fallback=allow_cpu_fallback)
            resolved = "coreml"
            _log_providers(providers, resolved)
            return providers, resolved

        # No accelerator available for device=auto
        if APPLE_SILICON_HOST and not allow_cpu_fallback:
            # On Apple Silicon, we should have CoreML - fail if not available
            error_msg = (
                "device=auto on Apple Silicon but CoreMLExecutionProvider is not available. "
                "This likely means onnxruntime-coreml is not installed. "
                "Install it or set --allow-cpu-fallback to enable CPU fallback."
            )
            LOGGER.error(error_msg)
            raise AcceleratorUnavailableError(error_msg)

        # Fallback to CPU (with warning)
        LOGGER.warning(
            "device=auto falling back to CPU; no accelerator providers available (providers=%s)",
            available or ["CPUExecutionProvider"],
        )
        if not explicit_cpu:
            print("[WARN] No accelerator available; running on CPU. Consider installing onnxruntime-coreml.")

    # Fallback to CPU for unknown device values or when no accelerators available
    _log_providers(providers, resolved)
    return providers, resolved


def _init_retinaface(
    model_name: str,
    device: str,
    score_thresh: float = RETINAFACE_SCORE_THRESHOLD,
    coreml_input_size: tuple[int, int] | None = None,
    allow_cpu_fallback: bool = False,
) -> tuple[Any, str]:
    """Initialize RetinaFace detector with the specified device.

    Args:
        model_name: InsightFace model name
        device: Device preference (auto/cuda/coreml/cpu)
        score_thresh: Detection score threshold
        coreml_input_size: Optional CoreML input size
        allow_cpu_fallback: If False (default), fails fast when accelerator unavailable

    Raises:
        AcceleratorUnavailableError: If accelerator requested but unavailable and
                                      allow_cpu_fallback=False
    """
    try:
        from insightface.model_zoo import get_model  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("insightface is required for RetinaFace detection") from exc

    providers, resolved = _onnx_providers_for(device, allow_cpu_fallback=allow_cpu_fallback)
    model = get_model(model_name)
    if model is None:
        raise RuntimeError(
            f"RetinaFace weights '{model_name}' not found. Install insightface models or run scripts/fetch_models.py."
        )
    ctx_id = 0 if resolved == "cuda" else -1
    # InsightFace 0.7.x configures detection threshold at prepare-time
    # (detect() no longer accepts a `threshold` kwarg).
    prepare_kwargs = {
        "ctx_id": ctx_id,
        "providers": providers,
        "nms": RETINAFACE_NMS,
        "det_thresh": float(score_thresh),
    }
    # Use CoreML-specific detection size if available and running on CoreML
    if resolved == "coreml":
        input_size = coreml_input_size or RETINAFACE_COREML_DET_SIZE
        if input_size:
            prepare_kwargs["input_size"] = input_size
    elif RETINAFACE_DET_SIZE:
        prepare_kwargs["input_size"] = RETINAFACE_DET_SIZE
    try:
        model.prepare(**prepare_kwargs)
    except TypeError:
        prepare_kwargs.pop("input_size", None)
        model.prepare(**prepare_kwargs)
    return model, resolved


def _init_arcface(model_name: str, device: str, allow_cpu_fallback: bool = False):
    """Initialize ArcFace embedder with the specified device.

    Args:
        model_name: InsightFace model name
        device: Device preference (auto/cuda/coreml/cpu)
        allow_cpu_fallback: If False (default), fails fast when accelerator unavailable

    Raises:
        AcceleratorUnavailableError: If accelerator requested but unavailable and
                                      allow_cpu_fallback=False
    """
    try:
        from insightface.model_zoo import get_model  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("insightface is required for ArcFace embeddings") from exc

    providers, resolved = _onnx_providers_for(device, allow_cpu_fallback=allow_cpu_fallback)
    model = get_model(model_name)
    if model is None:
        raise RuntimeError(
            f"ArcFace weights '{model_name}' not found. Install insightface models or run scripts/fetch_models.py."
        )
    ctx_id = 0 if resolved == "cuda" else -1
    model.prepare(ctx_id=ctx_id, providers=providers)
    return model, resolved


def ensure_retinaface_ready(device: str, det_thresh: float | None = None) -> tuple[bool, Optional[str], Optional[str]]:
    """Lightweight readiness probe for API/CLI preflight checks."""

    try:
        from insightface.app import FaceAnalysis  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        return False, f"insightface import failed: {exc}", None

    providers, resolved = _onnx_providers_for(device)
    ctx_id = 0 if resolved == "cuda" else -1
    profile = os.environ.get("RETINAFACE_PROFILE", "buffalo_l")
    detector = None
    try:
        try:
            detector = FaceAnalysis(name=profile, providers=providers)
        except TypeError:
            detector = FaceAnalysis(name=profile)
        # Use CoreML-specific detection size if available and running on CoreML
        if resolved == "coreml" and RETINAFACE_COREML_DET_SIZE:
            det_size = RETINAFACE_COREML_DET_SIZE
        else:
            det_size = RETINAFACE_DET_SIZE or (640, 640)
        detector.prepare(
            ctx_id=ctx_id,
            det_size=det_size,
        )
    except Exception as exc:  # pragma: no cover - surfaced via API tests
        return False, str(exc), resolved
    finally:
        detector = None
    return True, None, resolved


def ensure_arcface_ready(device: str) -> tuple[bool, Optional[str], Optional[str]]:
    """Ensure ArcFace weights/materials can initialize before running embeds."""

    model = None
    try:
        model, resolved = _init_arcface(ARC_FACE_MODEL_NAME, device)
    except Exception as exc:  # pragma: no cover - surfaced via API tests
        return False, str(exc), None
    finally:
        if model is not None:
            del model
    return True, None, resolved


def pick_device(explicit: str | None = None) -> str:
    """Return the safest device available.

    Order of preference: explicit override → CUDA → MPS → CPU.
    Values returned are what Ultralytics expects ("cpu", "mps", "cuda"/"0").
    """

    if explicit and explicit not in {"auto", ""}:
        return explicit

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():  # pragma: no cover - depends on env
            return "0"
        mps_available = getattr(torch.backends, "mps", None)
        if mps_available is not None and mps_available.is_available():  # pragma: no cover - mac only
            return "mps"
    except Exception:  # pragma: no cover - torch import/runtime guard
        # Torch import issues should fall back to CPU without crashing CLI.
        pass

    return "cpu"


def resolve_device(device: str | None = None, logger: logging.Logger | None = None) -> str:
    """Resolve a torch-compatible device string for torch-backed models.

    NOTE: This is intentionally torch-centric (cpu/cuda/mps). ONNX providers are tracked separately.
    """
    _requested, resolved, reason = _resolve_torch_device_request(device)
    if logger and reason:
        logger.info("[device] resolved=%s (reason=%s requested=%s)", resolved, reason, device)
    return resolved


def _onnx_provider_label(device: str | None) -> str:
    """Normalize a device selection into ONNX provider semantics for diagnostics."""
    normalized = (device or "auto").strip().lower()
    if normalized in {"0", "cuda", "gpu"}:
        return "cuda"
    if normalized in {"coreml", "mps", "metal", "apple"}:
        return "coreml"
    if normalized == "cpu":
        return "cpu"
    return "auto"


def _resolve_torch_device_request(device: str | None) -> tuple[str, str, str | None]:
    """Resolve a torch device string from a pipeline device request.

    - ONNX providers allow "coreml"; torch does not. Map "coreml" → "mps" (else "cpu") and record why.
    """
    requested_raw = (device or "auto").strip().lower()

    def _has_cuda() -> bool:
        try:
            import torch  # type: ignore

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _has_mps() -> bool:
        try:
            import torch  # type: ignore

            mps = getattr(torch.backends, "mps", None)
            return bool(mps is not None and mps.is_available())
        except Exception:
            return False

    if requested_raw in {"0", "cuda", "gpu"}:
        requested = "cuda"
        if _has_cuda():  # pragma: no cover - depends on env
            return requested, "cuda", None
        return requested, "cpu", "CUDA requested but not available; using cpu"

    if requested_raw in {"mps", "metal", "apple"}:
        requested = "mps"
        if _has_mps():  # pragma: no cover - mac only
            return requested, "mps", None
        return requested, "cpu", "MPS requested but not available; using cpu"

    if requested_raw == "coreml":
        requested = "mps"
        if _has_mps():  # pragma: no cover - mac only
            return requested, "mps", "CoreML requested; torch models run on mps"
        return requested, "cpu", "CoreML requested; mps not available; torch models run on cpu"

    if requested_raw == "cpu":
        return "cpu", "cpu", None

    # auto (or unknown): prefer CUDA, then MPS, then CPU.
    if _has_cuda():  # pragma: no cover - depends on env
        return "cuda", "cuda", "auto: selected cuda"
    if _has_mps():  # pragma: no cover - mac only
        return "mps", "mps", "auto: selected mps"
    return "cpu", "cpu", "auto: selected cpu"


def _normalize_detector_choice(detector: str | None) -> str:
    if detector:
        value = detector.strip().lower()
        if value in DETECTOR_CHOICES:
            return value
    return DEFAULT_DETECTOR


def _normalize_tracker_choice(tracker: str | None) -> str:
    if tracker:
        value = tracker.strip().lower()
        if value in TRACKER_CHOICES:
            return value
    return DEFAULT_TRACKER


def _normalize_scene_detector_choice(scene_detector: str | None) -> str:
    if scene_detector:
        value = scene_detector.strip().lower()
        if value in SCENE_DETECTOR_CHOICES:
            return value
    return SCENE_DETECTOR_DEFAULT


def _valid_face_box(bbox: np.ndarray, score: float, *, min_score: float, min_area: float) -> bool:
    # Validate bbox has valid numeric coordinates
    try:
        if len(bbox) < 4:
            return False
        width = float(bbox[2]) - float(bbox[0])
        height = float(bbox[3]) - float(bbox[1])
        area = max(width, 0.0) * max(height, 0.0)
    except (TypeError, ValueError, IndexError):
        return False
    if score < min_score:
        return False
    if area < min_area:
        return False
    try:
        ratio = width / max(height, 1e-6)
        return FACE_RATIO_BOUNDS[0] <= ratio <= FACE_RATIO_BOUNDS[1]
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def _nms_detections(
    detections: list[tuple[np.ndarray, float, np.ndarray | None]],
    thresh: float,
) -> list[tuple[np.ndarray, float, np.ndarray | None]]:
    ordered = sorted(range(len(detections)), key=lambda idx: detections[idx][1], reverse=True)
    keep: list[tuple[np.ndarray, float, np.ndarray | None]] = []
    while ordered:
        current_idx = ordered.pop(0)
        current = detections[current_idx]
        keep.append(current)
        remaining: list[int] = []
        for idx in ordered:
            iou = _bbox_iou(current[0].tolist(), detections[idx][0].tolist())
            if iou < thresh:
                remaining.append(idx)
        ordered = remaining
    return keep


@dataclass
class TrackAccumulator:
    track_id: int
    class_id: int | str
    first_ts: float
    last_ts: float
    first_frame_idx: int = -1
    last_frame_idx: int = -1
    frame_count: int = 0
    samples: List[dict] = field(default_factory=list)

    def add(
        self,
        ts: float,
        frame_idx: int,
        bbox_xyxy: List[float],
        *,
        confidence: float | None = None,
        landmarks: List[float] | None = None,
    ) -> None:
        self.frame_count += 1
        self.last_ts = ts
        if self.first_frame_idx < 0:
            self.first_frame_idx = frame_idx
        self.last_frame_idx = frame_idx
        limit = TRACK_SAMPLE_LIMIT
        if limit is None or len(self.samples) < limit:
            sample = {
                "frame_idx": frame_idx,
                "ts": round(float(ts), 4),
                "bbox_xyxy": [round(float(coord), 4) for coord in bbox_xyxy],
            }
            if landmarks:
                sample["landmarks"] = [round(float(val), 4) for val in landmarks]
            if confidence is not None:
                sample["conf"] = round(float(confidence), 4)
            self.samples.append(sample)

    def to_row(self) -> dict:
        row = {
            "track_id": self.track_id,
            "class": self.class_id,
            "first_ts": round(float(self.first_ts), 4),
            "last_ts": round(float(self.last_ts), 4),
            "frame_count": self.frame_count,
            "pipeline_ver": PIPELINE_VERSION,
        }
        if self.first_frame_idx >= 0:
            row["first_frame_idx"] = int(self.first_frame_idx)
        if self.last_frame_idx >= 0:
            row["last_frame_idx"] = int(self.last_frame_idx)
        if self.samples:
            row["bboxes_sampled"] = self.samples
        return row


@dataclass
class DetectionSample:
    bbox: np.ndarray
    conf: float
    class_idx: int
    class_label: str
    landmarks: np.ndarray | None = None
    embedding: np.ndarray | None = None


@dataclass
class TrackedObject:
    track_id: int
    bbox: np.ndarray
    conf: float
    class_idx: int
    class_label: str
    det_index: int | None = None
    landmarks: np.ndarray | None = None


@dataclass
class GateConfig:
    appear_t_hard: float = GATE_APPEAR_T_HARD_DEFAULT
    appear_t_soft: float = GATE_APPEAR_T_SOFT_DEFAULT
    appear_streak: int = GATE_APPEAR_STREAK_DEFAULT
    gate_iou: float = GATE_IOU_THRESHOLD_DEFAULT
    proto_momentum: float = GATE_PROTO_MOMENTUM_DEFAULT
    emb_every: int | None = None


@dataclass
class GateTrackState:
    proto: np.ndarray | None = None
    last_box: np.ndarray | None = None
    low_sim_streak: int = 0


TrackEmbeddingSample = Tuple[float, np.ndarray]


def _l2_normalize(vec: np.ndarray | None) -> np.ndarray | None:
    if vec is None:
        return None
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec
    return (vec / norm).astype(np.float32)


def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Compute cosine similarity between two L2-normalized vectors."""
    if a is None or b is None:
        return None
    # Check if arrays contain valid numeric values (no None, NaN, or Inf)
    try:
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return None
    except (TypeError, ValueError):
        # Array contains None or other non-numeric values
        return None

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    denom = float(norm_a * norm_b)
    if denom <= 0.0:
        return None
    return float(np.dot(a, b) / denom)


class AppearanceGate:
    def __init__(self, config: GateConfig) -> None:
        self.config = config
        self._states: dict[int, GateTrackState] = {}
        self.stats: Counter[str] = Counter()

    def reset_all(self) -> None:
        self._states.clear()

    def prune(self, active_ids: set[int]) -> None:
        for tracker_id in list(self._states.keys()):
            if tracker_id not in active_ids:
                self._states.pop(tracker_id, None)

    def needs_embedding(self, tracker_id: int) -> bool:
        """Check if a track needs an embedding computed.

        Returns True if:
        - Track is new (not in _states)
        - Track exists but has no prototype yet (proto is None)

        This ensures we always get an embedding for the first detection of a track,
        regardless of the embedding stride. Critical fix for multi-person track bug.
        """
        if tracker_id not in self._states:
            return True
        state = self._states[tracker_id]
        return state.proto is None

    def tracks_needing_embedding(self, tracker_ids: list[int]) -> set[int]:
        """Return set of track IDs that need embeddings computed.

        Used to force embedding computation for new tracks even when
        frame_idx % stride != 0.
        """
        return {tid for tid in tracker_ids if self.needs_embedding(tid)}

    def process(
        self,
        tracker_id: int,
        bbox: np.ndarray,
        embedding: np.ndarray | None,
        frame_idx: int,
    ) -> tuple[bool, str | None, float | None, float]:
        state = self._states.setdefault(tracker_id, GateTrackState())
        bbox_arr = bbox.astype(np.float32, copy=True)
        similarity = _cosine_similarity(embedding, state.proto)
        iou = 1.0
        if state.last_box is not None:
            iou = float(_bbox_iou(state.last_box.tolist(), bbox_arr.tolist()))
        split = False
        reason: str | None = None
        low_similarity = similarity is not None and similarity < self.config.appear_t_soft
        if similarity is not None and similarity < self.config.appear_t_hard:
            split = True
            reason = "hard"
        elif low_similarity:
            state.low_sim_streak += 1
            if state.low_sim_streak >= self.config.appear_streak:
                split = True
                reason = "streak"
        else:
            state.low_sim_streak = 0
        if not split and iou < self.config.gate_iou:
            split = True
            reason = "iou"
        if split:
            self.stats["splits_total"] += 1
            if reason:
                self.stats[f"split_{reason}"] += 1
            LOGGER.info(
                "[gate] split track=%s f=%s sim=%s iou=%.3f reason=%s",
                tracker_id,
                frame_idx,
                f"{similarity:.3f}" if similarity is not None else "n/a",
                iou,
                reason or "unknown",
            )
            state.low_sim_streak = 0
            state.proto = _l2_normalize(embedding.copy()) if embedding is not None else None
        else:
            if similarity is not None:
                self.stats["sim_sum"] += similarity
                self.stats["sim_count"] += 1
            if embedding is not None and not low_similarity:
                if state.proto is None:
                    state.proto = _l2_normalize(embedding.copy())
                else:
                    mixed = self.config.proto_momentum * state.proto + (1.0 - self.config.proto_momentum) * embedding
                    state.proto = _l2_normalize(mixed)
        state.last_box = bbox_arr
        return split, reason, similarity, iou

    def summary(self) -> Dict[str, Any]:
        sim_count = self.stats.get("sim_count", 0)
        avg_sim = None
        if sim_count:
            avg_sim = self.stats.get("sim_sum", 0.0) / max(sim_count, 1)
        return {
            "splits": {
                "hard": self.stats.get("split_hard", 0),
                "streak": self.stats.get("split_streak", 0),
                "iou": self.stats.get("split_iou", 0),
                "total": self.stats.get("splits_total", 0),
            },
            "avg_sim_kept": round(avg_sim, 4) if avg_sim is not None else None,
        }


def _parse_ep_id_for_show(ep_id: str) -> Optional[str]:
    """Extract show_id from ep_id (e.g., 'rhobh-s05e01' -> 'rhobh')."""
    import re

    match = re.match(r"^(?P<show>.+)-s\d{2}e\d{2}$", ep_id, re.IGNORECASE)
    if match:
        return match.group("show").lower()
    return None


def _load_show_seeds(show_id: str) -> List[Dict[str, Any]]:
    """Load all seed embeddings for a show from facebank."""
    if not show_id:
        return []

    try:
        from apps.api.services.facebank import FacebankService

        facebank_svc = FacebankService(data_root=DATA_ROOT)
        seeds = facebank_svc.get_all_seeds_for_show(show_id)
        return seeds
    except Exception as exc:
        LOGGER.debug("Failed to load seeds for show %s: %s", show_id, exc)
        return []


def _find_best_seed_match(
    embedding: np.ndarray,
    seeds: List[Dict[str, Any]],
    min_sim: float = SEED_BOOST_MIN_SIM,
) -> Optional[Tuple[str, float]]:
    """Find the best matching seed for an embedding.

    Returns (cast_id, similarity) if match found above threshold, else None.
    """
    if not seeds or embedding is None:
        return None

    best_sim = -1.0
    best_cast_id = None

    for seed in seeds:
        seed_emb = np.array(seed.get("embedding", []), dtype=np.float32)
        if seed_emb.size == 0:
            continue

        # Cosine similarity
        sim = float(np.dot(embedding, seed_emb) / (np.linalg.norm(embedding) * np.linalg.norm(seed_emb) + 1e-12))

        if sim > best_sim:
            best_sim = sim
            best_cast_id = seed.get("cast_id")

    if best_sim >= min_sim and best_cast_id:
        return (best_cast_id, best_sim)

    return None


class _TrackerDetections:
    """Lightweight structure that mimics ultralytics' Boxes for BYTETracker inputs."""

    def __init__(self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> None:
        self.xyxy = boxes.astype(np.float32)
        self.conf = scores.astype(np.float32)
        self.cls = classes.astype(np.float32)
        self._xywh: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, idx) -> "_TrackerDetections":
        """Support boolean-mask / slice indexing used by ultralytics' BYTETracker."""
        return _TrackerDetections(
            self.xyxy[idx],
            self.conf[idx],
            self.cls[idx],
        )

    @property
    def xywh(self) -> np.ndarray:
        if self._xywh is None:
            self._xywh = self.xyxy.copy()
            self._xywh[:, 2] = self._xywh[:, 2] - self._xywh[:, 0]
            self._xywh[:, 3] = self._xywh[:, 3] - self._xywh[:, 1]
            self._xywh[:, 0] = self._xywh[:, 0] + self._xywh[:, 2] / 2
            self._xywh[:, 1] = self._xywh[:, 1] + self._xywh[:, 3] / 2
        return self._xywh

    @property
    def xywhr(self) -> np.ndarray:
        return self.xywh


def _tracker_inputs_from_samples(
    detections: list[DetectionSample],
) -> _TrackerDetections:
    if detections:
        # Validate and filter out detections with invalid bboxes before vstack
        valid_samples: list[DetectionSample] = []
        for sample in detections:
            bbox, bbox_err = _safe_bbox_or_none(sample.bbox)
            if bbox is None:
                LOGGER.debug(
                    "Dropping detection before tracker due to invalid bbox: %s",
                    bbox_err,
                )
                continue
            sample.bbox = np.array(bbox, dtype=np.float32)
            valid_samples.append(sample)

        if valid_samples:
            boxes = np.vstack([sample.bbox for sample in valid_samples]).astype(np.float32)
            scores = np.asarray([sample.conf for sample in valid_samples], dtype=np.float32)
            classes = np.asarray([sample.class_idx for sample in valid_samples], dtype=np.float32)
            return _TrackerDetections(boxes, scores, classes)

    return _TrackerDetections(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
    )


@dataclass
class ByteTrackRuntimeConfig:
    """Runtime configuration for ByteTrack builds."""

    track_high_thresh: float = TRACK_HIGH_THRESH_DEFAULT
    new_track_thresh: float = TRACK_NEW_THRESH_DEFAULT
    match_thresh: float = BYTE_TRACK_MATCH_THRESH_DEFAULT
    track_low_thresh: float = 0.1
    track_buffer_base: int = TRACK_BUFFER_BASE_DEFAULT
    min_box_area: float = BYTE_TRACK_MIN_BOX_AREA_DEFAULT

    def __post_init__(self) -> None:
        self.track_high_thresh = min(max(float(self.track_high_thresh), 0.0), 1.0)
        self.new_track_thresh = min(max(float(self.new_track_thresh), 0.0), 1.0)
        self.match_thresh = min(max(float(self.match_thresh), 0.0), 1.0)
        self.track_buffer_base = max(int(self.track_buffer_base), 1)
        self.min_box_area = max(float(self.min_box_area), 0.0)

    def scaled_buffer(self, stride: int) -> int:
        stride_value = max(int(stride), 1)
        scale = max(1.0, float(stride_value) / 3.0)
        effective = max(int(round(self.track_buffer_base * scale)), self.track_buffer_base)
        return max(effective, 1)

    def summary(self, stride: int) -> Dict[str, Any]:
        return {
            "track_high_thresh": round(self.track_high_thresh, 3),
            "new_track_thresh": round(self.new_track_thresh, 3),
            "match_thresh": round(self.match_thresh, 3),
            "track_buffer": self.scaled_buffer(stride),
            "track_buffer_base": self.track_buffer_base,
            "min_box_area": round(self.min_box_area, 3),
            "stride": max(int(stride), 1),
        }


class ByteTrackAdapter:
    """Wrapper around ultralytics BYTETracker for direct invocation."""

    def __init__(
        self,
        frame_rate: float = 30.0,
        stride: int = 1,
        config: ByteTrackRuntimeConfig | None = None,
    ) -> None:
        self.frame_rate = max(frame_rate, 1)
        self.stride = max(stride, 1)
        self.config = config or ByteTrackRuntimeConfig()
        self._effective_buffer = self.config.scaled_buffer(self.stride)
        self._config_snapshot = self.config.summary(self.stride)
        self._tracker = self._build_tracker()

    def _build_tracker(self):
        from types import SimpleNamespace

        from ultralytics.trackers.byte_tracker import BYTETracker

        cfg = SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=self.config.track_high_thresh,
            track_low_thresh=self.config.track_low_thresh,
            new_track_thresh=self.config.new_track_thresh,
            track_buffer=self._effective_buffer,
            match_thresh=self.config.match_thresh,
            min_box_area=self.config.min_box_area,
            fuse_score=False,
        )
        return BYTETracker(cfg, frame_rate=self.frame_rate)

    @property
    def config_summary(self) -> Dict[str, Any]:
        return dict(self._config_snapshot)

    def update(self, detections: list[DetectionSample], frame_idx: int, image) -> list[TrackedObject]:
        det_struct = _tracker_inputs_from_samples(detections)
        tracks = self._tracker.update(det_struct, image)
        tracked: list[TrackedObject] = []
        if tracks.size == 0:
            return tracked
        for row in tracks:
            bbox = np.asarray(row[:4], dtype=np.float32)
            track_id = int(row[4])
            score = float(row[5])
            class_idx = int(row[6]) if len(row) > 6 else 0
            det_index = int(row[7]) if len(row) > 7 else None
            label = ""
            landmarks = None
            if det_index is not None and 0 <= det_index < len(detections):
                det = detections[det_index]
                label = det.class_label
                class_idx = det.class_idx
                landmarks = det.landmarks
            tracked.append(
                TrackedObject(
                    track_id=track_id,
                    bbox=bbox,
                    conf=score,
                    class_idx=class_idx,
                    class_label=label,
                    det_index=det_index,
                    landmarks=landmarks,
                )
            )
        return tracked

    def reset(self) -> None:
        self._tracker = self._build_tracker()


class StrongSortAdapter:
    """Adapter around Ultralytics BOT-SORT tracker (used as a StrongSORT-style ReID tracker)."""

    def __init__(self, frame_rate: float = 30.0) -> None:
        self.frame_rate = max(frame_rate, 1)
        self._tracker = self._build_tracker()

    def _build_tracker(self):
        from types import SimpleNamespace

        try:
            from ultralytics.trackers.bot_sort import BOTSORT
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError("StrongSORT tracker unavailable; ensure ultralytics>=8.2.70 is installed.") from exc

        cfg = SimpleNamespace(
            tracker_type="strongsort",
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            track_buffer=30,
            match_thresh=0.8,
            min_box_area=BYTE_TRACK_MIN_BOX_AREA,
            gmc_method=os.environ.get("SCREENALYTICS_GMC_METHOD", DEFAULT_GMC_METHOD),
            proximity_thresh=float(os.environ.get("SCREENALYTICS_REID_PROXIMITY", "0.6")),
            appearance_thresh=float(os.environ.get("SCREENALYTICS_REID_APPEARANCE", "0.7")),
            with_reid=_env_flag("SCREENALYTICS_REID_ENABLED", DEFAULT_REID_ENABLED),
            model=os.environ.get("SCREENALYTICS_REID_MODEL", DEFAULT_REID_MODEL) or "auto",
            fuse_score=_env_flag("SCREENALYTICS_REID_FUSE_SCORE", False),
        )
        return BOTSORT(cfg, frame_rate=self.frame_rate)

    def update(self, detections: list[DetectionSample], frame_idx: int, image) -> list[TrackedObject]:
        det_struct = _tracker_inputs_from_samples(detections)
        tracks = self._tracker.update(det_struct, image)
        tracked: list[TrackedObject] = []
        if tracks.size == 0:
            return tracked
        for row in tracks:
            bbox = np.asarray(row[:4], dtype=np.float32)
            track_id = int(row[4])
            score = float(row[5])
            class_idx = int(row[6]) if len(row) > 6 else 0
            det_index = int(row[7]) if len(row) > 7 else None
            label = ""
            landmarks = None
            if det_index is not None and 0 <= det_index < len(detections):
                det = detections[det_index]
                label = det.class_label
                class_idx = det.class_idx
                landmarks = det.landmarks
            tracked.append(
                TrackedObject(
                    track_id=track_id,
                    bbox=bbox,
                    conf=score,
                    class_idx=class_idx,
                    class_label=label,
                    det_index=det_index,
                    landmarks=landmarks,
                )
            )
        return tracked

    def reset(self) -> None:
        self._tracker = self._build_tracker()


class RetinaFaceDetectorBackend:
    def __init__(
        self,
        device: str,
        score_thresh: float = RETINAFACE_SCORE_THRESHOLD,
        *,
        coreml_input_size: tuple[int, int] | None = None,
        allow_cpu_fallback: bool = False,
    ) -> None:
        self.device = device
        self.score_thresh = max(min(float(score_thresh or RETINAFACE_SCORE_THRESHOLD), 1.0), 0.0)
        # Use config-based min_size (convert dimension to area)
        # In wide shot mode, use smaller min face size for better small face detection
        if WIDE_SHOT_MODE_ENABLED:
            min_size_dim = WIDE_SHOT_MIN_FACE_SIZE
            # Use wide shot confidence threshold
            self.score_thresh = max(min(float(WIDE_SHOT_CONFIDENCE_TH), 1.0), 0.0)
        else:
            min_size_dim = DETECTION_MIN_SIZE
        self.min_area = float(min_size_dim * min_size_dim)  # Convert pixel dimension to area
        self._model = None
        self._resolved_device: Optional[str] = None
        self._coreml_input_size = coreml_input_size
        self.allow_cpu_fallback = allow_cpu_fallback

    def _lazy_model(self):
        if self._model is not None:
            return self._model
        try:
            model, resolved = _init_retinaface(
                self.model_name,
                self.device,
                self.score_thresh,
                coreml_input_size=self._coreml_input_size,
                allow_cpu_fallback=self.allow_cpu_fallback,
            )
        except Exception as exc:
            raise RuntimeError(f"{RETINAFACE_HELP} ({exc})") from exc
        self._resolved_device = resolved
        self._model = model
        return self._model

    @property
    def model_name(self) -> str:
        return RETINAFACE_MODEL_NAME

    @property
    def resolved_device(self) -> str:
        if self._resolved_device is None:
            self.ensure_ready()
        return self._resolved_device or "cpu"

    def ensure_ready(self) -> None:
        self._lazy_model()

    def detect(self, image) -> list[DetectionSample]:
        model = self._lazy_model()
        # Threshold + input size configured during model.prepare. Some InsightFace
        # RetinaFace builds still require an explicit input_size, so pass it when
        # available.
        detect_kwargs = {}
        input_size = getattr(model, "input_size", None) or RETINAFACE_DET_SIZE
        if input_size:
            detect_kwargs["input_size"] = input_size
        bboxes, landmarks = model.detect(image, **detect_kwargs)
        if bboxes is None or len(bboxes) == 0:
            return []
        pending: list[tuple[np.ndarray, float, np.ndarray | None]] = []
        for idx in range(len(bboxes)):
            raw = bboxes[idx]
            score = float(raw[4]) if raw.shape[0] >= 5 else float(self.score_thresh)
            bbox = raw[:4].astype(np.float32)
            if not _valid_face_box(bbox, score, min_score=self.score_thresh, min_area=self.min_area):
                continue
            kps = None
            if landmarks is not None and idx < len(landmarks):
                kps = landmarks[idx].astype(np.float32).reshape(-1)
            pending.append((bbox, score, kps))
        filtered = _nms_detections(pending, RETINAFACE_NMS) if pending else []
        samples: list[DetectionSample] = []
        for bbox, score, kps in filtered:
            samples.append(
                DetectionSample(
                    bbox=bbox.astype(np.float32),
                    conf=score,
                    class_idx=0,
                    class_label=FACE_CLASS_LABEL,
                    landmarks=kps.copy() if isinstance(kps, np.ndarray) else None,
                )
            )
        return samples


def _build_face_detector(
    detector: str,
    device: str,
    score_thresh: float = RETINAFACE_SCORE_THRESHOLD,
    coreml_input_size: tuple[int, int] | None = None,
    allow_cpu_fallback: bool = False,
):
    return RetinaFaceDetectorBackend(
        device,
        score_thresh=score_thresh,
        coreml_input_size=coreml_input_size,
        allow_cpu_fallback=allow_cpu_fallback,
    )


class PersonFallbackDetector:
    """YOLO-based person detector for fallback when faces are too small to detect.

    When face detection fails (e.g., in wide shots), this detector finds people
    and estimates face regions from the upper portion of their body bounding boxes.
    """

    PERSON_CLASS_ID = 0  # COCO person class

    def __init__(
        self,
        device: str = "cpu",
        confidence_thresh: float = 0.50,
        min_body_height: int = 100,
        face_region_ratio: float = 0.25,
        max_detections: int = 10,
    ) -> None:
        self.device = device
        self.confidence_thresh = confidence_thresh
        self.min_body_height = min_body_height
        self.face_region_ratio = face_region_ratio
        self.max_detections = max_detections
        self._model = None
        self._load_attempted = False
        self._load_error: str | None = None
        self._load_status: str = "uninitialized"
        self.invocations = 0
        self.detections_emitted = 0

    def _lazy_model(self):
        """Lazily load YOLO model on first use."""
        if self._load_attempted:
            return self._model
        self._load_attempted = True
        try:
            from ultralytics import YOLO

            # Use YOLOv8n (nano) for speed - only need person detection
            self._model = YOLO("yolov8n.pt")
            # Force model to specified device
            if self.device and self.device != "cpu":
                self._model.to(self.device)
            LOGGER.info("Loaded YOLOv8n for person fallback detection on device=%s", self.device)
            self._load_status = "ok"
            self._load_error = None
        except ModuleNotFoundError as exc:
            self._load_status = "error"
            if getattr(exc, "name", None) == "ultralytics":
                self._load_error = "ultralytics_missing"
            else:
                self._load_error = f"import_error: {exc}"
            LOGGER.warning("Failed to load YOLO for person fallback: %s", self._load_error)
            self._model = None
        except ImportError as exc:
            self._load_status = "error"
            self._load_error = f"ultralytics_import_error: {exc}"
            LOGGER.warning("Failed to load YOLO for person fallback: %s", self._load_error)
            self._model = None
        except Exception as exc:
            self._load_status = "error"
            self._load_error = f"yolo_load_error: {type(exc).__name__}: {exc}"
            LOGGER.warning("Failed to load YOLO for person fallback: %s", self._load_error)
            self._model = None
        return self._model

    @property
    def load_status(self) -> str:
        return self._load_status

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def detect_persons(self, image: np.ndarray) -> list[DetectionSample]:
        """Detect persons and return estimated face regions.

        Args:
            image: BGR image array

        Returns:
            List of DetectionSample with estimated face bounding boxes
        """
        if not PERSON_FALLBACK_ENABLED:
            return []

        self.invocations += 1
        model = self._lazy_model()
        if model is None:
            return []

        try:
            # Run YOLO inference
            results = model(image, verbose=False, conf=self.confidence_thresh, classes=[self.PERSON_CLASS_ID])
            if not results or len(results) == 0:
                return []

            detections = []
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return []

            # Process person detections
            for i, box in enumerate(boxes):
                if i >= self.max_detections:
                    break

                # Get body bounding box (xyxy format)
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                body_height = y2 - y1
                body_width = x2 - x1

                # Skip if body is too small
                if body_height < self.min_body_height:
                    continue

                conf = float(box.conf[0].cpu().numpy())

                # Estimate face region from top portion of body
                # Face is typically in upper 20-25% of body, centered horizontally
                face_height = body_height * self.face_region_ratio
                face_width = min(face_height * 0.8, body_width * 0.6)  # Face is roughly square

                # Center the face region horizontally
                body_center_x = (x1 + x2) / 2
                face_x1 = body_center_x - face_width / 2
                face_x2 = body_center_x + face_width / 2
                face_y1 = y1  # Top of body
                face_y2 = y1 + face_height

                # Create detection sample
                face_bbox = np.array([face_x1, face_y1, face_x2, face_y2], dtype=np.float32)
                detections.append(
                    DetectionSample(
                        bbox=face_bbox,
                        conf=conf * 0.8,  # Discount confidence since this is estimated
                        class_idx=0,
                        class_label="face_estimated",  # Mark as estimated face
                        landmarks=None,  # No landmarks from body detection
                    )
                )

            LOGGER.debug("Person fallback: found %d estimated faces from %d persons", len(detections), len(boxes))
            self.detections_emitted += len(detections)
            return detections

        except Exception as exc:
            LOGGER.warning("Person fallback detection failed: %s", exc)
            return []


def _build_person_fallback_detector(device: str) -> PersonFallbackDetector | None:
    """Build person fallback detector if enabled in config."""
    if not PERSON_FALLBACK_ENABLED:
        return None
    # Guardrail: torch does not accept "coreml" as a device string.
    if device.strip().lower() == "coreml":
        _requested, resolved, _reason = _resolve_torch_device_request("coreml")
        device = resolved
    return PersonFallbackDetector(
        device=device,
        confidence_thresh=PERSON_FALLBACK_CONFIDENCE_TH,
        min_body_height=PERSON_FALLBACK_MIN_BODY_HEIGHT,
        face_region_ratio=PERSON_FALLBACK_FACE_REGION_RATIO,
        max_detections=PERSON_FALLBACK_MAX_PER_FRAME,
    )


def _build_tracker_adapter(
    tracker: str,
    frame_rate: float,
    stride: int = 1,
    config: ByteTrackRuntimeConfig | None = None,
) -> ByteTrackAdapter | StrongSortAdapter:
    if tracker == "strongsort":
        return StrongSortAdapter(frame_rate=frame_rate)
    return ByteTrackAdapter(frame_rate=frame_rate, stride=stride, config=config)


def _bytetrack_config_from_args(args: argparse.Namespace) -> ByteTrackRuntimeConfig:
    return ByteTrackRuntimeConfig(
        track_high_thresh=getattr(args, "track_high_thresh", TRACK_HIGH_THRESH_DEFAULT) or TRACK_HIGH_THRESH_DEFAULT,
        new_track_thresh=getattr(args, "new_track_thresh", TRACK_NEW_THRESH_DEFAULT) or TRACK_NEW_THRESH_DEFAULT,
        match_thresh=BYTE_TRACK_MATCH_THRESH_DEFAULT,
        track_low_thresh=0.1,
        track_buffer_base=getattr(args, "track_buffer", TRACK_BUFFER_BASE_DEFAULT) or TRACK_BUFFER_BASE_DEFAULT,
        min_box_area=getattr(args, "min_box_area", BYTE_TRACK_MIN_BOX_AREA_DEFAULT) or BYTE_TRACK_MIN_BOX_AREA_DEFAULT,
    )


# =============================================================================
# Embedding Backend Abstraction
# =============================================================================

class EmbeddingBackend(Protocol):
    """Protocol for face embedding backends.

    Supports multiple embedding implementations:
    - pytorch: PyTorch/ONNX Runtime via InsightFace (default)
    - tensorrt: TensorRT-accelerated ArcFace (GPU-optimized)
    """

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        """Encode face crops to embeddings.

        Args:
            crops: List of BGR face crops (any size, will be resized to 112x112)

        Returns:
            L2-normalized embeddings array of shape (N, 512)
        """
        ...

    def ensure_ready(self) -> None:
        """Ensure model is loaded and ready for inference."""
        ...


class TensorRTEmbeddingBackend:
    """TensorRT-accelerated ArcFace embedding backend.

    Uses TensorRT for GPU-accelerated face embedding inference.
    Provides ~5x speedup over PyTorch/ONNX Runtime on compatible GPUs.
    """

    def __init__(
        self,
        config_path: str = "config/pipeline/arcface_tensorrt.yaml",
        engine_path: Optional[str] = None,
    ):
        """
        Initialize TensorRT embedding backend.

        Args:
            config_path: Path to TensorRT configuration YAML
            engine_path: Explicit path to TensorRT engine (overrides config)
        """
        self.config_path = config_path
        self.engine_path = engine_path
        self._engine = None
        self._resolved_device = "tensorrt"

    def ensure_ready(self) -> None:
        """Load TensorRT engine and prepare for inference."""
        if self._engine is not None:
            return

        try:
            import tensorrt  # noqa: F401  # type: ignore
            import pycuda.driver  # noqa: F401  # type: ignore

            # Load config and build/load engine
            if self.engine_path:
                engine_path = Path(self.engine_path)
            else:
                from FEATURES.arcface_tensorrt.src.tensorrt_builder import TensorRTConfig, build_or_load_engine

                config = TensorRTConfig.from_yaml(Path(self.config_path))
                engine_path, _ = build_or_load_engine(config)

            if engine_path is None:
                raise RuntimeError("Failed to build or load TensorRT engine")

            from FEATURES.arcface_tensorrt.src.tensorrt_inference import TensorRTArcFace

            self._engine = TensorRTArcFace(engine_path=engine_path)
            # Warm engine once so fallback happens before embedding loop.
            self._engine._load_engine()
            LOGGER.info("TensorRT embedding backend ready: %s", engine_path)

        except ImportError as e:
            raise ImportError(
                f"TensorRT backend requires tensorrt and pycuda: {e}. "
                "Install with: pip install tensorrt pycuda"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TensorRT backend: {e}") from e

    @property
    def resolved_device(self) -> str:
        """Return the resolved device name."""
        return self._resolved_device

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        """Encode face crops to embeddings using TensorRT.

        Args:
            crops: List of BGR face crops

        Returns:
            L2-normalized embeddings array of shape (N, 512)
        """
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)

        self.ensure_ready()

        # Resize crops to 112x112 for ArcFace
        resized_batch: list[np.ndarray] = []
        valid_indices: list[int] = []
        embeddings: list[np.ndarray] = [np.zeros(512, dtype=np.float32)] * len(crops)

        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            resized = _resize_for_arcface(crop)
            if resized is None:
                continue
            resized_batch.append(resized)
            valid_indices.append(i)

        if not resized_batch:
            return np.vstack(embeddings)

        # Stack into batch array
        batch_array = np.stack(resized_batch, axis=0)

        # Run TensorRT inference
        feats = self._engine.embed(batch_array)

        # Place embeddings back into result array
        for idx, valid_idx in enumerate(valid_indices):
            embeddings[valid_idx] = feats[idx]

        return np.vstack(embeddings)


class _FallbackEmbeddingBackend:
    def __init__(
        self,
        primary: EmbeddingBackend,
        fallback: EmbeddingBackend,
        *,
        primary_label: str,
        fallback_label: str,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_label = primary_label
        self._fallback_label = fallback_label
        self._active: EmbeddingBackend | None = None
        self._active_label: str | None = None
        self._fallback_reason: str | None = None

    def _activate_primary(self) -> None:
        if self._active is not None:
            return
        try:
            self._primary.ensure_ready()
            self._active = self._primary
            self._active_label = self._primary_label
            self._fallback_reason = None
        except Exception as exc:
            LOGGER.warning(
                "[EMBED] Falling back from %s to %s: %s",
                self._primary_label,
                self._fallback_label,
                exc,
            )
            self._fallback.ensure_ready()
            self._active = self._fallback
            self._active_label = self._fallback_label
            self._fallback_reason = str(exc)

    def ensure_ready(self) -> None:
        self._activate_primary()
        if self._active is not None:
            self._active.ensure_ready()

    @property
    def resolved_device(self) -> str:
        if self._active is None:
            try:
                return getattr(self._primary, "resolved_device")
            except Exception:
                return getattr(self._fallback, "resolved_device", "cpu")
        return getattr(self._active, "resolved_device", "cpu")

    @property
    def active_backend_label(self) -> str:
        """Return the currently active backend label (primary vs fallback)."""
        if self._active_label is not None:
            return self._active_label
        return self._primary_label

    @property
    def fallback_reason(self) -> str | None:
        """Return the most recent reason for falling back (if any)."""
        return self._fallback_reason

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        self._activate_primary()
        if self._active is None:
            return np.zeros((len(crops), 512), dtype=np.float32)
        try:
            return self._active.encode(crops)
        except Exception as exc:
            if self._active is self._primary:
                LOGGER.warning(
                    "[EMBED] TensorRT encode failed; falling back to %s: %s",
                    self._fallback_label,
                    exc,
                )
                self._fallback.ensure_ready()
                self._active = self._fallback
                self._active_label = self._fallback_label
                self._fallback_reason = str(exc)
                return self._active.encode(crops)
            raise


def get_embedding_backend(
    backend_type: str = "pytorch",
    device: str = "auto",
    tensorrt_config: str = "config/pipeline/arcface_tensorrt.yaml",
    allow_cpu_fallback: bool = True,
) -> EmbeddingBackend:
    """
    Factory function to get the appropriate embedding backend.

    Args:
        backend_type: "pytorch" or "tensorrt"
        device: Device for PyTorch backend ("auto", "cuda", "cpu", "mps")
        tensorrt_config: Path to TensorRT config YAML
        allow_cpu_fallback: Allow CPU fallback for PyTorch backend

    Returns:
        Embedding backend instance
    """
    if backend_type == "tensorrt":
        tensorrt_backend = TensorRTEmbeddingBackend(config_path=tensorrt_config)
        if allow_cpu_fallback:
            return _FallbackEmbeddingBackend(
                tensorrt_backend,
                ArcFaceEmbedder(device, allow_cpu_fallback=allow_cpu_fallback),
                primary_label="tensorrt",
                fallback_label="pytorch",
            )
        return tensorrt_backend
    else:
        # Default to PyTorch/ONNX Runtime backend
        return ArcFaceEmbedder(device, allow_cpu_fallback=allow_cpu_fallback)


class ArcFaceEmbedder:
    def __init__(self, device: str, allow_cpu_fallback: bool = True) -> None:
        self.device = device
        self.allow_cpu_fallback = allow_cpu_fallback
        self._model = None
        self._resolved_device: Optional[str] = None

    def _lazy_model(self):
        if self._model is not None:
            return self._model
        try:
            model, resolved = _init_arcface(ARC_FACE_MODEL_NAME, self.device, allow_cpu_fallback=self.allow_cpu_fallback)
        except Exception as exc:
            raise RuntimeError(
                f"ArcFace init failed: {exc}. Install insightface + models or run scripts/fetch_models.py."
            ) from exc
        self._resolved_device = resolved
        self._model = model
        return self._model

    def ensure_ready(self) -> None:
        self._lazy_model()

    @property
    def resolved_device(self) -> str:
        if self._resolved_device is None:
            self.ensure_ready()
        return self._resolved_device or "cpu"

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        model = self._lazy_model()

        # Prepare batch: resize all crops and track valid indices
        resized_batch: list[np.ndarray] = []
        valid_indices: list[int] = []
        embeddings: list[np.ndarray] = [np.zeros(512, dtype=np.float32)] * len(crops)

        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            resized = _resize_for_arcface(crop)
            if resized is None:
                continue  # Skip faces too small to resize
            resized_batch.append(resized)
            valid_indices.append(i)

        if not resized_batch:
            return np.vstack(embeddings)

        # Batch inference: stack images into (N, H, W, C) array
        # InsightFace's get_feat supports batch processing via stacked arrays
        try:
            batch_array = np.stack(resized_batch, axis=0)
            # Try batch inference first (much faster)
            feats = model.get_feat(batch_array)
            feats = np.asarray(feats, dtype=np.float32)
            if feats.ndim == 1:
                # Single image case
                feats = feats.reshape(1, -1)

            # L2 normalize each embedding
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # Avoid division by zero
            feats = feats / norms

            # Place normalized embeddings back into result array
            for idx, valid_idx in enumerate(valid_indices):
                embeddings[valid_idx] = feats[idx]

        except Exception as batch_err:
            # Fallback to per-image inference if batch fails (common with mixed-size batches)
            LOGGER.debug("Batch embedding using per-image mode: %s", batch_err)
            for i, resized in zip(valid_indices, resized_batch):
                try:
                    feat = model.get_feat(resized)
                    vec = np.asarray(feat, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    embeddings[i] = vec
                except Exception:
                    # Skip faces that fail embedding (zero vector already in place)
                    pass

        return np.vstack(embeddings)


# Alias for backwards compatibility
PyTorchEmbeddingBackend = ArcFaceEmbedder


def _resize_for_arcface(image):
    """Resize image to ArcFace input size (112x112).

    Returns None if image has invalid dimensions (too small to resize).
    """
    import cv2  # type: ignore

    # Check for valid dimensions before resize
    if image is None:
        return None
    if not hasattr(image, 'shape') or len(image.shape) < 2:
        return None
    if image.size == 0:
        return None
    h, w = image.shape[:2]
    # Require minimum 10x10 pixels - tiny crops cause OpenCV resize errors
    MIN_CROP_DIM = 10
    if h < MIN_CROP_DIM or w < MIN_CROP_DIM:
        return None

    # Ensure image is contiguous and has correct dtype
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)

    # Ensure 3-channel uint8 for proper resize
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    target = (112, 112)
    try:
        resized = cv2.resize(image, target)
        return resized
    except cv2.error:
        return None


def _letterbox_square(image, size: int = 112, pad_value: int = 127):
    """Resize with aspect preservation and center pad to a square canvas."""
    import cv2  # type: ignore

    arr = to_u8_bgr(image)
    if arr is None or arr.size == 0:
        return np.full((size, size, 3), pad_value, dtype=np.uint8)
    h, w = arr.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((size, size, 3), pad_value, dtype=np.uint8)
    scale = min(size / max(w, 1), size / max(h, 1))
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    resized = cv2.resize(arr, (new_w, new_h))
    canvas = np.full((size, size, 3), pad_value, dtype=resized.dtype)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def _safe_bbox_or_none(
    bbox: list[float] | np.ndarray,
) -> tuple[list[float] | None, str | None]:
    """
    Validate bbox coordinates are finite numbers before cropping operations.

    Returns:
        (validated_bbox, error_message) - bbox is None if validation fails

    This prevents NoneType multiplication errors by catching invalid bboxes
    at call sites before they reach _prepare_face_crop's margin calculations.
    """
    if bbox is None:
        return None, "bbox_is_none"

    try:
        # Convert to list if numpy array
        if isinstance(bbox, np.ndarray):
            bbox_list = bbox.tolist()
        else:
            bbox_list = list(bbox)

        # Ensure exactly 4 coordinates
        if len(bbox_list) != 4:
            return None, f"bbox_wrong_length_{len(bbox_list)}"

        # Validate each coordinate is a finite number
        for i, coord in enumerate(bbox_list):
            if coord is None:
                return None, f"bbox_coord_{i}_is_none"

            try:
                val = float(coord)
                if not np.isfinite(val):
                    return None, f"bbox_coord_{i}_not_finite_{val}"
            except (TypeError, ValueError) as e:
                return None, f"bbox_coord_{i}_invalid_{e}"

        # Return validated bbox as list of floats
        return [float(c) for c in bbox_list], None

    except Exception as e:
        return None, f"bbox_validation_error_{e}"


class VideoValidationError(Exception):
    """Raised when video file validation fails."""

    def __init__(self, path: Path, message: str, details: dict | None = None):
        self.path = path
        self.details = details or {}
        super().__init__(f"Video validation failed for {path}: {message}")


def validate_video_file(
    video_path: Path,
    check_first_frame: bool = True,
    min_frames: int = 1,
    min_duration_sec: float = 0.0,
) -> tuple[bool, str | None, dict]:
    """Validate video file is readable and has valid content.

    Args:
        video_path: Path to the video file
        check_first_frame: If True, attempt to read first frame
        min_frames: Minimum number of frames required
        min_duration_sec: Minimum duration in seconds

    Returns:
        Tuple of (is_valid, error_message, video_info):
        - is_valid: True if validation passed
        - error_message: None if valid, otherwise specific error message
        - video_info: Dict with video properties (width, height, fps, frame_count, duration)
    """
    video_info: dict = {"path": str(video_path)}

    if not video_path.exists():
        return False, f"Video file not found: {video_path}", video_info

    # Try to get file size
    try:
        video_info["file_size_bytes"] = video_path.stat().st_size
        if video_info["file_size_bytes"] == 0:
            return False, "Video file is empty (0 bytes)", video_info
    except OSError as e:
        return False, f"Cannot read video file: {e}", video_info

    cap = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "Cannot open video file – file may be corrupted or use unsupported codec", video_info

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_info.update({
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
        })

        # Validate dimensions
        if width <= 0 or height <= 0:
            return False, f"Invalid video dimensions: {width}x{height}", video_info

        # Validate FPS
        if fps <= 0:
            return False, f"Invalid video FPS: {fps}", video_info

        # Validate frame count
        if frame_count < min_frames:
            return False, f"Video has only {frame_count} frames, minimum {min_frames} required", video_info

        # Calculate duration
        duration_sec = frame_count / fps if fps > 0 else 0.0
        video_info["duration_sec"] = duration_sec

        if duration_sec < min_duration_sec:
            return False, f"Video duration {duration_sec:.1f}s is less than minimum {min_duration_sec:.1f}s", video_info

        # Try to read first frame if requested
        if check_first_frame:
            ok, first_frame = cap.read()
            if not ok or first_frame is None:
                return False, "Cannot read first frame – video may be corrupted", video_info
            if first_frame.size == 0:
                return False, "First frame is empty – video may be corrupted", video_info
            video_info["first_frame_readable"] = True

        return True, None, video_info

    except cv2.error as e:
        return False, f"OpenCV error reading video: {e}", video_info
    except Exception as e:
        return False, f"Unexpected error validating video: {e}", video_info
    finally:
        if cap is not None:
            cap.release()


def require_valid_video(
    video_path: Path,
    check_first_frame: bool = True,
    min_frames: int = 1,
    min_duration_sec: float = 0.0,
) -> dict:
    """Validate video file and raise VideoValidationError if invalid.

    Args:
        video_path: Path to the video file
        check_first_frame: If True, attempt to read first frame
        min_frames: Minimum number of frames required
        min_duration_sec: Minimum duration in seconds

    Returns:
        Video info dict with properties

    Raises:
        VideoValidationError: If validation fails
    """
    is_valid, error_msg, video_info = validate_video_file(
        video_path, check_first_frame, min_frames, min_duration_sec
    )
    if not is_valid:
        raise VideoValidationError(video_path, error_msg or "Unknown validation error", video_info)
    return video_info


def _prepare_face_crop(
    image,
    bbox: list[float],
    landmarks: list[float] | None,
    margin: float = 0.20,
    *,
    align: bool = True,
    detector_mode: str = "retinaface",
    adaptive_margin: bool = True,
) -> tuple[np.ndarray | None, str | None]:
    """Prepare face crop with optional adaptive margins.

    Args:
        image: Source image
        bbox: Face bounding box [x1, y1, x2, y2]
        landmarks: Optional facial landmarks
        margin: Base margin ratio (default 0.15 = 15%)
        align: Whether to use landmark-based alignment
        detector_mode: Detector name for mode-specific handling
        adaptive_margin: Scale margin based on bbox size (default True)

    Returns:
        (cropped_image, error_message) tuple
    """
    import numpy as _np

    normalized_mode = (detector_mode or "retinaface").lower()
    # For simulated detector, use the bbox it computed (centered on brightest pixels)
    # instead of letterboxing the full image. This preserves the useful crop.
    # Fall through to bbox-based cropping logic below.

    if align and landmarks and len(landmarks) >= 10 and normalized_mode != "simulated":
        pts = _np.asarray(landmarks, dtype=_np.float32).reshape(-1, 2)
        # Some detectors (simulated RetinaFace fallback) emit duplicated
        # landmarks, which breaks the InsightFace alignment helper and results
        # in uniform crops. Require a minimum spread before attempting
        # landmark-based alignment.
        if _np.all(_np.isfinite(pts)) and _np.max(_np.ptp(pts, axis=0)) >= 1.0:
            try:
                from insightface.utils import face_align  # type: ignore

                aligned = face_align.norm_crop(image, landmark=pts)
                return to_u8_bgr(aligned), None
            except Exception:
                # If alignment fails, fall back to bbox-based cropping.
                pass

    x1, y1, x2, y2 = bbox

    # Validate and convert bbox coordinates to floats
    try:
        x1 = float(x1) if x1 is not None else None
        y1 = float(y1) if y1 is not None else None
        x2 = float(x2) if x2 is not None else None
        y2 = float(y2) if y2 is not None else None
    except (TypeError, ValueError) as e:
        return None, f"invalid_bbox_type_{e}"

    # Check for None values after conversion attempt
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None, f"invalid_bbox_none_values_{x1}_{y1}_{x2}_{y2}"

    # Compute dimensions
    try:
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
    except (TypeError, ValueError) as e:
        return None, f"invalid_bbox_coordinates_{e}"

    # Validate margin factor to prevent None multiplication errors
    try:
        margin = max(float(margin), 0.0)
    except (TypeError, ValueError):
        margin = 0.15  # Default fallback

    # Adaptive margin: smaller faces get more margin to capture context
    # Larger faces get less margin to avoid over-cropping
    if adaptive_margin:
        bbox_area = width * height
        # Scale factor: smaller faces (< 5000 px²) get +20% margin
        #               medium faces (5000-15000 px²) get base margin
        #               larger faces (> 15000 px²) get -20% margin
        if bbox_area < 5000:
            margin_scale = 1.2  # +20% for small faces
        elif bbox_area > 15000:
            margin_scale = 0.8  # -20% for large faces
        else:
            margin_scale = 1.0  # base margin for medium faces
        effective_margin = margin * margin_scale
    else:
        effective_margin = margin

    expand_x = width * effective_margin
    expand_y = height * effective_margin
    expanded_box = [
        x1 - expand_x,
        y1 - expand_y,
        x2 + expand_x,
        y2 + expand_y,
    ]
    crop, _, err = safe_crop(image, expanded_box)
    if crop is None:
        return None, err or "crop_failed"
    return crop, None


def _estimate_blur_score(image) -> float:
    import cv2  # type: ignore

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(variance)


def _make_skip_face_row(
    ep_id: str,
    track_id: int,
    frame_idx: int,
    ts_val: float,
    bbox: list[float],
    detector_choice: str,
    reason: str,
    *,
    crop_rel_path: str | None = None,
    crop_s3_key: str | None = None,
    thumb_rel_path: str | None = None,
    thumb_s3_key: str | None = None,
    # Structured skip data for smarter rescue decisions
    blur_score: float | None = None,
    confidence: float | None = None,
    contrast: float | None = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "ep_id": ep_id,
        "face_id": f"face_{track_id:04d}_{frame_idx:06d}",
        "track_id": track_id,
        "frame_idx": frame_idx,
        "ts": ts_val,
        "bbox_xyxy": bbox,
        "detector": detector_choice,
        "pipeline_ver": PIPELINE_VERSION,
        "skip": reason,
    }
    # Store structured quality metrics for smarter rescue decisions
    # This allows rescue logic to pick the "best" skipped face
    skip_data: Dict[str, Any] = {"reason": reason.split(":")[0] if ":" in reason else reason}
    if blur_score is not None:
        skip_data["blur_score"] = round(blur_score, 2)
    if confidence is not None:
        skip_data["confidence"] = round(confidence, 3)
    if contrast is not None:
        skip_data["contrast"] = round(contrast, 2)
    row["skip_data"] = skip_data

    if crop_rel_path:
        row["crop_rel_path"] = crop_rel_path
    if crop_s3_key:
        row["crop_s3_key"] = crop_s3_key
    if thumb_rel_path:
        row["thumb_rel_path"] = thumb_rel_path
    if thumb_s3_key:
        row["thumb_s3_key"] = thumb_s3_key
    return row


def _resolved_max_gap(configured_gap: int, analyzed_fps: float | None) -> int:
    configured = max(1, int(configured_gap))
    if analyzed_fps and analyzed_fps > 0 and TRACK_MAX_GAP_SEC > 0:
        cadence_cap = int(max(1, round(analyzed_fps * TRACK_MAX_GAP_SEC)))
        return max(1, min(configured, cadence_cap))
    return configured


class TrackRecorder:
    """Maintains exported track ids, metrics, and sampled boxes."""

    def __init__(self, *, max_gap: int, remap_ids: bool) -> None:
        self.max_gap = max(1, int(max_gap))
        self.remap_ids = remap_ids
        self._next_export_id = 1
        self._mapping: dict[int, dict[str, int]] = {}
        self._active_exports: set[int] = set()
        self._accumulators: dict[int, TrackAccumulator] = {}
        self.metrics = {
            "tracks_born": 0,
            "tracks_lost": 0,
            "id_switches": 0,
            "forced_splits": 0,
        }

    def _spawn_export_id(self) -> int:
        export_id = self._next_export_id
        self._next_export_id += 1
        self._active_exports.add(export_id)
        self.metrics["tracks_born"] += 1
        return export_id

    def _complete_track(self, export_id: int) -> None:
        if export_id in self._active_exports:
            self._active_exports.remove(export_id)
            self.metrics["tracks_lost"] += 1

    def record(
        self,
        *,
        tracker_track_id: int,
        frame_idx: int,
        ts: float,
        bbox: list[float] | np.ndarray,
        class_label: int | str,
        landmarks: list[float] | None = None,
        confidence: float | None = None,
        force_new_track: bool = False,
    ) -> int:
        if isinstance(bbox, np.ndarray):
            bbox_values = bbox.tolist()
        else:
            bbox_values = bbox
        export_id: int
        mapping = self._mapping.get(tracker_track_id)
        if self.remap_ids:
            start_new = mapping is None
            if mapping is not None:
                gap = frame_idx - mapping.get("last_frame", frame_idx)
                if gap > self.max_gap:
                    self.metrics["id_switches"] += 1
                    self._complete_track(mapping["export_id"])
                    start_new = True
                elif force_new_track:
                    self.metrics["forced_splits"] += 1
                    self._complete_track(mapping["export_id"])
                    start_new = True
            if start_new:
                export_id = self._spawn_export_id()
            else:
                export_id = mapping["export_id"]
            self._mapping[tracker_track_id] = {
                "export_id": export_id,
                "last_frame": frame_idx,
            }
        else:
            if mapping is None:
                export_id = tracker_track_id
                self._active_exports.add(export_id)
                self._mapping[tracker_track_id] = {
                    "export_id": export_id,
                    "last_frame": frame_idx,
                }
                self.metrics["tracks_born"] += 1
            else:
                export_id = mapping["export_id"]
                mapping["last_frame"] = frame_idx
        track = self._accumulators.get(export_id)
        if track is None:
            track = TrackAccumulator(track_id=export_id, class_id=class_label, first_ts=ts, last_ts=ts)
            self._accumulators[export_id] = track
        track.add(ts, frame_idx, bbox_values, confidence=confidence, landmarks=landmarks)
        return export_id

    def finalize(self) -> None:
        for export_id in list(self._active_exports):
            self._complete_track(export_id)
        self._mapping.clear()

    def on_cut(self, frame_idx: int | None = None) -> None:
        """Force-complete all active exports so new IDs spawn after a hard cut."""

        if not self._mapping:
            return
        forced = 0
        for mapping in list(self._mapping.values()):
            self._complete_track(mapping["export_id"])
            forced += 1
        self._mapping.clear()
        if forced:
            self.metrics["forced_splits"] += forced

    def rows(self) -> list[dict]:
        payload: list[dict] = []
        for track in sorted(self._accumulators.values(), key=lambda item: item.track_id):
            payload.append(track.to_row())
        return payload

    @property
    def active_track_count(self) -> int:
        return len(self._active_exports)

    def top_long_tracks(self, limit: int = 5) -> list[dict]:
        longest = sorted(self._accumulators.values(), key=lambda item: item.frame_count, reverse=True)[:limit]
        return [
            {"track_id": track.track_id, "frame_count": track.frame_count} for track in longest if track.frame_count > 0
        ]


def _bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def _try_import(module: str):
    try:
        return __import__(module)
    except ImportError:
        return None


def sanitize_xyxy(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Round + clamp XYXY boxes to integer pixel coordinates, skipping empty windows."""
    if width <= 0 or height <= 0:
        return None
    x1_int = int(max(0, min(round(x1), width - 1)))
    y1_int = int(max(0, min(round(y1), height - 1)))
    x2_int = int(max(0, min(round(x2), width)))
    y2_int = int(max(0, min(round(y2), height)))
    if x2_int <= x1_int or y2_int <= y1_int:
        return None
    return x1_int, y1_int, x2_int, y2_int


def _image_stats(image) -> tuple[float, float, float]:
    arr = np.asarray(image)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.nanmin(arr)), float(np.nanmax(arr)), float(np.nanmean(arr))


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.size == 0:
        return arr
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))
        if mx <= 1.0 and mn >= 0.0:
            arr = (arr * 255.0).round()
        elif mn >= -1.0 and mx <= 1.0:
            arr = ((arr + 1.0) * 127.5).round()
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        return np.clip(arr, 0, 255).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.integer):
        arr = np.clip(arr.astype(np.int64), 0, 255)
        return arr.astype(np.uint8)
    return arr.astype(np.uint8, copy=False)


def encode_jpeg_bytes(image, *, quality: int = 85, color: str = "bgr") -> bytes:
    """Encode an image to JPEG bytes without writing to disk.

    Args:
        image: Input image (numpy array or array-like)
        quality: JPEG quality (1-100)
        color: Input color space ("bgr" or "rgb")

    Returns:
        JPEG-encoded bytes
    """
    import cv2  # type: ignore

    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("Cannot encode empty image")
    arr = np.ascontiguousarray(_normalize_to_uint8(arr))
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        mode = color.lower()
        if mode == "rgb":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif mode not in {"bgr", "rgb"}:
            raise ValueError(f"Unsupported color mode '{color}'")
    else:
        raise ValueError(f"Unsupported image shape for JPEG encode: {arr.shape}")
    arr = np.ascontiguousarray(arr)
    jpeg_quality = max(1, min(int(quality or 85), 100))
    success, encoded = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


def encode_png_bytes(image, *, compression: int = 3, color: str = "bgr") -> bytes:
    """Encode an image to PNG bytes without writing to disk (lossless).

    Args:
        image: Input image (numpy array or array-like)
        compression: PNG compression level (0-9, default 3)
        color: Input color space ("bgr" or "rgb")

    Returns:
        PNG-encoded bytes
    """
    import cv2  # type: ignore

    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("Cannot encode empty image")
    arr = np.ascontiguousarray(_normalize_to_uint8(arr))
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        mode = color.lower()
        if mode == "rgb":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif mode not in {"bgr", "rgb"}:
            raise ValueError(f"Unsupported color mode '{color}'")
    else:
        raise ValueError(f"Unsupported image shape for PNG encode: {arr.shape}")
    arr = np.ascontiguousarray(arr)
    compression = max(0, min(int(compression), 9))
    success, encoded = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


def save_png(path: str | Path, image, *, compression: int = 3, color: str = "bgr") -> None:
    """Normalize + persist an image to PNG (lossless), ensuring non-blank uint8 BGR data."""
    import cv2  # type: ignore

    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError(f"Cannot save empty image to {path}")
    arr = np.ascontiguousarray(_normalize_to_uint8(arr))
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        mode = color.lower()
        if mode == "rgb":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif mode not in {"bgr", "rgb"}:
            raise ValueError(f"Unsupported color mode '{color}'")
    else:
        raise ValueError(f"Unsupported image shape for PNG write: {arr.shape}")
    arr = np.ascontiguousarray(arr)
    compression = max(0, min(int(compression), 9))
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), arr, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {out_path}")


def save_jpeg(path: str | Path, image, *, quality: int = 85, color: str = "bgr") -> None:
    """Normalize + persist an image to JPEG, ensuring non-blank uint8 BGR data."""
    import cv2  # type: ignore

    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError(f"Cannot save empty image to {path}")
    arr = np.ascontiguousarray(_normalize_to_uint8(arr))
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        mode = color.lower()
        if mode == "rgb":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif mode not in {"bgr", "rgb"}:
            raise ValueError(f"Unsupported color mode '{color}'")
    else:
        raise ValueError(f"Unsupported image shape for JPEG write: {arr.shape}")
    arr = np.ascontiguousarray(arr)
    jpeg_quality = max(1, min(int(quality or 85), 100))
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), arr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {out_path}")


class ThumbWriter:
    """Writes thumbnails for tracks, optionally using storage backend.

    Supports two modes:
    - Direct filesystem writes (default, for local/hybrid backends)
    - Storage backend writes (when storage_backend is provided and backend_type is 's3')

    For hybrid storage mode, thumbnails are written locally first for speed,
    then synced to S3 after job completion via the storage backend's sync_to_s3().
    """

    def __init__(
        self,
        ep_id: str,
        size: int = 256,
        jpeg_quality: int = 85,
        storage_backend=None,
        use_png: bool = True,  # Default to PNG for maximum quality
        png_compression: int = 3,
        run_id: str | None = None,
    ) -> None:
        self.ep_id = ep_id
        self.size = size
        self.jpeg_quality = max(1, min(int(jpeg_quality or 85), 100))
        self.use_png = use_png
        self.png_compression = max(0, min(int(png_compression), 9))
        self.root_dir = _frames_root_for_run(ep_id, run_id) / "thumbs"
        self._storage_backend = storage_backend
        # Determine if we should use backend for writes (only for pure S3 mode)
        self._use_backend_writes = (
            storage_backend is not None
            and hasattr(storage_backend, "backend_type")
            and storage_backend.backend_type == "s3"
        )
        # For local/hybrid modes, still create local directories
        if not self._use_backend_writes:
            self.root_dir.mkdir(parents=True, exist_ok=True)
        self._stat_samples = 0
        self._stat_limit = 10
        try:
            import cv2  # type: ignore

            self._cv2 = cv2
        except ImportError:
            self._cv2 = None
        self._last_thumb_meta: Dict[str, Any] = {}
        # Track bytes written for progress reporting
        self.bytes_written = 0
        self.thumbs_written = 0

    def write(
        self,
        image,
        bbox: List[float],
        track_id: int,
        frame_idx: int,
        *,
        prepared_crop: np.ndarray | None = None,
    ) -> tuple[str | None, Path | None]:
        self._last_thumb_meta = {"source_shape": None, "source_kind": None}
        if self._cv2 is None or image is None:
            return None, None
        source = None
        source_kind = None
        if prepared_crop is not None:
            arr = np.asarray(prepared_crop)
            if arr.size > 0:
                source = prepared_crop
                source_kind = "prepared"
        if source is None:
            crop, clipped_bbox, err = safe_crop(image, bbox)
            if crop is None:
                LOGGER.debug(
                    "Skipping thumb track=%s frame=%s reason=%s",
                    track_id,
                    frame_idx,
                    err,
                )
                return None, None
            source = crop
            source_kind = "fallback"
        arr = np.asarray(source)
        if arr.size == 0:
            return None, None
        variance = float(np.var(arr))
        if variance < FACE_MIN_STD:
            LOGGER.debug(
                "Skipping thumb track=%s frame=%s low_variance=%.4f source=%s",
                track_id,
                frame_idx,
                variance,
                source_kind,
            )
            return None, None
        thumb = self._letterbox(source)
        if self._stat_samples < self._stat_limit:
            mn, mx, mean = _image_stats(thumb)
            LOGGER.info(
                "thumb stats track=%s frame=%s min=%.3f max=%.3f mean=%.3f",
                track_id,
                frame_idx,
                mn,
                mx,
                mean,
            )
            if mx - mn < 1e-6:
                LOGGER.warning("Nearly constant thumb track=%s frame=%s", track_id, frame_idx)
            self._stat_samples += 1
        ext = ".png" if self.use_png else ".jpg"
        rel_path = Path(f"track_{track_id:04d}/thumb_{frame_idx:06d}{ext}")
        abs_path = self.root_dir / rel_path

        # Write thumbnail using storage backend (S3) or direct filesystem
        if self._use_backend_writes and self._storage_backend is not None:
            # Direct-to-S3 mode: encode to bytes and upload
            try:
                if self.use_png:
                    img_data = encode_png_bytes(thumb, compression=self.png_compression, color="bgr")
                else:
                    img_data = encode_jpeg_bytes(thumb, quality=self.jpeg_quality, color="bgr")
                # entity_id includes track and frame info
                entity_id = f"track_{track_id:04d}/thumb_{frame_idx:06d}"
                result = self._storage_backend.write_thumbnail(
                    self.ep_id, "track", entity_id, img_data,
                    content_type="image/png" if self.use_png else "image/jpeg",
                )
                if result.success:
                    self.thumbs_written += 1
                    self.bytes_written += len(img_data)
                    self._last_thumb_meta = {
                        "source_shape": tuple(int(x) for x in source.shape[:2]),
                        "source_kind": source_kind,
                    }
                    return rel_path.as_posix(), None  # No local path in S3 mode
                else:
                    LOGGER.warning("Failed to upload thumb %s: %s", rel_path, result.error)
                    return None, None
            except Exception as exc:
                LOGGER.warning("Failed to encode/upload thumb %s: %s", rel_path, exc)
                return None, None
        else:
            # Local/hybrid mode: write directly to disk
            ok, reason = safe_imwrite(
                abs_path, thumb, self.jpeg_quality,
                use_png=self.use_png, png_compression=self.png_compression
            )
            if not ok:
                LOGGER.warning("Failed to write thumb %s: %s", abs_path, reason)
                return None, None
            self.thumbs_written += 1
            if abs_path.exists():
                self.bytes_written += abs_path.stat().st_size
            self._last_thumb_meta = {
                "source_shape": tuple(int(x) for x in source.shape[:2]),
                "source_kind": source_kind,
            }
            return rel_path.as_posix(), abs_path

    def _letterbox(self, crop):
        if self._cv2 is None:
            return np.zeros((self.size, self.size, 3), dtype=np.uint8)
        if crop is None or crop.size == 0:
            return np.zeros((self.size, self.size, 3), dtype=np.uint8)
        h, w = crop.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((self.size, self.size, 3), dtype=np.uint8)
        scale = min(self.size / max(w, 1), self.size / max(h, 1))
        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)
        resized = self._cv2.resize(crop, (new_w, new_h))
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        top = (self.size - new_h) // 2
        left = (self.size - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized
        return canvas


def _faces_embed_path(ep_id: str, run_id: str | None = None) -> Path:
    embed_dir = DATA_ROOT / "embeds" / ep_id
    if run_id:
        embed_dir = embed_dir / "runs" / run_layout.normalize_run_id(run_id)
    embed_dir.mkdir(parents=True, exist_ok=True)
    return embed_dir / "faces.npy"


class ProgressEmitter:
    """Emit structured progress to stdout + optional file for SSE/polling.

    Supports two modes for file persistence:
    - Direct filesystem writes (default, for local/hybrid backends)
    - Storage backend writes (when storage_backend is provided)

    The storage backend approach allows progress to be stored in S3 or other
    backends for distributed systems where local filesystem isn't shared.
    """

    VERSION = 3

    def __init__(
        self,
        ep_id: str,
        file_path: str | Path | None,
        *,
        frames_total: int,
        secs_total: float | None,
        stride: int,
        fps_detected: float | None,
        fps_requested: float | None,
        frame_interval: int | None = None,
        run_id: str | None = None,
        storage_backend=None,
    ) -> None:
        import uuid

        self.ep_id = ep_id
        self.run_id = run_id or str(uuid.uuid4())
        self.path = Path(file_path).expanduser() if file_path else None
        self._storage_backend = storage_backend
        # Use storage backend for progress writes only if S3-only mode
        self._use_backend_writes = (
            storage_backend is not None
            and hasattr(storage_backend, "backend_type")
            and storage_backend.backend_type == "s3"
        )
        # Still create local directory for local/hybrid modes
        if self.path and not self._use_backend_writes:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames_total = max(int(frames_total or 0), 0)
        self.secs_total = float(secs_total) if secs_total else None
        self.stride = max(int(stride), 1)
        self.fps_detected = float(fps_detected) if fps_detected else None
        self.fps_requested = float(fps_requested) if fps_requested else None
        default_interval = PROGRESS_FRAME_STEP
        chosen_interval = frame_interval if frame_interval is not None else default_interval
        self._frame_interval = max(int(chosen_interval), 1)
        self._start_ts = time.time()
        self._last_frames = 0
        self._last_phase: str | None = None
        self._last_step: str | None = None
        self._last_emit_time: float = 0.0  # Time of last emission for time-based fallback
        self._device: str | None = None
        self._detector: str | None = None
        self._tracker: str | None = None
        self._resolved_device: str | None = None
        self._closed = False

    def _now(self) -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def _should_emit(self, frames_done: int, phase: str, step: str | None, force: bool) -> bool:
        if force:
            return True
        if phase != self._last_phase:
            return True
        if step != self._last_step:
            return True
        # Frame-based check
        if (frames_done - self._last_frames) >= self._frame_interval:
            return True
        # Time-based fallback: emit at least every PROGRESS_TIME_INTERVAL seconds
        if (time.time() - self._last_emit_time) >= PROGRESS_TIME_INTERVAL:
            return True
        return False

    def _compose_payload(
        self,
        frames_done: int,
        phase: str,
        device: str | None,
        summary: Dict[str, object] | None,
        error: str | None,
        detector: str | None,
        tracker: str | None,
        resolved_device: str | None,
        extra: Dict[str, Any] | None = None,
    ) -> Dict[str, object]:
        secs_done = time.time() - self._start_ts
        fps_infer = None
        if secs_done > 0 and frames_done >= 0:
            fps_infer = frames_done / secs_done

        # Scene detection phases should NOT inherit face detector/tracker values
        # Scene detector is only in summary, not top-level fields
        is_scene_phase = phase.startswith("scene_detect")
        detector_value = None if is_scene_phase else (detector or self._detector)
        tracker_value = None if is_scene_phase else (tracker or self._tracker)
        resolved_device_value = None if is_scene_phase else (resolved_device or self._resolved_device)

        payload: Dict[str, object] = {
            "progress_version": self.VERSION,
            "ep_id": self.ep_id,
            "run_id": self.run_id,
            "phase": phase,
            "frames_done": frames_done,
            "frames_total": self.frames_total,
            "secs_done": round(float(secs_done), 3),
            "secs_total": round(float(self.secs_total), 3) if self.secs_total else None,
            "device": device or self._device,
            "fps_infer": round(float(fps_infer), 3) if fps_infer else None,
            "fps_detected": (round(float(self.fps_detected), 3) if self.fps_detected else None),
            "fps_requested": (round(float(self.fps_requested), 3) if self.fps_requested else None),
            "stride": self.stride,
            "updated_at": self._now(),
            "detector": detector_value,
            "tracker": tracker_value,
            "resolved_device": resolved_device_value,
        }
        if summary:
            payload["summary"] = summary
        if error:
            payload["error"] = error
        if extra:
            payload.update(extra)
        return payload

    def _write_payload(self, payload: Dict[str, object]) -> None:
        line = json.dumps(payload, sort_keys=True)
        print(line, flush=True)

        # Structured logging for episode-wide grep
        phase = payload.get("phase", "")
        step = payload.get("step", "")
        frames = payload.get("frames_done", 0)
        total = payload.get("frames_total", 0)
        vt = payload.get("video_time")
        vtotal = payload.get("video_total")
        fps = payload.get("fps_infer")
        run_id_short = self.run_id[:8] if self.run_id else "unknown"

        if vt is not None and vtotal is not None:
            log_msg = (
                f"[job={self.ep_id} run={run_id_short} phase={phase} step={step} "
                f"frames={frames}/{total} vt={vt:.1f}/{vtotal:.1f} fps={fps or 0.0:.2f}]"
            )
            LOGGER.info(log_msg)
            # Print to stdout for local mode streaming
            if LOCAL_MODE_INSTRUMENTATION:
                print(log_msg, flush=True)
        else:
            log_msg = (
                f"[job={self.ep_id} run={run_id_short} phase={phase} step={step} "
                f"frames={frames}/{total} fps={fps or 0.0:.2f}]"
            )
            LOGGER.info(log_msg)
            # Print to stdout for local mode streaming
            if LOCAL_MODE_INSTRUMENTATION:
                print(log_msg, flush=True)

        stage_key = _log_stage_for_phase(str(phase))
        if stage_key and self.run_id:
            progress_pct = None
            if isinstance(total, (int, float)) and total > 0:
                try:
                    progress_pct = round((float(frames) / float(total)) * 100.0, 2)
                except (TypeError, ValueError, ZeroDivisionError):
                    progress_pct = None
            message = payload.get("message")
            if not isinstance(message, str) or not message.strip():
                message = f"{phase} progress"
            try:
                append_log(
                    self.ep_id,
                    self.run_id,
                    stage_key,
                    "INFO",
                    message,
                    progress=progress_pct,
                    meta={
                        "phase": phase,
                        "step": step,
                        "frames_done": frames,
                        "frames_total": total,
                    },
                )
            except Exception as exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to append progress log: %s", exc)

        # Persist progress to storage backend or local filesystem
        if self._use_backend_writes and self._storage_backend is not None:
            # S3-only mode: write progress via storage backend
            try:
                self._storage_backend.write_progress(self.ep_id, dict(payload))
            except Exception as exc:
                LOGGER.warning("Failed to write progress to storage backend: %s", exc)
        elif self.path:
            # Local/hybrid mode: write directly to filesystem
            tmp_path = self.path.with_suffix(".tmp")
            tmp_path.write_text(line, encoding="utf-8")
            tmp_path.replace(self.path)

    def emit(
        self,
        frames_done: int,
        *,
        phase: str,
        device: str | None = None,
        summary: Dict[str, object] | None = None,
        error: str | None = None,
        force: bool = False,
        detector: str | None = None,
        tracker: str | None = None,
        resolved_device: str | None = None,
        extra: Dict[str, Any] | None = None,
        **fields: Any,
    ) -> None:
        if self._closed:
            return
        frames_done = max(int(frames_done), 0)
        if self.frames_total and frames_done > self.frames_total:
            frames_done = self.frames_total

        # Extract step from extra dict if present
        step = None
        if extra and "step" in extra:
            step = extra.get("step")

        if not self._should_emit(frames_done, phase, step, force):
            return
        if device is not None:
            self._device = device
        if detector is not None:
            self._detector = detector
        if tracker is not None:
            self._tracker = tracker
        if resolved_device is not None:
            self._resolved_device = resolved_device
        combined_extra: Dict[str, Any] = {} if extra is None else dict(extra)
        if fields:
            combined_extra.update(fields)
        payload = self._compose_payload(
            frames_done,
            phase,
            device,
            summary,
            error,
            detector,
            tracker,
            resolved_device,
            combined_extra or None,
        )
        self._write_payload(payload)
        self._last_frames = frames_done
        self._last_phase = phase
        self._last_step = step
        self._last_emit_time = time.time()

    def complete(
        self,
        summary: Dict[str, object],
        device: str | None = None,
        detector: str | None = None,
        tracker: str | None = None,
        resolved_device: str | None = None,
        *,
        step: str | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        final_frames = self.frames_total or summary.get("frames_sampled") or self._last_frames
        final_frames = int(final_frames or 0)
        completion_extra: Dict[str, Any] = {} if extra is None else dict(extra)
        if step:
            completion_extra["step"] = step
        self.emit(
            final_frames,
            phase="done",
            device=device,
            summary=summary,
            force=True,
            detector=detector,
            tracker=tracker,
            resolved_device=resolved_device,
            extra=completion_extra or None,
        )

    def fail(self, error: str) -> None:
        self.emit(
            self._last_frames,
            phase="error",
            error=error,
            force=True,
            tracker=self._tracker,
        )

    @property
    def target_frames(self) -> int:
        return self.frames_total or 0

    def close(self) -> None:
        self._closed = True


def _crop_diag_meta(source: Any | None) -> Dict[str, Any]:
    if source is None:
        return {}
    payload: Dict[str, Any] = {}
    attempts = getattr(source, "_crop_attempts", None)
    if attempts is not None:
        payload["crop_attempts"] = int(attempts)
    counts = getattr(source, "_crop_error_counts", None)
    if counts is not None:
        try:
            mapped = {str(key): int(val) for key, val in dict(counts).items()}
        except Exception:
            mapped = None
        if mapped is not None:
            payload["crop_errors"] = mapped
    return payload


def _non_video_phase_meta(
    step: str | None = None,
    *,
    crop_diag_source: Any | None = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"video_time": None, "video_total": None}
    if step:
        meta["step"] = step
    meta.update(_crop_diag_meta(crop_diag_source))
    return meta


def _video_phase_meta(
    frames_done: int,
    frames_total: int | None,
    fps: float | None,
    step: str | None = None,
    *,
    crop_diag_source: Any | None = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if fps and fps > 0 and frames_total and frames_total > 0:
        video_total = frames_total / fps
        video_time = min(frames_done / fps, video_total)
        meta["video_total"] = round(video_total, 3)
        meta["video_time"] = round(video_time, 3)
    else:
        meta["video_time"] = None
        meta["video_total"] = None
    if step:
        meta["step"] = step
    meta.update(_crop_diag_meta(crop_diag_source))
    return meta


def _utcnow_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class StageStatusHeartbeat:
    ep_id: str
    run_id: str | None
    stage_key: str
    frames_total: int
    started_at: str
    heartbeat_interval: float = STAGE_HEARTBEAT_INTERVAL
    _last_tick: float = field(default_factory=time.time)
    frames_done_at: str | None = None
    finalize_started_at: str | None = None
    ended_at: str | None = None
    _enabled: bool = True

    def __post_init__(self) -> None:
        if not self.run_id:
            self._enabled = False
            return
        try:
            self.run_id = run_layout.normalize_run_id(self.run_id)
        except ValueError:
            self._enabled = False

    def update(
        self,
        *,
        done: int,
        phase: str,
        message: str | None = None,
        force: bool = False,
        mark_frames_done: bool = False,
        mark_finalize_start: bool = False,
        mark_end: bool = False,
    ) -> None:
        if not self._enabled:
            return
        now = time.time()
        if not force and (now - self._last_tick) < self.heartbeat_interval:
            return
        stamp = _utcnow_iso()
        if mark_frames_done and self.frames_done_at is None:
            self.frames_done_at = stamp
        if mark_finalize_start and self.finalize_started_at is None:
            self.finalize_started_at = stamp
        if mark_end:
            self.ended_at = stamp

        done_val = max(int(done or 0), 0)
        total_val = max(int(self.frames_total or 0), 0)
        pct = None
        if total_val > 0:
            pct = max(min(done_val / total_val, 1.0), 0.0)

        progress_payload = {
            "done": done_val,
            "total": total_val,
            "pct": pct,
            "phase": phase,
            "message": message,
            "last_update_at": stamp,
        }
        timestamps_payload = {
            "started_at": self.started_at,
            "frames_done_at": self.frames_done_at,
            "finalize_started_at": self.finalize_started_at,
            "ended_at": self.ended_at,
        }
        update_episode_status(
            self.ep_id,
            self.run_id,
            stage_key=self.stage_key,
            stage_update={"progress": progress_payload, "timestamps": timestamps_payload},
        )
        self._last_tick = now


def _run_with_heartbeat(
    action: Callable[[], Any],
    heartbeat: Callable[[], None],
    *,
    interval: float,
) -> Any:
    """Run a long action while emitting heartbeat updates at a fixed interval."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(action)
        while True:
            try:
                return future.result(timeout=interval)
            except TimeoutError:
                heartbeat()


def _manifests_dir_for_run(ep_id: str, run_id: str | None) -> Path:
    """Return the manifests dir for a run.

    Legacy layout (run_id is None):
        data/manifests/{ep_id}/

    Run-scoped layout:
        data/manifests/{ep_id}/runs/{run_id}/
    """
    root = get_path(ep_id, "detections").parent
    if not run_id:
        return root
    return run_layout.run_root(ep_id, run_layout.normalize_run_id(run_id))


def _tracks_path_for_run(ep_id: str, run_id: str | None) -> Path:
    if not run_id:
        return get_path(ep_id, "tracks")
    return _manifests_dir_for_run(ep_id, run_id) / "tracks.jsonl"


def _detections_path_for_run(ep_id: str, run_id: str | None) -> Path:
    if not run_id:
        return get_path(ep_id, "detections")
    return _manifests_dir_for_run(ep_id, run_id) / "detections.jsonl"


def _frames_root_for_run(ep_id: str, run_id: str | None) -> Path:
    """Return the frames root for a run (optional run-scoped layout).

    Legacy layout (run_id is None):
        data/frames/{ep_id}/

    Run-scoped layout:
        data/frames/{ep_id}/runs/{run_id}/
    """
    root = get_path(ep_id, "frames_root")
    if not run_id:
        return root
    return root / "runs" / run_layout.normalize_run_id(run_id)


def _body_tracking_dir_for_run(ep_id: str, run_id: str | None) -> Path:
    return _manifests_dir_for_run(ep_id, run_id) / "body_tracking"


def _promote_run_manifests_to_root(ep_id: str, run_id: str, filenames: Iterable[str]) -> None:
    """Copy run-scoped manifests into the legacy root manifests dir.

    This keeps backwards compatibility for tools/UI that still read from:
        data/manifests/{ep_id}/
    """
    run_id_norm = run_layout.normalize_run_id(run_id)
    src_dir = _manifests_dir_for_run(ep_id, run_id_norm)
    dest_dir = get_path(ep_id, "detections").parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    for name in filenames:
        cleaned = str(name).strip().lstrip("/\\")
        if not cleaned:
            continue
        src = src_dir / cleaned
        if not src.exists():
            continue
        dest = dest_dir / cleaned
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tmp)
        tmp.replace(dest)


_GIT_STATE_CACHE: dict[str, Any] | None = None


def _git_state() -> dict[str, Any]:
    global _GIT_STATE_CACHE
    if _GIT_STATE_CACHE is None:
        _GIT_STATE_CACHE = collect_git_state()
    return dict(_GIT_STATE_CACHE)


def _read_json_best_effort(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _delete_local_after_sync() -> bool:
    return os.environ.get("STORAGE_DELETE_LOCAL_AFTER_SYNC", "").strip().lower() in {"1", "true", "yes", "on"}


def _storage_summary_for_status(ep_id: str, run_id: str) -> dict[str, Any]:
    run_root = run_layout.run_root(ep_id, run_layout.normalize_run_id(run_id))
    allowlist = sorted(run_layout.RUN_ARTIFACT_ALLOWLIST)
    local_present = [name for name in allowlist if (run_root / name).exists()]
    summary: dict[str, Any] = {
        "delete_local_after_sync": _delete_local_after_sync(),
        "hydrated_from_s3": False,
        "local_present": {"count": len(local_present), "total": len(allowlist)},
        "remote_present": {"count": None, "total": len(allowlist), "checked": False},
    }
    try:
        storage = StorageService()
        if storage.s3_enabled():
            remote_present = 0
            for filename in allowlist:
                for s3_key in run_layout.run_artifact_s3_keys_for_read(ep_id, run_id, filename):
                    try:
                        if storage.object_exists(s3_key):
                            remote_present += 1
                            break
                    except Exception:
                        continue
            summary["remote_present"] = {
                "count": remote_present,
                "total": len(allowlist),
                "checked": True,
            }
    except Exception:
        pass
    return summary


def _env_payload_for_status(run_root: Path, marker_payload: dict[str, Any]) -> dict[str, Any]:
    env_payload: dict[str, Any] = {}
    env_diag = _read_json_best_effort(run_root / "env_diagnostics.json")
    if isinstance(env_diag, dict):
        env_payload["python_version"] = env_diag.get("python_version")
        env_payload["package_versions"] = env_diag.get("package_versions")
    torch_device = (
        marker_payload.get("torch_device_resolved")
        or marker_payload.get("resolved_device")
        or marker_payload.get("device")
    )
    onnx_provider = marker_payload.get("onnx_provider_resolved")
    if torch_device:
        env_payload["torch_device"] = torch_device
    if onnx_provider:
        env_payload["onnx_provider"] = onnx_provider
    env_payload["db_url_present"] = bool(
        os.environ.get("DB_URL") or os.environ.get("SCREENALYTICS_FAKE_DB")
    )
    return env_payload


def _update_episode_status_from_marker(
    ep_id: str,
    run_id: str,
    phase: str,
    marker_payload: dict[str, Any],
) -> None:
    try:
        stage_key, stage_update = stage_update_from_marker(
            ep_id=ep_id,
            run_id=run_id,
            phase=phase,
            marker_payload=marker_payload,
        )
        if not stage_key:
            return
        run_root = run_layout.run_root(ep_id, run_layout.normalize_run_id(run_id))
        update_episode_status(
            ep_id,
            run_id,
            stage_key=stage_key,
            stage_update=stage_update,
            git_info=_git_state(),
            env=_env_payload_for_status(run_root, marker_payload),
            storage=_storage_summary_for_status(ep_id, run_id),
        )
    except Exception as exc:
        LOGGER.debug("[episode_status] Failed to update: %s", exc)


def _write_run_marker(
    ep_id: str,
    phase: str,
    payload: Dict[str, Any],
    *,
    run_id: str | None = None,
) -> None:
    manifests_root = get_path(ep_id, "detections").parent
    runs_root = manifests_root / RUN_MARKERS_SUBDIR
    runs_root.mkdir(parents=True, exist_ok=True)

    marker_payload = dict(payload)
    if run_id:
        marker_payload["run_id"] = run_layout.normalize_run_id(run_id)
    else:
        # Avoid leaking ProgressEmitter.run_id into legacy markers when the job
        # was run without an explicit run_id (legacy/unscoped mode).
        marker_payload.pop("run_id", None)
    for key, value in _git_state().items():
        if value is not None and key not in marker_payload:
            marker_payload[key] = value

    # Legacy marker path (kept for UI compatibility)
    marker_path = runs_root / f"{phase}.json"
    marker_path.write_text(json.dumps(marker_payload, indent=2), encoding="utf-8")

    # Run-scoped marker path (for history/debugging)
    if run_id:
        run_root = run_layout.run_root(ep_id, run_layout.normalize_run_id(run_id))
        run_root.mkdir(parents=True, exist_ok=True)
        run_marker_path = run_root / f"{phase}.json"
        run_marker_path.write_text(json.dumps(marker_payload, indent=2), encoding="utf-8")
        _update_episode_status_from_marker(ep_id, run_id, phase, marker_payload)


class CropQualityThresholdExceeded(RuntimeError):
    """Raised when crop exports fail at an unacceptable rate."""


class FrameExporter:
    """Handles optional frame + crop JPEG exports for S3 sync.

    Supports two modes:
    - Direct filesystem writes (default, for local/hybrid backends)
    - Storage backend writes (when storage_backend is provided and supports it)

    For hybrid storage mode, frames/crops are written locally first for speed,
    then synced to S3 after job completion via the storage backend's sync_to_s3().
    """

    def __init__(
        self,
        ep_id: str,
        *,
        run_id: str | None = None,
        save_frames: bool,
        save_crops: bool,
        jpeg_quality: int,
        debug_logger: JsonlLogger | NullLogger | None = None,
        storage_backend=None,
        use_png: bool = True,  # Default to PNG for maximum quality
        png_compression: int = 3,  # PNG compression (0-9)
    ) -> None:
        self.ep_id = ep_id
        self.save_frames = save_frames
        self.save_crops = save_crops
        self.jpeg_quality = max(1, min(int(jpeg_quality or 85), 100))
        self.use_png = use_png
        self.png_compression = max(0, min(int(png_compression), 9))
        self.root_dir = _frames_root_for_run(ep_id, run_id)
        self.frames_dir = self.root_dir / "frames"
        self.crops_dir = self.root_dir / "crops"
        self._storage_backend = storage_backend
        # Determine if we should use backend for writes (only for pure S3 mode)
        self._use_backend_writes = (
            storage_backend is not None
            and hasattr(storage_backend, "backend_type")
            and storage_backend.backend_type == "s3"
        )
        # For local/hybrid modes, still create local directories
        if not self._use_backend_writes:
            if self.save_frames:
                self.frames_dir.mkdir(parents=True, exist_ok=True)
            if self.save_crops:
                self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.frames_written = 0
        self.crops_written = 0
        self._track_indexes: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self._stat_samples = 0
        self._stat_limit = 10
        self._crop_attempts = 0
        self._crop_error_counts: Counter[str] = Counter()
        self._fail_fast_threshold = 0.40  # Require 40% failure rate before aborting
        self._fail_fast_min_attempts = 50  # Require at least 50 attempts before checking
        self._fail_fast_reasons = {"near_uniform_gray", "tiny_file"}
        self.debug_logger = debug_logger
        # Track bytes written for progress reporting
        self.bytes_written = 0

    def _log_image_stats(self, kind: str, path: Path, image) -> None:
        if self._stat_samples >= self._stat_limit:
            return
        mn, mx, mean = _image_stats(image)
        LOGGER.info("%s stats %s min=%.3f max=%.3f mean=%.3f", kind, path, mn, mx, mean)
        if mx - mn < 1e-6:
            LOGGER.warning(
                "Nearly constant %s %s mn=%.6f mx=%.6f mean=%.6f",
                kind,
                path,
                mn,
                mx,
                mean,
            )
        self._stat_samples += 1

    def export(
        self,
        frame_idx: int,
        image,
        crops: List[Tuple[int, List[float]]],
        ts: float | None = None,
    ) -> None:
        if not (self.save_frames or self.save_crops):
            return
        if self.save_frames:
            ext = ".png" if self.use_png else ".jpg"
            frame_path = self.frames_dir / f"frame_{frame_idx:06d}{ext}"
            try:
                self._log_image_stats("frame", frame_path, image)
                if self._use_backend_writes and self._storage_backend is not None:
                    # Direct-to-S3 mode: encode to bytes and upload
                    if self.use_png:
                        img_data = encode_png_bytes(image, compression=self.png_compression, color="bgr")
                    else:
                        img_data = encode_jpeg_bytes(image, quality=self.jpeg_quality, color="bgr")
                    result = self._storage_backend.write_frame(
                        self.ep_id, frame_idx, img_data,
                        content_type="image/png" if self.use_png else "image/jpeg",
                    )
                    if result.success:
                        self.frames_written += 1
                        self.bytes_written += len(img_data)
                    else:
                        LOGGER.warning("Failed to upload frame %d: %s", frame_idx, result.error)
                else:
                    # Local/hybrid mode: write directly to disk
                    if self.use_png:
                        save_png(frame_path, image, compression=self.png_compression, color="bgr")
                    else:
                        save_jpeg(frame_path, image, quality=self.jpeg_quality, color="bgr")
                    self.frames_written += 1
                    if frame_path.exists():
                        self.bytes_written += frame_path.stat().st_size
            except Exception as exc:  # pragma: no cover - best effort
                LOGGER.warning("Failed to save frame %s: %s", frame_path, exc)
        if self.save_crops and crops:
            for track_id, bbox in crops:
                if track_id is None:
                    continue
                crop_path = self.crop_abs_path(track_id, frame_idx)
                try:
                    saved = self._write_crop(image, bbox, crop_path, track_id, frame_idx)
                except CropQualityThresholdExceeded as exc:
                    LOGGER.error(
                        "Aborting crop exports for track %s frame %s after quality threshold: %s",
                        track_id,
                        frame_idx,
                        exc,
                    )
                    raise
                except Exception as exc:  # pragma: no cover - best effort
                    LOGGER.warning("Failed to save crop %s: %s", crop_path, exc)
                    self._register_crop_attempt("exception")
                    saved = False
                if saved:
                    self.crops_written += 1
                    self._record_crop_index(track_id, frame_idx, ts)

    def crop_component(self, track_id: int, frame_idx: int) -> str:
        ext = ".png" if self.use_png else ".jpg"
        return f"track_{track_id:04d}/frame_{frame_idx:06d}{ext}"

    def crop_rel_path(self, track_id: int, frame_idx: int) -> str:
        return f"crops/{self.crop_component(track_id, frame_idx)}"

    def crop_abs_path(self, track_id: int, frame_idx: int) -> Path:
        return self.crops_dir / self.crop_component(track_id, frame_idx)

    def _record_crop_index(self, track_id: int, frame_idx: int, ts: float | None) -> None:
        if not self.save_crops:
            return
        key = self.crop_component(track_id, frame_idx)
        entry = {
            "key": key,
            "frame_idx": int(frame_idx),
            "ts": round(float(ts), 4) if ts is not None else None,
        }
        self._track_indexes.setdefault(track_id, {})[key] = entry

    def write_indexes(self) -> None:
        if not self.save_crops or not self._track_indexes:
            return
        for track_id, entries in self._track_indexes.items():
            if not entries:
                continue
            track_dir = self.crops_dir / f"track_{track_id:04d}"
            if not track_dir.exists():
                continue
            ordered = sorted(entries.values(), key=lambda item: item["frame_idx"])
            index_path = track_dir / "index.json"
            try:
                index_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
            except Exception as exc:  # pragma: no cover - best effort
                LOGGER.warning("Failed to write crop index %s: %s", index_path, exc)

    def _register_crop_attempt(self, reason: str | None) -> None:
        self._crop_attempts += 1
        if reason:
            self._crop_error_counts[reason] += 1
        if reason in self._fail_fast_reasons:
            self._maybe_fail_fast()

    def _maybe_fail_fast(self) -> None:
        if self._crop_attempts < self._fail_fast_min_attempts:
            return
        bad = sum(self._crop_error_counts.get(reason, 0) for reason in self._fail_fast_reasons)
        ratio = bad / max(self._crop_attempts, 1)
        if ratio >= self._fail_fast_threshold:
            raise CropQualityThresholdExceeded(
                f"Too many invalid crops ({bad}/{self._crop_attempts}, {ratio:.1%}); aborting export"
            )

    def _write_crop(
        self,
        image,
        bbox: List[float],
        crop_path: Path,
        track_id: int,
        frame_idx: int,
    ) -> bool:
        start = time.time()
        bbox_vals = [float(val) for val in bbox]
        crop, clipped_bbox, crop_err = safe_crop(image, bbox_vals)
        debug_payload: Dict[str, Any] | None = None
        if self.debug_logger:
            debug_payload = {
                "track_id": track_id,
                "frame_idx": frame_idx,
                "out": str(crop_path),
                "bbox": bbox_vals,
                "clipped_bbox": list(clipped_bbox) if clipped_bbox else None,
                "err_before_save": crop_err,
            }
        if crop is None:
            self._register_crop_attempt(crop_err or "no_crop")
            if debug_payload is not None:
                debug_payload.update(
                    {
                        "save_ok": False,
                        "save_err": crop_err or "no_crop",
                        "ms": int((time.time() - start) * 1000),
                    }
                )
                self._emit_debug(debug_payload)
            return False

        # Skip saving crops smaller than 16x16 pixels - too small for useful face recognition
        min_dim = min(crop.shape[0], crop.shape[1]) if crop.ndim >= 2 else 0
        if min_dim < 16:
            self._register_crop_attempt("too_small")
            if debug_payload is not None:
                debug_payload.update(
                    {
                        "shape": tuple(int(x) for x in crop.shape),
                        "save_ok": False,
                        "save_err": "too_small",
                        "ms": int((time.time() - start) * 1000),
                    }
                )
                self._emit_debug(debug_payload)
            return False

        # Write crop using storage backend (S3) or direct filesystem
        if self._use_backend_writes and self._storage_backend is not None:
            # Direct-to-S3 mode: encode to bytes and upload
            try:
                if self.use_png:
                    img_data = encode_png_bytes(crop, compression=self.png_compression, color="bgr")
                else:
                    img_data = encode_jpeg_bytes(crop, quality=self.jpeg_quality, color="bgr")
                result = self._storage_backend.write_crop(
                    self.ep_id, track_id, frame_idx, img_data,
                    content_type="image/png" if self.use_png else "image/jpeg",
                )
                ok = result.success
                save_err = result.error if not ok else None
                file_size = len(img_data) if ok else None
                if ok:
                    self.bytes_written += len(img_data)
            except Exception as exc:
                ok = False
                save_err = str(exc)
                file_size = None
        else:
            # Local/hybrid mode: write directly to disk
            ok, save_err = safe_imwrite(
                crop_path, crop, self.jpeg_quality,
                use_png=self.use_png, png_compression=self.png_compression
            )
            file_size = crop_path.stat().st_size if ok and crop_path.exists() else None
            if ok and file_size:
                self.bytes_written += file_size

        reason = save_err if not ok else None
        self._register_crop_attempt(reason)

        if debug_payload is not None:
            mn, mx, mean = _image_stats(crop)
            debug_payload.update(
                {
                    "shape": tuple(int(x) for x in crop.shape),
                    "dtype": str(crop.dtype),
                    "min": mn,
                    "max": mx,
                    "mean": mean,
                    "save_ok": bool(ok),
                    "save_err": save_err,
                    "file_size": file_size,
                    "ms": int((time.time() - start) * 1000),
                }
            )
            self._emit_debug(debug_payload)

        if not ok and save_err:
            LOGGER.warning("Failed to save crop %s: %s", crop_path, save_err)
        return bool(ok)

    def _emit_debug(self, payload: Dict[str, Any]) -> None:
        if not self.debug_logger:
            return
        try:
            self.debug_logger(payload)
        except Exception:  # pragma: no cover - best effort diagnostics
            # Debug logging must never break frame processing.
            pass


class FrameDecoder:
    """Random-access video frame reader with LRU cache.

    Caches decoded frames to avoid redundant decoding when multiple faces
    appear on the same frame. Uses an OrderedDict to implement LRU eviction.

    Args:
        video_path: Path to video file
        cache_size: Maximum number of frames to cache (default 50)
    """

    def __init__(self, video_path: Path, cache_size: int = 50) -> None:
        import cv2  # type: ignore

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Unable to open video {video_path}")

        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = max(1, cache_size)
        self._cache_hits = 0
        self._cache_misses = 0

    def read(self, frame_idx: int):
        """Read frame with LRU caching.

        Args:
            frame_idx: Frame index to read

        Returns:
            Decoded frame as numpy array

        Raises:
            RuntimeError: If frame decode fails
        """
        frame_idx = max(int(frame_idx), 0)

        # Check cache first
        if frame_idx in self._cache:
            self._cache_hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(frame_idx)
            return self._cache[frame_idx].copy()  # Return copy to prevent mutation

        # Cache miss - decode from video
        self._cache_misses += 1
        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"Failed to decode frame {frame_idx}")

        # Store in cache
        self._cache[frame_idx] = frame.copy()
        self._cache.move_to_end(frame_idx)

        # Evict oldest if cache exceeds size
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)  # Remove oldest (FIFO)

        return frame

    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache statistics for debugging."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
            "hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses),
        }

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self) -> None:  # pragma: no cover - defensive
        try:
            self.close()
        except Exception:
            # GC cleanup should never raise during interpreter shutdown.
            pass


def _copy_video(src: Path, dest: Path) -> None:
    """Copy video from source to destination, skipping if unchanged.

    Compares file size and modification time to avoid unnecessary copies
    of large video files when running detect multiple times.

    Args:
        src: Source video path
        dest: Destination video path
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        return

    # Skip copy if dest exists and matches source (size + mtime)
    if dest.exists():
        src_stat = src.stat()
        dest_stat = dest.stat()
        if src_stat.st_size == dest_stat.st_size and abs(src_stat.st_mtime - dest_stat.st_mtime) < 1.0:
            LOGGER.info(
                "Skipping video copy; destination matches source (size=%d)",
                src_stat.st_size,
            )
            return

    shutil.copy2(src, dest)


def _estimate_duration(frame_count: int, fps: float) -> float | None:
    if frame_count > 0 and fps > 0:
        return frame_count / fps
    return None


def _estimate_frame_budget(
    *,
    stride: int,
    target_fps: float | None,
    detected_fps: float,
    duration_sec: float | None,
    frame_count: int,
) -> int:
    stride = max(stride, 1)
    fps_source = target_fps if target_fps and target_fps > 0 else detected_fps
    if fps_source and fps_source > 0 and duration_sec:
        value = int(math.ceil((fps_source * duration_sec) / stride))
    elif frame_count > 0:
        value = int(math.ceil(frame_count / stride))
    else:
        value = 0
    return max(value, 1)


def _episode_ctx(ep_id: str) -> EpisodeContext | None:
    try:
        return episode_context_from_id(ep_id)
    except ValueError:
        LOGGER.warning("Unable to parse episode id '%s'; artifact prefixes unavailable", ep_id)
        return None


def _storage_context(
    ep_id: str,
) -> tuple[StorageService | None, EpisodeContext | None, Dict[str, str] | None]:
    # Default storage backend:
    # - Local mode (streamed subprocess): prefer local filesystem to avoid blocking the UI on S3 uploads.
    # - Non-local mode (Celery/CI): keep S3 default for parity with the API.
    storage_backend_env = os.environ.get("STORAGE_BACKEND")
    if storage_backend_env:
        storage_backend = storage_backend_env.lower()
    else:
        storage_backend = "local" if LOCAL_MODE_INSTRUMENTATION else "s3"
    storage: StorageService | None = None
    if storage_backend in {"s3", "minio"}:
        try:
            storage = StorageService()
        except Exception as exc:  # pragma: no cover - best effort init
            LOGGER.warning("Storage init failed (%s); disabling uploads", exc)
            storage = None
    ep_ctx = _episode_ctx(ep_id)
    prefixes = artifact_prefixes(ep_ctx) if ep_ctx else None
    return storage, ep_ctx, prefixes


@dataclass
class DiskSpaceCheck:
    """Result of disk space pre-check."""

    ok: bool
    available_bytes: int
    required_bytes: int
    path: str
    warning: str | None = None


# Typical file sizes for estimation (conservative estimates)
BYTES_PER_FRAME_JPEG = 50_000  # ~50KB per frame JPEG
BYTES_PER_CROP_JPEG = 10_000  # ~10KB per face crop JPEG
BYTES_PER_THUMB_JPEG = 20_000  # ~20KB per thumbnail
AVG_FACES_PER_FRAME = 2  # Average faces detected per frame
DISK_SPACE_SAFETY_MULTIPLIER = 1.5  # 50% safety margin


def estimate_disk_usage(
    frame_count: int,
    save_frames: bool = True,
    save_crops: bool = True,
    avg_faces_per_frame: float = AVG_FACES_PER_FRAME,
) -> int:
    """Estimate disk space required for processing an episode.

    Args:
        frame_count: Total frames to process
        save_frames: Whether full frames will be saved
        save_crops: Whether face crops will be saved
        avg_faces_per_frame: Average faces per frame (for crop estimation)

    Returns:
        Estimated bytes required
    """
    total = 0

    if save_frames:
        total += frame_count * BYTES_PER_FRAME_JPEG

    if save_crops:
        estimated_crops = int(frame_count * avg_faces_per_frame)
        total += estimated_crops * BYTES_PER_CROP_JPEG
        # Also estimate thumbnails
        total += estimated_crops * BYTES_PER_THUMB_JPEG

    # Add safety margin
    return int(total * DISK_SPACE_SAFETY_MULTIPLIER)


def check_disk_space(
    path: str | Path,
    required_bytes: int,
    *,
    warn_threshold_gb: float = 5.0,
    block_threshold_gb: float = 1.0,
) -> DiskSpaceCheck:
    """Check if sufficient disk space is available.

    Args:
        path: Path where files will be written
        required_bytes: Estimated bytes needed
        warn_threshold_gb: Warn if available space is below this (GB)
        block_threshold_gb: Block job if available space is below this (GB)

    Returns:
        DiskSpaceCheck with status and details
    """
    target_path = Path(path)
    # Find an existing parent directory to check space
    check_path = target_path
    while not check_path.exists() and check_path.parent != check_path:
        check_path = check_path.parent

    try:
        stat = shutil.disk_usage(check_path)
        available = stat.free
    except Exception as exc:
        LOGGER.warning("Unable to check disk space for %s: %s", path, exc)
        # Assume OK if we can't check
        return DiskSpaceCheck(
            ok=True,
            available_bytes=0,
            required_bytes=required_bytes,
            path=str(path),
            warning=f"Unable to check disk space: {exc}",
        )

    available_gb = available / (1024 ** 3)
    required_gb = required_bytes / (1024 ** 3)
    block_threshold_bytes = int(block_threshold_gb * (1024 ** 3))
    warn_threshold_bytes = int(warn_threshold_gb * (1024 ** 3))

    # Check if we have enough space
    if available < block_threshold_bytes:
        return DiskSpaceCheck(
            ok=False,
            available_bytes=available,
            required_bytes=required_bytes,
            path=str(path),
            warning=f"Critical: Only {available_gb:.1f}GB available (need {required_gb:.1f}GB, minimum {block_threshold_gb}GB)",
        )

    if available < required_bytes:
        return DiskSpaceCheck(
            ok=False,
            available_bytes=available,
            required_bytes=required_bytes,
            path=str(path),
            warning=f"Insufficient space: {available_gb:.1f}GB available but {required_gb:.1f}GB estimated needed",
        )

    warning = None
    if available < warn_threshold_bytes:
        warning = f"Low disk space: {available_gb:.1f}GB available"

    return DiskSpaceCheck(
        ok=True,
        available_bytes=available,
        required_bytes=required_bytes,
        path=str(path),
        warning=warning,
    )


def pre_check_disk_space(
    ep_id: str,
    frame_count: int,
    save_frames: bool,
    save_crops: bool,
    *,
    fail_on_insufficient: bool = True,
) -> DiskSpaceCheck:
    """Pre-check disk space before starting a job.

    Args:
        ep_id: Episode identifier
        frame_count: Estimated total frames
        save_frames: Whether frames will be saved
        save_crops: Whether crops will be saved
        fail_on_insufficient: If True, raise error on insufficient space

    Returns:
        DiskSpaceCheck result

    Raises:
        RuntimeError: If fail_on_insufficient=True and space is insufficient
    """
    required = estimate_disk_usage(frame_count, save_frames, save_crops)
    data_root = get_path(ep_id, "frames_root")

    check = check_disk_space(data_root, required)

    if check.warning:
        LOGGER.warning("Disk space check for %s: %s", ep_id, check.warning)

    if not check.ok and fail_on_insufficient:
        raise RuntimeError(
            f"Insufficient disk space for {ep_id}: "
            f"{check.available_bytes / (1024**3):.1f}GB available, "
            f"{check.required_bytes / (1024**3):.1f}GB estimated needed"
        )

    return check


@dataclass
class S3SyncResult:
    """Result of S3 sync operation."""

    success: bool
    stats: Dict[str, int]
    errors: List[str]
    partial: bool = False  # True if some uploads succeeded but others failed


@dataclass
class BackgroundUploadStatus:
    """Status of background S3 upload operations."""

    pending: int = 0
    completed: int = 0
    failed: int = 0
    bytes_uploaded: int = 0
    in_progress: bool = False
    errors: List[str] = field(default_factory=list)


class BackgroundS3Uploader:
    """Non-blocking S3 uploader using a bounded queue and thread pool.

    This allows the main pipeline to continue processing while uploads happen
    in the background. Use wait_for_completion() at the end to ensure all
    uploads finish and get final status.

    Usage:
        uploader = BackgroundS3Uploader(storage, max_workers=4, max_queue_size=100)
        uploader.start()

        # Submit uploads (non-blocking)
        uploader.submit_file(local_path, s3_key)
        uploader.submit_bytes(data, s3_key)

        # At end of pipeline
        status = uploader.wait_for_completion(timeout=300)
        if status.failed > 0:
            log.warning("Some uploads failed: %s", status.errors)
    """

    def __init__(
        self,
        storage: StorageService,
        *,
        max_workers: int = 4,
        max_queue_size: int = 100,
        max_retries: int = 3,
    ):
        """Initialize background uploader.

        Args:
            storage: StorageService instance for S3 operations
            max_workers: Maximum concurrent upload threads
            max_queue_size: Maximum pending uploads before submit blocks
            max_retries: Number of retry attempts per upload
        """
        self._storage = storage
        self._max_workers = max_workers
        self._max_queue_size = max_queue_size
        self._max_retries = max_retries

        self._executor: ThreadPoolExecutor | None = None
        self._futures: Dict[Future, str] = {}  # Future -> S3 key for tracking
        self._lock = threading.Lock()

        # Status tracking
        self._pending = 0
        self._completed = 0
        self._failed = 0
        self._bytes_uploaded = 0
        self._errors: List[str] = []
        self._started = False
        self._shutdown = False

    def start(self) -> None:
        """Start the background uploader thread pool."""
        if self._started:
            return
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="s3_upload",
        )
        self._started = True
        self._shutdown = False

    def _upload_file_with_retry(
        self, local_path: Path, s3_key: str, content_type: str | None = None
    ) -> Tuple[bool, int, str | None]:
        """Upload a file with retry logic.

        Returns: (success, bytes_uploaded, error_message)
        """
        if not local_path.exists():
            return False, 0, f"File not found: {local_path}"

        file_size = local_path.stat().st_size
        last_error: str | None = None

        for attempt in range(self._max_retries):
            try:
                if content_type:
                    success = self._storage.upload_file(
                        str(local_path), s3_key, content_type=content_type
                    )
                else:
                    success = self._storage.upload_file(str(local_path), s3_key)

                if success:
                    return True, file_size, None
                last_error = f"Upload returned False for {s3_key}"
            except Exception as exc:
                last_error = f"Upload failed (attempt {attempt + 1}/{self._max_retries}): {exc}"
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        return False, 0, last_error

    def _upload_bytes_with_retry(
        self, data: bytes, s3_key: str, content_type: str = "application/octet-stream"
    ) -> Tuple[bool, int, str | None]:
        """Upload bytes with retry logic.

        Returns: (success, bytes_uploaded, error_message)
        """
        last_error: str | None = None

        for attempt in range(self._max_retries):
            try:
                success = self._storage.upload_bytes(data, s3_key, content_type=content_type)
                if success:
                    return True, len(data), None
                last_error = f"Upload returned False for {s3_key}"
            except Exception as exc:
                last_error = f"Upload failed (attempt {attempt + 1}/{self._max_retries}): {exc}"
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)

        return False, 0, last_error

    def submit_file(
        self,
        local_path: Path,
        s3_key: str,
        *,
        content_type: str | None = None,
        block_if_full: bool = True,
    ) -> bool:
        """Submit a file for background upload.

        Args:
            local_path: Path to local file
            s3_key: Destination S3 key
            content_type: Optional content type override
            block_if_full: If True, block when queue is full; otherwise return False

        Returns:
            True if submitted, False if queue is full and block_if_full=False
        """
        if not self._started or self._shutdown:
            raise RuntimeError("Uploader not started or already shutdown")

        with self._lock:
            if not block_if_full and self._pending >= self._max_queue_size:
                return False
            self._pending += 1

        # Submit to thread pool
        future = self._executor.submit(
            self._upload_file_with_retry, local_path, s3_key, content_type
        )
        with self._lock:
            self._futures[future] = s3_key

        return True

    def submit_bytes(
        self,
        data: bytes,
        s3_key: str,
        *,
        content_type: str = "application/octet-stream",
        block_if_full: bool = True,
    ) -> bool:
        """Submit bytes for background upload.

        Args:
            data: Bytes to upload
            s3_key: Destination S3 key
            content_type: Content type header
            block_if_full: If True, block when queue is full; otherwise return False

        Returns:
            True if submitted, False if queue is full and block_if_full=False
        """
        if not self._started or self._shutdown:
            raise RuntimeError("Uploader not started or already shutdown")

        with self._lock:
            if not block_if_full and self._pending >= self._max_queue_size:
                return False
            self._pending += 1

        future = self._executor.submit(
            self._upload_bytes_with_retry, data, s3_key, content_type
        )
        with self._lock:
            self._futures[future] = s3_key

        return True

    def submit_directory(
        self,
        dir_path: Path,
        s3_prefix: str,
        *,
        pattern: str = "*",
        recursive: bool = True,
        skip_subdirs: Tuple[str, ...] = (),
    ) -> int:
        """Submit all files in a directory for upload.

        Args:
            dir_path: Local directory path
            s3_prefix: S3 key prefix (should end with /)
            pattern: Glob pattern for files
            recursive: Whether to recurse into subdirectories
            skip_subdirs: Subdirectory names to skip

        Returns:
            Number of files submitted
        """
        if not dir_path.exists():
            return 0

        submitted = 0
        glob_fn = dir_path.rglob if recursive else dir_path.glob

        for file_path in glob_fn(pattern):
            if not file_path.is_file():
                continue

            # Check if in skipped subdir
            rel_parts = file_path.relative_to(dir_path).parts
            if rel_parts and rel_parts[0] in skip_subdirs:
                continue

            rel_key = file_path.relative_to(dir_path).as_posix()
            s3_key = f"{s3_prefix}{rel_key}"

            # Determine content type
            suffix = file_path.suffix.lower()
            content_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".json": "application/json",
                ".jsonl": "application/json",
            }.get(suffix, "application/octet-stream")

            self.submit_file(file_path, s3_key, content_type=content_type)
            submitted += 1

        return submitted

    def get_status(self) -> BackgroundUploadStatus:
        """Get current upload status (non-blocking)."""
        # Process any completed futures
        self._process_completed_futures()

        with self._lock:
            return BackgroundUploadStatus(
                pending=self._pending,
                completed=self._completed,
                failed=self._failed,
                bytes_uploaded=self._bytes_uploaded,
                in_progress=self._pending > 0,
                errors=list(self._errors),
            )

    def _process_completed_futures(self) -> None:
        """Process completed futures and update stats."""
        with self._lock:
            completed_futures = [f for f in self._futures if f.done()]

        for future in completed_futures:
            with self._lock:
                s3_key = self._futures.pop(future, "unknown")

            try:
                success, bytes_uploaded, error = future.result()
                with self._lock:
                    self._pending -= 1
                    if success:
                        self._completed += 1
                        self._bytes_uploaded += bytes_uploaded
                    else:
                        self._failed += 1
                        if error:
                            self._errors.append(f"{s3_key}: {error}")
            except Exception as exc:
                with self._lock:
                    self._pending -= 1
                    self._failed += 1
                    self._errors.append(f"{s3_key}: {exc}")

    def wait_for_completion(self, *, timeout: float | None = None) -> BackgroundUploadStatus:
        """Wait for all pending uploads to complete.

        Args:
            timeout: Maximum time to wait in seconds (None = no limit)

        Returns:
            Final upload status
        """
        if not self._started or self._executor is None:
            return BackgroundUploadStatus()

        start_time = time.time()

        with self._lock:
            pending_futures = list(self._futures.keys())

        try:
            # Wait for all futures with timeout
            for future in as_completed(pending_futures, timeout=timeout):
                self._process_completed_futures()

                # Check overall timeout
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        break
        except TimeoutError:
            pass

        # Final processing
        self._process_completed_futures()

        return self.get_status()

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the uploader.

        Args:
            wait: If True, wait for pending uploads to complete
        """
        self._shutdown = True
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None
        self._started = False

    def __enter__(self) -> "BackgroundS3Uploader":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown(wait=True)


def _sync_artifacts_to_s3_async(
    ep_id: str,
    storage: StorageService | None,
    ep_ctx: EpisodeContext | None,
    exporter: FrameExporter | None,
    thumb_dir: Path | None = None,
    *,
    max_workers: int = 4,
    timeout: float = 600,
) -> S3SyncResult:
    """Sync artifacts to S3 using background uploads with ThreadPoolExecutor.

    This is a non-blocking alternative to _sync_artifacts_to_s3 that uses a
    thread pool for parallel uploads. The pipeline continues while uploads
    happen in background threads.

    Args:
        ep_id: Episode identifier
        storage: StorageService instance
        ep_ctx: Episode context for S3 key prefixes
        exporter: FrameExporter with frames/crops to upload
        thumb_dir: Optional thumbnail directory
        max_workers: Number of concurrent upload threads
        timeout: Maximum time to wait for all uploads (seconds)

    Returns:
        S3SyncResult with success status, stats, and any errors
    """
    stats: Dict[str, int] = {
        "manifests": 0,
        "frames": 0,
        "crops": 0,
        "thumbs_tracks": 0,
        "thumbs_identities": 0,
    }

    if storage is None or ep_ctx is None or not storage.s3_enabled() or not storage.write_enabled:
        return S3SyncResult(success=True, stats=stats, errors=[])

    prefixes = artifact_prefixes(ep_ctx)

    with BackgroundS3Uploader(storage, max_workers=max_workers) as uploader:
        # Submit manifests
        manifests_dir = get_path(ep_id, "detections").parent
        if manifests_dir.exists():
            stats["manifests"] = uploader.submit_directory(
                manifests_dir, prefixes["manifests"], pattern="*.json*"
            )

        # Submit frames
        if exporter and exporter.save_frames and exporter.frames_dir.exists():
            stats["frames"] = uploader.submit_directory(
                exporter.frames_dir, prefixes["frames"]
            )

        # Submit crops
        if exporter and exporter.save_crops and exporter.crops_dir.exists():
            stats["crops"] = uploader.submit_directory(
                exporter.crops_dir, prefixes["crops"]
            )

        # Submit thumbnails
        if thumb_dir is not None and thumb_dir.exists():
            stats["thumbs_tracks"] = uploader.submit_directory(
                thumb_dir,
                prefixes.get("thumbs_tracks", ""),
                skip_subdirs=("identities",),
            )

            identities_dir = thumb_dir / "identities"
            if identities_dir.exists():
                stats["thumbs_identities"] = uploader.submit_directory(
                    identities_dir, prefixes.get("thumbs_identities", "")
                )

        # Wait for completion
        final_status = uploader.wait_for_completion(timeout=timeout)

    # Convert to S3SyncResult
    errors = final_status.errors
    has_errors = len(errors) > 0 or final_status.failed > 0
    has_uploads = final_status.completed > 0
    partial = has_errors and has_uploads

    if final_status.failed > 0 and not errors:
        errors.append(f"{final_status.failed} uploads failed")

    return S3SyncResult(
        success=not has_errors,
        stats=stats,
        errors=errors,
        partial=partial,
    )


def _sync_artifacts_to_s3(
    ep_id: str,
    storage: StorageService | None,
    ep_ctx: EpisodeContext | None,
    exporter: FrameExporter | None,
    thumb_dir: Path | None = None,
    *,
    max_retries: int = 3,
    required_for_success: bool = True,
) -> S3SyncResult:
    """Sync artifacts to S3 storage with retry support.

    When required_for_success=True, the job should not be marked as complete
    until S3 sync succeeds. This ensures artifacts are accessible via S3 presigned URLs.

    Args:
        ep_id: Episode identifier
        storage: StorageService instance
        ep_ctx: Episode context for S3 key prefixes
        exporter: FrameExporter with frames/crops to upload
        thumb_dir: Optional thumbnail directory
        max_retries: Number of retry attempts for each upload type (default: 3)
        required_for_success: If True, sync failures should fail the job

    Returns:
        S3SyncResult with success status, stats, and any errors
    """
    stats: Dict[str, int] = {
        "manifests": 0,
        "frames": 0,
        "crops": 0,
        "thumbs_tracks": 0,
        "thumbs_identities": 0,
    }
    errors: List[str] = []

    if storage is None or ep_ctx is None or not storage.s3_enabled() or not storage.write_enabled:
        # No S3 configured - this is success if S3 not required
        return S3SyncResult(success=True, stats=stats, errors=[])

    prefixes = artifact_prefixes(ep_ctx)

    def _upload_with_retry(
        upload_fn, dir_path: Path, prefix: str, name: str, **kwargs
    ) -> Tuple[int, str | None]:
        """Upload with retry logic, returns (count, error_message)."""
        if not dir_path.exists():
            return 0, None

        last_error: str | None = None
        for attempt in range(max_retries):
            try:
                count = upload_fn(dir_path, prefix, **kwargs)
                return count, None
            except Exception as exc:
                last_error = f"{name} upload failed (attempt {attempt + 1}/{max_retries}): {exc}"
                LOGGER.warning(last_error)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s

        return 0, last_error

    # Upload manifests (always required)
    manifests_dir = get_path(ep_id, "detections").parent
    skip_manifests_subdirs: list[str] = []
    runs_dir = manifests_dir / RUN_MARKERS_SUBDIR
    if runs_dir.exists():
        try:
            for entry in runs_dir.iterdir():
                if entry.is_dir():
                    skip_manifests_subdirs.append(f"{RUN_MARKERS_SUBDIR}/{entry.name}")
        except OSError:
            pass
    count, err = _upload_with_retry(
        storage.upload_dir,
        manifests_dir,
        prefixes["manifests"],
        "manifests",
        skip_subdirs=skip_manifests_subdirs or None,
    )
    stats["manifests"] = count
    if err:
        errors.append(err)

    # Upload frames if exporter produced them
    if exporter and exporter.save_frames and exporter.frames_dir.exists():
        count, err = _upload_with_retry(
            storage.upload_dir, exporter.frames_dir, prefixes["frames"], "frames"
        )
        stats["frames"] = count
        if err:
            errors.append(err)

    # Upload crops if exporter produced them
    if exporter and exporter.save_crops and exporter.crops_dir.exists():
        count, err = _upload_with_retry(
            storage.upload_dir, exporter.crops_dir, prefixes["crops"], "crops"
        )
        stats["crops"] = count
        if err:
            errors.append(err)

    # Upload thumbnails
    if thumb_dir is not None and thumb_dir.exists():
        identities_dir = thumb_dir / "identities"
        count, err = _upload_with_retry(
            storage.upload_dir,
            thumb_dir,
            prefixes.get("thumbs_tracks", ""),
            "track_thumbs",
            skip_subdirs=("identities",),
        )
        stats["thumbs_tracks"] = count
        if err:
            errors.append(err)

        if identities_dir.exists():
            count, err = _upload_with_retry(
                storage.upload_dir,
                identities_dir,
                prefixes.get("thumbs_identities", ""),
                "identity_thumbs",
            )
            stats["thumbs_identities"] = count
            if err:
                errors.append(err)

    # Determine success
    has_errors = len(errors) > 0
    has_uploads = any(stats.values())
    partial = has_errors and has_uploads

    if required_for_success and has_errors:
        return S3SyncResult(success=False, stats=stats, errors=errors, partial=partial)

    return S3SyncResult(success=True, stats=stats, errors=errors, partial=partial)


def _report_s3_upload(
    progress: ProgressEmitter | None,
    sync_result: S3SyncResult | Dict[str, int],
    *,
    device: str | None,
    detector: str | None,
    tracker: str | None,
    resolved_device: str | None,
) -> None:
    """Report S3 upload status to progress emitter.

    Accepts either S3SyncResult (new format) or Dict[str, int] (legacy format).
    """
    if not progress:
        return

    # Handle both new S3SyncResult and legacy dict format
    if isinstance(sync_result, S3SyncResult):
        stats = sync_result.stats
        errors = sync_result.errors
        success = sync_result.success
    else:
        stats = sync_result
        errors = []
        success = True

    if not any(stats.values()) and not errors:
        return

    frames = progress.target_frames or 0
    summary: Dict[str, Any] = {"s3_uploads": stats}
    if errors:
        summary["s3_errors"] = errors
    if not success:
        summary["s3_sync_failed"] = True

    progress.emit(
        frames,
        phase="mirror_s3",
        device=device,
        summary=summary,
        detector=detector,
        tracker=tracker,
        resolved_device=resolved_device,
        force=True,
        error="; ".join(errors) if errors else None,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Run detection + tracking locally.")
    parser.add_argument(
        "--profile",
        choices=list(PROFILE_CHOICES),
        help="Performance profile (fast_cpu/low_power/balanced/high_accuracy) to apply default stride/FPS/save options.",
    )
    parser.add_argument("--ep-id", required=True, help="Episode identifier")
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional pipeline run identifier. When omitted, a new run_id is generated. Phase outputs are "
            "written under data/manifests/{ep_id}/runs/{run_id}/ and then promoted to the legacy root manifests dir."
        ),
    )
    parser.add_argument("--video", help="Path to source video (required for detect/track runs)")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride for detection (default: 1 = every frame)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Optional target FPS for downsampling before detection",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "coreml", "metal", "apple", "cuda"],
        default="auto",
        help="Execution device override (auto→CUDA/CoreML/CPU). Accepts auto/cpu/cuda/mps/coreml/metal/apple.",
    )
    parser.add_argument(
        "--embed-device",
        choices=["auto", "cpu", "mps", "coreml", "metal", "apple", "cuda"],
        default=None,
        help="Optional ArcFace/embedding device override. Defaults to --device when omitted.",
    )
    parser.add_argument(
        "--coreml-det-size",
        type=str,
        default=None,
        help=(
            "Override RetinaFace CoreML input resolution (e.g., 384x384 for an M1/M2 Air, 512x512+ for Pro/Max) "
            "to trade recall vs thermals on smaller Macs. Only applies when CoreML is the resolved provider."
        ),
    )
    parser.add_argument(
        "--coreml-only",
        action="store_true",
        default=True,
        help=(
            "Force CoreML-only execution without CPU fallback. Prevents CPU provider saturation "
            "but will fail if CoreML is unavailable. Useful for enforcing <300%% CPU budget on Apple Silicon. "
            "Default: enabled. Use --no-coreml-only or --allow-cpu-fallback to allow CPU fallback."
        ),
    )
    parser.add_argument(
        "--no-coreml-only",
        dest="coreml_only",
        action="store_false",
        help="Allow CPU fallback for CoreML execution (may exceed CPU budget).",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        default=False,
        help=(
            "Allow falling back to CPU if requested accelerator (coreml/cuda) is unavailable. "
            "If not set, fails fast with an error when the accelerator is unavailable. "
            "This prevents silent CPU execution that causes thermal issues on laptops."
        ),
    )
    parser.add_argument(
        "--detector",
        choices=list(DETECTOR_CHOICES),
        default=DEFAULT_DETECTOR,
        help="Face detector backend (RetinaFace only)",
    )
    parser.add_argument(
        "--tracker",
        choices=list(TRACKER_CHOICES),
        default=DEFAULT_TRACKER,
        help="Tracker backend (ByteTrack default, StrongSORT optional for occlusions)",
    )
    parser.add_argument(
        "--track-high-thresh",
        type=float,
        default=TRACK_HIGH_THRESH_DEFAULT,
        help="ByteTrack track_high_thresh gate (default: env or 0.5).",
    )
    parser.add_argument(
        "--new-track-thresh",
        type=float,
        default=TRACK_NEW_THRESH_DEFAULT,
        help="ByteTrack new_track_thresh gate (default: env or 0.5).",
    )
    parser.add_argument(
        "--track-buffer",
        type=int,
        default=TRACK_BUFFER_BASE_DEFAULT,
        help="Base ByteTrack track_buffer before stride scaling (default: env or 30).",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=BYTE_TRACK_MIN_BOX_AREA_DEFAULT,
        help="Minimum box area for ByteTrack gating (default: env or 20).",
    )
    parser.add_argument(
        "--scene-detector",
        choices=list(SCENE_DETECTOR_CHOICES),
        default=SCENE_DETECTOR_DEFAULT,
        help="Scene-cut prepass backend (PySceneDetect default, internal histogram fallback, or off)",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=SCENE_THRESHOLD_DEFAULT,
        help="Scene-cut threshold passed to the detector (PySceneDetect≈27, histogram fallback expects 0-2)",
    )
    parser.add_argument(
        "--scene-min-len",
        type=int,
        default=SCENE_MIN_LEN_DEFAULT,
        help="Minimum frames between scene cuts",
    )
    parser.add_argument(
        "--scene-warmup-dets",
        type=int,
        default=SCENE_WARMUP_DETS_DEFAULT,
        help="Frames of forced detection after each cut",
    )
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=RETINAFACE_SCORE_THRESHOLD,
        help="RetinaFace detection score threshold (0-1, default 0.5)",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=60,
        help="Maximum frame gap before splitting a track",
    )
    parser.add_argument(
        "--track-sample-limit",
        type=int,
        default=None,
        help="Optional max samples stored per track (0→all detections, default)",
    )
    parser.add_argument(
        "--max-samples-per-track",
        type=int,
        default=16,
        help="Maximum samples per track for embedding/export (default: 16)",
    )
    parser.add_argument(
        "--min-samples-per-track",
        type=int,
        default=4,
        help="Minimum samples per track if track is long enough (default: 4)",
    )
    parser.add_argument(
        "--sample-every-n-frames",
        type=int,
        default=4,
        help="Sample interval for per-track sampling (default: 4)",
    )
    parser.add_argument("--thumb-size", type=int, default=256, help="Square thumbnail size for faces")
    parser.add_argument(
        "--out-root",
        help="Data root override (defaults to SCREENALYTICS_DATA_ROOT or ./data)",
    )
    parser.add_argument("--progress-file", help="Progress JSON file to update during processing")
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save sampled frame JPGs under data/frames/{ep_id}",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save per-track crops (requires --save-frames or track IDs)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for frame exports (1-100)",
    )
    parser.add_argument("--faces-embed", action="store_true", help="Run faces embedding stage only")
    parser.add_argument("--cluster", action="store_true", help="Run clustering stage only")
    parser.add_argument(
        "--cluster-thresh",
        type=float,
        default=DEFAULT_CLUSTER_SIMILARITY,
        help="Minimum cosine similarity for merging tracks (converted to 1-sim distance)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum tracks per identity before splitting into singletons",
    )
    parser.add_argument(
        "--min-identity-sim",
        type=float,
        default=MIN_IDENTITY_SIMILARITY,
        help="Minimum cosine similarity for a track to remain in an identity cluster (outliers are split out)",
    )
    parser.add_argument(
        "--preserve-assigned",
        action="store_true",
        help="Preserve clusters that are assigned to cast members (don't recluster their tracks)",
    )
    parser.add_argument(
        "--ignore-preservation-errors",
        action="store_true",
        help="Continue clustering even if cast-assigned cluster preservation fails (may lose assignments)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose ONNX/model warnings (cleaner logs)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all debug output including ONNX/model warnings",
    )
    gate_group = parser.add_argument_group("Appearance gate")
    gate_group.add_argument(
        "--gate-enabled",
        dest="gate_enabled",
        action="store_true",
        default=GATE_ENABLED_DEFAULT,
        help="Enable appearance-based track splitting (default from tracking.yaml)",
    )
    gate_group.add_argument(
        "--no-gate",
        "--gate-disabled",
        dest="gate_enabled",
        action="store_false",
        help="Disable appearance gate (reduces forced splits on stable tracks)",
    )
    gate_group.add_argument(
        "--gate-appear-hard",
        type=float,
        default=GATE_APPEAR_T_HARD_DEFAULT,
        help="Force split when cosine similarity drops below this hard threshold (default 0.60)",
    )
    gate_group.add_argument(
        "--gate-appear-soft",
        type=float,
        default=GATE_APPEAR_T_SOFT_DEFAULT,
        help="Begin streak counting when similarity drops below this soft threshold (default 0.70)",
    )
    gate_group.add_argument(
        "--gate-appear-streak",
        type=int,
        default=GATE_APPEAR_STREAK_DEFAULT,
        help="Consecutive soft violations before forcing a split (default 2)",
    )
    gate_group.add_argument(
        "--gate-iou",
        type=float,
        default=GATE_IOU_THRESHOLD_DEFAULT,
        help="Minimum IoU to keep extending a track when appearance is uncertain (default 0.35)",
    )
    gate_group.add_argument(
        "--gate-proto-momentum",
        type=float,
        default=GATE_PROTO_MOMENTUM_DEFAULT,
        help="Momentum applied when updating per-track prototypes (0→current, 1→frozen)",
    )
    gate_group.add_argument(
        "--gate-emb-every",
        type=int,
        default=GATE_EMB_EVERY_DEFAULT,
        help="Frames between gate embeddings (0 uses detect stride)",
    )
    args = parser.parse_args(argv_list)
    args._raw_argv = argv_list
    return args


def _configure_logging(quiet: bool, verbose: bool) -> None:
    """Configure logging levels based on --quiet/--verbose flags."""
    if quiet and not verbose:
        # Suppress noisy model/runtime warnings
        logging.getLogger("onnxruntime").setLevel(logging.ERROR)
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("insightface").setLevel(logging.ERROR)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        # Also suppress warnings module for common ONNX/TF deprecation warnings
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*onnx.*", category=UserWarning)
    elif verbose:
        # Enable debug output
        logging.getLogger("episode_run").setLevel(logging.DEBUG)


def _finalize_run_exports(ep_id: str, run_id: str) -> None:
    try:
        from apps.api.services.run_export import build_and_upload_debug_pdf, run_segments_export
    except Exception as exc:
        LOGGER.warning("[export] Failed to import run_export: %s", exc)
        return

    try:
        build_and_upload_debug_pdf(
            ep_id=ep_id,
            run_id=run_id,
            upload_to_s3=False,
            write_index=True,
        )
    except Exception as exc:
        LOGGER.warning("[export] run_debug.pdf generation failed: %s", exc)

    try:
        run_segments_export(ep_id=ep_id, run_id=run_id)
    except Exception as exc:
        LOGGER.warning("[export] segments.parquet export failed: %s", exc)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    raw_argv: List[str] = getattr(args, "_raw_argv", [])

    # Apply performance profile defaults when provided (explicit CLI flags win)
    profile_cfg = _load_performance_profile(args.profile or os.environ.get("SCREENALYTICS_PERF_PROFILE"))
    if profile_cfg:
        if not _flag_present(raw_argv, "--stride"):
            try:
                args.stride = max(int(profile_cfg.get("frame_stride", args.stride)), 1)
            except (TypeError, ValueError):
                pass
        if not _flag_present(raw_argv, "--fps"):
            try:
                fps_value = float(profile_cfg.get("detection_fps_limit") or profile_cfg.get("max_fps") or 0.0)
            except (TypeError, ValueError):
                fps_value = 0.0
            if fps_value > 0:
                args.fps = fps_value
        if not _flag_present(raw_argv, "--save-frames") and "save_frames" in profile_cfg:
            args.save_frames = bool(profile_cfg.get("save_frames"))
        if not _flag_present(raw_argv, "--save-crops") and "save_crops" in profile_cfg:
            args.save_crops = bool(profile_cfg.get("save_crops"))
        if not _flag_present(raw_argv, "--coreml-det-size") and profile_cfg.get("coreml_input_size"):
            coreml_size = profile_cfg.get("coreml_input_size")
            if isinstance(coreml_size, (int, float)):
                coreml_size = f"{int(coreml_size)}x{int(coreml_size)}"
            args.coreml_det_size = str(coreml_size)

    # Apply wide shot mode override if enabled (increases input size for small face detection)
    # This overrides profile settings unless explicit CLI flag was provided
    if WIDE_SHOT_MODE_ENABLED and not _flag_present(raw_argv, "--coreml-det-size"):
        wide_size = WIDE_SHOT_INPUT_SIZE
        args.coreml_det_size = f"{wide_size}x{wide_size}"
        LOGGER.info("Wide shot mode: overriding coreml_det_size to %dx%d", wide_size, wide_size)
        # Also apply lower detection threshold for small faces
        if not _flag_present(raw_argv, "--det-thresh") and hasattr(args, "det_thresh"):
            args.det_thresh = WIDE_SHOT_CONFIDENCE_TH
            LOGGER.info("Wide shot mode: using lower detection threshold %.2f", WIDE_SHOT_CONFIDENCE_TH)

    # Configure logging based on --quiet/--verbose flags
    _configure_logging(getattr(args, "quiet", False), getattr(args, "verbose", False))

    coreml_size_arg = getattr(args, "coreml_det_size", None)
    if coreml_size_arg:
        args.coreml_det_size = _parse_retinaface_det_size(coreml_size_arg)
    else:
        args.coreml_det_size = None
    if hasattr(args, "det_thresh"):
        args.det_thresh = _normalize_det_thresh(getattr(args, "det_thresh", RETINAFACE_SCORE_THRESHOLD))
    args.scene_detector = _normalize_scene_detector_choice(getattr(args, "scene_detector", None))
    args.scene_threshold = max(float(getattr(args, "scene_threshold", SCENE_THRESHOLD_DEFAULT)), 0.0)
    args.scene_min_len = max(int(getattr(args, "scene_min_len", SCENE_MIN_LEN_DEFAULT)), 1)
    args.scene_warmup_dets = max(int(getattr(args, "scene_warmup_dets", SCENE_WARMUP_DETS_DEFAULT)), 0)
    args.track_high_thresh = min(
        max(
            float(getattr(args, "track_high_thresh", TRACK_HIGH_THRESH_DEFAULT) or TRACK_HIGH_THRESH_DEFAULT),
            0.0,
        ),
        1.0,
    )
    args.new_track_thresh = min(
        max(
            float(getattr(args, "new_track_thresh", TRACK_NEW_THRESH_DEFAULT) or TRACK_NEW_THRESH_DEFAULT),
            0.0,
        ),
        1.0,
    )
    args.track_buffer = max(
        int(getattr(args, "track_buffer", TRACK_BUFFER_BASE_DEFAULT) or TRACK_BUFFER_BASE_DEFAULT),
        1,
    )
    args.min_box_area = max(
        float(getattr(args, "min_box_area", BYTE_TRACK_MIN_BOX_AREA_DEFAULT) or BYTE_TRACK_MIN_BOX_AREA_DEFAULT),
        0.0,
    )
    cli_track_limit = getattr(args, "track_sample_limit", None)
    if cli_track_limit is not None:
        _set_track_sample_limit(cli_track_limit)
        if TRACK_SAMPLE_LIMIT is None:
            LOGGER.info("Track sampling disabled; persisting all detections per track.")
        else:
            LOGGER.info(
                "Track sampling limited to the first %s detections per track.",
                TRACK_SAMPLE_LIMIT,
            )
    data_root = (
        Path(args.out_root).expanduser()
        if args.out_root
        else Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
    )
    os.environ["SCREENALYTICS_DATA_ROOT"] = str(data_root)
    ensure_dirs(args.ep_id)

    # Ensure run_id is always set for run-scoped outputs/status
    raw_run_id = getattr(args, "run_id", None)
    run_id_missing = raw_run_id is None or not str(raw_run_id).strip()
    args.run_id = run_layout.get_or_create_run_id(args.ep_id, raw_run_id)
    if run_id_missing:
        LOGGER.info("[episode_run] Generated run_id=%s for ep_id=%s", args.run_id, args.ep_id)
        print(f"[episode_run] run_id={args.run_id}", file=sys.stderr)

    # ---------------------------------------------------------------------
    # Preflight: environment fingerprint (run-scoped when run_id is provided)
    # ---------------------------------------------------------------------
    try:
        from py_screenalytics.env_diagnostics import DEFAULT_ENV_PACKAGES, collect_env_diagnostics, write_env_diagnostics_json

        env_diag = collect_env_diagnostics(DEFAULT_ENV_PACKAGES)
        setattr(args, "_env_diagnostics", env_diag)

        LOGGER.info(
            "[env] python=%s pip=%s venv_active=%s sys.executable=%s sys.prefix=%s",
            env_diag.get("python_version"),
            env_diag.get("pip_version"),
            env_diag.get("venv_active"),
            env_diag.get("sys_executable"),
            env_diag.get("sys_prefix"),
        )
        pkg_versions = env_diag.get("package_versions")
        if isinstance(pkg_versions, dict) and pkg_versions:
            LOGGER.info("[env] package_versions=%s", pkg_versions)
        import_status = env_diag.get("import_status")
        if isinstance(import_status, dict) and import_status:
            LOGGER.info("[env] import_status=%s", import_status)

        if args.run_id:
            run_root = run_layout.run_root(args.ep_id, run_layout.normalize_run_id(args.run_id))
            env_path = run_root / "env_diagnostics.json"
            write_env_diagnostics_json(env_path, env_diag)
            LOGGER.info("[env] wrote %s", env_path)
    except Exception as exc:  # pragma: no cover - best-effort diagnostics
        LOGGER.warning("[env] Failed to capture env diagnostics: %s", exc)
    storage, ep_ctx, s3_prefixes = _storage_context(args.ep_id)

    phase_flags = [flag for flag in (args.faces_embed, args.cluster) if flag]
    if len(phase_flags) > 1:
        raise ValueError("Specify at most one of --faces-embed/--cluster per run")

    summary: Dict[str, Any] | None = None
    try:
        if args.faces_embed:
            summary = _run_faces_embed_stage(args, storage, ep_ctx, s3_prefixes)
        elif args.cluster:
            summary = _run_cluster_stage(args, storage, ep_ctx, s3_prefixes)
        else:
            summary = _run_detect_track_stage(args, storage, ep_ctx, s3_prefixes)

        stage = summary.get("stage", "detect_track")
        device_label = summary.get("device")
        analyzed_fps = summary.get("analyzed_fps")
        log_msg = f"stage={stage}"
        if device_label:
            log_msg += f" device={device_label}"
        if analyzed_fps:
            log_msg += f" analyzed_fps={analyzed_fps:.3f}"
        print(f"[episode_run] {log_msg}", file=sys.stderr)
        print("[episode_run] summary", summary, file=sys.stderr)
        return 0
    finally:
        if getattr(args, "run_id", None):
            _finalize_run_exports(args.ep_id, args.run_id)


def _gate_config_from_args(args: argparse.Namespace, frame_stride: int) -> GateConfig:
    def _clamp(value: float, *, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, float(value)))

    appear_hard = _clamp(getattr(args, "gate_appear_hard", GATE_APPEAR_T_HARD_DEFAULT))
    appear_soft = _clamp(getattr(args, "gate_appear_soft", GATE_APPEAR_T_SOFT_DEFAULT))
    streak = max(int(getattr(args, "gate_appear_streak", GATE_APPEAR_STREAK_DEFAULT)), 1)
    gate_iou = _clamp(getattr(args, "gate_iou", GATE_IOU_THRESHOLD_DEFAULT), lo=0.0, hi=1.0)
    proto_mom = _clamp(getattr(args, "gate_proto_momentum", GATE_PROTO_MOMENTUM_DEFAULT))
    emb_every = getattr(args, "gate_emb_every", None)
    if emb_every is None or int(emb_every) <= 0:
        emb_stride = max(frame_stride, 1)
    else:
        emb_stride = max(int(emb_every), 1)
    return GateConfig(
        appear_t_hard=appear_hard,
        appear_t_soft=max(appear_soft, appear_hard + 0.01),
        appear_streak=streak,
        gate_iou=gate_iou,
        proto_momentum=proto_mom,
        emb_every=emb_stride,
    )


def _gate_auto_rerun_decision(metrics: Dict[str, Any]) -> tuple[bool, str, Dict[str, Any]]:
    """Decide whether to auto-rerun tracking with gate disabled.

    Trigger only when forced_splits are extreme AND id_switches are low,
    indicating the appearance gate is over-splitting stable tracks.
    """

    def _safe_int(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return None

    forced_splits = _safe_int(metrics.get("forced_splits")) or 0
    id_switches = _safe_int(metrics.get("id_switches")) or 0
    gate_splits: int | None = None
    gate_share: float | None = None

    appearance_gate = metrics.get("appearance_gate")
    if isinstance(appearance_gate, dict):
        splits = appearance_gate.get("splits")
        if isinstance(splits, dict):
            gate_splits = _safe_int(splits.get("total"))
    if gate_splits is not None and forced_splits > 0:
        gate_share = gate_splits / forced_splits

    snapshot = {
        "forced_splits": forced_splits,
        "id_switches": id_switches,
        "gate_splits_total": gate_splits,
        "gate_splits_share": (round(gate_share, 3) if gate_share is not None else None),
        "thresholds": {
            "forced_splits": GATE_AUTO_RERUN_FORCED_SPLITS_THRESHOLD,
            "max_id_switches": GATE_AUTO_RERUN_MAX_ID_SWITCHES,
            "min_gate_share": GATE_AUTO_RERUN_MIN_GATE_SPLITS_SHARE,
        },
        "enabled": GATE_AUTO_RERUN_ENABLED,
    }

    if not GATE_AUTO_RERUN_ENABLED:
        return False, "disabled", snapshot
    if forced_splits < GATE_AUTO_RERUN_FORCED_SPLITS_THRESHOLD:
        return False, "forced_splits_below_threshold", snapshot
    if id_switches > GATE_AUTO_RERUN_MAX_ID_SWITCHES:
        return False, "id_switches_too_high", snapshot
    if gate_share is not None and gate_share < GATE_AUTO_RERUN_MIN_GATE_SPLITS_SHARE:
        return False, "gate_share_too_low", snapshot
    return True, "forced_splits_high_id_switches_low", snapshot


def _effective_stride(stride: int, target_fps: float | None, source_fps: float) -> int:
    stride = max(stride, 1)
    if target_fps and target_fps > 0 and source_fps > 0:
        fps_stride = max(int(round(source_fps / target_fps)), 1)
        stride = max(stride, fps_stride)
    return stride


def _probe_video(video_path: Path) -> Tuple[float, int]:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0:
        fps = 30.0
    return fps, frame_count


def _detect_fps(video_path: Path) -> float:
    fps, _ = _probe_video(video_path)
    if fps <= 0:
        fps = 24.0
    return fps


def _detect_scene_cuts_histogram(
    video_path: str | Path,
    *,
    thr: float = SCENE_THRESHOLD_DEFAULT,
    min_len: int = SCENE_MIN_LEN_DEFAULT,
    progress: ProgressEmitter | None = None,
) -> list[int]:
    """Lightweight HSV histogram scene-cut detector used as a fallback."""

    import cv2  # type: ignore

    threshold = max(min(float(thr or SCENE_THRESHOLD_DEFAULT), 2.0), 0.0)
    min_gap = max(int(min_len or SCENE_MIN_LEN_DEFAULT), 1)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    cuts: list[int] = []
    prev_hist = None
    last_cut = -(10**9)
    idx = 0
    target_frames = progress.target_frames if progress else 0
    emit_interval = 50  # Emit every 50 frames to reduce spam
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, None).flatten()
        if prev_hist is not None:
            diff = 1.0 - float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
            if diff > threshold and (idx - last_cut) >= min_gap:
                cuts.append(idx)
                last_cut = idx
        # Emit sparse updates (every 50 frames) instead of on every cut
        if progress and (idx % emit_interval == 0 or idx == target_frames - 1):
            frames_done = idx if idx >= 0 else 0
            if target_frames:
                frames_done = min(target_frames, frames_done)
            progress.emit(
                frames_done,
                phase="scene_detect:cut",
                summary={"count": len(cuts), "detector": "internal"},
                extra=_non_video_phase_meta(),
            )
        prev_hist = hist
        idx += 1
    cap.release()
    return cuts


def detect_scene_cuts_pyscenedetect(
    video_path: str | Path,
    *,
    threshold: float = SCENE_THRESHOLD_DEFAULT,
    min_len: int = SCENE_MIN_LEN_DEFAULT,
) -> list[int]:
    """Detect hard cuts via PySceneDetect's ContentDetector."""

    try:
        from scenedetect import SceneManager, open_video  # type: ignore
        from scenedetect.detectors import ContentDetector  # type: ignore
    except ImportError as exc:  # pragma: no cover - enforced via optional dependency
        raise RuntimeError(
            "PySceneDetect is required for scene detection. Install it via `pip install scenedetect>=0.6.4`."
        ) from exc

    video = open_video(str(video_path))
    manager = SceneManager()
    detector = ContentDetector(threshold=float(threshold), min_scene_len=max(int(min_len), 1))
    manager.add_detector(detector)
    try:
        manager.detect_scenes(video, show_progress=False)
        scenes = manager.get_scene_list()
    finally:
        close_handle = getattr(video, "close", None)
        if callable(close_handle):  # pragma: no cover - depends on backend
            close_handle()
        else:  # pragma: no cover - fallback path
            release_handle = getattr(video, "release", None)
            if callable(release_handle):
                release_handle()
    # Filter out frame 0 - it's the video start, not a cut
    return [start.get_frames() for (start, _end) in scenes if start.get_frames() > 0]


def detect_scene_cuts(
    video_path: str | Path,
    *,
    detector: str | None = None,
    thr: float = SCENE_THRESHOLD_DEFAULT,
    min_len: int = SCENE_MIN_LEN_DEFAULT,
    progress: ProgressEmitter | None = None,
) -> list[int]:
    """Run the configured scene-cut detector and emit consistent progress events."""

    detector_choice = _normalize_scene_detector_choice(detector)
    threshold_value = max(float(thr), 0.0)
    if detector_choice == "internal":
        threshold_value = max(min(threshold_value, 2.0), 0.0)
    summary_start = {
        "detector": detector_choice,
        "threshold": round(float(threshold_value), 3),
        "min_len": max(int(min_len), 1),
    }
    if progress:
        progress.emit(
            0,
            phase="scene_detect:cut",
            summary=summary_start,
            extra=_non_video_phase_meta("start"),
            force=True,
        )

    if detector_choice == "off":
        cuts: list[int] = []
    elif detector_choice == "pyscenedetect":
        cuts = detect_scene_cuts_pyscenedetect(
            video_path,
            threshold=threshold_value,
            min_len=min_len,
        )
    else:
        cuts = _detect_scene_cuts_histogram(
            video_path,
            thr=threshold_value,
            min_len=min_len,
            progress=progress,
        )

    if progress:
        total_frames = progress.target_frames or (cuts[-1] if cuts else 0)
        summary_done = {"count": len(cuts), "detector": detector_choice}
        if cuts:
            summary_done["first_cut"] = cuts[0]
        progress.emit(
            total_frames,
            phase="scene_detect:cut",
            summary=summary_done,
            extra=_non_video_phase_meta(),
        )
        progress.emit(
            total_frames,
            phase="scene_detect:done",
            summary=summary_done,
            force=True,
            extra=_non_video_phase_meta("done"),
        )
    return cuts


def _run_full_pipeline(
    args: argparse.Namespace,
    video_dest: Path,
    *,
    source_fps: float,
    progress: ProgressEmitter | None = None,
    target_fps: float | None = None,
    frame_exporter: FrameExporter | None = None,
    total_frames: int | None = None,
    video_fps: float | None = None,
) -> Tuple[
    int,
    int,
    int,
    str,
    str,
    float | None,
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, int],
    Dict[str, Any] | None,
    Dict[str, Any] | None,
]:
    import cv2  # type: ignore

    analyzed_fps = target_fps or source_fps
    if not analyzed_fps or analyzed_fps <= 0:
        analyzed_fps = _detect_fps(video_dest)

    # Detection/tracking stride (default: 1 = every frame for 24fps Bravo episodes)
    # This controls how often detection runs, NOT how many samples get embedded/exported.
    # Per-track sampling for embedding/export is controlled separately via
    # --max-samples-per-track, --min-samples-per-track, --sample-every-n-frames
    frame_stride = _effective_stride(args.stride, target_fps or analyzed_fps, source_fps)
    ts_fps = analyzed_fps if analyzed_fps and analyzed_fps > 0 else max(args.fps or 30.0, 1.0)

    # Initialize phase tracker for local mode instrumentation
    phase_tracker = PhaseTracker() if LOCAL_MODE_INSTRUMENTATION else None

    # Log structured config for Local mode diagnostics
    if LOCAL_MODE_INSTRUMENTATION:
        cpu_threads = os.environ.get("SCREENALYTICS_MAX_CPU_THREADS", "auto")
        save_crops = getattr(args, "save_crops", False)
        save_frames = getattr(args, "save_frames", False)
        detection_fps_limit = getattr(args, "fps", None)
        min_face_size = getattr(args, "min_box_area", MIN_FACE_AREA)
        config_lines = [
            f"[LOCAL MODE] detect_track config:",
            f"  device={args.device}, profile=balanced",
            f"  frame_stride={frame_stride} (requested={args.stride})",
            f"  detection_fps_limit={detection_fps_limit}",
            f"  min_face_size={min_face_size}",
            f"  cpu_threads={cpu_threads}",
            f"  save_crops={save_crops}, save_frames={save_frames}",
            f"  total_frames={total_frames or 'unknown'}",
        ]
        for line in config_lines:
            LOGGER.info(line)
            print(line)
    frames_goal = None
    if total_frames and total_frames > 0:
        frames_goal = int(total_frames)
    elif progress and progress.target_frames:
        frames_goal = progress.target_frames
    video_clock_fps = video_fps if video_fps and video_fps > 0 else (source_fps if source_fps > 0 else None)
    frame_exporter: FrameExporter | None = None

    def _progress_value(
        frame_index: int, *, include_current: bool = False, step: str | None = None
    ) -> tuple[int, Dict[str, Any]]:
        base = frame_index + (1 if include_current else 0)
        if base < 0:
            base = 0
        total = frames_goal or base
        value = base
        if frames_goal:
            value = min(frames_goal, base)
        meta = _video_phase_meta(
            value,
            total if total > 0 else None,
            video_clock_fps,
            step=step,
            crop_diag_source=frame_exporter,
        )
        return value, meta

    device = pick_device(args.device)
    onnx_provider_requested = _onnx_provider_label(args.device)
    torch_device_requested, torch_device_resolved, torch_device_reason = _resolve_torch_device_request(args.device)
    detector_choice = _normalize_detector_choice(getattr(args, "detector", None))
    tracker_choice = _normalize_tracker_choice(getattr(args, "tracker", None))
    args.detector = detector_choice
    args.tracker = tracker_choice
    det_thresh = _normalize_det_thresh(getattr(args, "det_thresh", RETINAFACE_SCORE_THRESHOLD))
    args.det_thresh = det_thresh

    # Determine CPU fallback policy from CLI flags
    # --allow-cpu-fallback OR --no-coreml-only enables CPU fallback
    allow_cpu_fallback = getattr(args, "allow_cpu_fallback", False) or not getattr(args, "coreml_only", True)

    detector_backend = _build_face_detector(
        detector_choice,
        device,
        det_thresh,
        coreml_input_size=getattr(args, "coreml_det_size", None),
        allow_cpu_fallback=allow_cpu_fallback,
    )
    detector_backend.ensure_ready()
    detector_device = getattr(detector_backend, "resolved_device", device)

    # Initialize person fallback detector if enabled (torch-backed)
    person_fallback_detector = _build_person_fallback_detector(torch_device_resolved)
    if person_fallback_detector:
        if torch_device_reason:
            LOGGER.info(
                "Person fallback detector initialized (torch_device=%s, reason=%s)",
                torch_device_resolved,
                torch_device_reason,
            )
        else:
            LOGGER.info("Person fallback detector initialized (torch_device=%s)", torch_device_resolved)

    tracker_label = tracker_choice
    if progress:
        start_frames, video_meta = _progress_value(-1, include_current=True)
        progress.emit(
            start_frames,
            phase="detect",
            device=device,
            detector=detector_choice,
            tracker=tracker_label,
            resolved_device=detector_device,
            force=True,
            extra=video_meta,
        )

    tracker_config = _bytetrack_config_from_args(args) if tracker_choice == "bytetrack" else None
    tracker_adapter = _build_tracker_adapter(
        tracker_choice,
        frame_rate=source_fps or 30.0,
        stride=frame_stride,
        config=tracker_config,
    )
    tracker_config_summary: Dict[str, Any] | None = None
    if tracker_choice == "bytetrack":
        tracker_config_summary = tracker_adapter.config_summary
    appearance_gate: AppearanceGate | None = None
    gate_embedder: ArcFaceEmbedder | None = None
    gate_config: GateConfig | None = None
    gate_embed_stride = frame_stride
    gate_enabled = bool(getattr(args, "gate_enabled", GATE_ENABLED_DEFAULT))
    if tracker_choice == "bytetrack" and gate_enabled:
        gate_config = _gate_config_from_args(args, frame_stride)
        appearance_gate = AppearanceGate(gate_config)
        gate_embed_stride = gate_config.emb_every or frame_stride
        try:
            gate_embedder = ArcFaceEmbedder(device, allow_cpu_fallback=allow_cpu_fallback)
            gate_embedder.ensure_ready()
        except Exception as exc:
            raise RuntimeError(
                "ArcFace gate embedder is required for ByteTrack gating. "
                "Install the ArcFace weights via scripts/fetch_models.py or rerun with --tracker strongsort."
            ) from exc
    elif tracker_choice == "bytetrack":
        LOGGER.info("[gate] gate_enabled=false; running ByteTrack without appearance-based splits")
    scene_detector_choice = _normalize_scene_detector_choice(getattr(args, "scene_detector", None))
    args.scene_detector = scene_detector_choice
    scene_threshold = max(float(getattr(args, "scene_threshold", SCENE_THRESHOLD_DEFAULT)), 0.0)
    scene_min_len = max(int(getattr(args, "scene_min_len", SCENE_MIN_LEN_DEFAULT)), 1)
    scene_warmup = max(int(getattr(args, "scene_warmup_dets", SCENE_WARMUP_DETS_DEFAULT)), 0)
    scene_cuts: list[int] = []
    scene_detect_start = time.time()
    if scene_detector_choice != "off":
        scene_cuts = detect_scene_cuts(
            str(video_dest),
            detector=scene_detector_choice,
            thr=scene_threshold,
            min_len=scene_min_len,
            progress=progress,
        )
    scene_detect_duration = time.time() - scene_detect_start

    # Record scene detection phase stats for local mode instrumentation
    if phase_tracker and scene_detector_choice != "off":
        phase_tracker.add_phase_stats(
            "scene_detect",
            frames_processed=total_frames or frames_goal or 0,
            frames_scanned=total_frames or frames_goal or 0,
            stride=1,  # Scene detection always processes all frames
            duration_seconds=scene_detect_duration,
        )
        scene_msg = (
            f"[LOCAL MODE] Phase=scene_detect frames={total_frames or frames_goal or '?'} "
            f"stride=1 wall_time_s={scene_detect_duration:.1f}s cuts={len(scene_cuts)}"
        )
        LOGGER.info(scene_msg)
        print(scene_msg)

    scene_summary = {
        "count": len(scene_cuts),
        "indices": scene_cuts,
        "detector": scene_detector_choice,
    }
    cut_ix = 0
    next_cut = scene_cuts[cut_ix] if scene_cuts else None
    frames_since_cut = 10**9
    warmup_window = 0
    warmup_cuts_applied = 0
    warmup_cap_total: int | None = None
    if scene_warmup > 0 and SCENE_WARMUP_CAP_RATIO_DEFAULT > 0 and frame_stride > 0:
        expected_total_frames = frames_goal or total_frames or 0
        expected_stride_hits_est = int(math.ceil(expected_total_frames / frame_stride)) if expected_total_frames > 0 else 0
        warmup_cap_total = int(math.floor(SCENE_WARMUP_CAP_RATIO_DEFAULT * expected_stride_hits_est))
    max_gap_frames = _resolved_max_gap(args.max_gap, analyzed_fps)
    if max_gap_frames != max(1, int(args.max_gap)):
        LOGGER.info(
            "Track max_gap capped to %s frames (configured=%s analyzed_fps=%.2f)",
            max_gap_frames,
            args.max_gap,
            analyzed_fps or 0.0,
        )
    recorder = TrackRecorder(max_gap=max_gap_frames, remap_ids=True)
    run_id = getattr(args, "run_id", None)
    det_path = _detections_path_for_run(args.ep_id, run_id)
    det_path.parent.mkdir(parents=True, exist_ok=True)
    track_path = _tracks_path_for_run(args.ep_id, run_id)
    det_count = 0
    frames_sampled = 0
    frames_sampled_stride = 0
    frames_sampled_forced_scene_warmup = 0
    frame_idx = 0
    last_diag_stats: Dict[str, Any] | None = None
    processed_frame_indices: list[int] = []

    def _diagnostic_stats(detections_in_frame: int, tracks_in_frame: int) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "frames_seen": frames_sampled,
            "detections_seen": det_count,
            "detections_in_frame": detections_in_frame,
            "tracks_in_frame": tracks_in_frame,
            "tracks_born": recorder.metrics["tracks_born"],
            "tracks_alive": recorder.active_track_count,
            "tracks_lost": recorder.metrics["tracks_lost"],
        }
        if tracker_config_summary:
            stats.setdefault("tracker_config", tracker_config_summary)
        return stats

    # Track detection confidence distribution for diagnostics
    detection_conf_hist = {
        "0.0-0.5": 0,
        "0.5-0.6": 0,
        "0.6-0.7": 0,
        "0.7-0.8": 0,
        "0.8-0.9": 0,
        "0.9-1.0": 0,
    }

    def _record_detection_conf(conf_value: float) -> None:
        if conf_value < 0.5:
            detection_conf_hist["0.0-0.5"] += 1
        elif conf_value < 0.6:
            detection_conf_hist["0.5-0.6"] += 1
        elif conf_value < 0.7:
            detection_conf_hist["0.6-0.7"] += 1
        elif conf_value < 0.8:
            detection_conf_hist["0.7-0.8"] += 1
        elif conf_value < 0.9:
            detection_conf_hist["0.8-0.9"] += 1
        else:
            detection_conf_hist["0.9-1.0"] += 1

    cap = cv2.VideoCapture(str(video_dest))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video {video_dest}")

    # Start timing for detect/track phase
    detect_track_start = time.time()

    try:
        with det_path.open("w", encoding="utf-8") as det_handle:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # Guard against empty/None frames before detection
                if frame is None or frame.size == 0:
                    LOGGER.warning(
                        "Skipping frame %d for %s: empty or None frame from video capture",
                        frame_idx,
                        args.ep_id,
                    )
                    frame_idx += 1
                    frames_since_cut += 1
                    continue

                if next_cut is not None and frame_idx >= next_cut:
                    reset_tracker = getattr(tracker_adapter, "reset", None)
                    if callable(reset_tracker):
                        reset_tracker()
                    if appearance_gate:
                        appearance_gate.reset_all()
                    recorder.on_cut(frame_idx)
                    frames_since_cut = 0
                    # Only warm up when the cut creates a stride gap (i.e., sampling coverage drops).
                    # Warmup covers the non-stride frames until the next stride hit, bounded by config + cap ratio.
                    if scene_warmup > 0 and frame_stride > 0:
                        gap_to_next_stride = (frame_stride - (frame_idx % frame_stride)) % frame_stride
                        warmup_window = min(scene_warmup, gap_to_next_stride) if gap_to_next_stride > 0 else 0
                        if warmup_window > 0:
                            warmup_cuts_applied += 1
                    else:
                        warmup_window = 0
                    cut_ix += 1
                    next_cut = scene_cuts[cut_ix] if cut_ix < len(scene_cuts) else None
                    if progress:
                        emit_frames, video_meta = _progress_value(frame_idx, include_current=False)
                        progress.emit(
                            emit_frames,
                            phase="track",
                            device=device,
                            detector=detector_choice,
                            tracker=tracker_label,
                            resolved_device=detector_device,
                            summary={"event": "reset_on_cut", "frame": frame_idx},
                            force=True,
                            extra=video_meta,
                        )
                should_sample = frame_idx % frame_stride == 0
                force_detect = (not should_sample) and warmup_window > 0 and frames_since_cut < warmup_window
                if warmup_cap_total is not None and frames_sampled_forced_scene_warmup >= warmup_cap_total:
                    force_detect = False
                if not (should_sample or force_detect):
                    frame_idx += 1
                    frames_since_cut += 1
                    continue

                frames_sampled += 1
                if should_sample:
                    frames_sampled_stride += 1
                else:
                    frames_sampled_forced_scene_warmup += 1
                processed_frame_indices.append(frame_idx)
                detect_frames, detect_meta = _progress_value(frame_idx, include_current=True)
                ts = frame_idx / ts_fps if ts_fps else 0.0

                # === BEGIN per-frame detect/track/crop guard ===
                # Wrap core detect/track/crop logic in targeted TypeError guard to handle
                # NoneType multiplication errors from malformed bboxes or margins.
                # If a frame fails with NoneType multiply error, skip it and continue processing.

                # Quarantine: Capture state for diagnostics if TypeError occurs
                quarantine_detections: list[tuple[np.ndarray, float]] = []
                quarantine_tracks: list[tuple[int, np.ndarray, float]] = []
                quarantine_stage = "init"

                try:
                    # DEBUG: Trace execution at frame start
                    if frames_sampled < 5:
                        LOGGER.error(
                            "[DEBUG] Frame %d START: entering detect/track/crop block",
                            frame_idx,
                        )

                    quarantine_stage = "detection"

                    try:
                        detections = detector_backend.detect(frame)
                        # DEBUG: Show detection count
                        if frames_sampled < 5:
                            LOGGER.error(
                                "[DEBUG] Frame %d: detector returned %d detections",
                                frame_idx,
                                len(detections),
                            )
                    except Exception as exc:
                        LOGGER.error(
                            "Face detection failed at frame %d for %s: %s",
                            frame_idx,
                            args.ep_id,
                            exc,
                            exc_info=True,
                        )
                        raise RuntimeError(f"Face detection failed at frame {frame_idx}") from exc
                    face_detections = [sample for sample in detections if sample.class_label == FACE_CLASS_LABEL]

                    # Person fallback: if face detection found few/no faces, try body detection
                    if PERSON_FALLBACK_ENABLED and person_fallback_detector is not None and len(face_detections) == 0:
                        person_fallback_detections = person_fallback_detector.detect_persons(frame)
                        if person_fallback_detections:
                            LOGGER.debug(
                                "Frame %d: face detection found 0 faces, person fallback found %d estimated faces",
                                frame_idx,
                                len(person_fallback_detections),
                            )
                            # Add estimated face detections from person detection
                            face_detections.extend(person_fallback_detections)

                    # DEBUG: Show face detection count
                    if frames_sampled < 5:
                        LOGGER.error(
                            "[DEBUG] Frame %d: filtered to %d face detections",
                            frame_idx,
                            len(face_detections),
                        )

                    # Validate detection bboxes before tracking to prevent NoneType multiply errors
                    validated_detections = []
                    invalid_bbox_count = 0
                    for det_sample in face_detections:
                        _record_detection_conf(float(det_sample.conf))

                        # Validate bbox coordinates
                        validated_bbox, bbox_err = _safe_bbox_or_none(det_sample.bbox)
                        if validated_bbox is None:
                            invalid_bbox_count += 1
                            LOGGER.warning(
                                "Dropping detection at frame %d for %s: invalid bbox (%s) - bbox=%s",
                                frame_idx,
                                args.ep_id,
                                bbox_err,
                                det_sample.bbox,
                            )
                            continue

                        # Update detection with validated bbox
                        det_sample.bbox = np.array(validated_bbox)
                        validated_detections.append(det_sample)

                    if invalid_bbox_count > 0:
                        LOGGER.info(
                            "Frame %d for %s: dropped %d/%d detections due to invalid bboxes",
                            frame_idx,
                            args.ep_id,
                            invalid_bbox_count,
                            len(face_detections),
                        )

                    # DEBUG: Show validation results
                    if frames_sampled < 5:
                        LOGGER.error(
                            "[DEBUG] Frame %d: after detection bbox validation: %d valid, %d invalid",
                            frame_idx,
                            len(validated_detections),
                            invalid_bbox_count,
                        )

                    # Quarantine: Capture validated detections for diagnostics
                    quarantine_detections = [(det.bbox, float(det.conf)) for det in validated_detections[:10]]
                    quarantine_stage = "tracking"

                    # Wrap tracker update in specific try/except to catch NoneType multiply errors
                    try:
                        raw_tracked_objects = tracker_adapter.update(validated_detections, frame_idx, frame)
                    except TypeError as e:
                        msg = str(e)
                        if "NoneType" in msg and "*" in msg:
                            LOGGER.error(
                                "[TRACKER ERROR] Frame %d: NoneType multiply in tracker_adapter.update() with %d validated detections: %s",
                                frame_idx,
                                len(validated_detections),
                                msg,
                            )
                            # Log first few detection bboxes for diagnosis
                            for i, det in enumerate(validated_detections[:3]):
                                LOGGER.error(
                                    "[TRACKER ERROR] Detection %d/%d: bbox=%s conf=%.3f",
                                    i + 1,
                                    len(validated_detections),
                                    det.bbox,
                                    det.conf,
                                )
                            raise  # Re-raise to be caught by outer per-frame guard
                        raise

                    # DEBUG: Show tracker output
                    if frames_sampled < 5:
                        LOGGER.error(
                            "[DEBUG] Frame %d: tracker returned %d raw tracked objects",
                            frame_idx,
                            len(raw_tracked_objects),
                        )

                    # Validate tracked object bboxes (ByteTrack may return invalid bboxes)
                    tracked_objects = []
                    invalid_track_count = 0
                    for track_obj in raw_tracked_objects:
                        validated_track_bbox, track_bbox_err = _safe_bbox_or_none(track_obj.bbox)
                        if validated_track_bbox is None:
                            invalid_track_count += 1
                            LOGGER.warning(
                                "Dropping tracked object %s at frame %d for %s: invalid bbox (%s) - bbox=%s",
                                track_obj.track_id,
                                frame_idx,
                                args.ep_id,
                                track_bbox_err,
                                track_obj.bbox,
                            )
                            continue
                        # Update track object with validated bbox
                        track_obj.bbox = np.array(validated_track_bbox)
                        tracked_objects.append(track_obj)

                    if invalid_track_count > 0:
                        LOGGER.warning(
                            "Frame %d for %s: dropped %d/%d tracked objects due to invalid bboxes",
                            frame_idx,
                            args.ep_id,
                            invalid_track_count,
                            len(raw_tracked_objects),
                        )

                    # DEBUG: Log tracked object validation results
                    if frames_sampled < 5:
                        LOGGER.error(
                            "[DEBUG] Frame %d: after tracked object bbox validation: %d valid, %d invalid",
                            frame_idx,
                            len(tracked_objects),
                            invalid_track_count,
                        )

                    # Quarantine: Capture validated tracks for diagnostics
                    quarantine_tracks = [
                        (
                            obj.track_id,
                            obj.bbox,
                            float(obj.conf) if obj.conf is not None else 0.0,
                        )
                        for obj in tracked_objects[:10]
                    ]
                    quarantine_stage = "gate_and_crop"

                    diag_stats = _diagnostic_stats(len(validated_detections), len(tracked_objects))
                    # Preserve skipped_none_multiply counter from previous iterations
                    if last_diag_stats and "skipped_none_multiply" in last_diag_stats:
                        diag_stats["skipped_none_multiply"] = last_diag_stats["skipped_none_multiply"]
                    last_diag_stats = diag_stats
                    if progress:
                        detect_extra = dict(detect_meta)
                        detect_extra["detect_track_stats"] = diag_stats
                        progress.emit(
                            detect_frames,
                            phase="detect",
                            device=device,
                            detector=detector_choice,
                            tracker=tracker_label,
                            resolved_device=detector_device,
                            extra=detect_extra,
                        )

                    if frames_sampled > 0 and frames_sampled % TRACKING_DIAG_INTERVAL == 0:
                        if tracker_config_summary:
                            LOGGER.info(
                                "Tracking diag ep=%s frame=%d sampled=%d detections_total=%d tracks_alive=%d "
                                "tracks_born=%d detections_frame=%d tracks_frame=%d stride=%d "
                                "track_high=%.2f new_track=%.2f match_thresh=%.2f track_buffer=%d min_box_area=%.2f",
                                args.ep_id,
                                frame_idx,
                                frames_sampled,
                                det_count,
                                recorder.active_track_count,
                                recorder.metrics["tracks_born"],
                                len(validated_detections),
                                len(tracked_objects),
                                frame_stride,
                                tracker_config_summary.get("track_high_thresh", 0.0),
                                tracker_config_summary.get("new_track_thresh", 0.0),
                                tracker_config_summary.get("match_thresh", 0.0),
                                tracker_config_summary.get("track_buffer", 0),
                                tracker_config_summary.get("min_box_area", 0.0),
                            )
                        else:
                            LOGGER.info(
                                "Tracking diag ep=%s frame=%d sampled=%d detections_total=%d tracks_alive=%d "
                                "tracks_born=%d detections_frame=%d tracks_frame=%d stride=%d",
                                args.ep_id,
                                frame_idx,
                                frames_sampled,
                                det_count,
                                recorder.active_track_count,
                                recorder.metrics["tracks_born"],
                                len(validated_detections),
                                len(tracked_objects),
                                frame_stride,
                            )

                    crop_records: list[tuple[int, list[float]]] = []
                    gate_embeddings: dict[int, np.ndarray | None] = {}
                    should_embed_gate = False
                    # Tracks that MUST have embeddings computed (new tracks without prototypes)
                    # This is critical to prevent multi-person track contamination
                    tracks_needing_embed: set[int] = set()
                    if appearance_gate:
                        should_embed_gate = True if frames_since_cut < scene_warmup else False
                        stride_for_gate = gate_embed_stride or frame_stride
                        if stride_for_gate > 1 and not should_embed_gate:
                            should_embed_gate = frame_idx % stride_for_gate == 0
                        else:
                            should_embed_gate = True

                        # CRITICAL FIX: Always compute embeddings for new tracks without prototypes
                        # This prevents the bug where a track starts on a non-embedding frame,
                        # then a different person enters on an embedding frame and becomes the prototype
                        tracks_needing_embed = appearance_gate.tracks_needing_embedding(
                            [obj.track_id for obj in tracked_objects]
                        )
                        if tracks_needing_embed and not should_embed_gate:
                            LOGGER.debug(
                                "[gate] Force embedding for %d new track(s) at frame %d: %s",
                                len(tracks_needing_embed),
                                frame_idx,
                                list(tracks_needing_embed)[:5],  # Log first 5 to avoid spam
                            )

                        if (should_embed_gate or tracks_needing_embed) and gate_embedder and tracked_objects:
                            # DEBUG: Show gate embedding processing
                            if frames_sampled < 5:
                                LOGGER.error(
                                    "[DEBUG] Frame %d: processing gate embeddings for %d tracks",
                                    frame_idx,
                                    len(tracked_objects),
                                )

                            embed_inputs: list[np.ndarray] = []
                            embed_track_ids: list[int] = []
                            for obj in tracked_objects:
                                # Skip tracks that don't need embedding on non-embedding frames
                                if not should_embed_gate and obj.track_id not in tracks_needing_embed:
                                    continue
                                # Validate bbox before cropping to prevent NoneType multiply errors
                                validated_bbox, bbox_err = _safe_bbox_or_none(obj.bbox)
                                if validated_bbox is None:
                                    LOGGER.debug(
                                        "Gate embedding skipped for track %s frame %d: invalid bbox (%s)",
                                        obj.track_id,
                                        frame_idx,
                                        bbox_err,
                                    )
                                    continue

                                landmarks_list = None
                                if obj.landmarks is not None:
                                    landmarks_list = (
                                        obj.landmarks.tolist()
                                        if isinstance(obj.landmarks, np.ndarray)
                                        else obj.landmarks
                                    )
                                crop, crop_err = _prepare_face_crop(
                                    frame,
                                    validated_bbox,
                                    landmarks_list,
                                    margin=0.2,
                                )
                                if crop is None:
                                    if crop_err:
                                        LOGGER.debug(
                                            "Gate crop failed for track %s: %s",
                                            obj.track_id,
                                            crop_err,
                                        )
                                    continue
                                aligned = _resize_for_arcface(crop)
                                embed_inputs.append(aligned)
                                embed_track_ids.append(obj.track_id)
                            if embed_inputs:
                                encoded = gate_embedder.encode(embed_inputs)
                                for idx, tid in enumerate(embed_track_ids):
                                    embedding_vec = encoded[idx]
                                    # Validate embedding contains finite values before storing
                                    try:
                                        if embedding_vec is not None and np.all(np.isfinite(embedding_vec)):
                                            gate_embeddings[tid] = embedding_vec
                                        else:
                                            gate_embeddings[tid] = None
                                            if embedding_vec is not None:
                                                LOGGER.warning(
                                                    "Gate embedding for track %s at frame %d contains invalid values (NaN/Inf/None), discarding",
                                                    tid,
                                                    frame_idx,
                                                )
                                    except (TypeError, ValueError):
                                        # Embedding contains None or non-numeric values
                                        gate_embeddings[tid] = None
                                        LOGGER.warning(
                                            "Gate embedding for track %s at frame %d is not a valid numeric array, discarding",
                                            tid,
                                            frame_idx,
                                        )
                        for obj in tracked_objects:
                            gate_embeddings.setdefault(obj.track_id, None)
                        # TODO(perf): Persist gate_embeddings to reuse in faces embed stage.
                        # Would require: (1) extending tracks.jsonl schema to include gate_embedding,
                        # (2) saving embeddings in TrackRecorder, (3) loading in faces embed, (4)
                        # matching gate embedding to appropriate face sample per track. Requires
                        # invasive schema changes and careful handling of missing/mismatched cases.

                    if not validated_detections:
                        if frame_exporter and frame_exporter.save_frames:
                            frame_exporter.export(frame_idx, frame, [], ts=ts)
                        frame_idx += 1
                        frames_since_cut += 1
                        continue

                    active_ids: set[int] = set()
                    for obj in tracked_objects:
                        active_ids.add(obj.track_id)
                        class_value = FACE_CLASS_LABEL
                        landmarks = None
                        if obj.landmarks is not None:
                            landmarks = (
                                obj.landmarks.tolist() if isinstance(obj.landmarks, np.ndarray) else obj.landmarks
                            )
                        force_split = False
                        if appearance_gate:
                            embedding_vec = gate_embeddings.get(obj.track_id)
                            force_split, _, _, _ = appearance_gate.process(
                                obj.track_id,
                                obj.bbox,
                                embedding_vec,
                                frame_idx,
                            )
                        export_id = recorder.record(
                            tracker_track_id=obj.track_id,
                            frame_idx=frame_idx,
                            ts=ts,
                            bbox=obj.bbox,
                            class_label=class_value,
                            landmarks=landmarks,
                            confidence=(float(obj.conf) if obj.conf is not None else None),
                            force_new_track=force_split,
                        )
                        bbox_list = [round(float(coord), 4) for coord in obj.bbox.tolist()]
                        conf_value = float(obj.conf)

                        row = {
                            "ep_id": args.ep_id,
                            "ts": round(float(ts), 4),
                            "frame_idx": frame_idx,
                            "class": class_value,
                            "conf": round(conf_value, 4),
                            "bbox_xyxy": bbox_list,
                            "track_id": export_id,
                            "model": detector_backend.model_name,
                            "detector": detector_choice,
                            "tracker": tracker_label,
                            "pipeline_ver": PIPELINE_VERSION,
                            "fps": (round(float(analyzed_fps), 4) if analyzed_fps else None),
                        }
                        if landmarks:
                            row["landmarks"] = [round(float(val), 4) for val in landmarks]
                        det_handle.write(json.dumps(row) + "\n")
                        det_count += 1
                        if frame_exporter and frame_exporter.save_crops:
                            crop_records.append((export_id, bbox_list))

                    if appearance_gate:
                        appearance_gate.prune(active_ids)

                    if frame_exporter and (frame_exporter.save_frames or crop_records):
                        frame_exporter.export(frame_idx, frame, crop_records, ts=ts)

                    # DEBUG: Frame processing completed successfully
                    if frames_sampled < 5:
                        LOGGER.error(
                            "[DEBUG] Frame %d END: completed all detect/track/crop processing successfully",
                            frame_idx,
                        )

                    # === END per-frame detect/track/crop guard ===
                except TypeError as e:
                    # Only catch NoneType multiplication errors from malformed bboxes/margins
                    msg = str(e)
                    if "NoneType" in msg and "*" in msg:
                        # Log with full traceback to identify exact location
                        import traceback

                        tb_str = traceback.format_exc()

                        # Build quarantine report with specific detection/track data
                        quarantine_report = {
                            "frame": frame_idx,
                            "stage": quarantine_stage,
                            "error": msg,
                            "detections_count": len(quarantine_detections),
                            "tracks_count": len(quarantine_tracks),
                        }

                        # Log detailed quarantine information with full traceback
                        LOGGER.error(
                            "[QUARANTINE] ❌ NoneType multiply at frame %d (stage=%s) for %s: %s",
                            frame_idx,
                            quarantine_stage,
                            args.ep_id,
                            msg,
                            exc_info=True,
                        )

                        # Log detection bboxes that were in play
                        if quarantine_detections:
                            LOGGER.error(
                                "[QUARANTINE] Detections (%d) at frame %d:",
                                len(quarantine_detections),
                                frame_idx,
                            )
                            for idx, (bbox, conf) in enumerate(quarantine_detections[:5]):
                                bbox_safe = bbox.tolist() if hasattr(bbox, "tolist") else bbox
                                LOGGER.error(
                                    "[QUARANTINE]   Det %d: bbox=%s conf=%.3f",
                                    idx,
                                    bbox_safe,
                                    conf,
                                )
                            quarantine_report["sample_detections"] = [
                                {
                                    "bbox": (bbox.tolist() if hasattr(bbox, "tolist") else bbox),
                                    "conf": conf,
                                }
                                for bbox, conf in quarantine_detections[:5]
                            ]

                        # Log track bboxes that were in play
                        if quarantine_tracks:
                            LOGGER.error(
                                "[QUARANTINE] Tracks (%d) at frame %d:",
                                len(quarantine_tracks),
                                frame_idx,
                            )
                            for idx, (track_id, bbox, conf) in enumerate(quarantine_tracks[:5]):
                                bbox_safe = bbox.tolist() if hasattr(bbox, "tolist") else bbox
                                LOGGER.error(
                                    "[QUARANTINE]   Track %d (tid=%d): bbox=%s conf=%.3f",
                                    idx,
                                    track_id,
                                    bbox_safe,
                                    conf,
                                )
                            quarantine_report["sample_tracks"] = [
                                {
                                    "track_id": tid,
                                    "bbox": (bbox.tolist() if hasattr(bbox, "tolist") else bbox),
                                    "conf": conf,
                                }
                                for tid, bbox, conf in quarantine_tracks[:5]
                            ]

                        # Traceback already logged via exc_info=True above, but also log formatted version
                        LOGGER.error("[QUARANTINE] Formatted traceback:\n%s", tb_str)

                        # Track crop errors for diagnostics
                        if last_diag_stats is None:
                            last_diag_stats = _diagnostic_stats(0, 0)
                        skipped = last_diag_stats.get("skipped_none_multiply", 0)
                        last_diag_stats["skipped_none_multiply"] = skipped + 1

                        # Emit quarantine event via progress so it's visible in UI/Health page
                        if progress:
                            quarantine_frames, quarantine_meta = _progress_value(frame_idx, include_current=False)
                            quarantine_extra = dict(quarantine_meta)
                            quarantine_extra["quarantine"] = quarantine_report
                            quarantine_extra["skipped_none_multiply"] = last_diag_stats["skipped_none_multiply"]
                            progress.emit(
                                quarantine_frames,
                                phase="track",
                                device=device,
                                detector=detector_choice,
                                tracker=tracker_label,
                                resolved_device=detector_device,
                                summary={
                                    "event": "none_multiply_skip",
                                    "frame": frame_idx,
                                    "stage": quarantine_stage,
                                },
                                force=True,
                                extra=quarantine_extra,
                            )

                        # Reset tracker state to clear corrupted state (similar to scene-cut reset)
                        LOGGER.warning(
                            "[QUARANTINE] Resetting tracker and appearance gate state at frame %d to clear corrupted state",
                            frame_idx,
                        )
                        reset_tracker = getattr(tracker_adapter, "reset", None)
                        if callable(reset_tracker):
                            reset_tracker()
                        if appearance_gate:
                            appearance_gate.reset_all()

                        frame_idx += 1
                        frames_since_cut += 1
                        continue
                    # Re-raise if it's a different TypeError
                    raise

                if progress:
                    track_frames, track_meta = _progress_value(frame_idx, include_current=True)
                    track_extra = dict(track_meta)
                    if last_diag_stats:
                        track_extra["detect_track_stats"] = last_diag_stats
                    progress.emit(
                        track_frames,
                        phase="track",
                        device=device,
                        detector=detector_choice,
                        tracker=tracker_label,
                        resolved_device=detector_device,
                        summary={
                            "tracks_born": recorder.metrics["tracks_born"],
                            "tracks_lost": recorder.metrics["tracks_lost"],
                            "id_switches": recorder.metrics["id_switches"],
                            "forced_splits": recorder.metrics.get("forced_splits", 0),
                        },
                        extra=track_extra,
                    )
                frame_idx += 1
                frames_since_cut += 1
    finally:
        cap.release()

    # Calculate detect/track phase duration
    detect_track_duration = time.time() - detect_track_start
    stride_observed_median: int | None = None
    if len(processed_frame_indices) >= 2:
        deltas = [
            processed_frame_indices[idx] - processed_frame_indices[idx - 1]
            for idx in range(1, len(processed_frame_indices))
        ]
        deltas_arr = np.asarray(deltas, dtype=np.int32)
        if deltas_arr.size > 0:
            stride_observed_median = int(np.median(deltas_arr))

    video_duration_est_s: float | None = None
    if video_clock_fps and video_clock_fps > 0 and frame_idx > 0:
        video_duration_est_s = frame_idx / float(video_clock_fps)

    effective_fps_processing: float | None = None
    if detect_track_duration > 0 and frames_sampled > 0:
        effective_fps_processing = frames_sampled / detect_track_duration

    rtf: float | None = None
    if video_duration_est_s and video_duration_est_s > 0:
        rtf = detect_track_duration / video_duration_est_s

    expected_by_stride: int | None = None
    if frame_stride > 0 and frame_idx > 0:
        expected_by_stride = int(math.ceil(frame_idx / frame_stride))

    LOGGER.info(
        "[detect_track] video_duration_s=%s wall_time_s=%.1f rtf=%s effective_fps_processing=%s "
        "frames_scanned_total=%d face_detect_frames_processed=%d (stride_requested=%s stride_effective=%d stride_observed_median=%s "
        "stride_hits=%d forced_scene_warmup=%d expected_by_stride=%s scene_warmup_dets=%d scene_cuts=%d)",
        f"{video_duration_est_s:.3f}" if video_duration_est_s is not None else "unknown",
        detect_track_duration,
        f"{rtf:.3f}" if rtf is not None else "unknown",
        f"{effective_fps_processing:.3f}" if effective_fps_processing is not None else "unknown",
        frame_idx,
        frames_sampled,
        getattr(args, "stride", None),
        frame_stride,
        stride_observed_median if stride_observed_median is not None else "unknown",
        frames_sampled_stride,
        frames_sampled_forced_scene_warmup,
        expected_by_stride if expected_by_stride is not None else "unknown",
        scene_warmup,
        len(scene_cuts),
    )

    # Record detect/track phase stats for local mode instrumentation
    if phase_tracker:
        phase_tracker.add_phase_stats(
            "detect",
            frames_processed=frames_sampled,
            frames_scanned=frame_idx,
            stride=frame_stride,
            duration_seconds=detect_track_duration,
        )
        detect_msg = (
            f"[LOCAL MODE] Phase=detect frames_processed={frames_sampled} "
            f"frames_scanned={frame_idx} stride={frame_stride} wall_time_s={detect_track_duration:.1f}s"
        )
        LOGGER.info(detect_msg)
        print(detect_msg)

    # Guard: Ensure detect loop actually processed frames
    if frames_sampled == 0:
        error_msg = (
            f"Detect phase completed with 0 frames processed for {args.ep_id}. "
            f"This indicates video read failure, immediate loop exit, or misconfiguration. "
            f"Video path: {video_dest}, frame_idx reached: {frame_idx}, total_frames: {total_frames}"
        )
        LOGGER.error(error_msg)
        raise RuntimeError(error_msg)

    if progress and frame_idx > 0:
        detect_done_index = max(frame_idx - 1, 0)
        detect_done_frames, detect_done_meta = _progress_value(detect_done_index, include_current=True, step="done")
        if last_diag_stats:
            detect_done_meta = dict(detect_done_meta)
            detect_done_meta["detect_track_stats"] = last_diag_stats
        progress.emit(
            detect_done_frames,
            phase="detect",
            device=device,
            detector=detector_choice,
            tracker=tracker_label,
            resolved_device=detector_device,
            force=True,
            extra=detect_done_meta,
        )

    recorder.finalize()
    if frame_exporter:
        frame_exporter.write_indexes()
    track_rows = recorder.rows()
    final_diag_stats = last_diag_stats or _diagnostic_stats(0, 0)
    final_diag_stats["wall_time_s"] = round(float(detect_track_duration), 3)
    final_diag_stats["frames_scanned_total"] = int(frame_idx)
    final_diag_stats["face_detect_frames_processed"] = int(frames_sampled)
    final_diag_stats["face_detect_frames_processed_stride"] = int(frames_sampled_stride)
    final_diag_stats["face_detect_frames_processed_forced_scene_warmup"] = int(frames_sampled_forced_scene_warmup)
    final_diag_stats["stride_requested"] = getattr(args, "stride", None)
    final_diag_stats["stride_effective"] = int(frame_stride)
    final_diag_stats["stride_observed_median"] = stride_observed_median
    final_diag_stats["expected_frames_by_stride"] = expected_by_stride
    final_diag_stats["effective_fps_processing"] = round(float(effective_fps_processing), 3) if effective_fps_processing is not None else None
    final_diag_stats["rtf"] = round(float(rtf), 3) if rtf is not None else None
    final_diag_stats["video_duration_est_s"] = round(float(video_duration_est_s), 3) if video_duration_est_s is not None else None
    final_diag_stats["scene_detect_wall_time_s"] = round(float(scene_detect_duration), 3)
    final_diag_stats["scene_cuts_total"] = int(len(scene_cuts))
    final_diag_stats["scene_cut_count"] = int(len(scene_cuts))
    final_diag_stats["scene_warmup_dets"] = int(scene_warmup)
    if frames_sampled_stride > 0:
        final_diag_stats["forced_scene_warmup_ratio"] = round(frames_sampled_forced_scene_warmup / frames_sampled_stride, 6)
    else:
        final_diag_stats["forced_scene_warmup_ratio"] = None
    final_diag_stats["warmup_cuts_applied"] = int(warmup_cuts_applied)
    final_diag_stats["warmup_frames_per_cut_effective"] = (
        round(frames_sampled_forced_scene_warmup / max(warmup_cuts_applied, 1), 6) if warmup_cuts_applied else 0.0
    )
    final_diag_stats["wall_time_per_processed_frame_s"] = (
        round(float(detect_track_duration) / max(frames_sampled, 1), 9) if detect_track_duration is not None else None
    )

    # Device semantics: ONNX provider vs torch device (avoid "coreml" being treated as torch device).
    final_diag_stats["onnx_provider_requested"] = onnx_provider_requested
    final_diag_stats["onnx_provider_resolved"] = str(detector_device or "").strip() or None
    final_diag_stats["torch_device_requested"] = torch_device_requested
    final_diag_stats["torch_device_resolved"] = torch_device_resolved
    final_diag_stats["torch_device_fallback_reason"] = torch_device_reason

    # YOLO/person fallback diagnostics (torch-backed; should never receive "coreml" as a torch device).
    final_diag_stats["yolo_fallback_enabled"] = bool(PERSON_FALLBACK_ENABLED)
    if person_fallback_detector is None:
        final_diag_stats["yolo_fallback_device"] = None
        final_diag_stats["yolo_fallback_load_status"] = "disabled" if not PERSON_FALLBACK_ENABLED else "not_initialized"
        final_diag_stats["yolo_fallback_disabled_reason"] = "disabled" if not PERSON_FALLBACK_ENABLED else None
        final_diag_stats["yolo_fallback_invocations"] = 0
        final_diag_stats["yolo_fallback_detections_added"] = 0
    else:
        final_diag_stats["yolo_fallback_device"] = str(getattr(person_fallback_detector, "device", "")).strip() or None
        final_diag_stats["yolo_fallback_load_status"] = getattr(person_fallback_detector, "load_status", "unknown")
        final_diag_stats["yolo_fallback_disabled_reason"] = getattr(person_fallback_detector, "load_error", None)
        final_diag_stats["yolo_fallback_invocations"] = int(getattr(person_fallback_detector, "invocations", 0) or 0)
        final_diag_stats["yolo_fallback_detections_added"] = int(
            getattr(person_fallback_detector, "detections_emitted", 0) or 0
        )

    # CRITICAL: Validate that tracking produced results when detections exist
    if det_count > 0 and len(track_rows) == 0:
        cfg = tracker_config_summary or {}
        LOGGER.error(
            "ZERO TRACKS produced for ep_id=%s despite %d detections (tracks=%d). "
            "ByteTrack config → track_high=%.2f new_track=%.2f match_thresh=%.2f track_buffer=%d "
            "min_box_area=%.2f stride=%d det_thresh=%.2f. "
            "Lower the ByteTrack thresholds or rerun detect/track to rebuild tracks.",
            args.ep_id,
            det_count,
            len(track_rows),
            cfg.get("track_high_thresh", TRACK_HIGH_THRESH_DEFAULT),
            cfg.get("new_track_thresh", TRACK_NEW_THRESH_DEFAULT),
            cfg.get("match_thresh", BYTE_TRACK_MATCH_THRESH_DEFAULT),
            cfg.get("track_buffer", TRACK_BUFFER_BASE_DEFAULT),
            cfg.get("min_box_area", BYTE_TRACK_MIN_BOX_AREA_DEFAULT),
            frame_stride,
            args.det_thresh,
        )

    for row in track_rows:
        row["ep_id"] = args.ep_id
        row["detector"] = detector_choice
        row["tracker"] = tracker_label
    _write_jsonl(track_path, track_rows)
    metrics = {
        "tracks_born": recorder.metrics["tracks_born"],
        "tracks_lost": recorder.metrics["tracks_lost"],
        "id_switches": recorder.metrics["id_switches"],
        "forced_splits": recorder.metrics.get("forced_splits", 0),
        "longest_tracks": recorder.top_long_tracks(),
        "max_gap_frames": recorder.max_gap,
    }
    metrics["tracking_gate"] = {
        "enabled": bool(gate_enabled),
        "config": (
            {
                "appear_t_hard": getattr(gate_config, "appear_t_hard", None),
                "appear_t_soft": getattr(gate_config, "appear_t_soft", None),
                "appear_streak": getattr(gate_config, "appear_streak", None),
                "gate_iou": getattr(gate_config, "gate_iou", None),
                "proto_momentum": getattr(gate_config, "proto_momentum", None),
                "emb_every": getattr(gate_config, "emb_every", None),
            }
            if gate_config is not None
            else None
        ),
    }
    if tracker_config_summary:
        metrics["tracker_config"] = tracker_config_summary
    if final_diag_stats:
        metrics["detect_track_stats"] = final_diag_stats
    if appearance_gate:
        metrics["appearance_gate"] = appearance_gate.summary()
    if progress:
        final_track_index = max(frame_idx - 1, 0)
        track_done_frames, track_done_meta = _progress_value(final_track_index, include_current=True, step="done")
        if final_diag_stats:
            track_done_meta = dict(track_done_meta)
            track_done_meta["detect_track_stats"] = final_diag_stats
        progress.emit(
            track_done_frames,
            phase="track",
            device=device,
            detector=detector_choice,
            tracker=tracker_label,
            resolved_device=detector_device,
            summary=metrics,
            force=True,
            extra=track_done_meta,
        )

    # Log final phase summary for local mode instrumentation
    if phase_tracker:
        phase_tracker.log_summary(args.ep_id)

    return (
        det_count,
        len(track_rows),
        frames_sampled,
        device,
        detector_device,
        analyzed_fps,
        metrics,
        scene_summary,
        detection_conf_hist,
        final_diag_stats,
        tracker_config_summary,
    )


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _status_artifact_paths(ep_id: str, run_id: str, stage_key: str) -> dict[str, str]:
    try:
        artifacts = stage_artifacts(ep_id, run_id, stage_key)
    except Exception:
        return {}
    paths: dict[str, str] = {}
    for entry in artifacts:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        path = entry.get("path")
        if not isinstance(label, str) or not isinstance(path, str):
            continue
        if entry.get("exists"):
            paths[label] = path
    return paths


def _log_stage_for_phase(phase: str | None) -> str | None:
    if not phase:
        return None
    normalized = normalize_stage_key(phase)
    if normalized:
        return normalized
    phase_lower = str(phase).strip().lower()
    if phase_lower.startswith("scene_detect") or phase_lower in {"detect", "track"}:
        return "detect"
    return None


def _blocked_details(gate: GateResult) -> tuple[BlockedReason, StageBlockedInfo]:
    reasons = gate.reasons or [GateReason(code="blocked", message="Stage blocked by prerequisites", details=None)]
    primary = reasons[0]
    details = {
        "reasons": [reason.as_dict() for reason in reasons],
        "suggested_actions": list(gate.suggested_actions),
    }
    blocked_reason = BlockedReason(
        code=primary.code,
        message=primary.message,
        details=details,
    )
    blocked_info = StageBlockedInfo(reasons=list(reasons), suggested_actions=list(gate.suggested_actions))
    return blocked_reason, blocked_info


def _maybe_run_body_tracking(
    *,
    ep_id: str,
    run_id: str | None,
    effective_run_id: str | None,
    video_path: Path,
    import_status: Dict[str, Any] | None = None,
    emit_manifests: bool = True,
) -> Dict[str, Any] | None:
    config = _load_body_tracking_config()
    stage_run_id = effective_run_id or run_id
    if stage_run_id:
        gate = check_prereqs("body_tracking", ep_id, stage_run_id, config=config)
        if not gate.ok:
            blocked_reason, blocked_info = _blocked_details(gate)
            should_block = blocked_update_needed(ep_id, stage_run_id, "body_tracking", blocked_reason)
            if should_block:
                try:
                    try:
                        append_log(
                            ep_id,
                            stage_run_id,
                            "body_tracking",
                            "WARNING",
                            "stage blocked",
                            progress=0.0,
                            meta={
                                "reason_code": blocked_reason.code,
                                "reason_message": blocked_reason.message,
                                "suggested_actions": list(gate.suggested_actions),
                            },
                        )
                    except Exception as log_exc:  # pragma: no cover - best effort log write
                        LOGGER.debug("[run_logs] Failed to log body_tracking blocked: %s", log_exc)
                    write_stage_blocked(
                        ep_id,
                        stage_run_id,
                        "body_tracking",
                        blocked_reason,
                    )
                except Exception as status_exc:  # pragma: no cover - best effort status update
                    LOGGER.warning("[episode_status] Failed to mark body_tracking blocked: %s", status_exc)
                if emit_manifests:
                    try:
                        write_stage_manifest(
                            ep_id,
                            stage_run_id,
                            "body_tracking",
                            "BLOCKED",
                            started_at=_utcnow_iso(),
                            finished_at=_utcnow_iso(),
                            duration_s=None,
                            blocked=blocked_info,
                            thresholds={
                                "enabled": (config.get("body_tracking") or {}).get("enabled"),
                            },
                        )
                    except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                        LOGGER.warning("[manifest] Failed to write body_tracking blocked manifest: %s", manifest_exc)
            return None
    else:
        enabled = bool((config.get("body_tracking") or {}).get("enabled", False))
        if not enabled:
            return None

    output_dir = _body_tracking_dir_for_run(ep_id, run_id)
    config_path = REPO_ROOT / "config" / "pipeline" / "body_detection.yaml"
    fusion_config_path = REPO_ROOT / "config" / "pipeline" / "track_fusion.yaml"

    started_at = _utcnow_iso()
    stage_heartbeat = StageStatusHeartbeat(
        ep_id=ep_id,
        run_id=effective_run_id,
        stage_key="body_tracking",
        frames_total=1,
        started_at=started_at,
    )
    if effective_run_id:
        try:
            write_stage_started(
                ep_id,
                effective_run_id,
                "body_tracking",
                started_at=datetime.fromisoformat(started_at.replace("Z", "+00:00")),
            )
        except Exception as exc:  # pragma: no cover - best effort status update
            LOGGER.warning("[episode_status] Failed to mark body_tracking start: %s", exc)
        try:
            append_log(
                ep_id,
                effective_run_id,
                "body_tracking",
                "INFO",
                "stage started",
                progress=0.0,
                meta={"video_path": str(video_path)},
            )
        except Exception as log_exc:  # pragma: no cover - best effort log write
            LOGGER.debug("[run_logs] Failed to log body_tracking start: %s", log_exc)
    stage_heartbeat.update(
        done=0,
        phase="running_frames",
        message="Starting body tracking",
        force=True,
    )
    try:
        from FEATURES.body_tracking.src.body_tracking_runner import BodyTrackingRunner
    except Exception as exc:
        LOGGER.warning(
            "[body_tracking] Disabled due to import error (install deps to enable): %s",
            exc,
        )
        payload = {
            "phase": "body_tracking",
            "status": "error",
            "version": APP_VERSION,
            "run_id": effective_run_id,
            "started_at": started_at,
            "finished_at": _utcnow_iso(),
            "error": f"import_error: {exc}",
        }
        if effective_run_id:
            try:
                write_stage_failed(
                    ep_id,
                    effective_run_id,
                    "body_tracking",
                    error_code="import_error",
                    error_message=str(exc),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark body_tracking failure: %s", status_exc)
            if emit_manifests:
                try:
                    write_stage_manifest(
                        ep_id,
                        effective_run_id,
                        "body_tracking",
                        "FAILED",
                        started_at=started_at,
                        finished_at=payload.get("finished_at"),
                        duration_s=None,
                        error=StageErrorInfo(code="import_error", message=str(exc)),
                        thresholds={
                            "enabled": (config.get("body_tracking") or {}).get("enabled"),
                        },
                        artifacts=_status_artifact_paths(ep_id, effective_run_id, "body_tracking"),
                    )
                except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                    LOGGER.warning("[manifest] Failed to write body_tracking failed manifest: %s", manifest_exc)
            try:
                append_log(
                    ep_id,
                    effective_run_id,
                    "body_tracking",
                    "ERROR",
                    "stage failed",
                    progress=100.0,
                    meta={"error_code": "import_error", "error_message": str(exc)},
                )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log body_tracking failure: %s", log_exc)
        stage_heartbeat.update(
            done=0,
            phase="error",
            message=f"Import error: {exc}",
            force=True,
            mark_end=True,
        )
        _write_run_marker(ep_id, "body_tracking", payload, run_id=run_id)
        return payload

    try:
        runner = BodyTrackingRunner(
            episode_id=ep_id,
            config_path=config_path if config_path.exists() else None,
            fusion_config_path=fusion_config_path if fusion_config_path.exists() else None,
            video_path=video_path,
            output_dir=output_dir,
            skip_existing=True,
        )
        det_path = _run_with_heartbeat(
            runner.run_detection,
            lambda: stage_heartbeat.update(
                done=0,
                phase="running_frames",
                message="Detecting bodies",
            ),
            interval=stage_heartbeat.heartbeat_interval,
        )
        if effective_run_id:
            try:
                append_log(
                    ep_id,
                    effective_run_id,
                    "body_tracking",
                    "INFO",
                    "body detection complete",
                    progress=40.0,
                    meta={"det_path": str(det_path) if det_path else None},
                )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log body detection: %s", log_exc)
        tracks_path = _run_with_heartbeat(
            runner.run_tracking,
            lambda: stage_heartbeat.update(
                done=0,
                phase="running_frames",
                message="Tracking bodies",
            ),
            interval=stage_heartbeat.heartbeat_interval,
        )
        if effective_run_id:
            try:
                append_log(
                    ep_id,
                    effective_run_id,
                    "body_tracking",
                    "INFO",
                    "body tracking complete",
                    progress=70.0,
                    meta={"tracks_path": str(tracks_path) if tracks_path else None},
                )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log body tracking: %s", log_exc)
        tracker_backend_configured = getattr(runner, "tracker_backend_configured", None)
        tracker_backend_actual = getattr(runner, "tracker_backend_actual", None)
        tracker_fallback_reason = getattr(runner, "tracker_fallback_reason", None)

        embeddings_path: Path | None = None
        embeddings_note: str | None = None
        reid_enabled_config = bool(getattr(runner.config, "reid_enabled", False))
        reid_embeddings_generated = False
        reid_skip_reason: str | None = None
        reid_comparisons_performed = 0

        torchreid_import_ok: bool | None = None
        torchreid_status: str | None = None
        torchreid_error: str | None = None
        torchreid_version: str | None = None
        torchreid_runtime_ok: bool | None = None
        torchreid_runtime_error: str | None = None
        torchreid_import_error: str | None = None
        if isinstance(import_status, dict):
            torchreid_state = import_status.get("torchreid")
            if isinstance(torchreid_state, dict):
                torchreid_status = torchreid_state.get("status")
                torchreid_error = torchreid_state.get("error")
                torchreid_version = torchreid_state.get("version")
                if isinstance(torchreid_status, str) and torchreid_status.strip():
                    torchreid_import_ok = torchreid_status.strip() == "ok"

        stage_heartbeat.update(
            done=1,
            phase="finalizing",
            message="Finalizing body tracking outputs",
            force=True,
            mark_frames_done=True,
            mark_finalize_start=True,
        )

        if not reid_enabled_config:
            embeddings_note = "disabled"
            reid_skip_reason = "disabled"
        else:
            try:
                from py_screenalytics.torchreid_compat import get_torchreid_feature_extractor

                get_torchreid_feature_extractor()
                torchreid_import_ok = True
            except ImportError as exc:
                torchreid_import_ok = False
                torchreid_import_error = str(exc)
            except Exception as exc:
                torchreid_import_ok = False
                torchreid_import_error = f"{type(exc).__name__}: {exc}"

        if not reid_enabled_config:
            pass
        elif torchreid_import_ok is False:
            reid_skip_reason = (
                f"torchreid_{torchreid_status}"
                if torchreid_status and torchreid_status != "ok"
                else "torchreid_import_error"
            )
            if torchreid_import_error and torchreid_import_error.startswith("torchreid_missing:"):
                reid_skip_reason = "torchreid_missing"
            elif torchreid_import_error and torchreid_import_error.startswith("torchreid_runtime_error:"):
                reid_skip_reason = "torchreid_runtime_error"
            torchreid_runtime_ok = False
            torchreid_runtime_error = torchreid_import_error or torchreid_error or "torchreid import failed"
            embeddings_note = f"import_error: {torchreid_runtime_error}"
            LOGGER.warning("[body_tracking] Re-ID embeddings skipped: %s", embeddings_note)
        else:
            try:
                embeddings_path = _run_with_heartbeat(
                    runner.run_embedding,
                    lambda: stage_heartbeat.update(
                        done=1,
                        phase="finalizing",
                        message="Generating body embeddings",
                    ),
                    interval=stage_heartbeat.heartbeat_interval,
                )
                reid_embeddings_generated = True
                torchreid_runtime_ok = True
            except ImportError as exc:
                torchreid_runtime_ok = False
                torchreid_runtime_error = str(exc)
                reid_skip_reason = "torchreid_runtime_error" if torchreid_import_ok is True else "torchreid_import_error"
                embeddings_note = f"{'runtime_error' if torchreid_import_ok is True else 'import_error'}: {exc}"
                LOGGER.warning("[body_tracking] Re-ID embeddings skipped: %s", embeddings_note)
            except Exception as exc:
                torchreid_runtime_ok = False
                torchreid_runtime_error = f"{type(exc).__name__}: {exc}"
                if torchreid_import_ok is True:
                    reid_skip_reason = "torchreid_runtime_error"
                    embeddings_note = f"runtime_error: {type(exc).__name__}: {exc}"
                else:
                    reid_skip_reason = "reid_error"
                    embeddings_note = f"error: {type(exc).__name__}: {exc}"
                LOGGER.warning("[body_tracking] Re-ID embeddings failed: %s", embeddings_note)

        if effective_run_id:
            try:
                append_log(
                    ep_id,
                    effective_run_id,
                    "body_tracking",
                    "INFO",
                    "body re-id embeddings processed",
                    progress=85.0,
                    meta={
                        "reid_enabled": reid_enabled_config,
                        "reid_embeddings_generated": reid_embeddings_generated,
                        "reid_skip_reason": reid_skip_reason,
                    },
                )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log body re-id status: %s", log_exc)

        body_reid = {
            "enabled_config": reid_enabled_config,
            "enabled_effective": bool(reid_enabled_config and reid_embeddings_generated),
            "reid_embeddings_generated": bool(reid_embeddings_generated),
            "reid_skip_reason": reid_skip_reason,
            "reid_comparisons_performed": int(reid_comparisons_performed),
            "torchreid_import_ok": torchreid_import_ok,
            "torchreid_version": torchreid_version,
            "torchreid_runtime_ok": torchreid_runtime_ok,
            "torchreid_runtime_error": torchreid_runtime_error,
        }

        # Always materialize placeholder embedding artifacts so run-scoped bundles are consistent.
        # Fusion can still proceed without embeddings; empty arrays safely no-op the Re-ID path.
        if embeddings_path is None:
            embeddings_path = runner.embeddings_path
        if not embeddings_path.exists():
            try:
                dim_raw = getattr(runner.config, "reid_embedding_dim", 256)
                dim = max(int(dim_raw) if dim_raw is not None else 256, 1)
            except (TypeError, ValueError):
                dim = 256
            try:
                np.save(embeddings_path, np.zeros((0, dim), dtype=np.float32))
            except Exception as exc:
                LOGGER.warning("[body_tracking] Failed to write placeholder body_embeddings.npy: %s", exc)
        if not runner.embeddings_meta_path.exists():
            try:
                meta = {
                    "embedding_dim": None,
                    "model_name": getattr(runner.config, "reid_model", None),
                    "num_embeddings": 0,
                    "entries": [],
                    "reid_enabled": reid_enabled_config,
                    "reid_skip_reason": reid_skip_reason,
                    "note": embeddings_note,
                }
                runner.embeddings_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception as exc:
                LOGGER.warning("[body_tracking] Failed to write placeholder body_embeddings_meta.json: %s", exc)

        # Emit a lightweight metrics artifact (allowlisted for run-scoped S3 sync).
        if not runner.metrics_path.exists():
            try:
                metrics_payload = {
                    "episode_id": ep_id,
                    "generated_at": _utcnow_iso(),
                    "output_dir": str(output_dir),
                    "import_status": import_status,
                    "reid_enabled": reid_enabled_config,
                    "reid_note": embeddings_note,
                    "body_reid": body_reid,
                    "tracker_backend_configured": tracker_backend_configured,
                    "tracker_backend_actual": tracker_backend_actual,
                    "tracker_fallback_reason": tracker_fallback_reason,
                    "body_tracker_backend_configured": tracker_backend_configured,
                    "body_tracker_backend_actual": tracker_backend_actual,
                    "body_tracker_fallback_reason": tracker_fallback_reason,
                    "artifacts": {
                        "body_detections": str(det_path),
                        "body_tracks": str(tracks_path),
                        "body_embeddings": str(embeddings_path),
                        "body_embeddings_meta": str(runner.embeddings_meta_path),
                    },
                }
                runner.metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
            except Exception as exc:
                LOGGER.warning("[body_tracking] Failed to write body_metrics.json: %s", exc)

        if run_id:
            promote = [
                "body_tracking/body_detections.jsonl",
                "body_tracking/body_tracks.jsonl",
            ]
            if embeddings_path and embeddings_path.exists():
                promote.append("body_tracking/body_embeddings.npy")
            if runner.embeddings_meta_path.exists():
                promote.append("body_tracking/body_embeddings_meta.json")
            if runner.metrics_path.exists():
                promote.append("body_tracking/body_metrics.json")
            _promote_run_manifests_to_root(ep_id, run_id, promote)

        payload: Dict[str, Any] = {
            "phase": "body_tracking",
            "status": "success",
            "version": APP_VERSION,
            "run_id": effective_run_id,
            "started_at": started_at,
            "finished_at": _utcnow_iso(),
            "import_status": import_status,
            "reid_enabled": reid_enabled_config,
            "reid_note": embeddings_note,
            "body_reid": body_reid,
            "tracker_backend_configured": tracker_backend_configured,
            "tracker_backend_actual": tracker_backend_actual,
            "tracker_fallback_reason": tracker_fallback_reason,
            "body_tracker_backend_configured": tracker_backend_configured,
            "body_tracker_backend_actual": tracker_backend_actual,
            "body_tracker_fallback_reason": tracker_fallback_reason,
            "artifacts": {
                "local": {
                    "body_tracking_dir": str(output_dir),
                    "body_detections": str(det_path),
                    "body_tracks": str(tracks_path),
                    "body_embeddings": str(embeddings_path) if embeddings_path and embeddings_path.exists() else None,
                    "body_embeddings_meta": (
                        str(runner.embeddings_meta_path) if runner.embeddings_meta_path.exists() else None
                    ),
                    "body_metrics": str(runner.metrics_path) if runner.metrics_path.exists() else None,
                }
            },
        }
        if effective_run_id:
            try:
                write_stage_finished(
                    ep_id,
                    effective_run_id,
                    "body_tracking",
                    artifact_paths=_status_artifact_paths(ep_id, effective_run_id, "body_tracking"),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark body_tracking success: %s", status_exc)
            if emit_manifests:
                try:
                    write_stage_manifest(
                        ep_id,
                        effective_run_id,
                        "body_tracking",
                        "SUCCESS",
                        started_at=started_at,
                        finished_at=payload.get("finished_at"),
                        duration_s=None,
                        counts=None,
                        thresholds={
                            "reid_enabled": reid_enabled_config,
                        },
                        artifacts=_status_artifact_paths(ep_id, effective_run_id, "body_tracking"),
                        model_versions={
                            "tracker_backend": tracker_backend_actual or tracker_backend_configured,
                            "reid_model": getattr(runner.config, "reid_model", None),
                        },
                    )
                except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                    LOGGER.warning("[manifest] Failed to write body_tracking success manifest: %s", manifest_exc)
            try:
                append_log(
                    ep_id,
                    effective_run_id,
                    "body_tracking",
                    "INFO",
                    "stage finished",
                    progress=100.0,
                )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log body_tracking success: %s", log_exc)
        stage_heartbeat.update(
            done=1,
            phase="done",
            message="Body tracking complete",
            force=True,
            mark_end=True,
        )
        _write_run_marker(ep_id, "body_tracking", payload, run_id=run_id)
        return payload
    except Exception as exc:
        LOGGER.exception("[body_tracking] Failed for ep_id=%s run_id=%s", ep_id, effective_run_id)
        payload = {
            "phase": "body_tracking",
            "status": "error",
            "version": APP_VERSION,
            "run_id": effective_run_id,
            "started_at": started_at,
            "finished_at": _utcnow_iso(),
            "import_status": import_status,
            "error": str(exc),
        }
        if effective_run_id:
            try:
                write_stage_failed(
                    ep_id,
                    effective_run_id,
                    "body_tracking",
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark body_tracking failure: %s", status_exc)
            if emit_manifests:
                try:
                    write_stage_manifest(
                        ep_id,
                        effective_run_id,
                        "body_tracking",
                        "FAILED",
                        started_at=started_at,
                        finished_at=payload.get("finished_at"),
                        duration_s=None,
                        error=StageErrorInfo(code=type(exc).__name__, message=str(exc)),
                        thresholds={
                            "enabled": (config.get("body_tracking") or {}).get("enabled"),
                        },
                        artifacts=_status_artifact_paths(ep_id, effective_run_id, "body_tracking"),
                    )
                except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                    LOGGER.warning("[manifest] Failed to write body_tracking failed manifest: %s", manifest_exc)
            try:
                append_log(
                    ep_id,
                    effective_run_id,
                    "body_tracking",
                    "ERROR",
                    "stage failed",
                    progress=100.0,
                    meta={"error_code": type(exc).__name__, "error_message": str(exc)},
                )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log body_tracking failure: %s", log_exc)
        stage_heartbeat.update(
            done=1,
            phase="error",
            message=f"Body tracking failed: {exc}",
            force=True,
            mark_end=True,
        )
        _write_run_marker(ep_id, "body_tracking", payload, run_id=run_id)
        return payload


def _maybe_run_body_tracking_fusion(
    *,
    ep_id: str,
    run_id: str | None,
    effective_run_id: str | None,
    emit_manifests: bool = True,
) -> Dict[str, Any] | None:
    """Run body tracking fusion and screen-time comparison.

    This should be called AFTER cluster stage completes, since fusion requires:
    - body_tracks.jsonl (from detect_track body_tracking)
    - faces.jsonl (from faces_embed)

    Returns payload dict on success/error, None if skipped (disabled or missing prereqs).
    """
    config = _load_body_tracking_config()
    stage_run_id = effective_run_id or run_id
    if stage_run_id:
        gate = check_prereqs("track_fusion", ep_id, stage_run_id, config=config)
        if not gate.ok:
            blocked_reason, blocked_info = _blocked_details(gate)
            LOGGER.debug("[body_tracking_fusion] Gate blocked: %s", blocked_reason.message)
            should_block = blocked_update_needed(ep_id, stage_run_id, "track_fusion", blocked_reason)
            if should_block:
                try:
                    try:
                        append_log(
                            ep_id,
                            stage_run_id,
                            "track_fusion",
                            "WARNING",
                            "stage blocked",
                            progress=0.0,
                            meta={
                                "reason_code": blocked_reason.code,
                                "reason_message": blocked_reason.message,
                                "suggested_actions": list(gate.suggested_actions),
                            },
                        )
                    except Exception as log_exc:  # pragma: no cover - best effort log write
                        LOGGER.debug("[run_logs] Failed to log track_fusion blocked: %s", log_exc)
                    write_stage_blocked(
                        ep_id,
                        stage_run_id,
                        "track_fusion",
                        blocked_reason,
                    )
                except Exception as status_exc:  # pragma: no cover - best effort status update
                    LOGGER.warning("[episode_status] Failed to mark track_fusion blocked: %s", status_exc)
                if emit_manifests:
                    try:
                        write_stage_manifest(
                            ep_id,
                            stage_run_id,
                            "track_fusion",
                            "BLOCKED",
                            started_at=_utcnow_iso(),
                            finished_at=_utcnow_iso(),
                            duration_s=None,
                            blocked=blocked_info,
                            thresholds={
                                "enabled": (config.get("body_tracking") or {}).get("enabled"),
                            },
                        )
                    except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                        LOGGER.warning("[manifest] Failed to write track_fusion blocked manifest: %s", manifest_exc)
            return None
    else:
        enabled = bool((config.get("body_tracking") or {}).get("enabled", False))
        if not enabled:
            LOGGER.debug("[body_tracking_fusion] Body tracking disabled, skipping fusion")
            return None

    output_dir = _body_tracking_dir_for_run(ep_id, run_id)
    manifests_dir = _manifests_dir_for_run(ep_id, stage_run_id)
    config_path = REPO_ROOT / "config" / "pipeline" / "body_detection.yaml"
    fusion_config_path = REPO_ROOT / "config" / "pipeline" / "track_fusion.yaml"

    started_at = _utcnow_iso()
    stage_heartbeat = StageStatusHeartbeat(
        ep_id=ep_id,
        run_id=effective_run_id,
        stage_key="track_fusion",
        frames_total=1,
        started_at=started_at,
    )
    if effective_run_id:
        try:
            write_stage_started(
                ep_id,
                effective_run_id,
                "track_fusion",
                started_at=datetime.fromisoformat(started_at.replace("Z", "+00:00")),
            )
        except Exception as exc:  # pragma: no cover - best effort status update
            LOGGER.warning("[episode_status] Failed to mark track_fusion start: %s", exc)
        try:
            append_log(
                ep_id,
                effective_run_id,
                "track_fusion",
                "INFO",
                "stage started",
                progress=0.0,
            )
        except Exception as log_exc:  # pragma: no cover - best effort log write
            LOGGER.debug("[run_logs] Failed to log track_fusion start: %s", log_exc)
    stage_heartbeat.update(
        done=0,
        phase="running_frames",
        message="Starting track fusion",
        force=True,
    )
    try:
        from FEATURES.body_tracking.src.body_tracking_runner import BodyTrackingRunner
    except Exception as exc:
        LOGGER.warning(
            "[body_tracking_fusion] Disabled due to import error: %s",
            exc,
        )
        payload = {
            "phase": "body_tracking_fusion",
            "status": "error",
            "version": APP_VERSION,
            "run_id": effective_run_id,
            "started_at": started_at,
            "finished_at": _utcnow_iso(),
            "error": f"import_error: {exc}",
        }
        if effective_run_id:
            try:
                write_stage_failed(
                    ep_id,
                    effective_run_id,
                    "track_fusion",
                    error_code="import_error",
                    error_message=str(exc),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark track_fusion failure: %s", status_exc)
            if emit_manifests:
                try:
                    write_stage_manifest(
                        ep_id,
                        effective_run_id,
                        "track_fusion",
                        "FAILED",
                        started_at=started_at,
                        finished_at=payload.get("finished_at"),
                        duration_s=None,
                        error=StageErrorInfo(code="import_error", message=str(exc)),
                        artifacts=_status_artifact_paths(ep_id, effective_run_id, "track_fusion"),
                    )
                except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                    LOGGER.warning("[manifest] Failed to write track_fusion failed manifest: %s", manifest_exc)
            try:
                append_log(
                    ep_id,
                    effective_run_id,
                    "track_fusion",
                    "ERROR",
                    "stage failed",
                    progress=100.0,
                    meta={"error_code": "import_error", "error_message": str(exc)},
                )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log track_fusion failure: %s", log_exc)
        stage_heartbeat.update(
            done=0,
            phase="error",
            message=f"Import error: {exc}",
            force=True,
            mark_end=True,
        )
        _write_run_marker(ep_id, "body_tracking_fusion", payload, run_id=run_id)
        return payload

    try:
        # Video path not needed for fusion (only uses existing artifacts)
        runner = BodyTrackingRunner(
            episode_id=ep_id,
            config_path=config_path if config_path.exists() else None,
            fusion_config_path=fusion_config_path if fusion_config_path.exists() else None,
            # Provide the canonical local video path to avoid legacy path discovery failures.
            # Fusion/comparison do not require the video, but the runner resolves it eagerly.
            video_path=get_path(ep_id, "video"),
            output_dir=output_dir,
            manifests_dir=manifests_dir,
            skip_existing=True,
        )

        # Run fusion (associates face tracks with body tracks)
        fusion_path = _run_with_heartbeat(
            runner.run_fusion,
            lambda: stage_heartbeat.update(
                done=0,
                phase="running_frames",
                message="Fusing face/body tracks",
            ),
            interval=stage_heartbeat.heartbeat_interval,
        )
        LOGGER.info("[body_tracking_fusion] Fusion complete: %s", fusion_path)
        if effective_run_id:
            try:
                append_log(
                    ep_id,
                    effective_run_id,
                    "track_fusion",
                    "INFO",
                    "fusion complete",
                    progress=50.0,
                    meta={"fusion_path": str(fusion_path) if fusion_path else None},
                )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log fusion progress: %s", log_exc)

        # Run comparison (calculates face-only vs face+body screen time)
        comparison_path: Path | None = None
        try:
            stage_heartbeat.update(
                done=1,
                phase="finalizing",
                message="Computing fusion comparison",
                force=True,
                mark_frames_done=True,
                mark_finalize_start=True,
            )
            comparison_path = _run_with_heartbeat(
                runner.run_comparison,
                lambda: stage_heartbeat.update(
                    done=1,
                    phase="finalizing",
                    message="Computing fusion comparison",
                ),
                interval=stage_heartbeat.heartbeat_interval,
            )
            LOGGER.info("[body_tracking_fusion] Comparison complete: %s", comparison_path)
            if effective_run_id:
                try:
                    append_log(
                        ep_id,
                        effective_run_id,
                        "track_fusion",
                        "INFO",
                        "comparison complete",
                        progress=80.0,
                        meta={
                            "comparison_path": str(comparison_path) if comparison_path else None,
                        },
                    )
                except Exception as log_exc:  # pragma: no cover - best effort log write
                    LOGGER.debug("[run_logs] Failed to log comparison progress: %s", log_exc)
        except Exception as exc:
            LOGGER.warning("[body_tracking_fusion] Comparison failed: %s", exc)
            if effective_run_id:
                try:
                    append_log(
                        ep_id,
                        effective_run_id,
                        "track_fusion",
                        "WARNING",
                        "comparison failed",
                        progress=80.0,
                        meta={"error_message": str(exc)},
                    )
                except Exception as log_exc:  # pragma: no cover - best effort log write
                    LOGGER.debug("[run_logs] Failed to log comparison failure: %s", log_exc)

        # Promote artifacts to root manifests dir
        if run_id:
            promote = []
            if fusion_path and fusion_path.exists():
                promote.append("body_tracking/track_fusion.json")
            if comparison_path and comparison_path.exists():
                promote.append("body_tracking/screentime_comparison.json")
            if runner.metrics_path.exists():
                promote.append("body_tracking/body_metrics.json")
            if promote:
                _promote_run_manifests_to_root(ep_id, run_id, promote)

        payload: Dict[str, Any] = {
            "phase": "body_tracking_fusion",
            "status": "success",
            "version": APP_VERSION,
            "run_id": effective_run_id,
            "started_at": started_at,
            "finished_at": _utcnow_iso(),
            "artifacts": {
                "local": {
                    "track_fusion": str(fusion_path) if fusion_path and fusion_path.exists() else None,
                    "screentime_comparison": str(comparison_path) if comparison_path and comparison_path.exists() else None,
                }
            },
        }
        if effective_run_id:
            try:
                write_stage_finished(
                    ep_id,
                    effective_run_id,
                    "track_fusion",
                    artifact_paths=_status_artifact_paths(ep_id, effective_run_id, "track_fusion"),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark track_fusion success: %s", status_exc)
            if emit_manifests:
                try:
                    write_stage_manifest(
                        ep_id,
                        effective_run_id,
                        "track_fusion",
                        "SUCCESS",
                        started_at=started_at,
                        finished_at=payload.get("finished_at"),
                        duration_s=None,
                        artifacts=_status_artifact_paths(ep_id, effective_run_id, "track_fusion"),
                        model_versions={
                            "tracker_backend": getattr(runner, "tracker_backend_actual", None),
                        },
                    )
                except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                    LOGGER.warning("[manifest] Failed to write track_fusion success manifest: %s", manifest_exc)
            try:
                append_log(
                    ep_id,
                    effective_run_id,
                    "track_fusion",
                    "INFO",
                    "stage finished",
                    progress=100.0,
                )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log track_fusion success: %s", log_exc)
        stage_heartbeat.update(
            done=1,
            phase="done",
            message="Track fusion complete",
            force=True,
            mark_end=True,
        )
        _write_run_marker(ep_id, "body_tracking_fusion", payload, run_id=run_id)
        return payload
    except Exception as exc:
        LOGGER.exception("[body_tracking_fusion] Failed for ep_id=%s run_id=%s", ep_id, effective_run_id)
        payload = {
            "phase": "body_tracking_fusion",
            "status": "error",
            "version": APP_VERSION,
            "run_id": effective_run_id,
            "started_at": started_at,
            "finished_at": _utcnow_iso(),
            "error": str(exc),
        }
        if effective_run_id:
            try:
                write_stage_failed(
                    ep_id,
                    effective_run_id,
                    "track_fusion",
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark track_fusion failure: %s", status_exc)
            if emit_manifests:
                try:
                    write_stage_manifest(
                        ep_id,
                        effective_run_id,
                        "track_fusion",
                        "FAILED",
                        started_at=started_at,
                        finished_at=payload.get("finished_at"),
                        duration_s=None,
                        error=StageErrorInfo(code=type(exc).__name__, message=str(exc)),
                        artifacts=_status_artifact_paths(ep_id, effective_run_id, "track_fusion"),
                    )
                except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                    LOGGER.warning("[manifest] Failed to write track_fusion failed manifest: %s", manifest_exc)
            try:
                append_log(
                    ep_id,
                    effective_run_id,
                    "track_fusion",
                    "ERROR",
                    "stage failed",
                    progress=100.0,
                    meta={"error_code": type(exc).__name__, "error_message": str(exc)},
                )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log track_fusion failure: %s", log_exc)
        stage_heartbeat.update(
            done=1,
            phase="error",
            message=f"Track fusion failed: {exc}",
            force=True,
            mark_end=True,
        )
        _write_run_marker(ep_id, "body_tracking_fusion", payload, run_id=run_id)
        return payload


def _run_detect_track_stage(
    args: argparse.Namespace,
    storage: StorageService | None,
    ep_ctx: EpisodeContext | None,
    s3_prefixes: Dict[str, str] | None,
) -> Dict[str, Any]:
    if not args.video:
        raise ValueError("--video is required for detect/track runs")
    video_src = Path(args.video)
    if not video_src.exists():
        raise FileNotFoundError(f"Video not found: {video_src}")
    video_dest = get_path(args.ep_id, "video")
    _copy_video(video_src, video_dest)

    # Validate video file before processing
    LOGGER.info("Validating video file: %s", video_dest)
    video_info = require_valid_video(
        video_dest,
        check_first_frame=True,
        min_frames=10,  # Need at least a few frames for meaningful processing
        min_duration_sec=0.5,  # At least half a second of video
    )
    LOGGER.info(
        "Video validation passed: %dx%d, %.1f fps, %d frames, %.1f sec",
        video_info.get("width", 0),
        video_info.get("height", 0),
        video_info.get("fps", 0),
        video_info.get("frame_count", 0),
        video_info.get("duration_sec", 0),
    )

    source_fps, frame_count = _probe_video(video_dest)
    target_fps = args.fps if args.fps and args.fps > 0 else None
    duration_sec = _estimate_duration(frame_count, source_fps)
    if duration_sec is None and frame_count > 0:
        fallback_fps = target_fps or source_fps or 30.0
        if fallback_fps > 0:
            duration_sec = frame_count / fallback_fps
    frames_total = frame_count
    if frames_total <= 0:
        frames_total = _estimate_frame_budget(
            stride=args.stride,
            target_fps=target_fps,
            detected_fps=source_fps,
            duration_sec=duration_sec,
            frame_count=frame_count,
        )

    progress = ProgressEmitter(
        args.ep_id,
        args.progress_file,
        frames_total=frames_total,
        secs_total=duration_sec,
        stride=args.stride,
        fps_detected=source_fps,
        fps_requested=target_fps,
        run_id=getattr(args, "run_id", None),
    )

    # Emit initial progress before model initialization to ensure progress.json
    # exists even if the job fails early (e.g., model load error, video access issue)
    progress.emit(
        frames_done=0,
        phase="init",
        force=True,
    )

    save_frames = bool(args.save_frames)
    save_crops = bool(args.save_crops)
    jpeg_quality = max(1, min(int(args.jpeg_quality or 85), 100))
    detector_choice = _normalize_detector_choice(getattr(args, "detector", None))
    tracker_choice = _normalize_tracker_choice(getattr(args, "tracker", None))
    frame_exporter = (
        FrameExporter(
            args.ep_id,
            run_id=getattr(args, "run_id", None),
            save_frames=save_frames,
            save_crops=save_crops,
            jpeg_quality=jpeg_quality,
            debug_logger=None,
        )
        if (save_frames or save_crops)
        else None
    )

    run_id = getattr(args, "run_id", None)
    emit_manifests = getattr(args, "emit_manifests", True)
    started_at = _utcnow_iso()
    if run_id:
        gate = check_prereqs("detect", args.ep_id, run_id)
        if not gate.ok:
            blocked_reason, blocked_info = _blocked_details(gate)
            if blocked_update_needed(args.ep_id, run_id, "detect", blocked_reason):
                try:
                    write_stage_blocked(args.ep_id, run_id, "detect", blocked_reason)
                except Exception as status_exc:  # pragma: no cover - best effort status update
                    LOGGER.warning("[episode_status] Failed to mark detect blocked: %s", status_exc)
                if emit_manifests:
                    try:
                        write_stage_manifest(
                            args.ep_id,
                            run_id,
                            "detect",
                            "BLOCKED",
                            started_at=started_at,
                            finished_at=_utcnow_iso(),
                            duration_s=None,
                            blocked=blocked_info,
                            thresholds={
                                "det_thresh": getattr(args, "det_thresh", None),
                                "track_high_thresh": getattr(args, "track_high_thresh", None),
                                "new_track_thresh": getattr(args, "new_track_thresh", None),
                                "min_box_area": getattr(args, "min_box_area", None),
                                "stride": getattr(args, "stride", None),
                                "scene_threshold": getattr(args, "scene_threshold", None),
                                "scene_min_len": getattr(args, "scene_min_len", None),
                            },
                        )
                    except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                        LOGGER.warning("[manifest] Failed to write detect blocked manifest: %s", manifest_exc)
            raise RuntimeError(blocked_reason.message)
    if run_id:
        try:
            write_stage_started(args.ep_id, run_id, "detect")
        except Exception as exc:  # pragma: no cover - best effort status update
            LOGGER.warning("[episode_status] Failed to mark detect start: %s", exc)
    try:
        (
            det_count,
            track_count,
            frames_sampled,
            pipeline_device,
            detector_device,
            analyzed_fps,
            track_metrics,
            scene_summary,
            detection_conf_hist,
            detect_track_stats,
            tracker_config_summary,
        ) = _run_full_pipeline(
            args,
            video_dest,
            source_fps=source_fps,
            progress=progress,
            target_fps=target_fps,
            frame_exporter=frame_exporter,
            total_frames=frames_total,
            video_fps=source_fps,
        )

        # Adaptive gate auto-rerun: if the appearance gate is over-splitting, rerun with gate disabled
        gate_enabled_initial = bool(getattr(args, "gate_enabled", GATE_ENABLED_DEFAULT))
        gate_auto_rerun: Dict[str, Any] = {"triggered": False, "reason": "not_evaluated"}
        if tracker_choice == "bytetrack" and gate_enabled_initial:
            triggered, reason, decision = _gate_auto_rerun_decision(track_metrics)
            gate_auto_rerun = {
                "triggered": bool(triggered),
                "reason": reason,
                "decision": decision,
                "selected": "initial",
                "initial": {
                    "gate_enabled": True,
                    "detections": det_count,
                    "tracks": track_count,
                    "metrics": track_metrics,
                },
            }
            if triggered:
                LOGGER.warning(
                    "[gate] Auto-rerun triggered (forced_splits=%s id_switches=%s); rerunning with gate_enabled=false",
                    decision.get("forced_splits"),
                    decision.get("id_switches"),
                )
                rerun_args = argparse.Namespace(**vars(args))
                rerun_args.gate_enabled = False
                try:
                    (
                        det_count_rerun,
                        track_count_rerun,
                        frames_sampled_rerun,
                        pipeline_device_rerun,
                        detector_device_rerun,
                        analyzed_fps_rerun,
                        track_metrics_rerun,
                        scene_summary_rerun,
                        detection_conf_hist_rerun,
                        detect_track_stats_rerun,
                        tracker_config_summary_rerun,
                    ) = _run_full_pipeline(
                        rerun_args,
                        video_dest,
                        source_fps=source_fps,
                        progress=None,  # avoid re-emitting progress for the recovery pass
                        target_fps=target_fps,
                        frame_exporter=frame_exporter,
                        total_frames=frames_total,
                        video_fps=source_fps,
                    )

                    det_count = det_count_rerun
                    track_count = track_count_rerun
                    frames_sampled = frames_sampled_rerun
                    pipeline_device = pipeline_device_rerun
                    detector_device = detector_device_rerun
                    analyzed_fps = analyzed_fps_rerun
                    track_metrics = track_metrics_rerun
                    scene_summary = scene_summary_rerun
                    detection_conf_hist = detection_conf_hist_rerun
                    detect_track_stats = detect_track_stats_rerun
                    tracker_config_summary = tracker_config_summary_rerun
                    args.gate_enabled = False

                    gate_auto_rerun["selected"] = "rerun"
                    gate_auto_rerun["rerun"] = {
                        "gate_enabled": False,
                        "detections": det_count_rerun,
                        "tracks": track_count_rerun,
                        "metrics": track_metrics_rerun,
                    }
                except Exception as exc:  # pragma: no cover - best effort recovery
                    LOGGER.warning("[gate] Auto-rerun failed; keeping initial gated output: %s", exc)
                    gate_auto_rerun["error"] = str(exc)
        elif tracker_choice == "bytetrack":
            gate_auto_rerun = {"triggered": False, "reason": "gate_disabled"}

        effective_run_id = run_id or progress.run_id
        manifests_dir = _manifests_dir_for_run(args.ep_id, run_id)

        if det_count == 0:
            LOGGER.warning(
                "Detect/track completed with 0 detections for %s; resulting tracks=%d.",
                args.ep_id,
                track_count,
            )
        if track_count == 0:
            LOGGER.error(
                "Detect/track completed with 0 tracks for %s despite %d detections. "
                "Review thresholds and rerun detect/track.",
                args.ep_id,
                det_count,
            )

        # Write track_metrics.json alongside the run's manifests (then promote to legacy root).
        scene_summary = scene_summary or {"count": 0}
        metrics_path = manifests_dir / "track_metrics.json"
        metrics_payload = {
            "ep_id": args.ep_id,
            "generated_at": _utcnow_iso(),
            "metrics": track_metrics,
            "scene_cuts": scene_summary,
            "tracking_gate": {
                "enabled": bool(getattr(args, "gate_enabled", GATE_ENABLED_DEFAULT)),
                "auto_rerun": gate_auto_rerun,
            },
        }
        try:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - best effort diagnostics
            LOGGER.warning("Failed to write track metrics for %s: %s", args.ep_id, exc)

        if run_id:
            _promote_run_manifests_to_root(
                args.ep_id,
                run_id,
                ("detections.jsonl", "tracks.jsonl", "track_metrics.json"),
            )

        body_tracking_summary = _maybe_run_body_tracking(
            ep_id=args.ep_id,
            run_id=run_id,
            effective_run_id=effective_run_id,
            video_path=video_dest,
            import_status=(
                getattr(args, "_env_diagnostics", {}).get("import_status")
                if isinstance(getattr(args, "_env_diagnostics", None), dict)
                else None
            ),
            emit_manifests=emit_manifests,
        )

        s3_sync_result = _sync_artifacts_to_s3(args.ep_id, storage, ep_ctx, frame_exporter)
        mirror_tracker = tracker_choice
        try:
            face_backend_actual = "ultralytics.bytetrack" if tracker_choice == "bytetrack" else tracker_choice
            body_backend_actual = None
            body_fallback_reason = None
            if isinstance(body_tracking_summary, dict):
                body_backend_actual = (
                    body_tracking_summary.get("body_tracker_backend_actual")
                    or body_tracking_summary.get("tracker_backend_actual")
                )
                body_fallback_reason = (
                    body_tracking_summary.get("body_tracker_fallback_reason")
                    or body_tracking_summary.get("tracker_fallback_reason")
                )
            if isinstance(body_backend_actual, str) and body_backend_actual.strip():
                mirror_tracker = f"face={face_backend_actual} body={body_backend_actual.strip()}"
                if isinstance(body_fallback_reason, str) and body_fallback_reason.strip():
                    mirror_tracker = f"{mirror_tracker} (body_fallback_reason={body_fallback_reason.strip()})"
            else:
                mirror_tracker = face_backend_actual
        except Exception:
            mirror_tracker = tracker_choice
        _report_s3_upload(
            progress,
            s3_sync_result,
            device=pipeline_device,
            detector=detector_choice,
            tracker=mirror_tracker,
            resolved_device=detector_device,
        )
        if not s3_sync_result.success:
            LOGGER.error("S3 sync failed for %s: %s", args.ep_id, s3_sync_result.errors)
        track_ratio = round(track_count / det_count, 3) if det_count > 0 else 0.0
        summary: Dict[str, Any] = {
            "stage": "detect_track",
            "ep_id": args.ep_id,
            "run_id": effective_run_id,
            "detections": det_count,
            "tracks": track_count,
            "detections_total": det_count,
            "tracks_total": track_count,
            "track_ratio": track_ratio,
            "track_to_detection_ratio": track_ratio,
            "detection_confidence_histogram": detection_conf_hist,
            "confidence_histogram": detection_conf_hist,
            "frames_sampled": frames_sampled,
            "frames_total": progress.target_frames,
            "frames_exported": frame_exporter.frames_written if frame_exporter else 0,
            "crops_exported": frame_exporter.crops_written if frame_exporter else 0,
            "device": pipeline_device,
            "requested_device": args.device,
            "resolved_device": detector_device,
            "analyzed_fps": analyzed_fps,
            "detector": detector_choice,
            "tracker": tracker_choice,
            "metrics": track_metrics,
            "artifacts": {
                "local": {
                    "detections": str(_detections_path_for_run(args.ep_id, run_id)),
                    "tracks": str(_tracks_path_for_run(args.ep_id, run_id)),
                    "manifests_dir": str(manifests_dir),
                    "active_mirror_dir": str(get_path(args.ep_id, "detections").parent),
                    "frames_dir": (
                        str(frame_exporter.frames_dir) if frame_exporter and frame_exporter.save_frames else None
                    ),
                    "crops_dir": (
                        str(frame_exporter.crops_dir) if frame_exporter and frame_exporter.save_crops else None
                    ),
                    "track_metrics": str(metrics_path),
                },
                "s3_prefixes": s3_prefixes,
            },
        }
        if body_tracking_summary:
            summary["body_tracking"] = body_tracking_summary
        if detect_track_stats:
            summary["detect_track_stats"] = detect_track_stats
        if tracker_config_summary:
            summary["tracker_config"] = tracker_config_summary
        scene_count = scene_summary.get("count")
        scene_cuts_payload: Dict[str, Any] = {"count": int(scene_count) if isinstance(scene_count, int) else 0}
        indices = scene_summary.get("indices")
        if isinstance(indices, list):
            scene_cuts_payload["indices"] = indices
        detector_label = scene_summary.get("detector")
        if isinstance(detector_label, str):
            scene_cuts_payload["detector"] = detector_label
        summary["scene_cuts"] = scene_cuts_payload
        summary["scene_cuts_total"] = scene_cuts_payload["count"]
        frames_for_scene_rate = summary.get("frames_total") or frames_sampled or progress.target_frames or 1
        summary["scene_cuts_per_1k_frames"] = (
            round(
                (scene_cuts_payload["count"] / max(frames_for_scene_rate, 1)) * 1000.0,
                3,
            )
            if frames_for_scene_rate
            else 0.0
        )
        completion_extra = _crop_diag_meta(frame_exporter)
        if detect_track_stats:
            completion_extra["detect_track_stats"] = detect_track_stats
        progress.complete(
            summary,
            device=pipeline_device,
            detector=detector_choice,
            tracker=tracker_choice,
            resolved_device=detector_device,
            step="detect_track",
            extra=completion_extra or None,
        )
        # Brief delay to ensure final progress event is written and readable
        time.sleep(0.2)
        finished_at = _utcnow_iso()
        runtime_stats: Dict[str, Any] = detect_track_stats if isinstance(detect_track_stats, dict) else {}
        marker_detect_wall_time_s = runtime_stats.get("wall_time_s")
        marker_rtf = runtime_stats.get("rtf")
        if (
            isinstance(marker_detect_wall_time_s, (int, float))
            and isinstance(progress.secs_total, (int, float))
            and progress.secs_total > 0
        ):
            marker_rtf = round(float(marker_detect_wall_time_s) / float(progress.secs_total), 3)
        _write_run_marker(
            args.ep_id,
            "detect_track",
            {
                "phase": "detect_track",
                "status": "success",
                "version": APP_VERSION,
                "run_id": effective_run_id,
                "import_status": (
                    getattr(args, "_env_diagnostics", {}).get("import_status")
                    if isinstance(getattr(args, "_env_diagnostics", None), dict)
                    else None
                ),
                "detections": det_count,
                "tracks": track_count,
                "detector": detector_choice,
                "detector_model_name": (RETINAFACE_MODEL_NAME if detector_choice == "retinaface" else detector_choice),
                "tracker": tracker_choice,
                "tracker_backend_configured": tracker_choice,
                "tracker_backend_actual": (
                    "ultralytics.bytetrack" if tracker_choice == "bytetrack" else tracker_choice
                ),
                "tracker_fallback_reason": None,
                "face_tracker_backend_configured": tracker_choice,
                "face_tracker_backend_actual": (
                    "ultralytics.bytetrack" if tracker_choice == "bytetrack" else tracker_choice
                ),
                "face_tracker_fallback_reason": None,
                "stride": args.stride,
                # Frame accounting (why frames_processed != frames_total/stride)
                "frames_scanned_total": runtime_stats.get("frames_scanned_total"),
                "face_detect_frames_processed": runtime_stats.get("face_detect_frames_processed"),
                "face_detect_frames_processed_stride": runtime_stats.get("face_detect_frames_processed_stride"),
                "face_detect_frames_processed_forced_scene_warmup": runtime_stats.get(
                    "face_detect_frames_processed_forced_scene_warmup"
                ),
                "stride_effective": runtime_stats.get("stride_effective"),
                "stride_observed_median": runtime_stats.get("stride_observed_median"),
                "expected_frames_by_stride": runtime_stats.get("expected_frames_by_stride"),
                # Runtime diagnostics (wall-clock vs video duration)
                "detect_wall_time_s": marker_detect_wall_time_s,
                "effective_fps_processing": runtime_stats.get("effective_fps_processing"),
                "rtf": marker_rtf,
                "scene_detect_wall_time_s": runtime_stats.get("scene_detect_wall_time_s"),
                "det_thresh": args.det_thresh,
                "max_gap": getattr(args, "max_gap", None),
                "scene_detector": args.scene_detector,
                "scene_threshold": args.scene_threshold,
                "scene_min_len": args.scene_min_len,
                "scene_warmup_dets": args.scene_warmup_dets,
                "scene_cut_count": runtime_stats.get("scene_cut_count"),
                "warmup_cuts_applied": runtime_stats.get("warmup_cuts_applied"),
                "warmup_frames_per_cut_effective": runtime_stats.get("warmup_frames_per_cut_effective"),
                "forced_scene_warmup_ratio": runtime_stats.get("forced_scene_warmup_ratio"),
                "wall_time_per_processed_frame_s": runtime_stats.get("wall_time_per_processed_frame_s"),
                "track_high_thresh": getattr(args, "track_high_thresh", None),
                "new_track_thresh": getattr(args, "new_track_thresh", None),
                "fps": analyzed_fps,
                "frames_total": progress.target_frames,
                "video_duration_sec": progress.secs_total,
                "save_frames": save_frames,
                "save_crops": save_crops,
                "jpeg_quality": jpeg_quality,
                "device": pipeline_device,
                "requested_device": args.device,
                "resolved_device": detector_device,
                "onnx_provider_requested": runtime_stats.get("onnx_provider_requested") or _onnx_provider_label(args.device),
                "onnx_provider_resolved": runtime_stats.get("onnx_provider_resolved") or detector_device,
                "torch_device_requested": runtime_stats.get("torch_device_requested"),
                "torch_device_resolved": runtime_stats.get("torch_device_resolved"),
                "torch_device_fallback_reason": runtime_stats.get("torch_device_fallback_reason"),
                "yolo_fallback_enabled": runtime_stats.get("yolo_fallback_enabled"),
                "yolo_fallback_device": runtime_stats.get("yolo_fallback_device"),
                "yolo_fallback_load_status": runtime_stats.get("yolo_fallback_load_status"),
                "yolo_fallback_disabled_reason": runtime_stats.get("yolo_fallback_disabled_reason"),
                "yolo_fallback_invocations": runtime_stats.get("yolo_fallback_invocations"),
                "yolo_fallback_detections_added": runtime_stats.get("yolo_fallback_detections_added"),
                "tracking_gate": {
                    "enabled": bool(getattr(args, "gate_enabled", GATE_ENABLED_DEFAULT)),
                    "auto_rerun": gate_auto_rerun,
                },
                "started_at": started_at,
                "finished_at": finished_at,
            },
            run_id=run_id,
        )
        if run_id:
            try:
                write_stage_finished(
                    args.ep_id,
                    run_id,
                    "detect",
                    counts={"detections": det_count, "tracks": track_count},
                    metrics={"detections": det_count, "tracks": track_count},
                    artifact_paths=_status_artifact_paths(args.ep_id, run_id, "detect"),
                )
            except Exception as exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark detect success: %s", exc)
            if emit_manifests:
                try:
                    write_stage_manifest(
                        args.ep_id,
                        run_id,
                        "detect",
                        "SUCCESS",
                        started_at=started_at,
                        finished_at=finished_at,
                        duration_s=None,
                        counts={"detections": det_count, "tracks": track_count},
                        thresholds={
                            "det_thresh": getattr(args, "det_thresh", None),
                            "track_high_thresh": getattr(args, "track_high_thresh", None),
                            "new_track_thresh": getattr(args, "new_track_thresh", None),
                            "min_box_area": getattr(args, "min_box_area", None),
                            "stride": getattr(args, "stride", None),
                            "scene_threshold": getattr(args, "scene_threshold", None),
                            "scene_min_len": getattr(args, "scene_min_len", None),
                        },
                        artifacts=_status_artifact_paths(args.ep_id, run_id, "detect"),
                        model_versions={
                            "detector": detector_choice,
                            "tracker": tracker_choice,
                        },
                    )
                except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                    LOGGER.warning("[manifest] Failed to write detect success manifest: %s", manifest_exc)
        return summary
    except Exception as exc:
        if run_id:
            try:
                write_stage_failed(
                    args.ep_id,
                    run_id,
                    "detect",
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark detect failure: %s", status_exc)
            if emit_manifests:
                try:
                    write_stage_manifest(
                        args.ep_id,
                        run_id,
                        "detect",
                        "FAILED",
                        started_at=started_at,
                        finished_at=_utcnow_iso(),
                        duration_s=None,
                        error=StageErrorInfo(code=type(exc).__name__, message=str(exc)),
                        thresholds={
                            "det_thresh": getattr(args, "det_thresh", None),
                            "track_high_thresh": getattr(args, "track_high_thresh", None),
                            "new_track_thresh": getattr(args, "new_track_thresh", None),
                            "min_box_area": getattr(args, "min_box_area", None),
                            "stride": getattr(args, "stride", None),
                            "scene_threshold": getattr(args, "scene_threshold", None),
                            "scene_min_len": getattr(args, "scene_min_len", None),
                        },
                        artifacts=_status_artifact_paths(args.ep_id, run_id, "detect"),
                        model_versions={
                            "detector": detector_choice,
                            "tracker": tracker_choice,
                        },
                    )
                except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                    LOGGER.warning("[manifest] Failed to write detect failed manifest: %s", manifest_exc)
        progress.fail(str(exc))
        raise
    finally:
        progress.close()


def _load_alignment_quality_index(manifest_dir: Path) -> Dict[Tuple[int, int], float]:
    """
    Load alignment quality scores from aligned_faces.jsonl.

    Args:
        manifest_dir: Path to episode manifest directory

    Returns:
        Dict mapping (track_id, frame_idx) to alignment_quality score
    """
    aligned_faces_path = manifest_dir / "face_alignment" / "aligned_faces.jsonl"
    if not aligned_faces_path.exists():
        return {}

    index: Dict[Tuple[int, int], float] = {}
    try:
        with open(aligned_faces_path) as f:
            for line in f:
                data = json.loads(line)
                track_id = data.get("track_id")
                frame_idx = data.get("frame_idx")
                quality = data.get("alignment_quality")

                if track_id is not None and frame_idx is not None and quality is not None:
                    index[(int(track_id), int(frame_idx))] = float(quality)

        if index:
            LOGGER.info("Loaded %d alignment quality scores from %s", len(index), aligned_faces_path)
    except Exception as e:
        LOGGER.warning("Failed to load alignment quality index: %s", e)
        return {}

    return index


def _load_alignment_landmarks_index(aligned_faces_path: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Load 68-point landmarks + quality from aligned_faces.jsonl for reuse in faces_embed."""
    if not aligned_faces_path.exists():
        return {}

    index: Dict[Tuple[int, int], Dict[str, Any]] = {}
    try:
        with open(aligned_faces_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                track_id = data.get("track_id")
                frame_idx = data.get("frame_idx")
                if track_id is None or frame_idx is None:
                    continue
                key = (int(track_id), int(frame_idx))
                entry: Dict[str, Any] = {}
                landmarks_68 = data.get("landmarks_68")
                if isinstance(landmarks_68, list) and landmarks_68:
                    entry["landmarks_68"] = landmarks_68
                alignment_quality = data.get("alignment_quality")
                if alignment_quality is not None:
                    try:
                        entry["alignment_quality"] = float(alignment_quality)
                    except (TypeError, ValueError):
                        pass
                alignment_quality_source = data.get("alignment_quality_source")
                if isinstance(alignment_quality_source, str) and alignment_quality_source.strip():
                    entry["alignment_quality_source"] = alignment_quality_source.strip()
                pose_yaw = data.get("pose_yaw")
                if pose_yaw is not None:
                    try:
                        entry["pose_yaw"] = float(pose_yaw)
                    except (TypeError, ValueError):
                        pass
                pose_pitch = data.get("pose_pitch")
                if pose_pitch is not None:
                    try:
                        entry["pose_pitch"] = float(pose_pitch)
                    except (TypeError, ValueError):
                        pass
                pose_roll = data.get("pose_roll")
                if pose_roll is not None:
                    try:
                        entry["pose_roll"] = float(pose_roll)
                    except (TypeError, ValueError):
                        pass
                pose_err = data.get("pose_reprojection_error_px")
                if pose_err is not None:
                    try:
                        entry["pose_reprojection_error_px"] = float(pose_err)
                    except (TypeError, ValueError):
                        pass
                pose_source = data.get("pose_source")
                if isinstance(pose_source, str) and pose_source.strip():
                    entry["pose_source"] = pose_source.strip()
                if entry:
                    index[key] = entry
    except Exception as exc:
        LOGGER.warning("Failed to load aligned_faces.jsonl: %s", exc)
        return {}

    if index:
        LOGGER.info("Loaded %d aligned face rows from %s", len(index), aligned_faces_path)
    return index


def _run_faces_embed_stage(
    args: argparse.Namespace,
    storage: StorageService | None,
    ep_ctx: EpisodeContext | None,
    s3_prefixes: Dict[str, str] | None,
) -> Dict[str, Any]:
    # Log startup for local mode streaming
    if LOCAL_MODE_INSTRUMENTATION:
        device = args.device or "auto"
        crop_interval_frames = getattr(args, "sample_every_n_frames", 4)
        print(f"[LOCAL MODE] faces_embed starting for {args.ep_id}", flush=True)
        print(f"  device={device}  •  crop_interval_frames={crop_interval_frames}", flush=True)

    run_id = getattr(args, "run_id", None)
    emit_manifests = getattr(args, "emit_manifests", True)
    if run_id:
        gate = check_prereqs("faces", args.ep_id, run_id)
        if not gate.ok:
            blocked_reason, blocked_info = _blocked_details(gate)
            if blocked_update_needed(args.ep_id, run_id, "faces", blocked_reason):
                try:
                    write_stage_blocked(args.ep_id, run_id, "faces", blocked_reason)
                except Exception as status_exc:  # pragma: no cover - best effort status update
                    LOGGER.warning("[episode_status] Failed to mark faces blocked: %s", status_exc)
                if emit_manifests:
                    try:
                        write_stage_manifest(
                            args.ep_id,
                            run_id,
                            "faces",
                            "BLOCKED",
                            started_at=_utcnow_iso(),
                            finished_at=_utcnow_iso(),
                            duration_s=None,
                            blocked=blocked_info,
                            thresholds={
                                "max_samples_per_track": getattr(args, "max_samples_per_track", None),
                                "min_samples_per_track": getattr(args, "min_samples_per_track", None),
                                "sample_every_n_frames": getattr(args, "sample_every_n_frames", None),
                            },
                        )
                    except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                        LOGGER.warning("[manifest] Failed to write faces blocked manifest: %s", manifest_exc)
            raise RuntimeError(blocked_reason.message)
        try:
            write_stage_started(args.ep_id, run_id, "faces")
        except Exception as exc:  # pragma: no cover - best effort status update
            LOGGER.warning("[episode_status] Failed to mark faces start: %s", exc)
    track_path = _tracks_path_for_run(args.ep_id, run_id)
    try:
        # Validate tracks.jsonl exists and has usable data
        require_manifest(
            track_path,
            "tracks.jsonl",
            required_fields=["track_id"],
            hint="run detect/track first",
        )

        # Load embedding config early (used for backend selection and alignment gating)
        embedding_config = _load_embedding_config()

        # Sort samples by frame to enable batch embedding per frame
        # Apply per-track sampling to limit embedding/export volume
        samples = _load_track_samples(
            track_path,
            sort_by_frame=True,
            max_samples_per_track=getattr(args, "max_samples_per_track", 16),
            min_samples_per_track=getattr(args, "min_samples_per_track", 4),
            sample_every_n_frames=getattr(args, "sample_every_n_frames", 4),
        )
        if not samples:
            raise ManifestValidationError(
                "tracks.jsonl",
                "File exists but contains no usable track samples – "
                "detect/track likely produced tracks without valid bounding boxes",
                track_path,
            )
    except ManifestValidationError as exc:
        if run_id:
            try:
                write_stage_failed(
                    args.ep_id,
                    run_id,
                    "faces",
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark faces failure: %s", status_exc)
            if emit_manifests:
                try:
                    write_stage_manifest(
                        args.ep_id,
                        run_id,
                        "faces",
                        "FAILED",
                        started_at=_utcnow_iso(),
                        finished_at=_utcnow_iso(),
                        duration_s=None,
                        error=StageErrorInfo(code=type(exc).__name__, message=str(exc)),
                        thresholds={
                            "max_samples_per_track": getattr(args, "max_samples_per_track", None),
                            "min_samples_per_track": getattr(args, "min_samples_per_track", None),
                            "sample_every_n_frames": getattr(args, "sample_every_n_frames", None),
                        },
                        artifacts=_status_artifact_paths(args.ep_id, run_id, "faces"),
                    )
                except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                    LOGGER.warning("[manifest] Failed to write faces failed manifest: %s", manifest_exc)
        raise
    except Exception as exc:
        if run_id:
            try:
                write_stage_failed(
                    args.ep_id,
                    run_id,
                    "faces",
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark faces failure: %s", status_exc)
        raise

    # Load face-alignment + alignment-quality gate config
    manifests_dir = _manifests_dir_for_run(args.ep_id, run_id)
    alignment_config = _load_alignment_config()
    face_alignment_cfg = alignment_config.get("face_alignment", {}) if isinstance(alignment_config.get("face_alignment"), dict) else {}
    alignment_run_enabled = bool(face_alignment_cfg.get("enabled", False))
    alignment_output_cfg = face_alignment_cfg.get("output", {}) if isinstance(face_alignment_cfg.get("output"), dict) else {}
    alignment_crop_size = int(alignment_output_cfg.get("crop_size") or 112)
    alignment_crop_margin = float(alignment_output_cfg.get("crop_margin") or 0.0)
    quality_gate_cfg = face_alignment_cfg.get("quality_gate", {}) if isinstance(face_alignment_cfg.get("quality_gate"), dict) else {}
    legacy_quality_gate_cfg = alignment_config.get("quality_gating", {}) if isinstance(alignment_config.get("quality_gating"), dict) else {}
    legacy_embed_face_alignment_cfg = (
        embedding_config.get("face_alignment", {}) if isinstance(embedding_config.get("face_alignment"), dict) else {}
    )

    quality_gate_enabled = bool(quality_gate_cfg.get("enabled", False))
    if not quality_gate_cfg and legacy_quality_gate_cfg:
        quality_gate_enabled = bool(legacy_quality_gate_cfg.get("enabled", quality_gate_enabled))
    if not quality_gate_cfg and not legacy_quality_gate_cfg:
        quality_gate_enabled = bool(
            quality_gate_enabled
            or legacy_embed_face_alignment_cfg.get("enabled", False)
            or legacy_embed_face_alignment_cfg.get("use_for_embedding", False)
        )

    quality_gate_threshold_raw = (
        quality_gate_cfg.get("threshold")
        if quality_gate_cfg
        else legacy_quality_gate_cfg.get("threshold", legacy_embed_face_alignment_cfg.get("min_alignment_quality", 0.3))
    )
    try:
        quality_gate_threshold = float(quality_gate_threshold_raw)
    except (TypeError, ValueError):
        quality_gate_threshold = 0.3
    quality_gate_threshold = max(0.0, min(quality_gate_threshold, 1.0))

    head_pose_cfg = (
        alignment_config.get("head_pose_3d", {})
        if isinstance(alignment_config.get("head_pose_3d"), dict)
        else {}
    )
    head_pose_enabled = alignment_run_enabled and bool(head_pose_cfg.get("enabled", False))
    head_pose_every_n_frames_raw = head_pose_cfg.get("run_every_n_frames", 10)
    try:
        head_pose_every_n_frames = int(head_pose_every_n_frames_raw)
    except (TypeError, ValueError):
        head_pose_every_n_frames = 10
    head_pose_every_n_frames = max(head_pose_every_n_frames, 1)
    head_pose_run_on_uncertain = bool(head_pose_cfg.get("run_on_uncertain", True))
    head_pose_uncertainty_threshold_raw = head_pose_cfg.get("uncertainty_threshold", quality_gate_threshold)
    try:
        head_pose_uncertainty_threshold = float(head_pose_uncertainty_threshold_raw)
    except (TypeError, ValueError):
        head_pose_uncertainty_threshold = float(quality_gate_threshold)
    head_pose_uncertainty_threshold = max(0.0, min(head_pose_uncertainty_threshold, 1.0))
    head_pose_computed = 0
    head_pose_failed = 0

    quality_gate_mode = str(quality_gate_cfg.get("mode", "drop") or "drop").strip().lower()
    if quality_gate_mode not in {"drop", "downweight"}:
        LOGGER.warning("Unknown face_alignment.quality_gate.mode=%r; falling back to 'drop'", quality_gate_mode)
        quality_gate_mode = "drop"

    alignment_quality_index: Dict[Tuple[int, int], float] = {}
    if quality_gate_enabled:
        alignment_quality_index = _load_alignment_quality_index(manifests_dir)
        if alignment_quality_index:
            LOGGER.info(
                "Alignment quality gate enabled: threshold=%.2f, mode=%s, faces_with_quality=%d",
                quality_gate_threshold,
                quality_gate_mode,
                len(alignment_quality_index),
            )
        else:
            LOGGER.info("Alignment quality gate enabled but no aligned_faces.jsonl found")

    skipped_alignment_quality = 0  # Counter for dropped faces
    downweighted_alignment_quality = 0  # Counter for downweighted faces

    faces_total = len(samples)
    if LOCAL_MODE_INSTRUMENTATION:
        print(f"  faces_total={faces_total}", flush=True)
    progress = ProgressEmitter(
        args.ep_id,
        args.progress_file,
        frames_total=faces_total,
        secs_total=None,
        stride=1,
        fps_detected=None,
        fps_requested=None,
        run_id=run_id,
    )
    requested_embed_device = getattr(args, "embed_device", None) or args.device
    device = pick_device(requested_embed_device)
    save_frames = bool(args.save_frames)
    save_crops = bool(args.save_crops)
    jpeg_quality = max(1, min(int(args.jpeg_quality or 85), 100))
    frames_root = _frames_root_for_run(args.ep_id, run_id)
    debug_logger_obj: JsonlLogger | NullLogger | None = None
    if save_crops and debug_thumbs_enabled():
        debug_logger_obj = init_debug_logger(args.ep_id, frames_root)
    exporter = (
        FrameExporter(
            args.ep_id,
            run_id=run_id,
            save_frames=save_frames,
            save_crops=save_crops,
            jpeg_quality=jpeg_quality,
            debug_logger=debug_logger_obj,
        )
        if (save_frames or save_crops)
        else None
    )

    def _phase_meta(step: str | None = None) -> Dict[str, Any]:
        return _non_video_phase_meta(step, crop_diag_source=exporter)

    thumb_writer = ThumbWriter(
        args.ep_id,
        size=int(getattr(args, "thumb_size", 256)),
        run_id=run_id,
    )
    detector_choice = _infer_detector_from_tracks(track_path) or DEFAULT_DETECTOR
    tracker_choice = _infer_tracker_from_tracks(track_path) or DEFAULT_TRACKER
    # Determine CPU fallback policy from CLI flags
    # --allow-cpu-fallback OR --no-coreml-only enables CPU fallback
    allow_cpu_fallback = getattr(args, "allow_cpu_fallback", False) or not getattr(args, "coreml_only", True)

    # Get backend selection from embedding config (already loaded earlier)
    embedding_backend_configured = embedding_config.get("embedding", {}).get("backend", "pytorch")
    if isinstance(embedding_backend_configured, str):
        embedding_backend_configured = embedding_backend_configured.strip().lower() or "pytorch"
    else:
        embedding_backend_configured = "pytorch"
    embedding_backend_configured_effective = embedding_backend_configured
    embedding_backend_platform_reason: str | None = None
    if embedding_backend_configured == "tensorrt":
        system = platform.system()
        _requested_cuda, resolved_cuda, _reason = _resolve_torch_device_request("cuda")
        cuda_available = resolved_cuda == "cuda"
        if system != "Linux":
            embedding_backend_configured_effective = "pytorch"
            embedding_backend_platform_reason = "TensorRT not supported on this platform; using PyTorch backend."
        elif not cuda_available:
            embedding_backend_configured_effective = "pytorch"
            embedding_backend_platform_reason = "TensorRT requires CUDA; CUDA not available; using PyTorch backend."
    tensorrt_config_rel = embedding_config.get("embedding", {}).get(
        "tensorrt_config",
        "config/pipeline/arcface_tensorrt.yaml",
    )
    # Resolve relative config paths against REPO_ROOT so subprocess CWD doesn't matter.
    if isinstance(tensorrt_config_rel, str) and tensorrt_config_rel.strip():
        tensorrt_config_path = tensorrt_config_rel.strip()
    else:
        tensorrt_config_path = "config/pipeline/arcface_tensorrt.yaml"
    if not Path(tensorrt_config_path).is_absolute():
        tensorrt_config_path = str(REPO_ROOT / tensorrt_config_path)
    fallback_cfg = embedding_config.get("fallback", {}) if isinstance(embedding_config.get("fallback"), dict) else {}
    fallback_to_pytorch = bool(fallback_cfg.get("fallback_to_pytorch", True))
    allow_embedding_fallback = allow_cpu_fallback
    if embedding_backend_configured_effective == "tensorrt":
        allow_embedding_fallback = allow_cpu_fallback or fallback_to_pytorch

    # Emit progress during model loading (can take time for CoreML compilation)
    print(
        "[INIT] Loading embedding backend "
        f"(configured={embedding_backend_configured}, effective={embedding_backend_configured_effective}, device={device})...",
        flush=True,
    )
    embedder = get_embedding_backend(
        backend_type=embedding_backend_configured_effective,
        device=device,
        tensorrt_config=tensorrt_config_path,
        allow_cpu_fallback=allow_embedding_fallback,
    )
    embedder.ensure_ready()
    embed_device = embedder.resolved_device
    embedding_backend_actual = (
        getattr(embedder, "active_backend_label", None) or embedding_backend_configured_effective
    )
    embedding_backend_fallback_reason = getattr(embedder, "fallback_reason", None) or embedding_backend_platform_reason
    embedding_model_name = ARC_FACE_MODEL_NAME
    crop_interval_frames = getattr(args, "sample_every_n_frames", None)
    print(
        "[INIT] Embedding backend ready "
        f"(configured={embedding_backend_configured}, effective={embedding_backend_configured_effective}, "
        f"resolved_device={embed_device}, "
        f"crop_interval_frames={crop_interval_frames}, allow_cpu_fallback={allow_embedding_fallback})",
        flush=True,
    )
    LOGGER.info(
        "[faces_embed] embedding_backend_configured=%s embedding_backend_effective=%s embedding_backend_actual=%s resolved_device=%s model=%s",
        embedding_backend_configured,
        embedding_backend_configured_effective,
        embedding_backend_actual,
        embed_device,
        embedding_model_name,
    )

    manifests_dir = _manifests_dir_for_run(args.ep_id, run_id)
    faces_path = manifests_dir / "faces.jsonl"
    aligned_faces_dir = manifests_dir / "face_alignment"
    aligned_faces_path = aligned_faces_dir / "aligned_faces.jsonl"
    aligned_faces_tmp_path = aligned_faces_dir / "aligned_faces.jsonl.tmp"
    aligned_faces_handle = None
    alignment_landmarks_index: Dict[Tuple[int, int], Dict[str, Any]] = {}
    alignment_written_keys: set[Tuple[int, int]] = set()
    alignment_used_count = 0
    alignment_failed_count = 0
    fan2d_mod = None
    fan_aligner = None
    if alignment_run_enabled:
        aligned_faces_dir.mkdir(parents=True, exist_ok=True)
        alignment_landmarks_index = _load_alignment_landmarks_index(aligned_faces_path)
        try:
            aligned_faces_handle = open(aligned_faces_tmp_path, "w", encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("Unable to write aligned_faces.jsonl (%s); continuing without artifact output", exc)
            aligned_faces_handle = None
    video_path = get_path(args.ep_id, "video")
    frame_decoder: FrameDecoder | None = None
    track_embeddings: Dict[int, List[TrackEmbeddingSample]] = defaultdict(list)
    track_best_thumb: Dict[int, tuple[float, str, str | None]] = {}
    embeddings_array: List[np.ndarray] = []

    # Load seeds for seed-based face matching
    show_seeds = []
    seed_match_stats = {"enabled": SEED_BOOST_ENABLED, "matches": 0, "total": 0}
    if SEED_BOOST_ENABLED:
        show_id = _parse_ep_id_for_show(args.ep_id)
        if show_id:
            show_seeds = _load_show_seeds(show_id)
            if show_seeds:
                LOGGER.info("Loaded %d seed embeddings for show %s", len(show_seeds), show_id)

    faces_done = 0
    started_at = _utcnow_iso()
    stage_heartbeat = StageStatusHeartbeat(
        ep_id=args.ep_id,
        run_id=run_id,
        stage_key="faces",
        frames_total=faces_total,
        started_at=started_at,
    )
    if run_id:
        try:
            write_stage_started(
                args.ep_id,
                run_id,
                "faces",
                started_at=datetime.fromisoformat(started_at.replace("Z", "+00:00")),
            )
        except Exception as exc:  # pragma: no cover - best effort status update
            LOGGER.warning("[episode_status] Failed to mark faces start: %s", exc)
    faces_embed_succeeded = False
    try:
        def _emit_faces_progress(
            done: int,
            *,
            subphase: str,
            message: str | None = None,
            force: bool = False,
            summary: Dict[str, Any] | None = None,
            step: str | None = None,
            mark_frames_done: bool = False,
            mark_finalize_start: bool = False,
            mark_end: bool = False,
        ) -> None:
            extra = _phase_meta(step)
            if message:
                extra["message"] = message
            progress.emit(
                done,
                phase="faces_embed",
                device=device,
                detector=detector_choice,
                tracker=tracker_choice,
                resolved_device=embed_device,
                summary=summary,
                force=force,
                extra=extra,
            )
            stage_heartbeat.update(
                done=done,
                phase=subphase,
                message=message,
                force=force,
                mark_frames_done=mark_frames_done,
                mark_finalize_start=mark_finalize_start,
                mark_end=mark_end,
            )

        _emit_faces_progress(
            0,
            subphase="running_frames",
            message="Embedding faces",
            force=True,
        )
        rows: List[Dict[str, Any]] = []
        
        # Group samples by frame_idx for batch embedding (CPU optimization)
        # This reduces CoreML invocations from N faces to M frames (where M << N)
        # Example: 24,061 faces → ~800 frames = 96.7% reduction in model calls
        samples_by_frame = []
        for frame_idx, frame_group in groupby(samples, key=lambda s: s["frame_idx"]):
            samples_by_frame.append((frame_idx, list(frame_group)))
        
        LOGGER.info("Processing %d faces across %d frames (avg %.1f faces/frame)", 
                    len(samples), len(samples_by_frame), len(samples) / max(len(samples_by_frame), 1))
        
        # Process all faces from each frame together in a single batch
        frames_processed = 0
        total_frames = len(samples_by_frame)
        for frame_idx, frame_samples in samples_by_frame:
            # Decode frame ONCE for all faces in this frame
            if not video_path.exists():
                raise FileNotFoundError("Local video not found for crop export")
            if frame_decoder is None:
                print(f"[INIT] Opening video decoder for {video_path.name}...", flush=True)
                frame_decoder = FrameDecoder(video_path)
                print(f"[INIT] Video decoder ready", flush=True)

            # Wrap decode in try/except to handle corrupted frames gracefully
            image = None
            try:
                image = frame_decoder.read(frame_idx)
                frame_std = float(np.std(image)) if image is not None else 0.0
                if image is None or frame_std < 1.0:
                    LOGGER.warning(
                        "Low-variance frame %s std=%.4f; retrying decode",
                        frame_idx,
                        frame_std,
                    )
                    image = frame_decoder.read(frame_idx)
                    frame_std = float(np.std(image)) if image is not None else 0.0
            except (RuntimeError, Exception) as decode_exc:
                LOGGER.error(
                    "Decode failure on frame %s: %s",
                    frame_idx,
                    decode_exc,
                )
                image = None
                frame_std = 0.0

            if image is None or frame_std < 1.0:
                # Skip all samples in this bad frame
                for sample in frame_samples:
                    LOGGER.error(
                        "Skipping frame %s for track %s due to bad_source_frame",
                        frame_idx,
                        sample["track_id"],
                    )
                    rows.append(
                        _make_skip_face_row(
                            args.ep_id,
                            sample["track_id"],
                            frame_idx,
                            round(float(sample["ts"]), 4),
                            sample["bbox_xyxy"],
                            detector_choice,
                            "bad_source_frame",
                        )
                    )
                    faces_done = min(faces_total, faces_done + 1)
                _emit_faces_progress(
                    faces_done,
                    subphase="running_frames",
                    message="Embedding faces",
                )
                continue

            # Prepare batch: collect all valid crops from this frame
            batch_crops: List[np.ndarray] = []
            batch_metadata: List[Dict[str, Any]] = []
            
            for sample in frame_samples:
                crop_rel_path = None
                crop_s3_key = None
                thumb_rel_path = None
                thumb_s3_key = None
                raw_conf = sample.get("conf")
                if raw_conf is None:
                    raw_conf = sample.get("confidence")
                conf = float(raw_conf) if raw_conf is not None else 1.0
                quality = max(min(conf, 1.0), 0.0)
                bbox = sample["bbox_xyxy"]
                track_id = sample["track_id"]
                ts_val = round(float(sample["ts"]), 4)
                landmarks = sample.get("landmarks")

                # Alignment quality gate (cached; drop-only)
                if quality_gate_enabled and quality_gate_mode == "drop" and alignment_quality_index:
                    aq_key = (int(track_id), int(frame_idx))
                    aq_score = alignment_quality_index.get(aq_key)
                    if aq_score is not None and aq_score < quality_gate_threshold:
                        rows.append(
                            _make_skip_face_row(
                                args.ep_id,
                                track_id,
                                frame_idx,
                                ts_val,
                                bbox,
                                detector_choice,
                                f"low_alignment_quality:{aq_score:.3f}",
                            )
                        )
                        skipped_alignment_quality += 1
                        faces_done = min(faces_total, faces_done + 1)
                        continue

                # Export frame/crop if requested
                if exporter and image is not None:
                    exporter.export(frame_idx, image, [(track_id, bbox)], ts=ts_val)
                    if exporter.save_crops:
                        crop_rel_path = exporter.crop_rel_path(track_id, frame_idx)
                        if s3_prefixes and s3_prefixes.get("crops"):
                            crop_s3_key = f"{s3_prefixes['crops']}{exporter.crop_component(track_id, frame_idx)}"

                # Validate bbox before cropping to prevent NoneType multiply errors
                validated_bbox, bbox_err = _safe_bbox_or_none(bbox)
                if validated_bbox is None:
                    rows.append(
                        _make_skip_face_row(
                            args.ep_id,
                            track_id,
                            frame_idx,
                            ts_val,
                            bbox if bbox is not None else [],
                            detector_choice,
                            f"invalid_bbox_{bbox_err}",
                            crop_rel_path=crop_rel_path,
                            crop_s3_key=crop_s3_key,
                            thumb_rel_path=thumb_rel_path,
                            thumb_s3_key=thumb_s3_key,
                        )
                    )
                    faces_done = min(faces_total, faces_done + 1)
                    continue

                crop, crop_err = _prepare_face_crop(image, validated_bbox, landmarks)
                if crop is None:
                    rows.append(
                        _make_skip_face_row(
                            args.ep_id,
                            track_id,
                            frame_idx,
                            ts_val,
                            bbox,
                            detector_choice,
                            crop_err or "crop_failed",
                            crop_rel_path=crop_rel_path,
                            crop_s3_key=crop_s3_key,
                            thumb_rel_path=None,
                            thumb_s3_key=None,
                        )
                    )
                    faces_done = min(faces_total, faces_done + 1)
                    continue

                alignment_used = False
                alignment_quality_score: float | None = None
                alignment_quality_source: str | None = None
                alignment_source: str | None = None
                alignment_quality_weight: float | None = None
                pose_yaw: float | None = None
                pose_pitch: float | None = None
                pose_roll: float | None = None
                pose_reprojection_error_px: float | None = None
                pose_source: str | None = None
                if alignment_run_enabled:
                    face_alignment_cfg = alignment_config.get("face_alignment", {}) if isinstance(
                        alignment_config.get("face_alignment"), dict
                    ) else {}
                    processing_cfg = face_alignment_cfg.get("processing", {}) if isinstance(
                        face_alignment_cfg.get("processing"), dict
                    ) else {}
                    model_cfg = face_alignment_cfg.get("model", {}) if isinstance(
                        face_alignment_cfg.get("model"), dict
                    ) else {}
                    quality_cfg = face_alignment_cfg.get("quality", {}) if isinstance(
                        face_alignment_cfg.get("quality"), dict
                    ) else {}
                    key = (int(track_id), int(frame_idx))
                    cached = alignment_landmarks_index.get(key)
                    landmarks_68 = None
                    if isinstance(cached, dict) and cached.get("landmarks_68") is not None:
                        if fan2d_mod is None:
                            try:
                                from py_screenalytics.face_alignment import fan2d as _fan2d  # type: ignore

                                fan2d_mod = _fan2d
                            except Exception as exc:
                                LOGGER.warning("Face alignment enabled but fan2d module unavailable: %s", exc)
                                fan2d_mod = False
                        if fan2d_mod:
                            landmarks_68 = fan2d_mod.coerce_landmarks_68(cached.get("landmarks_68"))
                        if landmarks_68 is not None:
                            alignment_source = "aligned_faces"
                            alignment_quality_source = str(cached.get("alignment_quality_source") or "heuristic")
                            aq_cached = cached.get("alignment_quality")
                            if aq_cached is not None:
                                try:
                                    alignment_quality_score = float(aq_cached)
                                except (TypeError, ValueError):
                                    alignment_quality_score = None
                            pose_yaw_cached = cached.get("pose_yaw")
                            if pose_yaw_cached is not None:
                                try:
                                    pose_yaw = float(pose_yaw_cached)
                                except (TypeError, ValueError):
                                    pose_yaw = None
                            pose_pitch_cached = cached.get("pose_pitch")
                            if pose_pitch_cached is not None:
                                try:
                                    pose_pitch = float(pose_pitch_cached)
                                except (TypeError, ValueError):
                                    pose_pitch = None
                            pose_roll_cached = cached.get("pose_roll")
                            if pose_roll_cached is not None:
                                try:
                                    pose_roll = float(pose_roll_cached)
                                except (TypeError, ValueError):
                                    pose_roll = None
                            pose_err_cached = cached.get("pose_reprojection_error_px")
                            if pose_err_cached is not None:
                                try:
                                    pose_reprojection_error_px = float(pose_err_cached)
                                except (TypeError, ValueError):
                                    pose_reprojection_error_px = None
                            pose_source_cached = cached.get("pose_source")
                            if isinstance(pose_source_cached, str) and pose_source_cached.strip():
                                pose_source = pose_source_cached.strip()
                    if landmarks_68 is None:
                        if fan2d_mod is None:
                            try:
                                from py_screenalytics.face_alignment import fan2d as _fan2d  # type: ignore

                                fan2d_mod = _fan2d
                            except Exception as exc:
                                LOGGER.warning("Face alignment enabled but fan2d module unavailable: %s", exc)
                                fan2d_mod = False
                        if fan2d_mod:
                            if fan_aligner is None:
                                fan_aligner = fan2d_mod.Fan2dAligner(
                                    landmarks_type=str(model_cfg.get("landmarks_type", "2D")),
                                    device=str(processing_cfg.get("device", "auto")),
                                    flip_input=bool(model_cfg.get("flip_input", False)),
                                )
                            try:
                                landmarks_68 = fan_aligner.align_face(image, validated_bbox)
                            except Exception as exc:
                                alignment_failed_count += 1
                                landmarks_68 = None
                                if alignment_failed_count <= 3:
                                    LOGGER.warning("FAN alignment failed for track=%s frame=%s: %s", track_id, frame_idx, exc)
                            if landmarks_68 is not None:
                                alignment_source = "fan2d"
                                alignment_quality_source = "heuristic"
                                try:
                                    alignment_quality_score = fan2d_mod.compute_alignment_quality(
                                        validated_bbox,
                                        landmarks_68,
                                        min_face_size=int(quality_cfg.get("min_face_size", 20)),
                                    )
                                except Exception:
                                    alignment_quality_score = None
                                alignment_landmarks_index[key] = {
                                    "landmarks_68": landmarks_68,
                                    "alignment_quality": alignment_quality_score,
                                    "alignment_quality_source": alignment_quality_source,
                                }
                    if landmarks_68 is not None:
                        if head_pose_enabled and pose_yaw is None:
                            should_run_pose = (int(frame_idx) % head_pose_every_n_frames) == 0
                            if (
                                not should_run_pose
                                and head_pose_run_on_uncertain
                                and alignment_quality_score is not None
                            ):
                                should_run_pose = alignment_quality_score < head_pose_uncertainty_threshold
                            if should_run_pose:
                                try:
                                    from py_screenalytics.face_alignment import head_pose as _head_pose  # type: ignore

                                    estimate = _head_pose.estimate_head_pose_pnp(
                                        landmarks_68,
                                        image_shape=image.shape,
                                    )
                                    if estimate is not None:
                                        pose_yaw = estimate.yaw
                                        pose_pitch = estimate.pitch
                                        pose_roll = estimate.roll
                                        pose_reprojection_error_px = estimate.reprojection_error_px
                                        pose_source = estimate.source
                                        head_pose_computed += 1
                                        cached_entry = alignment_landmarks_index.get(key)
                                        if isinstance(cached_entry, dict):
                                            cached_entry["pose_yaw"] = pose_yaw
                                            cached_entry["pose_pitch"] = pose_pitch
                                            cached_entry["pose_roll"] = pose_roll
                                            cached_entry["pose_reprojection_error_px"] = pose_reprojection_error_px
                                            cached_entry["pose_source"] = pose_source
                                            alignment_landmarks_index[key] = cached_entry
                                except Exception as exc:
                                    head_pose_failed += 1
                                    if head_pose_failed <= 3:
                                        LOGGER.warning(
                                            "Head pose estimation failed for track=%s frame=%s: %s",
                                            track_id,
                                            frame_idx,
                                            exc,
                                        )

                        if aligned_faces_handle and key not in alignment_written_keys:
                            try:
                                aligned_faces_handle.write(
                                    json.dumps(
                                        fan2d_mod.aligned_face_row(
                                            track_id=track_id,
                                            frame_idx=frame_idx,
                                            bbox_xyxy=validated_bbox,
                                            confidence=conf,
                                            landmarks_68=landmarks_68,
                                            alignment_quality=alignment_quality_score,
                                            alignment_quality_source=alignment_quality_source or "heuristic",
                                            pose_yaw=pose_yaw,
                                            pose_pitch=pose_pitch,
                                            pose_roll=pose_roll,
                                            pose_reprojection_error_px=pose_reprojection_error_px,
                                            pose_source=pose_source,
                                        )
                                    )
                                    + "\n"
                                )
                                alignment_written_keys.add(key)
                            except Exception:
                                pass

                        if (
                            quality_gate_enabled
                            and alignment_quality_score is not None
                            and alignment_quality_score < quality_gate_threshold
                        ):
                            if quality_gate_mode == "drop":
                                rows.append(
                                    _make_skip_face_row(
                                        args.ep_id,
                                        track_id,
                                        frame_idx,
                                        ts_val,
                                        bbox,
                                        detector_choice,
                                        f"low_alignment_quality:{alignment_quality_score:.3f}",
                                        crop_rel_path=crop_rel_path,
                                        crop_s3_key=crop_s3_key,
                                        thumb_rel_path=None,
                                        thumb_s3_key=None,
                                    )
                                )
                                skipped_alignment_quality += 1
                                faces_done = min(faces_total, faces_done + 1)
                                continue
                            if quality_gate_mode == "downweight":
                                denom = max(quality_gate_threshold, 1e-6)
                                factor = max(0.0, min(float(alignment_quality_score) / denom, 1.0))
                                alignment_quality_weight = factor
                                quality = max(min(quality * factor, 1.0), 0.0)
                                downweighted_alignment_quality += 1

                        if fan2d_mod and alignment_source in {"fan2d", "aligned_faces"}:
                            try:
                                crop = fan2d_mod.align_face_crop(
                                    image,
                                    landmarks_68,
                                    crop_size=alignment_crop_size,
                                    margin=alignment_crop_margin,
                                )
                                alignment_used = True
                                alignment_used_count += 1
                            except Exception as exc:
                                alignment_failed_count += 1
                                if alignment_failed_count <= 3:
                                    LOGGER.warning(
                                        "Aligned crop failed for track=%s frame=%s (source=%s): %s",
                                        track_id,
                                        frame_idx,
                                        alignment_source,
                                        exc,
                                    )

                crop_std = float(np.std(crop))
                blur_score = _estimate_blur_score(crop)
                skip_reason: str | None = None
                skip_meta: str | None = None

                # Use lower thresholds for single-face tracks to avoid orphaned clusters
                # Single-face tracks have no redundancy, so we're more permissive
                track_sample_count = sample.get("track_sample_count", 999)
                is_single_face_track = track_sample_count == 1

                min_conf = FACE_MIN_CONFIDENCE_SINGLE if is_single_face_track else FACE_MIN_CONFIDENCE
                min_std = FACE_MIN_STD_SINGLE if is_single_face_track else FACE_MIN_STD
                min_blur = FACE_MIN_BLUR_SINGLE if is_single_face_track else FACE_MIN_BLUR

                if conf < min_conf:
                    skip_reason = "low_confidence"
                    skip_meta = f"{conf:.2f}"
                elif crop_std < min_std:
                    skip_reason = "low_contrast"
                    skip_meta = f"{crop_std:.2f}"
                elif blur_score < min_blur:
                    skip_reason = "blurry"
                    skip_meta = f"{blur_score:.1f}"
                if skip_reason:
                    reason = f"{skip_reason}:{skip_meta}" if skip_meta else skip_reason
                    rows.append(
                        _make_skip_face_row(
                            args.ep_id,
                            track_id,
                            frame_idx,
                            ts_val,
                            bbox,
                            detector_choice,
                            reason,
                            crop_rel_path=crop_rel_path,
                            crop_s3_key=crop_s3_key,
                            thumb_rel_path=None,
                            thumb_s3_key=None,
                            # Pass quality metrics for smarter rescue decisions
                            blur_score=blur_score,
                            confidence=conf,
                            contrast=crop_std,
                        )
                    )
                    faces_done = min(faces_total, faces_done + 1)
                    continue

                # Quality checks passed - create thumbnail for this valid face
                if image is not None:
                    thumb_rel_path, _ = thumb_writer.write(
                        image,
                        validated_bbox,
                        track_id,
                        frame_idx,
                        prepared_crop=crop,
                    )
                    if thumb_rel_path and s3_prefixes and s3_prefixes.get("thumbs_tracks"):
                        thumb_s3_key = f"{s3_prefixes['thumbs_tracks']}{thumb_rel_path}"

                # Add to batch for embedding
                batch_crops.append(crop)
                batch_metadata.append({
                    "sample": sample,
                    "bbox": bbox,
                    "validated_bbox": validated_bbox,
                    "landmarks": landmarks,
                    "alignment_used": alignment_used,
                    "alignment_source": alignment_source,
                    "alignment_quality": alignment_quality_score,
                    "alignment_quality_weight": alignment_quality_weight,
                    "pose_yaw": pose_yaw,
                    "pose_pitch": pose_pitch,
                    "pose_roll": pose_roll,
                    "pose_reprojection_error_px": pose_reprojection_error_px,
                    "pose_source": pose_source,
                    "crop_rel_path": crop_rel_path,
                    "crop_s3_key": crop_s3_key,
                    "thumb_rel_path": thumb_rel_path,
                    "thumb_s3_key": thumb_s3_key,
                    "conf": conf,
                    "quality": quality,
                    "track_id": track_id,
                    "ts_val": ts_val,
                })

            # BATCH EMBEDDING: Process all valid crops from this frame in ONE CoreML call
            # This reduces CPU by 60% - from 24,061 calls → ~800 calls (96.7% reduction)
            if batch_crops:
                # Log first few frames for debugging slow starts
                if frames_processed < 3:
                    print(f"[EMBED] Frame {frame_idx}: embedding {len(batch_crops)} faces...", flush=True)
                embed_start = time.time()
                embeddings = embedder.encode(batch_crops)
                embed_time = time.time() - embed_start
                if frames_processed < 3:
                    print(f"[EMBED] Frame {frame_idx}: done in {embed_time:.2f}s", flush=True)
                frames_processed += 1
                
                for embedding_vec, meta in zip(embeddings, batch_metadata):
                    track_id = meta["track_id"]
                    ts_val = meta["ts_val"]
                    bbox = meta["bbox"]
                    
                    # Check for zero-norm embedding (invalid)
                    embedding_norm = float(np.linalg.norm(embedding_vec))
                    if embedding_norm < 1e-6:
                        rows.append(
                            _make_skip_face_row(
                                args.ep_id,
                                track_id,
                                frame_idx,
                                ts_val,
                                bbox,
                                detector_choice,
                                "zero_norm_embedding",
                                crop_rel_path=meta["crop_rel_path"],
                                crop_s3_key=meta["crop_s3_key"],
                                thumb_rel_path=meta["thumb_rel_path"],
                                thumb_s3_key=meta["thumb_s3_key"],
                            )
                        )
                        faces_done = min(faces_total, faces_done + 1)
                        continue

                    track_embeddings[track_id].append((float(meta["quality"]), embedding_vec.copy()))
                    embeddings_array.append(embedding_vec)

                    # Check for seed match
                    seed_cast_id = None
                    seed_similarity = None
                    if show_seeds and SEED_BOOST_ENABLED:
                        seed_match_stats["total"] += 1
                        match_result = _find_best_seed_match(embedding_vec, show_seeds, min_sim=SEED_BOOST_MIN_SIM)
                        if match_result:
                            seed_cast_id, seed_similarity = match_result
                            seed_match_stats["matches"] += 1
                    
                    # Track best thumbnail
                    if meta["thumb_rel_path"]:
                        prev = track_best_thumb.get(track_id)
                        if not prev or meta["quality"] > prev[0]:
                            track_best_thumb[track_id] = (meta["quality"], meta["thumb_rel_path"], meta["thumb_s3_key"])

                    face_row = {
                        "ep_id": args.ep_id,
                        "face_id": f"face_{track_id:04d}_{frame_idx:06d}",
                        "track_id": track_id,
                        "frame_idx": frame_idx,
                        "ts": ts_val,
                        "bbox_xyxy": bbox,
                        "conf": round(float(meta["conf"]), 4),
                        "quality": round(float(meta["quality"]), 4),
                        "embedding": embedding_vec.tolist(),
                        "embedding_model": embedding_model_name,
                        "detector": detector_choice,
                        "pipeline_ver": PIPELINE_VERSION,
                    }
                    if meta["crop_rel_path"]:
                        face_row["crop_rel_path"] = meta["crop_rel_path"]
                    if meta["crop_s3_key"]:
                        face_row["crop_s3_key"] = meta["crop_s3_key"]
                    if meta["thumb_rel_path"]:
                        face_row["thumb_rel_path"] = meta["thumb_rel_path"]
                    if meta["thumb_s3_key"]:
                        face_row["thumb_s3_key"] = meta["thumb_s3_key"]
                    if meta["landmarks"]:
                        face_row["landmarks"] = [round(float(val), 4) for val in meta["landmarks"]]
                    pose_yaw_val = meta.get("pose_yaw")
                    if pose_yaw_val is not None:
                        try:
                            face_row["pose_yaw"] = round(float(pose_yaw_val), 3)
                        except (TypeError, ValueError):
                            pass
                    pose_pitch_val = meta.get("pose_pitch")
                    if pose_pitch_val is not None:
                        try:
                            face_row["pose_pitch"] = round(float(pose_pitch_val), 3)
                        except (TypeError, ValueError):
                            pass
                    pose_roll_val = meta.get("pose_roll")
                    if pose_roll_val is not None:
                        try:
                            face_row["pose_roll"] = round(float(pose_roll_val), 3)
                        except (TypeError, ValueError):
                            pass
                    pose_err_val = meta.get("pose_reprojection_error_px")
                    if pose_err_val is not None:
                        try:
                            face_row["pose_reprojection_error_px"] = round(float(pose_err_val), 3)
                        except (TypeError, ValueError):
                            pass
                    pose_source_val = meta.get("pose_source")
                    if isinstance(pose_source_val, str) and pose_source_val.strip():
                        face_row["pose_source"] = pose_source_val.strip()
                    if meta.get("alignment_used"):
                        face_row["alignment_used"] = True
                        if meta.get("alignment_source"):
                            face_row["alignment_source"] = str(meta["alignment_source"])
                        aq_val = meta.get("alignment_quality")
                        if aq_val is not None:
                            try:
                                face_row["alignment_quality"] = round(float(aq_val), 4)
                            except (TypeError, ValueError):
                                pass
                        aq_weight = meta.get("alignment_quality_weight")
                        if aq_weight is not None:
                            try:
                                face_row["alignment_quality_weight"] = round(float(aq_weight), 4)
                            except (TypeError, ValueError):
                                pass
                    if seed_cast_id:
                        face_row["seed_cast_id"] = seed_cast_id
                        face_row["seed_similarity"] = round(float(seed_similarity), 4)
                    rows.append(face_row)
                    faces_done = min(faces_total, faces_done + 1)

            # Emit progress per frame (not per face) - reduces progress overhead
            _emit_faces_progress(
                faces_done,
                subphase="running_frames",
                message="Embedding faces",
            )
        _emit_faces_progress(
            faces_done,
            subphase="running_frames",
            message="Embedding faces",
        )

        # Force emit final progress after loop completes
        _emit_faces_progress(
            faces_done,
            subphase="running_frames",
            message="Embedding faces",
            force=True,
        )

        finalize_message = "Finalizing embeddings / writing artifacts / syncing..."
        _emit_faces_progress(
            faces_done,
            subphase="finalizing",
            message=finalize_message,
            step="finalizing",
            force=True,
            mark_frames_done=True,
            mark_finalize_start=True,
        )

        _write_jsonl(faces_path, rows)
        if SEED_BOOST_ENABLED and seed_match_stats["total"] > 0:
            LOGGER.info(
                "Seed matching: %d/%d faces matched seeds (%.1f%%)",
                seed_match_stats["matches"],
                seed_match_stats["total"],
                100.0 * seed_match_stats["matches"] / seed_match_stats["total"],
            )

        # Log frame cache statistics
        if frame_decoder:
            cache_stats = frame_decoder.get_cache_stats()
            LOGGER.info(
                "Frame cache: %d hits, %d misses (%.1f%% hit rate, %d frames cached)",
                cache_stats["hits"],
                cache_stats["misses"],
                100.0 * cache_stats["hit_rate"],
                cache_stats["size"],
            )

        embed_path = _faces_embed_path(args.ep_id, run_id=run_id)
        if embeddings_array:
            np.save(embed_path, np.vstack(embeddings_array))
        else:
            np.save(embed_path, np.zeros((0, 512), dtype=np.float32))

        coherence_result = _update_track_embeddings(track_path, track_embeddings, track_best_thumb, embedding_model_name)
        if exporter:
            exporter.write_indexes()

        if run_id:
            _promote_run_manifests_to_root(args.ep_id, run_id, ("faces.jsonl", "tracks.jsonl"))

        # Build summary before S3 sync; finalize timing after sync completes.
        summary: Dict[str, Any] = {
            "stage": "faces_embed",
            "ep_id": args.ep_id,
            "run_id": run_id or progress.run_id,
            "faces": len(rows),
            "device": device,
            "requested_device": requested_embed_device,
            "resolved_device": embed_device,
            "detector": detector_choice,
            "tracker": tracker_choice,
            "embedding_model": embedding_model_name,
            "frames_exported": (exporter.frames_written if exporter and exporter.save_frames else 0),
            "crops_exported": (exporter.crops_written if exporter and exporter.save_crops else 0),
            "artifacts": {
                "local": {
                    "faces": str(faces_path),
                    "tracks": str(track_path),
                    "manifests_dir": str(manifests_dir),
                    "active_mirror_dir": str(get_path(args.ep_id, "detections").parent),
                    "frames_dir": (str(exporter.frames_dir) if exporter and exporter.save_frames else None),
                    "crops_dir": (str(exporter.crops_dir) if exporter and exporter.save_crops else None),
                    "thumbs_dir": str(thumb_writer.root_dir),
                    "faces_embeddings": str(embed_path),
                    "aligned_faces": (str(aligned_faces_path) if alignment_run_enabled else None),
                },
                "s3_prefixes": s3_prefixes,
            },
            "stats": {
                "faces": len(rows),
                "embedding_model": embedding_model_name,
                "alignment_used": (alignment_used_count if alignment_run_enabled else 0),
            },
            "coherence": coherence_result.get("coherence_stats", {}),
        }
        if head_pose_enabled:
            summary["head_pose_3d"] = {
                "enabled": True,
                "run_every_n_frames": head_pose_every_n_frames,
                "run_on_uncertain": head_pose_run_on_uncertain,
                "uncertainty_threshold": head_pose_uncertainty_threshold,
                "computed": head_pose_computed,
                "failed": head_pose_failed,
            }
        # Add mixed tracks if any were found
        if coherence_result.get("mixed_tracks"):
            summary["mixed_tracks"] = coherence_result["mixed_tracks"]

        def _finalize_heartbeat() -> None:
            _emit_faces_progress(
                faces_done,
                subphase="finalizing",
                message=finalize_message,
                step="finalizing",
                force=True,
            )

        # Keep heartbeats alive during long sync/finalization steps.
        s3_sync_result = _run_with_heartbeat(
            lambda: _sync_artifacts_to_s3(args.ep_id, storage, ep_ctx, exporter, thumb_writer.root_dir),
            _finalize_heartbeat,
            interval=STAGE_HEARTBEAT_INTERVAL,
        )
        summary["artifacts"]["s3_uploads"] = s3_sync_result.stats
        if s3_sync_result.errors:
            summary["artifacts"]["s3_errors"] = s3_sync_result.errors
        if not s3_sync_result.success:
            LOGGER.error("S3 sync failed for %s: %s", args.ep_id, s3_sync_result.errors)

        finished_at = _utcnow_iso()
        try:
            start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
            summary["runtime_sec"] = max((end_dt - start_dt).total_seconds(), 0.0)
        except ValueError:
            pass

        _emit_faces_progress(
            len(rows),
            subphase="done",
            message="Completed",
            summary=summary,
            step="done",
            force=True,
            mark_end=True,
        )
        progress.complete(
            summary,
            device=device,
            detector=detector_choice,
            tracker=tracker_choice,
            resolved_device=embed_device,
            step="faces_embed",
            extra=_phase_meta(),
        )

        # Brief delay to ensure final progress event is written and readable.
        time.sleep(0.2)
        _write_run_marker(
            args.ep_id,
            "faces_embed",
            {
                "phase": "faces_embed",
                "status": "success",
                "version": APP_VERSION,
                "run_id": run_id or progress.run_id,
                "faces": len(rows),
                "save_frames": save_frames,
                "save_crops": save_crops,
                "jpeg_quality": jpeg_quality,
                "thumb_size": thumb_writer.size,
                "device": device,
                "requested_device": requested_embed_device,
                "resolved_device": embed_device,
                "embedding_backend_configured": embedding_backend_configured,
                "embedding_backend_configured_effective": embedding_backend_configured_effective,
                "embedding_backend_actual": embedding_backend_actual,
                "embedding_backend_fallback_reason": embedding_backend_fallback_reason,
                "embedding_model_name": embedding_model_name,
                "started_at": started_at,
                "finished_at": finished_at,
            },
            run_id=run_id,
        )
        if run_id:
            try:
                write_stage_finished(
                    args.ep_id,
                    run_id,
                    "faces",
                    counts={"faces": len(rows)},
                    metrics={"faces": len(rows)},
                    artifact_paths=_status_artifact_paths(args.ep_id, run_id, "faces"),
                )
            except Exception as exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark faces success: %s", exc)
            if emit_manifests:
                try:
                    write_stage_manifest(
                        args.ep_id,
                        run_id,
                        "faces",
                        "SUCCESS",
                        started_at=started_at,
                        finished_at=finished_at,
                        duration_s=None,
                        counts={
                            "faces": len(rows),
                            "alignment_used": alignment_used_count if alignment_run_enabled else 0,
                        },
                        thresholds={
                            "max_samples_per_track": getattr(args, "max_samples_per_track", None),
                            "min_samples_per_track": getattr(args, "min_samples_per_track", None),
                            "sample_every_n_frames": getattr(args, "sample_every_n_frames", None),
                            "alignment_quality_threshold": quality_gate_threshold if quality_gate_enabled else None,
                            "alignment_quality_mode": quality_gate_mode if quality_gate_enabled else None,
                        },
                        artifacts=_status_artifact_paths(args.ep_id, run_id, "faces"),
                        model_versions={
                            "embedding_model": embedding_model_name,
                            "detector": detector_choice,
                            "tracker": tracker_choice,
                            "embedding_backend": embedding_backend_actual,
                        },
                    )
                except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                    LOGGER.warning("[manifest] Failed to write faces success manifest: %s", manifest_exc)

        # Log completion for local mode streaming
        if LOCAL_MODE_INSTRUMENTATION:
            runtime_sec = summary.get("runtime_sec", 0)
            print(f"[LOCAL MODE] faces_embed completed for {args.ep_id}", flush=True)
            print(f"  faces={len(rows)}, runtime={runtime_sec:.1f}s", flush=True)

        # Log alignment quality gate stats
        if quality_gate_enabled and (skipped_alignment_quality > 0 or downweighted_alignment_quality > 0):
            LOGGER.info(
                "Alignment quality gate: mode=%s, threshold=%.2f, skipped=%d, downweighted=%d (total=%d)",
                quality_gate_mode,
                quality_gate_threshold,
                skipped_alignment_quality,
                downweighted_alignment_quality,
                faces_total,
            )
            summary["alignment_quality_gating"] = {
                "enabled": True,
                "mode": quality_gate_mode,
                "threshold": quality_gate_threshold,
                "skipped": skipped_alignment_quality,
                "downweighted": downweighted_alignment_quality,
                "total": faces_total,
                "skip_rate": skipped_alignment_quality / max(faces_total, 1),
                "downweight_rate": downweighted_alignment_quality / max(faces_total, 1),
            }

        faces_embed_succeeded = True
        return summary
    except ManifestValidationError as exc:
        if run_id:
            try:
                write_stage_failed(
                    args.ep_id,
                    run_id,
                    "faces",
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark faces failure: %s", status_exc)
            if emit_manifests:
                try:
                    write_stage_manifest(
                        args.ep_id,
                        run_id,
                        "faces",
                        "FAILED",
                        started_at=started_at if "started_at" in locals() else _utcnow_iso(),
                        finished_at=_utcnow_iso(),
                        duration_s=None,
                        error=StageErrorInfo(code=type(exc).__name__, message=str(exc)),
                        thresholds={
                            "max_samples_per_track": getattr(args, "max_samples_per_track", None),
                            "min_samples_per_track": getattr(args, "min_samples_per_track", None),
                            "sample_every_n_frames": getattr(args, "sample_every_n_frames", None),
                        },
                        artifacts=_status_artifact_paths(args.ep_id, run_id, "faces"),
                    )
                except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                    LOGGER.warning("[manifest] Failed to write faces failed manifest: %s", manifest_exc)
        progress.fail(str(exc))
        raise
    except Exception as exc:
        if run_id:
            try:
                write_stage_failed(
                    args.ep_id,
                    run_id,
                    "faces",
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark faces failure: %s", status_exc)
            if emit_manifests:
                try:
                    write_stage_manifest(
                        args.ep_id,
                        run_id,
                        "faces",
                        "FAILED",
                        started_at=started_at if "started_at" in locals() else _utcnow_iso(),
                        finished_at=_utcnow_iso(),
                        duration_s=None,
                        error=StageErrorInfo(code=type(exc).__name__, message=str(exc)),
                        thresholds={
                            "max_samples_per_track": getattr(args, "max_samples_per_track", None),
                            "min_samples_per_track": getattr(args, "min_samples_per_track", None),
                            "sample_every_n_frames": getattr(args, "sample_every_n_frames", None),
                        },
                        artifacts=_status_artifact_paths(args.ep_id, run_id, "faces"),
                    )
                except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                    LOGGER.warning("[manifest] Failed to write faces failed manifest: %s", manifest_exc)
        progress.fail(str(exc))
        raise
    finally:
        if frame_decoder:
            frame_decoder.close()
        if debug_logger_obj:
            debug_logger_obj.close()
        if aligned_faces_handle:
            try:
                aligned_faces_handle.close()
            except Exception:
                pass
            if faces_embed_succeeded:
                try:
                    aligned_faces_tmp_path.replace(aligned_faces_path)
                except Exception as exc:
                    LOGGER.warning("Failed to finalize aligned_faces.jsonl: %s", exc)
            else:
                try:
                    aligned_faces_tmp_path.unlink()
                except Exception:
                    pass
        progress.close()


def _max_pairwise_cosine_distance(vectors: np.ndarray) -> float:
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        return 0.0
    max_dist = 0.0
    count = vectors.shape[0]
    for i in range(count):
        vi = vectors[i]
        for j in range(i + 1, count):
            vj = vectors[j]
            sim = float(np.dot(vi, vj))
            sim = max(min(sim, 1.0), -1.0)
            dist = 1.0 - sim
            if dist > max_dist:
                max_dist = dist
    return max_dist


def _select_track_prototype(
    samples: List[TrackEmbeddingSample],
) -> tuple[np.ndarray | None, int, float | None]:
    if not samples:
        return None, 0, None
    ranked = sorted(samples, key=lambda item: item[0], reverse=True)
    capped = ranked[:TRACK_PROTO_MAX_SAMPLES]
    vectors: list[np.ndarray] = []
    weights: list[float] = []
    for score, vec in capped:
        normed = _l2_normalize(vec)
        if normed is None:
            continue
        vectors.append(normed)
        weights.append(max(float(score), 1e-3))
    if not vectors:
        return None, 0, None
    stack = np.vstack(vectors)
    weight_arr = np.asarray(weights, dtype=np.float32)
    weight_sum = float(weight_arr.sum())
    if weight_sum <= 0:
        weight_arr = np.ones(len(vectors), dtype=np.float32) / len(vectors)
    else:
        weight_arr = weight_arr / weight_sum
    proto = np.sum(stack * weight_arr[:, None], axis=0)
    proto = _l2_normalize(proto)
    if proto is None:
        return None, stack.shape[0], None
    sims = np.array([float(np.dot(vec, proto)) for vec in stack], dtype=np.float32)
    sims = np.clip(sims, -1.0, 1.0)
    if stack.shape[0] > 2:
        median_sim = float(np.median(sims))
        cutoff = max(TRACK_PROTO_SIM_MIN, median_sim - TRACK_PROTO_SIM_DELTA)
        keep_mask = sims >= cutoff
        if keep_mask.any() and keep_mask.sum() < stack.shape[0]:
            stack = stack[keep_mask]
            weight_arr = weight_arr[keep_mask]
            weight_arr = weight_arr / max(float(weight_arr.sum()), 1e-6)
            proto = np.sum(stack * weight_arr[:, None], axis=0)
            proto = _l2_normalize(proto)
            sims = np.array([float(np.dot(vec, proto)) for vec in stack], dtype=np.float32)
            sims = np.clip(sims, -1.0, 1.0)
    spread = _max_pairwise_cosine_distance(stack)
    return proto, stack.shape[0], spread


# Threshold for flagging a track as potentially containing multiple people
# A spread ≥0.30 indicates the max pairwise cosine distance between embeddings
# is quite high, suggesting the track may have different faces
TRACK_COHERENCE_WARN_THRESHOLD = float(os.environ.get("TRACK_COHERENCE_WARN", "0.30"))


def _update_track_embeddings(
    track_path: Path,
    track_embeddings: Dict[int, List[TrackEmbeddingSample]],
    track_best_thumb: Dict[int, tuple[float, str, str | None]],
    embedding_model: str,
) -> Dict[str, Any]:
    """Update tracks with embedding info and return coherence validation stats.

    Returns:
        Dict with:
        - mixed_tracks: list of track_ids with spread >= threshold
        - coherence_stats: summary statistics
    """
    coherence_result: Dict[str, Any] = {
        "mixed_tracks": [],
        "coherence_stats": {
            "total_tracks": 0,
            "tracks_with_embeddings": 0,
            "tracks_flagged_mixed": 0,
            "max_spread": 0.0,
            "avg_spread": 0.0,
        },
    }
    if not track_path.exists():
        return coherence_result

    rows = list(_iter_jsonl(track_path))
    updated: List[dict] = []
    spreads: List[float] = []
    mixed_tracks: List[int] = []

    for row in rows:
        track_id = int(row.get("track_id", -1))
        samples = track_embeddings.get(track_id) or []
        if samples:
            proto_vec, sample_count, spread = _select_track_prototype(samples)
            row["faces_count"] = len(samples)
            if proto_vec is not None:
                row["face_embedding"] = proto_vec.tolist()
                row["face_embedding_model"] = embedding_model
                row["face_embedding_samples"] = sample_count
                if spread is not None:
                    spread_rounded = round(float(spread), 4)
                    row["face_embedding_spread"] = spread_rounded
                    spreads.append(spread_rounded)

                    # Flag tracks with high spread as potentially mixed
                    if spread_rounded >= TRACK_COHERENCE_WARN_THRESHOLD:
                        mixed_tracks.append(track_id)
                        row["coherence_warning"] = "high_spread"
                        LOGGER.warning(
                            "[COHERENCE] Track %d has high embedding spread %.3f (threshold %.2f) - "
                            "may contain multiple people",
                            track_id,
                            spread_rounded,
                            TRACK_COHERENCE_WARN_THRESHOLD,
                        )
        thumb_info = track_best_thumb.get(track_id)
        if thumb_info:
            _, rel_path, s3_key = thumb_info
            row["thumb_rel_path"] = rel_path
            if s3_key:
                row["thumb_s3_key"] = s3_key
        updated.append(row)

    if updated:
        _write_jsonl(track_path, updated)

    # Compute coherence stats
    coherence_result["mixed_tracks"] = mixed_tracks
    coherence_result["coherence_stats"]["total_tracks"] = len(rows)
    coherence_result["coherence_stats"]["tracks_with_embeddings"] = len(spreads)
    coherence_result["coherence_stats"]["tracks_flagged_mixed"] = len(mixed_tracks)
    if spreads:
        coherence_result["coherence_stats"]["max_spread"] = round(max(spreads), 4)
        coherence_result["coherence_stats"]["avg_spread"] = round(
            sum(spreads) / len(spreads), 4
        )

    if mixed_tracks:
        LOGGER.info(
            "[COHERENCE] Embed stage flagged %d tracks as potentially mixed (spread >= %.2f): %s",
            len(mixed_tracks),
            TRACK_COHERENCE_WARN_THRESHOLD,
            mixed_tracks[:10] if len(mixed_tracks) > 10 else mixed_tracks,
        )

    return coherence_result


def _run_cluster_stage(
    args: argparse.Namespace,
    storage: StorageService | None,
    ep_ctx: EpisodeContext | None,
    s3_prefixes: Dict[str, str] | None,
) -> Dict[str, Any]:
    # Log startup for local mode streaming
    if LOCAL_MODE_INSTRUMENTATION:
        device = args.device or "auto"
        cluster_thresh = getattr(args, "cluster_thresh", 0.7)
        print(f"[LOCAL MODE] cluster starting for {args.ep_id}", flush=True)
        print(f"  device={device}, cluster_thresh={cluster_thresh}", flush=True)

    run_id = getattr(args, "run_id", None)
    emit_manifests = getattr(args, "emit_manifests", True)
    if run_id:
        gate = check_prereqs("cluster", args.ep_id, run_id)
        if not gate.ok:
            blocked_reason, blocked_info = _blocked_details(gate)
            if blocked_update_needed(args.ep_id, run_id, "cluster", blocked_reason):
                try:
                    write_stage_blocked(args.ep_id, run_id, "cluster", blocked_reason)
                except Exception as status_exc:  # pragma: no cover - best effort status update
                    LOGGER.warning("[episode_status] Failed to mark cluster blocked: %s", status_exc)
                if emit_manifests:
                    try:
                        write_stage_manifest(
                            args.ep_id,
                            run_id,
                            "cluster",
                            "BLOCKED",
                            started_at=_utcnow_iso(),
                            finished_at=_utcnow_iso(),
                            duration_s=None,
                            blocked=blocked_info,
                            thresholds={
                                "cluster_thresh": getattr(args, "cluster_thresh", None),
                                "min_identity_sim": getattr(args, "min_identity_sim", None),
                                "min_cluster_size": getattr(args, "min_cluster_size", None),
                            },
                        )
                    except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                        LOGGER.warning("[manifest] Failed to write cluster blocked manifest: %s", manifest_exc)
            raise RuntimeError(blocked_reason.message)
    manifests_dir = _manifests_dir_for_run(args.ep_id, run_id)
    faces_path = manifests_dir / "faces.jsonl"
    started_at: str | None = None
    try:
        # Validate faces.jsonl exists, is non-empty, and has required fields
        faces_row_count = require_manifest(
            faces_path,
            "faces.jsonl",
            required_fields=["track_id", "embedding"],
            hint="run faces embedding first",
        )
        faces_rows = list(_iter_jsonl(faces_path))
        # Double-check we have usable embeddings
        usable_faces = [r for r in faces_rows if r.get("embedding") and len(r.get("embedding", [])) > 0]
        if not usable_faces:
            raise ManifestValidationError(
                "faces.jsonl",
                f"File has {len(faces_rows)} rows but none contain valid embeddings – "
                "faces embedding stage may have failed to compute embeddings",
                faces_path,
            )
        faces_total = len(usable_faces)
        if LOCAL_MODE_INSTRUMENTATION:
            print(f"  faces_total={faces_total} (validated with embeddings)", flush=True)
        faces_per_track: Dict[int, int] = defaultdict(int)
        for face_row in usable_faces:
            track_id_val = face_row.get("track_id")
            try:
                track_key = int(track_id_val)
            except (TypeError, ValueError):
                continue
            faces_per_track[track_key] += 1
        track_path = _tracks_path_for_run(args.ep_id, run_id)
        # Validate tracks.jsonl for cross-reference
        require_manifest(
            track_path,
            "tracks.jsonl",
            required_fields=["track_id"],
            hint="run detect/track first",
        )
        track_rows = list(_iter_jsonl(track_path))
        detector_choice = _infer_detector_from_tracks(track_path) or DEFAULT_DETECTOR
        tracker_choice = _infer_tracker_from_tracks(track_path) or DEFAULT_TRACKER
        distance_threshold = _cluster_distance_threshold(args.cluster_thresh)
        flagged_tracks: set[int] = set()
        for row in track_rows:
            track_id_val = row.get("track_id")
            try:
                track_id = int(track_id_val)
            except (TypeError, ValueError):
                continue
            spread_val = row.get("face_embedding_spread")
            if spread_val is None:
                continue
            try:
                spread = float(spread_val)
            except (TypeError, ValueError):
                continue
            if spread >= distance_threshold:
                flagged_tracks.add(track_id)
        if flagged_tracks:
            LOGGER.info(
                "Pre-cluster sanity: %s tracks flagged as mixed (spread>=%.3f)",
                len(flagged_tracks),
                distance_threshold,
            )
    except ManifestValidationError as exc:
        if run_id:
            try:
                write_stage_failed(
                    args.ep_id,
                    run_id,
                    "cluster",
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark cluster failure: %s", status_exc)
            if emit_manifests:
                try:
                    write_stage_manifest(
                        args.ep_id,
                        run_id,
                        "cluster",
                        "FAILED",
                        started_at=started_at or _utcnow_iso(),
                        finished_at=_utcnow_iso(),
                        duration_s=None,
                        error=StageErrorInfo(code=type(exc).__name__, message=str(exc)),
                        thresholds={
                            "cluster_thresh": getattr(args, "cluster_thresh", None),
                            "min_identity_sim": getattr(args, "min_identity_sim", None),
                            "min_cluster_size": getattr(args, "min_cluster_size", None),
                        },
                        artifacts=_status_artifact_paths(args.ep_id, run_id, "cluster"),
                    )
                except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                    LOGGER.warning("[manifest] Failed to write cluster failed manifest: %s", manifest_exc)
        raise
    except Exception as exc:
        if run_id:
            try:
                write_stage_failed(
                    args.ep_id,
                    run_id,
                    "cluster",
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark cluster failure: %s", status_exc)
        raise

    # =========================================================================
    # PRESERVE CAST-ASSIGNED CLUSTERS
    # Load existing identities and protect clusters linked to cast members
    # =========================================================================
    preserved_identities: List[dict] = []
    preserved_track_ids: Set[int] = set()
    max_preserved_id: int = 0
    ignore_preservation_errors = getattr(args, "ignore_preservation_errors", False)
    preservation_error: str | None = None

    identities_path = manifests_dir / "identities.json"
    # Default to True to preserve cast-assigned clusters (original behavior)
    preserve_assigned = bool(getattr(args, "preserve_assigned", True))
    if preserve_assigned and identities_path.exists():
        try:
            existing_data = json.loads(identities_path.read_text(encoding="utf-8"))
            existing_identities = existing_data.get("identities", [])

            # Count how many identities have person_id (cast-assigned)
            assigned_count = sum(1 for i in existing_identities if i.get("person_id"))
            if assigned_count > 0:
                LOGGER.info(
                    "Found %d existing identities, %d with cast assignments",
                    len(existing_identities),
                    assigned_count,
                )

            # Load people service to check cast assignments
            try:
                from apps.api.services.people import PeopleService
                people_service = PeopleService()

                # Parse show_slug from ep_id (e.g., "rhobh-s05e14" -> "RHOBH")
                import re
                ep_match = re.match(r"^(?P<show>.+)-s\d{2}e\d{2}$", args.ep_id, re.IGNORECASE)
                show_slug = ep_match.group("show").upper() if ep_match else None

                if show_slug:
                    for identity in existing_identities:
                        person_id = identity.get("person_id")
                        if not person_id:
                            continue

                        # Check if this person is linked to a cast member
                        person = people_service.get_person(show_slug, person_id)
                        if person and person.get("cast_id"):
                            # This cluster is assigned to a cast member - preserve it!
                            preserved_identities.append(identity)
                            for tid in identity.get("track_ids", []):
                                try:
                                    preserved_track_ids.add(int(tid))
                                except (TypeError, ValueError):
                                    pass

                            # Track max identity ID for renumbering new clusters
                            identity_id = identity.get("identity_id", "")
                            if identity_id.startswith("id_"):
                                try:
                                    id_num = int(identity_id[3:])
                                    max_preserved_id = max(max_preserved_id, id_num)
                                except ValueError:
                                    pass

                    if preserved_identities:
                        LOGGER.info(
                            "Preserving %d clusters (%d tracks) assigned to cast members",
                            len(preserved_identities),
                            len(preserved_track_ids),
                        )
                else:
                    # Could not parse show slug - if there are assigned identities, this is a problem
                    if assigned_count > 0:
                        preservation_error = (
                            f"Cannot preserve cast assignments: unable to parse show slug from "
                            f"ep_id '{args.ep_id}' (expected format: 'showname-s##e##'). "
                            f"{assigned_count} existing assignments may be lost."
                        )
            except ImportError:
                if assigned_count > 0:
                    preservation_error = (
                        f"Cannot preserve cast assignments: PeopleService not available. "
                        f"{assigned_count} existing assignments may be lost."
                    )
                else:
                    LOGGER.debug("PeopleService not available; no existing assignments to preserve")
            except Exception as exc:
                if assigned_count > 0:
                    preservation_error = (
                        f"Cannot preserve cast assignments: {exc}. "
                        f"{assigned_count} existing assignments may be lost."
                    )
                else:
                    LOGGER.warning("Failed to check cast assignments: %s", exc)
        except json.JSONDecodeError as exc:
            preservation_error = (
                f"Cannot preserve cast assignments: identities.json is corrupted ({exc}). "
                "Existing assignments may be lost if you proceed."
            )
        except OSError as exc:
            preservation_error = (
                f"Cannot preserve cast assignments: failed to read identities.json ({exc}). "
                "Existing assignments may be lost if you proceed."
            )

    # Handle preservation errors - fail hard unless explicitly ignored
    if preserve_assigned and preservation_error:
        if ignore_preservation_errors:
            LOGGER.warning(
                "CLUSTER PRESERVATION WARNING: %s (proceeding due to --ignore-preservation-errors)",
                preservation_error,
            )
        else:
            raise RuntimeError(
                f"CLUSTER PRESERVATION FAILED: {preservation_error}\n"
                "Use --ignore-preservation-errors to proceed anyway (may lose cast assignments)."
            )

    progress = ProgressEmitter(
        args.ep_id,
        args.progress_file,
        frames_total=faces_total,
        secs_total=None,
        stride=1,
        fps_detected=None,
        fps_requested=None,
        run_id=run_id,
    )
    device = pick_device(args.device)
    started_at = _utcnow_iso()
    stage_heartbeat = StageStatusHeartbeat(
        ep_id=args.ep_id,
        run_id=run_id,
        stage_key="cluster",
        frames_total=faces_total,
        started_at=started_at,
    )
    if run_id:
        try:
            write_stage_started(
                args.ep_id,
                run_id,
                "cluster",
                started_at=datetime.fromisoformat(started_at.replace("Z", "+00:00")),
            )
        except Exception as exc:  # pragma: no cover - best effort status update
            LOGGER.warning("[episode_status] Failed to mark cluster start: %s", exc)

    def _emit_cluster_progress(
        done: int,
        *,
        subphase: str,
        message: str | None = None,
        force: bool = False,
        summary: Dict[str, Any] | None = None,
        step: str | None = None,
        mark_frames_done: bool = False,
        mark_finalize_start: bool = False,
        mark_end: bool = False,
    ) -> None:
        extra = _non_video_phase_meta(step)
        if message:
            extra["message"] = message
        progress.emit(
            done,
            phase="cluster",
            device=device,
            detector=detector_choice,
            tracker=tracker_choice,
            resolved_device=device,
            summary=summary,
            force=force,
            extra=extra,
        )
        stage_heartbeat.update(
            done=done,
            phase=subphase,
            message=message,
            force=force,
            mark_frames_done=mark_frames_done,
            mark_finalize_start=mark_finalize_start,
            mark_end=mark_end,
        )

    _emit_cluster_progress(
        0,
        subphase="running_frames",
        message="Clustering identities",
        force=True,
    )

    try:
        embedding_rows: List[np.ndarray] = []
        track_ids: List[int] = []
        track_index: Dict[int, dict] = {}
        forced_singletons: List[List[int]] = []
        tracks_with_embeddings: Set[int] = set()

        for row in track_rows:
            track_id = int(row.get("track_id", -1))
            track_index[track_id] = row
            # Skip preserved tracks (assigned to cast members) - they stay in their existing cluster
            if track_id in preserved_track_ids:
                continue
            if track_id in flagged_tracks:
                forced_singletons.append([track_id])
                continue
            embed = row.get("face_embedding")
            if embed:
                embedding_rows.append(np.asarray(embed, dtype="float32"))
                track_ids.append(track_id)
                tracks_with_embeddings.add(track_id)

        # Add tracks with no accepted embeddings as forced singletons
        # This ensures tracks whose faces were all skipped still appear in identities.json
        for track_id, row in track_index.items():
            # Skip preserved tracks
            if track_id in preserved_track_ids:
                continue
            if track_id not in tracks_with_embeddings and track_id not in flagged_tracks:
                # Track has no embedding and wasn't already flagged
                faces_count = faces_per_track.get(track_id, 0)
                if faces_count > 0:  # Only if we saw faces for this track (even if skipped)
                    forced_singletons.append([track_id])
                    LOGGER.info(
                        "Track %d has no accepted embeddings (%d faces seen but all skipped); adding as forced singleton",
                        track_id,
                        faces_count,
                    )

        # Allow empty clustering if we have preserved identities
        if not embedding_rows and not forced_singletons and not preserved_identities:
            raise RuntimeError("No track embeddings available; rerun faces_embed with detector enabled")

        # === FIX 2: Pre-clustering validation ===
        # Reject tracks with low internal similarity before they can pollute clusters
        pre_cluster_rejected: List[Tuple[int, str]] = []
        if embedding_rows and track_index:
            track_ids, embedding_rows, pre_cluster_rejected = _validate_track_embeddings_for_clustering(
                track_ids,
                embedding_rows,
                track_index,
                min_internal_sim=0.70,  # Reject tracks with <70% internal consistency
            )
            if pre_cluster_rejected:
                LOGGER.info(
                    "Pre-clustering: rejected %d tracks with low internal similarity",
                    len(pre_cluster_rejected),
                )

        track_groups: Dict[int, List[int]] = defaultdict(list)
        if embedding_rows:
            labels = _cluster_embeddings(np.vstack(embedding_rows), args.cluster_thresh)
            for tid, label in zip(track_ids, labels):
                track_groups[label].append(tid)

        # Build track embeddings index for outlier removal and cohesion check
        track_embeddings: Dict[int, np.ndarray] = {}
        for tid, embed in zip(track_ids, embedding_rows):
            track_embeddings[tid] = embed

        # Remove low-similarity outliers from clusters
        min_identity_sim = max(0.0, min(float(args.min_identity_sim), 0.99))
        outlier_tracks: List[Tuple[int, str]] = []
        if min_identity_sim > 0.0 and track_groups:
            track_groups, outlier_tracks = _remove_low_similarity_outliers(
                track_groups,
                track_embeddings,
                min_identity_sim,
            )
            if outlier_tracks:
                LOGGER.info(
                    "Removed %d outlier tracks with similarity < %.2f from their identity clusters",
                    len(outlier_tracks),
                    min_identity_sim,
                )

        # === FIX 3: Cohesion quality gate ===
        # Split clusters with low cohesion into singletons (likely different people)
        split_clusters: List[Tuple[List[int], float, str]] = []
        if track_groups:
            track_groups, split_clusters = _split_low_cohesion_clusters(
                track_groups,
                track_embeddings,
                min_cohesion=MIN_CLUSTER_COHESION,
            )
            if split_clusters:
                total_tracks_split = sum(len(tracks) for tracks, _, _ in split_clusters)
                LOGGER.info(
                    "Cohesion quality gate: split %d low-cohesion clusters (%d tracks) into singletons",
                    len(split_clusters),
                    total_tracks_split,
                )

        # Add pre-cluster rejected tracks to outliers (they become singletons)
        outlier_tracks.extend(pre_cluster_rejected)

        min_cluster = max(1, int(args.min_cluster_size))
        identity_payload: List[dict] = []
        thumb_root = _frames_root_for_run(args.ep_id, run_id) / "thumbs"
        faces_done = 0
        # Start identity counter after max preserved ID to avoid collisions
        identity_counter = max_preserved_id + 1
        candidate_groups: List[List[int]] = list(track_groups.values())
        candidate_groups.extend(forced_singletons)

        # Add outlier tracks as separate single-track identities
        outlier_singletons: List[List[int]] = [[tid] for tid, _ in outlier_tracks]
        candidate_groups.extend(outlier_singletons)

        # Build outlier lookup for manifest metadata
        outlier_map: Dict[int, str] = {tid: reason for tid, reason in outlier_tracks}
        for tids in candidate_groups:
            buckets = [tids]
            if len(tids) < min_cluster:
                buckets = [[tid] for tid in tids]
            for bucket in buckets:
                identity_id = f"id_{identity_counter:04d}"
                identity_counter += 1
                identity_faces = sum(faces_per_track.get(tid, 0) for tid in bucket)
                if identity_faces <= 0:
                    identity_faces = len(bucket)
                rep_track_id = max(
                    bucket,
                    key=lambda tid: track_index.get(tid, {}).get("faces_count", 0),
                )
                rep_rel, rep_s3 = _materialize_identity_thumb(
                    thumb_root,
                    track_index.get(rep_track_id),
                    identity_id,
                    s3_prefixes,
                )

                # Check if any tracks in this identity are outliers
                outlier_reasons = [outlier_map[tid] for tid in bucket if tid in outlier_map]
                is_outlier_identity = len(outlier_reasons) > 0
                is_singleton_outlier = len(bucket) == 1 and is_outlier_identity

                # Compute identity cohesion (average similarity to centroid)
                cohesion_score: float | None = None
                min_sim_to_centroid: float | None = None
                if len(bucket) > 1:
                    bucket_embeds = [track_embeddings[tid] for tid in bucket if tid in track_embeddings]
                    if len(bucket_embeds) >= 2:
                        centroid = np.mean(bucket_embeds, axis=0)
                        norm_centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
                        sims = [_cosine_similarity(emb, norm_centroid) for emb in bucket_embeds]
                        cohesion_score = float(np.mean(sims))
                        min_sim_to_centroid = float(np.min(sims))

                identity_record = {
                    "identity_id": identity_id,
                    "label": None,
                    "track_ids": bucket,
                    "size": identity_faces,
                    "rep_thumb_rel_path": rep_rel,
                    "rep_thumb_s3_key": rep_s3,
                }

                # Add outlier metadata
                if is_singleton_outlier:
                    identity_record["outlier_reason"] = outlier_reasons[0]
                    identity_record["low_cohesion"] = True

                # Add cohesion stats
                if cohesion_score is not None:
                    identity_record["cohesion"] = round(cohesion_score, 4)
                if min_sim_to_centroid is not None:
                    identity_record["min_identity_sim"] = round(min_sim_to_centroid, 4)

                # Flag low cohesion if cohesion or min_sim is below threshold
                if not is_outlier_identity:
                    if cohesion_score is not None and cohesion_score < 0.80:
                        identity_record["low_cohesion"] = True
                    elif min_sim_to_centroid is not None and min_sim_to_centroid < min_identity_sim:
                        identity_record["low_cohesion"] = True

                identity_payload.append(identity_record)
                faces_done = min(faces_total, faces_done + identity_faces)
                _emit_cluster_progress(
                    faces_done,
                    subphase="running_frames",
                    message="Clustering identities",
                )

        # Force emit final progress after loop completes
        _emit_cluster_progress(
            faces_total,
            subphase="running_frames",
            message="Clustering identities",
            force=True,
        )

        # Merge preserved identities (cast-assigned) with newly clustered identities
        # Preserved identities come first to maintain their original IDs
        all_identities = preserved_identities + identity_payload
        preserved_faces = sum(
            identity.get("size", 0) for identity in preserved_identities
        )

        identities_path = manifests_dir / "identities.json"
        low_cohesion_count = sum(1 for identity in all_identities if identity.get("low_cohesion"))
        payload = {
            "ep_id": args.ep_id,
            "pipeline_ver": PIPELINE_VERSION,
            "config": {
                "cluster_thresh": args.cluster_thresh,
                "min_cluster_size": min_cluster,
                "min_identity_sim": min_identity_sim,
            },
            "stats": {
                "faces": faces_total,
                "clusters": len(all_identities),
                "mixed_tracks": len(flagged_tracks),
                "outlier_tracks": len(outlier_tracks),
                "low_cohesion_identities": low_cohesion_count,
                "preserved_clusters": len(preserved_identities),
                "preserved_tracks": len(preserved_track_ids),
            },
            "identities": all_identities,
        }
        identities_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if run_id:
            _promote_run_manifests_to_root(
                args.ep_id,
                run_id,
                ("tracks.jsonl", "faces.jsonl", "identities.json"),
            )

        if preserved_identities:
            LOGGER.info(
                "Cluster stage complete: %d preserved clusters, %d new clusters",
                len(preserved_identities),
                len(identity_payload),
            )

        # Generate track representatives and cluster centroids
        try:
            from apps.api.services.track_reps import generate_track_reps_and_centroids

            LOGGER.info(
                "Generating track representatives and cluster centroids for %s",
                args.ep_id,
            )
            track_reps_summary = generate_track_reps_and_centroids(args.ep_id, run_id=run_id)
            LOGGER.info(
                "Generated %d track reps and %d cluster centroids",
                track_reps_summary.get("tracks_with_reps", 0),
                track_reps_summary.get("centroids_computed", 0),
            )
        except Exception as exc:
            LOGGER.warning("Failed to generate track representatives: %s", exc)

        # Build summary (S3 sync + fusion may augment this before completion).
        summary: Dict[str, Any] = {
            "stage": "cluster",
            "ep_id": args.ep_id,
            "run_id": run_id or progress.run_id,
            "identities_count": len(identity_payload),
            "faces_count": faces_total,
            "device": device,
            "requested_device": args.device,
            "resolved_device": device,
            "detector": detector_choice,
            "tracker": tracker_choice,
            "artifacts": {
                "local": {
                    "faces": str(faces_path),
                    "identities": str(identities_path),
                    "tracks": str(track_path),
                    "manifests_dir": str(manifests_dir),
                    "active_mirror_dir": str(get_path(args.ep_id, "detections").parent),
                },
                "s3_prefixes": s3_prefixes,
            },
            "stats": payload["stats"],
        }

        finalize_message = "Finalizing clusters / syncing artifacts..."
        _emit_cluster_progress(
            faces_total,
            subphase="finalizing",
            message=finalize_message,
            step="finalizing",
            force=True,
            mark_frames_done=True,
            mark_finalize_start=True,
        )

        # Run body tracking fusion (best-effort, requires body_tracks + faces to exist).
        try:
            fusion_result = _maybe_run_body_tracking_fusion(
                ep_id=args.ep_id,
                run_id=run_id,
                effective_run_id=run_id or progress.run_id,
                emit_manifests=emit_manifests,
            )
            if fusion_result and fusion_result.get("status") == "success":
                summary["body_tracking_fusion"] = fusion_result
        except Exception as exc:
            LOGGER.warning("[cluster] Body tracking fusion failed (non-fatal): %s", exc)

        # Guardrail: when STORAGE_REQUIRE_S3=true (or EXPORT_REQUIRE_S3=true), do not report completion
        # until S3 sync succeeds. This prevents the "cluster complete" UI race where an immediate
        # PDF export can't hydrate a consistent run-scoped bundle from S3.
        s3_required = False
        for var in ("STORAGE_REQUIRE_S3", "EXPORT_REQUIRE_S3"):
            if os.environ.get(var, "").strip().lower() in ("1", "true", "yes", "on"):
                s3_required = True
                break

        def _finalize_tick(step_label: str, message: str) -> None:
            _emit_cluster_progress(
                faces_total,
                subphase="finalizing",
                message=message,
                step=step_label,
                force=True,
            )

        # S3 sync should run after all artifacts are written so S3-first mode (delete local)
        # doesn't race exports/reporting that hydrate from S3.
        _finalize_tick("s3_sync_legacy", "Syncing artifacts to S3...")
        s3_sync_result = _run_with_heartbeat(
            lambda: _sync_artifacts_to_s3(args.ep_id, storage, ep_ctx, exporter=None, thumb_dir=thumb_root),
            lambda: _finalize_tick("s3_sync_legacy", "Syncing artifacts to S3..."),
            interval=STAGE_HEARTBEAT_INTERVAL,
        )
        summary["artifacts"]["s3_uploads"] = s3_sync_result.stats
        if s3_sync_result.errors:
            summary["artifacts"]["s3_errors"] = s3_sync_result.errors
        if not s3_sync_result.success:
            LOGGER.error("S3 sync failed for %s: %s", args.ep_id, s3_sync_result.errors)
            if s3_required:
                raise RuntimeError(f"S3 sync failed for {args.ep_id}: {s3_sync_result.errors}")

        # Sync run-scoped artifacts to S3 with deterministic key layout
        if run_id:
            try:
                from apps.api.services.run_artifact_store import sync_run_artifacts_to_s3

                _finalize_tick("s3_sync_run_scoped", "Syncing run bundle to S3...")
                run_sync_result = _run_with_heartbeat(
                    lambda: sync_run_artifacts_to_s3(
                        ep_id=args.ep_id,
                        run_id=run_id,
                        # Fail loud when STORAGE_REQUIRE_S3=true so runs aren't marked complete without artifacts.
                        fail_on_error=s3_required,
                    ),
                    lambda: _finalize_tick("s3_sync_run_scoped", "Syncing run bundle to S3..."),
                    interval=STAGE_HEARTBEAT_INTERVAL,
                )
                summary["artifacts"]["run_scoped_s3"] = run_sync_result.to_dict()
                if run_sync_result.success:
                    LOGGER.info(
                        "[cluster] Run-scoped S3 sync: %d artifacts uploaded to %s",
                        run_sync_result.uploaded_count,
                        run_sync_result.s3_prefix,
                    )
                else:
                    LOGGER.warning(
                        "[cluster] Run-scoped S3 sync failed: %s",
                        run_sync_result.errors,
                    )
            except Exception as exc:
                LOGGER.warning("[cluster] Run-scoped S3 sync skipped: %s", exc)
                summary["artifacts"]["run_scoped_s3"] = {"error": str(exc)}
                if s3_required:
                    raise

        finished_at = _utcnow_iso()
        try:
            start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
            summary["runtime_sec"] = max((end_dt - start_dt).total_seconds(), 0.0)
        except ValueError:
            pass

        # Emit completion after fusion + S3 sync so downstream steps see a consistent run bundle.
        _emit_cluster_progress(
            faces_total,
            subphase="done",
            message="Completed",
            summary=summary,
            step="done",
            force=True,
            mark_end=True,
        )
        progress.complete(
            summary,
            device=device,
            detector=detector_choice,
            tracker=tracker_choice,
            resolved_device=device,
            step="cluster",
            extra=_non_video_phase_meta(),
        )

        # Brief delay to ensure final progress event is written and readable.
        time.sleep(0.2)
        _write_run_marker(
            args.ep_id,
            "cluster",
            {
                "phase": "cluster",
                "status": "success",
                "version": APP_VERSION,
                "run_id": run_id or progress.run_id,
                "faces": faces_total,
                "identities": len(identity_payload),
                "cluster_thresh": args.cluster_thresh,
                "min_cluster_size": args.min_cluster_size,
                "min_identity_sim": args.min_identity_sim,
                "device": device,
                "requested_device": args.device,
                "resolved_device": device,
                "started_at": started_at,
                "finished_at": finished_at,
            },
            run_id=run_id,
        )
        if run_id:
            try:
                write_stage_finished(
                    args.ep_id,
                    run_id,
                    "cluster",
                    counts={"faces": faces_total, "identities": len(identity_payload)},
                    metrics={"faces": faces_total, "identities": len(identity_payload)},
                    artifact_paths=_status_artifact_paths(args.ep_id, run_id, "cluster"),
                )
            except Exception as exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark cluster success: %s", exc)
            if emit_manifests:
                try:
                    write_stage_manifest(
                        args.ep_id,
                        run_id,
                        "cluster",
                        "SUCCESS",
                        started_at=started_at,
                        finished_at=finished_at,
                        duration_s=None,
                        counts={"faces": faces_total, "identities": len(identity_payload)},
                        thresholds={
                            "cluster_thresh": getattr(args, "cluster_thresh", None),
                            "min_identity_sim": getattr(args, "min_identity_sim", None),
                            "min_cluster_size": getattr(args, "min_cluster_size", None),
                        },
                        artifacts=_status_artifact_paths(args.ep_id, run_id, "cluster"),
                        model_versions={
                            "detector": detector_choice,
                            "tracker": tracker_choice,
                        },
                    )
                except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                    LOGGER.warning("[manifest] Failed to write cluster success manifest: %s", manifest_exc)

        if run_id:
            run_layout.write_active_run_id(
                args.ep_id,
                run_id,
                extra={"phase": "cluster", "status": "success", "finished_at": finished_at},
            )

        # Log completion for local mode streaming
        if LOCAL_MODE_INSTRUMENTATION:
            runtime_sec = summary.get("runtime_sec", 0)
            identities = len(identity_payload)
            print(f"[LOCAL MODE] cluster completed for {args.ep_id}", flush=True)
            print(f"  identities={identities}, faces={faces_total}, runtime={runtime_sec:.1f}s", flush=True)

        return summary
    except ManifestValidationError as exc:
        if run_id:
            try:
                write_stage_failed(
                    args.ep_id,
                    run_id,
                    "cluster",
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark cluster failure: %s", status_exc)
            if emit_manifests:
                try:
                    write_stage_manifest(
                        args.ep_id,
                        run_id,
                        "cluster",
                        "FAILED",
                        started_at=started_at if "started_at" in locals() else _utcnow_iso(),
                        finished_at=_utcnow_iso(),
                        duration_s=None,
                        error=StageErrorInfo(code=type(exc).__name__, message=str(exc)),
                        thresholds={
                            "cluster_thresh": getattr(args, "cluster_thresh", None),
                            "min_identity_sim": getattr(args, "min_identity_sim", None),
                            "min_cluster_size": getattr(args, "min_cluster_size", None),
                        },
                        artifacts=_status_artifact_paths(args.ep_id, run_id, "cluster"),
                    )
                except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                    LOGGER.warning("[manifest] Failed to write cluster failed manifest: %s", manifest_exc)
        progress.fail(str(exc))
        raise
    except Exception as exc:
        if run_id:
            try:
                write_stage_failed(
                    args.ep_id,
                    run_id,
                    "cluster",
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark cluster failure: %s", status_exc)
            if emit_manifests:
                try:
                    write_stage_manifest(
                        args.ep_id,
                        run_id,
                        "cluster",
                        "FAILED",
                        started_at=started_at if "started_at" in locals() else _utcnow_iso(),
                        finished_at=_utcnow_iso(),
                        duration_s=None,
                        error=StageErrorInfo(code=type(exc).__name__, message=str(exc)),
                        thresholds={
                            "cluster_thresh": getattr(args, "cluster_thresh", None),
                            "min_identity_sim": getattr(args, "min_identity_sim", None),
                            "min_cluster_size": getattr(args, "min_cluster_size", None),
                        },
                        artifacts=_status_artifact_paths(args.ep_id, run_id, "cluster"),
                    )
                except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                    LOGGER.warning("[manifest] Failed to write cluster failed manifest: %s", manifest_exc)
        progress.fail(str(exc))
        raise
    finally:
        progress.close()


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


class ManifestValidationError(Exception):
    """Raised when manifest validation fails with specific context."""

    def __init__(self, manifest_type: str, message: str, path: Path | None = None):
        self.manifest_type = manifest_type
        self.path = path
        super().__init__(f"{manifest_type}: {message}")


def validate_manifest(
    path: Path,
    manifest_type: str,
    required_fields: list[str] | None = None,
    min_rows: int = 1,
) -> tuple[bool, str | None, int]:
    """Validate a JSONL manifest file with detailed error reporting.

    Args:
        path: Path to the manifest file
        manifest_type: Human-readable name (e.g., "tracks.jsonl", "faces.jsonl")
        required_fields: List of field names that must be present in each row
        min_rows: Minimum number of valid rows required (default 1)

    Returns:
        Tuple of (is_valid, error_message, row_count)
        - is_valid: True if validation passed
        - error_message: None if valid, otherwise specific error message
        - row_count: Number of valid rows found (0 if file doesn't exist or is invalid)

    Raises:
        ManifestValidationError: If validation fails (only if raise_on_error=True)
    """
    if not path.exists():
        return False, f"{manifest_type} not found at {path}", 0

    # Check for empty file
    try:
        file_size = path.stat().st_size
        if file_size == 0:
            return (
                False,
                f"{manifest_type} exists but is empty (0 bytes) – "
                "the previous pipeline stage likely produced no valid output",
                0,
            )
    except OSError as e:
        return False, f"Cannot read {manifest_type}: {e}", 0

    # Count rows and validate required fields
    valid_rows = 0
    invalid_rows = 0
    missing_fields_examples: list[str] = []

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    invalid_rows += 1
                    continue

                if not isinstance(obj, dict):
                    invalid_rows += 1
                    continue

                # Check required fields
                if required_fields:
                    missing = [f for f in required_fields if f not in obj]
                    if missing:
                        if len(missing_fields_examples) < 3:
                            missing_fields_examples.append(
                                f"line {line_num}: missing {missing}"
                            )
                        invalid_rows += 1
                        continue

                valid_rows += 1
    except OSError as e:
        return False, f"Error reading {manifest_type}: {e}", 0

    # Build error message based on findings
    if valid_rows == 0:
        if invalid_rows > 0:
            details = ""
            if missing_fields_examples:
                details = f" Examples: {'; '.join(missing_fields_examples)}"
            return (
                False,
                f"{manifest_type} contains {invalid_rows} rows but none have required data.{details}",
                0,
            )
        return (
            False,
            f"{manifest_type} exists but contains no valid data rows – "
            "the previous pipeline stage likely produced no valid output",
            0,
        )

    if valid_rows < min_rows:
        return (
            False,
            f"{manifest_type} has only {valid_rows} valid rows, minimum {min_rows} required",
            valid_rows,
        )

    # Validation passed
    return True, None, valid_rows


def require_manifest(
    path: Path,
    manifest_type: str,
    required_fields: list[str] | None = None,
    min_rows: int = 1,
    hint: str | None = None,
) -> int:
    """Validate manifest and raise ManifestValidationError if invalid.

    Args:
        path: Path to the manifest file
        manifest_type: Human-readable name (e.g., "tracks.jsonl")
        required_fields: List of field names that must be present
        min_rows: Minimum number of valid rows required
        hint: Additional hint to add to error message (e.g., "run detect/track first")

    Returns:
        Number of valid rows found

    Raises:
        ManifestValidationError: If validation fails
    """
    is_valid, error_msg, row_count = validate_manifest(
        path, manifest_type, required_fields, min_rows
    )
    if not is_valid:
        full_msg = error_msg or f"{manifest_type} validation failed"
        if hint:
            full_msg = f"{full_msg}. Hint: {hint}"
        raise ManifestValidationError(manifest_type, full_msg, path)
    return row_count


def _sample_track_uniformly(
    track_samples: List[Dict[str, Any]],
    max_samples: int,
    min_samples: int,
    sample_interval: int,
) -> List[Dict[str, Any]]:
    """Sample track uniformly to limit number of samples per track.

    Args:
        track_samples: All samples for a single track, sorted by frame_idx
        max_samples: Maximum number of samples to keep
        min_samples: Minimum number of samples if track is long enough
        sample_interval: Fallback interval for uniform sampling

    Returns:
        Sampled subset of track_samples
    """
    if not track_samples:
        return []

    interval = max(int(sample_interval or 1), 1)

    # First: enforce a minimum spacing between sampled frames (when requested).
    # This is the "crop interval" control used by Faces Harvest.
    sampled: list[Dict[str, Any]] = []
    if interval <= 1:
        sampled = list(track_samples)
    else:
        last_frame: int | None = None
        for sample in track_samples:
            try:
                frame_idx = int(sample.get("frame_idx") or 0)
            except (TypeError, ValueError):
                frame_idx = 0
            if last_frame is None or (frame_idx - last_frame) >= interval:
                sampled.append(sample)
                last_frame = frame_idx
        # Keep the final frame sample to anchor the end of the track.
        if sampled:
            try:
                last_sample_frame = int(sampled[-1].get("frame_idx") or 0)
            except (TypeError, ValueError):
                last_sample_frame = None
            try:
                final_frame = int(track_samples[-1].get("frame_idx") or 0)
            except (TypeError, ValueError):
                final_frame = None
            if final_frame is not None and last_sample_frame is not None and final_frame != last_sample_frame:
                sampled.append(track_samples[-1])

    # Second: cap volume with uniform downsampling when we still have too many samples.
    if len(sampled) > max_samples:
        uniform: list[Dict[str, Any]] = []
        n = len(sampled)

        # Calculate step size for uniform sampling
        if max_samples >= 2:
            step = (n - 1) / (max_samples - 1)
            indices = [int(round(i * step)) for i in range(max_samples)]
        else:
            indices = [n // 2]  # Take middle frame if max_samples == 1

        # Ensure we don't duplicate indices
        indices = sorted(set(indices))

        for idx in indices:
            if 0 <= idx < n:
                uniform.append(sampled[idx])
        sampled = uniform

    # Ensure we meet min_samples if the track is long enough
    if len(sampled) < min_samples and len(track_samples) >= min_samples:
        # Add more samples uniformly
        additional_needed = min_samples - len(sampled)
        n_full = len(track_samples)
        step = n_full / (min_samples + 1)
        additional_indices = [int(round((i + 1) * step)) for i in range(additional_needed)]
        for idx in additional_indices:
            if 0 <= idx < n_full and track_samples[idx] not in sampled:
                sampled.append(track_samples[idx])

        # Re-sort by frame_idx
        sampled.sort(key=lambda s: s["frame_idx"])

    return sampled[:max_samples]


def _load_track_samples(
    track_path: Path,
    sort_by_frame: bool = False,
    max_samples_per_track: int | None = None,
    min_samples_per_track: int | None = None,
    sample_every_n_frames: int | None = None,
) -> List[Dict[str, Any]]:
    """Load track samples from tracks.jsonl with optional per-track sampling.

    Args:
        track_path: Path to tracks.jsonl
        sort_by_frame: If True, sort samples by frame_idx for batch processing
        max_samples_per_track: Maximum samples per track (None = no limit)
        min_samples_per_track: Minimum samples per track if track is long enough
        sample_every_n_frames: Fallback interval for uniform sampling

    Returns:
        List of sample dicts with track_id, frame_idx, bbox_xyxy, etc.
    """
    # Default sampling parameters
    if max_samples_per_track is None:
        max_samples_per_track = 16
    if min_samples_per_track is None:
        min_samples_per_track = 4
    if sample_every_n_frames is None:
        sample_every_n_frames = 4

    # Group samples by track_id first
    tracks_samples: Dict[int, List[Dict[str, Any]]] = {}

    for row in _iter_jsonl(track_path):
        track_id = int(row.get("track_id", -1))
        bbox_samples = row.get("bboxes_sampled") or []
        if not bbox_samples:
            fallback = {
                "frame_idx": int(row.get("first_frame_idx") or 0),
                "ts": float(row.get("first_ts") or 0.0),
                "bbox_xyxy": row.get("bbox_xyxy") or [0, 0, 10, 10],
            }
            bbox_samples = [fallback]

        track_samples_list = []
        for sample in bbox_samples:
            frame_idx = int(sample.get("frame_idx") or 0)
            ts = float(sample.get("ts") or 0.0)
            bbox = sample.get("bbox_xyxy") or [0, 0, 10, 10]
            if not isinstance(bbox, list) or len(bbox) != 4:
                bbox = [0, 0, 10, 10]
            sample_dict = {
                "track_id": track_id,
                "frame_idx": frame_idx,
                "ts": ts,
                "bbox_xyxy": [float(val) for val in bbox],
            }
            landmarks = sample.get("landmarks")
            if isinstance(landmarks, list) and landmarks:
                sample_dict["landmarks"] = [float(val) for val in landmarks]
            track_samples_list.append(sample_dict)

        # Sort by frame_idx within this track
        track_samples_list.sort(key=lambda s: s["frame_idx"])
        tracks_samples[track_id] = track_samples_list

    # Apply per-track sampling
    samples: List[Dict[str, Any]] = []
    for track_id, track_samples_list in tracks_samples.items():
        sampled = _sample_track_uniformly(
            track_samples_list,
            max_samples_per_track,
            min_samples_per_track,
            sample_every_n_frames,
        )
        # Add track_sample_count to each sample for adaptive quality thresholds
        # Single-face tracks get more permissive quality gates to avoid orphaned clusters
        track_sample_count = len(sampled)
        for s in sampled:
            s["track_sample_count"] = track_sample_count
        samples.extend(sampled)

    if sort_by_frame:
        samples.sort(key=lambda s: s["frame_idx"])

    return samples


def _infer_detector_from_tracks(track_path: Path) -> str | None:
    if not track_path.exists():
        return None
    try:
        with track_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                detector = row.get("detector")
                if detector:
                    return str(detector)
    except FileNotFoundError:
        return None
    return None


def _infer_tracker_from_tracks(track_path: Path) -> str | None:
    if not track_path.exists():
        return None
    try:
        with track_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tracker = row.get("tracker")
                if tracker:
                    return str(tracker).lower()
    except FileNotFoundError:
        return None
    return None


def _cluster_distance_threshold(similarity: float) -> float:
    sim = min(
        max(
            float(similarity if similarity is not None else DEFAULT_CLUSTER_SIMILARITY),
            0.0,
        ),
        0.999,
    )
    distance = 1.0 - sim
    return max(distance, 0.01)


def _cluster_embeddings(matrix: np.ndarray, threshold: float) -> np.ndarray:
    """Cluster embeddings using centroid-based hierarchical clustering.

    Uses scipy's linkage with 'centroid' method, which compares cluster centroids
    rather than averaging all pairwise distances. This is more robust when clusters
    contain noisy embeddings (e.g., from single-frame tracks or extreme poses).

    Args:
        matrix: N x D embedding matrix (each row is a track's prototype embedding)
        threshold: Similarity threshold (0-1). Clusters with centroid similarity
                   above this threshold will be merged.

    Returns:
        Array of cluster labels for each input embedding.
    """
    if matrix.shape[0] == 1:
        return np.array([0], dtype=int)

    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
    normalized = matrix / norms

    # Compute pairwise cosine distances (1 - similarity)
    # For normalized vectors, cosine distance = 1 - dot product
    distances = pdist(normalized, metric='cosine')

    # Hierarchical clustering with centroid linkage
    # 'centroid' method: distance between clusters is the distance between their centroids
    # This is more robust than 'average' when clusters have noisy outliers
    Z = linkage(distances, method='centroid')

    # Convert similarity threshold to distance threshold
    distance_threshold = _cluster_distance_threshold(threshold)

    # Cut the dendrogram at the threshold
    # fcluster returns labels starting from 1, convert to 0-indexed
    labels = fcluster(Z, t=distance_threshold, criterion='distance')

    return labels - 1  # Convert to 0-indexed labels


# NOTE: Duplicate _cosine_similarity removed - using single definition at line ~590
# to prevent function redefinition and ensure validation is always applied.


def _remove_low_similarity_outliers(
    track_groups: Dict[int, List[int]],
    track_embeddings: Dict[int, np.ndarray],
    min_sim: float,
) -> Tuple[Dict[int, List[int]], List[Tuple[int, str]]]:
    """Remove tracks that have low similarity to their identity centroid.

    Returns:
        - Updated track_groups with outliers removed
        - List of (track_id, reason) tuples for outlier tracks
    """
    outliers: List[Tuple[int, str]] = []
    updated_groups: Dict[int, List[int]] = {}

    for label, track_ids in track_groups.items():
        if len(track_ids) < 2:
            # Single-track identities always pass
            updated_groups[label] = track_ids
            continue

        # Compute cluster centroid
        cluster_embeds = [track_embeddings[tid] for tid in track_ids if tid in track_embeddings]
        if not cluster_embeds:
            updated_groups[label] = track_ids
            continue

        centroid = np.mean(cluster_embeds, axis=0)
        norm_centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        # Check similarity of each track to centroid
        kept_tracks = []
        for tid in track_ids:
            if tid not in track_embeddings:
                kept_tracks.append(tid)
                continue

            track_embed = track_embeddings[tid]
            similarity = _cosine_similarity(track_embed, norm_centroid)

            if similarity >= min_sim:
                kept_tracks.append(tid)
            else:
                outliers.append((tid, f"low_identity_similarity_{similarity:.3f}"))
                LOGGER.info(
                    "Track %d removed from cluster %d (similarity %.3f < %.3f)",
                    tid,
                    label,
                    similarity,
                    min_sim,
                )

        if kept_tracks:
            updated_groups[label] = kept_tracks

    return updated_groups, outliers


def _validate_track_embeddings_for_clustering(
    track_ids: List[int],
    embedding_rows: List[np.ndarray],
    track_index: Dict[int, Dict[str, Any]],
    min_internal_sim: float = 0.70,
) -> Tuple[List[int], List[np.ndarray], List[Tuple[int, str]]]:
    """Pre-clustering validation: reject tracks with poor internal consistency.

    Tracks with low internal similarity (intra-track sim) are unreliable and
    can pollute clusters when grouped with others.

    Args:
        track_ids: List of track IDs to validate
        embedding_rows: Corresponding embeddings for each track
        track_index: Track metadata containing internal_sim scores
        min_internal_sim: Minimum required internal similarity (default 0.70)

    Returns:
        - Filtered track_ids (those that passed)
        - Filtered embedding_rows (corresponding embeddings)
        - List of (track_id, reason) for rejected tracks
    """
    valid_track_ids = []
    valid_embeddings = []
    rejected: List[Tuple[int, str]] = []

    for tid, embed in zip(track_ids, embedding_rows):
        track_info = track_index.get(tid, {})
        internal_sim = track_info.get("internal_sim")

        # If no internal_sim available (single face track), allow through
        if internal_sim is None:
            valid_track_ids.append(tid)
            valid_embeddings.append(embed)
            continue

        if internal_sim >= min_internal_sim:
            valid_track_ids.append(tid)
            valid_embeddings.append(embed)
        else:
            rejected.append((tid, f"low_internal_sim_{internal_sim:.3f}"))
            LOGGER.info(
                "Track %d rejected from clustering: internal_sim=%.3f < %.3f",
                tid, internal_sim, min_internal_sim,
            )

    return valid_track_ids, valid_embeddings, rejected


def _split_low_cohesion_clusters(
    track_groups: Dict[int, List[int]],
    track_embeddings: Dict[int, np.ndarray],
    min_cohesion: float = MIN_CLUSTER_COHESION,
) -> Tuple[Dict[int, List[int]], List[Tuple[List[int], float, str]]]:
    """Quality gate: split clusters with cohesion below threshold into singletons.

    Low cohesion indicates the tracks in a cluster are not similar enough
    to each other - they are likely different people incorrectly grouped.

    Args:
        track_groups: Cluster label -> list of track IDs
        track_embeddings: Track ID -> embedding vector
        min_cohesion: Minimum cohesion threshold (default from MIN_CLUSTER_COHESION)

    Returns:
        - Updated track_groups with low-cohesion clusters split
        - List of (original_track_ids, cohesion, reason) for split clusters
    """
    updated_groups: Dict[int, List[int]] = {}
    split_clusters: List[Tuple[List[int], float, str]] = []
    next_label = max(track_groups.keys()) + 1 if track_groups else 0

    for label, track_ids in track_groups.items():
        if len(track_ids) < 2:
            # Single-track clusters always pass
            updated_groups[label] = track_ids
            continue

        # Compute cluster cohesion (mean pairwise similarity)
        cluster_embeds = [track_embeddings[tid] for tid in track_ids if tid in track_embeddings]
        if len(cluster_embeds) < 2:
            updated_groups[label] = track_ids
            continue

        # Compute centroid
        centroid = np.mean(cluster_embeds, axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)

        # Compute cohesion as mean similarity to centroid
        similarities = []
        for embed in cluster_embeds:
            sim = _cosine_similarity(embed, centroid_norm)
            similarities.append(sim)
        cohesion = float(np.mean(similarities)) if similarities else 0.0

        if cohesion >= min_cohesion:
            # Cluster passes quality gate
            updated_groups[label] = track_ids
        else:
            # Split into singleton clusters
            split_clusters.append((track_ids.copy(), cohesion, f"cohesion_{cohesion:.3f}_below_{min_cohesion:.2f}"))
            LOGGER.warning(
                "Splitting cluster %d into %d singletons: cohesion=%.3f < %.3f (tracks: %s)",
                label, len(track_ids), cohesion, min_cohesion, track_ids,
            )
            for tid in track_ids:
                updated_groups[next_label] = [tid]
                next_label += 1

    return updated_groups, split_clusters


def _materialize_identity_thumb(
    thumb_root: Path,
    track_row: dict | None,
    identity_id: str,
    s3_prefixes: Dict[str, str] | None,
) -> tuple[str | None, str | None]:
    if not track_row:
        return None, None
    rel = track_row.get("thumb_rel_path")
    if not rel:
        return None, None
    source = thumb_root / rel
    if not source.exists():
        return None, None
    dest_rel = Path("identities") / identity_id / "rep.jpg"
    dest = thumb_root / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, dest)
    s3_key = None
    if s3_prefixes and s3_prefixes.get("thumbs_identities"):
        s3_key = f"{s3_prefixes['thumbs_identities']}{identity_id}/rep.jpg"
    return dest_rel.as_posix(), s3_key


def _build_identity_clusters(
    faces_by_track: Dict[int, List[Dict[str, Any]]],
    s3_prefixes: Dict[str, str] | None,
) -> List[Dict[str, Any]]:
    track_ids = sorted(track_id for track_id in faces_by_track.keys() if track_id is not None)
    clusters: List[List[int]] = []
    current: List[int] = []
    for track_id in track_ids:
        current.append(track_id)
        if len(current) == 2:
            clusters.append(current)
            current = []
    if current:
        clusters.append(current)
    if not clusters and faces_by_track:
        for track_id in track_ids:
            clusters.append([track_id])

    if not clusters:
        return []

    identities: List[Dict[str, Any]] = []
    for idx, track_group in enumerate(clusters, start=1):
        track_faces: List[Dict[str, Any]] = []
        for track_id in track_group:
            track_faces.extend(faces_by_track.get(track_id, []))
        track_faces.sort(key=lambda face: face.get("ts", 0.0))
        count = len(track_faces)
        rep_face = track_faces[0] if track_faces else None
        identity = {
            "identity_id": f"id_{idx:04d}",
            "label": f"Identity {idx:02d}",
            "track_ids": track_group,
            "count": count,
            "samples": [face.get("face_id") for face in track_faces[:3] if face.get("face_id")],
        }
        rep_payload = _rep_payload(rep_face, s3_prefixes)
        if rep_payload:
            identity["rep"] = rep_payload
        identities.append(identity)
    return identities


def _rep_payload(face: Dict[str, Any] | None, s3_prefixes: Dict[str, str] | None) -> Dict[str, Any] | None:
    if not face:
        return None
    rep: Dict[str, Any] = {
        "track_id": face.get("track_id"),
        "frame_idx": face.get("frame_idx"),
        "ts": face.get("ts"),
    }
    if face.get("crop_rel_path"):
        rep["crop_rel_path"] = face["crop_rel_path"]
    s3_key = face.get("crop_s3_key")
    if not s3_key and s3_prefixes and s3_prefixes.get("crops"):
        track_id = face.get("track_id")
        frame_idx = face.get("frame_idx")
        if track_id is not None and frame_idx is not None:
            # Default to PNG (matches FrameExporter default use_png=True)
            s3_key = f"{s3_prefixes['crops']}track_{int(track_id):04d}/frame_{int(frame_idx):06d}.png"
    if s3_key:
        rep["s3_key"] = s3_key
    return rep


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
