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
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import logging

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.api.services.storage import (
    EpisodeContext,
    StorageService,
    artifact_prefixes,
    episode_context_from_id,
)
from py_screenalytics.artifacts import ensure_dirs, get_path
from tools._img_utils import safe_crop, safe_imwrite, to_u8_bgr
from tools.debug_thumbs import init_debug_logger, debug_thumbs_enabled, NullLogger, JsonlLogger

PIPELINE_VERSION = os.environ.get("SCREENALYTICS_PIPELINE_VERSION", "2025-11-11")
APP_VERSION = os.environ.get("SCREENALYTICS_APP_VERSION", PIPELINE_VERSION)
TRACKER_CONFIG = os.environ.get("SCREENALYTICS_TRACKER_CONFIG", "bytetrack.yaml")
TRACKER_NAME = Path(TRACKER_CONFIG).stem if TRACKER_CONFIG else "bytetrack"
PROGRESS_FRAME_STEP = int(os.environ.get("SCREENALYTICS_PROGRESS_FRAME_STEP", 25))
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
MIN_FACE_AREA = 20.0
FACE_RATIO_BOUNDS = (0.5, 2.0)
RETINAFACE_SCORE_THRESHOLD = 0.5
RETINAFACE_NMS = 0.45

RUN_MARKERS_SUBDIR = "runs"
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

BYTE_TRACK_MIN_BOX_AREA_DEFAULT = max(
    _env_float("SCREENALYTICS_MIN_BOX_AREA", _env_float("BYTE_TRACK_MIN_BOX_AREA", 20.0)),
    0.0,
)
DEFAULT_GMC_METHOD = os.environ.get("SCREENALYTICS_GMC_METHOD", "sparseOptFlow")
DEFAULT_REID_MODEL = os.environ.get("SCREENALYTICS_REID_MODEL", "yolov8n-cls.pt")
DEFAULT_REID_ENABLED = os.environ.get("SCREENALYTICS_REID_ENABLED", "1").lower() in {"1", "true", "yes"}
RETINAFACE_HELP = "RetinaFace weights missing or could not initialize. See README 'Models' or run scripts/fetch_models.py."
ARC_FACE_HELP = "ArcFace weights missing or could not initialize. See README 'Models' or run scripts/fetch_models.py."
GATE_APPEAR_T_HARD_DEFAULT = float(os.environ.get("TRACK_GATE_APPEAR_HARD", "0.60"))
GATE_APPEAR_T_SOFT_DEFAULT = float(os.environ.get("TRACK_GATE_APPEAR_SOFT", "0.70"))
GATE_APPEAR_STREAK_DEFAULT = max(int(os.environ.get("TRACK_GATE_APPEAR_STREAK", "2")), 1)
GATE_IOU_THRESHOLD_DEFAULT = float(os.environ.get("TRACK_GATE_IOU", "0.35"))
GATE_PROTO_MOMENTUM_DEFAULT = min(max(float(os.environ.get("TRACK_GATE_PROTO_MOM", "0.90")), 0.0), 1.0)
GATE_EMB_EVERY_DEFAULT = max(int(os.environ.get("TRACK_GATE_EMB_EVERY", "0")), 0)
TRACK_BUFFER_BASE_DEFAULT = max(
    _env_int("SCREENALYTICS_TRACK_BUFFER", _env_int("BYTE_TRACK_BUFFER", 30)),
    1,
)
BYTE_TRACK_MATCH_THRESH_DEFAULT = _env_float("BYTE_TRACK_MATCH_THRESH", 0.75)
TRACK_HIGH_THRESH_DEFAULT = _env_float(
    "SCREENALYTICS_TRACK_HIGH_THRESH",
    _env_float("BYTE_TRACK_HIGH_THRESH", 0.5),
)
TRACK_NEW_THRESH_DEFAULT = _env_float(
    "SCREENALYTICS_NEW_TRACK_THRESH",
    _env_float("BYTE_TRACK_NEW_TRACK_THRESH", 0.5),
)
TRACK_MAX_GAP_SEC = float(os.environ.get("TRACK_MAX_GAP_SEC", "0.5"))
TRACK_PROTO_MAX_SAMPLES = max(int(os.environ.get("TRACK_PROTO_MAX_SAMPLES", "6")), 2)
TRACK_PROTO_SIM_DELTA = float(os.environ.get("TRACK_PROTO_SIM_DELTA", "0.08"))
TRACK_PROTO_SIM_MIN = float(os.environ.get("TRACK_PROTO_SIM_MIN", "0.6"))
DEFAULT_CLUSTER_SIMILARITY = float(os.environ.get("SCREENALYTICS_CLUSTER_SIM", "0.7"))
MIN_IDENTITY_SIMILARITY = float(os.environ.get("SCREENALYTICS_MIN_IDENTITY_SIM", "0.50"))
FACE_MIN_CONFIDENCE = float(os.environ.get("FACES_MIN_CONF", "0.60"))
FACE_MIN_BLUR = float(os.environ.get("FACES_MIN_BLUR", "35.0"))
FACE_MIN_STD = float(os.environ.get("FACES_MIN_STD", "1.0"))
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
SCENE_WARMUP_DETS_DEFAULT = max(_env_int("SCENE_WARMUP_DETS", 3), 0)


def _normalize_device_label(device: str | None) -> str:
    normalized = (device or "cpu").lower()
    if normalized in {"0", "cuda", "gpu"}:
        return "cuda"
    return normalized


def _onnx_providers_for(device: str | None) -> tuple[list[str], str]:
    normalized = (device or "auto").lower()
    providers: list[str] = ["CPUExecutionProvider"]
    resolved = "cpu"
    if normalized in {"cuda", "0", "gpu", "auto"}:
        try:
            import onnxruntime as ort  # type: ignore

            available = ort.get_available_providers()
        except Exception:
            available = []
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            resolved = "cuda"
            return providers, resolved
        if normalized in {"cuda", "0", "gpu"}:
            LOGGER.warning("CUDA requested for RetinaFace/ArcFace but CUDAExecutionProvider unavailable; falling back to CPU")
    if normalized in {"mps", "metal", "apple"}:
        return ["CPUExecutionProvider"], "cpu"
    return providers, resolved


def _init_retinaface(model_name: str, device: str, score_thresh: float = RETINAFACE_SCORE_THRESHOLD) -> tuple[Any, str]:
    try:
        from insightface.model_zoo import get_model  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("insightface is required for RetinaFace detection") from exc

    providers, resolved = _onnx_providers_for(device)
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
    if RETINAFACE_DET_SIZE:
        prepare_kwargs["input_size"] = RETINAFACE_DET_SIZE
    try:
        model.prepare(**prepare_kwargs)
    except TypeError:
        prepare_kwargs.pop("input_size", None)
        model.prepare(**prepare_kwargs)
    return model, resolved


def _init_arcface(model_name: str, device: str):
    try:
        from insightface.model_zoo import get_model  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("insightface is required for ArcFace embeddings") from exc

    providers, resolved = _onnx_providers_for(device)
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
        detector.prepare(
            ctx_id=ctx_id,
            det_size=RETINAFACE_DET_SIZE or (640, 640),
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
    if a is None or b is None:
        return None
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
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
    match = re.match(r'^(?P<show>.+)-s\d{2}e\d{2}$', ep_id, re.IGNORECASE)
    if match:
        return match.group('show').lower()
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
    min_sim: float = SEED_BOOST_MIN_SIM
) -> Optional[Tuple[str, float]]:
    """Find the best matching seed for an embedding.
    
    Returns (cast_id, similarity) if match found above threshold, else None.
    """
    if not seeds or embedding is None:
        return None
    
    best_sim = -1.0
    best_cast_id = None
    
    for seed in seeds:
        seed_emb = np.array(seed.get('embedding', []), dtype=np.float32)
        if seed_emb.size == 0:
            continue
        
        # Cosine similarity
        sim = float(np.dot(embedding, seed_emb) / (np.linalg.norm(embedding) * np.linalg.norm(seed_emb) + 1e-12))
        
        if sim > best_sim:
            best_sim = sim
            best_cast_id = seed.get('cast_id')
    
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


def _tracker_inputs_from_samples(detections: list[DetectionSample]) -> _TrackerDetections:
    if detections:
        boxes = np.vstack([sample.bbox for sample in detections]).astype(np.float32)
        scores = np.asarray([sample.conf for sample in detections], dtype=np.float32)
        classes = np.asarray([sample.class_idx for sample in detections], dtype=np.float32)
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
            raise RuntimeError(
                "StrongSORT tracker unavailable; ensure ultralytics>=8.2.70 is installed."
            ) from exc

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
    def __init__(self, device: str, score_thresh: float = RETINAFACE_SCORE_THRESHOLD) -> None:
        self.device = device
        self.score_thresh = max(min(float(score_thresh or RETINAFACE_SCORE_THRESHOLD), 1.0), 0.0)
        self.min_area = MIN_FACE_AREA
        self._model = None
        self._resolved_device: Optional[str] = None

    def _lazy_model(self):
        if self._model is not None:
            return self._model
        try:
            model, resolved = _init_retinaface(self.model_name, self.device, self.score_thresh)
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


def _build_face_detector(detector: str, device: str, score_thresh: float = RETINAFACE_SCORE_THRESHOLD):
    return RetinaFaceDetectorBackend(device, score_thresh=score_thresh)


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
        track_high_thresh=getattr(args, "track_high_thresh", TRACK_HIGH_THRESH_DEFAULT)
        or TRACK_HIGH_THRESH_DEFAULT,
        new_track_thresh=getattr(args, "new_track_thresh", TRACK_NEW_THRESH_DEFAULT)
        or TRACK_NEW_THRESH_DEFAULT,
        match_thresh=BYTE_TRACK_MATCH_THRESH_DEFAULT,
        track_low_thresh=0.1,
        track_buffer_base=getattr(args, "track_buffer", TRACK_BUFFER_BASE_DEFAULT)
        or TRACK_BUFFER_BASE_DEFAULT,
        min_box_area=getattr(args, "min_box_area", BYTE_TRACK_MIN_BOX_AREA_DEFAULT)
        or BYTE_TRACK_MIN_BOX_AREA_DEFAULT,
    )


class ArcFaceEmbedder:
    def __init__(self, device: str) -> None:
        self.device = device
        self._model = None
        self._resolved_device: Optional[str] = None

    def _lazy_model(self):
        if self._model is not None:
            return self._model
        try:
            model, resolved = _init_arcface(ARC_FACE_MODEL_NAME, self.device)
        except Exception as exc:
            raise RuntimeError(f"ArcFace init failed: {exc}. Install insightface + models or run scripts/fetch_models.py.") from exc
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
        embeddings: list[np.ndarray] = []
        for crop in crops:
            if crop is None or crop.size == 0:
                embeddings.append(np.zeros(512, dtype=np.float32))
                continue
            resized = _resize_for_arcface(crop)
            feat = model.get_feat(resized)
            vec = np.asarray(feat, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return np.vstack(embeddings)


def _resize_for_arcface(image):
    import cv2  # type: ignore

    target = (112, 112)
    resized = cv2.resize(image, target)
    return resized


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


def _prepare_face_crop(
    image,
    bbox: list[float],
    landmarks: list[float] | None,
    margin: float = 0.15,
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
            self._mapping[tracker_track_id] = {"export_id": export_id, "last_frame": frame_idx}
        else:
            if mapping is None:
                export_id = tracker_track_id
                self._active_exports.add(export_id)
                self._mapping[tracker_track_id] = {"export_id": export_id, "last_frame": frame_idx}
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
            {"track_id": track.track_id, "frame_count": track.frame_count}
            for track in longest
            if track.frame_count > 0
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


def sanitize_xyxy(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int] | None:
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
    def __init__(self, ep_id: str, size: int = 256, jpeg_quality: int = 85) -> None:
        self.ep_id = ep_id
        self.size = size
        self.jpeg_quality = max(1, min(int(jpeg_quality or 85), 100))
        self.root_dir = get_path(ep_id, "frames_root") / "thumbs"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._stat_samples = 0
        self._stat_limit = 10
        try:
            import cv2  # type: ignore

            self._cv2 = cv2
        except ImportError:
            self._cv2 = None
        self._last_thumb_meta: Dict[str, Any] = {}

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
                LOGGER.debug("Skipping thumb track=%s frame=%s reason=%s", track_id, frame_idx, err)
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
            LOGGER.info("thumb stats track=%s frame=%s min=%.3f max=%.3f mean=%.3f", track_id, frame_idx, mn, mx, mean)
            if mx - mn < 1e-6:
                LOGGER.warning("Nearly constant thumb track=%s frame=%s", track_id, frame_idx)
            self._stat_samples += 1
        rel_path = Path(f"track_{track_id:04d}/thumb_{frame_idx:06d}.jpg")
        abs_path = self.root_dir / rel_path
        ok, reason = safe_imwrite(abs_path, thumb, self.jpeg_quality)
        if not ok:
            LOGGER.warning("Failed to write thumb %s: %s", abs_path, reason)
            return None, None
        self._last_thumb_meta = {
            "source_shape": tuple(int(x) for x in source.shape[:2]),
            "source_kind": source_kind,
        }
        return rel_path.as_posix(), abs_path

    def _letterbox(self, crop):
        if self._cv2 is None:
            return np.zeros((self.size, self.size, 3), dtype=np.uint8)
        if crop.size == 0:
            crop = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        h, w = crop.shape[:2]
        scale = min(self.size / max(w, 1), self.size / max(h, 1))
        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)
        resized = self._cv2.resize(crop, (new_w, new_h))
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        top = (self.size - new_h) // 2
        left = (self.size - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized
        return canvas


def _faces_embed_path(ep_id: str) -> Path:
    embed_dir = DATA_ROOT / "embeds" / ep_id
    embed_dir.mkdir(parents=True, exist_ok=True)
    return embed_dir / "faces.npy"
class ProgressEmitter:
    """Emit structured progress to stdout + optional file for SSE/polling."""

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
    ) -> None:
        import uuid
        self.ep_id = ep_id
        self.run_id = run_id or str(uuid.uuid4())
        self.path = Path(file_path).expanduser() if file_path else None
        if self.path:
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
        return (frames_done - self._last_frames) >= self._frame_interval

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
            "fps_detected": round(float(self.fps_detected), 3) if self.fps_detected else None,
            "fps_requested": round(float(self.fps_requested), 3) if self.fps_requested else None,
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
            LOGGER.info(
                "[job=%s run=%s phase=%s step=%s frames=%s/%s vt=%.1f/%.1f fps=%.2f]",
                self.ep_id, run_id_short, phase, step, frames, total, vt, vtotal, fps or 0.0
            )
        else:
            LOGGER.info(
                "[job=%s run=%s phase=%s step=%s frames=%s/%s fps=%.2f]",
                self.ep_id, run_id_short, phase, step, frames, total, fps or 0.0
            )

        if self.path:
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
        self.emit(self._last_frames, phase="error", error=error, force=True, tracker=self._tracker)

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


def _write_run_marker(ep_id: str, phase: str, payload: Dict[str, Any]) -> None:
    manifests_dir = get_path(ep_id, "detections").parent
    run_dir = manifests_dir / RUN_MARKERS_SUBDIR
    run_dir.mkdir(parents=True, exist_ok=True)
    marker_path = run_dir / f"{phase}.json"
    marker_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class CropQualityThresholdExceeded(RuntimeError):
    """Raised when crop exports fail at an unacceptable rate."""


class FrameExporter:
    """Handles optional frame + crop JPEG exports for S3 sync."""

    def __init__(
        self,
        ep_id: str,
        *,
        save_frames: bool,
        save_crops: bool,
        jpeg_quality: int,
        debug_logger: JsonlLogger | NullLogger | None = None,
    ) -> None:
        self.ep_id = ep_id
        self.save_frames = save_frames
        self.save_crops = save_crops
        self.jpeg_quality = max(1, min(int(jpeg_quality or 85), 100))
        self.root_dir = get_path(ep_id, "frames_root")
        self.frames_dir = self.root_dir / "frames"
        self.crops_dir = self.root_dir / "crops"
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

    def _log_image_stats(self, kind: str, path: Path, image) -> None:
        if self._stat_samples >= self._stat_limit:
            return
        mn, mx, mean = _image_stats(image)
        LOGGER.info("%s stats %s min=%.3f max=%.3f mean=%.3f", kind, path, mn, mx, mean)
        if mx - mn < 1e-6:
            LOGGER.warning("Nearly constant %s %s mn=%.6f mx=%.6f mean=%.6f", kind, path, mn, mx, mean)
        self._stat_samples += 1

    def export(self, frame_idx: int, image, crops: List[Tuple[int, List[float]]], ts: float | None = None) -> None:
        if not (self.save_frames or self.save_crops):
            return
        if self.save_frames:
            frame_path = self.frames_dir / f"frame_{frame_idx:06d}.jpg"
            try:
                self._log_image_stats("frame", frame_path, image)
                save_jpeg(frame_path, image, quality=self.jpeg_quality, color="bgr")
                self.frames_written += 1
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
        return f"track_{track_id:04d}/frame_{frame_idx:06d}.jpg"

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

        ok, save_err = safe_imwrite(crop_path, crop, self.jpeg_quality)
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
                    "file_size": crop_path.stat().st_size if ok and crop_path.exists() else None,
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
            LOGGER.info("Skipping video copy; destination matches source (size=%d)", src_stat.st_size)
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
 

def _storage_context(ep_id: str) -> tuple[StorageService | None, EpisodeContext | None, Dict[str, str] | None]:
    storage_backend = os.environ.get("STORAGE_BACKEND", "local").lower()
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


def _sync_artifacts_to_s3(
    ep_id: str,
    storage: StorageService | None,
    ep_ctx: EpisodeContext | None,
    exporter: FrameExporter | None,
    thumb_dir: Path | None = None,
) -> Dict[str, int]:
    """Sync artifacts to S3 storage.

    TODO(perf): Make uploads asynchronous using ThreadPoolExecutor to avoid blocking.
    Current implementation uploads synchronously which blocks the worker. However, this
    is called AFTER completion events are emitted (see line 3336), so users already see
    "done" status. Making async would require: (1) thread pool management, (2) error
    handling from background threads, (3) ensuring all uploads complete before exit, (4)
    progress reporting across threads. Benefit is marginal since user must wait anyway.
    """
    stats = {"manifests": 0, "frames": 0, "crops": 0, "thumbs_tracks": 0, "thumbs_identities": 0}
    if storage is None or ep_ctx is None or not storage.s3_enabled() or not storage.write_enabled:
        return stats
    prefixes = artifact_prefixes(ep_ctx)
    manifests_dir = get_path(ep_id, "detections").parent
    stats["manifests"] = storage.upload_dir(manifests_dir, prefixes["manifests"])
    frames_root = get_path(ep_id, "frames_root")
    frames_dir = frames_root / "frames"
    crops_dir = frames_root / "crops"
    # Only upload frames/crops if they were actually produced in this run
    # The elif branches that uploaded stale artifacts have been removed to prevent
    # publishing old frames/crops from prior runs when exports are disabled
    if exporter and exporter.save_frames and exporter.frames_dir.exists():
        stats["frames"] = storage.upload_dir(exporter.frames_dir, prefixes["frames"])
    if exporter and exporter.save_crops and exporter.crops_dir.exists():
        stats["crops"] = storage.upload_dir(exporter.crops_dir, prefixes["crops"])
    if thumb_dir is not None and thumb_dir.exists():
        identities_dir = thumb_dir / "identities"
        stats["thumbs_tracks"] = storage.upload_dir(
            thumb_dir,
            prefixes.get("thumbs_tracks", ""),
            skip_subdirs=("identities",),
        )
        if identities_dir.exists():
            stats["thumbs_identities"] = storage.upload_dir(
                identities_dir,
                prefixes.get("thumbs_identities", ""),
            )
    return stats


def _report_s3_upload(
    progress: ProgressEmitter | None,
    stats: Dict[str, int],
    *,
    device: str | None,
    detector: str | None,
    tracker: str | None,
    resolved_device: str | None,
) -> None:
    if not progress:
        return
    if not any(stats.values()):
        return
    frames = progress.target_frames or 0
    progress.emit(
        frames,
        phase="mirror_s3",
        device=device,
        summary={"s3_uploads": stats},
        detector=detector,
        tracker=tracker,
        resolved_device=resolved_device,
        force=True,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection + tracking locally.")
    parser.add_argument("--ep-id", required=True, help="Episode identifier")
    parser.add_argument("--video", help="Path to source video (required for detect/track runs)")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for detection (default: 1 = every frame)")
    parser.add_argument(
        "--fps",
        type=float,
        help="Optional target FPS for downsampling before detection",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Execution device override (auto→CUDA/MPS/CPU)",
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
        default=30,
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
    parser.add_argument("--save-frames", action="store_true", help="Save sampled frame JPGs under data/frames/{ep_id}")
    parser.add_argument("--save-crops", action="store_true", help="Save per-track crops (requires --save-frames or track IDs)")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality for frame exports (1-100)")
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
    gate_group = parser.add_argument_group("Appearance gate")
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
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(args, "det_thresh"):
        args.det_thresh = _normalize_det_thresh(getattr(args, "det_thresh", RETINAFACE_SCORE_THRESHOLD))
    args.scene_detector = _normalize_scene_detector_choice(getattr(args, "scene_detector", None))
    args.scene_threshold = max(float(getattr(args, "scene_threshold", SCENE_THRESHOLD_DEFAULT)), 0.0)
    args.scene_min_len = max(int(getattr(args, "scene_min_len", SCENE_MIN_LEN_DEFAULT)), 1)
    args.scene_warmup_dets = max(int(getattr(args, "scene_warmup_dets", SCENE_WARMUP_DETS_DEFAULT)), 0)
    args.track_high_thresh = min(
        max(float(getattr(args, "track_high_thresh", TRACK_HIGH_THRESH_DEFAULT) or TRACK_HIGH_THRESH_DEFAULT), 0.0),
        1.0,
    )
    args.new_track_thresh = min(
        max(float(getattr(args, "new_track_thresh", TRACK_NEW_THRESH_DEFAULT) or TRACK_NEW_THRESH_DEFAULT), 0.0),
        1.0,
    )
    args.track_buffer = max(int(getattr(args, "track_buffer", TRACK_BUFFER_BASE_DEFAULT) or TRACK_BUFFER_BASE_DEFAULT), 1)
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
            LOGGER.info("Track sampling limited to the first %s detections per track.", TRACK_SAMPLE_LIMIT)
    data_root = (
        Path(args.out_root).expanduser()
        if args.out_root
        else Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
    )
    os.environ["SCREENALYTICS_DATA_ROOT"] = str(data_root)
    ensure_dirs(args.ep_id)
    storage, ep_ctx, s3_prefixes = _storage_context(args.ep_id)

    phase_flags = [flag for flag in (args.faces_embed, args.cluster) if flag]
    if len(phase_flags) > 1:
        raise ValueError("Specify at most one of --faces-embed/--cluster per run")

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
    last_cut = -10**9
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
    return [start.get_frames() for (start, _end) in scenes]


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
    frames_goal = None
    if total_frames and total_frames > 0:
        frames_goal = int(total_frames)
    elif progress and progress.target_frames:
        frames_goal = progress.target_frames
    video_clock_fps = video_fps if video_fps and video_fps > 0 else (source_fps if source_fps > 0 else None)
    frame_exporter: FrameExporter | None = None

    def _progress_value(frame_index: int, *, include_current: bool = False, step: str | None = None) -> tuple[int, Dict[str, Any]]:
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
    detector_choice = _normalize_detector_choice(getattr(args, "detector", None))
    tracker_choice = _normalize_tracker_choice(getattr(args, "tracker", None))
    args.detector = detector_choice
    args.tracker = tracker_choice
    det_thresh = _normalize_det_thresh(getattr(args, "det_thresh", RETINAFACE_SCORE_THRESHOLD))
    args.det_thresh = det_thresh
    detector_backend = _build_face_detector(detector_choice, device, det_thresh)
    detector_backend.ensure_ready()
    detector_device = getattr(detector_backend, "resolved_device", device)
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
    if tracker_choice == "bytetrack":
        gate_config = _gate_config_from_args(args, frame_stride)
        appearance_gate = AppearanceGate(gate_config)
        gate_embed_stride = gate_config.emb_every or frame_stride
        try:
            gate_embedder = ArcFaceEmbedder(device)
            gate_embedder.ensure_ready()
        except Exception as exc:
            raise RuntimeError(
                "ArcFace gate embedder is required for ByteTrack gating. "
                "Install the ArcFace weights via scripts/fetch_models.py or rerun with --tracker strongsort."
            ) from exc
    scene_detector_choice = _normalize_scene_detector_choice(getattr(args, "scene_detector", None))
    args.scene_detector = scene_detector_choice
    scene_threshold = max(float(getattr(args, "scene_threshold", SCENE_THRESHOLD_DEFAULT)), 0.0)
    scene_min_len = max(int(getattr(args, "scene_min_len", SCENE_MIN_LEN_DEFAULT)), 1)
    scene_warmup = max(int(getattr(args, "scene_warmup_dets", SCENE_WARMUP_DETS_DEFAULT)), 0)
    scene_cuts: list[int] = []
    if scene_detector_choice != "off":
        scene_cuts = detect_scene_cuts(
            str(video_dest),
            detector=scene_detector_choice,
            thr=scene_threshold,
            min_len=scene_min_len,
            progress=progress,
        )
    scene_summary = {"count": len(scene_cuts), "indices": scene_cuts, "detector": scene_detector_choice}
    cut_ix = 0
    next_cut = scene_cuts[cut_ix] if scene_cuts else None
    frames_since_cut = 10**9
    max_gap_frames = _resolved_max_gap(args.max_gap, analyzed_fps)
    if max_gap_frames != max(1, int(args.max_gap)):
        LOGGER.info(
            "Track max_gap capped to %s frames (configured=%s analyzed_fps=%.2f)",
            max_gap_frames,
            args.max_gap,
            analyzed_fps or 0.0,
        )
    recorder = TrackRecorder(max_gap=max_gap_frames, remap_ids=True)
    det_path = get_path(args.ep_id, "detections")
    det_path.parent.mkdir(parents=True, exist_ok=True)
    track_path = get_path(args.ep_id, "tracks")
    det_count = 0
    frames_sampled = 0
    frame_idx = 0
    last_diag_stats: Dict[str, Any] | None = None

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

    try:
        with det_path.open("w", encoding="utf-8") as det_handle:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if next_cut is not None and frame_idx >= next_cut:
                    reset_tracker = getattr(tracker_adapter, "reset", None)
                    if callable(reset_tracker):
                        reset_tracker()
                    if appearance_gate:
                        appearance_gate.reset_all()
                    recorder.on_cut(frame_idx)
                    frames_since_cut = 0
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
                force_detect = frames_since_cut < scene_warmup
                should_sample = frame_idx % frame_stride == 0
                if not (should_sample or force_detect):
                    frame_idx += 1
                    frames_since_cut += 1
                    continue

                frames_sampled += 1
                detect_frames, detect_meta = _progress_value(frame_idx, include_current=True)
                ts = frame_idx / ts_fps if ts_fps else 0.0

                try:
                    detections = detector_backend.detect(frame)
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
                for det_sample in face_detections:
                    _record_detection_conf(float(det_sample.conf))
                tracked_objects = tracker_adapter.update(face_detections, frame_idx, frame)
                diag_stats = _diagnostic_stats(len(face_detections), len(tracked_objects))
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
                            len(face_detections),
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
                            len(face_detections),
                            len(tracked_objects),
                            frame_stride,
                        )

                crop_records: list[tuple[int, list[float]]] = []
                gate_embeddings: dict[int, np.ndarray | None] = {}
                should_embed_gate = False
                if appearance_gate:
                    should_embed_gate = True if frames_since_cut < scene_warmup else False
                    stride_for_gate = gate_embed_stride or frame_stride
                    if stride_for_gate > 1 and not should_embed_gate:
                        should_embed_gate = frame_idx % stride_for_gate == 0
                    else:
                        should_embed_gate = True
                    if should_embed_gate and gate_embedder and tracked_objects:
                        embed_inputs: list[np.ndarray] = []
                        embed_track_ids: list[int] = []
                        for obj in tracked_objects:
                            landmarks_list = None
                            if obj.landmarks is not None:
                                landmarks_list = (
                                    obj.landmarks.tolist() if isinstance(obj.landmarks, np.ndarray) else obj.landmarks
                                )
                            crop, crop_err = _prepare_face_crop(
                                frame,
                                obj.bbox.tolist(),
                                landmarks_list,
                                margin=0.2,
                            )
                            if crop is None:
                                if crop_err:
                                    LOGGER.debug("Gate crop failed for track %s: %s", obj.track_id, crop_err)
                                continue
                            aligned = _resize_for_arcface(crop)
                            embed_inputs.append(aligned)
                            embed_track_ids.append(obj.track_id)
                        if embed_inputs:
                            encoded = gate_embedder.encode(embed_inputs)
                            for idx, tid in enumerate(embed_track_ids):
                                gate_embeddings[tid] = encoded[idx]
                    for obj in tracked_objects:
                        gate_embeddings.setdefault(obj.track_id, None)
                    # TODO(perf): Persist gate_embeddings to reuse in faces embed stage.
                    # Would require: (1) extending tracks.jsonl schema to include gate_embedding,
                    # (2) saving embeddings in TrackRecorder, (3) loading in faces embed, (4)
                    # matching gate embedding to appropriate face sample per track. Requires
                    # invasive schema changes and careful handling of missing/mismatched cases.

                if not face_detections:
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
                        landmarks = obj.landmarks.tolist() if isinstance(obj.landmarks, np.ndarray) else obj.landmarks
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
                        confidence=float(obj.conf) if obj.conf is not None else None,
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
                        "fps": round(float(analyzed_fps), 4) if analyzed_fps else None,
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
            save_frames=save_frames,
            save_crops=save_crops,
            jpeg_quality=jpeg_quality,
            debug_logger=None,
        )
        if (save_frames or save_crops)
        else None
    )

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

        manifests_dir = get_path(args.ep_id, "detections").parent

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

        s3_stats = _sync_artifacts_to_s3(args.ep_id, storage, ep_ctx, frame_exporter)
        _report_s3_upload(
            progress,
            s3_stats,
            device=pipeline_device,
            detector=detector_choice,
            tracker=tracker_choice,
            resolved_device=detector_device,
        )
        track_ratio = round(track_count / det_count, 3) if det_count > 0 else 0.0
        summary: Dict[str, Any] = {
            "stage": "detect_track",
            "ep_id": args.ep_id,
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
            "resolved_device": detector_device,
            "analyzed_fps": analyzed_fps,
            "detector": detector_choice,
            "tracker": tracker_choice,
            "metrics": track_metrics,
            "artifacts": {
                "local": {
                    "detections": str(get_path(args.ep_id, "detections")),
                    "tracks": str(get_path(args.ep_id, "tracks")),
                    "manifests_dir": str(manifests_dir),
                    "frames_dir": str(frame_exporter.frames_dir) if frame_exporter and frame_exporter.save_frames else None,
                    "crops_dir": str(frame_exporter.crops_dir) if frame_exporter and frame_exporter.save_crops else None,
                },
                "s3_prefixes": s3_prefixes,
                "s3_uploads": s3_stats,
            },
        }
        if detect_track_stats:
            summary["detect_track_stats"] = detect_track_stats
        if tracker_config_summary:
            summary["tracker_config"] = tracker_config_summary
        scene_summary = scene_summary or {"count": 0}
        metrics_path = manifests_dir / "track_metrics.json"
        metrics_payload = {
            "ep_id": args.ep_id,
            "generated_at": _utcnow_iso(),
            "metrics": track_metrics,
            "scene_cuts": scene_summary,
        }
        try:
            metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
            summary["artifacts"]["local"]["track_metrics"] = str(metrics_path)
        except OSError as exc:  # pragma: no cover - best effort diagnostics
            LOGGER.warning("Failed to write track metrics for %s: %s", args.ep_id, exc)
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
            round((scene_cuts_payload["count"] / max(frames_for_scene_rate, 1)) * 1000.0, 3)
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
        return summary
    except Exception as exc:
        progress.fail(str(exc))
        raise
    finally:
        progress.close()


def _run_faces_embed_stage(
    args: argparse.Namespace,
    storage: StorageService | None,
    ep_ctx: EpisodeContext | None,
    s3_prefixes: Dict[str, str] | None,
) -> Dict[str, Any]:
    track_path = get_path(args.ep_id, "tracks")
    if not track_path.exists():
        raise FileNotFoundError("tracks.jsonl not found; run detect/track first")
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
        raise RuntimeError("No track samples available for faces embedding")

    faces_total = len(samples)
    progress = ProgressEmitter(
        args.ep_id,
        args.progress_file,
        frames_total=faces_total,
        secs_total=None,
        stride=1,
        fps_detected=None,
        fps_requested=None,
    )
    device = pick_device(args.device)
    save_frames = bool(args.save_frames)
    save_crops = bool(args.save_crops)
    jpeg_quality = max(1, min(int(args.jpeg_quality or 85), 100))
    frames_root = get_path(args.ep_id, "frames_root")
    debug_logger_obj: JsonlLogger | NullLogger | None = None
    if save_crops and debug_thumbs_enabled():
        debug_logger_obj = init_debug_logger(args.ep_id, frames_root)
    exporter = (
        FrameExporter(
            args.ep_id,
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
    thumb_writer = ThumbWriter(args.ep_id, size=int(getattr(args, "thumb_size", 256)))
    detector_choice = _infer_detector_from_tracks(track_path) or DEFAULT_DETECTOR
    tracker_choice = _infer_tracker_from_tracks(track_path) or DEFAULT_TRACKER
    embedder = ArcFaceEmbedder(device)
    embedder.ensure_ready()
    embed_device = embedder.resolved_device
    embedding_model_name = ARC_FACE_MODEL_NAME

    manifests_dir = get_path(args.ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
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
    try:
        progress.emit(
            0,
            phase="faces_embed",
            device=device,
            detector=detector_choice,
            tracker=tracker_choice,
            resolved_device=embed_device,
            force=True,
            extra=_phase_meta(),
        )
        rows: List[Dict[str, Any]] = []
        for sample in samples:
            crop_rel_path = None
            crop_s3_key = None
            thumb_rel_path = None
            thumb_s3_key = None
            embedding_vec: np.ndarray | None = None
            raw_conf = sample.get("conf")
            if raw_conf is None:
                raw_conf = sample.get("confidence")
            conf = float(raw_conf) if raw_conf is not None else 1.0
            quality = max(min(conf, 1.0), 0.0)
            bbox = sample["bbox_xyxy"]
            track_id = sample["track_id"]
            frame_idx = sample["frame_idx"]
            ts_val = round(float(sample["ts"]), 4)
            landmarks = sample.get("landmarks")

            if not video_path.exists():
                raise FileNotFoundError("Local video not found for crop export")
            if frame_decoder is None:
                frame_decoder = FrameDecoder(video_path)

            # Wrap decode in try/except to handle corrupted frames gracefully
            image = None
            try:
                image = frame_decoder.read(frame_idx)
                frame_std = float(np.std(image)) if image is not None else 0.0
                if image is None or frame_std < 1.0:
                    LOGGER.warning(
                        "Low-variance frame %s std=%.4f; retrying decode for track %s", frame_idx, frame_std, track_id
                    )
                    image = frame_decoder.read(frame_idx)
                    frame_std = float(np.std(image)) if image is not None else 0.0
            except (RuntimeError, Exception) as decode_exc:
                LOGGER.error("Decode failure on frame %s for track %s: %s", frame_idx, track_id, decode_exc)
                image = None
                frame_std = 0.0

            if image is None or frame_std < 1.0:
                LOGGER.error("Skipping frame %s for track %s due to bad_source_frame", frame_idx, track_id)
                rows.append(
                    _make_skip_face_row(
                        args.ep_id,
                        track_id,
                        frame_idx,
                        ts_val,
                        bbox,
                        detector_choice,
                        "bad_source_frame",
                    )
                )
                faces_done = min(faces_total, faces_done + 1)
                progress.emit(
                    faces_done,
                    phase="faces_embed",
                    device=device,
                    detector=detector_choice,
                    tracker=tracker_choice,
                    resolved_device=embed_device,
                    extra=_phase_meta(),
                )
                continue

            if exporter and image is not None:
                exporter.export(frame_idx, image, [(track_id, bbox)], ts=ts_val)
                if exporter.save_crops:
                    crop_rel_path = exporter.crop_rel_path(track_id, frame_idx)
                    if s3_prefixes and s3_prefixes.get("crops"):
                        crop_s3_key = f"{s3_prefixes['crops']}{exporter.crop_component(track_id, frame_idx)}"

            crop, crop_err = _prepare_face_crop(image, bbox, landmarks)
            if image is not None:
                thumb_rel_path, _ = thumb_writer.write(
                    image,
                    bbox,
                    track_id,
                    frame_idx,
                    prepared_crop=crop,
                )
                if thumb_rel_path and s3_prefixes and s3_prefixes.get("thumbs_tracks"):
                    thumb_s3_key = f"{s3_prefixes['thumbs_tracks']}{thumb_rel_path}"
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
                        thumb_rel_path=thumb_rel_path,
                        thumb_s3_key=thumb_s3_key,
                    )
                )
                faces_done = min(faces_total, faces_done + 1)
                progress.emit(
                    faces_done,
                    phase="faces_embed",
                    device=device,
                    detector=detector_choice,
                    tracker=tracker_choice,
                    resolved_device=embed_device,
                    extra=_phase_meta(),
                )
                continue

            crop_std = float(np.std(crop))
            blur_score = _estimate_blur_score(crop)
            skip_reason: str | None = None
            skip_meta: str | None = None
            if conf < FACE_MIN_CONFIDENCE:
                skip_reason = "low_confidence"
                skip_meta = f"{conf:.2f}"
            elif crop_std < FACE_MIN_STD:
                skip_reason = "low_contrast"
                skip_meta = f"{crop_std:.2f}"
            elif blur_score < FACE_MIN_BLUR:
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
                        thumb_rel_path=thumb_rel_path,
                        thumb_s3_key=thumb_s3_key,
                    )
                )
                faces_done = min(faces_total, faces_done + 1)
                progress.emit(
                    faces_done,
                    phase="faces_embed",
                    device=device,
                    detector=detector_choice,
                    tracker=tracker_choice,
                    resolved_device=embed_device,
                    extra=_phase_meta(),
                )
                continue

            # TODO(perf): Batch embeddings per frame by grouping samples and calling
            # embedder.encode() once with all crops from same frame. Currently we call
            # encode() per face. With frame caching (FrameDecoder LRU cache), decode
            # cost is already eliminated for repeated frames. Batching would improve GPU
            # utilization but requires restructuring 300+ lines of embed logic with
            # complex quality checks and early exits. Samples are now sorted by frame_idx
            # to enable future batching if profiling shows significant GPU idle time.
            encoded = embedder.encode([crop])
            if encoded.size:
                embedding_vec = encoded[0]
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
                            crop_rel_path=crop_rel_path,
                            crop_s3_key=crop_s3_key,
                            thumb_rel_path=thumb_rel_path,
                            thumb_s3_key=thumb_s3_key,
                        )
                    )
                    faces_done = min(faces_total, faces_done + 1)
                    progress.emit(
                        faces_done,
                        phase="faces_embed",
                        device=device,
                        detector=detector_choice,
                        tracker=tracker_choice,
                        resolved_device=embed_device,
                        extra=_phase_meta(),
                    )
                    continue
            else:
                # Encoder returned empty array - skip this face
                rows.append(
                    _make_skip_face_row(
                        args.ep_id,
                        track_id,
                        frame_idx,
                        ts_val,
                        bbox,
                        detector_choice,
                        "embedding_failed",
                        crop_rel_path=crop_rel_path,
                        crop_s3_key=crop_s3_key,
                        thumb_rel_path=thumb_rel_path,
                        thumb_s3_key=thumb_s3_key,
                    )
                )
                faces_done = min(faces_total, faces_done + 1)
                progress.emit(
                    faces_done,
                    phase="faces_embed",
                    device=device,
                    detector=detector_choice,
                    tracker=tracker_choice,
                    resolved_device=embed_device,
                    extra=_phase_meta(),
                )
                continue

            track_embeddings[track_id].append((float(quality), embedding_vec.copy()))
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
            if thumb_rel_path:
                prev = track_best_thumb.get(track_id)
                score = quality
                if not prev or score > prev[0]:
                    track_best_thumb[track_id] = (score, thumb_rel_path, thumb_s3_key)

            face_row = {
                "ep_id": args.ep_id,
                "face_id": f"face_{track_id:04d}_{frame_idx:06d}",
                "track_id": track_id,
                "frame_idx": frame_idx,
                "ts": ts_val,
                "bbox_xyxy": bbox,
                "conf": round(float(conf), 4),
                "quality": round(float(quality), 4),
                "embedding": embedding_vec.tolist(),
                "embedding_model": embedding_model_name,
                "detector": detector_choice,
                "pipeline_ver": PIPELINE_VERSION,
            }
            if crop_rel_path:
                face_row["crop_rel_path"] = crop_rel_path
            if crop_s3_key:
                face_row["crop_s3_key"] = crop_s3_key
            if thumb_rel_path:
                face_row["thumb_rel_path"] = thumb_rel_path
            if thumb_s3_key:
                face_row["thumb_s3_key"] = thumb_s3_key
            if landmarks:
                face_row["landmarks"] = [round(float(val), 4) for val in landmarks]
            if seed_cast_id:
                face_row["seed_cast_id"] = seed_cast_id
                face_row["seed_similarity"] = round(float(seed_similarity), 4)
            rows.append(face_row)
            faces_done = min(faces_total, faces_done + 1)
        progress.emit(
            faces_done,
            phase="faces_embed",
            device=device,
            detector=detector_choice,
            tracker=tracker_choice,
            resolved_device=embed_device,
            extra=_phase_meta(),
        )

        # Force emit final progress after loop completes
        progress.emit(
            len(rows),
            phase="faces_embed",
            device=device,
            detector=detector_choice,
            tracker=tracker_choice,
            resolved_device=embed_device,
            extra=_phase_meta(),
            force=True,
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

        embed_path = _faces_embed_path(args.ep_id)
        if embeddings_array:
            np.save(embed_path, np.vstack(embeddings_array))
        else:
            np.save(embed_path, np.zeros((0, 512), dtype=np.float32))

        _update_track_embeddings(track_path, track_embeddings, track_best_thumb, embedding_model_name)
        if exporter:
            exporter.write_indexes()

        # Build preliminary summary for completion events (before S3 sync)
        finished_at = _utcnow_iso()
        summary: Dict[str, Any] = {
            "stage": "faces_embed",
            "ep_id": args.ep_id,
            "faces": len(rows),
            "device": device,
            "resolved_device": embed_device,
            "detector": detector_choice,
            "tracker": tracker_choice,
            "embedding_model": embedding_model_name,
            "frames_exported": exporter.frames_written if exporter and exporter.save_frames else 0,
            "crops_exported": exporter.crops_written if exporter and exporter.save_crops else 0,
            "artifacts": {
                "local": {
                    "faces": str(faces_path),
                    "tracks": str(track_path),
                    "manifests_dir": str(manifests_dir),
                    "frames_dir": str(exporter.frames_dir) if exporter and exporter.save_frames else None,
                    "crops_dir": str(exporter.crops_dir) if exporter and exporter.save_crops else None,
                    "thumbs_dir": str(thumb_writer.root_dir),
                    "faces_embeddings": str(embed_path),
                },
                "s3_prefixes": s3_prefixes,
            },
            "stats": {"faces": len(rows), "embedding_model": embedding_model_name},
        }

        # Emit completion BEFORE S3 sync (which might hang or take long)
        progress.emit(
            len(rows),
            phase="faces_embed",
            device=device,
            detector=detector_choice,
            tracker=tracker_choice,
            resolved_device=embed_device,
            summary=summary,
            force=True,
            extra=_phase_meta("done"),
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

        # Now do S3 sync after completion is signaled
        s3_stats = _sync_artifacts_to_s3(args.ep_id, storage, ep_ctx, exporter, thumb_writer.root_dir)
        summary["artifacts"]["s3_uploads"] = s3_stats
        # Brief delay to ensure final progress event is written and readable
        time.sleep(0.2)
        _write_run_marker(
            args.ep_id,
            "faces_embed",
            {
                "phase": "faces_embed",
                "status": "success",
                "version": APP_VERSION,
                "faces": len(rows),
                "started_at": started_at,
                "finished_at": finished_at,
            },
        )
        return summary
    except Exception as exc:
        progress.fail(str(exc))
        raise
    finally:
        if frame_decoder:
            frame_decoder.close()
        if debug_logger_obj:
            debug_logger_obj.close()
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


def _select_track_prototype(samples: List[TrackEmbeddingSample]) -> tuple[np.ndarray | None, int, float | None]:
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


def _update_track_embeddings(
    track_path: Path,
    track_embeddings: Dict[int, List[TrackEmbeddingSample]],
    track_best_thumb: Dict[int, tuple[float, str, str | None]],
    embedding_model: str,
) -> None:
    if not track_path.exists():
        return
    rows = list(_iter_jsonl(track_path))
    updated: List[dict] = []
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
                    row["face_embedding_spread"] = round(float(spread), 4)
        thumb_info = track_best_thumb.get(track_id)
        if thumb_info:
            _, rel_path, s3_key = thumb_info
            row["thumb_rel_path"] = rel_path
            if s3_key:
                row["thumb_s3_key"] = s3_key
        updated.append(row)
    if updated:
        _write_jsonl(track_path, updated)


def _run_cluster_stage(
    args: argparse.Namespace,
    storage: StorageService | None,
    ep_ctx: EpisodeContext | None,
    s3_prefixes: Dict[str, str] | None,
) -> Dict[str, Any]:
    manifests_dir = get_path(args.ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
    if not faces_path.exists():
        raise FileNotFoundError("faces.jsonl not found; run faces embedding first")
    faces_rows = list(_iter_jsonl(faces_path))
    if not faces_rows:
        raise RuntimeError("faces.jsonl is empty; cannot cluster")
    faces_total = len(faces_rows)
    faces_per_track: Dict[int, int] = defaultdict(int)
    for face_row in faces_rows:
        track_id_val = face_row.get("track_id")
        try:
            track_key = int(track_id_val)
        except (TypeError, ValueError):
            continue
        faces_per_track[track_key] += 1
    track_path = get_path(args.ep_id, "tracks")
    if not track_path.exists():
        raise FileNotFoundError("tracks.jsonl not found; run detect/track first")
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

    progress = ProgressEmitter(
        args.ep_id,
        args.progress_file,
        frames_total=faces_total,
        secs_total=None,
        stride=1,
        fps_detected=None,
        fps_requested=None,
    )
    phase_meta = _non_video_phase_meta()
    device = pick_device(args.device)
    progress.emit(
        0,
        phase="cluster",
        device=device,
        detector=detector_choice,
        tracker=tracker_choice,
        resolved_device=device,
        force=True,
        extra=phase_meta,
    )

    started_at = _utcnow_iso()
    try:
        embedding_rows: List[np.ndarray] = []
        track_ids: List[int] = []
        track_index: Dict[int, dict] = {}
        forced_singletons: List[List[int]] = []
        tracks_with_embeddings: Set[int] = set()

        for row in track_rows:
            track_id = int(row.get("track_id", -1))
            track_index[track_id] = row
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

        if not embedding_rows and not forced_singletons:
            raise RuntimeError("No track embeddings available; rerun faces_embed with detector enabled")

        track_groups: Dict[int, List[int]] = defaultdict(list)
        if embedding_rows:
            labels = _cluster_embeddings(np.vstack(embedding_rows), args.cluster_thresh)
            for tid, label in zip(track_ids, labels):
                track_groups[label].append(tid)

        # Build track embeddings index for outlier removal
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

        min_cluster = max(1, int(args.min_cluster_size))
        identity_payload: List[dict] = []
        thumb_root = get_path(args.ep_id, "frames_root") / "thumbs"
        faces_done = 0
        identity_counter = 1
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
                rep_track_id = max(bucket, key=lambda tid: track_index.get(tid, {}).get("faces_count", 0))
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
                progress.emit(
                    faces_done,
                    phase="cluster",
                    device=device,
                    detector=detector_choice,
                    tracker=tracker_choice,
                    resolved_device=device,
                    extra=phase_meta,
                )

        # Force emit final progress after loop completes
        progress.emit(
            faces_total,
            phase="cluster",
            device=device,
            detector=detector_choice,
            tracker=tracker_choice,
            resolved_device=device,
            extra=phase_meta,
            force=True,
        )

        identities_path = manifests_dir / "identities.json"
        low_cohesion_count = sum(1 for identity in identity_payload if identity.get("low_cohesion"))
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
                "clusters": len(identity_payload),
                "mixed_tracks": len(flagged_tracks),
                "outlier_tracks": len(outlier_tracks),
                "low_cohesion_identities": low_cohesion_count,
            },
            "identities": identity_payload,
        }
        identities_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Generate track representatives and cluster centroids
        try:
            from apps.api.services.track_reps import generate_track_reps_and_centroids
            LOGGER.info("Generating track representatives and cluster centroids for %s", args.ep_id)
            track_reps_summary = generate_track_reps_and_centroids(args.ep_id)
            LOGGER.info(
                "Generated %d track reps and %d cluster centroids",
                track_reps_summary.get("track_reps_count", 0),
                track_reps_summary.get("cluster_centroids_count", 0),
            )
        except Exception as exc:
            LOGGER.warning("Failed to generate track representatives: %s", exc)

        # Build preliminary summary for completion events (before S3 sync)
        finished_at = _utcnow_iso()
        summary: Dict[str, Any] = {
            "stage": "cluster",
            "ep_id": args.ep_id,
            "identities_count": len(identity_payload),
            "faces_count": faces_total,
            "device": device,
            "resolved_device": device,
            "detector": detector_choice,
            "tracker": tracker_choice,
            "artifacts": {
                "local": {
                    "faces": str(faces_path),
                    "identities": str(identities_path),
                    "tracks": str(track_path),
                },
                "s3_prefixes": s3_prefixes,
            },
            "stats": payload["stats"],
        }

        # Emit completion BEFORE S3 sync (which might hang or take long)
        progress.emit(
            faces_total,
            phase="cluster",
            device=device,
            detector=detector_choice,
            tracker=tracker_choice,
            resolved_device=device,
            summary=summary,
            force=True,
            extra=_non_video_phase_meta("done"),
        )
        progress.complete(
            summary,
            device=device,
            detector=detector_choice,
            tracker=tracker_choice,
            resolved_device=device,
            step="cluster",
            extra=phase_meta,
        )

        # Now do S3 sync after completion is signaled
        s3_stats = _sync_artifacts_to_s3(args.ep_id, storage, ep_ctx, exporter=None, thumb_dir=thumb_root)
        summary["artifacts"]["s3_uploads"] = s3_stats
        # Brief delay to ensure final progress event is written and readable
        time.sleep(0.2)
        _write_run_marker(
            args.ep_id,
            "cluster",
            {
                "phase": "cluster",
                "status": "success",
                "version": APP_VERSION,
                "faces": faces_total,
                "identities": len(identity_payload),
                "started_at": started_at,
                "finished_at": finished_at,
            },
        )
        return summary
    except Exception as exc:
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
    if len(track_samples) <= max_samples:
        return track_samples

    # Uniform sampling across the track's frame range
    sampled = []
    n = len(track_samples)

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
            sampled.append(track_samples[idx])

    # Ensure we meet min_samples if the track is long enough
    if len(sampled) < min_samples and len(track_samples) >= min_samples:
        # Add more samples uniformly
        additional_needed = min_samples - len(sampled)
        step = n / (min_samples + 1)
        additional_indices = [int(round((i + 1) * step)) for i in range(additional_needed)]
        for idx in additional_indices:
            if 0 <= idx < n and track_samples[idx] not in sampled:
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
    sim = min(max(float(similarity if similarity is not None else DEFAULT_CLUSTER_SIMILARITY), 0.0), 0.999)
    distance = 1.0 - sim
    return max(distance, 0.01)


def _cluster_embeddings(matrix: np.ndarray, threshold: float) -> np.ndarray:
    if matrix.shape[0] == 1:
        return np.array([0], dtype=int)
    from sklearn.cluster import AgglomerativeClustering

    distance_threshold = _cluster_distance_threshold(threshold)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    return model.fit_predict(matrix)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized vectors."""
    norm_a = np.linalg.norm(a) + 1e-12
    norm_b = np.linalg.norm(b) + 1e-12
    return float(np.dot(a / norm_a, b / norm_b))


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
            s3_key = f"{s3_prefixes['crops']}track_{int(track_id):04d}/frame_{int(frame_idx):06d}.jpg"
    if s3_key:
        rep["s3_key"] = s3_key
    return rep


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
