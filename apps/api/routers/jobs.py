from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any, List, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from py_screenalytics.artifacts import ensure_dirs, get_path

from apps.api.routers import facebank as facebank_router
from apps.api.services.episodes import EpisodeStore
from apps.api.services import jobs as jobs_service
from apps.api.services.jobs import JobNotFoundError, JobService
from apps.api.services.storage import (
    StorageService,
    artifact_prefixes,
    episode_context_from_id,
)

router = APIRouter()
LOGGER = logging.getLogger(__name__)
JOB_SERVICE = JobService()
EPISODE_STORE = EpisodeStore()
DETECTOR_CHOICES = {"retinaface"}
TRACKER_CHOICES = {"bytetrack", "strongsort"}
_DEFAULT_DETECTOR_RAW = os.getenv("DEFAULT_DETECTOR", "retinaface").lower()
DEFAULT_DETECTOR_ENV = _DEFAULT_DETECTOR_RAW if _DEFAULT_DETECTOR_RAW in DETECTOR_CHOICES else "retinaface"
DEFAULT_TRACKER_ENV = os.getenv("DEFAULT_TRACKER", "bytetrack").lower()
SCENE_DETECTOR_CHOICES = getattr(jobs_service, "SCENE_DETECTOR_CHOICES", ("pyscenedetect", "internal", "off"))
SCENE_DETECTOR_DEFAULT = getattr(jobs_service, "SCENE_DETECTOR_DEFAULT", "pyscenedetect")
SCENE_THRESHOLD_DEFAULT = getattr(jobs_service, "SCENE_THRESHOLD_DEFAULT", 27.0)
SCENE_MIN_LEN_DEFAULT = getattr(jobs_service, "SCENE_MIN_LEN_DEFAULT", 12)
SCENE_WARMUP_DETS_DEFAULT = getattr(jobs_service, "SCENE_WARMUP_DETS_DEFAULT", 3)
DEFAULT_CLUSTER_SIMILARITY = float(os.getenv("SCREENALYTICS_CLUSTER_SIM", "0.7"))
MIN_IDENTITY_SIMILARITY = float(os.getenv("SCREENALYTICS_MIN_IDENTITY_SIM", "0.50"))
DEFAULT_JPEG_QUALITY = int(os.getenv("SCREENALYTICS_JPEG_QUALITY", "72"))
DEFAULT_MIN_FRAMES_BETWEEN_CROPS = int(os.getenv("SCREENALYTICS_MIN_FRAMES_BETWEEN_CROPS", "32"))
CLEANUP_ACTIONS = ("split_tracks", "reembed", "recluster", "group_clusters")
CleanupAction = Literal["split_tracks", "reembed", "recluster", "group_clusters"]


LEGACY_SUFFIX = "st" "ub"
LEGACY_KEYS = ("use_" + LEGACY_SUFFIX, LEGACY_SUFFIX)


def _has_legacy_marker(payload: dict | None, query_params: dict[str, str]) -> bool:
    if any(key in query_params for key in LEGACY_KEYS):
        return True
    if payload is None:
        return False
    return any(key in payload for key in LEGACY_KEYS)


async def _reject_legacy_payload(request: Request) -> None:
    query_markers = dict(request.query_params)
    payload_data: dict | None = None
    try:
        body_bytes = await request.body()
    except Exception:
        body_bytes = b""
    if body_bytes:
        try:
            parsed = json.loads(body_bytes)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            payload_data = parsed
    if _has_legacy_marker(payload_data, query_markers):
        raise HTTPException(status_code=400, detail="Stub mode is not supported.")


DEVICE_LITERAL = Literal["auto", "cpu", "mps", "coreml", "metal", "apple", "cuda"]
# Performance profiles:
# - low_power: Lower resource usage for laptops/quiet operation (stride 8, ≤8fps)
# - balanced: Default setting for most workloads (stride 5, ≤24fps)
# - high_accuracy: Maximum recall for powerful systems (stride 1, 30fps)
# - fast_cpu: Alias for low_power
# - performance: Legacy alias for high_accuracy
PROFILE_LITERAL = Literal["fast_cpu", "low_power", "balanced", "performance", "high_accuracy"]


class DetectRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    video: str = Field(..., description="Source video path or URL")
    profile: PROFILE_LITERAL | None = Field(
        None,
        description="Performance profile (fast_cpu/low_power/balanced/high_accuracy). Overrides stride/FPS/min_size defaults.",
    )
    stride: int = Field(
        4,
        description="Frame stride for detection sampling (default 4 aligns with detect+track 42-minute runs)",
    )
    fps: float | None = Field(None, description="Optional target FPS for sampling")
    device: DEVICE_LITERAL = Field(
        "auto",
        description="Execution device (auto→CUDA→CoreML→CPU; accepts coreml/metal/apple aliases)",
    )


class TrackRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")


class DetectTrackRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    profile: PROFILE_LITERAL | None = Field(
        None,
        description="Performance profile (fast_cpu/low_power/balanced/high_accuracy). Overrides stride/FPS/min_size defaults.",
    )
    stride: int = Field(6, description="Frame stride for detection sampling (default: 6)")
    fps: float | None = Field(None, description="Optional target FPS for sampling")
    device: DEVICE_LITERAL = Field(
        "auto",
        description="Execution device (auto→CUDA→CoreML→CPU; accepts coreml/metal/apple aliases)",
    )
    save_frames: bool = Field(False, description="Sample full-frame JPGs to S3/local frames root")
    save_crops: bool = Field(False, description="Save per-track crops (requires tracks)")
    jpeg_quality: int = Field(
        DEFAULT_JPEG_QUALITY,
        ge=1,
        le=100,
        description="JPEG quality for frame/crop exports",
    )
    detector: str = Field(
        DEFAULT_DETECTOR_ENV,
        description="Face detector backend (retinaface)",
    )
    tracker: str = Field(
        DEFAULT_TRACKER_ENV,
        description="Tracker backend (bytetrack or strongsort)",
    )
    max_gap: int | None = Field(30, ge=1, description="Frame gap before forcing a new track")
    det_thresh: float | None = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="RetinaFace detection threshold (0-1)",
    )
    scene_detector: Literal[SCENE_DETECTOR_CHOICES] = Field(
        SCENE_DETECTOR_DEFAULT,
        description="Scene-cut detector backend (PySceneDetect default, internal fallback, or off)",
    )
    scene_threshold: float = Field(
        SCENE_THRESHOLD_DEFAULT,
        ge=0.0,
        description="Scene-cut threshold passed to the selected detector",
    )
    scene_min_len: int = Field(
        SCENE_MIN_LEN_DEFAULT,
        ge=1,
        description="Minimum frames between detected cuts",
    )
    scene_warmup_dets: int = Field(
        SCENE_WARMUP_DETS_DEFAULT,
        ge=0,
        description="Frames forced to run detection immediately after a cut",
    )
    track_high_thresh: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional ByteTrack track_high_thresh override (default 0.5 or env)",
    )
    new_track_thresh: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional ByteTrack new_track_thresh override (default 0.5 or env)",
    )
    track_buffer: int | None = Field(
        None,
        ge=1,
        description="Optional ByteTrack base track_buffer before stride scaling",
    )
    min_box_area: float | None = Field(
        None,
        ge=0.0,
        description="Optional ByteTrack min_box_area override",
    )
    cpu_threads: int | None = Field(
        None,
        ge=1,
        le=16,
        description="CPU thread limit for ML libraries (OMP, MKL, etc.)",
    )


class FacesEmbedRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    profile: PROFILE_LITERAL | None = Field(
        None,
        description="Performance profile (fast_cpu/low_power/balanced/high_accuracy). Controls quality gating and sampling.",
    )
    device: DEVICE_LITERAL | None = Field(
        None,
        description="Execution device (auto→CUDA→CoreML→CPU; accepts coreml/metal/apple aliases)",
    )
    save_frames: bool = Field(False, description="Export sampled frames alongside crops")
    save_crops: bool = Field(False, description="Export crops to data/frames + S3")
    jpeg_quality: int = Field(DEFAULT_JPEG_QUALITY, ge=1, le=100, description="JPEG quality for face crops")
    min_frames_between_crops: int = Field(
        DEFAULT_MIN_FRAMES_BETWEEN_CROPS,
        ge=1,
        le=1000,
        description="Minimum frame gap between successive crops on the same track",
    )
    thumb_size: int = Field(256, ge=64, le=512, description="Square thumbnail size")
    cpu_threads: int | None = Field(
        None,
        ge=1,
        le=16,
        description="CPU thread limit for ML libraries (OMP, MKL, etc.)",
    )


class ClusterRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    profile: PROFILE_LITERAL | None = Field(
        None,
        description="Performance profile (fast_cpu/low_power/balanced/high_accuracy). Controls clustering thresholds.",
    )
    device: DEVICE_LITERAL | None = Field(
        None,
        description="Execution device (defaults to server auto-detect with CUDA/CoreML fallbacks)",
    )
    cluster_thresh: float = Field(
        DEFAULT_CLUSTER_SIMILARITY,
        ge=0.2,
        le=0.99,
        description="Minimum cosine similarity for clustering (converted to 1-sim distance)",
    )
    min_cluster_size: int = Field(2, ge=1, description="Minimum tracks per identity")
    min_identity_sim: float = Field(
        MIN_IDENTITY_SIMILARITY,
        ge=0.0,
        le=0.99,
        description="Minimum cosine similarity for a track to remain in an identity cluster",
    )
    clear_assignments: bool = Field(
        True,
        description="Clear all existing cluster-to-person assignments before clustering. "
        "When True (default), all old assignments are removed and clusters start fresh.",
    )


class CleanupJobRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    profile: PROFILE_LITERAL | None = Field(
        None,
        description="Performance profile (fast_cpu/low_power/balanced/high_accuracy). Applies to all cleanup stages.",
    )
    stride: int = Field(4, ge=1, le=50)
    fps: float | None = Field(None, ge=0.0)
    device: DEVICE_LITERAL = Field("auto", description="Detect/track device (auto→CUDA→CoreML→CPU)")
    embed_device: DEVICE_LITERAL = Field("auto", description="Faces embed device (supports coreml/metal/apple alias)")
    detector: str = Field(DEFAULT_DETECTOR_ENV, description="Detector backend (retinaface)")
    tracker: str = Field(DEFAULT_TRACKER_ENV, description="Tracker backend")
    max_gap: int = Field(30, ge=1, le=240)
    det_thresh: float | None = Field(None, ge=0.0, le=1.0)
    save_frames: bool = False
    save_crops: bool = False
    jpeg_quality: int = Field(DEFAULT_JPEG_QUALITY, ge=50, le=100)
    scene_detector: Literal[SCENE_DETECTOR_CHOICES] = Field(SCENE_DETECTOR_DEFAULT)
    scene_threshold: float = Field(SCENE_THRESHOLD_DEFAULT, ge=0.0)
    scene_min_len: int = Field(SCENE_MIN_LEN_DEFAULT, ge=1)
    scene_warmup_dets: int = Field(SCENE_WARMUP_DETS_DEFAULT, ge=0)
    cluster_thresh: float = Field(DEFAULT_CLUSTER_SIMILARITY, ge=0.2, le=0.99)
    min_cluster_size: int = Field(2, ge=1, le=50)
    min_identity_sim: float = Field(MIN_IDENTITY_SIMILARITY, ge=0.0, le=0.99)
    thumb_size: int = Field(256, ge=64, le=512)
    actions: List[CleanupAction] = Field(default_factory=lambda: list(CLEANUP_ACTIONS))
    write_back: bool = Field(True, description="Update people.json with grouping assignments")


class FacebankBackfillRequest(BaseModel):
    show_id: str = Field(..., description="Show identifier (e.g., RHOBH)")
    cast_id: str | None = Field(None, description="Optional cast id to limit reprocessing")
    dry_run: bool = Field(
        False,
        description="When True, report actions without writing files or uploading to S3",
    )


class EpisodeRunRequest(BaseModel):
    """Request model for running the full episode processing pipeline."""

    ep_id: str = Field(..., description="Episode identifier (e.g., 'rhobh-s05e14')")
    profile: PROFILE_LITERAL | None = Field(
        None,
        description="Performance profile: low_power, balanced, performance (or aliases fast_cpu, high_accuracy). Overrides stride defaults.",
    )
    device: DEVICE_LITERAL = Field(
        "auto",
        description="Execution device (auto, cpu, cuda, coreml)",
    )
    stride: int = Field(
        1,
        ge=1,
        le=30,
        description="Frame stride for detection (1 = every frame)",
    )
    det_thresh: float = Field(
        0.65,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold",
    )
    cluster_thresh: float = Field(
        0.75,
        ge=0.2,
        le=0.99,
        description="Clustering similarity threshold",
    )
    save_crops: bool = Field(False, description="Save per-track face crops")
    save_frames: bool = Field(False, description="Save full frame JPGs")
    reuse_detections: bool = Field(
        False,
        description="Skip detect_track if artifacts exist (dev mode)",
    )
    reuse_embeddings: bool = Field(
        False,
        description="Skip faces_embed if artifacts exist (dev mode)",
    )


@router.post("/jobs/episode-run")
def start_episode_run(req: EpisodeRunRequest) -> dict:
    """Start a full episode processing pipeline job.

    This endpoint runs the complete pipeline:
    1. detect_track - Face detection and tracking
    2. faces_embed - Face embedding extraction
    3. cluster - Identity clustering

    The job runs asynchronously. Use GET /jobs/{job_id} to poll status.

    Args:
        req: Episode run configuration

    Returns:
        {"job_id": "...", "status": "running"}
    """
    video_path = _validate_episode_ready(req.ep_id)

    try:
        record = JOB_SERVICE.start_episode_run_job(
            ep_id=req.ep_id,
            video_path=video_path,
            device=req.device,
            stride=req.stride,
            det_thresh=req.det_thresh,
            cluster_thresh=req.cluster_thresh,
            save_crops=req.save_crops,
            save_frames=req.save_frames,
            reuse_detections=req.reuse_detections,
            reuse_embeddings=req.reuse_embeddings,
            profile=req.profile,
        )
        return {"job_id": record["job_id"], "status": "running"}

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Failed to start episode run job: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/jobs/facebank/backfill_display")
def backfill_facebank_display(req: FacebankBackfillRequest) -> dict:
    """Regenerate missing or low-resolution facebank display derivatives."""
    stats = _run_facebank_backfill(req.show_id, cast_id=req.cast_id, dry_run=req.dry_run)
    return stats


def _artifact_summary(ep_id: str) -> dict:
    ensure_dirs(ep_id)
    return {
        "video": str(get_path(ep_id, "video")),
        "detections": str(get_path(ep_id, "detections")),
        "tracks": str(get_path(ep_id, "tracks")),
        "frames_root": str(get_path(ep_id, "frames_root")),
    }


def _validate_episode_ready(ep_id: str) -> Path:
    video_path = get_path(ep_id, "video")
    flat_video_path = video_path.parent.parent / f"{ep_id}.mp4"
    record = EPISODE_STORE.get(ep_id)
    if not record and not video_path.exists() and flat_video_path.exists():
        try:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(flat_video_path, video_path)
            LOGGER.info("Mirrored %s into canonical episode path %s", flat_video_path, video_path)
        except OSError:
            video_path = flat_video_path
        try:
            EPISODE_STORE.upsert_ep_id(ep_id=ep_id, show_slug=ep_id, season=0, episode=0)
        except Exception as exc:
            LOGGER.debug("Failed to upsert ep_id=%s into EpisodeStore during validation: %s", ep_id, exc)
            record = None
        else:
            record = EPISODE_STORE.get(ep_id)
    if not video_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Episode video not mirrored locally; run Mirror from S3.",
        )
    if not video_path.is_file():
        raise HTTPException(status_code=400, detail="Episode video path is invalid.")
    try:
        with video_path.open("rb"):
            # Just touching the file verifies readable bytes on disk.
            pass
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Episode video unreadable: {exc}") from exc
    return video_path


def _s3_prefixes(ep_id: str) -> dict | None:
    try:
        ctx = episode_context_from_id(ep_id)
    except ValueError:
        return None
    return artifact_prefixes(ctx)


def _progress_file_path(ep_id: str) -> Path:
    manifests_dir = get_path(ep_id, "detections").parent
    manifests_dir.mkdir(parents=True, exist_ok=True)
    return manifests_dir / "progress.json"


def _normalize_detector(detector: str | None) -> str:
    fallback = DEFAULT_DETECTOR_ENV or "retinaface"
    value = (detector or fallback).strip().lower()
    if value not in DETECTOR_CHOICES:
        raise HTTPException(status_code=400, detail=f"Unsupported detector '{detector}'")
    return value


def _normalize_tracker(tracker: str | None) -> str:
    fallback = DEFAULT_TRACKER_ENV or "bytetrack"
    value = (tracker or fallback).strip().lower()
    if value not in TRACKER_CHOICES:
        raise HTTPException(status_code=400, detail=f"Unsupported tracker '{tracker}'")
    return value


def _normalize_scene_detector(scene_detector: str | None) -> str:
    fallback = SCENE_DETECTOR_DEFAULT or "pyscenedetect"
    value = (scene_detector or fallback).strip().lower()
    if value not in SCENE_DETECTOR_CHOICES:
        raise HTTPException(status_code=400, detail=f"Unsupported scene detector '{scene_detector}'")
    return value


def _coerce_cpu_threads(value: Any) -> int | None:
    try:
        threads = int(value)
    except (TypeError, ValueError):
        return None
    if threads < 1:
        return None
    return min(threads, 16)


def _resolve_detect_track_inputs(req: DetectTrackRequest, resolved_device: str | None) -> dict[str, Any]:
    """Apply profile defaults (including laptop low_power) to detect/track inputs."""

    fields_set = getattr(req, "__fields_set__", set())
    profile_value = req.profile or jobs_service.default_profile_for_device(req.device, resolved_device)
    if profile_value == "fast_cpu":
        profile_value = "low_power"
    profile_cfg = jobs_service.load_performance_profile(profile_value) if profile_value else {}

    stride_value = req.stride
    if "stride" not in fields_set and profile_cfg.get("frame_stride"):
        try:
            stride_value = max(int(profile_cfg["frame_stride"]), 1)
        except (TypeError, ValueError):
            stride_value = req.stride

    fps_value = req.fps if req.fps and req.fps > 0 else None
    if fps_value is None:
        fps_raw = profile_cfg.get("detection_fps_limit") or profile_cfg.get("max_fps")
        try:
            fps_numeric = float(fps_raw)
        except (TypeError, ValueError):
            fps_numeric = None
        if fps_numeric and fps_numeric > 0:
            fps_value = fps_numeric

    save_frames_value = req.save_frames
    if "save_frames" not in fields_set and "save_frames" in profile_cfg:
        save_frames_value = bool(profile_cfg.get("save_frames"))

    save_crops_value = req.save_crops
    if "save_crops" not in fields_set and "save_crops" in profile_cfg:
        save_crops_value = bool(profile_cfg.get("save_crops"))

    cpu_threads_value = req.cpu_threads
    if cpu_threads_value is None:
        cpu_threads_value = _coerce_cpu_threads(profile_cfg.get("cpu_threads"))

    return {
        "profile": profile_value,
        "profile_cfg": profile_cfg,
        "stride": stride_value,
        "fps": fps_value,
        "save_frames": save_frames_value,
        "save_crops": save_crops_value,
        "cpu_threads": cpu_threads_value,
    }


def _build_detect_track_command(
    req: DetectTrackRequest,
    video_path: Path,
    progress_path: Path,
    detector_value: str | None = None,
    tracker_value: str | None = None,
    det_thresh: float | None = None,
    scene_detector: str | None = None,
    scene_threshold: float | None = None,
    scene_min_len: int | None = None,
    scene_warmup_dets: int | None = None,
    track_high_thresh: float | None = None,
    new_track_thresh: float | None = None,
    track_buffer: int | None = None,
    min_box_area: float | None = None,
    device_value: str | None = None,
    stride_override: int | None = None,
    fps_override: float | None = None,
    save_frames: bool | None = None,
    save_crops: bool | None = None,
    profile_name: str | None = None,
) -> List[str]:
    detector_value = _normalize_detector(detector_value or req.detector)
    tracker_value = _normalize_tracker(tracker_value or req.tracker)
    device_value = device_value or req.device or "auto"
    scene_detector = _normalize_scene_detector(scene_detector or req.scene_detector)
    scene_threshold = scene_threshold if scene_threshold is not None else req.scene_threshold
    scene_min_len = scene_min_len if scene_min_len is not None else req.scene_min_len
    scene_warmup_dets = scene_warmup_dets if scene_warmup_dets is not None else req.scene_warmup_dets
    det_thresh_value = det_thresh if det_thresh is not None else req.det_thresh
    track_high_thresh = track_high_thresh if track_high_thresh is not None else req.track_high_thresh
    new_track_thresh = new_track_thresh if new_track_thresh is not None else req.new_track_thresh
    track_buffer = track_buffer if track_buffer is not None else req.track_buffer
    min_box_area = min_box_area if min_box_area is not None else req.min_box_area
    stride_value = stride_override if stride_override is not None else req.stride
    fps_value = fps_override if fps_override is not None else req.fps
    save_frames_value = req.save_frames if save_frames is None else bool(save_frames)
    save_crops_value = req.save_crops if save_crops is None else bool(save_crops)
    profile_value = profile_name if profile_name is not None else req.profile

    command: List[str] = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id",
        req.ep_id,
        "--video",
        str(video_path),
        "--stride",
        str(stride_value),
        "--device",
        device_value,
        "--progress-file",
        str(progress_path),
    ]
    if profile_value:
        command += ["--profile", profile_value]
    if fps_value is not None and fps_value > 0:
        command += ["--fps", str(fps_value)]
    if save_frames_value:
        command.append("--save-frames")
    if save_crops_value:
        command.append("--save-crops")
    if req.jpeg_quality and req.jpeg_quality != DEFAULT_JPEG_QUALITY:
        command += ["--jpeg-quality", str(req.jpeg_quality)]
    command += ["--detector", detector_value]
    command += ["--tracker", tracker_value]
    if track_high_thresh is not None:
        command += ["--track-high-thresh", str(track_high_thresh)]
    if new_track_thresh is not None:
        command += ["--new-track-thresh", str(new_track_thresh)]
    if track_buffer is not None:
        command += ["--track-buffer", str(track_buffer)]
    if min_box_area is not None:
        command += ["--min-box-area", str(min_box_area)]
    if req.max_gap:
        command += ["--max-gap", str(req.max_gap)]
    if det_thresh_value is not None:
        command += ["--det-thresh", str(det_thresh_value)]
    command += ["--scene-detector", scene_detector]
    command += ["--scene-threshold", str(scene_threshold)]
    command += ["--scene-min-len", str(max(scene_min_len, 1))]
    command += ["--scene-warmup-dets", str(max(scene_warmup_dets, 0))]
    return command


def _build_faces_command(req: FacesEmbedRequest, progress_path: Path) -> List[str]:
    device_value = req.device or "auto"
    command: List[str] = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id",
        req.ep_id,
        "--faces-embed",
        "--device",
        device_value,
        "--progress-file",
        str(progress_path),
    ]
    if req.profile:
        command += ["--profile", req.profile]
    if req.save_frames:
        command.append("--save-frames")
    if req.save_crops:
        command.append("--save-crops")
    if req.jpeg_quality and req.jpeg_quality != DEFAULT_JPEG_QUALITY:
        command += ["--jpeg-quality", str(req.jpeg_quality)]
    if (
        req.min_frames_between_crops
        and req.min_frames_between_crops != DEFAULT_MIN_FRAMES_BETWEEN_CROPS
    ):
        command += ["--sample-every-n-frames", str(req.min_frames_between_crops)]
    if req.thumb_size and req.thumb_size != 256:
        command += ["--thumb-size", str(req.thumb_size)]
    return command


def _build_cluster_command(req: ClusterRequest, progress_path: Path) -> List[str]:
    device_value = req.device or "auto"
    command: List[str] = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id",
        req.ep_id,
        "--cluster",
        "--device",
        device_value,
        "--progress-file",
        str(progress_path),
    ]
    if req.profile:
        command += ["--profile", req.profile]
    command += ["--cluster-thresh", str(req.cluster_thresh)]
    command += ["--min-cluster-size", str(req.min_cluster_size)]
    command += ["--min-identity-sim", str(req.min_identity_sim)]
    return command


def _format_sse(event_name: str, payload: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"


def _parse_progress_line(line: str) -> dict | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if "phase" not in payload:
        return None
    return payload


def _count_lines(path: Path) -> int:
    # Bug 9 fix: Avoid TOCTOU race by catching exceptions instead of exists() check
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except (OSError, FileNotFoundError):
        return 0


def _load_progress_payload(progress_path: Path | None) -> dict | None:
    # Bug 9 fix: Avoid TOCTOU race by catching exceptions instead of exists() check
    if not progress_path:
        return None
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, FileNotFoundError):
        # File may not exist, be empty, or contain invalid JSON
        return None
    return payload if isinstance(payload, dict) else None


def _wants_sse(request: Request) -> bool:
    accept = (request.headers.get("accept") or "").lower()
    return "text/event-stream" in accept


def _run_job_with_optional_sse(
    command: List[str], request: Request, progress_file: Path | None = None, cpu_threads: int | None = None
):
    env = os.environ.copy()
    # Apply CPU thread limits if specified
    if cpu_threads is not None:
        env["SCREENALYTICS_MAX_CPU_THREADS"] = str(cpu_threads)
        env["OMP_NUM_THREADS"] = str(cpu_threads)
        env["MKL_NUM_THREADS"] = str(cpu_threads)
        env["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
        env["VECLIB_MAXIMUM_THREADS"] = str(cpu_threads)
        env["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
        env["OPENCV_NUM_THREADS"] = str(cpu_threads)
        env["ORT_INTRA_OP_NUM_THREADS"] = str(cpu_threads)
        env["ORT_INTER_OP_NUM_THREADS"] = "1"  # Keep inter-op at 1 for stability
    if _wants_sse(request):
        generator = _stream_progress_command(command, env, request, progress_file=progress_file)
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        progress_payload = None
        if progress_file and progress_file.exists():
            try:
                progress_payload = json.loads(progress_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                progress_payload = None
        detail: dict[str, Any] = {
            "error": "episode_run_failed",
            "stderr": completed.stderr.strip(),
        }
        if progress_payload:
            detail["progress"] = progress_payload
            ep_value = progress_payload.get("ep_id") if isinstance(progress_payload, dict) else None
            if isinstance(ep_value, str):
                detail.setdefault("ep_id", ep_value)
        raise HTTPException(
            status_code=500,
            detail=detail,
        )
    return completed


async def _stream_progress_command(
    command: List[str],
    env: dict,
    request: Request,
    *,
    progress_file: Path | None = None,
):
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(PROJECT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    saw_terminal = False
    last_payload: dict[str, Any] | None = None
    last_run_id: str | None = None
    stdout_task: asyncio.Task | None = None
    stderr_task: asyncio.Task | None = None
    try:
        assert proc.stdout is not None
        stdout_task = asyncio.create_task(proc.stdout.readline())
        if proc.stderr is not None:
            stderr_task = asyncio.create_task(proc.stderr.readline())
        while stdout_task or stderr_task:
            pending = [task for task in (stdout_task, stderr_task) if task]
            if not pending:
                break
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            if stdout_task in done:
                raw_line = stdout_task.result()
                stdout_task = None
                if raw_line:
                    line = raw_line.decode().strip()
                    if line:
                        payload = _parse_progress_line(line)
                        if payload is None:
                            log_payload: dict[str, Any] = {
                                "phase": "log",
                                "stream": "stdout",
                                "message": line,
                            }
                            if last_run_id:
                                log_payload["run_id"] = last_run_id
                            yield _format_sse("log", log_payload)
                        else:
                            last_payload = payload
                            run_id_value = payload.get("run_id")
                            if isinstance(run_id_value, str):
                                last_run_id = run_id_value
                            phase = str(payload.get("phase", "")).lower()
                            if phase == "done":
                                event_name = "done"
                                saw_terminal = True
                            elif phase == "error":
                                event_name = "error"
                                saw_terminal = True
                            else:
                                event_name = "progress"
                            yield _format_sse(event_name, payload)
                            if await request.is_disconnected():
                                proc.terminate()
                                break
                    stdout_task = asyncio.create_task(proc.stdout.readline())
                else:
                    stdout_task = None
            if stderr_task in done:
                raw_err = stderr_task.result()
                stderr_task = None
                if raw_err:
                    err_line = raw_err.decode().strip()
                    if err_line:
                        err_payload: dict[str, Any] = {
                            "phase": "log",
                            "stream": "stderr",
                            "message": err_line,
                        }
                        if last_run_id:
                            err_payload["run_id"] = last_run_id
                        yield _format_sse("log", err_payload)
                        if await request.is_disconnected():
                            proc.terminate()
                            break
                    stderr_task = asyncio.create_task(proc.stderr.readline())
                else:
                    stderr_task = None
        await proc.wait()
        if proc.returncode not in (0, None) and not saw_terminal:
            stderr_text = ""
            if proc.stderr:
                stderr_bytes = await proc.stderr.read()
                stderr_text = stderr_bytes.decode().strip()
            payload = {
                "phase": "error",
                "error": stderr_text or f"episode_run exited with code {proc.returncode}",
                "return_code": proc.returncode,
            }
            yield _format_sse("error", payload)
        elif proc.returncode in (0, None) and not saw_terminal:
            synthesized: dict[str, Any] | None = None
            if last_payload and isinstance(last_payload.get("summary"), dict):
                synthesized = dict(last_payload)
            elif progress_file and progress_file.exists():
                try:
                    file_payload = json.loads(progress_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    file_payload = None
                if isinstance(file_payload, dict):
                    synthesized = file_payload
            if synthesized is None:
                synthesized = {}
            synthesized["phase"] = "done"
            yield _format_sse("done", synthesized)
    finally:
        for task in (stdout_task, stderr_task):
            if task:
                task.cancel()
        if proc.returncode is None:
            proc.kill()


@router.post("/detect")
async def enqueue_detect(req: DetectRequest, request: Request) -> dict:
    await _reject_legacy_payload(request)
    artifacts = _artifact_summary(req.ep_id)

    command: List[str] = [
        "python",
        "tools/episode_run.py",
        "--ep-id",
        req.ep_id,
        "--video",
        req.video,
        "--stride",
        str(req.stride),
    ]
    if req.fps is not None:
        command += ["--fps", str(req.fps)]
    command += ["--device", req.device]

    return {
        "job": "detect",
        "ep_id": req.ep_id,
        "queued": False,
        "command": command,
        "artifacts": artifacts,
        "note": "Queue integration pending; run command manually for now.",
    }


@router.post("/track")
def enqueue_track(req: TrackRequest) -> dict:
    artifacts = _artifact_summary(req.ep_id)
    command = [
        "python",
        "FEATURES/tracking/src/bytetrack_runner.py",
        "--ep-id",
        req.ep_id,
        "--detections",
        artifacts["detections"],
        "--output",
        artifacts["tracks"],
    ]
    return {
        "job": "track",
        "ep_id": req.ep_id,
        "queued": False,
        "command": command,
        "artifacts": artifacts,
        "note": "Tracking runs locally; detection must be prepared first.",
    }


@router.post("/detect_track")
async def run_detect_track(req: DetectTrackRequest, request: Request):
    await _reject_legacy_payload(request)
    artifacts = _artifact_summary(req.ep_id)
    video_path = _validate_episode_ready(req.ep_id)
    detector_value = _normalize_detector(req.detector)
    tracker_value = _normalize_tracker(req.tracker)
    scene_detector_value = _normalize_scene_detector(req.scene_detector)
    resolved_device = req.device
    try:
        if getattr(jobs_service, "episode_run", None) and hasattr(jobs_service.episode_run, "resolve_device"):
            resolved_device = jobs_service.episode_run.resolve_device(req.device, LOGGER)  # type: ignore[attr-defined]
    except Exception:
        resolved_device = req.device
    try:
        JOB_SERVICE.ensure_retinaface_ready(detector_value, resolved_device, req.det_thresh)
    except ValueError as exc:
        # Model validation failed - provide actionable error message
        raise HTTPException(
            status_code=400,
            detail=f"Model initialization failed: {exc}",
        ) from exc
    except (ImportError, ModuleNotFoundError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Missing dependencies for detector '{detector_value}': {exc}. "
            "Install ML stack with: pip install -r requirements-ml.txt",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error initializing detector '{detector_value}' on device '{req.device}': {exc}",
        ) from exc
    progress_path = _progress_file_path(req.ep_id)
    try:
        progress_path.unlink()
    except FileNotFoundError:
        # Missing progress files are normal when a prior run never started.
        pass
    # Remove downstream artifacts before launching a new detect/track run
    # This ensures faces/cluster artifacts don't reference obsolete track IDs
    # NOTE: detections.jsonl and tracks.jsonl are NOT deleted here - they are written
    # atomically via temp files and will only be replaced on successful completion
    manifests_dir = get_path(req.ep_id, "detections").parent
    embeds_dir = manifests_dir.parent / "embeds" / req.ep_id
    stale_paths = [
        manifests_dir / "faces.jsonl",
        manifests_dir / "identities.json",
        embeds_dir / "faces.npy",
        embeds_dir / "tracks.npy",
        embeds_dir / "track_ids.json",
    ]
    for stale_path in stale_paths:
        try:
            stale_path.unlink()
            LOGGER.info("Removed stale artifact before detect/track rerun: %s", stale_path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            LOGGER.warning("Failed to remove stale artifact %s: %s", stale_path, exc)
    effective = _resolve_detect_track_inputs(req, resolved_device)
    command = _build_detect_track_command(
        req,
        video_path,
        progress_path,
        detector_value,
        tracker_value,
        req.det_thresh,
        scene_detector_value,
        req.scene_threshold,
        req.scene_min_len,
        req.scene_warmup_dets,
        req.track_high_thresh,
        req.new_track_thresh,
        req.track_buffer,
        req.min_box_area,
        resolved_device,
        stride_override=effective["stride"],
        fps_override=effective["fps"],
        save_frames=effective["save_frames"],
        save_crops=effective["save_crops"],
        profile_name=effective["profile"],
    )
    result = _run_job_with_optional_sse(
        command,
        request,
        progress_file=progress_path,
        cpu_threads=effective["cpu_threads"],
    )
    if isinstance(result, StreamingResponse):
        return result

    detections_count = _count_lines(Path(artifacts["detections"]))
    tracks_count = _count_lines(Path(artifacts["tracks"]))
    progress_payload = _load_progress_payload(progress_path) or {}
    progress_resolved_device = progress_payload.get("resolved_device")
    resolved_device_out = progress_resolved_device or resolved_device

    return {
        "job": "detect_track",
        "ep_id": req.ep_id,
        "command": command,
        "device": req.device,
        "resolved_device": resolved_device_out,
        "profile": effective["profile"],
        "stride": effective["stride"],
        "save_frames": effective["save_frames"],
        "save_crops": effective["save_crops"],
        "cpu_threads": effective["cpu_threads"],
        "detector": detector_value,
        "tracker": tracker_value,
        "scene_detector": scene_detector_value,
        "detections_count": detections_count,
        "tracks_count": tracks_count,
        "artifacts": artifacts,
        "progress_file": str(progress_path),
    }


@router.post("/faces_embed")
async def run_faces_embed(req: FacesEmbedRequest, request: Request):
    await _reject_legacy_payload(request)
    _validate_episode_ready(req.ep_id)
    track_path = get_path(req.ep_id, "tracks")
    if not track_path.exists():
        raise HTTPException(status_code=400, detail="tracks.jsonl not found; run detect/track first")
    device_value = req.device or "auto"
    try:
        JOB_SERVICE.ensure_arcface_ready(device_value)
    except ValueError as exc:
        # Model validation failed - provide actionable error message
        raise HTTPException(
            status_code=400,
            detail=f"Face embedding model initialization failed: {exc}",
        ) from exc
    except (ImportError, ModuleNotFoundError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Missing dependencies for face embedding: {exc}. "
            "Install ML stack with: pip install -r requirements-ml.txt",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error initializing ArcFace on device '{device_value}': {exc}",
        ) from exc
    progress_path = _progress_file_path(req.ep_id)
    command = _build_faces_command(req, progress_path)
    result = _run_job_with_optional_sse(command, request, progress_file=progress_path, cpu_threads=req.cpu_threads)
    if isinstance(result, StreamingResponse):
        return result

    manifests_dir = get_path(req.ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
    faces_count = _count_lines(faces_path)
    frames_dir = get_path(req.ep_id, "frames_root") / "frames"
    progress_payload = _load_progress_payload(progress_path)
    resolved_device = progress_payload.get("resolved_device") if progress_payload else None
    return {
        "job": "faces_embed",
        "ep_id": req.ep_id,
        "faces_count": faces_count,
        "device": device_value,
        "resolved_device": resolved_device,
        "artifacts": {
            "faces": str(faces_path),
            "tracks": str(track_path),
            "frames": str(frames_dir) if req.save_frames else None,
            "s3_prefixes": _s3_prefixes(req.ep_id),
        },
        "progress_file": str(progress_path),
    }


@router.post("/cluster")
async def run_cluster(req: ClusterRequest, request: Request):
    await _reject_legacy_payload(request)
    manifests_dir = get_path(req.ep_id, "detections").parent
    faces_path = manifests_dir / "faces.jsonl"
    if not faces_path.exists():
        raise HTTPException(status_code=400, detail="faces.jsonl not found; run faces_embed first")

    # Clear all existing cluster-to-person assignments before clustering
    # This ensures old cluster IDs and person links are removed so Faces Review starts fresh
    cleared_count = 0
    if req.clear_assignments:
        try:
            from apps.api.services.grouping import GroupingService

            grouping_service = GroupingService()
            cleared_count = grouping_service._clear_person_assignments(req.ep_id)
            if cleared_count > 0:
                LOGGER.info(
                    "[%s] Cleared %d existing assignment(s) before clustering",
                    req.ep_id,
                    cleared_count,
                )
        except Exception as exc:
            LOGGER.warning(
                "[%s] Failed to clear assignments before clustering: %s",
                req.ep_id,
                exc,
            )

    progress_path = _progress_file_path(req.ep_id)
    command = _build_cluster_command(req, progress_path)
    result = _run_job_with_optional_sse(command, request, progress_file=progress_path)
    if isinstance(result, StreamingResponse):
        return result

    identities_path = manifests_dir / "identities.json"
    identities_count = 0
    faces_count = _count_lines(faces_path)
    if identities_path.exists():
        try:
            identities_doc = json.loads(identities_path.read_text(encoding="utf-8"))
            identities_count = len(identities_doc.get("identities", []))
            faces_count = int(identities_doc.get("stats", {}).get("faces", faces_count))
        except json.JSONDecodeError:
            # A partially written identities.json should not break the API response.
            pass
    progress_payload = _load_progress_payload(progress_path)
    resolved_device = progress_payload.get("resolved_device") if progress_payload else None
    return {
        "job": "cluster",
        "ep_id": req.ep_id,
        "identities_count": identities_count,
        "faces_count": faces_count,
        "device": req.device or "auto",
        "resolved_device": resolved_device,
        "artifacts": {
            "identities": str(identities_path),
            "faces": str(faces_path),
            "s3_prefixes": _s3_prefixes(req.ep_id),
        },
        "progress_file": str(progress_path),
    }


@router.post("/detect_track_async")
async def enqueue_detect_track_async(req: DetectTrackRequest, request: Request) -> dict:
    await _reject_legacy_payload(request)
    artifacts = _artifact_summary(req.ep_id)
    video_path = _validate_episode_ready(req.ep_id)
    detector_value = _normalize_detector(req.detector)
    tracker_value = _normalize_tracker(req.tracker)
    scene_detector_value = _normalize_scene_detector(req.scene_detector)
    resolved_device = req.device
    try:
        if getattr(jobs_service, "episode_run", None) and hasattr(jobs_service.episode_run, "resolve_device"):
            resolved_device = jobs_service.episode_run.resolve_device(req.device, LOGGER)  # type: ignore[attr-defined]
    except Exception:
        resolved_device = req.device
    effective = _resolve_detect_track_inputs(req, resolved_device)
    try:
        job = JOB_SERVICE.start_detect_track_job(
            ep_id=req.ep_id,
            stride=effective["stride"],
            fps=effective["fps"],
            device=req.device,
            video_path=video_path,
            profile=effective["profile"],
            cpu_threads=effective["cpu_threads"],
            save_frames=effective["save_frames"],
            save_crops=effective["save_crops"],
            jpeg_quality=req.jpeg_quality,
            detector=detector_value,
            tracker=tracker_value,
            max_gap=req.max_gap,
            det_thresh=req.det_thresh,
            scene_detector=scene_detector_value,
            scene_threshold=req.scene_threshold,
            scene_min_len=req.scene_min_len,
            scene_warmup_dets=req.scene_warmup_dets,
            track_high_thresh=req.track_high_thresh,
            new_track_thresh=req.new_track_thresh,
            track_buffer=req.track_buffer,
            min_box_area=req.min_box_area,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "job_id": job["job_id"],
        "ep_id": req.ep_id,
        "state": job["state"],
        "started_at": job["started_at"],
        "progress_file": job["progress_file"],
        "requested": job.get("requested"),
        "artifacts": artifacts,
    }


@router.post("/faces_embed_async")
async def enqueue_faces_embed_async(req: FacesEmbedRequest, request: Request) -> dict:
    await _reject_legacy_payload(request)
    # Ensure the local mirror exists so asynchronous jobs don't fail later
    _validate_episode_ready(req.ep_id)
    try:
        job = JOB_SERVICE.start_faces_embed_job(
            ep_id=req.ep_id,
            device=req.device or "auto",
            profile=req.profile,
            cpu_threads=req.cpu_threads,
            save_frames=req.save_frames,
            save_crops=req.save_crops,
            jpeg_quality=req.jpeg_quality,
            thumb_size=req.thumb_size,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "job_id": job["job_id"],
        "ep_id": req.ep_id,
        "state": job["state"],
        "started_at": job["started_at"],
        "progress_file": job.get("progress_file"),
        "requested": job.get("requested"),
    }


@router.post("/cluster_async")
async def enqueue_cluster_async(req: ClusterRequest, request: Request) -> dict:
    await _reject_legacy_payload(request)
    try:
        job = JOB_SERVICE.start_cluster_job(
            ep_id=req.ep_id,
            device=req.device or "auto",
            cluster_thresh=req.cluster_thresh,
            min_cluster_size=req.min_cluster_size,
            min_identity_sim=req.min_identity_sim,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "job_id": job["job_id"],
        "ep_id": req.ep_id,
        "state": job["state"],
        "started_at": job["started_at"],
        "progress_file": job.get("progress_file"),
        "requested": job.get("requested"),
    }


@router.post("/episode_cleanup_async")
async def enqueue_episode_cleanup_async(req: CleanupJobRequest, request: Request) -> dict:
    await _reject_legacy_payload(request)
    artifacts = _artifact_summary(req.ep_id)
    video_path = _validate_episode_ready(req.ep_id)
    fps_value = req.fps if req.fps and req.fps > 0 else None
    detector_value = _normalize_detector(req.detector)
    tracker_value = _normalize_tracker(req.tracker)
    scene_detector_value = _normalize_scene_detector(req.scene_detector)
    actions = req.actions or list(CLEANUP_ACTIONS)
    try:
        job = JOB_SERVICE.start_episode_cleanup_job(
            ep_id=req.ep_id,
            video_path=video_path,
            stride=req.stride,
            fps=fps_value,
            device=req.device,
            embed_device=req.embed_device or req.device,
            save_frames=req.save_frames,
            save_crops=req.save_crops,
            jpeg_quality=req.jpeg_quality,
            detector=detector_value,
            tracker=tracker_value,
            max_gap=req.max_gap,
            det_thresh=req.det_thresh,
            scene_detector=scene_detector_value,
            scene_threshold=req.scene_threshold,
            scene_min_len=req.scene_min_len,
            scene_warmup_dets=req.scene_warmup_dets,
            cluster_thresh=req.cluster_thresh,
            min_cluster_size=req.min_cluster_size,
            min_identity_sim=req.min_identity_sim,
            thumb_size=req.thumb_size,
            actions=actions,
            write_back=req.write_back,
            profile=req.profile,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "job_id": job["job_id"],
        "ep_id": req.ep_id,
        "state": job["state"],
        "started_at": job["started_at"],
        "progress_file": job.get("progress_file"),
        "requested": job.get("requested"),
        "artifacts": artifacts,
    }


class AnalyzeScreenTimeRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    run_id: str | None = Field(
        None,
        description=(
            "Optional pipeline run identifier. When omitted, defaults to the most recent successful run "
            "(when available)."
        ),
    )
    quality_min: float | None = Field(None, ge=0.0, le=1.0, description="Minimum face quality threshold")
    gap_tolerance_s: float | None = Field(None, ge=0.0, description="Gap tolerance in seconds")
    use_video_decode: bool | None = Field(None, description="Use video decode for precise timestamps")
    screen_time_mode: Literal["faces", "tracks"] | None = Field(
        None, description="How to derive intervals for screen time aggregation"
    )
    edge_padding_s: float | None = Field(None, ge=0.0, description="Edge padding (seconds) applied to each interval")
    track_coverage_min: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum detection coverage when using track mode",
    )
    preset: str | None = Field(
        None,
        description="Named preset defined in config/pipeline/screen_time_v2.yaml",
    )


@router.post("/screen_time/analyze")
async def analyze_screen_time(req: AnalyzeScreenTimeRequest, request: Request) -> dict:
    """Analyze per-cast screen time from assigned faces and tracks."""
    await _reject_legacy_payload(request)
    try:
        job = JOB_SERVICE.start_screen_time_job(
            ep_id=req.ep_id,
            run_id=req.run_id,
            quality_min=req.quality_min,
            gap_tolerance_s=req.gap_tolerance_s,
            use_video_decode=req.use_video_decode,
            screen_time_mode=req.screen_time_mode,
            edge_padding_s=req.edge_padding_s,
            track_coverage_min=req.track_coverage_min,
            preset=req.preset,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "job_id": job["job_id"],
        "ep_id": req.ep_id,
        "state": job["state"],
        "started_at": job["started_at"],
        "progress_file": job.get("progress_file"),
        "requested": job.get("requested"),
    }


class BodyTrackingRequest(BaseModel):
    ep_id: str = Field(..., description="Episode identifier")
    run_id: str = Field(..., description="Pipeline run identifier (required for body tracking)")


@router.post("/body_tracking/run")
async def run_body_tracking(req: BodyTrackingRequest, request: Request) -> dict:
    """Run body tracking for a specific run_id."""
    await _reject_legacy_payload(request)
    try:
        job = JOB_SERVICE.start_body_tracking_job(ep_id=req.ep_id, run_id=req.run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "job_id": job["job_id"],
        "ep_id": req.ep_id,
        "state": job["state"],
        "started_at": job["started_at"],
        "progress_file": job.get("progress_file"),
        "requested": job.get("requested"),
    }


@router.post("/body_tracking/fusion")
async def run_body_tracking_fusion(req: BodyTrackingRequest, request: Request) -> dict:
    """Run face↔body fusion + screen-time comparison for a specific run_id."""
    await _reject_legacy_payload(request)
    try:
        job = JOB_SERVICE.start_body_tracking_fusion_job(ep_id=req.ep_id, run_id=req.run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "job_id": job["job_id"],
        "ep_id": req.ep_id,
        "state": job["state"],
        "started_at": job["started_at"],
        "progress_file": job.get("progress_file"),
        "requested": job.get("requested"),
    }


class VideoExportRequest(BaseModel):
    """Request model for video export job."""

    ep_id: str = Field(..., description="Episode identifier")
    include_unidentified: bool = Field(
        True,
        description="Include faces without cast assignment (shown in gray)",
    )
    output_fps: float | None = Field(
        None,
        ge=1,
        le=30,
        description="Output FPS (default: 15fps for smaller file)",
    )


@router.post("/video_export")
async def export_video_with_overlays(req: VideoExportRequest, request: Request) -> dict:
    """Export full episode video with face overlay annotations.

    Creates a video with bounding boxes and name labels for all detected
    and identified faces. The video is uploaded to S3 when complete.

    This is a long-running job - poll the job progress endpoint to track status.
    """
    await _reject_legacy_payload(request)
    try:
        job = JOB_SERVICE.start_video_export_job(
            ep_id=req.ep_id,
            include_unidentified=req.include_unidentified,
            output_fps=req.output_fps,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "job_id": job["job_id"],
        "ep_id": req.ep_id,
        "state": job["state"],
        "started_at": job["started_at"],
        "progress_file": job.get("progress_file"),
        "requested": job.get("requested"),
    }


@router.get("")
def list_jobs(ep_id: str | None = None, job_type: str | None = None, limit: int = 50) -> dict:
    """List all jobs, optionally filtered by episode and/or job type."""
    jobs = JOB_SERVICE.list_jobs(ep_id=ep_id, job_type=job_type, limit=limit)
    # Return simplified job records
    simplified_jobs = []
    for job in jobs:
        simplified_jobs.append(
            {
                "job_id": job["job_id"],
                "job_type": job.get("job_type"),
                "ep_id": job["ep_id"],
                "state": job["state"],
                "started_at": job["started_at"],
                "ended_at": job.get("ended_at"),
                "error": job.get("error"),
                "requested": job.get("requested"),
            }
        )
    return {"jobs": simplified_jobs, "count": len(simplified_jobs)}


@router.get("/{job_id}/progress")
def get_job_progress(job_id: str) -> dict:
    try:
        job = JOB_SERVICE.get(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    progress = JOB_SERVICE.get_progress(job_id) or {}
    track_metrics = job.get("track_metrics")
    if not track_metrics and isinstance(progress, dict):
        track_metrics = progress.get("track_metrics")
    return {
        "job_id": job_id,
        "ep_id": job["ep_id"],
        "state": job["state"],
        "started_at": job["started_at"],
        "ended_at": job.get("ended_at"),
        "track_metrics": track_metrics,
        "progress": progress,
    }


@router.get("/{job_id}")
def job_details(job_id: str) -> dict:
    try:
        job = JOB_SERVICE.get(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job["job_id"],
        "ep_id": job["ep_id"],
        "state": job["state"],
        "started_at": job["started_at"],
        "ended_at": job.get("ended_at"),
        "summary": job.get("summary"),
        "error": job.get("error"),
        "track_metrics": job.get("track_metrics"),
        "requested": job.get("requested"),
        "progress_file": job.get("progress_file"),
    }


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    try:
        job = JOB_SERVICE.cancel(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "state": job.get("state"),
        "ended_at": job.get("ended_at"),
        "error": job.get("error"),
    }


def _run_facebank_backfill(show_id: str, *, cast_id: str | None, dry_run: bool) -> dict:
    service = facebank_router.facebank_service
    show_dir = service.facebank_dir / show_id
    if not show_dir.exists():
        raise HTTPException(status_code=404, detail=f"Show '{show_id}' has no facebank data")

    if cast_id:
        cast_dirs = [show_dir / cast_id]
        if not cast_dirs[0].exists():
            raise HTTPException(status_code=404, detail=f"Cast '{cast_id}' has no facebank data")
    else:
        cast_dirs = sorted(path for path in show_dir.iterdir() if path.is_dir())

    storage = StorageService()
    stats = {"updated": 0, "skipped": 0, "failed": 0, "low_res": 0}

    for cast_path in cast_dirs:
        cid = cast_path.name
        data = service._load_facebank(show_id, cid)
        seeds = data.get("seeds", [])
        if not seeds:
            continue
        changed = False
        seeds_dir = service._seeds_dir(show_id, cid)
        for seed in seeds:
            status, low_res = _rebuild_seed_display(
                show_id,
                cid,
                seed,
                seeds_dir,
                storage,
                dry_run=dry_run,
            )
            if status == "skipped":
                stats["skipped"] += 1
                continue
            if status == "failed":
                stats["failed"] += 1
                continue
            stats["updated"] += 1
            if low_res:
                stats["low_res"] += 1
            if not dry_run:
                changed = True
        if changed and not dry_run:
            service._save_facebank(show_id, cid, data)
    return stats


def _rebuild_seed_display(
    show_id: str,
    cast_id: str,
    seed: dict,
    seeds_dir: Path,
    storage: StorageService,
    *,
    dry_run: bool,
) -> tuple[str, bool]:
    storage_id = seed.get("storage_seed_id") or _guess_storage_id(seed)
    if not storage_id:
        storage_id = seed.get("fb_id")
    if not storage_id:
        LOGGER.warning("Seed %s/%s missing identifier; skipping", show_id, seed.get("fb_id"))
        return "failed", False

    display_path = _resolve_seed_path(seed, "image_uri", seeds_dir, storage_id, "_d")
    long_side = _display_long_side(seed, display_path)
    has_key = bool(seed.get("display_s3_key") or seed.get("image_s3_key"))
    if long_side is not None and long_side > 128 and has_key:
        return "skipped", bool(seed.get("display_low_res"))

    orig_path = _resolve_seed_path(seed, "orig_uri", seeds_dir, storage_id, "_o")
    embed_path = _resolve_seed_path(seed, "embed_uri", seeds_dir, storage_id, "_e")

    display_bgr = None
    display_dims: list[int] | None = None
    low_res = False

    if orig_path and orig_path.exists():
        source_bgr = _load_image_bgr(orig_path)
        bbox = (seed.get("quality") or {}).get("display_bbox")
        cropped = _crop_with_bbox(source_bgr, bbox)
        try:
            resized, dims, low_res = facebank_router._resize_display_image(cropped)
        except ValueError:
            resized, dims, low_res = facebank_router._resize_display_image(source_bgr)
        display_bgr = resized
        display_dims = dims
    elif embed_path and embed_path.exists():
        display_bgr, display_dims = _upscale_embed_display(embed_path)
        low_res = True
    else:
        LOGGER.warning(
            "Seed %s/%s missing orig/embed derivatives; cannot rebuild display",
            show_id,
            seed.get("fb_id"),
        )
        return "failed", False

    if dry_run:
        return "updated", low_res

    target_path = display_path
    if target_path is None:
        target_path = seeds_dir / f"{storage_id}_d.{facebank_router.SEED_DISPLAY_FORMAT}"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    facebank_router._save_derivative(
        display_bgr.astype(np.uint8, copy=False),
        target_path,
        facebank_router.SEED_DISPLAY_FORMAT,
    )

    key = storage.upload_facebank_seed(
        show_id,
        cast_id,
        storage_id,
        target_path,
        object_name=f"seeds/{target_path.name}",
        content_type_hint=facebank_router.DISPLAY_MIME,
    )

    seed["storage_seed_id"] = storage_id
    seed["image_uri"] = str(target_path)
    seed["display_uri"] = str(target_path)
    seed["display_dims"] = display_dims
    seed["display_low_res"] = low_res
    quality = seed.get("quality") or {}
    quality["display_dims"] = display_dims
    quality["display_low_res"] = low_res
    seed["quality"] = quality
    if key:
        seed["display_s3_key"] = key
        seed["image_s3_key"] = key
        seed["display_key"] = key
    return "updated", low_res


def _guess_storage_id(seed: dict) -> str | None:
    for key in ("image_uri", "display_s3_key", "image_s3_key", "display_key"):
        value = seed.get(key)
        if not value:
            continue
        stem = Path(value).stem
        if "_" in stem:
            return stem.split("_", 1)[0]
    return seed.get("fb_id")


def _resolve_seed_path(seed: dict, field: str, seeds_dir: Path, storage_id: str | None, suffix: str) -> Path | None:
    candidates: list[Path] = []
    raw = seed.get(field)
    if raw:
        candidates.append(Path(raw))
    if storage_id:
        for ext in (".png", ".jpg", ".jpeg"):
            candidates.append(seeds_dir / f"{storage_id}{suffix}{ext}")
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return path
        if not path.is_absolute():
            alt = (facebank_router.facebank_service.data_root / path).resolve()
            if alt.exists():
                return alt
            alt2 = seeds_dir / path.name
            if alt2.exists():
                return alt2
    return None


def _display_long_side(seed: dict, display_path: Path | None) -> int | None:
    dims = seed.get("display_dims")
    if isinstance(dims, (list, tuple)) and len(dims) == 2 and all(isinstance(v, (int, float)) for v in dims):
        width, height = int(dims[0]), int(dims[1])
        if width > 0 and height > 0:
            return max(width, height)
    if display_path and display_path.exists():
        with Image.open(display_path) as img:
            return max(img.size)
    return None


def _load_image_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        corrected = ImageOps.exif_transpose(img)
        rgb = np.asarray(corrected.convert("RGB"))
    return np.ascontiguousarray(rgb[..., ::-1])


def _crop_with_bbox(image_bgr: np.ndarray, bbox: list[float] | None) -> np.ndarray:
    if not bbox or len(bbox) != 4:
        return image_bgr
    x1, y1, x2, y2 = [int(round(val)) for val in bbox]
    h, w = image_bgr.shape[:2]
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(max(x2, x1 + 1), w)
    y2 = min(max(y2, y1 + 1), h)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return image_bgr
    return crop


def _upscale_embed_display(embed_path: Path) -> tuple[np.ndarray, list[int]]:
    with Image.open(embed_path) as img:
        rgb = np.asarray(img.convert("RGB"))
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    h, w = bgr.shape[:2]
    long_side = max(h, w)
    target = max(256, long_side)
    if long_side == 0:
        return bgr, [w, h]
    if long_side >= target:
        return bgr, [w, h]
    scale = target / long_side
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    resample_attr = getattr(Image, "Resampling", Image)
    resample_filter = getattr(resample_attr, "LANCZOS", getattr(Image, "BICUBIC", Image.NEAREST))
    pil_img = Image.fromarray(bgr[..., ::-1])
    resized = pil_img.resize((new_w, new_h), resample=resample_filter)
    pil_img.close()
    arr = np.asarray(resized)[..., ::-1]
    resized.close()
    return np.ascontiguousarray(arr), [new_w, new_h]
