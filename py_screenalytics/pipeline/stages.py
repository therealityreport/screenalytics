"""Pipeline stage implementations.

This module provides the core stage functions for the episode processing pipeline.
It acts as the canonical source for stage logic, abstracting away the underlying
implementations.

The stages are:
1. detect_track - Face detection + tracking (RetinaFace + ByteTrack)
2. faces_embed - Face embedding extraction (ArcFace)
3. cluster - Identity clustering (Agglomerative)

Usage:
    from py_screenalytics.pipeline.stages import (
        run_detect_track,
        run_faces_embed,
        run_cluster,
    )

    result = run_detect_track(episode_id, video_path, config)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

# Ensure project root is in path for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from py_screenalytics.pipeline.constants import (
    ensure_artifact_dirs,
    get_artifact_path,
    ArtifactKind,
)

if TYPE_CHECKING:
    from py_screenalytics.pipeline.episode_engine import EpisodeRunConfig

LOGGER = logging.getLogger("pipeline.stages")


@dataclass
class StageContext:
    """Context passed to stage functions."""

    episode_id: str
    video_path: Path
    data_root: Optional[Path]

    # Storage service for S3 uploads
    storage: Any = None
    episode_context: Any = None
    s3_prefixes: Optional[Dict[str, str]] = None


def _utcnow_iso() -> str:
    """Return current UTC time as ISO format string."""
    return datetime.now(timezone.utc).isoformat()


def _get_storage_context(episode_id: str) -> tuple:
    """Get storage service and episode context for S3 uploads."""
    try:
        from apps.api.services.storage import (
            StorageService,
            artifact_prefixes,
            episode_context_from_id,
        )

        storage_backend = os.environ.get("STORAGE_BACKEND", "s3").lower()
        storage = None
        if storage_backend in {"s3", "minio"}:
            try:
                storage = StorageService()
            except Exception as exc:
                LOGGER.warning("Storage init failed (%s); disabling uploads", exc)
                storage = None

        try:
            ep_ctx = episode_context_from_id(episode_id)
        except ValueError:
            LOGGER.warning("Unable to parse episode id '%s'", episode_id)
            ep_ctx = None

        prefixes = artifact_prefixes(ep_ctx) if ep_ctx else None
        return storage, ep_ctx, prefixes

    except ImportError:
        LOGGER.warning("Storage services not available")
        return None, None, None


def _config_to_args_namespace(config: "EpisodeRunConfig", episode_id: str, video_path: Path) -> Any:
    """Convert EpisodeRunConfig to argparse.Namespace for legacy stage functions.

    This is a transitional helper that allows us to call the existing stage
    implementations with the new config structure.
    """
    import argparse

    args = argparse.Namespace()

    # Core identifiers
    args.ep_id = episode_id
    args.video = str(video_path)
    args.run_id = config.run_id

    # Device settings
    args.device = config.device
    args.embed_device = config.embed_device
    args.allow_cpu_fallback = config.allow_cpu_fallback
    args.coreml_det_size = config.coreml_det_size

    # Detection settings
    args.detector = config.detector
    args.det_thresh = config.det_thresh

    # Tracking settings
    args.tracker = config.tracker
    args.stride = config.stride
    args.fps = config.target_fps
    args.track_buffer = config.track_buffer
    args.track_high_thresh = config.track_high_thresh
    args.new_track_thresh = config.new_track_thresh
    args.match_thresh = config.match_thresh
    args.min_box_area = config.min_box_area
    args.max_gap = config.max_gap

    # Scene detection
    args.scene_detector = config.scene_detector
    args.scene_threshold = config.scene_threshold
    args.scene_min_len = config.scene_min_len
    args.scene_warmup_dets = config.scene_warmup_dets

    # Embedding settings
    args.max_samples_per_track = config.max_samples_per_track
    args.min_samples_per_track = config.min_samples_per_track
    args.sample_every_n_frames = config.sample_every_n_frames

    # Clustering settings
    args.cluster_thresh = config.cluster_thresh
    args.min_identity_sim = config.min_identity_sim
    args.min_cluster_size = config.min_cluster_size

    # Export settings
    args.save_frames = config.save_frames
    args.save_crops = config.save_crops
    args.thumb_size = config.thumb_size
    args.jpeg_quality = config.jpeg_quality

    # Output settings
    args.out_root = str(config.data_root) if config.data_root else None
    args.progress_file = str(config.progress_file) if config.progress_file else None

    # Flags for stage selection (these are set by the caller)
    args.faces_embed = False
    args.cluster = False

    # Additional settings with defaults
    args.track_sample_limit = None
    args.quiet = False
    args.verbose = False
    args.emit_manifests = False

    # Gate config defaults
    args.gate_appear_hard = 0.75
    args.gate_appear_soft = 0.82
    args.gate_appear_streak = 2
    args.gate_iou = 0.50
    args.gate_proto_momentum = 0.90
    args.gate_emb_every = 10

    return args


def run_detect_track(
    episode_id: str,
    video_path: str | Path,
    config: "EpisodeRunConfig",
) -> Dict[str, Any]:
    """Run detection and tracking stage.

    Args:
        episode_id: Episode identifier
        video_path: Path to source video
        config: Pipeline configuration

    Returns:
        Dictionary with stage results including:
        - stage: "detect_track"
        - success: bool
        - frames_processed: int
        - tracks: int
        - detections: int
        - artifacts: dict of artifact paths
        - runtime_sec: float
    """
    video_path = Path(video_path)
    started_at = _utcnow_iso()
    start_time = time.time()

    # Set up data root
    if config.data_root:
        os.environ["SCREENALYTICS_DATA_ROOT"] = str(config.data_root)

    # Ensure directories exist
    ensure_artifact_dirs(episode_id, config.data_root)

    try:
        # Import the stage implementation
        from tools.episode_run import _run_detect_track_stage

        # Convert config to args namespace
        args = _config_to_args_namespace(config, episode_id, video_path)

        # Get storage context
        storage, ep_ctx, s3_prefixes = _get_storage_context(episode_id)

        # Run the stage
        summary = _run_detect_track_stage(args, storage, ep_ctx, s3_prefixes)

        runtime_sec = time.time() - start_time
        summary["runtime_sec"] = runtime_sec
        summary["success"] = True

        return summary

    except Exception as exc:
        LOGGER.exception("detect_track failed: %s", exc)
        return {
            "stage": "detect_track",
            "success": False,
            "error": str(exc),
            "started_at": started_at,
            "finished_at": _utcnow_iso(),
            "runtime_sec": time.time() - start_time,
        }


def run_faces_embed(
    episode_id: str,
    video_path: str | Path,
    config: "EpisodeRunConfig",
) -> Dict[str, Any]:
    """Run face embedding stage.

    Args:
        episode_id: Episode identifier
        video_path: Path to source video (needed for frame extraction)
        config: Pipeline configuration

    Returns:
        Dictionary with stage results including:
        - stage: "faces_embed"
        - success: bool
        - faces: int
        - artifacts: dict of artifact paths
        - runtime_sec: float
    """
    video_path = Path(video_path)
    started_at = _utcnow_iso()
    start_time = time.time()

    # Set up data root
    if config.data_root:
        os.environ["SCREENALYTICS_DATA_ROOT"] = str(config.data_root)

    try:
        # Import the stage implementation
        from tools.episode_run import _run_faces_embed_stage

        # Convert config to args namespace
        args = _config_to_args_namespace(config, episode_id, video_path)
        args.faces_embed = True

        # Get storage context
        storage, ep_ctx, s3_prefixes = _get_storage_context(episode_id)

        # Run the stage
        summary = _run_faces_embed_stage(args, storage, ep_ctx, s3_prefixes)

        runtime_sec = time.time() - start_time
        summary["runtime_sec"] = runtime_sec
        summary["success"] = True

        return summary

    except Exception as exc:
        LOGGER.exception("faces_embed failed: %s", exc)
        return {
            "stage": "faces_embed",
            "success": False,
            "error": str(exc),
            "started_at": started_at,
            "finished_at": _utcnow_iso(),
            "runtime_sec": time.time() - start_time,
        }


def run_cluster(
    episode_id: str,
    video_path: str | Path,  # May not be needed but kept for API consistency
    config: "EpisodeRunConfig",
) -> Dict[str, Any]:
    """Run clustering stage.

    Args:
        episode_id: Episode identifier
        video_path: Path to source video (for consistency, may not be used)
        config: Pipeline configuration

    Returns:
        Dictionary with stage results including:
        - stage: "cluster"
        - success: bool
        - identities: int
        - artifacts: dict of artifact paths
        - runtime_sec: float
    """
    video_path = Path(video_path) if video_path else None
    started_at = _utcnow_iso()
    start_time = time.time()

    # Set up data root
    if config.data_root:
        os.environ["SCREENALYTICS_DATA_ROOT"] = str(config.data_root)

    try:
        # Import the stage implementation
        from tools.episode_run import _run_cluster_stage

        # Convert config to args namespace
        args = _config_to_args_namespace(config, episode_id, video_path or Path("."))
        args.cluster = True

        # Get storage context
        storage, ep_ctx, s3_prefixes = _get_storage_context(episode_id)

        # Run the stage
        summary = _run_cluster_stage(args, storage, ep_ctx, s3_prefixes)

        runtime_sec = time.time() - start_time
        summary["runtime_sec"] = runtime_sec
        summary["success"] = True

        return summary

    except Exception as exc:
        LOGGER.exception("cluster failed: %s", exc)
        return {
            "stage": "cluster",
            "success": False,
            "error": str(exc),
            "started_at": started_at,
            "finished_at": _utcnow_iso(),
            "runtime_sec": time.time() - start_time,
        }


def check_artifacts_exist(episode_id: str, stage: str, data_root: Path | None = None) -> Dict[str, bool]:
    """Check which artifacts exist for a given stage.

    Useful for implementing artifact reuse in dev mode.

    Args:
        episode_id: Episode identifier
        stage: Stage name (detect_track, faces_embed, cluster)
        data_root: Optional data root override

    Returns:
        Dictionary mapping artifact name to existence boolean
    """
    result = {}

    if stage == "detect_track":
        artifacts = [ArtifactKind.DETECTIONS, ArtifactKind.TRACKS]
    elif stage == "faces_embed":
        artifacts = [ArtifactKind.FACES, ArtifactKind.FACES_EMBEDDINGS]
    elif stage == "cluster":
        artifacts = [ArtifactKind.IDENTITIES]
    else:
        return result

    for artifact in artifacts:
        path = get_artifact_path(episode_id, artifact, data_root)
        result[artifact.value] = path.exists()

    return result
