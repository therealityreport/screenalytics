"""Episode processing engine - UI-agnostic pipeline orchestration.

This module provides the core engine for running the screen-time pipeline.
It is designed to be called from CLI tools, API endpoints, or Streamlit UIs
without any UI-specific dependencies.

Usage:
    from py_screenalytics.pipeline import run_episode, EpisodeRunConfig

    config = EpisodeRunConfig(
        device="auto",
        stride=1,
        save_crops=True,
    )
    result = run_episode("rhobh-s05e14", "/path/to/video.mp4", config)
    print(f"Processed {result.tracks_count} tracks, {result.identities_count} identities")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from py_screenalytics.pipeline.constants import (
    PIPELINE_VERSION,
    ArtifactKind,
    get_artifact_path,
    ensure_artifact_dirs,
    DEFAULT_DETECTOR,
    DEFAULT_TRACKER,
    DEFAULT_DEVICE,
    DEFAULT_DET_THRESH,
    DEFAULT_CLUSTER_THRESH,
    DEFAULT_MIN_IDENTITY_SIM,
    DEFAULT_THUMB_SIZE,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_SAMPLES_PER_TRACK,
    DEFAULT_MIN_SAMPLES_PER_TRACK,
    DEFAULT_SAMPLE_EVERY_N_FRAMES,
    DEFAULT_TRACK_BUFFER,
    DEFAULT_MATCH_THRESH,
    DEFAULT_TRACK_HIGH_THRESH,
    DEFAULT_NEW_TRACK_THRESH,
    DEFAULT_MIN_BOX_AREA,
)
from py_screenalytics import run_layout
from py_screenalytics.episode_status import (
    normalize_stage_key,
    stage_artifacts,
    write_stage_failed,
    write_stage_finished,
    write_stage_started,
)

LOGGER = logging.getLogger("episode_engine")


class PipelineStage(str, Enum):
    """Pipeline stages that can be run individually or in sequence."""

    DETECT_TRACK = "detect_track"
    FACES_EMBED = "faces_embed"
    CLUSTER = "cluster"
    ALL = "all"  # Run all stages in sequence


@dataclass
class EpisodeRunConfig:
    """Configuration for running the episode pipeline.

    This dataclass captures all parameters needed to run the pipeline.
    All fields have sensible defaults so minimal configuration is needed.
    """

    # =========================================================================
    # Device and execution settings
    # =========================================================================
    device: str = DEFAULT_DEVICE
    """Execution device: auto, cpu, cuda, coreml, mps"""

    embed_device: Optional[str] = None
    """Optional separate device for embedding (defaults to device)"""

    allow_cpu_fallback: bool = False
    """If True, falls back to CPU when accelerator unavailable"""

    # =========================================================================
    # Detection settings
    # =========================================================================
    detector: str = DEFAULT_DETECTOR
    """Face detector backend (retinaface)"""

    det_thresh: float = DEFAULT_DET_THRESH
    """Detection confidence threshold (0-1)"""

    coreml_det_size: Optional[Tuple[int, int]] = None
    """Optional CoreML detection size override (width, height)"""

    # =========================================================================
    # Tracking settings
    # =========================================================================
    tracker: str = DEFAULT_TRACKER
    """Tracker backend (bytetrack, strongsort)"""

    stride: int = 1
    """Frame stride for detection (1 = every frame)"""

    target_fps: Optional[float] = None
    """Optional target FPS for downsampling"""

    track_buffer: int = DEFAULT_TRACK_BUFFER
    """ByteTrack track buffer (frames to keep lost tracks)"""

    track_high_thresh: float = DEFAULT_TRACK_HIGH_THRESH
    """ByteTrack high detection threshold"""

    new_track_thresh: float = DEFAULT_NEW_TRACK_THRESH
    """ByteTrack new track threshold"""

    match_thresh: float = DEFAULT_MATCH_THRESH
    """ByteTrack matching threshold"""

    min_box_area: float = DEFAULT_MIN_BOX_AREA
    """Minimum bounding box area for tracking"""

    max_gap: int = 60
    """Maximum frame gap before splitting a track"""

    # =========================================================================
    # Scene detection settings
    # =========================================================================
    scene_detector: str = "pyscenedetect"
    """Scene detector: pyscenedetect, internal, off"""

    scene_threshold: float = 27.0
    """Scene cut threshold"""

    scene_min_len: int = 12
    """Minimum frames between scene cuts"""

    scene_warmup_dets: int = 3
    """Frames of forced detection after each cut"""

    # =========================================================================
    # Embedding settings
    # =========================================================================
    max_samples_per_track: int = DEFAULT_MAX_SAMPLES_PER_TRACK
    """Maximum face samples to embed per track"""

    min_samples_per_track: int = DEFAULT_MIN_SAMPLES_PER_TRACK
    """Minimum samples if track is long enough"""

    sample_every_n_frames: int = DEFAULT_SAMPLE_EVERY_N_FRAMES
    """Sample interval for per-track sampling"""

    # =========================================================================
    # Clustering settings
    # =========================================================================
    cluster_thresh: float = DEFAULT_CLUSTER_THRESH
    """Clustering similarity threshold"""

    min_identity_sim: float = DEFAULT_MIN_IDENTITY_SIM
    """Minimum similarity to identity centroid"""

    min_cluster_size: int = 1
    """Minimum tracks per identity cluster"""

    # =========================================================================
    # Export settings
    # =========================================================================
    save_frames: bool = False
    """Save full frame JPGs"""

    save_crops: bool = False
    """Save per-track face crops"""

    thumb_size: int = DEFAULT_THUMB_SIZE
    """Square thumbnail size for face crops"""

    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    """JPEG quality for exports (1-100)"""

    # =========================================================================
    # Output settings
    # =========================================================================
    data_root: Optional[Path] = None
    """Override data root directory"""

    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    """Optional callback for progress updates"""

    progress_file: Optional[Path] = None
    """Optional file to write progress JSON"""

    # =========================================================================
    # Run scoping
    # =========================================================================
    run_id: Optional[str] = None
    """Run-scoped identifier (generated when omitted)"""

    # =========================================================================
    # Stage selection
    # =========================================================================
    stages: PipelineStage = PipelineStage.ALL
    """Which stages to run"""

    # =========================================================================
    # Dev-mode options (for faster iteration)
    # =========================================================================
    reuse_detections: bool = False
    """If True and detections/tracks exist, skip detect_track stage"""

    reuse_embeddings: bool = False
    """If True and face embeddings exist, skip faces_embed stage"""

    skip_disk_check: bool = False
    """If True, skip disk space pre-check (warn instead of error)"""

    force_recluster: bool = True
    """Always rerun clustering (for threshold tuning). Default True."""

    resume: bool = False
    """If True, resume from last checkpoint (skip already completed stages)"""

    checkpoint_file: Optional[Path] = None
    """Custom path for checkpoint file (default: checkpoint.json in episode manifests dir)"""

    def __post_init__(self) -> None:
        """Validate and normalize configuration values."""
        # Normalize device strings
        self.device = self.device.lower() if self.device else DEFAULT_DEVICE
        if self.embed_device:
            self.embed_device = self.embed_device.lower()

        # Clamp thresholds to valid ranges
        self.det_thresh = max(0.0, min(1.0, self.det_thresh))
        self.cluster_thresh = max(0.0, min(0.999, self.cluster_thresh))
        self.min_identity_sim = max(0.0, min(0.99, self.min_identity_sim))
        self.track_high_thresh = max(0.0, min(1.0, self.track_high_thresh))
        self.new_track_thresh = max(0.0, min(1.0, self.new_track_thresh))
        self.match_thresh = max(0.0, min(1.0, self.match_thresh))

        # Ensure positive integers
        self.stride = max(1, self.stride)
        self.track_buffer = max(1, self.track_buffer)
        self.max_gap = max(1, self.max_gap)
        self.max_samples_per_track = max(1, self.max_samples_per_track)
        self.min_samples_per_track = max(1, self.min_samples_per_track)
        self.sample_every_n_frames = max(1, self.sample_every_n_frames)
        self.min_cluster_size = max(1, self.min_cluster_size)
        self.thumb_size = max(16, self.thumb_size)
        self.jpeg_quality = max(1, min(100, self.jpeg_quality))
        self.min_box_area = max(0.0, self.min_box_area)

        # Convert data_root to Path if string
        if isinstance(self.data_root, str):
            self.data_root = Path(self.data_root).expanduser()

        if self.run_id is not None:
            self.run_id = run_layout.normalize_run_id(str(self.run_id))


@dataclass
class StageResult:
    """Result from running a single pipeline stage."""

    stage: str
    """Stage name (detect_track, faces_embed, cluster)"""

    success: bool
    """Whether the stage completed successfully"""

    started_at: str
    """ISO timestamp when stage started"""

    finished_at: str
    """ISO timestamp when stage finished"""

    runtime_sec: float
    """Runtime in seconds"""

    # Stage-specific counts
    frames_processed: int = 0
    detections_count: int = 0
    tracks_count: int = 0
    faces_count: int = 0
    identities_count: int = 0

    # Device info
    device: Optional[str] = None
    resolved_device: Optional[str] = None

    # Artifacts produced
    artifacts: Dict[str, str] = field(default_factory=dict)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Error info if failed
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stage": self.stage,
            "success": self.success,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "runtime_sec": round(self.runtime_sec, 2),
            "frames_processed": self.frames_processed,
            "detections_count": self.detections_count,
            "tracks_count": self.tracks_count,
            "faces_count": self.faces_count,
            "identities_count": self.identities_count,
            "device": self.device,
            "resolved_device": self.resolved_device,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
            "error": self.error,
        }


@dataclass
class EpisodeRunResult:
    """Result from running the full episode pipeline.

    Contains aggregate metrics and per-stage results.
    """

    episode_id: str
    """Episode identifier"""

    success: bool
    """Whether all stages completed successfully"""

    started_at: str
    """ISO timestamp when pipeline started"""

    finished_at: str
    """ISO timestamp when pipeline finished"""

    runtime_sec: float
    """Total runtime in seconds"""

    run_id: Optional[str] = None
    """Run identifier for run-scoped artifacts"""

    # Aggregate counts from final stage
    frames_processed: int = 0
    tracks_count: int = 0
    faces_count: int = 0
    identities_count: int = 0

    # Pipeline configuration
    config: Optional[EpisodeRunConfig] = None

    # Per-stage results
    stages: List[StageResult] = field(default_factory=list)

    # Final artifacts
    artifacts: Dict[str, str] = field(default_factory=dict)

    # Pipeline version
    version: str = PIPELINE_VERSION

    # Error info if failed
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "success": self.success,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "runtime_sec": round(self.runtime_sec, 2),
            "frames_processed": self.frames_processed,
            "tracks_count": self.tracks_count,
            "faces_count": self.faces_count,
            "identities_count": self.identities_count,
            "stages": [s.to_dict() for s in self.stages],
            "artifacts": self.artifacts,
            "version": self.version,
            "error": self.error,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def _utcnow_iso() -> str:
    """Return current UTC time as ISO format string."""
    return datetime.now(timezone.utc).isoformat()


def _status_artifact_paths(ep_id: str, run_id: str, stage_key: str) -> Dict[str, str]:
    normalized = normalize_stage_key(stage_key) or stage_key
    try:
        artifacts = stage_artifacts(ep_id, run_id, normalized)
    except Exception:
        return {}
    paths: Dict[str, str] = {}
    for entry in artifacts:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        path = entry.get("path")
        if isinstance(label, str) and isinstance(path, str) and entry.get("exists"):
            paths[label] = path
    return paths


def _stage_counts(stage: PipelineStage, summary: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if stage == PipelineStage.DETECT_TRACK:
        detections = summary.get("detections")
        tracks = summary.get("tracks")
        if isinstance(detections, int):
            counts["detections"] = detections
        if isinstance(tracks, int):
            counts["tracks"] = tracks
    elif stage == PipelineStage.FACES_EMBED:
        faces = summary.get("faces")
        if isinstance(faces, int):
            counts["faces"] = faces
    elif stage == PipelineStage.CLUSTER:
        identities = summary.get("identities_count", summary.get("identities"))
        faces = summary.get("faces_count", summary.get("faces"))
        if isinstance(identities, int):
            counts["identities"] = identities
        if isinstance(faces, int):
            counts["faces"] = faces
    return counts


# Disk space constants
MIN_DISK_SPACE_GB = 5.0  # Minimum free space required to start
SPACE_MULTIPLIER = 3.0  # Estimated output is ~3x video size (crops, frames, manifests)


def _check_disk_space(
    video_path: Path,
    data_root: Optional[Path] = None,
    warn_only: bool = False,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Check if there's enough disk space for processing.

    Estimates required space based on video file size and checks available space
    on the target filesystem (data_root or current directory).

    Args:
        video_path: Path to the source video file
        data_root: Optional data root directory (defaults to current directory)
        warn_only: If True, returns warning instead of error

    Returns:
        Tuple of (ok, message, details):
        - ok: True if sufficient space
        - message: None if ok, otherwise error/warning message
        - details: Dict with space information for logging
    """
    # Determine check path (use data_root or video parent directory)
    check_path = data_root if data_root else video_path.parent
    if not check_path.exists():
        check_path = Path.cwd()

    try:
        # Get disk usage statistics
        usage = shutil.disk_usage(check_path)
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)

        # Estimate required space from video size
        video_size_gb = video_path.stat().st_size / (1024**3) if video_path.exists() else 0.0
        estimated_required_gb = max(video_size_gb * SPACE_MULTIPLIER, MIN_DISK_SPACE_GB)

        details = {
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
            "video_size_gb": round(video_size_gb, 2),
            "estimated_required_gb": round(estimated_required_gb, 2),
            "check_path": str(check_path),
        }

        if free_gb < MIN_DISK_SPACE_GB:
            message = (
                f"Critically low disk space: {free_gb:.1f}GB available, "
                f"minimum {MIN_DISK_SPACE_GB:.1f}GB required. "
                f"Free up space on {check_path} before processing."
            )
            return False, message, details

        if free_gb < estimated_required_gb:
            message = (
                f"Potentially insufficient disk space: {free_gb:.1f}GB available, "
                f"estimated {estimated_required_gb:.1f}GB needed for video ({video_size_gb:.1f}GB). "
                f"Processing may fail partway through."
            )
            if warn_only:
                LOGGER.warning(message)
                return True, message, details
            return False, message, details

        return True, None, details

    except OSError as exc:
        LOGGER.warning("Could not check disk space: %s", exc)
        return True, None, {"error": str(exc)}  # Proceed on check failure


# ============================================================================
# Checkpoint/Resume helpers
# ============================================================================

def _get_checkpoint_path(
    episode_id: str,
    data_root: Optional[Path] = None,
    custom_path: Optional[Path] = None,
) -> Path:
    """Get the path to the checkpoint file for an episode."""
    if custom_path:
        return custom_path
    # Use the episode manifests directory
    manifests_dir = get_artifact_path(episode_id, ArtifactKind.MANIFESTS, data_root)
    return manifests_dir / "checkpoint.json"


def _load_checkpoint(
    episode_id: str,
    data_root: Optional[Path] = None,
    custom_path: Optional[Path] = None,
) -> dict[str, Any] | None:
    """Load checkpoint for an episode if it exists.

    Returns:
        Checkpoint dict with keys like 'completed_stages', 'last_stage', 'started_at',
        or None if no checkpoint exists.
    """
    checkpoint_path = _get_checkpoint_path(episode_id, data_root, custom_path)
    if not checkpoint_path.exists():
        return None

    try:
        with checkpoint_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            LOGGER.info(
                "Loaded checkpoint for %s: completed stages = %s",
                episode_id,
                data.get("completed_stages", []),
            )
            return data
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Failed to load checkpoint for %s: %s", episode_id, exc)
        return None


def _save_checkpoint(
    episode_id: str,
    completed_stages: list[str],
    last_stage: str,
    data_root: Optional[Path] = None,
    custom_path: Optional[Path] = None,
) -> None:
    """Save checkpoint after a successful stage.

    Args:
        episode_id: Episode identifier
        completed_stages: List of completed stage names
        last_stage: Name of the most recently completed stage
        data_root: Optional data root directory
        custom_path: Optional custom checkpoint file path
    """
    checkpoint_path = _get_checkpoint_path(episode_id, data_root, custom_path)

    checkpoint_data = {
        "episode_id": episode_id,
        "completed_stages": completed_stages,
        "last_stage": last_stage,
        "updated_at": _utcnow_iso(),
        "pipeline_version": PIPELINE_VERSION,
    }

    try:
        # Ensure directory exists
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)
        LOGGER.debug("Saved checkpoint: %s completed", completed_stages)
    except OSError as exc:
        LOGGER.warning("Failed to save checkpoint: %s", exc)


def _clear_checkpoint(
    episode_id: str,
    data_root: Optional[Path] = None,
    custom_path: Optional[Path] = None,
) -> None:
    """Clear checkpoint file after successful completion."""
    checkpoint_path = _get_checkpoint_path(episode_id, data_root, custom_path)
    try:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            LOGGER.debug("Cleared checkpoint for %s", episode_id)
    except OSError as exc:
        LOGGER.warning("Failed to clear checkpoint: %s", exc)


def run_stage(
    episode_id: str,
    video_path: str | Path,
    config: EpisodeRunConfig,
    stage: PipelineStage,
) -> StageResult:
    """Run a single pipeline stage.

    This function provides a clean interface to run individual stages.
    It calls the stage implementations in py_screenalytics.pipeline.stages.

    Args:
        episode_id: Episode identifier (e.g., "rhobh-s05e14")
        video_path: Path to source video file
        config: Pipeline configuration
        stage: Which stage to run

    Returns:
        StageResult with stage metrics and artifacts
    """
    if stage == PipelineStage.ALL:
        raise ValueError("Use run_episode() for running all stages")

    video_path = Path(video_path)
    started_at = _utcnow_iso()
    start_time = time.time()

    try:
        config.run_id = run_layout.get_or_create_run_id(episode_id, config.run_id)
    except ValueError as exc:
        return StageResult(
            stage=stage.value,
            success=False,
            started_at=started_at,
            finished_at=_utcnow_iso(),
            runtime_sec=time.time() - start_time,
            error=str(exc),
        )

    # Set up data root environment variable
    if config.data_root:
        os.environ["SCREENALYTICS_DATA_ROOT"] = str(config.data_root)

    # Ensure directories exist
    ensure_artifact_dirs(episode_id, config.data_root)

    if config.run_id:
        try:
            write_stage_started(episode_id, config.run_id, stage.value)
        except Exception as exc:  # pragma: no cover - best effort status update
            LOGGER.warning("[episode_status] Failed to mark %s start: %s", stage.value, exc)

    try:
        # Import stage implementations from the stages module
        from py_screenalytics.pipeline.stages import (
            run_detect_track,
            run_faces_embed,
            run_cluster,
        )

        # Run the appropriate stage
        if stage == PipelineStage.DETECT_TRACK:
            summary = run_detect_track(episode_id, video_path, config)
        elif stage == PipelineStage.FACES_EMBED:
            summary = run_faces_embed(episode_id, video_path, config)
        elif stage == PipelineStage.CLUSTER:
            summary = run_cluster(episode_id, video_path, config)
        else:
            raise ValueError(f"Unknown stage: {stage}")

        finished_at = _utcnow_iso()
        runtime_sec = time.time() - start_time

        # Check for stage-level failure
        if not summary.get("success", True):
            if config.run_id:
                try:
                    write_stage_failed(
                        episode_id,
                        config.run_id,
                        stage.value,
                        error_code="stage_failed",
                        error_message=str(summary.get("error", "Unknown error")),
                    )
                except Exception as status_exc:  # pragma: no cover - best effort status update
                    LOGGER.warning("[episode_status] Failed to mark %s failure: %s", stage.value, status_exc)
            return StageResult(
                stage=stage.value,
                success=False,
                started_at=started_at,
                finished_at=finished_at,
                runtime_sec=runtime_sec,
                error=summary.get("error", "Unknown error"),
            )

        # Extract results from summary
        if config.run_id:
            try:
                counts = _stage_counts(stage, summary)
                write_stage_finished(
                    episode_id,
                    config.run_id,
                    stage.value,
                    counts=counts,
                    metrics=counts or None,
                    artifact_paths=_status_artifact_paths(episode_id, config.run_id, stage.value),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark %s success: %s", stage.value, status_exc)
        return StageResult(
            stage=stage.value,
            success=True,
            started_at=started_at,
            finished_at=finished_at,
            runtime_sec=runtime_sec,
            frames_processed=summary.get("frames_processed", 0),
            detections_count=summary.get("detections", 0),
            tracks_count=summary.get("tracks", 0),
            faces_count=summary.get("faces", 0),
            identities_count=summary.get("identities", 0),
            device=summary.get("device"),
            resolved_device=summary.get("resolved_device"),
            artifacts=summary.get("artifacts", {}).get("local", {}),
            metadata=summary,
        )

    except Exception as exc:
        finished_at = _utcnow_iso()
        runtime_sec = time.time() - start_time
        LOGGER.exception("Stage %s failed: %s", stage.value, exc)

        if config.run_id:
            try:
                write_stage_failed(
                    episode_id,
                    config.run_id,
                    stage.value,
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[episode_status] Failed to mark %s failure: %s", stage.value, status_exc)
        return StageResult(
            stage=stage.value,
            success=False,
            started_at=started_at,
            finished_at=finished_at,
            runtime_sec=runtime_sec,
            error=str(exc),
        )


def _load_metrics_from_artifacts(
    episode_id: str,
    data_root: Optional[Path] = None,
) -> Dict[str, int]:
    """Load aggregate metrics from existing artifacts.

    This is used when stages are skipped due to reuse flags, to ensure
    EpisodeRunResult still has correct metric counts.

    Args:
        episode_id: Episode identifier
        data_root: Optional data root override

    Returns:
        Dictionary with tracks_count, faces_count, identities_count
    """
    metrics: Dict[str, int] = {}

    # Load tracks count from tracks.jsonl
    tracks_path = get_artifact_path(episode_id, ArtifactKind.TRACKS, data_root)
    if tracks_path.exists():
        try:
            with tracks_path.open("r", encoding="utf-8") as f:
                metrics["tracks_count"] = sum(1 for line in f if line.strip())
        except Exception:
            pass

    # Load faces count from faces.jsonl
    faces_path = get_artifact_path(episode_id, ArtifactKind.FACES, data_root)
    if faces_path.exists():
        try:
            with faces_path.open("r", encoding="utf-8") as f:
                metrics["faces_count"] = sum(1 for line in f if line.strip())
        except Exception:
            pass

    # Load identities count from identities.json
    identities_path = get_artifact_path(episode_id, ArtifactKind.IDENTITIES, data_root)
    if identities_path.exists():
        try:
            with identities_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                identities_list = data.get("identities", [])
                metrics["identities_count"] = len(identities_list)
        except Exception:
            pass

    return metrics


def run_episode(
    episode_id: str,
    video_path: str | Path,
    config: Optional[EpisodeRunConfig] = None,
) -> EpisodeRunResult:
    """Run the complete episode processing pipeline.

    This is the main entry point for processing an episode. It orchestrates
    all pipeline stages (detect_track → faces_embed → cluster) and returns
    a comprehensive result object.

    Args:
        episode_id: Episode identifier (e.g., "rhobh-s05e14")
        video_path: Path to source video file
        config: Optional pipeline configuration (uses defaults if not provided)

    Returns:
        EpisodeRunResult with aggregate metrics and per-stage results

    Example:
        >>> from py_screenalytics.pipeline import run_episode, EpisodeRunConfig
        >>> config = EpisodeRunConfig(device="coreml", save_crops=True)
        >>> result = run_episode("rhobh-s05e14", "/path/to/video.mp4", config)
        >>> print(f"Found {result.identities_count} identities")
    """
    if config is None:
        config = EpisodeRunConfig()

    video_path = Path(video_path)
    started_at = _utcnow_iso()
    start_time = time.time()

    try:
        config.run_id = run_layout.get_or_create_run_id(episode_id, config.run_id)
    except ValueError as exc:
        return EpisodeRunResult(
            episode_id=episode_id,
            run_id=config.run_id,
            success=False,
            started_at=started_at,
            finished_at=_utcnow_iso(),
            runtime_sec=time.time() - start_time,
            config=config,
            error=str(exc),
        )

    LOGGER.info("Pipeline run_id=%s for episode %s", config.run_id, episode_id)

    # Validate video path
    if not video_path.exists():
        return EpisodeRunResult(
            episode_id=episode_id,
            run_id=config.run_id,
            success=False,
            started_at=started_at,
            finished_at=_utcnow_iso(),
            runtime_sec=time.time() - start_time,
            config=config,
            error=f"Video file not found: {video_path}",
        )

    # Pre-flight disk space check
    space_ok, space_msg, space_details = _check_disk_space(
        video_path,
        data_root=config.data_root,
        warn_only=config.skip_disk_check,
    )
    if space_details:
        LOGGER.info(
            "Disk space check: %.1fGB free of %.1fGB total, "
            "estimated %.1fGB needed for %.1fGB video",
            space_details.get("free_gb", 0),
            space_details.get("total_gb", 0),
            space_details.get("estimated_required_gb", 0),
            space_details.get("video_size_gb", 0),
        )
    if not space_ok:
        return EpisodeRunResult(
            episode_id=episode_id,
            run_id=config.run_id,
            success=False,
            started_at=started_at,
            finished_at=_utcnow_iso(),
            runtime_sec=time.time() - start_time,
            config=config,
            error=space_msg or "Insufficient disk space",
        )

    # Set up data root
    if config.data_root:
        os.environ["SCREENALYTICS_DATA_ROOT"] = str(config.data_root)

    # Ensure directories exist
    ensure_artifact_dirs(episode_id, config.data_root)

    stage_results: List[StageResult] = []
    final_artifacts: Dict[str, str] = {}
    completed_stages: List[str] = []

    # Load checkpoint if resume enabled
    checkpoint_data: dict[str, Any] | None = None
    if config.resume:
        checkpoint_data = _load_checkpoint(
            episode_id, config.data_root, config.checkpoint_file
        )
        if checkpoint_data:
            completed_stages = checkpoint_data.get("completed_stages", [])
            LOGGER.info(
                "Resuming from checkpoint: %d stages already completed",
                len(completed_stages),
            )

    # Determine which stages to run
    if config.stages == PipelineStage.ALL:
        stages_to_run = [
            PipelineStage.DETECT_TRACK,
            PipelineStage.FACES_EMBED,
            PipelineStage.CLUSTER,
        ]
    else:
        stages_to_run = [config.stages]

    # Run each stage
    for stage in stages_to_run:
        # Check if we should skip this stage due to reuse flags or checkpoint
        should_skip = False
        skip_reason = None

        # Check checkpoint first (if resuming)
        if config.resume and stage.value in completed_stages:
            should_skip = True
            skip_reason = f"already completed (checkpoint resume)"

        elif stage == PipelineStage.DETECT_TRACK and config.reuse_detections:
            # Check if detections and tracks exist
            from py_screenalytics.pipeline.stages import check_artifacts_exist
            artifacts = check_artifacts_exist(episode_id, "detect_track", config.data_root)
            if artifacts.get("detections") and artifacts.get("tracks"):
                should_skip = True
                skip_reason = "reuse_detections enabled and artifacts exist"

        elif stage == PipelineStage.FACES_EMBED and config.reuse_embeddings:
            # Check if face embeddings exist
            from py_screenalytics.pipeline.stages import check_artifacts_exist
            artifacts = check_artifacts_exist(episode_id, "faces_embed", config.data_root)
            if artifacts.get("faces") and artifacts.get("faces_embeddings"):
                should_skip = True
                skip_reason = "reuse_embeddings enabled and artifacts exist"

        elif stage == PipelineStage.CLUSTER and not config.force_recluster:
            # Check if clustering artifacts exist and force_recluster is False
            from py_screenalytics.pipeline.stages import check_artifacts_exist
            artifacts = check_artifacts_exist(episode_id, "cluster", config.data_root)
            if artifacts.get("identities"):
                should_skip = True
                skip_reason = "force_recluster=False and identities.json exists"

        if should_skip:
            LOGGER.info("Skipping stage %s: %s", stage.value, skip_reason)
            # Create a skipped stage result
            stage_results.append(StageResult(
                stage=stage.value,
                success=True,
                started_at=_utcnow_iso(),
                finished_at=_utcnow_iso(),
                runtime_sec=0.0,
                metadata={"skipped": True, "reason": skip_reason},
            ))
            if config.progress_callback:
                config.progress_callback({
                    "type": "stage_skipped",
                    "stage": stage.value,
                    "episode_id": episode_id,
                    "reason": skip_reason,
                })
            continue

        LOGGER.info("Running stage: %s for episode %s", stage.value, episode_id)

        # Emit progress callback if configured
        if config.progress_callback:
            config.progress_callback({
                "type": "stage_start",
                "stage": stage.value,
                "episode_id": episode_id,
            })

        result = run_stage(episode_id, video_path, config, stage)
        stage_results.append(result)

        # Update final artifacts
        if result.artifacts:
            final_artifacts.update(result.artifacts)

        # Emit progress callback
        if config.progress_callback:
            config.progress_callback({
                "type": "stage_complete",
                "stage": stage.value,
                "episode_id": episode_id,
                "success": result.success,
                "runtime_sec": result.runtime_sec,
            })

        # Stop if stage failed
        if not result.success:
            LOGGER.error("Stage %s failed, stopping pipeline", stage.value)
            break

        # Save checkpoint after successful stage
        completed_stages.append(stage.value)
        _save_checkpoint(
            episode_id,
            completed_stages,
            stage.value,
            config.data_root,
            config.checkpoint_file,
        )

    finished_at = _utcnow_iso()
    runtime_sec = time.time() - start_time

    # Aggregate metrics from appropriate stages (not just the last one)
    # Each metric comes from the stage that actually produces it:
    # - frames_processed, detections_count, tracks_count: detect_track stage
    # - faces_count: faces_embed stage
    # - identities_count: cluster stage
    aggregated_frames = 0
    aggregated_tracks = 0
    aggregated_faces = 0
    aggregated_identities = 0

    for result in stage_results:
        if result.stage == PipelineStage.DETECT_TRACK.value:
            if result.frames_processed > 0:
                aggregated_frames = result.frames_processed
            if result.tracks_count > 0:
                aggregated_tracks = result.tracks_count
        elif result.stage == PipelineStage.FACES_EMBED.value:
            if result.faces_count > 0:
                aggregated_faces = result.faces_count
        elif result.stage == PipelineStage.CLUSTER.value:
            if result.identities_count > 0:
                aggregated_identities = result.identities_count
            # Clustering may also report faces if it read them from artifacts
            if result.faces_count > 0 and aggregated_faces == 0:
                aggregated_faces = result.faces_count

    # If stages were skipped due to reuse flags, try to load metrics from artifacts
    if aggregated_tracks == 0 or aggregated_faces == 0 or aggregated_identities == 0:
        try:
            loaded_metrics = _load_metrics_from_artifacts(episode_id, config.data_root)
            if aggregated_tracks == 0 and loaded_metrics.get("tracks_count", 0) > 0:
                aggregated_tracks = loaded_metrics["tracks_count"]
            if aggregated_faces == 0 and loaded_metrics.get("faces_count", 0) > 0:
                aggregated_faces = loaded_metrics["faces_count"]
            if aggregated_identities == 0 and loaded_metrics.get("identities_count", 0) > 0:
                aggregated_identities = loaded_metrics["identities_count"]
        except Exception as exc:
            LOGGER.debug("Could not load metrics from artifacts: %s", exc)

    last_result = stage_results[-1] if stage_results else None
    all_success = all(r.success for r in stage_results)

    # Clear checkpoint after successful completion of all stages
    if all_success and len(completed_stages) == len(stages_to_run):
        _clear_checkpoint(episode_id, config.data_root, config.checkpoint_file)
        LOGGER.info("Pipeline completed successfully, checkpoint cleared")

    return EpisodeRunResult(
        episode_id=episode_id,
        run_id=config.run_id,
        success=all_success,
        started_at=started_at,
        finished_at=finished_at,
        runtime_sec=runtime_sec,
        frames_processed=aggregated_frames,
        tracks_count=aggregated_tracks,
        faces_count=aggregated_faces,
        identities_count=aggregated_identities,
        config=config,
        stages=stage_results,
        artifacts=final_artifacts,
        error=last_result.error if last_result and not last_result.success else None,
    )
