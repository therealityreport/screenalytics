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

    force_recluster: bool = True
    """Always rerun clustering (for threshold tuning). Default True."""

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

    # Set up data root environment variable
    if config.data_root:
        os.environ["SCREENALYTICS_DATA_ROOT"] = str(config.data_root)

    # Ensure directories exist
    ensure_artifact_dirs(episode_id, config.data_root)

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
            return StageResult(
                stage=stage.value,
                success=False,
                started_at=started_at,
                finished_at=finished_at,
                runtime_sec=runtime_sec,
                error=summary.get("error", "Unknown error"),
            )

        # Extract results from summary
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

        return StageResult(
            stage=stage.value,
            success=False,
            started_at=started_at,
            finished_at=finished_at,
            runtime_sec=runtime_sec,
            error=str(exc),
        )


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

    # Validate video path
    if not video_path.exists():
        return EpisodeRunResult(
            episode_id=episode_id,
            success=False,
            started_at=started_at,
            finished_at=_utcnow_iso(),
            runtime_sec=time.time() - start_time,
            config=config,
            error=f"Video file not found: {video_path}",
        )

    # Set up data root
    if config.data_root:
        os.environ["SCREENALYTICS_DATA_ROOT"] = str(config.data_root)

    # Ensure directories exist
    ensure_artifact_dirs(episode_id, config.data_root)

    stage_results: List[StageResult] = []
    final_artifacts: Dict[str, str] = {}

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
        # Check if we should skip this stage due to reuse flags
        should_skip = False
        skip_reason = None

        if stage == PipelineStage.DETECT_TRACK and config.reuse_detections:
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

    finished_at = _utcnow_iso()
    runtime_sec = time.time() - start_time

    # Aggregate results from final stage
    last_result = stage_results[-1] if stage_results else None
    all_success = all(r.success for r in stage_results)

    return EpisodeRunResult(
        episode_id=episode_id,
        success=all_success,
        started_at=started_at,
        finished_at=finished_at,
        runtime_sec=runtime_sec,
        frames_processed=last_result.frames_processed if last_result else 0,
        tracks_count=last_result.tracks_count if last_result else 0,
        faces_count=last_result.faces_count if last_result else 0,
        identities_count=last_result.identities_count if last_result else 0,
        config=config,
        stages=stage_results,
        artifacts=final_artifacts,
        error=last_result.error if last_result and not last_result.success else None,
    )
