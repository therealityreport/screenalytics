"""Screen time episode processing pipeline.

This module provides a UI-agnostic engine for running the face detection,
embedding, and clustering pipeline on TV episodes.
"""

from __future__ import annotations

from py_screenalytics.pipeline.episode_engine import (
    EpisodeRunConfig,
    EpisodeRunResult,
    StageResult,
    run_episode,
    run_stage,
    PipelineStage,
)
from py_screenalytics.pipeline.constants import (
    ARTIFACT_KINDS,
    ArtifactKind,
    get_artifact_path,
    ensure_artifact_dirs,
    PIPELINE_VERSION,
)
from py_screenalytics.pipeline.stages import (
    run_detect_track,
    run_faces_embed,
    run_cluster,
    check_artifacts_exist,
)

__all__ = [
    # Main API
    "run_episode",
    "run_stage",
    # Individual stage functions
    "run_detect_track",
    "run_faces_embed",
    "run_cluster",
    # Config and Results
    "EpisodeRunConfig",
    "EpisodeRunResult",
    "StageResult",
    "PipelineStage",
    # Constants
    "ARTIFACT_KINDS",
    "ArtifactKind",
    "get_artifact_path",
    "ensure_artifact_dirs",
    "PIPELINE_VERSION",
    # Utilities
    "check_artifacts_exist",
]
