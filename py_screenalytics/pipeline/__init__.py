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
    get_artifact_path,
    PIPELINE_VERSION,
)

__all__ = [
    # Main API
    "run_episode",
    "run_stage",
    # Config and Results
    "EpisodeRunConfig",
    "EpisodeRunResult",
    "StageResult",
    "PipelineStage",
    # Constants
    "ARTIFACT_KINDS",
    "get_artifact_path",
    "PIPELINE_VERSION",
]
