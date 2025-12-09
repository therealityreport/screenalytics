"""Pipeline constants and artifact path definitions.

This module defines the canonical IO contract for all pipeline artifacts.
Any changes to artifact locations or schemas MUST be reflected here.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Dict, Literal

# Pipeline version - update when making breaking changes to artifact schema
PIPELINE_VERSION = "2025-12-03"

# Default data root - can be overridden via SCREENALYTICS_DATA_ROOT env var
DEFAULT_DATA_ROOT = Path("data")


def _data_root() -> Path:
    """Return the configured data root directory."""
    raw = os.environ.get("SCREENALYTICS_DATA_ROOT")
    base = Path(raw).expanduser() if raw else DEFAULT_DATA_ROOT
    return base


class ArtifactKind(str, Enum):
    """Canonical artifact types produced by the pipeline."""

    # Video input
    VIDEO = "video"

    # Manifest files (JSONL)
    DETECTIONS = "detections"
    TRACKS = "tracks"
    FACES = "faces"
    IDENTITIES = "identities"
    TRACK_REPS = "track_reps"

    # Frame exports
    FRAMES_ROOT = "frames_root"
    FRAMES_DIR = "frames"
    CROPS_DIR = "crops"
    THUMBS_DIR = "thumbs"

    # Embeddings (numpy arrays)
    FACES_EMBEDDINGS = "faces_embeddings"

    # Run markers / metadata
    RUN_MARKERS = "runs"


# Map of artifact kind -> relative path pattern
# {ep_id} is replaced with the episode identifier
ARTIFACT_PATHS: Dict[ArtifactKind, str] = {
    ArtifactKind.VIDEO: "videos/{ep_id}/episode.mp4",
    ArtifactKind.DETECTIONS: "manifests/{ep_id}/detections.jsonl",
    ArtifactKind.TRACKS: "manifests/{ep_id}/tracks.jsonl",
    ArtifactKind.FACES: "manifests/{ep_id}/faces.jsonl",
    ArtifactKind.IDENTITIES: "manifests/{ep_id}/identities.json",
    ArtifactKind.TRACK_REPS: "manifests/{ep_id}/track_reps.jsonl",
    ArtifactKind.FRAMES_ROOT: "frames/{ep_id}",
    ArtifactKind.FRAMES_DIR: "frames/{ep_id}/frames",
    ArtifactKind.CROPS_DIR: "frames/{ep_id}/crops",
    ArtifactKind.THUMBS_DIR: "frames/{ep_id}/thumbs",
    ArtifactKind.FACES_EMBEDDINGS: "manifests/{ep_id}/faces_embeddings.npy",
    ArtifactKind.RUN_MARKERS: "manifests/{ep_id}/runs",
}

# Legacy string-based kind map for backward compatibility
ARTIFACT_KINDS: Dict[str, str] = {k.value: v for k, v in ARTIFACT_PATHS.items()}


def get_artifact_path(
    ep_id: str,
    kind: ArtifactKind | str,
    data_root: Path | str | None = None,
) -> Path:
    """Return the absolute path for a given artifact kind.

    Args:
        ep_id: Episode identifier (e.g., "rhobh-s05e14")
        kind: Artifact type (ArtifactKind enum or string)
        data_root: Optional override for data root directory

    Returns:
        Absolute path to the artifact

    Raises:
        ValueError: If kind is not recognized
    """
    root = Path(data_root) if data_root else _data_root()

    # Handle both enum and string inputs
    if isinstance(kind, str):
        kind_key = kind.lower()
        if kind_key not in ARTIFACT_KINDS:
            raise ValueError(f"Unknown artifact kind '{kind}'")
        rel_pattern = ARTIFACT_KINDS[kind_key]
    else:
        rel_pattern = ARTIFACT_PATHS[kind]

    rel_path = rel_pattern.format(ep_id=ep_id)
    return root / rel_path


def ensure_artifact_dirs(ep_id: str, data_root: Path | str | None = None) -> None:
    """Create parent directories for all artifacts associated with an episode.

    Args:
        ep_id: Episode identifier
        data_root: Optional override for data root directory
    """
    for kind in [
        ArtifactKind.VIDEO,
        ArtifactKind.DETECTIONS,
        ArtifactKind.FRAMES_ROOT,
        ArtifactKind.RUN_MARKERS,
    ]:
        path = get_artifact_path(ep_id, kind, data_root)
        if kind in {ArtifactKind.FRAMES_ROOT, ArtifactKind.RUN_MARKERS}:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)


# Detection/Tracking defaults
DEFAULT_DETECTOR = "retinaface"
DEFAULT_TRACKER = "bytetrack"
DEFAULT_DEVICE = "auto"

# Model names
RETINAFACE_MODEL_NAME = os.environ.get("RETINAFACE_MODEL", "retinaface_r50_v1")
ARCFACE_MODEL_NAME = os.environ.get("ARCFACE_MODEL", "arcface_r100_v1")

# Threshold defaults
DEFAULT_DET_THRESH = 0.65
DEFAULT_CLUSTER_THRESH = 0.75
DEFAULT_MIN_IDENTITY_SIM = 0.50

# Frame export defaults
DEFAULT_THUMB_SIZE = 256
DEFAULT_JPEG_QUALITY = 85
DEFAULT_MAX_SAMPLES_PER_TRACK = 16
DEFAULT_MIN_SAMPLES_PER_TRACK = 4
DEFAULT_SAMPLE_EVERY_N_FRAMES = 4

# Tracking defaults
DEFAULT_TRACK_BUFFER = 15
DEFAULT_MATCH_THRESH = 0.85
DEFAULT_TRACK_HIGH_THRESH = 0.45
DEFAULT_NEW_TRACK_THRESH = 0.70
# Minimum face bounding box area - lowered from 20.0 to capture smaller faces
DEFAULT_MIN_BOX_AREA = 10.0
