"""
Face Alignment Sandbox

FAN-based face alignment with 68-point landmarks for improved embedding quality.
Future: LUVLi quality gating and 3DDFA_V2 head pose estimation.

Usage:
    python -m FEATURES.face_alignment --episode-id EP_ID

Artifacts produced (under data/manifests/{ep_id}/face_alignment/):
    - aligned_faces.jsonl
    - aligned_crops/ (optional)

Pipeline Integration:
    from FEATURES.face_alignment.src import (
        run_face_alignment_stage,
        is_face_alignment_enabled,
    )
"""

from .run_fan_alignment import (
    FANAligner,
    AlignedFace,
    run_fan_alignment,
    get_5_point_landmarks,
    align_face_crop,
)
from .load_detections import load_face_detections, load_face_tracks
from .export_aligned_faces import export_aligned_faces, compute_alignment_stats
from .pipeline_integration import (
    load_face_alignment_config,
    is_face_alignment_enabled,
    run_face_alignment_stage,
    load_aligned_crops_for_embedding,
)

__all__ = [
    # Core alignment
    "FANAligner",
    "AlignedFace",
    "run_fan_alignment",
    "get_5_point_landmarks",
    "align_face_crop",
    # Loading
    "load_face_detections",
    "load_face_tracks",
    # Export
    "export_aligned_faces",
    "compute_alignment_stats",
    # Pipeline integration
    "load_face_alignment_config",
    "is_face_alignment_enabled",
    "run_face_alignment_stage",
    "load_aligned_crops_for_embedding",
]
