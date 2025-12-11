"""
Body Tracking Sandbox

Experimental body detection, tracking, and Re-ID fusion pipeline.
This is a sandboxed implementation that does not modify the production faces pipeline.

Usage:
    python -m FEATURES.body_tracking --episode-id EP_ID

Artifacts produced (under data/manifests/{ep_id}/body_tracking/):
    - body_detections.jsonl
    - body_tracks.jsonl
    - body_embeddings.npy + body_embeddings_meta.json
    - track_fusion.json
    - body_metrics.json
"""

from .detect_bodies import BodyDetector, detect_bodies
from .track_bodies import BodyTracker, track_bodies
from .body_embeddings import BodyEmbedder, compute_body_embeddings
from .track_fusion import TrackFusion, fuse_face_body_tracks
from .screentime_compare import ScreenTimeComparator, compare_screen_time

__all__ = [
    "BodyDetector",
    "detect_bodies",
    "BodyTracker",
    "track_bodies",
    "BodyEmbedder",
    "compute_body_embeddings",
    "TrackFusion",
    "fuse_face_body_tracks",
    "ScreenTimeComparator",
    "compare_screen_time",
]
