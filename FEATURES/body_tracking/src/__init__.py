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
    - screentime_comparison.json
    - body_metrics.json

Analytics Integration:
    from FEATURES.body_tracking.src import (
        compute_analytics,
        validate_acceptance_metrics,
    )
"""

from .detect_bodies import BodyDetector, detect_bodies
from .track_bodies import BodyTracker, track_bodies
from .body_embeddings import BodyEmbedder, compute_body_embeddings
from .track_fusion import TrackFusion, fuse_face_body_tracks
from .screentime_compare import ScreenTimeComparator, compare_screen_time
from .analytics_integration import (
    EpisodeAnalytics,
    IdentityScreenTime,
    compute_analytics,
    validate_acceptance_metrics,
    run_analytics,
)

__all__ = [
    # Detection
    "BodyDetector",
    "detect_bodies",
    # Tracking
    "BodyTracker",
    "track_bodies",
    # Embeddings
    "BodyEmbedder",
    "compute_body_embeddings",
    # Fusion
    "TrackFusion",
    "fuse_face_body_tracks",
    # Screen time
    "ScreenTimeComparator",
    "compare_screen_time",
    # Analytics
    "EpisodeAnalytics",
    "IdentityScreenTime",
    "compute_analytics",
    "validate_acceptance_metrics",
    "run_analytics",
]
