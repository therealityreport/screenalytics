"""Pipeline diagnostics service.

Collects diagnostic data at each pipeline stage and generates
AI-powered explanations for why faces weren't detected, tracked,
embedded, or clustered.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths and config loading
# ---------------------------------------------------------------------------

def _manifests_dir(ep_id: str) -> Path:
    """Get manifests directory for episode."""
    from apps.api.services.storage import get_path
    return get_path(ep_id, "detections").parent


def _diagnostics_dir(ep_id: str) -> Path:
    """Get diagnostics subdirectory for episode."""
    path = _manifests_dir(ep_id) / "diagnostics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_config(config_name: str) -> Dict[str, Any]:
    """Load a pipeline config YAML file."""
    config_path = Path("config/pipeline") / f"{config_name}.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_pipeline_thresholds() -> Dict[str, Any]:
    """Load all relevant pipeline thresholds for diagnostics."""
    embed_config = _load_config("faces_embed_sampling")
    cluster_config = _load_config("clustering")

    quality_gating = embed_config.get("quality_gating", {})

    return {
        "embedding": {
            "min_blur_score": quality_gating.get("min_blur_score", 18.0),
            "min_confidence": quality_gating.get("min_confidence", 0.45),
            "min_quality_score": quality_gating.get("min_quality_score", 1.5),
            "min_std": quality_gating.get("min_std", 0.5),
            "max_yaw_angle": quality_gating.get("max_yaw_angle", 60.0),
            "max_pitch_angle": quality_gating.get("max_pitch_angle", 45.0),
        },
        "clustering": {
            "cluster_thresh": cluster_config.get("cluster_thresh", 0.52),
            "min_identity_sim": cluster_config.get("min_identity_sim", 0.45),
            "min_cluster_size": cluster_config.get("min_cluster_size", 1),
        },
    }


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ConfigChange:
    """Suggested configuration change."""
    file: str
    key: str
    current: Any
    suggested: Any
    reason: str


@dataclass
class AIAnalysis:
    """AI-generated analysis of a pipeline issue."""
    explanation: str
    root_cause: str
    blocked_by: str  # e.g., "embedding:min_blur_score"
    suggested_fixes: List[str]
    config_changes: List[ConfigChange]


@dataclass
class TrackDiagnostics:
    """Diagnostic data for a single track."""
    track_id: int
    ep_id: str

    # Pipeline stage progression
    stage_reached: str  # "detected" | "tracked" | "embedded" | "clustered"
    stage_failed: Optional[str] = None

    # Detection stage
    detection_conf: Optional[float] = None
    bbox_size: Optional[tuple] = None

    # Track stage
    frame_count: int = 0
    first_frame: Optional[int] = None
    last_frame: Optional[int] = None

    # Embedding stage
    faces_count: int = 0
    faces_skipped: int = 0
    faces_with_embeddings: int = 0
    skip_reasons: List[str] = field(default_factory=list)
    max_blur_score: Optional[float] = None
    avg_blur_score: Optional[float] = None
    embedding_generated: bool = False

    # Clustering stage
    cluster_id: Optional[str] = None
    nearest_cluster_sim: Optional[float] = None
    identity_sim: Optional[float] = None

    # Config thresholds used
    config_thresholds: Dict[str, Any] = field(default_factory=dict)

    # AI analysis (populated after OpenAI call)
    ai_analysis: Optional[AIAnalysis] = None

    # Metadata
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if not self.config_thresholds:
            self.config_thresholds = get_pipeline_thresholds()


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for a track."""
    track_id: int
    ep_id: str
    generated_at: str
    diagnostics: TrackDiagnostics
    ai_analysis: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "track_id": self.track_id,
            "ep_id": self.ep_id,
            "generated_at": self.generated_at,
            "stage_reached": self.diagnostics.stage_reached,
            "stage_failed": self.diagnostics.stage_failed,
            "raw_data": {
                "frame_count": self.diagnostics.frame_count,
                "faces_count": self.diagnostics.faces_count,
                "faces_skipped": self.diagnostics.faces_skipped,
                "faces_with_embeddings": self.diagnostics.faces_with_embeddings,
                "skip_reasons": self.diagnostics.skip_reasons,
                "max_blur_score": self.diagnostics.max_blur_score,
                "avg_blur_score": self.diagnostics.avg_blur_score,
                "embedding_generated": self.diagnostics.embedding_generated,
                "cluster_id": self.diagnostics.cluster_id,
            },
            "config_used": self.diagnostics.config_thresholds,
        }
        if self.ai_analysis:
            result["ai_analysis"] = self.ai_analysis
        return result


# ---------------------------------------------------------------------------
# Data collection functions
# ---------------------------------------------------------------------------

def _load_faces_for_track(ep_id: str, track_id: int) -> List[Dict[str, Any]]:
    """Load all faces for a specific track."""
    faces_path = _manifests_dir(ep_id) / "faces.jsonl"
    if not faces_path.exists():
        return []

    faces = []
    with faces_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                face = json.loads(line)
                if face.get("track_id") == track_id:
                    faces.append(face)
            except json.JSONDecodeError:
                continue
    return faces


def _load_track(ep_id: str, track_id: int) -> Optional[Dict[str, Any]]:
    """Load a specific track by ID."""
    tracks_path = _manifests_dir(ep_id) / "tracks.jsonl"
    if not tracks_path.exists():
        return None

    with tracks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                track = json.loads(line)
                if track.get("track_id") == track_id:
                    return track
            except json.JSONDecodeError:
                continue
    return None


def _load_identities(ep_id: str) -> Dict[str, Any]:
    """Load identities.json for episode."""
    identities_path = _manifests_dir(ep_id) / "identities.json"
    if not identities_path.exists():
        return {"identities": []}

    with identities_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_cluster_for_track(ep_id: str, track_id: int) -> Optional[str]:
    """Find which cluster a track belongs to."""
    identities = _load_identities(ep_id)
    for identity in identities.get("identities", []):
        track_ids = identity.get("track_ids", [])
        if track_id in track_ids or str(track_id) in [str(t) for t in track_ids]:
            return identity.get("identity_id") or identity.get("cluster_id")
    return None


def collect_track_diagnostics(ep_id: str, track_id: int) -> TrackDiagnostics:
    """Collect all diagnostic data for a track.

    This function gathers data from faces.jsonl, tracks.jsonl, and identities.json
    to determine:
    - How far the track progressed in the pipeline
    - Why it might have failed at a particular stage
    - Relevant metrics and thresholds
    """
    # Load track data
    track = _load_track(ep_id, track_id)
    faces = _load_faces_for_track(ep_id, track_id)
    cluster_id = _find_cluster_for_track(ep_id, track_id)

    # Determine pipeline stage
    if not track and not faces:
        stage_reached = "unknown"
        stage_failed = "detected"
    elif track:
        stage_reached = "tracked"
    else:
        stage_reached = "detected"

    # Analyze faces
    faces_count = len(faces)
    faces_skipped = 0
    faces_with_embeddings = 0
    skip_reasons = []
    blur_scores = []

    for face in faces:
        # Check skip status
        skip = face.get("skip")
        if skip:
            faces_skipped += 1
            skip_reasons.append(str(skip))

        # Check embedding
        if face.get("embedding") or face.get("has_embedding"):
            faces_with_embeddings += 1

        # Collect blur scores
        blur = face.get("blur")
        if blur is not None:
            blur_scores.append(float(blur))
        # Also check skip_data for blur
        skip_data = face.get("skip_data", {})
        if skip_data.get("blur_score") is not None:
            blur_scores.append(float(skip_data["blur_score"]))

    # Calculate blur stats
    max_blur = max(blur_scores) if blur_scores else None
    avg_blur = sum(blur_scores) / len(blur_scores) if blur_scores else None

    # Determine embedding status
    embedding_generated = faces_with_embeddings > 0

    # Update stage based on embedding/clustering
    if embedding_generated:
        stage_reached = "embedded"
    if cluster_id:
        stage_reached = "clustered"

    # Determine which stage failed
    stage_failed = None
    if stage_reached == "tracked" and not embedding_generated:
        stage_failed = "embedded"
    elif stage_reached == "embedded" and not cluster_id:
        stage_failed = "clustered"
    elif stage_reached == "detected":
        stage_failed = "tracked"

    # Extract track metadata
    frame_count = 0
    first_frame = None
    last_frame = None
    detection_conf = None

    if track:
        frame_count = track.get("frame_count", 0)
        first_frame = track.get("first_frame_idx")
        last_frame = track.get("last_frame_idx")
        # Get confidence from first bbox
        bboxes = track.get("bboxes_sampled", [])
        if bboxes:
            detection_conf = bboxes[0].get("conf")
    elif faces:
        # Fallback to face data
        frame_count = len(faces)
        frame_indices = [f.get("frame_idx") for f in faces if f.get("frame_idx") is not None]
        if frame_indices:
            first_frame = min(frame_indices)
            last_frame = max(frame_indices)
        if faces:
            detection_conf = faces[0].get("conf") or faces[0].get("detector_conf")

    return TrackDiagnostics(
        track_id=track_id,
        ep_id=ep_id,
        stage_reached=stage_reached,
        stage_failed=stage_failed,
        detection_conf=detection_conf,
        frame_count=frame_count,
        first_frame=first_frame,
        last_frame=last_frame,
        faces_count=faces_count,
        faces_skipped=faces_skipped,
        faces_with_embeddings=faces_with_embeddings,
        skip_reasons=list(set(skip_reasons)),  # Deduplicate
        max_blur_score=max_blur,
        avg_blur_score=avg_blur,
        embedding_generated=embedding_generated,
        cluster_id=cluster_id,
    )


def collect_unclustered_tracks(ep_id: str) -> List[int]:
    """Get list of track IDs that are not in any cluster."""
    # Load all tracks
    tracks_path = _manifests_dir(ep_id) / "tracks.jsonl"
    if not tracks_path.exists():
        return []

    all_track_ids = set()
    with tracks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                track = json.loads(line)
                tid = track.get("track_id")
                if tid is not None:
                    all_track_ids.add(int(tid))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    # Load clustered track IDs
    identities = _load_identities(ep_id)
    clustered_ids = set()
    for identity in identities.get("identities", []):
        for tid in identity.get("track_ids", []):
            try:
                clustered_ids.add(int(tid))
            except (ValueError, TypeError):
                continue

    # Return unclustered
    return sorted(all_track_ids - clustered_ids)


# ---------------------------------------------------------------------------
# Report saving/loading
# ---------------------------------------------------------------------------

def save_diagnostic_report(report: DiagnosticReport) -> Path:
    """Save diagnostic report to JSON file."""
    diagnostics_dir = _diagnostics_dir(report.ep_id)
    filename = f"track_{report.track_id:04d}_diagnostic.json"
    path = diagnostics_dir / filename

    with path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    LOGGER.info(f"Saved diagnostic report: {path}")
    return path


def load_diagnostic_report(ep_id: str, track_id: int) -> Optional[DiagnosticReport]:
    """Load existing diagnostic report if available."""
    diagnostics_dir = _diagnostics_dir(ep_id)
    filename = f"track_{track_id:04d}_diagnostic.json"
    path = diagnostics_dir / filename

    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct DiagnosticReport (simplified)
        diag = TrackDiagnostics(
            track_id=data.get("track_id", track_id),
            ep_id=data.get("ep_id", ep_id),
            stage_reached=data.get("stage_reached", "unknown"),
            stage_failed=data.get("stage_failed"),
        )

        raw = data.get("raw_data", {})
        diag.faces_count = raw.get("faces_count", 0)
        diag.faces_skipped = raw.get("faces_skipped", 0)
        diag.faces_with_embeddings = raw.get("faces_with_embeddings", 0)
        diag.skip_reasons = raw.get("skip_reasons", [])
        diag.max_blur_score = raw.get("max_blur_score")
        diag.embedding_generated = raw.get("embedding_generated", False)
        diag.config_thresholds = data.get("config_used", {})

        return DiagnosticReport(
            track_id=track_id,
            ep_id=ep_id,
            generated_at=data.get("generated_at", ""),
            diagnostics=diag,
            ai_analysis=data.get("ai_analysis"),
        )
    except Exception as e:
        LOGGER.warning(f"Failed to load diagnostic report: {e}")
        return None


def list_diagnostic_reports(ep_id: str) -> List[Dict[str, Any]]:
    """List all diagnostic reports for an episode."""
    diagnostics_dir = _diagnostics_dir(ep_id)
    reports = []

    for path in diagnostics_dir.glob("track_*_diagnostic.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                reports.append({
                    "track_id": data.get("track_id"),
                    "stage_reached": data.get("stage_reached"),
                    "stage_failed": data.get("stage_failed"),
                    "has_ai_analysis": bool(data.get("ai_analysis")),
                    "generated_at": data.get("generated_at"),
                })
        except Exception:
            continue

    return sorted(reports, key=lambda x: x.get("track_id", 0))
