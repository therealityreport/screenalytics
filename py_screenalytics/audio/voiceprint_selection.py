"""Voiceprint segment selection for cast members.

Selects clean, exclusive segments per cast member for voiceprint creation.

Rules:
- Only segments from manually assigned clusters
- Exclusive diarization (no overlapping speakers)
- Duration 4-20s (never >30s per Pyannote limit)
- Prefer higher turn-level confidence
- Up to 3 segments per cast member
- Minimum 10s total clean speech required to create voiceprint
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import DiarizationSegment, VoiceCluster, VoiceprintIdentificationConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class VoiceprintCandidate:
    """A candidate segment for voiceprint creation."""

    cast_id: str
    cast_name: Optional[str]
    segment_id: str
    start: float
    end: float
    duration: float
    confidence: Optional[float]
    source: str  # "manual_assignment"
    cluster_id: str
    overlap_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cast_id": self.cast_id,
            "cast_name": self.cast_name,
            "segment_id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "confidence": self.confidence,
            "source": self.source,
            "cluster_id": self.cluster_id,
            "overlap_ratio": self.overlap_ratio,
        }


@dataclass
class CastVoiceprintSelection:
    """Selection result for a single cast member."""

    cast_id: str
    cast_name: Optional[str]
    status: str  # "ready" | "insufficient_data" | "no_assignments"
    candidates: List[VoiceprintCandidate] = field(default_factory=list)
    total_duration_s: float = 0.0
    min_required_s: float = 10.0
    mean_confidence: Optional[float] = None
    score: Optional[float] = None  # duration * confidence
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cast_id": self.cast_id,
            "cast_name": self.cast_name,
            "status": self.status,
            "candidates": [c.to_dict() for c in self.candidates],
            "total_duration_s": self.total_duration_s,
            "min_required_s": self.min_required_s,
            "mean_confidence": self.mean_confidence,
            "score": self.score,
            "reason": self.reason,
        }


def select_voiceprint_segments(
    ep_id: str,
    diarization_segments: List[DiarizationSegment],
    manual_assignments: Dict[str, Dict[str, Any]],
    voice_clusters: List[VoiceCluster],
    cast_lookup: Dict[str, str],  # cast_id -> cast_name
    config: Optional[VoiceprintIdentificationConfig] = None,
) -> Dict[str, CastVoiceprintSelection]:
    """Select clean, exclusive segments per cast member for voiceprint creation.

    This function identifies the best segments for each cast member based on:
    - Manual assignments (cluster_id -> cast_id mapping)
    - Segment quality (duration, confidence, no overlap)

    Args:
        ep_id: Episode identifier
        diarization_segments: All diarization segments for the episode
        manual_assignments: Dict mapping cluster_id -> {cast_id, assigned_by, ...}
        voice_clusters: Voice cluster data linking segments to clusters
        cast_lookup: Dict mapping cast_id -> display name
        config: Optional configuration (uses defaults if not provided)

    Returns:
        Dict mapping cast_id -> CastVoiceprintSelection
    """
    if config is None:
        config = VoiceprintIdentificationConfig()

    LOGGER.info(f"[{ep_id}] Selecting voiceprint segments from {len(diarization_segments)} diarization segments")
    LOGGER.info(f"[{ep_id}] Found {len(manual_assignments)} manual assignments")

    # Build reverse lookup: cluster_id -> cast_id (only for manual assignments)
    cluster_to_cast: Dict[str, str] = {}
    for cluster_id, assignment_data in manual_assignments.items():
        if assignment_data.get("assigned_by") == "user":
            cast_id = assignment_data.get("cast_id")
            if cast_id:
                cluster_to_cast[cluster_id] = cast_id

    LOGGER.info(f"[{ep_id}] {len(cluster_to_cast)} clusters have manual user assignments")

    # Build segment_id -> DiarizationSegment lookup
    segment_lookup: Dict[str, DiarizationSegment] = {}
    for seg in diarization_segments:
        seg_id = seg.get_segment_id()
        segment_lookup[seg_id] = seg

    # Build cluster_id -> list of segment_ids
    cluster_segments: Dict[str, List[str]] = {}
    for vc in voice_clusters:
        cluster_id = vc.voice_cluster_id
        seg_ids = [s.get_segment_id() for s in vc.segments]
        cluster_segments[cluster_id] = seg_ids

    # Group segments by cast_id
    cast_segments: Dict[str, List[VoiceprintCandidate]] = {}

    for cluster_id, cast_id in cluster_to_cast.items():
        if cluster_id not in cluster_segments:
            LOGGER.warning(f"[{ep_id}] Cluster {cluster_id} not found in voice clusters")
            continue

        seg_ids = cluster_segments[cluster_id]
        cast_name = cast_lookup.get(cast_id)

        for seg_id in seg_ids:
            if seg_id not in segment_lookup:
                continue

            seg = segment_lookup[seg_id]
            duration = seg.end - seg.start

            # Filter by duration
            if duration < config.min_segment_duration:
                continue
            if duration > config.max_segment_duration:
                # Truncate to max duration (Pyannote limit is 30s)
                continue

            # Filter by overlap (exclusive segments only)
            overlap_ratio = seg.overlap_ratio or 0.0
            if overlap_ratio > 0.05:  # Allow up to 5% overlap
                continue

            candidate = VoiceprintCandidate(
                cast_id=cast_id,
                cast_name=cast_name,
                segment_id=seg_id,
                start=seg.start,
                end=seg.end,
                duration=duration,
                confidence=seg.confidence,
                source="manual_assignment",
                cluster_id=cluster_id,
                overlap_ratio=overlap_ratio,
            )

            if cast_id not in cast_segments:
                cast_segments[cast_id] = []
            cast_segments[cast_id].append(candidate)

    # Build selection results per cast
    results: Dict[str, CastVoiceprintSelection] = {}

    # Get all cast members that have manual assignments
    all_cast_ids = set(cluster_to_cast.values())

    for cast_id in all_cast_ids:
        cast_name = cast_lookup.get(cast_id)
        candidates = cast_segments.get(cast_id, [])

        if not candidates:
            results[cast_id] = CastVoiceprintSelection(
                cast_id=cast_id,
                cast_name=cast_name,
                status="no_assignments",
                total_duration_s=0.0,
                min_required_s=config.min_total_clean_speech_per_cast,
                reason="No valid segments found in manual assignments",
            )
            continue

        # Sort by confidence (highest first), then by duration (longest first)
        candidates.sort(
            key=lambda c: (c.confidence or 0.0, c.duration),
            reverse=True,
        )

        # Select top N candidates
        selected = candidates[: config.max_segments_per_cast]

        # Calculate totals
        total_duration = sum(c.duration for c in selected)
        confidences = [c.confidence for c in selected if c.confidence is not None]
        mean_conf = sum(confidences) / len(confidences) if confidences else None
        score = total_duration * (mean_conf or 0.0) if mean_conf else None

        # Check minimum duration requirement (HARD RULE #3)
        if total_duration < config.min_total_clean_speech_per_cast:
            LOGGER.warning(
                f"[{ep_id}] Cast {cast_id} ({cast_name}): only {total_duration:.1f}s clean speech "
                f"(need {config.min_total_clean_speech_per_cast}s+)"
            )
            results[cast_id] = CastVoiceprintSelection(
                cast_id=cast_id,
                cast_name=cast_name,
                status="insufficient_data",
                candidates=selected,
                total_duration_s=total_duration,
                min_required_s=config.min_total_clean_speech_per_cast,
                mean_confidence=mean_conf,
                score=score,
                reason=f"Only {total_duration:.1f}s of clean speech (minimum {config.min_total_clean_speech_per_cast}s required)",
            )
        else:
            LOGGER.info(
                f"[{ep_id}] Cast {cast_id} ({cast_name}): {len(selected)} segments, "
                f"{total_duration:.1f}s total, mean_conf={mean_conf:.2f if mean_conf else 'N/A'}"
            )
            results[cast_id] = CastVoiceprintSelection(
                cast_id=cast_id,
                cast_name=cast_name,
                status="ready",
                candidates=selected,
                total_duration_s=total_duration,
                min_required_s=config.min_total_clean_speech_per_cast,
                mean_confidence=mean_conf,
                score=score,
            )

    # Log summary
    ready_count = sum(1 for r in results.values() if r.status == "ready")
    insufficient_count = sum(1 for r in results.values() if r.status == "insufficient_data")
    no_assign_count = sum(1 for r in results.values() if r.status == "no_assignments")

    LOGGER.info(
        f"[{ep_id}] Voiceprint selection complete: "
        f"{ready_count} ready, {insufficient_count} insufficient data, {no_assign_count} no assignments"
    )

    return results


def save_selection_artifact(
    ep_id: str,
    selection: Dict[str, CastVoiceprintSelection],
    output_path: Path,
) -> None:
    """Save voiceprint selection artifact for debugging.

    Args:
        ep_id: Episode identifier
        selection: Selection results from select_voiceprint_segments
        output_path: Path to save JSON artifact
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "ep_id": ep_id,
        "schema_version": "voiceprint_selection_v1",
        "summary": {
            "total_cast": len(selection),
            "ready": sum(1 for s in selection.values() if s.status == "ready"),
            "insufficient_data": sum(1 for s in selection.values() if s.status == "insufficient_data"),
            "no_assignments": sum(1 for s in selection.values() if s.status == "no_assignments"),
        },
        "cast_selections": {cast_id: s.to_dict() for cast_id, s in selection.items()},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    LOGGER.info(f"[{ep_id}] Saved voiceprint selection artifact to {output_path}")


def load_selection_artifact(
    artifact_path: Path,
) -> Dict[str, CastVoiceprintSelection]:
    """Load voiceprint selection artifact.

    Args:
        artifact_path: Path to selection JSON file

    Returns:
        Dict mapping cast_id -> CastVoiceprintSelection
    """
    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: Dict[str, CastVoiceprintSelection] = {}

    for cast_id, sel_data in data.get("cast_selections", {}).items():
        candidates = [
            VoiceprintCandidate(
                cast_id=c["cast_id"],
                cast_name=c.get("cast_name"),
                segment_id=c["segment_id"],
                start=c["start"],
                end=c["end"],
                duration=c["duration"],
                confidence=c.get("confidence"),
                source=c.get("source", "manual_assignment"),
                cluster_id=c.get("cluster_id", ""),
                overlap_ratio=c.get("overlap_ratio"),
            )
            for c in sel_data.get("candidates", [])
        ]

        results[cast_id] = CastVoiceprintSelection(
            cast_id=cast_id,
            cast_name=sel_data.get("cast_name"),
            status=sel_data.get("status", "unknown"),
            candidates=candidates,
            total_duration_s=sel_data.get("total_duration_s", 0.0),
            min_required_s=sel_data.get("min_required_s", 10.0),
            mean_confidence=sel_data.get("mean_confidence"),
            score=sel_data.get("score"),
            reason=sel_data.get("reason"),
        )

    return results
