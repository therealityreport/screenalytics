"""Review queue generation for low-confidence segments.

Generates a list of segments that need human review based on:
- Low diarization confidence
- Low identification confidence
- No identification match found
- Uncertain transcript decisions
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .identification_pass import IdentificationResult, TranscriptSegmentDecision
from .models import DiarizationSegment, VoiceprintIdentificationConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class ReviewQueueEntry:
    """A segment that needs human review."""

    segment_id: str
    start: float
    end: float
    duration: float
    text: Optional[str]
    current_cast_id: Optional[str]
    current_cast_name: Optional[str]
    diar_speaker: Optional[str]
    diar_confidence: Optional[float]
    ident_cast_id: Optional[str]
    ident_cast_name: Optional[str]
    ident_confidence: Optional[float]
    reason: str  # "low_diar_conf" | "low_ident_conf" | "no_match" | "uncertain"
    priority: int  # 1=high, 2=medium, 3=low
    decision: Optional[str]  # From transcript regeneration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "text": self.text,
            "current_cast_id": self.current_cast_id,
            "current_cast_name": self.current_cast_name,
            "diar_speaker": self.diar_speaker,
            "diar_confidence": self.diar_confidence,
            "ident_cast_id": self.ident_cast_id,
            "ident_cast_name": self.ident_cast_name,
            "ident_confidence": self.ident_confidence,
            "reason": self.reason,
            "priority": self.priority,
            "decision": self.decision,
        }


@dataclass
class ReviewQueueSummary:
    """Summary of review queue."""

    total_entries: int
    by_reason: Dict[str, int]
    by_priority: Dict[int, int]
    total_duration_s: float
    high_priority_duration_s: float


def generate_review_queue(
    ep_id: str,
    decisions: List[TranscriptSegmentDecision],
    diarization_segments: List[DiarizationSegment],
    identification_result: Optional[IdentificationResult] = None,
    config: Optional[VoiceprintIdentificationConfig] = None,
    output_path: Optional[Path] = None,
) -> List[ReviewQueueEntry]:
    """Generate queue of low-confidence segments for human review.

    Flags segments where:
    - Transcript decision was "uncertain"
    - Diarization confidence < diar_conf_threshold (e.g., 70)
    - Identification confidence < ident_conf_threshold (e.g., 60)
    - No identification match found

    Args:
        ep_id: Episode identifier
        decisions: Transcript segment decisions from regeneration
        diarization_segments: Original diarization segments
        identification_result: Result from identification pass
        config: Optional configuration
        output_path: Path to save review queue JSON

    Returns:
        List of ReviewQueueEntry sorted by priority
    """
    if config is None:
        config = VoiceprintIdentificationConfig()

    if not config.review_queue_enabled:
        LOGGER.info(f"[{ep_id}] Review queue generation disabled")
        return []

    LOGGER.info(f"[{ep_id}] Generating review queue from {len(decisions)} decisions")

    # Build diarization confidence lookup
    diar_conf_lookup: Dict[str, float] = {}
    diar_speaker_lookup: Dict[str, str] = {}
    for seg in diarization_segments:
        seg_id = seg.get_segment_id()
        if seg.confidence is not None:
            diar_conf_lookup[seg_id] = seg.confidence
        diar_speaker_lookup[seg_id] = seg.speaker

    # Build identification confidence lookup from result
    ident_conf_lookup: Dict[str, Dict[str, float]] = {}
    if identification_result and identification_result.voiceprints:
        for vp in identification_result.voiceprints:
            speaker = vp.get("speaker")
            confidence = vp.get("confidence", {})
            if speaker:
                ident_conf_lookup[speaker] = confidence

    # Process each decision
    queue: List[ReviewQueueEntry] = []

    for decision in decisions:
        reasons: List[str] = []
        priority = 3  # Default: low

        seg_id = decision.segment_id
        diar_conf = diar_conf_lookup.get(seg_id)
        diar_speaker = diar_speaker_lookup.get(seg_id)

        # Check if decision was uncertain
        if decision.decision == "uncertain":
            reasons.append("uncertain")
            priority = min(priority, 1)  # High priority

        # Check diarization confidence
        if diar_conf is not None and diar_conf < config.diar_conf_threshold:
            reasons.append("low_diar_conf")
            priority = min(priority, 2)  # Medium priority

        # Check identification confidence
        if decision.ident_confidence is not None:
            if decision.ident_confidence < config.ident_conf_threshold:
                reasons.append("low_ident_conf")
                priority = min(priority, 2)
        else:
            if decision.decision != "keep_manual":
                # No identification match and wasn't manually assigned
                reasons.append("no_match")
                priority = min(priority, 1)  # High priority

        # Only add to queue if there's a reason
        if reasons:
            entry = ReviewQueueEntry(
                segment_id=seg_id,
                start=decision.start,
                end=decision.end,
                duration=decision.end - decision.start,
                text=decision.text if hasattr(decision, "text") else None,
                current_cast_id=decision.final_cast_id,
                current_cast_name=decision.final_cast_name,
                diar_speaker=diar_speaker,
                diar_confidence=diar_conf,
                ident_cast_id=decision.ident_cast_id,
                ident_cast_name=decision.ident_cast_name,
                ident_confidence=decision.ident_confidence,
                reason=reasons[0],  # Primary reason
                priority=priority,
                decision=decision.decision,
            )
            queue.append(entry)

    # Sort by priority (1=high first), then by start time
    queue.sort(key=lambda e: (e.priority, e.start))

    # Generate summary
    summary = _generate_summary(queue)

    LOGGER.info(
        f"[{ep_id}] Review queue: {summary.total_entries} entries, "
        f"{summary.high_priority_duration_s:.1f}s high priority"
    )

    # Save if output path provided
    if output_path:
        save_review_queue(ep_id, queue, summary, output_path)

    return queue


def _generate_summary(queue: List[ReviewQueueEntry]) -> ReviewQueueSummary:
    """Generate summary statistics for review queue."""
    by_reason: Dict[str, int] = {}
    by_priority: Dict[int, int] = {}
    total_duration = 0.0
    high_priority_duration = 0.0

    for entry in queue:
        # Count by reason
        by_reason[entry.reason] = by_reason.get(entry.reason, 0) + 1

        # Count by priority
        by_priority[entry.priority] = by_priority.get(entry.priority, 0) + 1

        # Sum durations
        total_duration += entry.duration
        if entry.priority == 1:
            high_priority_duration += entry.duration

    return ReviewQueueSummary(
        total_entries=len(queue),
        by_reason=by_reason,
        by_priority=by_priority,
        total_duration_s=total_duration,
        high_priority_duration_s=high_priority_duration,
    )


def save_review_queue(
    ep_id: str,
    queue: List[ReviewQueueEntry],
    summary: Optional[ReviewQueueSummary] = None,
    output_path: Optional[Path] = None,
) -> None:
    """Save review queue to JSON file.

    Args:
        ep_id: Episode identifier
        queue: Review queue entries
        summary: Optional pre-computed summary
        output_path: Output file path
    """
    if output_path is None:
        import os
        data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
        output_path = data_root / "manifests" / ep_id / "review_queue.voiceprints.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if summary is None:
        summary = _generate_summary(queue)

    artifact = {
        "ep_id": ep_id,
        "schema_version": "review_queue_v1",
        "summary": {
            "total_entries": summary.total_entries,
            "by_reason": summary.by_reason,
            "by_priority": {
                "high": summary.by_priority.get(1, 0),
                "medium": summary.by_priority.get(2, 0),
                "low": summary.by_priority.get(3, 0),
            },
            "total_duration_s": summary.total_duration_s,
            "high_priority_duration_s": summary.high_priority_duration_s,
        },
        "queue": [e.to_dict() for e in queue],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    LOGGER.info(f"[{ep_id}] Saved review queue to {output_path}")


def load_review_queue(artifact_path: Path) -> List[ReviewQueueEntry]:
    """Load review queue from artifact file."""
    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queue = []
    for item in data.get("queue", []):
        entry = ReviewQueueEntry(
            segment_id=item.get("segment_id", ""),
            start=item.get("start", 0),
            end=item.get("end", 0),
            duration=item.get("duration", 0),
            text=item.get("text"),
            current_cast_id=item.get("current_cast_id"),
            current_cast_name=item.get("current_cast_name"),
            diar_speaker=item.get("diar_speaker"),
            diar_confidence=item.get("diar_confidence"),
            ident_cast_id=item.get("ident_cast_id"),
            ident_cast_name=item.get("ident_cast_name"),
            ident_confidence=item.get("ident_confidence"),
            reason=item.get("reason", "unknown"),
            priority=item.get("priority", 3),
            decision=item.get("decision"),
        )
        queue.append(entry)

    return queue


def get_high_priority_segments(queue: List[ReviewQueueEntry]) -> List[ReviewQueueEntry]:
    """Get only high-priority segments from the queue."""
    return [e for e in queue if e.priority == 1]


def get_segments_by_cast(
    queue: List[ReviewQueueEntry],
    cast_id: Optional[str] = None,
) -> Dict[str, List[ReviewQueueEntry]]:
    """Group queue entries by current cast assignment.

    Args:
        queue: Review queue entries
        cast_id: Optional filter for specific cast member

    Returns:
        Dict mapping cast_id -> list of entries
    """
    by_cast: Dict[str, List[ReviewQueueEntry]] = {}

    for entry in queue:
        key = entry.current_cast_id or "UNASSIGNED"
        if cast_id is not None and key != cast_id:
            continue
        if key not in by_cast:
            by_cast[key] = []
        by_cast[key].append(entry)

    return by_cast
