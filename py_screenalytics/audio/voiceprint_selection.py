"""Voiceprint segment selection for cast members.

Selects clean, exclusive segments per cast member for voiceprint creation.

Rules:
- Only segments from manually assigned speaker groups
- Exclusive diarization (no overlapping speakers)
- Duration 4-20s (never >30s per Pyannote limit)
- Prefer higher turn-level confidence
- Up to 3 segments per cast member
- Minimum 10s total clean speech required to create voiceprint
- **NEW**: Skip segments with multiple ASR utterances (indicates mixed speakers)
- **NEW**: Skip segments with dialogue patterns (response phrases, etc.)

IMPORTANT: This module operates on speaker groups (from audio_speaker_groups.json)
and manual assignments (from audio_speaker_assignments.json), NOT voice clusters.
Voice clusters (VC_XX) are a separate abstraction used for UI/clustering features.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    ASRSegment,
    AudioSpeakerGroupsManifest,
    SpeakerGroup,
    VoiceprintIdentificationConfig,
)

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Utterance Analysis & Dialogue Detection
# =============================================================================

# Response phrases that suggest a different speaker is responding
DIALOGUE_RESPONSE_PATTERNS = [
    r"^(oh|oh my|oh no|yes|yeah|no|nope|uh huh|mhm|hmm|wow|what|really|right)\b",
    r"^(i know|i mean|i think|i feel|i love|i'm)\b",
    r"^(wait|okay|ok|so|but|and|well)\b",
    r"^(thank you|thanks|exactly|absolutely|definitely|totally)\b",
]

# Phrases that typically appear in announcer/narrator speech (different from cast dialogue)
NARRATOR_PATTERNS = [
    r"(this season|next week|later this season|coming up|previously on)",
    r"(the real housewives of)",
    r"(on bravo|on peacock)",
]


@dataclass
class UtteranceAnalysis:
    """Analysis of ASR utterances within a diarization segment."""

    segment_start: float
    segment_end: float
    utterance_count: int
    utterances: List[Dict[str, Any]]  # [{start, end, text, confidence}, ...]
    dialogue_risk: str  # "none" | "low" | "high"
    dialogue_reasons: List[str]
    has_narrator_speech: bool
    is_clean_for_voiceprint: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_start": self.segment_start,
            "segment_end": self.segment_end,
            "utterance_count": self.utterance_count,
            "utterances": self.utterances,
            "dialogue_risk": self.dialogue_risk,
            "dialogue_reasons": self.dialogue_reasons,
            "has_narrator_speech": self.has_narrator_speech,
            "is_clean_for_voiceprint": self.is_clean_for_voiceprint,
            "rejection_reason": self.rejection_reason,
        }


def get_overlapping_utterances(
    asr_segments: List[ASRSegment],
    seg_start: float,
    seg_end: float,
    tolerance: float = 0.3,
    min_overlap_ratio: float = 0.3,
) -> List[Dict[str, Any]]:
    """Get ASR utterances that overlap with a diarization segment.

    Args:
        asr_segments: List of ASR segments
        seg_start: Diarization segment start time
        seg_end: Diarization segment end time
        tolerance: Time tolerance for matching
        min_overlap_ratio: Minimum overlap ratio to include

    Returns:
        List of overlapping utterances as dicts
    """
    overlapping = []

    for asr in asr_segments:
        # Calculate overlap
        overlap_start = max(asr.start, seg_start - tolerance)
        overlap_end = min(asr.end, seg_end + tolerance)
        overlap_duration = max(0, overlap_end - overlap_start)

        asr_duration = asr.end - asr.start
        if asr_duration <= 0:
            continue

        overlap_ratio = overlap_duration / asr_duration

        if overlap_ratio >= min_overlap_ratio:
            overlapping.append({
                "start": asr.start,
                "end": asr.end,
                "text": asr.text,
                "confidence": asr.confidence,
                "overlap_ratio": overlap_ratio,
            })

    return sorted(overlapping, key=lambda x: x["start"])


def detect_dialogue_patterns(utterances: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Detect if utterances within a segment suggest multiple speakers.

    Returns:
        Tuple of (risk_level: "none"|"low"|"high", reasons: List[str])
    """
    if len(utterances) <= 1:
        return "none", []

    reasons = []
    risk_score = 0

    # Check for response patterns in second+ utterances
    for i, utt in enumerate(utterances[1:], start=1):
        text = utt.get("text", "").strip().lower()

        for pattern in DIALOGUE_RESPONSE_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                reasons.append(f"Response pattern in utterance {i+1}: '{text[:30]}...'")
                risk_score += 2
                break

    # Check for multiple distinct "I" statements with gaps
    i_statements = []
    for utt in utterances:
        text = utt.get("text", "").strip().lower()
        if re.match(r"^i('m| am| was| have| had| think| feel| love| know| mean| want)\b", text):
            i_statements.append(utt)

    if len(i_statements) >= 2:
        # Check if there's a gap between I-statements (suggests different speakers)
        for j in range(1, len(i_statements)):
            gap = i_statements[j]["start"] - i_statements[j-1]["end"]
            if gap > 0.5:  # More than 0.5s gap between personal statements
                reasons.append(f"Multiple 'I' statements with {gap:.1f}s gap (different speakers?)")
                risk_score += 3
                break

    # Short back-and-forth utterances (typical of dialogue)
    short_utterances = [u for u in utterances if len(u.get("text", "").split()) <= 5]
    if len(short_utterances) >= 2 and len(utterances) >= 3:
        reasons.append(f"{len(short_utterances)} short utterances suggest dialogue")
        risk_score += 2

    # Time gaps between utterances
    for i in range(1, len(utterances)):
        gap = utterances[i]["start"] - utterances[i-1]["end"]
        if gap > 1.0:
            reasons.append(f"{gap:.1f}s pause between utterances (speaker change?)")
            risk_score += 1

    # Determine risk level
    if risk_score >= 4:
        return "high", reasons
    elif risk_score >= 2:
        return "low", reasons
    else:
        return "none", reasons


def detect_narrator_speech(utterances: List[Dict[str, Any]]) -> bool:
    """Detect if any utterance contains narrator/announcer speech."""
    for utt in utterances:
        text = utt.get("text", "").lower()
        for pattern in NARRATOR_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
    return False


def analyze_segment_utterances(
    asr_segments: List[ASRSegment],
    seg_start: float,
    seg_end: float,
    max_utterances_for_voiceprint: int = 1,
) -> UtteranceAnalysis:
    """Analyze ASR utterances within a diarization segment.

    This provides ADVISORY analysis for voiceprint quality. The heuristics
    here guide auto-selection but do NOT permanently block segments.

    IMPORTANT: is_clean_for_voiceprint is now a soft recommendation:
    - True = safe for auto-selection
    - False = needs manual review, but can still be force-included

    Only truly problematic cases are auto-marked as not clean:
    - Clear narrator/bumper lines (opt-in override available)
    - Very high dialogue risk (multiple distinct speaker patterns)

    Multiple utterances from the SAME speaker are NOT automatically rejected.

    Args:
        asr_segments: Full list of ASR segments
        seg_start: Diarization segment start
        seg_end: Diarization segment end
        max_utterances_for_voiceprint: Max utterances for "ideal" clean segment

    Returns:
        UtteranceAnalysis with advisory clean status and reasons
    """
    utterances = get_overlapping_utterances(asr_segments, seg_start, seg_end)
    utterance_count = len(utterances)

    dialogue_risk, dialogue_reasons = detect_dialogue_patterns(utterances)
    has_narrator = detect_narrator_speech(utterances)

    # ADVISORY logic: only auto-reject truly problematic cases
    # Multiple utterances alone is NOT a reason to reject
    is_clean = True
    rejection_reason = None

    # Check for truly problematic cases that warrant auto-exclusion
    if has_narrator:
        # Narrator speech is a strong signal this isn't a cast confessional
        # Still overridable via force_include, but auto-excluded
        is_clean = False
        rejection_reason = "Contains narrator/announcer speech (overridable)"
    elif dialogue_risk == "high" and utterance_count >= 3:
        # Only reject if BOTH high dialogue risk AND many utterances
        # This catches true back-and-forth dialogue, not single sentences split by ASR
        is_clean = False
        rejection_reason = f"High dialogue risk with {utterance_count} utterances (overridable)"
    # Note: utterance_count > 1 alone is NOT a rejection reason anymore
    # Note: dialogue_risk == "low" or "high" with few utterances is just a warning

    return UtteranceAnalysis(
        segment_start=seg_start,
        segment_end=seg_end,
        utterance_count=utterance_count,
        utterances=utterances,
        dialogue_risk=dialogue_risk,
        dialogue_reasons=dialogue_reasons,
        has_narrator_speech=has_narrator,
        is_clean_for_voiceprint=is_clean,
        rejection_reason=rejection_reason,
    )


def analyze_speaker_group_purity(
    group_segments: List[Dict[str, Any]],
    asr_segments: List[ASRSegment],
) -> Tuple[str, List[str]]:
    """Analyze if segments within a speaker group likely belong to the same person.

    Looks for signs that different segments contain different speakers.

    Args:
        group_segments: List of segment dicts with start/end times
        asr_segments: Full ASR transcript

    Returns:
        Tuple of (purity: "high"|"medium"|"low", warnings: List[str])
    """
    if len(group_segments) <= 1:
        return "high", []

    warnings = []
    issues = 0

    # Collect all utterances per segment
    segment_utterances = []
    for seg in group_segments:
        utts = get_overlapping_utterances(asr_segments, seg["start"], seg["end"])
        segment_utterances.append(utts)

    # Check for narrator speech in some but not all segments
    has_narrator_segments = []
    for i, utts in enumerate(segment_utterances):
        if detect_narrator_speech(utts):
            has_narrator_segments.append(i)

    if 0 < len(has_narrator_segments) < len(group_segments):
        warnings.append(
            f"Mixed content: {len(has_narrator_segments)} of {len(group_segments)} "
            f"segments contain narrator speech"
        )
        issues += 2

    # Check for very different speech patterns across segments
    # (e.g., one segment has short reactive phrases, another has long statements)
    avg_words_per_segment = []
    for utts in segment_utterances:
        total_words = sum(len(u.get("text", "").split()) for u in utts)
        avg_words_per_segment.append(total_words)

    if avg_words_per_segment:
        min_words = min(avg_words_per_segment)
        max_words = max(avg_words_per_segment)
        if max_words > 0 and min_words / max_words < 0.2 and max_words > 10:
            warnings.append(
                f"Speech length varies widely ({min_words}-{max_words} words) - "
                f"possible different speakers"
            )
            issues += 1

    # Determine purity level
    if issues >= 2:
        return "low", warnings
    elif issues >= 1:
        return "medium", warnings
    else:
        return "high", warnings


# =============================================================================
# Voiceprint Override Manifest
# =============================================================================


@dataclass
class VoiceprintSegmentOverride:
    """Manual override for a segment's voiceprint eligibility."""

    source: str  # "pyannote" | "gpt4o" | etc.
    speaker_group_id: str  # e.g., "pyannote:SPEAKER_00"
    segment_id: str  # e.g., "py_0007"
    override: str  # "force_include" | "force_exclude"
    override_by: str = "user"
    override_at: Optional[str] = None  # ISO timestamp
    reason: Optional[str] = None  # Why the override was applied

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "speaker_group_id": self.speaker_group_id,
            "segment_id": self.segment_id,
            "override": self.override,
            "override_by": self.override_by,
            "override_at": self.override_at,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceprintSegmentOverride":
        return cls(
            source=data.get("source", ""),
            speaker_group_id=data.get("speaker_group_id", ""),
            segment_id=data.get("segment_id", ""),
            override=data.get("override", ""),
            override_by=data.get("override_by", "user"),
            override_at=data.get("override_at"),
            reason=data.get("reason"),
        )


@dataclass
class VoiceprintOverridesManifest:
    """Manifest of manual voiceprint segment overrides for an episode."""

    ep_id: str
    schema_version: str = "voiceprint_overrides_v1"
    overrides: List[VoiceprintSegmentOverride] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ep_id": self.ep_id,
            "schema_version": self.schema_version,
            "overrides": [o.to_dict() for o in self.overrides],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceprintOverridesManifest":
        return cls(
            ep_id=data.get("ep_id", ""),
            schema_version=data.get("schema_version", "voiceprint_overrides_v1"),
            overrides=[
                VoiceprintSegmentOverride.from_dict(o)
                for o in data.get("overrides", [])
            ],
        )

    def get_override(self, segment_id: str) -> Optional[str]:
        """Get override for a specific segment. Returns 'force_include', 'force_exclude', or None."""
        for o in self.overrides:
            if o.segment_id == segment_id:
                return o.override
        return None

    def get_override_lookup(self) -> Dict[str, str]:
        """Get lookup dict: segment_id -> override value."""
        return {o.segment_id: o.override for o in self.overrides}

    def set_override(
        self,
        source: str,
        speaker_group_id: str,
        segment_id: str,
        override: Optional[str],
        reason: Optional[str] = None,
    ) -> None:
        """Set or clear an override for a segment."""
        # Remove existing override for this segment
        self.overrides = [o for o in self.overrides if o.segment_id != segment_id]

        # Add new override if not None/null
        if override in ("force_include", "force_exclude"):
            self.overrides.append(VoiceprintSegmentOverride(
                source=source,
                speaker_group_id=speaker_group_id,
                segment_id=segment_id,
                override=override,
                override_by="user",
                override_at=datetime.now().isoformat(),
                reason=reason,
            ))


def load_voiceprint_overrides(manifest_path: Path) -> Optional[VoiceprintOverridesManifest]:
    """Load voiceprint overrides manifest from disk.

    Args:
        manifest_path: Path to audio_voiceprint_overrides.json

    Returns:
        VoiceprintOverridesManifest or None if file doesn't exist
    """
    if not manifest_path.exists():
        return None

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return VoiceprintOverridesManifest.from_dict(data)
    except Exception as e:
        LOGGER.error(f"Failed to load voiceprint overrides: {e}")
        return None


def save_voiceprint_overrides(manifest: VoiceprintOverridesManifest, output_path: Path) -> None:
    """Save voiceprint overrides manifest to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.to_dict(), indent=2),
        encoding="utf-8",
    )
    LOGGER.info(f"Saved voiceprint overrides to {output_path}")


# =============================================================================
# Speaker Assignment Manifest
# =============================================================================


@dataclass
class SpeakerAssignment:
    """A manual speaker-to-cast assignment.

    Supports both group-level and time-range-level assignments:
    - Group-level: start and end are None → applies to entire speaker group
    - Time-range: start and end are floats → applies only to that time window

    Time-range assignments take precedence over group-level assignments.
    """

    source: str  # "pyannote" | "gpt4o" | etc.
    speaker_group_id: str  # e.g., "pyannote:SPEAKER_00"
    cast_id: str
    cast_display_name: Optional[str]
    assigned_by: str  # "user" | "model"
    assigned_at: Optional[str]  # ISO timestamp
    # Time-range fields (None = group-level assignment)
    start: Optional[float] = None  # Segment start time (seconds)
    end: Optional[float] = None  # Segment end time (seconds)

    def is_time_range_assignment(self) -> bool:
        """Check if this is a time-range assignment vs group-level."""
        return self.start is not None and self.end is not None

    def overlaps_time_range(
        self, seg_start: float, seg_end: float, tolerance: float = 0.5
    ) -> bool:
        """Check if this assignment overlaps with a given time range.

        For group-level assignments (start/end=None), always returns True.
        For time-range assignments, checks overlap with tolerance.
        """
        if not self.is_time_range_assignment():
            return True  # Group-level applies to all segments in group

        # Calculate overlap
        overlap_start = max(self.start, seg_start - tolerance)
        overlap_end = min(self.end, seg_end + tolerance)
        return overlap_end > overlap_start

    def overlap_duration(self, seg_start: float, seg_end: float) -> float:
        """Calculate overlap duration with a segment (0 if no overlap)."""
        if not self.is_time_range_assignment():
            return seg_end - seg_start  # Full segment for group-level

        overlap_start = max(self.start, seg_start)
        overlap_end = min(self.end, seg_end)
        return max(0.0, overlap_end - overlap_start)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "source": self.source,
            "speaker_group_id": self.speaker_group_id,
            "cast_id": self.cast_id,
            "cast_display_name": self.cast_display_name,
            "assigned_by": self.assigned_by,
            "assigned_at": self.assigned_at,
        }
        # Only include start/end if this is a time-range assignment
        if self.start is not None:
            result["start"] = self.start
        if self.end is not None:
            result["end"] = self.end
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeakerAssignment":
        return cls(
            source=data.get("source", ""),
            speaker_group_id=data.get("speaker_group_id", ""),
            cast_id=data.get("cast_id", ""),
            cast_display_name=data.get("cast_display_name"),
            assigned_by=data.get("assigned_by", "user"),
            assigned_at=data.get("assigned_at"),
            start=data.get("start"),
            end=data.get("end"),
        )


@dataclass
class SpeakerAssignmentsManifest:
    """Manifest of manual speaker-to-cast assignments for an episode.

    Supports both group-level and time-range-level assignments.
    Time-range assignments take precedence over group-level.
    """

    ep_id: str
    schema_version: str = "audio_speaker_assignments_v2"  # v2 supports time ranges
    assignments: List[SpeakerAssignment] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ep_id": self.ep_id,
            "schema_version": self.schema_version,
            "assignments": [a.to_dict() for a in self.assignments],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeakerAssignmentsManifest":
        return cls(
            ep_id=data.get("ep_id", ""),
            schema_version=data.get("schema_version", "audio_speaker_assignments_v2"),
            assignments=[
                SpeakerAssignment.from_dict(a)
                for a in data.get("assignments", [])
            ],
        )

    def get_user_assignments(self) -> Dict[str, SpeakerAssignment]:
        """Get only user-confirmed GROUP-LEVEL assignments as {speaker_group_id: assignment}.

        NOTE: This returns only group-level assignments (start/end=None).
        For time-range-aware lookup, use find_assignment_for_segment().
        """
        return {
            a.speaker_group_id: a
            for a in self.assignments
            if a.assigned_by == "user" and not a.is_time_range_assignment()
        }

    def get_all_assignments_for_group(self, speaker_group_id: str) -> List[SpeakerAssignment]:
        """Get all assignments (group-level and time-range) for a speaker group."""
        return [
            a for a in self.assignments
            if a.speaker_group_id == speaker_group_id and a.assigned_by == "user"
        ]

    def find_assignment_for_segment(
        self,
        speaker_group_id: str,
        seg_start: float,
        seg_end: float,
        tolerance: float = 0.5,
    ) -> Optional[SpeakerAssignment]:
        """Find the applicable assignment for a specific segment time range.

        Priority:
        1. Time-range assignment with best overlap → use that cast
        2. Group-level assignment → use that cast
        3. None → no assignment

        If multiple time-range assignments overlap, choose the one with max overlap.
        If ambiguous (equal overlap), return None to avoid wrong assignment.

        Args:
            speaker_group_id: The speaker group to look up
            seg_start: Segment start time
            seg_end: Segment end time
            tolerance: Time tolerance for matching

        Returns:
            SpeakerAssignment or None if no applicable assignment
        """
        group_assignments = self.get_all_assignments_for_group(speaker_group_id)
        if not group_assignments:
            return None

        # Separate time-range and group-level assignments
        time_range_assignments = [a for a in group_assignments if a.is_time_range_assignment()]
        group_level_assignments = [a for a in group_assignments if not a.is_time_range_assignment()]

        # Check time-range assignments first (they take precedence)
        overlapping = []
        for a in time_range_assignments:
            if a.overlaps_time_range(seg_start, seg_end, tolerance):
                overlap = a.overlap_duration(seg_start, seg_end)
                overlapping.append((a, overlap))

        if overlapping:
            # Sort by overlap duration (descending)
            overlapping.sort(key=lambda x: x[1], reverse=True)

            # Check for ambiguity: if top 2 have same overlap, it's ambiguous
            if len(overlapping) >= 2:
                top_overlap = overlapping[0][1]
                second_overlap = overlapping[1][1]
                # If overlaps are nearly equal (within 0.1s), it's ambiguous
                if abs(top_overlap - second_overlap) < 0.1:
                    LOGGER.warning(
                        f"Ambiguous time-range assignments for {speaker_group_id} "
                        f"at {seg_start:.1f}-{seg_end:.1f}s: {[a[0].cast_id for a in overlapping[:2]]}"
                    )
                    return None

            return overlapping[0][0]

        # Fall back to group-level assignment
        if group_level_assignments:
            # Should only be one group-level, but take first if multiple
            return group_level_assignments[0]

        return None

    def get_cast_speaker_groups(self, cast_id: str) -> List[str]:
        """Get all speaker_group_ids assigned to a cast member."""
        return list(set(
            a.speaker_group_id
            for a in self.assignments
            if a.cast_id == cast_id and a.assigned_by == "user"
        ))

    def upsert_assignment(
        self,
        source: str,
        speaker_group_id: str,
        cast_id: str,
        cast_display_name: Optional[str],
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> SpeakerAssignment:
        """Add or update an assignment.

        For time-range assignments (start/end provided):
        - Removes any existing time-range assignment that exactly matches
        - Adds the new assignment

        For group-level assignments (start/end=None):
        - Removes any existing group-level assignment for this group
        - Adds the new assignment

        Args:
            source: Assignment source (pyannote, gpt4o, etc.)
            speaker_group_id: Speaker group ID
            cast_id: Cast member ID
            cast_display_name: Display name
            start: Optional start time for time-range assignment
            end: Optional end time for time-range assignment

        Returns:
            The created/updated SpeakerAssignment
        """
        is_time_range = start is not None and end is not None

        # Remove matching existing assignment
        if is_time_range:
            # Remove time-range assignment with same group and overlapping time
            self.assignments = [
                a for a in self.assignments
                if not (
                    a.speaker_group_id == speaker_group_id
                    and a.is_time_range_assignment()
                    and a.start is not None
                    and a.end is not None
                    and abs(a.start - start) < 0.5
                    and abs(a.end - end) < 0.5
                )
            ]
        else:
            # Remove group-level assignment for this group
            self.assignments = [
                a for a in self.assignments
                if not (
                    a.speaker_group_id == speaker_group_id
                    and not a.is_time_range_assignment()
                )
            ]

        # Create new assignment
        new_assignment = SpeakerAssignment(
            source=source,
            speaker_group_id=speaker_group_id,
            cast_id=cast_id,
            cast_display_name=cast_display_name,
            assigned_by="user",
            assigned_at=datetime.now().isoformat(),
            start=start,
            end=end,
        )
        self.assignments.append(new_assignment)
        return new_assignment

    def remove_assignment(
        self,
        speaker_group_id: str,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> bool:
        """Remove an assignment.

        Args:
            speaker_group_id: Speaker group ID
            start: Optional start time (for time-range removal)
            end: Optional end time (for time-range removal)

        Returns:
            True if an assignment was removed
        """
        initial_count = len(self.assignments)
        is_time_range = start is not None and end is not None

        if is_time_range:
            self.assignments = [
                a for a in self.assignments
                if not (
                    a.speaker_group_id == speaker_group_id
                    and a.is_time_range_assignment()
                    and a.start is not None
                    and abs(a.start - start) < 0.5
                )
            ]
        else:
            # Remove group-level only
            self.assignments = [
                a for a in self.assignments
                if not (
                    a.speaker_group_id == speaker_group_id
                    and not a.is_time_range_assignment()
                )
            ]

        return len(self.assignments) < initial_count


def load_speaker_assignments(manifest_path: Path) -> Optional[SpeakerAssignmentsManifest]:
    """Load speaker assignments manifest from disk.

    Args:
        manifest_path: Path to audio_speaker_assignments.json

    Returns:
        SpeakerAssignmentsManifest or None if file doesn't exist
    """
    if not manifest_path.exists():
        return None

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return SpeakerAssignmentsManifest.from_dict(data)
    except Exception as e:
        LOGGER.error(f"Failed to load speaker assignments: {e}")
        return None


def save_speaker_assignments(manifest: SpeakerAssignmentsManifest, output_path: Path) -> None:
    """Save speaker assignments manifest to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.to_dict(), indent=2),
        encoding="utf-8",
    )
    LOGGER.info(f"Saved speaker assignments to {output_path}")


# =============================================================================
# Voiceprint Selection
# =============================================================================


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
    speaker_group_id: str  # The speaker group this segment belongs to
    overlap_ratio: Optional[float] = None
    # Utterance analysis fields (advisory, not blocking)
    utterance_count: int = 1
    dialogue_risk: str = "none"  # "none" | "low" | "high"
    is_clean_for_voiceprint: bool = True  # Advisory auto-selection hint
    rejection_reason: Optional[str] = None
    # Manual override fields
    voiceprint_override: Optional[str] = None  # "force_include" | "force_exclude" | None
    voiceprint_selected: bool = True  # Effective value: auto + override

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
            "speaker_group_id": self.speaker_group_id,
            "overlap_ratio": self.overlap_ratio,
            "utterance_count": self.utterance_count,
            "dialogue_risk": self.dialogue_risk,
            "is_clean_for_voiceprint": self.is_clean_for_voiceprint,
            "rejection_reason": self.rejection_reason,
            "voiceprint_override": self.voiceprint_override,
            "voiceprint_selected": self.voiceprint_selected,
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
    speaker_groups_used: List[str] = field(default_factory=list)

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
            "speaker_groups_used": self.speaker_groups_used,
        }


def select_voiceprint_segments(
    ep_id: str,
    speaker_groups: AudioSpeakerGroupsManifest,
    manual_assignments: SpeakerAssignmentsManifest,
    cast_lookup: Dict[str, str],  # cast_id -> cast_name
    config: Optional[VoiceprintIdentificationConfig] = None,
    asr_segments: Optional[List[ASRSegment]] = None,
    voiceprint_overrides: Optional[VoiceprintOverridesManifest] = None,
) -> Dict[str, CastVoiceprintSelection]:
    """Select clean, exclusive segments per cast member for voiceprint creation.

    This function identifies the best segments for each cast member based on:
    - Manual assignments (speaker_group_id -> cast_id mapping from user)
      - Supports both group-level and time-range-level assignments
      - Time-range assignments take precedence over group-level
    - Segment quality (duration, confidence, no overlap)
    - Utterance analysis (ADVISORY - guides auto-selection, not hard blocking)
    - Manual overrides (force_include/force_exclude take precedence)

    Assignment precedence per segment:
    1. Time-range assignment with best overlap → use that cast
    2. Group-level assignment → use that cast
    3. None → segment is not manually assigned

    Override precedence:
    1. force_exclude → never use, even if heuristics say it's clean
    2. force_include → always use (within safety limits like ≤30s)
    3. None → use heuristics (prefer is_clean_for_voiceprint=True)

    IMPORTANT: This uses speaker groups from audio_speaker_groups.json and
    manual assignments from audio_speaker_assignments.json. It does NOT use
    voice clusters (VC_XX) - those are a separate abstraction.

    Args:
        ep_id: Episode identifier
        speaker_groups: Speaker groups manifest from audio_speaker_groups.json
        manual_assignments: Manual assignments from audio_speaker_assignments.json
        cast_lookup: Dict mapping cast_id -> display name
        config: Optional configuration (uses defaults if not provided)
        asr_segments: Optional ASR segments for utterance analysis
        voiceprint_overrides: Optional manual overrides for segment selection

    Returns:
        Dict mapping cast_id -> CastVoiceprintSelection
    """
    if config is None:
        config = VoiceprintIdentificationConfig()

    # Build override lookup
    override_lookup: Dict[str, str] = {}
    if voiceprint_overrides:
        override_lookup = voiceprint_overrides.get_override_lookup()

    # Count total segments across all sources
    total_segments = sum(
        sum(len(g.segments) for g in src.speakers)
        for src in speaker_groups.sources
    )
    LOGGER.info(f"[{ep_id}] Selecting voiceprint segments from {total_segments} speaker group segments")

    # Get all speaker groups that have ANY assignment (group-level or time-range)
    assigned_groups = set()
    for a in manual_assignments.assignments:
        if a.assigned_by == "user":
            assigned_groups.add(a.speaker_group_id)

    total_assignments = len(manual_assignments.assignments)
    group_level_count = len(manual_assignments.get_user_assignments())
    time_range_count = total_assignments - group_level_count

    LOGGER.info(
        f"[{ep_id}] Found {total_assignments} assignments: "
        f"{group_level_count} group-level, {time_range_count} time-range"
    )

    if not assigned_groups:
        LOGGER.warning(f"[{ep_id}] No user-confirmed speaker assignments found")
        return {}

    # Build speaker_group_id -> SpeakerGroup lookup
    group_lookup: Dict[str, SpeakerGroup] = {}
    for source in speaker_groups.sources:
        for group in source.speakers:
            group_lookup[group.speaker_group_id] = group

    # Group segments by cast_id using time-range-aware lookup
    cast_segments: Dict[str, List[VoiceprintCandidate]] = {}
    cast_groups: Dict[str, List[str]] = {}  # track which groups were used per cast

    for speaker_group_id in assigned_groups:
        # Get the speaker group
        group = group_lookup.get(speaker_group_id)
        if not group:
            LOGGER.warning(f"[{ep_id}] Speaker group {speaker_group_id} not found in manifest")
            continue

        # Process segments from this group
        for seg in group.segments:
            duration = seg.end - seg.start

            # Filter by duration
            if duration < config.min_segment_duration:
                continue
            if duration > config.max_segment_duration:
                # Skip segments that are too long (Pyannote limit is 30s)
                continue

            # Use time-range-aware assignment lookup
            assignment = manual_assignments.find_assignment_for_segment(
                speaker_group_id=speaker_group_id,
                seg_start=seg.start,
                seg_end=seg.end,
            )

            if not assignment:
                # No applicable assignment for this segment (or ambiguous)
                continue

            cast_id = assignment.cast_id
            cast_name = assignment.cast_display_name or cast_lookup.get(cast_id)

            # Track groups used per cast
            if cast_id not in cast_groups:
                cast_groups[cast_id] = []
            if speaker_group_id not in cast_groups[cast_id]:
                cast_groups[cast_id].append(speaker_group_id)

            # For now, we don't have overlap info in SpeakerSegment
            # This could be computed from diarization data if needed
            overlap_ratio = 0.0

            # Utterance analysis (if ASR segments provided)
            utterance_count = 1
            dialogue_risk = "none"
            is_clean = True
            rejection_reason = None

            if asr_segments:
                analysis = analyze_segment_utterances(
                    asr_segments=asr_segments,
                    seg_start=seg.start,
                    seg_end=seg.end,
                    max_utterances_for_voiceprint=1,  # For "ideal" clean segment
                )
                utterance_count = analysis.utterance_count
                dialogue_risk = analysis.dialogue_risk
                is_clean = analysis.is_clean_for_voiceprint
                rejection_reason = analysis.rejection_reason

            # Check for manual override
            override = override_lookup.get(seg.segment_id)

            # Compute effective voiceprint_selected based on override precedence:
            # 1. force_exclude → never use
            # 2. force_include → always use (we already passed duration checks)
            # 3. None → use heuristics (is_clean)
            if override == "force_exclude":
                voiceprint_selected = False
            elif override == "force_include":
                voiceprint_selected = True
            else:
                voiceprint_selected = is_clean

            candidate = VoiceprintCandidate(
                cast_id=cast_id,
                cast_name=cast_name,
                segment_id=seg.segment_id,
                start=seg.start,
                end=seg.end,
                duration=duration,
                confidence=seg.diar_confidence,  # Use diar_confidence from segment
                source="manual_assignment",
                speaker_group_id=speaker_group_id,
                overlap_ratio=overlap_ratio,
                utterance_count=utterance_count,
                dialogue_risk=dialogue_risk,
                is_clean_for_voiceprint=is_clean,
                rejection_reason=rejection_reason,
                voiceprint_override=override,
                voiceprint_selected=voiceprint_selected,
            )

            if cast_id not in cast_segments:
                cast_segments[cast_id] = []
            cast_segments[cast_id].append(candidate)

    # Build selection results per cast
    results: Dict[str, CastVoiceprintSelection] = {}

    # Get all cast members that have manual assignments
    all_cast_ids = set(a.cast_id for a in user_assignments.values())

    for cast_id in all_cast_ids:
        cast_name = cast_lookup.get(cast_id)
        candidates = cast_segments.get(cast_id, [])
        groups_used = cast_groups.get(cast_id, [])

        if not candidates:
            results[cast_id] = CastVoiceprintSelection(
                cast_id=cast_id,
                cast_name=cast_name,
                status="no_assignments",
                total_duration_s=0.0,
                min_required_s=config.min_total_clean_speech_per_cast,
                reason="No valid segments found in manual assignments",
                speaker_groups_used=groups_used,
            )
            continue

        # Filter to segments that are selected (respects overrides)
        # voiceprint_selected is True for:
        #   - force_include (always)
        #   - auto (is_clean_for_voiceprint=True and no override)
        # voiceprint_selected is False for:
        #   - force_exclude (always)
        #   - auto (is_clean_for_voiceprint=False and no override)
        selected_candidates = [c for c in candidates if c.voiceprint_selected]
        excluded_count = len(candidates) - len(selected_candidates)

        # Count force_include overrides
        force_include_count = sum(1 for c in candidates if c.voiceprint_override == "force_include")

        if excluded_count > 0:
            LOGGER.info(
                f"[{ep_id}] Cast {cast_id}: {len(selected_candidates)} selected, "
                f"{excluded_count} excluded ({force_include_count} force-included)"
            )

        if not selected_candidates:
            # All segments were filtered out
            rejection_reasons = [c.rejection_reason for c in candidates if c.rejection_reason]
            results[cast_id] = CastVoiceprintSelection(
                cast_id=cast_id,
                cast_name=cast_name,
                status="insufficient_data",
                candidates=candidates[:3],  # Keep some for debugging
                total_duration_s=sum(c.duration for c in candidates),
                min_required_s=config.min_total_clean_speech_per_cast,
                reason=f"All {len(candidates)} segments excluded. Consider force-including segments in Voices Review.",
                speaker_groups_used=groups_used,
            )
            continue

        # Sort by:
        # 1. force_include overrides first (user explicitly chose these)
        # 2. Then by dialogue_risk (none > low > high)
        # 3. Then by duration (longest first)
        # 4. Then by confidence
        def sort_key(c: VoiceprintCandidate) -> Tuple:
            risk_order = {"none": 0, "low": 1, "high": 2}
            return (
                0 if c.voiceprint_override == "force_include" else 1,  # force_include first
                risk_order.get(c.dialogue_risk, 2),  # lower risk better
                -c.duration,  # longer is better (negative for descending)
                -(c.confidence or 0),  # higher conf better
            )

        selected_candidates.sort(key=sort_key)

        # Select top N candidates
        selected = selected_candidates[: config.max_segments_per_cast]

        # Calculate totals
        total_duration = sum(c.duration for c in selected)
        confidences = [c.confidence for c in selected if c.confidence is not None]
        mean_conf = sum(confidences) / len(confidences) if confidences else None
        score = total_duration * (mean_conf or 1.0)  # Default confidence to 1.0 if not available

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
                speaker_groups_used=groups_used,
            )
        else:
            LOGGER.info(
                f"[{ep_id}] Cast {cast_id} ({cast_name}): {len(selected)} segments, "
                f"{total_duration:.1f}s total, groups={groups_used}"
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
                speaker_groups_used=groups_used,
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
        "schema_version": "voiceprint_selection_v2",  # Updated version for speaker groups
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
                speaker_group_id=c.get("speaker_group_id", c.get("cluster_id", "")),  # Support legacy
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
            speaker_groups_used=sel_data.get("speaker_groups_used", []),
        )

    return results
