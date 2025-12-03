"""Identification pass using Pyannote voiceprints.

Runs speaker identification on an episode using stored cast voiceprints,
then regenerates the speaker-attributed transcript.

MANUAL-FIRST, ML-SECOND RULES:
1. If segment has manual assignment: only override if allow_high_conf_override=True
   AND ident_confidence >= high_conf_override_threshold (80%)
2. If no manual assignment: assign if confidence >= ident_conf_threshold (60%)
3. Otherwise: mark as uncertain and add to review queue

IMPORTANT: This module uses speaker groups (from audio_speaker_groups.json) and
manual assignments (from audio_speaker_assignments.json), NOT voice clusters.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import (
    ASRSegment,
    AudioSpeakerGroupsManifest,
    SpeakerGroup,
    TranscriptRow,
    VoiceprintIdentificationConfig,
    WordTiming,
)
from .pyannote_api import (
    IdentificationJobResult,
    PyannoteAPIClient,
    PyannoteAPIError,
)
from .voiceprint_selection import (
    SpeakerAssignmentsManifest,
    load_speaker_assignments,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class IdentificationResult:
    """Result from running identification on an episode."""

    ep_id: str
    job_id: str
    status: str  # "success" | "failed"
    diarization: List[Dict[str, Any]] = field(default_factory=list)
    identification: List[Dict[str, Any]] = field(default_factory=list)
    voiceprints: List[Dict[str, Any]] = field(default_factory=list)
    cast_labels_used: List[str] = field(default_factory=list)
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ep_id": self.ep_id,
            "job_id": self.job_id,
            "status": self.status,
            "diarization_segments": len(self.diarization),
            "identification_segments": len(self.identification),
            "voiceprint_matches": len(self.voiceprints),
            "cast_labels_used": self.cast_labels_used,
            "error": self.error,
        }


@dataclass
class TranscriptSegmentDecision:
    """Decision record for a single transcript segment."""

    segment_id: str
    start: float
    end: float
    text: str
    had_manual_assignment: bool
    manual_cast_id: Optional[str]
    manual_cast_name: Optional[str]
    ident_cast_id: Optional[str]
    ident_cast_name: Optional[str]
    ident_confidence: Optional[float]
    decision: str  # "keep_manual" | "override_manual" | "assign_ident" | "uncertain"
    final_cast_id: Optional[str]
    final_cast_name: Optional[str]
    reason: str


def run_identification_pass(
    ep_id: str,
    show_id: str,
    audio_path: Path,
    config: Optional[VoiceprintIdentificationConfig] = None,
    artifacts_dir: Optional[Path] = None,
) -> IdentificationResult:
    """Run identification on episode using all available cast voiceprints.

    Args:
        ep_id: Episode identifier
        show_id: Show identifier
        audio_path: Path to episode audio file
        config: Optional configuration
        artifacts_dir: Directory to save artifacts

    Returns:
        IdentificationResult
    """
    if config is None:
        config = VoiceprintIdentificationConfig()

    if artifacts_dir is None:
        import os
        data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
        artifacts_dir = data_root / "manifests" / ep_id

    # Load cast members with voiceprints
    from apps.api.services.cast import CastService

    cast_service = CastService()
    cast_with_voiceprints = cast_service.list_cast_with_voiceprints(show_id)

    if not cast_with_voiceprints:
        LOGGER.warning(f"[{ep_id}] No cast members have voiceprints for show {show_id}")
        return IdentificationResult(
            ep_id=ep_id,
            job_id="",
            status="failed",
            error="No cast members have voiceprints",
        )

    LOGGER.info(f"[{ep_id}] Running identification with {len(cast_with_voiceprints)} cast voiceprints")

    # Build voiceprints array for API
    voiceprints_payload = [
        {
            "label": cast["cast_id"],
            "voiceprint": cast["voiceprint_blob"],
        }
        for cast in cast_with_voiceprints
    ]
    cast_labels = [cast["cast_id"] for cast in cast_with_voiceprints]

    # Create Pyannote client
    try:
        client = PyannoteAPIClient()
    except PyannoteAPIError as e:
        LOGGER.error(f"[{ep_id}] Failed to create Pyannote client: {e}")
        return IdentificationResult(
            ep_id=ep_id,
            job_id="",
            status="failed",
            error=f"Failed to create Pyannote client: {e}",
        )

    try:
        # Upload audio and get presigned URL
        media_url = client.upload_and_get_url(audio_path, expiry_seconds=7200)

        # Submit identification job
        job_id = client.submit_identification(
            media_url=media_url,
            voiceprints=voiceprints_payload,
            threshold=config.ident_matching_threshold,
            exclusive=config.ident_matching_exclusive,
        )

        LOGGER.info(f"[{ep_id}] Submitted identification job: {job_id}")

        # Poll for result
        result: IdentificationJobResult = client.poll_identification_job(
            job_id,
            max_wait=900.0,
        )

        if result.status != "succeeded":
            return IdentificationResult(
                ep_id=ep_id,
                job_id=job_id,
                status="failed",
                error=result.error or f"Job status: {result.status}",
                raw_response=result.raw_response,
            )

        ident_result = IdentificationResult(
            ep_id=ep_id,
            job_id=job_id,
            status="success",
            diarization=result.diarization,
            identification=result.identification,
            voiceprints=result.voiceprints,
            cast_labels_used=cast_labels,
            raw_response=result.raw_response,
        )

        # Save identification artifact
        artifact_path = artifacts_dir / "pyannote_identification.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ep_id": ep_id,
                    "job_id": job_id,
                    "status": "success",
                    "diarization": result.diarization,
                    "identification": result.identification,
                    "voiceprints": result.voiceprints,
                    "cast_labels_used": cast_labels,
                },
                f,
                indent=2,
            )

        LOGGER.info(f"[{ep_id}] Identification complete: {len(result.identification)} segments identified")

        return ident_result

    except PyannoteAPIError as e:
        LOGGER.error(f"[{ep_id}] Identification failed: {e}")
        return IdentificationResult(
            ep_id=ep_id,
            job_id="",
            status="failed",
            error=str(e),
        )
    finally:
        client.close()


def _build_confidence_lookup(
    voiceprints: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Build lookup: diarization_speaker -> {cast_id: confidence, ...}."""
    lookup: Dict[str, Dict[str, float]] = {}

    for vp in voiceprints:
        speaker = vp.get("speaker")
        confidence = vp.get("confidence", {})
        if speaker:
            lookup[speaker] = confidence

    return lookup


def _find_overlapping_identification(
    start: float,
    end: float,
    identification: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Find identification segments that overlap with the given time range."""
    overlapping = []

    for seg in identification:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)

        # Check for overlap
        overlap_start = max(start, seg_start)
        overlap_end = min(end, seg_end)

        if overlap_start < overlap_end:
            overlap_duration = overlap_end - overlap_start
            overlapping.append({
                **seg,
                "overlap_duration": overlap_duration,
            })

    return overlapping


def _get_best_match(
    overlapping: List[Dict[str, Any]],
    confidence_lookup: Dict[str, Dict[str, float]],
) -> Tuple[Optional[str], Optional[float]]:
    """Get the best cast match from overlapping identification segments.

    Returns (cast_id, confidence) or (None, None) if no match.
    """
    if not overlapping:
        return None, None

    # Weight by overlap duration and confidence
    cast_scores: Dict[str, float] = {}
    cast_confidences: Dict[str, float] = {}

    for seg in overlapping:
        match = seg.get("match")
        if not match:
            continue

        diar_speaker = seg.get("diarizationSpeaker")
        overlap_duration = seg.get("overlap_duration", 0)

        # Get confidence for this match
        speaker_conf = confidence_lookup.get(diar_speaker, {})
        confidence = speaker_conf.get(match, 0)

        # Weight score by overlap and confidence
        score = overlap_duration * (confidence / 100.0)

        if match not in cast_scores or score > cast_scores[match]:
            cast_scores[match] = score
            cast_confidences[match] = confidence

    if not cast_scores:
        return None, None

    # Return best match
    best_cast = max(cast_scores.keys(), key=lambda c: cast_scores[c])
    return best_cast, cast_confidences.get(best_cast)


def _build_time_based_manual_assignments(
    speaker_groups: AudioSpeakerGroupsManifest,
    manual_assignments: SpeakerAssignmentsManifest,
) -> List[Tuple[float, float, str, str]]:
    """Build time-based list of manual assignments: [(start, end, cast_id, speaker_group_id), ...].

    This allows us to look up manual assignments by time range rather than segment ID,
    which is necessary because ASR segments may not align exactly with speaker group segments.
    """
    time_assignments: List[Tuple[float, float, str, str]] = []

    user_assignments = manual_assignments.get_user_assignments()

    # Build speaker_group_id -> SpeakerGroup lookup
    group_lookup: Dict[str, SpeakerGroup] = {}
    for source in speaker_groups.sources:
        for group in source.speakers:
            group_lookup[group.speaker_group_id] = group

    # For each assignment, add all segments with time ranges
    for speaker_group_id, assignment in user_assignments.items():
        group = group_lookup.get(speaker_group_id)
        if not group:
            continue

        cast_id = assignment.cast_id
        for seg in group.segments:
            time_assignments.append((seg.start, seg.end, cast_id, speaker_group_id))

    # Sort by start time for efficient lookup
    time_assignments.sort(key=lambda x: x[0])

    return time_assignments


def _find_manual_assignment_for_time(
    start: float,
    end: float,
    time_assignments: List[Tuple[float, float, str, str]],
    min_overlap_ratio: float = 0.5,
) -> Tuple[Optional[str], Optional[str]]:
    """Find manual assignment that overlaps with given time range.

    Returns (cast_id, speaker_group_id) or (None, None) if no match.
    """
    best_overlap = 0.0
    best_cast_id = None
    best_group_id = None

    segment_duration = end - start
    if segment_duration <= 0:
        return None, None

    for (seg_start, seg_end, cast_id, group_id) in time_assignments:
        # Quick skip if no possible overlap
        if seg_end <= start:
            continue
        if seg_start >= end:
            break  # Since sorted by start, no more overlaps possible

        # Calculate overlap
        overlap_start = max(start, seg_start)
        overlap_end = min(end, seg_end)
        overlap = max(0, overlap_end - overlap_start)

        overlap_ratio = overlap / segment_duration

        if overlap_ratio >= min_overlap_ratio and overlap > best_overlap:
            best_overlap = overlap
            best_cast_id = cast_id
            best_group_id = group_id

    return best_cast_id, best_group_id


def regenerate_transcript_from_identification(
    ep_id: str,
    identification_result: IdentificationResult,
    asr_segments: List[ASRSegment],
    speaker_groups: AudioSpeakerGroupsManifest,
    manual_assignments: SpeakerAssignmentsManifest,
    cast_lookup: Dict[str, str],  # cast_id -> name
    config: Optional[VoiceprintIdentificationConfig] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[List[TranscriptRow], List[TranscriptSegmentDecision]]:
    """Regenerate speaker-attributed transcript using identification output.

    MANUAL-FIRST, ML-SECOND RULES:
    1. If segment has manual assignment: only override if allow_high_conf_override=True
       AND ident_confidence >= high_conf_override_threshold (80%)
    2. If no manual assignment: assign if confidence >= ident_conf_threshold (60%)
    3. Otherwise: mark as uncertain

    IMPORTANT: This uses speaker groups and manual assignments, NOT voice clusters.

    Args:
        ep_id: Episode identifier
        identification_result: Result from run_identification_pass
        asr_segments: ASR transcript segments
        speaker_groups: Speaker groups from audio_speaker_groups.json
        manual_assignments: Manual assignments from audio_speaker_assignments.json
        cast_lookup: Dict mapping cast_id -> display name
        config: Optional configuration
        output_dir: Directory to save output artifacts

    Returns:
        Tuple of (transcript_rows, decisions)
    """
    if config is None:
        config = VoiceprintIdentificationConfig()

    if output_dir is None:
        import os
        data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
        output_dir = data_root / "manifests" / ep_id

    LOGGER.info(f"[{ep_id}] Regenerating transcript from {len(asr_segments)} ASR segments")

    # Build confidence lookup
    confidence_lookup = _build_confidence_lookup(identification_result.voiceprints)

    # Build time-based manual assignment lookup
    time_assignments = _build_time_based_manual_assignments(speaker_groups, manual_assignments)
    LOGGER.info(f"[{ep_id}] Built {len(time_assignments)} time-based manual assignment entries")

    # Process each ASR segment
    transcript_rows: List[TranscriptRow] = []
    decisions: List[TranscriptSegmentDecision] = []

    for asr_seg in asr_segments:
        seg_id = asr_seg.get_segment_id()
        start = asr_seg.start
        end = asr_seg.end
        text = asr_seg.text

        # Check for manual assignment by time overlap
        manual_cast_id, speaker_group_id = _find_manual_assignment_for_time(
            start, end, time_assignments
        )
        manual_cast_name = cast_lookup.get(manual_cast_id) if manual_cast_id else None
        had_manual = manual_cast_id is not None

        # Find identification matches
        overlapping = _find_overlapping_identification(
            start, end,
            identification_result.identification,
        )
        ident_cast_id, ident_confidence = _get_best_match(overlapping, confidence_lookup)
        ident_cast_name = cast_lookup.get(ident_cast_id) if ident_cast_id else None

        # Apply MANUAL-FIRST, ML-SECOND decision logic
        decision: str
        final_cast_id: Optional[str]
        final_cast_name: Optional[str]
        reason: str

        if had_manual:
            # Has manual assignment - check if we should override
            if config.allow_high_conf_override and ident_confidence is not None:
                if ident_confidence >= config.high_conf_override_threshold:
                    # High confidence override allowed
                    decision = "override_manual"
                    final_cast_id = ident_cast_id
                    final_cast_name = ident_cast_name
                    reason = f"Identification confidence {ident_confidence:.0f}% >= {config.high_conf_override_threshold:.0f}% threshold"
                else:
                    # Keep manual - confidence not high enough
                    decision = "keep_manual"
                    final_cast_id = manual_cast_id
                    final_cast_name = manual_cast_name
                    reason = f"Identification confidence {ident_confidence:.0f}% < {config.high_conf_override_threshold:.0f}% (manual preserved)"
            else:
                # Override not allowed or no identification match
                decision = "keep_manual"
                final_cast_id = manual_cast_id
                final_cast_name = manual_cast_name
                reason = "Manual assignment preserved (override disabled)" if not config.allow_high_conf_override else "No identification match"
        else:
            # No manual assignment - check identification confidence
            if ident_confidence is not None and ident_confidence >= config.ident_conf_threshold:
                decision = "assign_ident"
                final_cast_id = ident_cast_id
                final_cast_name = ident_cast_name
                reason = f"Identification confidence {ident_confidence:.0f}% >= {config.ident_conf_threshold:.0f}% threshold"
            else:
                # Low confidence or no match - uncertain
                decision = "uncertain"
                final_cast_id = None
                final_cast_name = None
                if ident_confidence is not None:
                    reason = f"Identification confidence {ident_confidence:.0f}% < {config.ident_conf_threshold:.0f}% threshold"
                else:
                    reason = "No identification match found"

        # Create decision record
        decisions.append(TranscriptSegmentDecision(
            segment_id=seg_id,
            start=start,
            end=end,
            text=text[:100] + "..." if len(text) > 100 else text,
            had_manual_assignment=had_manual,
            manual_cast_id=manual_cast_id,
            manual_cast_name=manual_cast_name,
            ident_cast_id=ident_cast_id,
            ident_cast_name=ident_cast_name,
            ident_confidence=ident_confidence,
            decision=decision,
            final_cast_id=final_cast_id,
            final_cast_name=final_cast_name,
            reason=reason,
        ))

        # Create transcript row
        transcript_rows.append(TranscriptRow(
            start=start,
            end=end,
            speaker_id=final_cast_id or "UNKNOWN",
            speaker_display_name=final_cast_name or "Unknown Speaker",
            voice_cluster_id=speaker_group_id or "",  # Use speaker_group_id instead of cluster_id
            voice_bank_id="",
            text=text,
            conf=ident_confidence / 100.0 if ident_confidence else None,
            words=asr_seg.words,
        ))

    # Log decision summary
    keep_manual = sum(1 for d in decisions if d.decision == "keep_manual")
    override_manual = sum(1 for d in decisions if d.decision == "override_manual")
    assign_ident = sum(1 for d in decisions if d.decision == "assign_ident")
    uncertain = sum(1 for d in decisions if d.decision == "uncertain")

    LOGGER.info(
        f"[{ep_id}] Transcript regeneration decisions: "
        f"keep_manual={keep_manual}, override_manual={override_manual}, "
        f"assign_ident={assign_ident}, uncertain={uncertain}"
    )

    # Save transcript
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSONL
    jsonl_path = output_dir / "speaker_transcript.voiceprints.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in transcript_rows:
            f.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")

    LOGGER.info(f"[{ep_id}] Saved regenerated transcript to {jsonl_path}")

    # Save VTT
    vtt_path = output_dir / "speaker_transcript.voiceprints.vtt"
    _save_vtt(transcript_rows, vtt_path)

    # Save decisions artifact
    decisions_path = output_dir / "transcript_decisions.voiceprints.json"
    with open(decisions_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ep_id": ep_id,
                "schema_version": "transcript_decisions_v2",  # Updated version for speaker groups
                "summary": {
                    "total": len(decisions),
                    "keep_manual": keep_manual,
                    "override_manual": override_manual,
                    "assign_ident": assign_ident,
                    "uncertain": uncertain,
                },
                "config": {
                    "allow_high_conf_override": config.allow_high_conf_override,
                    "high_conf_override_threshold": config.high_conf_override_threshold,
                    "ident_conf_threshold": config.ident_conf_threshold,
                    "preserve_manual_assignments": config.preserve_manual_assignments,
                },
                "decisions": [
                    {
                        "segment_id": d.segment_id,
                        "start": d.start,
                        "end": d.end,
                        "had_manual_assignment": d.had_manual_assignment,
                        "manual_cast_id": d.manual_cast_id,
                        "manual_cast_name": d.manual_cast_name,
                        "ident_cast_id": d.ident_cast_id,
                        "ident_cast_name": d.ident_cast_name,
                        "ident_confidence": d.ident_confidence,
                        "decision": d.decision,
                        "final_cast_id": d.final_cast_id,
                        "final_cast_name": d.final_cast_name,
                        "reason": d.reason,
                    }
                    for d in decisions
                ],
            },
            f,
            indent=2,
        )

    return transcript_rows, decisions


def _save_vtt(transcript_rows: List[TranscriptRow], output_path: Path) -> None:
    """Save transcript as WebVTT file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")

        for i, row in enumerate(transcript_rows):
            # Format timestamps
            start_time = _format_vtt_time(row.start)
            end_time = _format_vtt_time(row.end)

            # Write cue
            f.write(f"{i + 1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"<v {row.speaker_display_name}>{row.text}\n\n")


def _format_vtt_time(seconds: float) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def load_identification_result(artifact_path: Path) -> IdentificationResult:
    """Load identification result from artifact file."""
    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return IdentificationResult(
        ep_id=data.get("ep_id", ""),
        job_id=data.get("job_id", ""),
        status=data.get("status", "unknown"),
        diarization=data.get("diarization", []),
        identification=data.get("identification", []),
        voiceprints=data.get("voiceprints", []),
        cast_labels_used=data.get("cast_labels_used", []),
    )
