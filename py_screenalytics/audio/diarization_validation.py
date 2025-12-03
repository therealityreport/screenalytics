"""
Diarization Validation Layer.

Validates diarization segments to catch issues early before they propagate downstream.
Checks for:
- Chronological ordering
- No negative durations
- No excessive overlaps
- Speaker label format consistency
- Minimum segment durations
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .models import DiarizationSegment

LOGGER = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A single validation issue found in diarization data."""
    severity: str  # "error", "warning", "info"
    code: str      # Short code like "NEG_DURATION", "OUT_OF_ORDER"
    message: str
    segment_index: Optional[int] = None
    segment_start: Optional[float] = None
    segment_end: Optional[float] = None


@dataclass
class ValidationResult:
    """Result of validating diarization segments."""
    is_valid: bool = True
    error_count: int = 0
    warning_count: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity == "error":
            self.error_count += 1
            self.is_valid = False
        elif issue.severity == "warning":
            self.warning_count += 1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [
                {
                    "severity": i.severity,
                    "code": i.code,
                    "message": i.message,
                    "segment_index": i.segment_index,
                    "segment_start": i.segment_start,
                    "segment_end": i.segment_end,
                }
                for i in self.issues
            ],
            "stats": self.stats,
        }


@dataclass
class ValidationConfig:
    """Configuration for diarization validation."""
    min_segment_duration_s: float = 0.1    # Minimum valid segment duration
    max_overlap_ratio: float = 0.5         # Maximum allowed overlap between consecutive segments
    max_gap_s: float = 30.0                # Maximum gap between segments (warn if exceeded)
    require_chronological: bool = True     # Require segments to be chronologically ordered
    speaker_label_pattern: Optional[str] = None  # Regex pattern for valid speaker labels
    max_speakers: int = 50                 # Maximum reasonable number of speakers


def validate_diarization(
    segments: List[DiarizationSegment],
    config: Optional[ValidationConfig] = None,
    source_name: str = "unknown",
) -> ValidationResult:
    """
    Validate a list of diarization segments.

    Args:
        segments: List of DiarizationSegment objects
        config: Validation configuration
        source_name: Name of the diarization source (for logging)

    Returns:
        ValidationResult with issues and statistics
    """
    config = config or ValidationConfig()
    result = ValidationResult()

    if not segments:
        result.add_issue(ValidationIssue(
            severity="warning",
            code="EMPTY_SEGMENTS",
            message=f"No diarization segments from {source_name}",
        ))
        return result

    # Collect statistics
    speakers = set()
    total_duration = 0.0
    min_start = float('inf')
    max_end = 0.0

    # Sort segments by start time for validation
    sorted_segments = sorted(segments, key=lambda s: (s.start, s.end))

    for i, seg in enumerate(sorted_segments):
        # Check for negative duration
        duration = seg.end - seg.start
        if duration < 0:
            result.add_issue(ValidationIssue(
                severity="error",
                code="NEG_DURATION",
                message=f"Segment {i} has negative duration ({duration:.3f}s): start={seg.start}, end={seg.end}",
                segment_index=i,
                segment_start=seg.start,
                segment_end=seg.end,
            ))
            continue

        # Check for too-short segments
        if duration < config.min_segment_duration_s:
            result.add_issue(ValidationIssue(
                severity="warning",
                code="SHORT_SEGMENT",
                message=f"Segment {i} is very short ({duration:.3f}s < {config.min_segment_duration_s}s)",
                segment_index=i,
                segment_start=seg.start,
                segment_end=seg.end,
            ))

        # Track speakers
        speakers.add(seg.speaker)
        total_duration += duration
        min_start = min(min_start, seg.start)
        max_end = max(max_end, seg.end)

        # Check chronological ordering (compare to original order)
        original_index = segments.index(seg)
        if config.require_chronological and original_index != i and i > 0:
            # Only report if this segment appears out of order in original list
            prev_in_original = segments[original_index - 1] if original_index > 0 else None
            if prev_in_original and prev_in_original.start > seg.start:
                result.add_issue(ValidationIssue(
                    severity="warning",
                    code="OUT_OF_ORDER",
                    message=f"Segment {original_index} appears out of chronological order",
                    segment_index=original_index,
                    segment_start=seg.start,
                    segment_end=seg.end,
                ))

        # Check overlap with previous segment
        if i > 0:
            prev_seg = sorted_segments[i - 1]
            overlap = prev_seg.end - seg.start
            if overlap > 0:
                overlap_ratio = overlap / min(duration, prev_seg.end - prev_seg.start)
                if overlap_ratio > config.max_overlap_ratio:
                    result.add_issue(ValidationIssue(
                        severity="warning",
                        code="EXCESSIVE_OVERLAP",
                        message=f"Segments {i-1} and {i} have excessive overlap ({overlap:.3f}s, ratio={overlap_ratio:.2f})",
                        segment_index=i,
                        segment_start=seg.start,
                        segment_end=seg.end,
                    ))

            # Check for large gaps
            gap = seg.start - prev_seg.end
            if gap > config.max_gap_s:
                result.add_issue(ValidationIssue(
                    severity="info",
                    code="LARGE_GAP",
                    message=f"Large gap ({gap:.1f}s) between segments {i-1} and {i}",
                    segment_index=i,
                    segment_start=seg.start,
                    segment_end=seg.end,
                ))

        # Check speaker label format
        if config.speaker_label_pattern:
            if not re.match(config.speaker_label_pattern, seg.speaker):
                result.add_issue(ValidationIssue(
                    severity="warning",
                    code="INVALID_SPEAKER_LABEL",
                    message=f"Segment {i} has invalid speaker label format: '{seg.speaker}'",
                    segment_index=i,
                    segment_start=seg.start,
                    segment_end=seg.end,
                ))

    # Check total speaker count
    if len(speakers) > config.max_speakers:
        result.add_issue(ValidationIssue(
            severity="warning",
            code="TOO_MANY_SPEAKERS",
            message=f"Unusually high speaker count ({len(speakers)} > {config.max_speakers})",
        ))

    # Check for duplicate speakers with different cases
    speaker_lower = {}
    for spk in speakers:
        lower = spk.lower()
        if lower in speaker_lower and speaker_lower[lower] != spk:
            result.add_issue(ValidationIssue(
                severity="warning",
                code="CASE_MISMATCH_SPEAKER",
                message=f"Speaker labels differ only by case: '{spk}' vs '{speaker_lower[lower]}'",
            ))
        speaker_lower[lower] = spk

    # Populate statistics
    result.stats = {
        "source": source_name,
        "segment_count": len(segments),
        "speaker_count": len(speakers),
        "total_speech_s": round(total_duration, 2),
        "audio_span_s": round(max_end - min_start, 2) if segments else 0,
        "speech_ratio": round(total_duration / (max_end - min_start), 3) if max_end > min_start else 0,
        "speakers": sorted(speakers),
    }

    # Log validation results
    if result.is_valid:
        LOGGER.info(
            f"Diarization validation passed for {source_name}: "
            f"{len(segments)} segments, {len(speakers)} speakers, {result.warning_count} warnings"
        )
    else:
        LOGGER.warning(
            f"Diarization validation FAILED for {source_name}: "
            f"{result.error_count} errors, {result.warning_count} warnings"
        )

    return result


def fix_diarization_issues(
    segments: List[DiarizationSegment],
    validation_result: ValidationResult,
) -> Tuple[List[DiarizationSegment], int]:
    """
    Attempt to fix common diarization issues.

    Args:
        segments: Original diarization segments
        validation_result: Validation result from validate_diarization()

    Returns:
        Tuple of (fixed segments, number of fixes applied)
    """
    if not segments:
        return segments, 0

    fixes_applied = 0
    fixed_segments = []

    # Sort by start time
    sorted_segments = sorted(segments, key=lambda s: (s.start, s.end))

    for i, seg in enumerate(sorted_segments):
        # Skip segments with negative duration
        if seg.end <= seg.start:
            LOGGER.debug(f"Removing segment {i} with invalid duration")
            fixes_applied += 1
            continue

        # Clip overlapping segments
        if fixed_segments:
            prev = fixed_segments[-1]
            if seg.start < prev.end:
                # Adjust start time to remove overlap
                adjusted_start = prev.end
                if adjusted_start < seg.end:
                    seg = DiarizationSegment(
                        segment_id=seg.segment_id,
                        start=adjusted_start,
                        end=seg.end,
                        speaker=seg.speaker,
                        confidence=seg.confidence,
                        overlap_ratio=seg.overlap_ratio,
                    )
                    LOGGER.debug(f"Adjusted segment {i} start from {seg.start} to {adjusted_start}")
                    fixes_applied += 1
                else:
                    # Segment is entirely within previous, skip it
                    LOGGER.debug(f"Removing segment {i} (entirely overlapped)")
                    fixes_applied += 1
                    continue

        fixed_segments.append(seg)

    LOGGER.info(f"Applied {fixes_applied} fixes to diarization segments")
    return fixed_segments, fixes_applied


def validate_and_fix(
    segments: List[DiarizationSegment],
    config: Optional[ValidationConfig] = None,
    source_name: str = "unknown",
    auto_fix: bool = True,
) -> Tuple[List[DiarizationSegment], ValidationResult]:
    """
    Validate diarization segments and optionally auto-fix issues.

    Args:
        segments: List of DiarizationSegment objects
        config: Validation configuration
        source_name: Name of the diarization source
        auto_fix: Whether to attempt automatic fixes

    Returns:
        Tuple of (possibly fixed segments, validation result)
    """
    # First validation pass
    result = validate_diarization(segments, config, source_name)

    # Apply fixes if requested and there are issues
    if auto_fix and not result.is_valid:
        fixed_segments, fixes = fix_diarization_issues(segments, result)
        if fixes > 0:
            # Re-validate after fixes
            result = validate_diarization(fixed_segments, config, source_name)
            result.stats["fixes_applied"] = fixes
            return fixed_segments, result

    return segments, result
