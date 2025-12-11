"""
Screen-time Comparison: Face-only vs Face+Body.

Compares screen time metrics between face-only tracking
and combined face+body tracking.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class TimeSegment:
    """A continuous time segment."""

    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    segment_type: str  # "face" | "body" | "both"

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    def to_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration": round(self.duration, 3),
            "frame_count": self.frame_count,
            "segment_type": self.segment_type,
        }


@dataclass
class ScreenTimeBreakdown:
    """Screen time breakdown for an identity."""

    identity_id: str

    # Face-only metrics
    face_only_duration: float = 0.0
    face_only_frames: int = 0
    face_only_segments: List[TimeSegment] = field(default_factory=list)

    # Face+body metrics (combined)
    combined_duration: float = 0.0
    combined_frames: int = 0
    combined_segments: List[TimeSegment] = field(default_factory=list)

    # Breakdown
    face_visible_duration: float = 0.0
    body_only_duration: float = 0.0

    # Delta
    duration_gain: float = 0.0
    duration_gain_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "identity_id": self.identity_id,
            "face_only": {
                "duration": round(self.face_only_duration, 3),
                "frames": self.face_only_frames,
                "segment_count": len(self.face_only_segments),
            },
            "combined": {
                "duration": round(self.combined_duration, 3),
                "frames": self.combined_frames,
                "segment_count": len(self.combined_segments),
            },
            "breakdown": {
                "face_visible_duration": round(self.face_visible_duration, 3),
                "body_only_duration": round(self.body_only_duration, 3),
            },
            "delta": {
                "duration_gain": round(self.duration_gain, 3),
                "duration_gain_pct": round(self.duration_gain_pct, 2),
            },
        }


@dataclass
class ComparisonSummary:
    """Summary of screen-time comparison across all identities."""

    total_identities: int = 0
    identities_with_gain: int = 0

    # Aggregate metrics
    total_face_only_duration: float = 0.0
    total_combined_duration: float = 0.0
    total_duration_gain: float = 0.0
    avg_duration_gain_pct: float = 0.0

    # Per-identity breakdowns
    breakdowns: List[ScreenTimeBreakdown] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_identities": self.total_identities,
                "identities_with_gain": self.identities_with_gain,
                "total_face_only_duration": round(self.total_face_only_duration, 3),
                "total_combined_duration": round(self.total_combined_duration, 3),
                "total_duration_gain": round(self.total_duration_gain, 3),
                "avg_duration_gain_pct": round(self.avg_duration_gain_pct, 2),
            },
            "breakdowns": [b.to_dict() for b in self.breakdowns],
        }


class ScreenTimeComparator:
    """Compares face-only vs face+body screen time."""

    def __init__(
        self,
        fps: float = 24.0,
        merge_short_gaps: bool = True,
        max_merge_gap_seconds: float = 1.0,
    ):
        self.fps = fps
        self.merge_short_gaps = merge_short_gaps
        self.max_merge_gap_seconds = max_merge_gap_seconds

    def _frames_to_time(self, frame: int) -> float:
        """Convert frame index to timestamp."""
        return frame / self.fps

    def _merge_segments(self, segments: List[TimeSegment]) -> List[TimeSegment]:
        """Merge adjacent segments with small gaps."""
        if not segments or not self.merge_short_gaps:
            return segments

        # Sort by start frame
        sorted_segs = sorted(segments, key=lambda s: s.start_frame)
        merged = [sorted_segs[0]]

        for seg in sorted_segs[1:]:
            last = merged[-1]
            gap = seg.start_time - last.end_time

            if gap <= self.max_merge_gap_seconds and seg.segment_type == last.segment_type:
                # Merge
                merged[-1] = TimeSegment(
                    start_frame=last.start_frame,
                    end_frame=seg.end_frame,
                    start_time=last.start_time,
                    end_time=seg.end_time,
                    segment_type=last.segment_type,
                )
            else:
                merged.append(seg)

        return merged

    def _frames_to_segments(
        self,
        frames: List[int],
        segment_type: str,
    ) -> List[TimeSegment]:
        """Convert list of frame indices to segments."""
        if not frames:
            return []

        sorted_frames = sorted(frames)
        segments = []
        start = sorted_frames[0]
        prev = start

        for frame in sorted_frames[1:]:
            if frame - prev > 1:
                # Gap detected, close current segment
                segments.append(TimeSegment(
                    start_frame=start,
                    end_frame=prev,
                    start_time=self._frames_to_time(start),
                    end_time=self._frames_to_time(prev),
                    segment_type=segment_type,
                ))
                start = frame
            prev = frame

        # Close final segment
        segments.append(TimeSegment(
            start_frame=start,
            end_frame=prev,
            start_time=self._frames_to_time(start),
            end_time=self._frames_to_time(prev),
            segment_type=segment_type,
        ))

        return segments

    def compute_breakdown(
        self,
        identity_id: str,
        face_frames: List[int],
        body_frames: List[int],
    ) -> ScreenTimeBreakdown:
        """Compute screen time breakdown for a single identity."""
        face_set = set(face_frames)
        body_set = set(body_frames)

        # Face-only metrics
        face_only_segments = self._frames_to_segments(list(face_set), "face")
        face_only_segments = self._merge_segments(face_only_segments)
        face_only_duration = sum(s.duration for s in face_only_segments)

        # Combined metrics
        combined_frames = face_set | body_set
        body_only_frames = body_set - face_set

        # Create combined segments with type annotation
        face_segments = self._frames_to_segments(list(face_set), "face")
        body_only_segs = self._frames_to_segments(list(body_only_frames), "body")
        combined_segments = self._merge_segments(face_segments + body_only_segs)
        combined_duration = sum(s.duration for s in combined_segments)

        # Body-only duration
        body_only_duration = sum(s.duration for s in body_only_segs)

        # Calculate gain
        duration_gain = combined_duration - face_only_duration
        duration_gain_pct = (duration_gain / face_only_duration * 100) if face_only_duration > 0 else 0.0

        return ScreenTimeBreakdown(
            identity_id=identity_id,
            face_only_duration=face_only_duration,
            face_only_frames=len(face_set),
            face_only_segments=face_only_segments,
            combined_duration=combined_duration,
            combined_frames=len(combined_frames),
            combined_segments=combined_segments,
            face_visible_duration=face_only_duration,
            body_only_duration=body_only_duration,
            duration_gain=duration_gain,
            duration_gain_pct=duration_gain_pct,
        )

    def compare(
        self,
        fused_identities: Dict[str, dict],
        face_tracks: Dict[int, dict],
        body_tracks: Dict[int, dict],
    ) -> ComparisonSummary:
        """
        Compare screen time across all fused identities.

        Args:
            fused_identities: Dict from track_fusion.json
            face_tracks: Dict of face tracks
            body_tracks: Dict of body tracks

        Returns:
            ComparisonSummary with per-identity breakdowns
        """
        breakdowns = []

        for identity_id, identity in fused_identities.items():
            # Collect face frames
            face_frames = []
            for face_track_id in identity.get("face_track_ids", []):
                track = face_tracks.get(face_track_id, {})
                for det in track.get("detections", []):
                    face_frames.append(det["frame_idx"])

            # Collect body frames
            body_frames = []
            for body_track_id in identity.get("body_track_ids", []):
                track = body_tracks.get(body_track_id, {})
                for det in track.get("detections", []):
                    body_frames.append(det["frame_idx"])

            breakdown = self.compute_breakdown(identity_id, face_frames, body_frames)
            breakdowns.append(breakdown)

        # Compute summary
        summary = ComparisonSummary(
            total_identities=len(breakdowns),
            identities_with_gain=sum(1 for b in breakdowns if b.duration_gain > 0),
            total_face_only_duration=sum(b.face_only_duration for b in breakdowns),
            total_combined_duration=sum(b.combined_duration for b in breakdowns),
            total_duration_gain=sum(b.duration_gain for b in breakdowns),
            breakdowns=breakdowns,
        )

        # Calculate average gain percentage
        gains = [b.duration_gain_pct for b in breakdowns if b.face_only_duration > 0]
        summary.avg_duration_gain_pct = np.mean(gains) if gains else 0.0

        return summary


def compare_screen_time(
    comparator: ScreenTimeComparator,
    identities_path: Path,
    fusion_path: Path,
    output_path: Path,
) -> ComparisonSummary:
    """
    Compare face-only vs face+body screen time.

    Args:
        comparator: ScreenTimeComparator instance
        identities_path: Path to episode identities.json
        fusion_path: Path to track_fusion.json
        output_path: Path to output comparison JSON

    Returns:
        ComparisonSummary
    """
    identities_path = Path(identities_path)
    fusion_path = Path(fusion_path)
    output_path = Path(output_path)

    # Load fusion results
    logger.info(f"Loading fusion results from: {fusion_path}")
    with open(fusion_path) as f:
        fusion_data = json.load(f)

    fused_identities = fusion_data.get("identities", {})
    logger.info(f"Loaded {len(fused_identities)} fused identities")

    # Load original identities for face track mapping
    logger.info(f"Loading identities from: {identities_path}")
    with open(identities_path) as f:
        identities_data = json.load(f)

    # Build face tracks dict from identities
    face_tracks: Dict[int, dict] = {}
    for identity in identities_data.get("identities", []):
        for track in identity.get("tracks", []):
            track_id = track.get("track_id")
            if track_id is not None:
                face_tracks[track_id] = track

    logger.info(f"Loaded {len(face_tracks)} face tracks")

    # Load body tracks from fusion directory
    body_tracks_path = output_path.parent / "body_tracks.jsonl"
    body_tracks: Dict[int, dict] = {}

    if body_tracks_path.exists():
        with open(body_tracks_path) as f:
            for line in f:
                track = json.loads(line)
                track_id = track.get("track_id")
                if track_id is not None:
                    body_tracks[track_id] = track
        logger.info(f"Loaded {len(body_tracks)} body tracks")

    # Run comparison
    summary = comparator.compare(fused_identities, face_tracks, body_tracks)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)

    logger.info(f"Saved comparison to: {output_path}")
    logger.info(f"  Total face-only duration: {summary.total_face_only_duration:.1f}s")
    logger.info(f"  Total combined duration: {summary.total_combined_duration:.1f}s")
    logger.info(f"  Duration gain: {summary.total_duration_gain:.1f}s ({summary.avg_duration_gain_pct:.1f}%)")

    return summary
