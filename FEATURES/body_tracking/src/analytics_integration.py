"""
Analytics Integration for Body Tracking.

Combines face-only and body tracking metrics for comprehensive
screen time analysis and acceptance matrix validation.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class IdentityScreenTime:
    """Screen time breakdown for a single identity."""

    identity_id: str
    identity_name: Optional[str] = None

    # Face-only metrics
    face_visible_seconds: float = 0.0
    face_visible_frames: int = 0

    # Body-only metrics (face not visible but body tracked)
    body_only_seconds: float = 0.0
    body_only_frames: int = 0

    # Combined metrics
    total_screen_time_seconds: float = 0.0
    total_frames: int = 0

    # Derived metrics
    body_contribution_pct: float = 0.0  # % of screen time from body-only
    gap_bridged_seconds: float = 0.0  # Time recovered by body tracking

    def to_dict(self) -> dict:
        return {
            "identity_id": self.identity_id,
            "identity_name": self.identity_name,
            "face_visible_seconds": round(self.face_visible_seconds, 3),
            "face_visible_frames": self.face_visible_frames,
            "body_only_seconds": round(self.body_only_seconds, 3),
            "body_only_frames": self.body_only_frames,
            "total_screen_time_seconds": round(self.total_screen_time_seconds, 3),
            "total_frames": self.total_frames,
            "body_contribution_pct": round(self.body_contribution_pct, 2),
            "gap_bridged_seconds": round(self.gap_bridged_seconds, 3),
        }


@dataclass
class EpisodeAnalytics:
    """Comprehensive analytics for an episode."""

    episode_id: str
    total_duration_seconds: float = 0.0
    fps: float = 24.0

    # Identity-level breakdowns
    identities: List[IdentityScreenTime] = field(default_factory=list)

    # Aggregate metrics
    total_face_visible_seconds: float = 0.0
    total_body_only_seconds: float = 0.0
    total_combined_seconds: float = 0.0

    # Acceptance metrics
    body_id_switch_rate: float = 0.0
    face_vs_body_gap_fraction: float = 0.0

    # Counts
    num_identities: int = 0
    num_identities_with_body_gain: int = 0

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "fps": self.fps,
            "summary": {
                "total_face_visible_seconds": round(self.total_face_visible_seconds, 3),
                "total_body_only_seconds": round(self.total_body_only_seconds, 3),
                "total_combined_seconds": round(self.total_combined_seconds, 3),
                "num_identities": self.num_identities,
                "num_identities_with_body_gain": self.num_identities_with_body_gain,
            },
            "acceptance_metrics": {
                "body_id_switch_rate": round(self.body_id_switch_rate, 4),
                "face_vs_body_gap_fraction": round(self.face_vs_body_gap_fraction, 4),
            },
            "identities": [i.to_dict() for i in self.identities],
        }


def load_screentime_comparison(manifest_dir: Path) -> Optional[Dict]:
    """Load screentime comparison from body tracking output."""
    comparison_path = manifest_dir / "body_tracking" / "screentime_comparison.json"

    if not comparison_path.exists():
        logger.warning(f"Screentime comparison not found: {comparison_path}")
        return None

    with open(comparison_path) as f:
        return json.load(f)


def load_track_fusion(manifest_dir: Path) -> Optional[Dict]:
    """Load track fusion results."""
    fusion_path = manifest_dir / "body_tracking" / "track_fusion.json"

    if not fusion_path.exists():
        logger.warning(f"Track fusion not found: {fusion_path}")
        return None

    with open(fusion_path) as f:
        return json.load(f)


def compute_body_id_switch_rate(
    body_tracks_path: Path,
    fps: float = 24.0,
) -> float:
    """
    Compute body ID switch rate from body tracks.

    ID switch rate = number of track discontinuities / total tracked time
    """
    if not body_tracks_path.exists():
        return 0.0

    total_duration = 0.0
    num_switches = 0

    with open(body_tracks_path) as f:
        for line in f:
            track = json.loads(line)
            duration = track.get("duration", 0.0)
            total_duration += duration

            # Count gaps within track as potential switches
            detections = track.get("detections", [])
            if len(detections) > 1:
                for i in range(1, len(detections)):
                    prev_frame = detections[i - 1].get("frame_idx", 0)
                    curr_frame = detections[i].get("frame_idx", 0)
                    gap = curr_frame - prev_frame
                    # Large gaps indicate potential ID switches
                    if gap > fps * 2:  # More than 2 seconds gap
                        num_switches += 1

    if total_duration == 0:
        return 0.0

    return num_switches / (total_duration / 60.0)  # Switches per minute


def compute_analytics(
    episode_id: str,
    manifest_dir: Path,
    fps: float = 24.0,
    episode_duration_seconds: Optional[float] = None,
) -> EpisodeAnalytics:
    """
    Compute comprehensive analytics for an episode.

    Args:
        episode_id: Episode identifier
        manifest_dir: Path to manifest directory
        fps: Frames per second
        episode_duration_seconds: Total episode duration (optional)

    Returns:
        EpisodeAnalytics with all computed metrics
    """
    manifest_dir = Path(manifest_dir)

    analytics = EpisodeAnalytics(
        episode_id=episode_id,
        fps=fps,
        total_duration_seconds=episode_duration_seconds or 0.0,
    )

    # Load screentime comparison if available
    comparison = load_screentime_comparison(manifest_dir)

    if comparison:
        breakdowns = comparison.get("breakdowns", [])

        for breakdown in breakdowns:
            identity_id = breakdown.get("identity_id", "unknown")

            face_only = breakdown.get("face_only", {})
            combined = breakdown.get("combined", {})
            breakdown_data = breakdown.get("breakdown", {})
            delta = breakdown.get("delta", {})

            face_visible_secs = face_only.get("duration", 0.0)
            combined_secs = combined.get("duration", 0.0)
            body_only_secs = breakdown_data.get("body_only_duration", 0.0)
            gap_bridged = delta.get("duration_gain", 0.0)

            identity_st = IdentityScreenTime(
                identity_id=identity_id,
                face_visible_seconds=face_visible_secs,
                face_visible_frames=face_only.get("frames", 0),
                body_only_seconds=body_only_secs,
                body_only_frames=int(body_only_secs * fps),
                total_screen_time_seconds=combined_secs,
                total_frames=combined.get("frames", 0),
                body_contribution_pct=(
                    (body_only_secs / combined_secs * 100)
                    if combined_secs > 0 else 0.0
                ),
                gap_bridged_seconds=gap_bridged,
            )

            analytics.identities.append(identity_st)

            # Aggregate
            analytics.total_face_visible_seconds += face_visible_secs
            analytics.total_body_only_seconds += body_only_secs
            analytics.total_combined_seconds += combined_secs

            if gap_bridged > 0:
                analytics.num_identities_with_body_gain += 1

        analytics.num_identities = len(analytics.identities)

        # Summary metrics
        summary = comparison.get("summary", {})
        analytics.face_vs_body_gap_fraction = (
            analytics.total_body_only_seconds / analytics.total_combined_seconds
            if analytics.total_combined_seconds > 0 else 0.0
        )

    # Compute body ID switch rate
    body_tracks_path = manifest_dir / "body_tracking" / "body_tracks.jsonl"
    analytics.body_id_switch_rate = compute_body_id_switch_rate(body_tracks_path, fps)

    return analytics


def validate_acceptance_metrics(
    analytics: EpisodeAnalytics,
    thresholds: Optional[Dict] = None,
) -> Dict[str, Tuple[bool, str]]:
    """
    Validate analytics against acceptance thresholds.

    Args:
        analytics: Computed analytics
        thresholds: Optional threshold overrides

    Returns:
        Dict of metric name to (passed, message) tuples
    """
    thresholds = thresholds or {
        "body_id_switch_rate_max": 0.10,  # Max 10% switches per minute
        "face_vs_body_gap_fraction_max": 0.50,  # Body-only should be < 50%
        "face_vs_body_gap_fraction_min": 0.05,  # Body should contribute at least 5%
    }

    results = {}

    # Body ID switch rate
    switch_rate = analytics.body_id_switch_rate
    max_switch = thresholds["body_id_switch_rate_max"]
    passed = switch_rate <= max_switch
    results["body_id_switch_rate"] = (
        passed,
        f"{switch_rate:.4f} {'<=' if passed else '>'} {max_switch} (max)"
    )

    # Face vs body gap fraction
    gap_frac = analytics.face_vs_body_gap_fraction
    max_gap = thresholds["face_vs_body_gap_fraction_max"]
    min_gap = thresholds["face_vs_body_gap_fraction_min"]

    if gap_frac > max_gap:
        results["face_vs_body_gap_fraction"] = (
            False,
            f"{gap_frac:.4f} > {max_gap} (max) - body tracking may be unreliable"
        )
    elif gap_frac < min_gap:
        results["face_vs_body_gap_fraction"] = (
            True,  # Not a failure, just a note
            f"{gap_frac:.4f} < {min_gap} (min) - body tracking adds minimal value"
        )
    else:
        results["face_vs_body_gap_fraction"] = (
            True,
            f"{gap_frac:.4f} in [{min_gap}, {max_gap}] - good balance"
        )

    return results


def write_analytics(analytics: EpisodeAnalytics, output_path: Path) -> None:
    """Write analytics to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(analytics.to_dict(), f, indent=2)

    logger.info(f"Analytics written to: {output_path}")


def run_analytics(
    episode_id: str,
    manifest_dir: Path,
    output_path: Optional[Path] = None,
    fps: float = 24.0,
) -> EpisodeAnalytics:
    """
    Run full analytics computation and optionally write output.

    Args:
        episode_id: Episode identifier
        manifest_dir: Path to manifest directory
        output_path: Optional output JSON path
        fps: Frames per second

    Returns:
        EpisodeAnalytics
    """
    logger.info(f"Computing analytics for {episode_id}")

    analytics = compute_analytics(episode_id, manifest_dir, fps)

    if output_path:
        write_analytics(analytics, output_path)

    # Validate and log
    validations = validate_acceptance_metrics(analytics)
    for metric, (passed, msg) in validations.items():
        status = "PASS" if passed else "WARN"
        logger.info(f"  [{status}] {metric}: {msg}")

    return analytics
