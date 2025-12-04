"""Golden episodes configuration for regression testing.

Golden episodes are small, hand-validated episodes used as fixed test cases
for regression detection. When pipeline changes cause metrics to drift outside
the expected ranges, it signals a potential regression.

Usage:
    from configs.golden_episodes import GOLDEN_EPISODES, get_golden_config

    # Get all golden episodes
    for ep_id, config in GOLDEN_EPISODES.items():
        print(f"{ep_id}: {config['description']}")

    # Get config for a specific episode
    config = get_golden_config("rhobh-s05e17")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExpectedMetrics:
    """Expected metric ranges for a golden episode.

    All ranges are inclusive [min, max]. A value is considered within range
    if min <= value <= max.

    Metrics:
        tracks_per_minute: Total tracks / duration in minutes
        short_track_fraction: Fraction of tracks with < 5 frames
        id_switch_rate: ID switches / total tracks
        identities_count: Number of identity clusters
        faces_count: Total face detections
    """

    tracks_per_minute: Tuple[float, float]
    short_track_fraction: Tuple[float, float]
    id_switch_rate: Tuple[float, float]
    identities_count: Tuple[int, int]
    faces_count: Tuple[int, int]

    # Optional: expected screen time per identity (seconds)
    # Maps identity label/id to (expected_sec, tolerance_sec)
    screen_time_expectations: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class GoldenEpisodeConfig:
    """Configuration for a golden episode."""

    episode_id: str
    description: str
    video_path: str  # Relative to data root
    expected_metrics: ExpectedMetrics

    # Pipeline config used for baseline (for reproducibility)
    baseline_config: Dict[str, Any] = field(default_factory=dict)

    # Notes about this episode
    notes: List[str] = field(default_factory=list)


# ==============================================================================
# Golden Episode #1: rhobh-s05e17
# ==============================================================================
# Short RHOBH clip (~1.7 minutes) that was manually validated.
# Initial baseline captured on 2025-12-04 from existing artifacts.
#
# IMPORTANT: These ranges are initial estimates based on current pipeline
# output. They may need to be tightened after manual review.
# ==============================================================================

RHOBH_S05E17 = GoldenEpisodeConfig(
    episode_id="rhobh-s05e17",
    description="Short RHOBH clip (1.7min) - Golden Episode #1",
    video_path="videos/rhobh-s05e17/episode.mp4",
    expected_metrics=ExpectedMetrics(
        # Baseline: 56.91 tracks/min
        # Range: +/- 15 to allow for minor detection variations
        tracks_per_minute=(42.0, 72.0),
        # Baseline: 0.388 (38% of tracks are short)
        # Range: +/- 0.15 to allow for tracking variations
        short_track_fraction=(0.24, 0.54),
        # Baseline: 0.0204 (2% ID switch rate)
        # Range: Allow up to 7% switches
        id_switch_rate=(0.0, 0.07),
        # Baseline: 47 identities
        # Range: +/- 10 clusters
        identities_count=(37, 57),
        # Baseline: 622 faces
        # Range: +/- 100 faces
        faces_count=(522, 722),
    ),
    baseline_config={
        "stride": 6,
        "detector": "retinaface",
        "tracker": "bytetrack",
        "det_thresh": 0.65,
        "cluster_thresh": 0.70,
        "pipeline_ver": "2025-11-11",
    },
    notes=[
        "Initial baseline captured 2025-12-04 from existing artifacts.",
        "Ranges are intentionally wide for initial setup.",
        "Tighten ranges after manual validation of results.",
    ],
)


# ==============================================================================
# Placeholder for Golden Episode #2
# ==============================================================================
# Uncomment and configure when adding a second golden episode.
#
# GOLDEN_EP_2 = GoldenEpisodeConfig(
#     episode_id="<episode-id>",
#     description="<description>",
#     video_path="videos/<ep_id>/episode.mp4",
#     expected_metrics=ExpectedMetrics(
#         tracks_per_minute=(0.0, 100.0),
#         short_track_fraction=(0.0, 1.0),
#         id_switch_rate=(0.0, 0.1),
#         identities_count=(0, 100),
#         faces_count=(0, 2000),
#     ),
# )


# ==============================================================================
# Golden Episodes Registry
# ==============================================================================

GOLDEN_EPISODES: Dict[str, GoldenEpisodeConfig] = {
    "rhobh-s05e17": RHOBH_S05E17,
    # Add more golden episodes here:
    # "episode-id": GOLDEN_EP_2,
}


def get_golden_config(episode_id: str) -> Optional[GoldenEpisodeConfig]:
    """Get the golden episode config for a given episode ID.

    Args:
        episode_id: Episode identifier (e.g., 'rhobh-s05e17')

    Returns:
        GoldenEpisodeConfig if found, None otherwise
    """
    return GOLDEN_EPISODES.get(episode_id)


def list_golden_episodes() -> List[str]:
    """Return list of all registered golden episode IDs."""
    return list(GOLDEN_EPISODES.keys())
