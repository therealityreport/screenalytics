"""Golden episode regression tests for screen-time pipeline.

This module tests that the screen-time pipeline produces consistent results
for golden episodes - small, hand-validated test cases.

When metrics drift outside expected ranges, it signals a regression in:
- Face detection/tracking logic
- Identity clustering
- Screen-time metric computation

Run with pytest:
    pytest tests/test_golden_screen_time.py -v

Run as standalone:
    python -m tests.test_golden_screen_time
    python -m tests.test_golden_screen_time rhobh-s05e17

Requirements:
- Golden episode video must exist at the configured path
- If reuse_artifacts=True, existing artifacts may be used (faster)
- If reuse_artifacts=False, runs full pipeline (slower, more thorough)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.golden_episodes import (
    GOLDEN_EPISODES,
    GoldenEpisodeConfig,
    ExpectedMetrics,
    get_golden_config,
    list_golden_episodes,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of checking a single metric."""

    name: str
    actual: float
    expected_min: float
    expected_max: float
    passed: bool
    message: str


@dataclass
class GoldenEpisodeResult:
    """Result of checking a golden episode."""

    episode_id: str
    passed: bool
    metric_results: List[MetricResult]
    error: Optional[str] = None


def _get_data_root() -> Path:
    """Get the data root directory."""
    raw = os.environ.get("SCREENALYTICS_DATA_ROOT")
    if raw:
        return Path(raw).expanduser()
    return PROJECT_ROOT / "data"


def _load_track_metrics(ep_id: str, data_root: Path) -> Optional[Dict[str, Any]]:
    """Load track_metrics.json for an episode."""
    path = data_root / f"manifests/{ep_id}/track_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_identities(ep_id: str, data_root: Path) -> Optional[Dict[str, Any]]:
    """Load identities.json for an episode."""
    path = data_root / f"manifests/{ep_id}/identities.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_tracks(ep_id: str, data_root: Path) -> Optional[List[Dict[str, Any]]]:
    """Load tracks.jsonl for an episode."""
    path = data_root / f"manifests/{ep_id}/tracks.jsonl"
    if not path.exists():
        return None
    tracks = []
    with open(path) as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))
    return tracks


def _get_video_duration(video_path: Path) -> Optional[float]:
    """Get video duration in seconds using ffprobe."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        pass
    return None


def compute_metrics_from_artifacts(
    ep_id: str,
    data_root: Path,
    video_path: Path,
) -> Dict[str, float]:
    """Compute regression metrics from existing artifacts.

    Args:
        ep_id: Episode identifier
        data_root: Path to data root directory
        video_path: Path to video file (for duration)

    Returns:
        Dictionary of metric name -> value
    """
    tracks = _load_tracks(ep_id, data_root)
    track_metrics = _load_track_metrics(ep_id, data_root)
    identities = _load_identities(ep_id, data_root)

    if not tracks or not track_metrics or not identities:
        raise ValueError(f"Missing artifacts for {ep_id}")

    duration_sec = _get_video_duration(video_path)
    if not duration_sec:
        raise ValueError(f"Could not determine video duration for {ep_id}")

    duration_min = duration_sec / 60

    # Compute metrics
    total_tracks = len(tracks)
    id_switches = track_metrics.get("metrics", {}).get("id_switches", 0)

    # Short track fraction (tracks with < 5 frames)
    SHORT_TRACK_THRESHOLD = 5
    short_tracks = sum(1 for t in tracks if t.get("frame_count", 0) < SHORT_TRACK_THRESHOLD)
    short_track_fraction = short_tracks / total_tracks if total_tracks > 0 else 0

    # Tracks per minute
    tracks_per_minute = total_tracks / duration_min if duration_min > 0 else 0

    # ID switch rate
    id_switch_rate = id_switches / total_tracks if total_tracks > 0 else 0

    # Identity and face counts
    identity_list = identities.get("identities", [])
    identities_count = len(identity_list)
    faces_count = sum(i.get("size", 0) for i in identity_list)

    return {
        "tracks_per_minute": tracks_per_minute,
        "short_track_fraction": short_track_fraction,
        "id_switch_rate": id_switch_rate,
        "identities_count": identities_count,
        "faces_count": faces_count,
    }


def check_metric(
    name: str,
    actual: float,
    expected_range: Tuple[float, float],
) -> MetricResult:
    """Check if a metric is within expected range.

    Args:
        name: Metric name
        actual: Actual value
        expected_range: (min, max) inclusive range

    Returns:
        MetricResult with pass/fail status
    """
    expected_min, expected_max = expected_range
    passed = expected_min <= actual <= expected_max

    if passed:
        message = f"{name}: {actual:.4f} (OK, in range [{expected_min:.4f}, {expected_max:.4f}])"
    else:
        if actual < expected_min:
            message = f"{name}: {actual:.4f} < {expected_min:.4f} (TOO LOW)"
        else:
            message = f"{name}: {actual:.4f} > {expected_max:.4f} (TOO HIGH)"

    return MetricResult(
        name=name,
        actual=actual,
        expected_min=expected_min,
        expected_max=expected_max,
        passed=passed,
        message=message,
    )


def check_golden_episode(
    config: GoldenEpisodeConfig,
    data_root: Optional[Path] = None,
    run_pipeline: bool = False,
    reuse_artifacts: bool = True,
) -> GoldenEpisodeResult:
    """Check a golden episode against expected metrics.

    Args:
        config: Golden episode configuration
        data_root: Override data root directory
        run_pipeline: If True, run the pipeline first (not just use artifacts)
        reuse_artifacts: If True, use existing artifacts when available

    Returns:
        GoldenEpisodeResult with pass/fail status and metric details
    """
    if data_root is None:
        data_root = _get_data_root()

    ep_id = config.episode_id
    video_path = data_root / config.video_path

    # Check video exists
    if not video_path.exists():
        return GoldenEpisodeResult(
            episode_id=ep_id,
            passed=False,
            metric_results=[],
            error=f"Video not found: {video_path}",
        )

    # Optionally run the pipeline
    if run_pipeline:
        try:
            from py_screenalytics.pipeline import run_episode, EpisodeRunConfig

            pipeline_config = EpisodeRunConfig(
                device="auto",
                stride=config.baseline_config.get("stride", 4),
                det_thresh=config.baseline_config.get("det_thresh", 0.65),
                cluster_thresh=config.baseline_config.get("cluster_thresh", 0.75),
                reuse_detections=reuse_artifacts,
                reuse_embeddings=reuse_artifacts,
                force_recluster=not reuse_artifacts,
                data_root=data_root,
            )

            result = run_episode(ep_id, video_path, pipeline_config)
            if not result.success:
                return GoldenEpisodeResult(
                    episode_id=ep_id,
                    passed=False,
                    metric_results=[],
                    error=f"Pipeline failed: {result.error}",
                )
        except Exception as exc:
            return GoldenEpisodeResult(
                episode_id=ep_id,
                passed=False,
                metric_results=[],
                error=f"Pipeline error: {exc}",
            )

    # Compute metrics from artifacts
    try:
        metrics = compute_metrics_from_artifacts(ep_id, data_root, video_path)
    except Exception as exc:
        return GoldenEpisodeResult(
            episode_id=ep_id,
            passed=False,
            metric_results=[],
            error=f"Failed to compute metrics: {exc}",
        )

    # Check each metric
    expected = config.expected_metrics
    results = [
        check_metric("tracks_per_minute", metrics["tracks_per_minute"], expected.tracks_per_minute),
        check_metric("short_track_fraction", metrics["short_track_fraction"], expected.short_track_fraction),
        check_metric("id_switch_rate", metrics["id_switch_rate"], expected.id_switch_rate),
        check_metric("identities_count", metrics["identities_count"], expected.identities_count),
        check_metric("faces_count", metrics["faces_count"], expected.faces_count),
    ]

    passed = all(r.passed for r in results)

    return GoldenEpisodeResult(
        episode_id=ep_id,
        passed=passed,
        metric_results=results,
        error=None,
    )


# ==============================================================================
# Pytest Tests
# ==============================================================================


@pytest.fixture
def data_root() -> Path:
    """Get the data root directory."""
    return _get_data_root()


@pytest.mark.parametrize("ep_id", list_golden_episodes())
def test_golden_episode(ep_id: str, data_root: Path):
    """Test that a golden episode's metrics are within expected ranges.

    This test uses existing artifacts (fast). For full pipeline testing,
    use the standalone script with --run-pipeline flag.
    """
    config = get_golden_config(ep_id)
    assert config is not None, f"No config found for {ep_id}"

    result = check_golden_episode(config, data_root=data_root, run_pipeline=False)

    # Print detailed results
    print(f"\n{'=' * 60}")
    print(f"Golden Episode: {ep_id}")
    print(f"Description: {config.description}")
    print(f"{'=' * 60}")

    if result.error:
        print(f"ERROR: {result.error}")
        pytest.fail(result.error)

    for metric in result.metric_results:
        status = "PASS" if metric.passed else "FAIL"
        print(f"[{status}] {metric.message}")

    if not result.passed:
        failed = [m for m in result.metric_results if not m.passed]
        pytest.fail(
            f"{len(failed)} metric(s) out of range:\n"
            + "\n".join(f"  - {m.message}" for m in failed)
        )


# ==============================================================================
# Standalone CLI
# ==============================================================================


def main(argv: List[str] | None = None) -> int:
    """Run golden episode regression checks.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        0 if all checks pass, 1 otherwise
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run golden episode regression checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Check all golden episodes
  %(prog)s rhobh-s05e17       # Check specific episode
  %(prog)s --run-pipeline     # Run pipeline before checking
  %(prog)s --list             # List available golden episodes
        """,
    )
    parser.add_argument(
        "episodes",
        nargs="*",
        help="Episode IDs to check (default: all golden episodes)",
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run the pipeline before checking (slower but more thorough)",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Don't reuse existing artifacts (force full pipeline run)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_episodes",
        help="List available golden episodes and exit",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override data root directory",
    )

    args = parser.parse_args(argv)

    # List mode
    if args.list_episodes:
        print("Available golden episodes:")
        for ep_id in list_golden_episodes():
            config = get_golden_config(ep_id)
            if config:
                print(f"  {ep_id}: {config.description}")
        return 0

    # Determine which episodes to check
    if args.episodes:
        episode_ids = args.episodes
    else:
        episode_ids = list_golden_episodes()

    if not episode_ids:
        print("No golden episodes configured!")
        return 1

    # Run checks
    data_root = args.data_root or _get_data_root()
    all_passed = True
    results: List[GoldenEpisodeResult] = []

    for ep_id in episode_ids:
        config = get_golden_config(ep_id)
        if not config:
            print(f"WARNING: No config found for {ep_id}, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"Checking: {ep_id}")
        print(f"Description: {config.description}")
        print(f"{'=' * 60}")

        result = check_golden_episode(
            config,
            data_root=data_root,
            run_pipeline=args.run_pipeline,
            reuse_artifacts=not args.no_reuse,
        )
        results.append(result)

        if result.error:
            print(f"ERROR: {result.error}")
            all_passed = False
            continue

        for metric in result.metric_results:
            status = "PASS" if metric.passed else "FAIL"
            print(f"[{status}] {metric.message}")

        if not result.passed:
            all_passed = False

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        if result.error:
            print(f"[{status}] {result.episode_id}: {result.error}")
        else:
            failed_count = sum(1 for m in result.metric_results if not m.passed)
            total_count = len(result.metric_results)
            print(f"[{status}] {result.episode_id}: {total_count - failed_count}/{total_count} metrics passed")

    if all_passed:
        print("\nAll golden episode checks passed!")
        return 0
    else:
        print("\nSome golden episode checks FAILED!")
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
