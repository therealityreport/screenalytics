#!/usr/bin/env python
"""
Face Alignment Evaluation Experiment.

Quantifies the impact of FAN alignment on recognition stability and screentime.

Usage:
    # Evaluate with alignment disabled
    python -m tools.experiments.face_alignment_eval --episode-id rhoslc-s06e01 --alignment-enabled false

    # Evaluate with alignment enabled
    python -m tools.experiments.face_alignment_eval --episode-id rhoslc-s06e01 --alignment-enabled true

    # Compare both modes
    python -m tools.experiments.face_alignment_eval --episode-id rhoslc-s06e01 --alignment-enabled both

    # Multiple episodes
    python -m tools.experiments.face_alignment_eval --episode-id ep1 ep2 ep3 --alignment-enabled both

Output:
    data/experiments/face_alignment_eval/{episode_id}.json
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


logger = logging.getLogger(__name__)


@dataclass
class TrackMetrics:
    """Metrics for a single face track."""

    track_id: int
    frame_count: int
    duration_seconds: float
    mean_confidence: float
    embedding_jitter: float = 0.0  # Mean cosine distance between consecutive embeddings


@dataclass
class EpisodeMetrics:
    """Metrics for an episode run."""

    episode_id: str
    alignment_enabled: bool
    timestamp: str = ""

    # Track-level metrics
    num_tracks: int = 0
    avg_track_length: float = 0.0
    total_track_duration: float = 0.0

    # ID switch metrics
    id_switch_count: int = 0
    id_switch_rate_per_minute: float = 0.0

    # Clustering metrics
    cluster_count: int = 0
    singleton_count: int = 0

    # Embedding quality
    mean_embedding_jitter: float = 0.0
    max_embedding_jitter: float = 0.0

    # Screen time
    total_screen_time_seconds: float = 0.0
    screen_time_per_identity: Dict[str, float] = field(default_factory=dict)

    # Runtime
    pipeline_runtime_seconds: float = 0.0

    # Alignment quality stats (only populated when alignment_enabled=True)
    alignment_quality_mean: Optional[float] = None
    alignment_quality_p05: Optional[float] = None
    alignment_quality_p95: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "alignment_enabled": self.alignment_enabled,
            "timestamp": self.timestamp,
            "track_metrics": {
                "num_tracks": self.num_tracks,
                "avg_track_length": round(self.avg_track_length, 2),
                "total_track_duration": round(self.total_track_duration, 3),
            },
            "id_switch_metrics": {
                "id_switch_count": self.id_switch_count,
                "id_switch_rate_per_minute": round(self.id_switch_rate_per_minute, 4),
            },
            "clustering_metrics": {
                "cluster_count": self.cluster_count,
                "singleton_count": self.singleton_count,
            },
            "embedding_quality": {
                "mean_embedding_jitter": round(self.mean_embedding_jitter, 6),
                "max_embedding_jitter": round(self.max_embedding_jitter, 6),
            },
            "screen_time": {
                "total_seconds": round(self.total_screen_time_seconds, 3),
                "per_identity": {
                    k: round(v, 3) for k, v in self.screen_time_per_identity.items()
                },
            },
            "runtime_seconds": round(self.pipeline_runtime_seconds, 2),
            "alignment_quality_stats": self._get_alignment_quality_stats(),
        }

    def _get_alignment_quality_stats(self) -> Optional[dict]:
        """Return alignment quality stats if available."""
        if self.alignment_quality_mean is None:
            return None
        return {
            "mean": round(self.alignment_quality_mean, 4),
            "p05": round(self.alignment_quality_p05, 4) if self.alignment_quality_p05 else None,
            "p95": round(self.alignment_quality_p95, 4) if self.alignment_quality_p95 else None,
        }


@dataclass
class ComparisonResult:
    """Comparison between aligned and non-aligned runs."""

    episode_id: str
    baseline: EpisodeMetrics  # alignment_enabled=False
    aligned: EpisodeMetrics  # alignment_enabled=True

    # Deltas
    delta_num_tracks: int = 0
    delta_avg_track_length: float = 0.0
    delta_id_switch_rate: float = 0.0
    delta_cluster_count: int = 0
    delta_embedding_jitter: float = 0.0
    delta_total_screen_time: float = 0.0
    delta_screen_time_per_identity: Dict[str, float] = field(default_factory=dict)

    # Improvement flags
    jitter_improved: bool = False
    id_switch_improved: bool = False
    track_length_improved: bool = False

    def compute_deltas(self):
        """Compute delta values from baseline and aligned."""
        self.delta_num_tracks = self.aligned.num_tracks - self.baseline.num_tracks
        self.delta_avg_track_length = self.aligned.avg_track_length - self.baseline.avg_track_length
        self.delta_id_switch_rate = self.aligned.id_switch_rate_per_minute - self.baseline.id_switch_rate_per_minute
        self.delta_cluster_count = self.aligned.cluster_count - self.baseline.cluster_count
        self.delta_embedding_jitter = self.aligned.mean_embedding_jitter - self.baseline.mean_embedding_jitter
        self.delta_total_screen_time = self.aligned.total_screen_time_seconds - self.baseline.total_screen_time_seconds

        # Per-identity screen time deltas
        all_ids = set(self.baseline.screen_time_per_identity.keys()) | set(
            self.aligned.screen_time_per_identity.keys()
        )
        for identity_id in all_ids:
            baseline_time = self.baseline.screen_time_per_identity.get(identity_id, 0.0)
            aligned_time = self.aligned.screen_time_per_identity.get(identity_id, 0.0)
            self.delta_screen_time_per_identity[identity_id] = aligned_time - baseline_time

        # Improvement flags (lower jitter/switch rate is better, higher track length is better)
        self.jitter_improved = self.delta_embedding_jitter < -0.001
        self.id_switch_improved = self.delta_id_switch_rate < -0.01
        self.track_length_improved = self.delta_avg_track_length > 0.5

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "baseline": self.baseline.to_dict(),
            "aligned": self.aligned.to_dict(),
            "deltas": {
                "num_tracks": self.delta_num_tracks,
                "avg_track_length": round(self.delta_avg_track_length, 2),
                "id_switch_rate_per_minute": round(self.delta_id_switch_rate, 4),
                "cluster_count": self.delta_cluster_count,
                "embedding_jitter": round(self.delta_embedding_jitter, 6),
                "total_screen_time_seconds": round(self.delta_total_screen_time, 3),
                "screen_time_per_identity": {
                    k: round(v, 3) for k, v in self.delta_screen_time_per_identity.items()
                },
            },
            "improvements": {
                "jitter_improved": self.jitter_improved,
                "id_switch_improved": self.id_switch_improved,
                "track_length_improved": self.track_length_improved,
            },
        }


def load_tracks(manifest_dir: Path) -> List[Dict]:
    """Load face tracks from manifest."""
    tracks_path = manifest_dir / "tracks.jsonl"
    if not tracks_path.exists():
        # Try faces.jsonl (older format)
        tracks_path = manifest_dir / "faces.jsonl"

    if not tracks_path.exists():
        logger.warning(f"No tracks found at {tracks_path}")
        return []

    tracks = []
    with open(tracks_path) as f:
        for line in f:
            tracks.append(json.loads(line))
    return tracks


def load_identities(manifest_dir: Path) -> Dict:
    """Load identity/cluster data from manifest."""
    identities_path = manifest_dir / "identities.json"
    if not identities_path.exists():
        return {}

    with open(identities_path) as f:
        return json.load(f)


def load_embeddings(manifest_dir: Path) -> Tuple[Optional[np.ndarray], List[Dict]]:
    """Load face embeddings and metadata."""
    embeddings_path = manifest_dir / "face_embeddings.npy"
    meta_path = manifest_dir / "face_embeddings_meta.json"

    if not embeddings_path.exists():
        return None, []

    embeddings = np.load(embeddings_path)

    meta = []
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return embeddings, meta


def load_alignment_quality_stats(manifest_dir: Path) -> Optional[Dict[str, float]]:
    """
    Load alignment quality statistics from aligned_faces.jsonl.

    Returns dict with 'mean', 'p05', 'p95' or None if file doesn't exist.
    """
    aligned_faces_path = manifest_dir / "face_alignment" / "aligned_faces.jsonl"

    if not aligned_faces_path.exists():
        logger.debug(f"No aligned faces found at {aligned_faces_path}")
        return None

    qualities = []
    with open(aligned_faces_path) as f:
        for line in f:
            data = json.loads(line)
            quality = data.get("alignment_quality")
            if quality is not None:
                qualities.append(quality)

    if not qualities:
        return None

    qualities_arr = np.array(qualities)
    return {
        "mean": float(np.mean(qualities_arr)),
        "p05": float(np.percentile(qualities_arr, 5)),
        "p95": float(np.percentile(qualities_arr, 95)),
    }


def compute_embedding_jitter(
    embeddings: np.ndarray,
    meta: List[Dict],
    tracks: List[Dict],
) -> Dict[int, float]:
    """
    Compute embedding jitter (instability) per track.

    Jitter = mean cosine distance between consecutive embeddings in same track.
    Lower is better - indicates more stable embeddings.
    """
    if embeddings is None or len(meta) == 0:
        return {}

    # Build index from meta to embedding
    track_embeddings: Dict[int, List[Tuple[int, np.ndarray]]] = {}

    for i, m in enumerate(meta):
        track_id = m.get("track_id")
        frame_idx = m.get("frame_idx", 0)
        if track_id is not None and i < len(embeddings):
            if track_id not in track_embeddings:
                track_embeddings[track_id] = []
            track_embeddings[track_id].append((frame_idx, embeddings[i]))

    # Compute jitter per track
    jitter_per_track = {}
    for track_id, frame_embs in track_embeddings.items():
        if len(frame_embs) < 2:
            jitter_per_track[track_id] = 0.0
            continue

        # Sort by frame
        frame_embs.sort(key=lambda x: x[0])

        # Compute cosine distances between consecutive embeddings
        distances = []
        for i in range(1, len(frame_embs)):
            emb1 = frame_embs[i - 1][1]
            emb2 = frame_embs[i][1]

            # Normalize
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)

            # Cosine distance = 1 - cosine_similarity
            cos_sim = np.dot(emb1_norm, emb2_norm)
            cos_dist = 1.0 - cos_sim
            distances.append(cos_dist)

        jitter_per_track[track_id] = float(np.mean(distances))

    return jitter_per_track


def compute_id_switch_rate(tracks: List[Dict], fps: float = 24.0) -> Tuple[int, float]:
    """
    Compute ID switch rate from tracks.

    Returns (switch_count, switches_per_minute)
    """
    total_duration = 0.0
    switch_count = 0

    for track in tracks:
        duration = track.get("duration", 0.0)
        if duration == 0 and "detections" in track:
            dets = track["detections"]
            if len(dets) > 0:
                start = dets[0].get("timestamp", dets[0].get("frame_idx", 0) / fps)
                end = dets[-1].get("timestamp", dets[-1].get("frame_idx", 0) / fps)
                duration = end - start

        total_duration += duration

        # Count gaps within track as potential switches
        detections = track.get("detections", [])
        if len(detections) > 1:
            for i in range(1, len(detections)):
                prev_frame = detections[i - 1].get("frame_idx", 0)
                curr_frame = detections[i].get("frame_idx", 0)
                gap = curr_frame - prev_frame
                # Large gap suggests ID switch within track
                if gap > fps * 2:  # > 2 seconds
                    switch_count += 1

    if total_duration == 0:
        return 0, 0.0

    rate_per_minute = switch_count / (total_duration / 60.0)
    return switch_count, rate_per_minute


def compute_screen_time(identities: Dict) -> Dict[str, float]:
    """Compute screen time per identity."""
    screen_time = {}

    for identity in identities.get("identities", []):
        identity_id = identity.get("identity_id", identity.get("cluster_id", "unknown"))
        total_time = 0.0

        for track in identity.get("tracks", []):
            duration = track.get("duration", 0.0)
            total_time += duration

        screen_time[str(identity_id)] = total_time

    return screen_time


def run_evaluation(
    episode_id: str,
    manifest_dir: Path,
    alignment_enabled: bool,
    fps: float = 24.0,
) -> EpisodeMetrics:
    """
    Run evaluation on existing pipeline output.

    Note: This reads existing artifacts rather than re-running the pipeline.
    For a full end-to-end comparison, you would need to run the pipeline twice
    with different alignment settings.
    """
    logger.info(f"Evaluating {episode_id} (alignment_enabled={alignment_enabled})")

    metrics = EpisodeMetrics(
        episode_id=episode_id,
        alignment_enabled=alignment_enabled,
        timestamp=datetime.now().isoformat(),
    )

    # Load artifacts
    tracks = load_tracks(manifest_dir)
    identities = load_identities(manifest_dir)
    embeddings, embed_meta = load_embeddings(manifest_dir)

    if not tracks:
        logger.warning(f"No tracks found for {episode_id}")
        return metrics

    # Track metrics
    metrics.num_tracks = len(tracks)
    track_lengths = [t.get("frame_count", len(t.get("detections", []))) for t in tracks]
    metrics.avg_track_length = float(np.mean(track_lengths)) if track_lengths else 0.0
    metrics.total_track_duration = sum(t.get("duration", 0.0) for t in tracks)

    # ID switch metrics
    metrics.id_switch_count, metrics.id_switch_rate_per_minute = compute_id_switch_rate(
        tracks, fps
    )

    # Clustering metrics
    if identities:
        identity_list = identities.get("identities", [])
        metrics.cluster_count = len(identity_list)
        metrics.singleton_count = sum(
            1 for i in identity_list if len(i.get("tracks", [])) == 1
        )

    # Embedding jitter
    jitter_per_track = compute_embedding_jitter(embeddings, embed_meta, tracks)
    if jitter_per_track:
        jitter_values = list(jitter_per_track.values())
        metrics.mean_embedding_jitter = float(np.mean(jitter_values))
        metrics.max_embedding_jitter = float(np.max(jitter_values))

    # Screen time
    metrics.screen_time_per_identity = compute_screen_time(identities)
    metrics.total_screen_time_seconds = sum(metrics.screen_time_per_identity.values())

    # Alignment quality stats (when alignment is enabled)
    if alignment_enabled:
        quality_stats = load_alignment_quality_stats(manifest_dir)
        if quality_stats:
            metrics.alignment_quality_mean = quality_stats["mean"]
            metrics.alignment_quality_p05 = quality_stats["p05"]
            metrics.alignment_quality_p95 = quality_stats["p95"]
            logger.info(f"  Alignment quality: mean={quality_stats['mean']:.4f}, "
                       f"p05={quality_stats['p05']:.4f}, p95={quality_stats['p95']:.4f}")

    logger.info(f"  Tracks: {metrics.num_tracks}, Avg length: {metrics.avg_track_length:.1f}")
    logger.info(f"  Clusters: {metrics.cluster_count}, Singletons: {metrics.singleton_count}")
    logger.info(f"  Embedding jitter: {metrics.mean_embedding_jitter:.6f}")
    logger.info(f"  ID switch rate: {metrics.id_switch_rate_per_minute:.4f}/min")

    return metrics


def run_comparison(
    episode_id: str,
    baseline_dir: Path,
    aligned_dir: Path,
    fps: float = 24.0,
) -> ComparisonResult:
    """
    Compare baseline (no alignment) vs aligned pipeline outputs.

    Args:
        episode_id: Episode identifier
        baseline_dir: Manifest dir for baseline run (alignment disabled)
        aligned_dir: Manifest dir for aligned run (alignment enabled)
        fps: Frames per second

    Returns:
        ComparisonResult with deltas
    """
    logger.info(f"Comparing baseline vs aligned for {episode_id}")

    baseline = run_evaluation(episode_id, baseline_dir, alignment_enabled=False, fps=fps)
    aligned = run_evaluation(episode_id, aligned_dir, alignment_enabled=True, fps=fps)

    result = ComparisonResult(
        episode_id=episode_id,
        baseline=baseline,
        aligned=aligned,
    )
    result.compute_deltas()

    logger.info("Comparison summary:")
    logger.info(f"  Delta tracks: {result.delta_num_tracks:+d}")
    logger.info(f"  Delta avg length: {result.delta_avg_track_length:+.2f}")
    logger.info(f"  Delta jitter: {result.delta_embedding_jitter:+.6f} ({'improved' if result.jitter_improved else 'no change'})")
    logger.info(f"  Delta ID switch rate: {result.delta_id_switch_rate:+.4f} ({'improved' if result.id_switch_improved else 'no change'})")

    return result


def save_results(results: Dict, output_path: Path) -> None:
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Face Alignment Evaluation Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate single episode (reads existing artifacts)
    python -m tools.experiments.face_alignment_eval --episode-id rhoslc-s06e01

    # Compare two manifest directories
    python -m tools.experiments.face_alignment_eval --episode-id ep1 \\
        --baseline-dir /path/to/baseline --aligned-dir /path/to/aligned

    # Output location
    python -m tools.experiments.face_alignment_eval --episode-id ep1 \\
        --output-dir data/experiments/my_eval
        """,
    )

    parser.add_argument(
        "--episode-id",
        nargs="+",
        required=True,
        help="Episode ID(s) to evaluate",
    )

    parser.add_argument(
        "--alignment-enabled",
        choices=["true", "false", "both"],
        default="true",
        help="Alignment mode: true, false, or both (compare)",
    )

    parser.add_argument(
        "--manifest-dir",
        type=Path,
        help="Override manifest directory (default: data/manifests/{episode_id})",
    )

    parser.add_argument(
        "--baseline-dir",
        type=Path,
        help="Baseline manifest directory for comparison mode",
    )

    parser.add_argument(
        "--aligned-dir",
        type=Path,
        help="Aligned manifest directory for comparison mode",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/face_alignment_eval"),
        help="Output directory for results",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Frames per second (default: 24.0)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Face Alignment Evaluation Experiment")
    logger.info(f"Episodes: {args.episode_id}")
    logger.info(f"Mode: alignment_enabled={args.alignment_enabled}")

    all_results = []

    for episode_id in args.episode_id:
        # Determine manifest directories
        default_manifest_dir = Path(f"data/manifests/{episode_id}")
        manifest_dir = args.manifest_dir or default_manifest_dir

        if args.alignment_enabled == "both":
            # Comparison mode
            baseline_dir = args.baseline_dir or manifest_dir
            aligned_dir = args.aligned_dir or manifest_dir

            if baseline_dir == aligned_dir:
                logger.warning(
                    "Baseline and aligned dirs are the same. "
                    "For true comparison, run pipeline twice with different settings."
                )

            result = run_comparison(episode_id, baseline_dir, aligned_dir, args.fps)
            all_results.append(result.to_dict())

        else:
            # Single mode
            alignment_enabled = args.alignment_enabled == "true"
            metrics = run_evaluation(episode_id, manifest_dir, alignment_enabled, args.fps)
            all_results.append(metrics.to_dict())

    # Save results
    output_path = args.output_dir / f"eval_{'_'.join(args.episode_id)}.json"
    save_results(
        {
            "experiment": "face_alignment_eval",
            "mode": args.alignment_enabled,
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
        },
        output_path,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for result in all_results:
        ep_id = result.get("episode_id", "unknown")
        print(f"\nEpisode: {ep_id}")

        if "deltas" in result:
            # Comparison mode
            print("  Mode: Comparison (baseline vs aligned)")
            deltas = result["deltas"]
            print(f"  Delta embedding jitter: {deltas['embedding_jitter']:+.6f}")
            print(f"  Delta ID switch rate: {deltas['id_switch_rate_per_minute']:+.4f}/min")
            print(f"  Delta avg track length: {deltas['avg_track_length']:+.2f}")

            improvements = result.get("improvements", {})
            if improvements.get("jitter_improved"):
                print("  [+] Jitter IMPROVED")
            if improvements.get("id_switch_improved"):
                print("  [+] ID switch rate IMPROVED")
            if improvements.get("track_length_improved"):
                print("  [+] Track length IMPROVED")
        else:
            # Single mode
            mode = "aligned" if result.get("alignment_enabled") else "baseline"
            print(f"  Mode: {mode}")
            emb = result.get("embedding_quality", {})
            ids = result.get("id_switch_metrics", {})
            tracks = result.get("track_metrics", {})
            print(f"  Embedding jitter: {emb.get('mean_embedding_jitter', 0):.6f}")
            print(f"  ID switch rate: {ids.get('id_switch_rate_per_minute', 0):.4f}/min")
            print(f"  Avg track length: {tracks.get('avg_track_length', 0):.2f}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
