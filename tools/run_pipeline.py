#!/usr/bin/env python
"""Thin CLI wrapper over the episode processing engine.

This script provides a simple command-line interface to run the full episode
pipeline using py_screenalytics.pipeline.run_episode().

Usage:
    python tools/run_pipeline.py --ep-id rhobh-s05e14 --video /path/to/video.mp4
    python tools/run_pipeline.py --ep-id rhobh-s05e14 --video /path/to/video.mp4 --device coreml --stride 4
    python tools/run_pipeline.py --ep-id rhobh-s05e14 --video /path/to/video.mp4 --reuse-detections --reuse-embeddings

For single-stage runs, use the original tools/episode_run.py with --faces-embed or --cluster flags.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Apply CPU limits before importing ML libraries
from apps.common.cpu_limits import apply_global_cpu_limits
apply_global_cpu_limits()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the full episode processing pipeline (detect → embed → cluster)."
    )

    # Required arguments
    parser.add_argument(
        "--ep-id",
        required=True,
        help="Episode identifier (e.g., 'rhobh-s05e14')",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to source video file",
    )

    # Device settings
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "coreml", "cuda"],
        default="auto",
        help="Execution device (default: auto)",
    )
    parser.add_argument(
        "--embed-device",
        choices=["auto", "cpu", "mps", "coreml", "cuda"],
        default=None,
        help="Separate device for face embedding (defaults to --device)",
    )

    # Detection settings
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=0.65,
        help="Detection confidence threshold (default: 0.65)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride for detection (default: 1 = every frame)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional target FPS for downsampling",
    )

    # Tracking settings
    parser.add_argument(
        "--track-buffer",
        type=int,
        default=15,
        help="ByteTrack track buffer (default: 15)",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=60,
        help="Maximum frame gap before splitting a track (default: 60)",
    )

    # Clustering settings
    parser.add_argument(
        "--cluster-thresh",
        type=float,
        default=0.75,
        help="Clustering similarity threshold (default: 0.75)",
    )
    parser.add_argument(
        "--min-identity-sim",
        type=float,
        default=0.50,
        help="Minimum similarity to identity centroid (default: 0.50)",
    )

    # Export settings
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save per-track face crops",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save full frame JPGs",
    )
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=256,
        help="Square thumbnail size (default: 256)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for exports (default: 85)",
    )

    # Output settings
    parser.add_argument(
        "--out-root",
        help="Data root override (defaults to SCREENALYTICS_DATA_ROOT or ./data)",
    )
    parser.add_argument(
        "--progress-file",
        help="Progress JSON file to update during processing",
    )

    # Dev-mode options
    parser.add_argument(
        "--reuse-detections",
        action="store_true",
        help="Skip detect_track if artifacts already exist",
    )
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Skip faces_embed if artifacts already exist",
    )

    # Output format
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Validate video path
    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        return 1

    # Set up data root
    data_root = None
    if args.out_root:
        data_root = Path(args.out_root).expanduser()
        os.environ["SCREENALYTICS_DATA_ROOT"] = str(data_root)

    # Import the engine
    from py_screenalytics.pipeline import run_episode, EpisodeRunConfig

    # Build config
    config = EpisodeRunConfig(
        device=args.device,
        embed_device=args.embed_device,
        det_thresh=args.det_thresh,
        stride=args.stride,
        target_fps=args.fps,
        track_buffer=args.track_buffer,
        max_gap=args.max_gap,
        cluster_thresh=args.cluster_thresh,
        min_identity_sim=args.min_identity_sim,
        save_crops=args.save_crops,
        save_frames=args.save_frames,
        thumb_size=args.thumb_size,
        jpeg_quality=args.jpeg_quality,
        data_root=data_root,
        progress_file=Path(args.progress_file) if args.progress_file else None,
        reuse_detections=args.reuse_detections,
        reuse_embeddings=args.reuse_embeddings,
    )

    if not args.quiet:
        print(f"[run_pipeline] Starting episode={args.ep_id} device={config.device} stride={config.stride}", file=sys.stderr)

    # Run the pipeline
    result = run_episode(args.ep_id, video_path, config)

    # Output results
    if args.json:
        print(result.to_json())
    else:
        if result.success:
            print(f"\n[run_pipeline] ✅ Success!", file=sys.stderr)
            print(f"  Episode:     {result.episode_id}", file=sys.stderr)
            print(f"  Tracks:      {result.tracks_count}", file=sys.stderr)
            print(f"  Faces:       {result.faces_count}", file=sys.stderr)
            print(f"  Identities:  {result.identities_count}", file=sys.stderr)
            print(f"  Runtime:     {result.runtime_sec:.1f}s", file=sys.stderr)

            # Per-stage breakdown
            print(f"\n  Stages:", file=sys.stderr)
            for stage in result.stages:
                skipped = stage.metadata.get("skipped", False) if stage.metadata else False
                if skipped:
                    print(f"    {stage.stage}: SKIPPED ({stage.metadata.get('reason', '')})", file=sys.stderr)
                else:
                    print(f"    {stage.stage}: {stage.runtime_sec:.1f}s", file=sys.stderr)

            # Print artifact paths
            print(f"\n  Artifacts:", file=sys.stderr)
            for name, path in result.artifacts.items():
                print(f"    {name}: {path}", file=sys.stderr)
        else:
            print(f"\n[run_pipeline] ❌ Failed: {result.error}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
