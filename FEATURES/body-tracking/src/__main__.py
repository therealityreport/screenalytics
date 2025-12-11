"""
CLI entrypoint for body tracking sandbox.

Usage:
    python -m FEATURES.body_tracking --episode-id EP_ID [options]
"""

import argparse
import logging
import sys
from pathlib import Path

from .body_tracking_runner import BodyTrackingRunner


def main():
    parser = argparse.ArgumentParser(
        description="Body Tracking + Re-ID Fusion Sandbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline on episode
    python -m FEATURES.body_tracking --episode-id rhoslc-s06e01

    # Detection only
    python -m FEATURES.body_tracking --episode-id rhoslc-s06e01 --stage detect

    # Tracking + embeddings
    python -m FEATURES.body_tracking --episode-id rhoslc-s06e01 --stage track

    # Track fusion only (requires face tracks)
    python -m FEATURES.body_tracking --episode-id rhoslc-s06e01 --stage fuse

    # Screen-time comparison
    python -m FEATURES.body_tracking --episode-id rhoslc-s06e01 --stage compare
        """,
    )

    parser.add_argument(
        "--episode-id",
        required=True,
        help="Episode ID to process (e.g., rhoslc-s06e01)",
    )

    parser.add_argument(
        "--stage",
        choices=["detect", "track", "embed", "fuse", "compare", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline/body_detection.yaml"),
        help="Path to body detection config",
    )

    parser.add_argument(
        "--fusion-config",
        type=Path,
        default=Path("config/pipeline/track_fusion.yaml"),
        help="Path to track fusion config",
    )

    parser.add_argument(
        "--video-path",
        type=Path,
        help="Override video path (default: from episode manifest)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory (default: data/manifests/{ep_id}/body_tracking/)",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Device for inference (default: auto)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip stages that have existing output files",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("body_tracking")

    logger.info(f"Body Tracking Sandbox - Episode: {args.episode_id}")
    logger.info(f"Stage: {args.stage}")

    if args.dry_run:
        logger.info("[DRY RUN] Would execute pipeline with:")
        logger.info(f"  Config: {args.config}")
        logger.info(f"  Fusion config: {args.fusion_config}")
        logger.info(f"  Device: {args.device}")
        return 0

    try:
        runner = BodyTrackingRunner(
            episode_id=args.episode_id,
            config_path=args.config,
            fusion_config_path=args.fusion_config,
            video_path=args.video_path,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            skip_existing=args.skip_existing,
        )

        if args.stage == "all":
            runner.run_full_pipeline()
        elif args.stage == "detect":
            runner.run_detection()
        elif args.stage == "track":
            runner.run_tracking()
        elif args.stage == "embed":
            runner.run_embedding()
        elif args.stage == "fuse":
            runner.run_fusion()
        elif args.stage == "compare":
            runner.run_comparison()

        logger.info("Pipeline completed successfully")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
