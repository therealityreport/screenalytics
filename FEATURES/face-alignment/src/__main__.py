"""
CLI entrypoint for face alignment sandbox.

Usage:
    python -m FEATURES.face_alignment --episode-id EP_ID [options]
"""

import argparse
import logging
import sys
from pathlib import Path

from .face_alignment_runner import FaceAlignmentRunner


def main():
    parser = argparse.ArgumentParser(
        description="Face Alignment Sandbox (FAN 68-point landmarks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full alignment pipeline on episode
    python -m FEATURES.face_alignment --episode-id rhoslc-s06e01

    # Alignment only (no crop export)
    python -m FEATURES.face_alignment --episode-id rhoslc-s06e01 --stage align

    # Export aligned crops
    python -m FEATURES.face_alignment --episode-id rhoslc-s06e01 --stage export

    # With custom stride (every 5th frame)
    python -m FEATURES.face_alignment --episode-id rhoslc-s06e01 --stride 5
        """,
    )

    parser.add_argument(
        "--episode-id",
        required=True,
        help="Episode ID to process (e.g., rhoslc-s06e01)",
    )

    parser.add_argument(
        "--stage",
        choices=["align", "export", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline/face_alignment.yaml"),
        help="Path to face alignment config",
    )

    parser.add_argument(
        "--video-path",
        type=Path,
        help="Override video path (default: from episode manifest)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory (default: data/manifests/{ep_id}/face_alignment/)",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for inference (default: auto)",
    )

    parser.add_argument(
        "--stride",
        type=int,
        help="Process every Nth frame (default: from config or 1)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config",
    )

    parser.add_argument(
        "--export-crops",
        action="store_true",
        help="Export aligned crop images to disk",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if output artifacts already exist",
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
    logger = logging.getLogger("face_alignment")

    logger.info(f"Face Alignment Sandbox - Episode: {args.episode_id}")
    logger.info(f"Stage: {args.stage}")

    if args.dry_run:
        logger.info("[DRY RUN] Would execute pipeline with:")
        logger.info(f"  Config: {args.config}")
        logger.info(f"  Device: {args.device}")
        logger.info(f"  Export crops: {args.export_crops}")
        return 0

    try:
        runner = FaceAlignmentRunner(
            episode_id=args.episode_id,
            config_path=args.config,
            video_path=args.video_path,
            output_dir=args.output_dir,
            device=args.device,
            stride=args.stride,
            batch_size=args.batch_size,
            export_crops=args.export_crops,
            skip_existing=args.skip_existing,
        )

        if args.stage == "all":
            runner.run_full_pipeline()
        elif args.stage == "align":
            runner.run_alignment()
        elif args.stage == "export":
            runner.run_export()

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
