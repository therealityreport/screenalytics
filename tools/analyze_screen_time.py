#!/usr/bin/env python3
"""Analyze screen time from cast-linked faces and tracks."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.api.services.screentime import ScreenTimeAnalyzer, ScreenTimeConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)


def emit_progress(phase: str, message: str, **kwargs):
    """Emit progress JSON to stdout for job monitoring."""
    progress = {
        "phase": phase,
        "message": message,
        **kwargs,
    }
    print(json.dumps(progress), flush=True)


def load_config(config_path: Path | None = None) -> dict:
    """Load pipeline config from YAML."""
    if config_path is None:
        config_path = REPO_ROOT / "config" / "pipeline" / "screen_time_v2.yaml"

    if not config_path.exists():
        LOGGER.warning(f"Config file not found: {config_path}, using defaults")
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze per-cast screen time from faces and tracks")
    parser.add_argument("--ep-id", required=True, help="Episode identifier (e.g., rhobh-s05e17)")
    parser.add_argument("--quality-min", type=float, help="Minimum face quality threshold (0.0-1.0)")
    parser.add_argument("--gap-tolerance-s", type=float, help="Gap tolerance in seconds")
    parser.add_argument(
        "--use-video-decode",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        help="Use video decode for timestamps (true/false)",
    )
    parser.add_argument("--config", type=Path, help="Path to custom config YAML")
    parser.add_argument("--progress-file", type=Path, help="Path to write progress JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for screen time analysis."""
    args = parse_args(argv)

    try:
        emit_progress("init", f"Starting screen time analysis for {args.ep_id}")

        # Load config
        config_dict = load_config(args.config)

        # Apply CLI overrides
        if args.quality_min is not None:
            config_dict["quality_min"] = args.quality_min
        if args.gap_tolerance_s is not None:
            config_dict["gap_tolerance_s"] = args.gap_tolerance_s
        if args.use_video_decode is not None:
            config_dict["use_video_decode"] = args.use_video_decode

        config = ScreenTimeConfig(
            quality_min=config_dict.get("quality_min", 0.7),
            gap_tolerance_s=config_dict.get("gap_tolerance_s", 0.5),
            use_video_decode=config_dict.get("use_video_decode", True),
        )

        LOGGER.info(
            f"Config: quality_min={config.quality_min}, "
            f"gap_tolerance_s={config.gap_tolerance_s}, "
            f"use_video_decode={config.use_video_decode}"
        )

        emit_progress("loading", "Loading episode artifacts and people data")

        # Run analyzer
        analyzer = ScreenTimeAnalyzer(config)
        metrics_data = analyzer.analyze_episode(args.ep_id)

        emit_progress(
            "analyzing",
            f"Analyzed {len(metrics_data.get('metrics', []))} cast members",
            cast_count=len(metrics_data.get("metrics", [])),
        )

        # Write outputs
        emit_progress("writing", "Writing screen time outputs")
        json_path, csv_path = analyzer.write_outputs(args.ep_id, metrics_data)

        emit_progress(
            "done",
            f"Screen time analysis complete",
            json_path=str(json_path),
            csv_path=str(csv_path),
            cast_count=len(metrics_data.get("metrics", [])),
        )

        LOGGER.info(f"Analysis complete: {json_path}, {csv_path}")
        return 0

    except FileNotFoundError as exc:
        emit_progress("error", f"Required artifact not found: {exc}")
        LOGGER.error(f"File not found: {exc}")
        return 1

    except ValueError as exc:
        emit_progress("error", f"Invalid input: {exc}")
        LOGGER.error(f"Invalid input: {exc}")
        return 1

    except Exception as exc:
        emit_progress("error", f"Screen time analysis failed: {exc}")
        LOGGER.exception("Screen time analysis failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
