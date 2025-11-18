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
CONFIG_KEYS = (
    "quality_min",
    "gap_tolerance_s",
    "use_video_decode",
    "screen_time_mode",
    "edge_padding_s",
    "track_coverage_min",
)


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


def resolve_config(raw_config: dict, preset_override: str | None = None) -> dict:
    """Resolve the effective config dictionary, honoring presets."""
    if raw_config is None:
        raw_config = {}

    presets = raw_config.get("screen_time_presets") or {}
    preset_name = (
        preset_override
        or raw_config.get("preset")
        or raw_config.get("screen_time_preset")
    )

    resolved: dict = {}
    if preset_name:
        preset_values = presets.get(preset_name)
        if preset_values:
            resolved.update(preset_values)
        else:
            LOGGER.warning(
                "Requested screen time preset '%s' not found. Falling back to inline values.",
                preset_name,
            )
    elif presets.get("default"):
        resolved.update(presets["default"])

    for key in CONFIG_KEYS:
        if key in raw_config:
            resolved[key] = raw_config[key]

    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze per-cast screen time from faces and tracks"
    )
    parser.add_argument(
        "--ep-id", required=True, help="Episode identifier (e.g., rhobh-s05e17)"
    )
    parser.add_argument(
        "--quality-min", type=float, help="Minimum face quality threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--gap-tolerance-s", type=float, help="Gap tolerance in seconds"
    )
    parser.add_argument(
        "--use-video-decode",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        help="Use video decode for timestamps (true/false)",
    )
    parser.add_argument(
        "--screen-time-mode",
        choices=["faces", "tracks"],
        help="Interval calculation mode",
    )
    parser.add_argument(
        "--edge-padding-s",
        type=float,
        help="Edge padding applied to each interval (seconds)",
    )
    parser.add_argument(
        "--track-coverage-min",
        type=float,
        help="Minimum detection coverage required when screen_time_mode=tracks",
    )
    parser.add_argument(
        "--preset", help="Name of the screen time preset defined in the YAML config"
    )
    parser.add_argument("--config", type=Path, help="Path to custom config YAML")
    parser.add_argument(
        "--progress-file", type=Path, help="Path to write progress JSON"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for screen time analysis."""
    args = parse_args(argv)

    try:
        emit_progress("init", f"Starting screen time analysis for {args.ep_id}")

        # Load config
        raw_config = load_config(args.config)
        resolved = resolve_config(raw_config, args.preset)

        # Apply CLI overrides
        cli_overrides = {
            "quality_min": args.quality_min,
            "gap_tolerance_s": args.gap_tolerance_s,
            "use_video_decode": args.use_video_decode,
            "screen_time_mode": args.screen_time_mode,
            "edge_padding_s": args.edge_padding_s,
            "track_coverage_min": args.track_coverage_min,
        }
        for key, value in cli_overrides.items():
            if value is not None:
                resolved[key] = value

        config = ScreenTimeConfig(
            quality_min=resolved.get("quality_min", 0.7),
            gap_tolerance_s=resolved.get("gap_tolerance_s", 0.5),
            use_video_decode=resolved.get("use_video_decode", True),
            screen_time_mode=resolved.get("screen_time_mode", "faces"),
            edge_padding_s=resolved.get("edge_padding_s", 0.0),
            track_coverage_min=resolved.get("track_coverage_min", 0.0),
        )

        LOGGER.info(
            "Config: preset=%s quality_min=%.2f gap_tolerance_s=%.2f use_video_decode=%s mode=%s edge_padding_s=%.2f track_coverage_min=%.2f",
            args.preset or raw_config.get("preset"),
            config.quality_min,
            config.gap_tolerance_s,
            config.use_video_decode,
            config.screen_time_mode,
            config.edge_padding_s,
            config.track_coverage_min,
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
            "Screen time analysis complete",
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
