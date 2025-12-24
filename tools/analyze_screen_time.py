#!/usr/bin/env python3
"""Analyze screen time from cast-linked faces and tracks."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.api.services.screentime import ScreenTimeAnalyzer, ScreenTimeConfig
from py_screenalytics import run_layout
from py_screenalytics.episode_status import (
    BlockedReason,
    blocked_update_needed,
    stage_artifacts,
    write_stage_blocked,
    write_stage_failed,
    write_stage_finished,
    write_stage_started,
)
from py_screenalytics.run_gates import GateReason, check_prereqs
from py_screenalytics.run_manifests import StageBlockedInfo, StageErrorInfo, write_stage_manifest
from py_screenalytics.run_logs import append_log

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


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _duration_s(started_at: str, ended_at: str) -> float | None:
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    delta = (end - start).total_seconds()
    return delta if delta >= 0 else None


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
    preset_name = preset_override or raw_config.get("preset") or raw_config.get("screen_time_preset")

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
    parser = argparse.ArgumentParser(description="Analyze per-cast screen time from faces and tracks")
    parser.add_argument("--ep-id", required=True, help="Episode identifier (e.g., rhobh-s05e17)")
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional pipeline run identifier. When provided, artifacts are read only from "
            "data/manifests/{ep_id}/runs/{run_id}/ and outputs are also written under that run."
        ),
    )
    parser.add_argument("--quality-min", type=float, help="Minimum face quality threshold (0.0-1.0)")
    parser.add_argument("--gap-tolerance-s", type=float, help="Gap tolerance in seconds")
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
    parser.add_argument("--preset", help="Name of the screen time preset defined in the YAML config")
    parser.add_argument("--config", type=Path, help="Path to custom config YAML")
    parser.add_argument("--progress-file", type=Path, help="Path to write progress JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for screen time analysis."""
    args = parse_args(argv)
    started_at = _utcnow_iso()

    try:
        run_id = run_layout.normalize_run_id(args.run_id) if args.run_id else None
        if run_id:
            try:
                append_log(args.ep_id, run_id, "screentime", "INFO", "stage started", progress=0.0)
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log screentime start: %s", log_exc)
        if run_id:
            gate = check_prereqs("screentime", args.ep_id, run_id)
            if not gate.ok:
                reasons = gate.reasons or []
                primary = reasons[0] if reasons else None
                blocked_reason = BlockedReason(
                    code=primary.code if primary else "blocked",
                    message=primary.message if primary else "Stage blocked by prerequisites",
                    details={
                        "reasons": [reason.as_dict() for reason in reasons],
                        "suggested_actions": list(gate.suggested_actions),
                    },
                )
                blocked_info = StageBlockedInfo(reasons=list(reasons), suggested_actions=list(gate.suggested_actions))
                should_block = blocked_update_needed(args.ep_id, run_id, "screentime", blocked_reason)
                if should_block:
                    try:
                        write_stage_blocked(args.ep_id, run_id, "screentime", blocked_reason)
                    except Exception as status_exc:  # pragma: no cover - best effort status update
                        LOGGER.warning("[screentime] Failed to mark screentime blocked: %s", status_exc)
                    try:
                        append_log(
                            args.ep_id,
                            run_id,
                            "screentime",
                            "WARNING",
                            "stage blocked",
                            progress=0.0,
                            meta={
                                "reason_code": blocked_reason.code,
                                "reason_message": blocked_reason.message,
                                "suggested_actions": list(gate.suggested_actions),
                            },
                        )
                    except Exception as log_exc:  # pragma: no cover - best effort log write
                        LOGGER.debug("[run_logs] Failed to log screentime blocked: %s", log_exc)
                    try:
                        write_stage_manifest(
                            args.ep_id,
                            run_id,
                            "screentime",
                            "BLOCKED",
                            started_at=started_at,
                            finished_at=_utcnow_iso(),
                            duration_s=None,
                            blocked=blocked_info,
                        )
                    except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                        LOGGER.warning("[screentime] Failed to write screentime blocked manifest: %s", manifest_exc)
                emit_progress("blocked", blocked_reason.message, run_id=run_id)
                return 1
        if run_id:
            try:
                write_stage_started(
                    args.ep_id,
                    run_id,
                    "screentime",
                    started_at=datetime.fromisoformat(started_at.replace("Z", "+00:00")),
                )
            except Exception as exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[screentime] Failed to mark screentime start: %s", exc)
        emit_progress("init", f"Starting screen time analysis for {args.ep_id}", run_id=run_id)

        # Load config
        raw_config = load_config(args.config)
        resolved = resolve_config(raw_config, args.preset)
        if run_id:
            try:
                append_log(
                    args.ep_id,
                    run_id,
                    "screentime",
                    "INFO",
                    "config resolved",
                    progress=20.0,
                    meta={"preset": args.preset or raw_config.get("preset")},
                )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log screentime config: %s", log_exc)

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

        emit_progress("loading", "Loading episode artifacts and people data", run_id=run_id)

        # Run analyzer
        analyzer = ScreenTimeAnalyzer(config)
        if run_id:
            try:
                append_log(args.ep_id, run_id, "screentime", "INFO", "analysis started", progress=40.0)
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log screentime analyze start: %s", log_exc)
        metrics_data = analyzer.analyze_episode(args.ep_id, run_id=run_id)
        if run_id:
            try:
                append_log(
                    args.ep_id,
                    run_id,
                    "screentime",
                    "INFO",
                    "analysis complete",
                    progress=70.0,
                    meta={"cast_count": len(metrics_data.get("metrics", []))},
                )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log screentime analyze complete: %s", log_exc)

        emit_progress(
            "analyzing",
            f"Analyzed {len(metrics_data.get('metrics', []))} cast members",
            cast_count=len(metrics_data.get("metrics", [])),
            run_id=run_id,
        )

        # Write outputs
        emit_progress("writing", "Writing screen time outputs", run_id=run_id)
        json_path, csv_path = analyzer.write_outputs(args.ep_id, metrics_data, run_id=run_id)
        if run_id:
            try:
                append_log(
                    args.ep_id,
                    run_id,
                    "screentime",
                    "INFO",
                    "outputs written",
                    progress=90.0,
                    meta={"json_path": str(json_path), "csv_path": str(csv_path)},
                )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log screentime outputs: %s", log_exc)

        emit_progress(
            "done",
            "Screen time analysis complete",
            json_path=str(json_path),
            csv_path=str(csv_path),
            cast_count=len(metrics_data.get("metrics", [])),
            run_id=run_id,
        )

        if run_id:
            finished_at = _utcnow_iso()
            metrics_count = len(metrics_data.get("metrics", []))
            try:
                write_stage_finished(
                    args.ep_id,
                    run_id,
                    "screentime",
                    counts={"metrics": metrics_count},
                    metrics={
                        "metrics_count": metrics_count,
                        "body_tracking_enabled": (
                            metrics_data.get("metadata", {}).get("body_tracking_enabled")
                            if isinstance(metrics_data.get("metadata"), dict)
                            else None
                        ),
                    },
                    artifact_paths={
                        entry["label"]: entry["path"]
                        for entry in stage_artifacts(args.ep_id, run_id, "screentime")
                        if isinstance(entry, dict) and entry.get("exists")
                    },
                )
            except Exception as exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[screentime] Failed to mark screentime success: %s", exc)
            try:
                append_log(
                    args.ep_id,
                    run_id,
                    "screentime",
                    "INFO",
                    "stage finished",
                    progress=100.0,
                    meta={"metrics_count": metrics_count},
                )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log screentime finish: %s", log_exc)
            try:
                write_stage_manifest(
                    args.ep_id,
                    run_id,
                    "screentime",
                    "SUCCESS",
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_s=None,
                    counts={"metrics": metrics_count},
                    thresholds={
                        "quality_min": config.quality_min,
                        "gap_tolerance_s": config.gap_tolerance_s,
                        "use_video_decode": config.use_video_decode,
                        "screen_time_mode": config.screen_time_mode,
                        "edge_padding_s": config.edge_padding_s,
                        "track_coverage_min": config.track_coverage_min,
                    },
                    artifacts={
                        entry["label"]: entry["path"]
                        for entry in stage_artifacts(args.ep_id, run_id, "screentime")
                        if isinstance(entry, dict) and entry.get("exists")
                    },
                )
            except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                LOGGER.warning("[screentime] Failed to write screentime success manifest: %s", manifest_exc)

        LOGGER.info(f"Analysis complete: {json_path}, {csv_path}")
        return 0

    except FileNotFoundError as exc:
        emit_progress("error", f"Required artifact not found: {exc}", run_id=args.run_id)
        LOGGER.error(f"File not found: {exc}")
        if args.run_id:
            run_id_norm = run_layout.normalize_run_id(args.run_id)
            blocked_reason = BlockedReason(
                code="missing_prereq",
                message=str(exc),
                details=None,
            )
            should_block = blocked_update_needed(args.ep_id, run_id_norm, "screentime", blocked_reason)
            try:
                if should_block:
                    write_stage_blocked(
                        args.ep_id,
                        run_id_norm,
                        "screentime",
                        blocked_reason,
                    )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[screentime] Failed to mark screentime blocked: %s", status_exc)
            try:
                if should_block:
                    append_log(
                        args.ep_id,
                        run_id_norm,
                        "screentime",
                        "WARNING",
                        "stage blocked",
                        progress=100.0,
                        meta={"error_message": str(exc)},
                    )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log screentime blocked: %s", log_exc)
            try:
                if should_block:
                    write_stage_manifest(
                        args.ep_id,
                        run_id_norm,
                        "screentime",
                        "BLOCKED",
                        started_at=started_at,
                        finished_at=_utcnow_iso(),
                        duration_s=None,
                        blocked=StageBlockedInfo(
                            reasons=[GateReason(code="missing_artifact", message=str(exc), details=None)],
                            suggested_actions=["Run upstream stages to generate screentime prerequisites."],
                        ),
                    )
            except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                LOGGER.warning("[screentime] Failed to write screentime blocked manifest: %s", manifest_exc)
        return 1

    except ValueError as exc:
        emit_progress("error", f"Invalid input: {exc}", run_id=args.run_id)
        LOGGER.error(f"Invalid input: {exc}")
        if args.run_id:
            run_id_norm = run_layout.normalize_run_id(args.run_id)
            try:
                write_stage_failed(
                    args.ep_id,
                    run_id_norm,
                    "screentime",
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[screentime] Failed to mark screentime failure: %s", status_exc)
            try:
                append_log(
                    args.ep_id,
                    run_id_norm,
                    "screentime",
                    "ERROR",
                    "stage failed",
                    progress=100.0,
                    meta={"error_code": type(exc).__name__, "error_message": str(exc)},
                )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log screentime failure: %s", log_exc)
            try:
                write_stage_manifest(
                    args.ep_id,
                    run_id_norm,
                    "screentime",
                    "FAILED",
                    started_at=started_at,
                    finished_at=_utcnow_iso(),
                    duration_s=None,
                    error=StageErrorInfo(code=type(exc).__name__, message=str(exc)),
                )
            except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                LOGGER.warning("[screentime] Failed to write screentime failed manifest: %s", manifest_exc)
        return 1

    except Exception as exc:
        emit_progress("error", f"Screen time analysis failed: {exc}", run_id=args.run_id)
        LOGGER.exception("Screen time analysis failed")
        if args.run_id:
            run_id_norm = run_layout.normalize_run_id(args.run_id)
            try:
                write_stage_failed(
                    args.ep_id,
                    run_id_norm,
                    "screentime",
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception as status_exc:  # pragma: no cover - best effort status update
                LOGGER.warning("[screentime] Failed to mark screentime failure: %s", status_exc)
            try:
                append_log(
                    args.ep_id,
                    run_id_norm,
                    "screentime",
                    "ERROR",
                    "stage failed",
                    progress=100.0,
                    meta={"error_code": type(exc).__name__, "error_message": str(exc)},
                )
            except Exception as log_exc:  # pragma: no cover - best effort log write
                LOGGER.debug("[run_logs] Failed to log screentime failure: %s", log_exc)
            try:
                write_stage_manifest(
                    args.ep_id,
                    run_id_norm,
                    "screentime",
                    "FAILED",
                    started_at=started_at,
                    finished_at=_utcnow_iso(),
                    duration_s=None,
                    error=StageErrorInfo(code=type(exc).__name__, message=str(exc)),
                )
            except Exception as manifest_exc:  # pragma: no cover - best effort manifest write
                LOGGER.warning("[screentime] Failed to write screentime failed manifest: %s", manifest_exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
