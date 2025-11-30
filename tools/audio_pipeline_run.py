#!/usr/bin/env python3
"""CLI entry point for audio pipeline with streaming output.

This script runs the audio pipeline and outputs progress/logs to stdout
for real-time streaming to the UI.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging():
    """Configure logging to output to stdout for streaming."""
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(name)s: %(message)s',
        stream=sys.stdout,
    )
    # Make sure logs are flushed immediately
    for handler in logging.root.handlers:
        handler.flush = lambda: sys.stdout.flush()


def emit_progress(phase: str, progress: float, message: str = "", **extra):
    """Emit a JSON progress line for the UI to parse."""
    data = {
        "phase": phase,
        "progress": round(progress, 2),
        "message": message,
        "timestamp": time.time(),
        **extra,
    }
    print(json.dumps(data), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run audio pipeline for an episode")
    parser.add_argument("--ep-id", required=True, help="Episode ID")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts")
    parser.add_argument("--asr-provider", default="openai_whisper",
                       choices=["openai_whisper", "gemini"],
                       help="ASR provider")
    parser.add_argument("--progress-file", help="Path to write progress JSON")

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("audio_pipeline_run")

    logger.info(f"Starting audio pipeline for {args.ep_id}")
    emit_progress("init", 0, f"Starting audio pipeline for {args.ep_id}")

    try:
        from py_screenalytics.audio.episode_audio_pipeline import (
            run_episode_audio_pipeline,
            AUDIO_PIPELINE_STEPS,
        )

        # Create a progress callback that emits to stdout
        def progress_callback(step: str, progress: float, message: str = ""):
            step_info = AUDIO_PIPELINE_STEPS.get(step, {})
            step_name = step_info.get("name", step)
            step_order = step_info.get("order", 0)
            total_steps = len(AUDIO_PIPELINE_STEPS)

            # Calculate overall progress
            completed_weight = sum(
                s["weight"] for s in AUDIO_PIPELINE_STEPS.values()
                if s["order"] < step_order
            )
            current_weight = step_info.get("weight", 10)
            total_weight = sum(s["weight"] for s in AUDIO_PIPELINE_STEPS.values())
            overall = (completed_weight + current_weight * progress) / total_weight

            emit_progress(
                step,
                overall,
                message or f"{step_name}: {progress*100:.0f}%",
                step_name=step_name,
                step_progress=progress,
                step_order=step_order,
                total_steps=total_steps,
            )

            # Also write to progress file if specified
            if args.progress_file:
                try:
                    progress_data = {
                        "step": step,
                        "step_name": step_name,
                        "step_progress": progress,
                        "overall_progress": overall,
                        "message": message,
                        "step_order": step_order,
                        "total_steps": total_steps,
                    }
                    Path(args.progress_file).write_text(json.dumps(progress_data))
                except Exception:
                    pass

        # Run the pipeline
        result = run_episode_audio_pipeline(
            args.ep_id,
            overwrite=args.overwrite,
            asr_provider=args.asr_provider,
            progress_callback=progress_callback,
        )

        if result.status == "succeeded":
            emit_progress("complete", 1.0, "Audio pipeline completed successfully",
                         voice_clusters=result.voice_cluster_count,
                         labeled_voices=result.labeled_voices,
                         unlabeled_voices=result.unlabeled_voices)
            logger.info(f"Audio pipeline completed: {result.voice_cluster_count} voice clusters")
            sys.exit(0)
        else:
            emit_progress("error", 0, f"Audio pipeline failed: {result.error}")
            logger.error(f"Audio pipeline failed: {result.error}")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Audio pipeline crashed: {e}")
        emit_progress("error", 0, f"Audio pipeline crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
