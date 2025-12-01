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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # dotenv not installed, rely on system environment


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


def _get_audio_paths(ep_id: str):
    """Get standard audio artifact paths for an episode."""
    import os
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
    audio_dir = data_root / "audio" / ep_id
    manifests_dir = data_root / "manifests" / ep_id

    return {
        "audio_vocals": audio_dir / "episode_vocals.wav",
        "audio_vocals_enhanced": audio_dir / "episode_vocals_enhanced.wav",
        "diarization": manifests_dir / "audio_diarization.jsonl",
        "voice_clusters": manifests_dir / "audio_voice_clusters.json",
        "voice_mapping": manifests_dir / "audio_voice_mapping.json",
        "asr_raw": manifests_dir / "audio_asr_raw.jsonl",
    }


def run_diarize_only(args):
    """Re-run only the diarization stage."""
    setup_logging()
    logger = logging.getLogger("audio_pipeline_run")

    ep_id = args.ep_id
    logger.info(f"Starting diarization-only for {ep_id}")
    emit_progress("diarize", 0, f"Starting diarization for {ep_id}",
                 step_name="Diarization", step_order=1, total_steps=1)

    try:
        paths = _get_audio_paths(ep_id)

        # Find input audio
        audio_path = paths["audio_vocals_enhanced"]
        if not audio_path.exists():
            audio_path = paths["audio_vocals"]
        if not audio_path.exists():
            emit_progress("error", 0, "No audio files found. Run full pipeline first.")
            sys.exit(1)

        from py_screenalytics.audio.diarization_pyannote import run_diarization
        from py_screenalytics.audio.episode_audio_pipeline import _load_config

        config = _load_config().diarization

        # Apply num_speakers override
        if args.num_speakers is not None:
            config = config.model_copy(update={"num_speakers": args.num_speakers})
            logger.info(f"Forcing {args.num_speakers} speakers")
            emit_progress("diarize", 0.1, f"Running diarization (forcing {args.num_speakers} speakers)...",
                         step_name="Diarization", step_order=1, total_steps=1)
        else:
            emit_progress("diarize", 0.1, "Running diarization (auto speaker detection)...",
                         step_name="Diarization", step_order=1, total_steps=1)

        segments = run_diarization(audio_path, paths["diarization"], config, overwrite=True)
        speakers = set(s.speaker for s in segments)

        emit_progress("complete", 1.0, f"Diarization complete: {len(speakers)} speakers, {len(segments)} segments",
                     step_name="Complete", step_order=1, total_steps=1,
                     speaker_count=len(speakers), segment_count=len(segments))

        logger.info(f"Diarization complete: {len(speakers)} speakers, {len(segments)} segments")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Diarization failed: {e}")
        emit_progress("error", 0, f"Diarization failed: {e}")
        sys.exit(1)


def run_transcribe_only(args):
    """Re-run only the transcription stage."""
    setup_logging()
    logger = logging.getLogger("audio_pipeline_run")

    ep_id = args.ep_id
    logger.info(f"Starting transcription-only for {ep_id}")
    emit_progress("transcribe", 0, f"Starting transcription for {ep_id}",
                 step_name="Transcription", step_order=1, total_steps=1)

    try:
        paths = _get_audio_paths(ep_id)

        # Find input audio
        audio_path = paths["audio_vocals_enhanced"]
        if not audio_path.exists():
            audio_path = paths["audio_vocals"]
        if not audio_path.exists():
            emit_progress("error", 0, "No audio files found. Run full pipeline first.")
            sys.exit(1)

        from py_screenalytics.audio.asr_openai import transcribe_audio as transcribe_openai
        from py_screenalytics.audio.asr_gemini import transcribe_audio as transcribe_gemini
        from py_screenalytics.audio.episode_audio_pipeline import _load_config

        config = _load_config().asr

        # Update provider if specified
        asr_provider = args.asr_provider
        provider = "gemini_3" if asr_provider in ("gemini", "gemini_3") else "openai_whisper"
        config = config.model_copy(update={"provider": provider})
        transcribe_fn = transcribe_gemini if provider == "gemini_3" else transcribe_openai

        emit_progress("transcribe", 0.2, f"Running transcription with {asr_provider}...",
                     step_name="Transcription", step_order=1, total_steps=1)

        segments = transcribe_fn(audio_path, paths["asr_raw"], config, overwrite=True)

        emit_progress("complete", 1.0, f"Transcription complete: {len(segments)} segments",
                     step_name="Complete", step_order=1, total_steps=1,
                     segment_count=len(segments))

        logger.info(f"Transcription complete: {len(segments)} segments")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Transcription failed: {e}")
        emit_progress("error", 0, f"Transcription failed: {e}")
        sys.exit(1)


def run_voices_only(args):
    """Re-run only the voice clustering stage."""
    setup_logging()
    logger = logging.getLogger("audio_pipeline_run")

    ep_id = args.ep_id
    logger.info(f"Starting voice clustering for {ep_id}")
    emit_progress("voices", 0, f"Starting voice clustering for {ep_id}",
                 step_name="Voice Clustering", step_order=1, total_steps=1)

    try:
        paths = _get_audio_paths(ep_id)

        if not paths["diarization"].exists():
            emit_progress("error", 0, "No diarization found. Run diarization first.")
            sys.exit(1)

        # Find input audio for embeddings
        audio_path = paths["audio_vocals_enhanced"]
        if not audio_path.exists():
            audio_path = paths["audio_vocals"]
        if not audio_path.exists():
            emit_progress("error", 0, "No audio files found. Run full pipeline first.")
            sys.exit(1)

        from py_screenalytics.audio.voice_clusters import cluster_episode_voices
        from py_screenalytics.audio.voice_bank import match_voice_clusters_to_bank
        from py_screenalytics.audio.diarization_pyannote import _load_diarization_manifest
        from py_screenalytics.audio.models import VoiceClusteringConfig, VoiceBankConfig

        # Get show_id from ep_id
        show_id = ep_id.rsplit("-", 1)[0] if "-" in ep_id else ep_id

        # Create clustering config with the threshold
        config = VoiceClusteringConfig(similarity_threshold=args.similarity_threshold)

        emit_progress("voices", 0.2, f"Clustering with threshold {args.similarity_threshold}...",
                     step_name="Voice Clustering", step_order=1, total_steps=1)

        # Load diarization segments
        segments = _load_diarization_manifest(paths["diarization"])

        # Run clustering
        clusters = cluster_episode_voices(
            audio_path,
            segments,
            paths["voice_clusters"],
            config,
            overwrite=True,
        )

        emit_progress("voices", 0.8, f"Matching clusters to voice bank...",
                     step_name="Voice Clustering", step_order=1, total_steps=1)

        # Re-run voice mapping to match clusters to voice bank
        voice_bank_config = VoiceBankConfig()
        match_voice_clusters_to_bank(
            show_id,
            clusters,
            paths["voice_mapping"],
            voice_bank_config,
            args.similarity_threshold,
            overwrite=True,
        )

        emit_progress("complete", 1.0, f"Voice clustering complete: {len(clusters)} clusters",
                     step_name="Complete", step_order=1, total_steps=1,
                     cluster_count=len(clusters))

        logger.info(f"Voice clustering complete: {len(clusters)} clusters")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Voice clustering failed: {e}")
        emit_progress("error", 0, f"Voice clustering failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run audio pipeline for an episode")
    parser.add_argument("--ep-id", required=True, help="Episode ID")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts")
    parser.add_argument("--asr-provider", default="openai_whisper",
                       choices=["openai_whisper", "gemini", "gemini_3"],
                       help="ASR provider (gemini and gemini_3 are equivalent)")
    parser.add_argument("--progress-file", help="Path to write progress JSON")

    # Incremental operation flags
    parser.add_argument("--diarize-only", action="store_true",
                       help="Re-run only diarization (requires existing audio files)")
    parser.add_argument("--transcribe-only", action="store_true",
                       help="Re-run only transcription (requires existing diarization)")
    parser.add_argument("--voices-only", action="store_true",
                       help="Re-run only voice clustering (requires existing diarization)")
    parser.add_argument("--num-speakers", type=int, default=None,
                       help="Force exact speaker count for diarization")
    parser.add_argument("--similarity-threshold", type=float, default=0.30,
                       help="Similarity threshold for voice clustering (lower = fewer clusters)")

    args = parser.parse_args()

    # Handle incremental operations
    if args.diarize_only:
        return run_diarize_only(args)
    if args.transcribe_only:
        return run_transcribe_only(args)
    if args.voices_only:
        return run_voices_only(args)

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
            emit_progress(
                "complete",
                1.0,
                "Audio pipeline completed successfully",
                voice_clusters=result.voice_cluster_count,
                labeled_voices=result.labeled_voices,
                unlabeled_voices=result.unlabeled_voices,
            )
            # Write final progress file so UI doesn't think we're still running
            if args.progress_file:
                try:
                    Path(args.progress_file).write_text(json.dumps({
                        "step": "complete",
                        "step_name": "Complete",
                        "step_progress": 1.0,
                        "overall_progress": 1.0,
                        "message": "Audio pipeline completed successfully",
                        "step_order": len(AUDIO_PIPELINE_STEPS),
                        "total_steps": len(AUDIO_PIPELINE_STEPS),
                        "status": "succeeded",
                    }))
                except Exception:
                    pass

            logger.info(f"Audio pipeline completed: {result.voice_cluster_count} voice clusters")
            sys.exit(0)
        else:
            emit_progress("error", 0, f"Audio pipeline failed: {result.error}")
            if args.progress_file:
                try:
                    Path(args.progress_file).write_text(json.dumps({
                        "step": "error",
                        "step_name": "Error",
                        "step_progress": 0,
                        "overall_progress": 0,
                        "message": f"Audio pipeline failed: {result.error}",
                        "step_order": len(AUDIO_PIPELINE_STEPS),
                        "total_steps": len(AUDIO_PIPELINE_STEPS),
                        "status": "error",
                    }))
                except Exception:
                    pass
            logger.error(f"Audio pipeline failed: {result.error}")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Audio pipeline crashed: {e}")
        emit_progress("error", 0, f"Audio pipeline crashed: {e}")
        if args.progress_file:
            try:
                Path(args.progress_file).write_text(json.dumps({
                    "step": "error",
                    "step_name": "Error",
                    "step_progress": 0,
                    "overall_progress": 0,
                    "message": f"Audio pipeline crashed: {e}",
                    "status": "error",
                    "step_order": 0,
                    "total_steps": len(AUDIO_PIPELINE_STEPS),
                }))
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
