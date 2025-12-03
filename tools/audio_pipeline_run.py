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
        "diarization_comparison": manifests_dir / "audio_diarization_comparison.json",
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

        # Clear old diarization data for a fresh start
        files_to_clear = [
            paths["diarization"],            # audio_diarization.jsonl
            paths["diarization_comparison"], # audio_diarization_comparison.json
            paths["voice_clusters"],         # audio_voice_clusters.json (depends on diarization)
            paths["voice_mapping"],          # audio_voice_mapping.json (depends on diarization)
        ]
        for old_file in files_to_clear:
            if old_file.exists():
                logger.info(f"Clearing old file: {old_file.name}")
                old_file.unlink()
        emit_progress("diarize", 0.05, "Cleared old diarization data",
                     step_name="Diarization", step_order=1, total_steps=1)

        from py_screenalytics.audio.diarization_pyannote import run_diarization
        from py_screenalytics.audio.episode_audio_pipeline import _load_config, _get_cast_count_for_episode

        config = _load_config().diarization

        # Apply speaker count overrides
        config_updates = {}
        if args.num_speakers is not None:
            config_updates["num_speakers"] = args.num_speakers
            logger.info(f"Forcing {args.num_speakers} speakers")
            emit_progress("diarize", 0.1, f"Running diarization (forcing {args.num_speakers} speakers)...",
                         step_name="Diarization", step_order=1, total_steps=1)
        elif args.min_speakers is not None or args.max_speakers is not None:
            if args.min_speakers is not None:
                config_updates["min_speakers"] = args.min_speakers
            if args.max_speakers is not None:
                config_updates["max_speakers"] = args.max_speakers
            logger.info(f"Speaker range: {args.min_speakers or config.min_speakers}-{args.max_speakers or config.max_speakers}")
            emit_progress("diarize", 0.1, f"Running diarization (speakers: {args.min_speakers or config.min_speakers}-{args.max_speakers or config.max_speakers})...",
                         step_name="Diarization", step_order=1, total_steps=1)
        else:
            # Auto-calculate speaker range from cast count
            cast_count = _get_cast_count_for_episode(ep_id)
            if cast_count > 0:
                auto_min = max(1, cast_count - 2)
                auto_max = cast_count + 5
                config_updates["min_speakers"] = auto_min
                config_updates["max_speakers"] = auto_max
                logger.info(f"Auto-calculated speaker range from {cast_count} cast members: {auto_min}-{auto_max}")
                emit_progress("diarize", 0.1, f"Running diarization (auto: {auto_min}-{auto_max} speakers from {cast_count} cast)...",
                             step_name="Diarization", step_order=1, total_steps=1)
            else:
                emit_progress("diarize", 0.1, "Running diarization (using config defaults)...",
                             step_name="Diarization", step_order=1, total_steps=1)

        if config_updates:
            config = config.model_copy(update=config_updates)

        # Log final config for debugging
        logger.info(f"Diarization config: backend={config.backend}, min_speakers={config.min_speakers}, max_speakers={config.max_speakers}, num_speakers={config.num_speakers}")
        emit_progress("diarize", 0.15, f"Config: backend={config.backend}, speakers={config.min_speakers}-{config.max_speakers}",
                     step_name="Diarization", step_order=1, total_steps=1)

        segments = run_diarization(audio_path, paths["diarization"], config, overwrite=True)
        speakers = set(s.speaker for s in segments)

        emit_progress("diarize", 0.7, f"Pyannote diarization complete: {len(speakers)} speakers",
                     step_name="Diarization", step_order=1, total_steps=1)

        # Regenerate diarization comparison using existing GPT-4o data
        emit_progress("diarize", 0.8, "Regenerating diarization comparison...",
                     step_name="Diarization", step_order=1, total_steps=1)
        try:
            from py_screenalytics.audio.episode_audio_pipeline import _save_diarization_comparison, _get_audio_paths as get_full_paths
            from py_screenalytics.audio.diarization_comparison import augment_diarization_comparison

            full_paths = get_full_paths(ep_id)

            # Load existing GPT-4o diarization if available
            gpt4o_segments = []
            gpt4o_path = full_paths.get("diarization_gpt4o")
            if gpt4o_path and gpt4o_path.exists():
                import json
                with gpt4o_path.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            gpt4o_segments.append(json.loads(line))
                logger.info(f"Loaded {len(gpt4o_segments)} GPT-4o segments for comparison")

            # Convert pyannote segments to dict format for comparison
            pyannote_dicts = [{"start": s.start, "end": s.end, "speaker": s.speaker} for s in segments]

            # Save comparison
            comparison_path = full_paths.get("diarization_comparison")
            if comparison_path:
                _save_diarization_comparison(pyannote_dicts, gpt4o_segments, comparison_path)
                logger.info(f"Saved diarization comparison to {comparison_path}")

                # Augment with transcript text
                transcript_path = full_paths.get("transcript_jsonl")
                if transcript_path and transcript_path.exists():
                    augment_diarization_comparison(comparison_path, transcript_path)
                    logger.info("Augmented comparison with transcript text")

        except Exception as e:
            logger.warning(f"Could not regenerate comparison: {e}")

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

        # Load diarization segments (prefer combined pyannote+GPT-4o if present)
        diar_path = paths.get("diarization_combined", paths["diarization"])
        segments = _load_diarization_manifest(diar_path if diar_path.exists() else paths["diarization"])

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


def run_voiceprint_refresh(args):
    """Run voiceprint identification refresh pipeline.

    This function:
    1. Selects clean segments from manually assigned clusters
    2. Creates voiceprints for cast members using Pyannote API
    3. Runs identification pass on full episode audio
    4. Regenerates transcript with cast names
    5. Generates review queue for low-confidence segments
    """
    setup_logging()
    logger = logging.getLogger("audio_pipeline_run")

    ep_id = args.ep_id
    logger.info(f"Starting voiceprint refresh for {ep_id}")
    emit_progress("voiceprint_refresh", 0, f"Starting voiceprint identification refresh for {ep_id}",
                 step_name="Voiceprint Refresh", step_order=1, total_steps=5)

    try:
        paths = _get_audio_paths(ep_id)

        if not paths["diarization"].exists():
            emit_progress("error", 0, "No diarization found. Run audio pipeline first.")
            sys.exit(1)

        # Find input audio
        audio_path = paths["audio_vocals_enhanced"]
        if not audio_path.exists():
            audio_path = paths["audio_vocals"]
        if not audio_path.exists():
            emit_progress("error", 0, "No audio files found. Run full pipeline first.")
            sys.exit(1)

        # Get show_id from ep_id
        show_id = ep_id.rsplit("-", 1)[0] if "-" in ep_id else ep_id

        # Build voiceprint overwrite policy
        policy = "always" if args.overwrite_voiceprints else "if_missing"

        emit_progress("voiceprint_refresh", 0.1, "Loading configuration...",
                     step_name="Voiceprint Refresh", step_order=1, total_steps=5)

        from py_screenalytics.audio.models import VoiceprintIdentificationConfig

        config = VoiceprintIdentificationConfig(
            voiceprint_overwrite_policy=policy,
            ident_matching_threshold=args.ident_threshold,
        )

        # Use async runner
        import asyncio
        from apps.api.jobs_audio import episode_voiceprint_refresh_async

        async def run_with_progress():
            def progress_cb(step: str, progress: float, message: str = ""):
                # Map steps to progress stages
                step_map = {
                    "select_segments": (1, "Selecting Segments"),
                    "create_voiceprints": (2, "Creating Voiceprints"),
                    "run_identification": (3, "Running Identification"),
                    "regenerate_transcript": (4, "Regenerating Transcript"),
                    "generate_review_queue": (5, "Generating Review Queue"),
                }
                step_order, step_name = step_map.get(step, (1, step))
                overall = (step_order - 1 + progress) / 5.0
                emit_progress("voiceprint_refresh", overall, message,
                            step_name=step_name, step_order=step_order, total_steps=5,
                            step_progress=progress)

            result = await episode_voiceprint_refresh_async(
                ep_id=ep_id,
                show_id=show_id,
                overwrite_voiceprints=args.overwrite_voiceprints,
                ident_threshold=args.ident_threshold,
                progress_callback=progress_cb,
            )
            return result

        result = asyncio.run(run_with_progress())

        status = result.get("status")

        if status == "succeeded" or status == "success":
            summary = result.get("summary", {})
            emit_progress(
                "complete", 1.0,
                f"Voiceprint refresh complete: {summary.get('voiceprints_created', 0)} voiceprints, "
                f"{summary.get('review_queue_count', 0)} items in review queue",
                step_name="Complete", step_order=5, total_steps=5,
                voiceprints_created=summary.get("voiceprints_created", 0),
                voiceprints_skipped=summary.get("voiceprints_skipped", 0),
                review_queue_count=summary.get("review_queue_count", 0),
            )
            logger.info(f"Voiceprint refresh complete for {ep_id}")
            sys.exit(0)
        elif status == "skipped":
            # No manual assignments - not an error, but nothing to do
            reason = result.get("reason", "unknown")
            message = result.get("message", f"Voiceprint refresh skipped: {reason}")
            emit_progress(
                "skipped", 1.0,
                message,
                step_name="Skipped", step_order=1, total_steps=1,
                reason=reason,
            )
            logger.info(f"Voiceprint refresh skipped for {ep_id}: {reason}")
            sys.exit(0)
        elif status == "queued":
            # Task was queued via Celery
            job_id = result.get("job_id", "unknown")
            emit_progress(
                "queued", 0.0,
                f"Voiceprint refresh queued (job_id: {job_id})",
                step_name="Queued", step_order=1, total_steps=1,
                job_id=job_id,
            )
            logger.info(f"Voiceprint refresh queued for {ep_id}: job_id={job_id}")
            sys.exit(0)
        else:
            emit_progress("error", 0, f"Voiceprint refresh failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Voiceprint refresh failed: {e}")
        emit_progress("error", 0, f"Voiceprint refresh failed: {e}")
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
    parser.add_argument("--min-speakers", type=int, default=None,
                       help="Minimum expected speakers (hint to diarization model)")
    parser.add_argument("--max-speakers", type=int, default=None,
                       help="Maximum expected speakers (hint to diarization model)")
    parser.add_argument("--similarity-threshold", type=float, default=0.30,
                       help="Similarity threshold for voice clustering (lower = fewer clusters)")

    # Voiceprint refresh flags
    parser.add_argument("--voiceprint-refresh", action="store_true",
                       help="Run voiceprint identification refresh (requires diarization + manual assignments)")
    parser.add_argument("--overwrite-voiceprints", action="store_true",
                       help="Force recreation of voiceprints even if they exist")
    parser.add_argument("--ident-threshold", type=int, default=60,
                       help="Confidence threshold for identification matching (0-100)")

    args = parser.parse_args()

    # Handle incremental operations
    if args.diarize_only:
        return run_diarize_only(args)
    if args.transcribe_only:
        return run_transcribe_only(args)
    if args.voices_only:
        return run_voices_only(args)
    if args.voiceprint_refresh:
        return run_voiceprint_refresh(args)

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
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
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
