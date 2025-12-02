"""Episode Audio Pipeline Orchestrator.

Main entry point for running the complete audio pipeline:
1. Extract original audio
2. Separation (MDX-Extra)
3. Enhance (Resemble)
4. Diarization
5. Voice embeddings + clustering + voice-bank mapping
6. ASR (OpenAI Whisper or Gemini)
7. Fuse diarization + ASR + voice-bank mapping into transcript
8. QC
9. Final export
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

import yaml

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    # Try to find .env in common locations
    for env_path in [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[3] / ".env",  # project root
    ]:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass  # dotenv not installed, rely on system environment

from .models import (
    AudioArtifacts,
    AudioPipelineConfig,
    AudioPipelineResult,
    ManifestArtifacts,
    QCStatus,
)

LOGGER = logging.getLogger(__name__)


# Pipeline step metadata for progress tracking
AUDIO_PIPELINE_STEPS = {
    "extract": {"name": "Extract Audio", "order": 1, "weight": 5},
    "separate": {"name": "Separate Vocals", "order": 2, "weight": 20},
    "enhance": {"name": "Enhance Audio", "order": 3, "weight": 15},
    "diarize": {"name": "Speaker Diarization", "order": 4, "weight": 20},
    "voices": {"name": "Voice Clustering", "order": 5, "weight": 10},
    "transcribe": {"name": "Transcription", "order": 6, "weight": 15},
    "fuse": {"name": "Fuse Transcript", "order": 7, "weight": 5},
    "export": {"name": "Export", "order": 8, "weight": 5},
    "qc": {"name": "Quality Control", "order": 9, "weight": 5},
}


def _load_config(config_path: Optional[Path] = None) -> AudioPipelineConfig:
    """Load audio pipeline configuration from YAML."""
    if config_path is None:
        # Find config relative to project root
        # Try common locations
        candidates = [
            Path("config/pipeline/audio.yaml"),
            Path(__file__).parents[3] / "config" / "pipeline" / "audio.yaml",
        ]

        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None or not config_path.exists():
        LOGGER.warning("Audio config not found, using defaults")
        return AudioPipelineConfig()

    with config_path.open("r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    return AudioPipelineConfig.from_yaml(yaml_data)


def _get_audio_paths(ep_id: str, data_root: Optional[Path] = None) -> dict:
    """Get paths for audio artifacts."""
    if data_root is None:
        data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))

    audio_dir = data_root / "audio" / ep_id
    manifests_dir = data_root / "manifests" / ep_id
    video_dir = data_root / "videos" / ep_id

    return {
        "video": video_dir / "episode.mp4",
        "audio_dir": audio_dir,
        "manifests_dir": manifests_dir,
        "original": audio_dir / "episode_original.wav",
        "vocals": audio_dir / "episode_vocals.wav",
        "vocals_enhanced": audio_dir / "episode_vocals_enhanced.wav",
        "final_voice_only": audio_dir / "episode_final_voice_only.wav",
        "diarization": manifests_dir / "audio_diarization.jsonl",
        "diarization_pyannote": manifests_dir / "audio_diarization_pyannote.jsonl",
        "diarization_gpt4o": manifests_dir / "audio_diarization_gpt4o.jsonl",
        "diarization_combined": manifests_dir / "audio_diarization_combined.jsonl",
        "diarization_comparison": manifests_dir / "audio_diarization_comparison.json",
        "asr_raw": manifests_dir / "audio_asr_raw.jsonl",
        "voice_clusters": manifests_dir / "audio_voice_clusters.json",
        "voice_clusters_gpt4o": manifests_dir / "audio_voice_clusters_gpt4o.json",
        "voice_mapping": manifests_dir / "audio_voice_mapping.json",
        "transcript_jsonl": manifests_dir / "episode_transcript.jsonl",
        "transcript_vtt": manifests_dir / "episode_transcript.vtt",
        "qc": manifests_dir / "audio_qc.json",
        "speaker_groups": manifests_dir / "audio_speaker_groups.json",
    }


def _get_show_id(ep_id: str) -> str:
    """Extract show ID from episode ID."""
    # ep_id format: show-sXXeYY (e.g., rhoslc-s06e02)
    parts = ep_id.split("-")
    if parts:
        return parts[0].upper()
    return ep_id.upper()


def _save_diarization_comparison(
    pyannote_segments: list,
    gpt4o_segments: list,
    output_path: Path,
) -> dict:
    """Compare pyannote and GPT-4o diarization outputs and save report.

    Args:
        pyannote_segments: Diarization segments from pyannote
        gpt4o_segments: Diarization segments from GPT-4o
        output_path: Path to save comparison JSON

    Returns:
        Comparison report dict
    """
    import json

    # Get pyannote stats
    pyannote_speakers = set(s.speaker for s in pyannote_segments)
    pyannote_duration = sum(s.end - s.start for s in pyannote_segments)

    # Get GPT-4o stats
    gpt4o_speakers = set(s.speaker for s in gpt4o_segments if s.speaker)
    gpt4o_duration = sum(s.end - s.start for s in gpt4o_segments) if gpt4o_segments else 0

    # Build comparison report
    comparison = {
        "pyannote": {
            "segment_count": len(pyannote_segments),
            "speaker_count": len(pyannote_speakers),
            "speakers": sorted(pyannote_speakers),
            "total_speech_duration_s": round(pyannote_duration, 2),
            "avg_segment_duration_s": round(pyannote_duration / len(pyannote_segments), 2) if pyannote_segments else 0,
        },
        "gpt4o": {
            "segment_count": len(gpt4o_segments),
            "speaker_count": len(gpt4o_speakers),
            "speakers": sorted(gpt4o_speakers),
            "total_speech_duration_s": round(gpt4o_duration, 2),
            "avg_segment_duration_s": round(gpt4o_duration / len(gpt4o_segments), 2) if gpt4o_segments else 0,
            "has_transcription": True if gpt4o_segments else False,
        },
        "comparison": {
            "speaker_count_diff": len(gpt4o_speakers) - len(pyannote_speakers),
            "segment_count_diff": len(gpt4o_segments) - len(pyannote_segments),
            "duration_diff_s": round(gpt4o_duration - pyannote_duration, 2),
        },
        # Sample segments for manual review
        "samples": {
            "pyannote_first_5": [
                {"start": s.start, "end": s.end, "speaker": s.speaker}
                for s in pyannote_segments[:5]
            ],
            "gpt4o_first_5": [
                {"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text[:100] if hasattr(s, 'text') else ""}
                for s in gpt4o_segments[:5]
            ],
        },
        # Full segment lists for downstream UI/flagging
        "segments": {
            "pyannote": [
                {
                    "start": s.start,
                    "end": s.end,
                    "speaker": s.speaker,
                }
                for s in pyannote_segments
            ],
            "gpt4o": [
                {
                    "segment_id": f"gpt4o_{i+1:04d}",
                    "start": s.start,
                    "end": s.end,
                    "speaker": getattr(s, "speaker", None),
                    "raw_text": getattr(s, "text", None)[:400] if getattr(s, "text", None) else None,
                }
                for i, s in enumerate(gpt4o_segments)
            ],
        },
    }

    # Save comparison
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    LOGGER.info(f"Diarization comparison saved: {output_path}")
    LOGGER.info(f"  Pyannote: {len(pyannote_speakers)} speakers, {len(pyannote_segments)} segments")
    LOGGER.info(f"  GPT-4o: {len(gpt4o_speakers)} speakers, {len(gpt4o_segments)} segments")

    return comparison


def run_episode_audio_pipeline(
    ep_id: str,
    overwrite: bool = False,
    asr_provider: Optional[str] = None,
    config_path: Optional[Path] = None,
    data_root: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, float, str], None]] = None,
) -> AudioPipelineResult:
    """Run the complete audio pipeline for an episode.

    Args:
        ep_id: Episode identifier
        overwrite: Whether to overwrite existing artifacts
        asr_provider: Override ASR provider (openai_whisper or gemini_3)
        config_path: Optional path to config file
        data_root: Optional data root directory
        progress_callback: Optional callback for progress updates
            Signature: callback(step: str, progress: float, message: str)

    Returns:
        AudioPipelineResult with all paths and metrics
    """
    LOGGER.info(f"Starting audio pipeline for episode: {ep_id}")

    # Load configuration
    config = _load_config(config_path)

    # Override ASR provider if specified
    if asr_provider:
        # Normalize aliases for backwards compatibility (CLI uses "gemini")
        provider_override = "gemini_3" if asr_provider == "gemini" else asr_provider
        config.asr.provider = provider_override

    # Get paths
    paths = _get_audio_paths(ep_id, data_root)
    show_id = _get_show_id(ep_id)

    # Ensure directories exist
    paths["audio_dir"].mkdir(parents=True, exist_ok=True)
    paths["manifests_dir"].mkdir(parents=True, exist_ok=True)

    # Initialize result
    result = AudioPipelineResult(
        ep_id=ep_id,
        audio_artifacts=AudioArtifacts(),
        manifest_artifacts=ManifestArtifacts(),
    )

    def _update_progress(step: str, progress: float, message: str):
        LOGGER.info(f"[{step}] {progress*100:.0f}% - {message}")
        if progress_callback:
            progress_callback(step, progress, message)

    try:
        # Step 1: Extract original audio
        _update_progress("extract", 0.0, "Extracting audio from video...")

        from .io import extract_audio_from_video

        if not paths["video"].exists():
            raise FileNotFoundError(f"Video not found: {paths['video']}")

        original_path, original_stats = extract_audio_from_video(
            paths["video"],
            paths["original"],
            sample_rate=config.export.sample_rate,
            bit_depth=config.export.bit_depth,
            overwrite=overwrite,
        )
        result.audio_artifacts.original = original_path
        result.duration_original_s = original_stats.duration_seconds

        _update_progress("extract", 1.0, f"Audio extracted: {original_stats.duration_seconds:.1f}s")

        # Step 2: Separation (MDX-Extra)
        _update_progress("separate", 0.0, "Separating vocals from accompaniment...")

        from .separation_mdx import separate_vocals

        vocals_path, _ = separate_vocals(
            original_path,
            paths["audio_dir"],
            config.separation,
            overwrite=overwrite,
        )
        result.audio_artifacts.vocals = vocals_path

        _update_progress("separate", 1.0, "Vocal separation complete")

        # Step 3: Enhance (Resemble)
        _update_progress("enhance", 0.0, "Enhancing vocals...")

        try:
            from .enhance_resemble import enhance_audio_resemble, check_api_available

            if check_api_available():
                enhanced_path = enhance_audio_resemble(
                    vocals_path,
                    paths["vocals_enhanced"],
                    config.enhance,
                    overwrite=overwrite,
                )
            else:
                LOGGER.warning("Resemble API not available, using local enhancement")
                from .enhance_resemble import enhance_audio_local
                enhanced_path = enhance_audio_local(
                    vocals_path,
                    paths["vocals_enhanced"],
                    overwrite=overwrite,
                )
        except Exception as e:
            LOGGER.warning(f"Enhancement failed, using vocals directly: {e}")
            enhanced_path = vocals_path

        result.audio_artifacts.vocals_enhanced = enhanced_path

        _update_progress("enhance", 1.0, "Enhancement complete")

        # Step 4: Diarization (dual mode: pyannote + GPT-4o for comparison)
        # Use vocals (separated, no music) for diarization - better for speaker detection
        # Enhancement can alter voice characteristics, so we use the cleaner separated vocals
        _update_progress("diarize", 0.0, "Running dual diarization (pyannote + GPT-4o)...")

        # 4a: Run pyannote diarization
        from .diarization_pyannote import run_diarization

        _update_progress("diarize", 0.1, "Running pyannote diarization...")
        pyannote_segments = run_diarization(
            vocals_path,  # Use separated vocals (no music) for better speaker detection
            paths["diarization_pyannote"],
            config.diarization,
            overwrite=overwrite,
        )
        LOGGER.info(f"Pyannote diarization: {len(pyannote_segments)} segments, "
                   f"{len(set(s.speaker for s in pyannote_segments))} speakers")

        # 4b: Run GPT-4o diarization (unified transcription + diarization)
        _update_progress("diarize", 0.5, "Running GPT-4o diarization...")
        gpt4o_diar_segments = []
        try:
            from .asr_openai import transcribe_with_diarization, check_api_available

            if check_api_available():
                gpt4o_segments = transcribe_with_diarization(
                    vocals_path,
                    paths["diarization_gpt4o"],
                    overwrite=overwrite,
                )
                gpt4o_speakers = len(set(s.speaker for s in gpt4o_segments if s.speaker))
                LOGGER.info(f"GPT-4o diarization: {len(gpt4o_segments)} segments, {gpt4o_speakers} speakers")
                # Convert GPT-4o diarization segments to DiarizationSegment for downstream use
                from .models import DiarizationSegment
                for seg in gpt4o_segments:
                    if seg.start is None or seg.end is None:
                        continue
                    if seg.end <= seg.start:
                        continue
                    speaker_label = seg.speaker or "GPT4O_SPK"
                    gpt4o_diar_segments.append(DiarizationSegment(
                        start=float(seg.start),
                        end=float(seg.end),
                        speaker=speaker_label,
                        confidence=seg.confidence,
                    ))
            else:
                LOGGER.warning("OpenAI API not available, skipping GPT-4o diarization")
                gpt4o_segments = []
        except Exception as e:
            LOGGER.warning(f"GPT-4o diarization failed: {e}")
            gpt4o_segments = []

        # 4c: Create comparison report
        _update_progress("diarize", 0.8, "Generating diarization comparison...")
        _save_diarization_comparison(
            pyannote_segments,
            gpt4o_segments,
            paths["diarization_comparison"],
        )

        # Use pyannote as primary manifest, but create a combined manifest for clustering (pyannote + GPT-4o)
        import shutil
        if paths["diarization_pyannote"].exists():
            shutil.copy(paths["diarization_pyannote"], paths["diarization"])
        result.manifest_artifacts.diarization_pyannote = paths["diarization_pyannote"]
        clustering_segments = pyannote_segments
        if gpt4o_diar_segments:
            try:
                from .diarization_pyannote import _save_diarization_manifest
                clustering_segments = pyannote_segments + gpt4o_diar_segments
                _save_diarization_manifest(clustering_segments, paths["diarization_combined"])
                result.manifest_artifacts.diarization_gpt4o = paths["diarization_gpt4o"]
            except Exception as err:
                LOGGER.warning(f"Failed to save combined diarization manifest: {err}")
                clustering_segments = pyannote_segments
        diarization_segments = pyannote_segments
        result.manifest_artifacts.diarization = paths["diarization"]

        # 4d: Build speaker groups manifest (primary surface for UI)
        from .speaker_groups import build_speaker_groups_manifest

        speaker_group_sources = {"pyannote": pyannote_segments}
        if gpt4o_diar_segments:
            speaker_group_sources["gpt4o"] = gpt4o_diar_segments

        speaker_groups_manifest = build_speaker_groups_manifest(
            ep_id,
            speaker_group_sources,
            paths["speaker_groups"],
            overwrite=overwrite,
        )
        result.manifest_artifacts.speaker_groups = paths["speaker_groups"]

        _update_progress("diarize", 1.0, f"Dual diarization complete: pyannote={len(pyannote_segments)}, gpt4o={len(gpt4o_segments)}")

        # Step 5: Voice clustering + voice bank mapping
        _update_progress("voices", 0.0, "Clustering voices...")

        from .voice_clusters import cluster_episode_voices
        from .voice_bank import match_voice_clusters_to_bank

        voice_clusters = cluster_episode_voices(
            vocals_path,  # Use separated vocals for voice embedding extraction
            clustering_segments,
            paths["voice_clusters"],
            config.voice_clustering,
            overwrite=overwrite,
            speaker_groups_manifest=speaker_groups_manifest,
        )
        result.manifest_artifacts.voice_clusters = paths["voice_clusters"]

        # Also produce GPT-4o-only clusters (one per diarization label) when available for diagnostics
        if gpt4o_diar_segments:
            try:
                from .voice_clusters import _clusters_from_diarization_labels, _save_voice_clusters

                gpt4o_clusters = _clusters_from_diarization_labels(gpt4o_diar_segments)
                _save_voice_clusters(gpt4o_clusters, paths["voice_clusters_gpt4o"])
                LOGGER.info(f"Saved GPT-4o-only clusters: {len(gpt4o_clusters)}")
            except Exception as err:
                LOGGER.warning(f"Failed to build GPT-4o-only clusters: {err}")

        _update_progress("voices", 0.5, f"Found {len(voice_clusters)} voice clusters")

        voice_mapping = match_voice_clusters_to_bank(
            show_id,
            voice_clusters,
            paths["voice_mapping"],
            config.voice_bank,
            config.voice_clustering.similarity_threshold,
            overwrite=overwrite,
        )
        result.manifest_artifacts.voice_mapping = paths["voice_mapping"]

        result.voice_cluster_count = len(voice_clusters)
        result.labeled_voices = sum(1 for m in voice_mapping if m.similarity is not None)
        result.unlabeled_voices = len(voice_mapping) - result.labeled_voices

        _update_progress("voices", 1.0, f"Voice mapping complete: {result.labeled_voices} labeled, {result.unlabeled_voices} unlabeled")

        # Step 6: ASR
        _update_progress("transcribe", 0.0, f"Transcribing with {config.asr.provider}...")

        # Normalize provider before selecting implementation
        if config.asr.provider == "gemini":
            config.asr.provider = "gemini_3"

        if config.asr.provider == "gemini_3":
            from .asr_gemini import transcribe_audio
        else:
            from .asr_openai import transcribe_audio

        asr_segments = transcribe_audio(
            enhanced_path,
            paths["asr_raw"],
            config.asr,
            overwrite=overwrite,
        )
        result.manifest_artifacts.asr_raw = paths["asr_raw"]

        _update_progress("transcribe", 1.0, f"Transcription complete: {len(asr_segments)} segments")

        # Step 7: Fuse diarization + ASR + voice mapping
        _update_progress("fuse", 0.0, "Generating final transcript...")

        from .fuse_diarization_asr import fuse_transcript

        transcript_rows = fuse_transcript(
            diarization_segments,
            asr_segments,
            voice_clusters,
            voice_mapping,
            speaker_groups_manifest,
            paths["transcript_jsonl"],
            paths["transcript_vtt"],
            config.export.vtt_include_speaker_notes,
            overwrite=overwrite,
            diarization_source="pyannote",
        )
        result.manifest_artifacts.transcript_jsonl = paths["transcript_jsonl"]
        result.manifest_artifacts.transcript_vtt = paths["transcript_vtt"]
        result.transcript_row_count = len(transcript_rows)

        # Enrich diarization comparison with canonical text + mixed-speaker flags
        try:
            from .diarization_comparison import augment_diarization_comparison

            augment_diarization_comparison(
                paths["diarization_comparison"],
                paths["transcript_jsonl"],
            )
        except Exception as exc:
            LOGGER.warning(f"Failed to augment diarization comparison: {exc}")

        _update_progress("fuse", 1.0, f"Transcript generated: {len(transcript_rows)} rows")

        # Step 8: Final export
        _update_progress("export", 0.0, "Exporting final audio...")

        from .export import export_final_audio

        final_path = export_final_audio(
            enhanced_path,
            paths["final_voice_only"],
            config.export,
            overwrite=overwrite,
        )
        result.audio_artifacts.final_voice_only = final_path

        from .io import get_audio_duration
        result.duration_final_s = get_audio_duration(final_path)

        _update_progress("export", 1.0, "Final audio exported")

        # Step 9: QC
        _update_progress("qc", 0.0, "Running quality checks...")

        from .io import compute_snr
        from .qc import run_qc_checks

        snr_db = None
        try:
            snr_db = compute_snr(enhanced_path)
        except Exception as e:
            LOGGER.warning(f"SNR calculation failed: {e}")

        qc_report = run_qc_checks(
            ep_id,
            config.qc,
            result.duration_original_s,
            result.duration_final_s,
            snr_db,
            diarization_segments,
            asr_segments,
            voice_clusters,
            voice_mapping,
            transcript_rows,
            paths["qc"],
            overwrite=overwrite,
        )
        result.manifest_artifacts.qc = paths["qc"]
        result.qc_status = qc_report.status

        result.metrics = {
            "snr_db": snr_db,
            "diarization_segments": len(diarization_segments),
            "asr_segments": len(asr_segments),
            "voice_clusters": len(voice_clusters),
            "labeled_voices": result.labeled_voices,
            "unlabeled_voices": result.unlabeled_voices,
            "transcript_rows": len(transcript_rows),
            "qc_warnings": len(qc_report.warnings),
            "qc_errors": len(qc_report.errors),
        }

        result.status = "succeeded"
        _update_progress("qc", 1.0, f"Pipeline complete: QC status = {qc_report.status.value}")

        LOGGER.info(f"Audio pipeline complete for {ep_id}: {result.status}")

    except Exception as e:
        LOGGER.exception(f"Audio pipeline failed for {ep_id}: {e}")
        result.status = "failed"
        result.qc_status = QCStatus.FAILED
        result.error = str(e)

    return result


def check_pipeline_prerequisites() -> dict:
    """Check if all required dependencies and API keys are available.

    Returns:
        Dict with status of each dependency
    """
    status = {
        "ffmpeg": False,
        "soundfile": False,
        "demucs": False,
        "pyannote": False,
        "openai": False,
        "gemini": False,
        "resemble": False,
    }

    # Check ffmpeg
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        status["ffmpeg"] = True
    except Exception:
        pass

    # Check Python packages
    try:
        import soundfile
        status["soundfile"] = True
    except ImportError:
        pass

    try:
        import demucs
        status["demucs"] = True
    except ImportError:
        pass

    try:
        import pyannote.audio
        status["pyannote"] = True
    except ImportError:
        pass

    # Check API keys
    if os.environ.get("OPENAI_API_KEY"):
        status["openai"] = True

    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        status["gemini"] = True

    if os.environ.get("RESEMBLE_API_KEY"):
        status["resemble"] = True

    return status


def sync_audio_artifacts_to_s3(
    ep_id: str,
    result: AudioPipelineResult,
    data_root: Optional[Path] = None,
) -> dict:
    """Sync audio pipeline artifacts to S3.

    Args:
        ep_id: Episode identifier
        result: Audio pipeline result with artifact paths
        data_root: Optional data root directory

    Returns:
        Dict with upload status for each artifact type
    """
    try:
        from apps.api.services.storage import (
            STORAGE,
            episode_context_from_id,
        )
    except ImportError:
        LOGGER.warning("Storage service not available, skipping S3 sync")
        return {"status": "unavailable"}

    if not STORAGE.write_enabled:
        LOGGER.debug("S3 write not enabled, skipping sync")
        return {"status": "disabled"}

    try:
        ep_ctx = episode_context_from_id(ep_id)
    except ValueError as e:
        LOGGER.warning(f"Failed to parse episode ID for S3 sync: {e}")
        return {"status": "error", "error": str(e)}

    show = ep_ctx.show_slug
    season = ep_ctx.season_number
    episode = ep_ctx.episode_number
    base_name = f"{show}_s{season:02d}e{episode:02d}"

    upload_status = {
        "audio": {},
        "transcripts": {},
        "qc": {},
        "diagnostics": {},
    }

    # Upload audio files
    audio_files = [
        ("original", result.audio_artifacts.original, f"{base_name}_original.wav"),
        ("vocals", result.audio_artifacts.vocals, f"{base_name}_vocals.wav"),
        ("vocals_enhanced", result.audio_artifacts.vocals_enhanced, f"{base_name}_vocals_enhanced.wav"),
        ("final_voice_only", result.audio_artifacts.final_voice_only, f"{base_name}_final_voice_only.wav"),
    ]

    for key, path, s3_name in audio_files:
        if path and path.exists():
            ok = STORAGE.put_artifact(ep_ctx, "audio", path, s3_name)
            upload_status["audio"][key] = ok

    # Upload transcripts
    transcript_files = [
        ("jsonl", result.manifest_artifacts.transcript_jsonl, f"{base_name}_transcript.jsonl"),
        ("vtt", result.manifest_artifacts.transcript_vtt, f"{base_name}_transcript.vtt"),
    ]

    for key, path, s3_name in transcript_files:
        if path and path.exists():
            ok = STORAGE.put_artifact(ep_ctx, "audio_transcripts", path, s3_name)
            upload_status["transcripts"][key] = ok

    # Upload QC and voice manifests
    qc_files = [
        ("qc", result.manifest_artifacts.qc, f"{base_name}_audio_qc.json"),
        ("voice_clusters", result.manifest_artifacts.voice_clusters, f"{base_name}_audio_voice_clusters.json"),
        ("voice_mapping", result.manifest_artifacts.voice_mapping, f"{base_name}_audio_voice_mapping.json"),
        ("speaker_groups", result.manifest_artifacts.speaker_groups, f"{base_name}_audio_speaker_groups.json"),
    ]

    for key, path, s3_name in qc_files:
        if path and path.exists():
            ok = STORAGE.put_artifact(ep_ctx, "audio_qc", path, s3_name)
            upload_status["qc"][key] = ok

    # Upload diagnostic GPT-4o-only clusters if present
    diag_path = paths.get("voice_clusters_gpt4o")
    if diag_path and diag_path.exists():
        ok = STORAGE.put_artifact(ep_ctx, "audio_qc", diag_path, f"{base_name}_audio_voice_clusters_gpt4o.json")
        upload_status["diagnostics"]["voice_clusters_gpt4o"] = ok

    LOGGER.info(f"S3 sync complete for {ep_id}: {upload_status}")
    upload_status["status"] = "ok"

    return upload_status
