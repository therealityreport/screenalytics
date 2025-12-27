"""Audio pipeline API endpoints.

Provides HTTP endpoints for:
- Starting audio pipeline jobs
- Querying job status
- Downloading transcripts
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_screenalytics.artifacts import get_path

router = APIRouter(prefix="/jobs", tags=["audio"])
edit_router = APIRouter(tags=["audio"])
LOGGER = logging.getLogger(__name__)

try:
    from celery.result import AsyncResult  # type: ignore
except Exception:  # pragma: no cover - optional celery dependency for tests
    AsyncResult = None  # type: ignore

try:
    from apps.api.jobs_audio import episode_audio_pipeline_async as episode_audio_pipeline_task
except Exception:  # pragma: no cover - optional audio pipeline dependency for tests
    episode_audio_pipeline_task = None  # type: ignore


# =============================================================================
# Request/Response Models
# =============================================================================


class AudioPipelineRequest(BaseModel):
    """Request to start audio pipeline."""
    ep_id: str = Field(..., description="Episode identifier")
    run_mode: Literal["queue", "local", "redis"] = Field(
        "queue",
        description="Run mode: 'queue' for Celery, 'local' for synchronous",
    )
    overwrite: bool = Field(False, description="Overwrite existing artifacts")
    asr_provider: Optional[Literal["openai_whisper", "gemini_3", "gemini"]] = Field(
        None,
        description="Override ASR provider (gemini_3 and gemini are equivalent)",
    )
    min_speakers: Optional[int] = Field(
        None,
        description="Minimum expected speakers (hint for diarization)",
    )
    max_speakers: Optional[int] = Field(
        None,
        description="Maximum expected speakers (hint for diarization)",
    )


class AudioFilesRequest(BaseModel):
    """Request for Phase 1: Create audio files."""
    ep_id: str = Field(..., description="Episode identifier")
    overwrite: bool = Field(False, description="Overwrite existing artifacts")


class DiarizeTranscribeRequest(BaseModel):
    """Request for Phase 2: Diarization + Transcription."""
    ep_id: str = Field(..., description="Episode identifier")
    asr_provider: Optional[Literal["openai_whisper", "gemini_3"]] = Field(
        None,
        description="Override ASR provider",
    )
    overwrite: bool = Field(False, description="Overwrite existing artifacts")


class FinalizeTranscriptRequest(BaseModel):
    """Request for Phase 4: Finalize transcript."""
    ep_id: str = Field(..., description="Episode identifier")
    overwrite: bool = Field(False, description="Overwrite existing artifacts")


class AudioPipelineResponse(BaseModel):
    """Response from starting audio pipeline."""
    job_id: Optional[str] = None
    job_type: Optional[str] = None
    status: str
    ep_id: str
    run_mode: Optional[str] = None
    error: Optional[str] = None


class AudioStatusResponse(BaseModel):
    """Response from audio status query."""
    ep_id: str
    status: str
    qc_status: Optional[str] = None
    summary: Optional[dict] = None
    artifacts: Optional[dict] = None
    error: Optional[str] = None


class VoiceAssignRequest(BaseModel):
    """Request to assign a voice cluster to a cast member or label."""
    voice_cluster_id: str = Field(..., description="Voice cluster ID to assign")
    cast_id: Optional[str] = Field(None, description="Cast member ID to assign to")
    custom_label: Optional[str] = Field(None, description="Custom label if not assigning to cast")


class VoiceAssignResponse(BaseModel):
    """Response from voice assignment."""
    voice_cluster_id: str
    speaker_id: str
    speaker_display_name: str
    voice_bank_id: str
    success: bool = True
    error: Optional[str] = None


class SmartSplitRequest(BaseModel):
    """Request payload for smart split."""
    source: str = Field("nemo", description="Diarization source (nemo, pyannote, gpt4o)")
    speaker_group_id: str
    segment_id: Optional[str] = Field(None, description="Segment identifier to split")
    start: Optional[float] = Field(None, description="Start time (seconds) if segment_id not provided")
    end: Optional[float] = Field(None, description="End time (seconds) if segment_id not provided")
    expected_voices: int = Field(2, ge=2, le=5, description="Expected number of voices within the segment")


class SpeakerAssignmentItem(BaseModel):
    """A single speaker group assignment.

    Supports both group-level and time-range-level assignments:
    - Group-level: start and end are omitted → applies to entire speaker group
    - Time-range: start and end are provided → applies only to that time window
    """
    source: str = Field("nemo", description="Diarization source (nemo, pyannote, gpt4o)")
    speaker_group_id: str = Field(..., description="Speaker group ID (e.g., nemo:speaker_0)")
    cast_id: str = Field(..., description="Cast member ID to assign to")
    cast_display_name: Optional[str] = Field(None, description="Cast member display name")
    # Time-range fields for segment-level assignments
    start: Optional[float] = Field(None, description="Segment start time (for time-range assignment)")
    end: Optional[float] = Field(None, description="Segment end time (for time-range assignment)")


class SpeakerAssignmentsRequest(BaseModel):
    """Request to set speaker assignments.

    Can include both group-level and time-range assignments.
    Time-range assignments take precedence over group-level.
    """
    assignments: List[SpeakerAssignmentItem] = Field(
        default_factory=list,
        description="List of speaker group to cast member assignments (supports time-range)"
    )


class SpeakerAssignmentsResponse(BaseModel):
    """Response containing speaker assignments."""
    ep_id: str
    assignments: List[Dict[str, Any]] = Field(default_factory=list)
    cast_members: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Available cast members for assignment"
    )
    speaker_groups: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Available speaker groups"
    )


class VoiceprintOverrideRequest(BaseModel):
    """Request to set a voiceprint override for a segment."""
    source: str = Field(..., description="Diarization source (pyannote, gpt4o)")
    speaker_group_id: str = Field(..., description="Speaker group ID (e.g., pyannote:SPEAKER_00)")
    segment_id: str = Field(..., description="Segment ID (e.g., py_0007)")
    voiceprint_override: Optional[str] = Field(
        None,
        description="Override value: 'force_include', 'force_exclude', or null to clear"
    )


class VoiceprintOverrideResponse(BaseModel):
    """Response after setting a voiceprint override."""
    status: str = Field(..., description="Status (ok, error)")
    segment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Segment state after override"
    )
    message: Optional[str] = Field(None, description="Optional message")


class SplitAtUtteranceRequest(BaseModel):
    """Request to split a speaker group segment at an utterance boundary."""
    source: str = Field(..., description="Diarization source (pyannote, gpt4o)")
    speaker_group_id: str = Field(..., description="Speaker group ID (e.g., pyannote:SPEAKER_00)")
    segment_id: str = Field(..., description="Segment ID to split (e.g., py_0007)")
    split_time: float = Field(..., description="Time to split at (typically utterance start)")
    target_speaker_group_id: Optional[str] = Field(
        None,
        description="Optional: move the second part to this speaker group. If not provided, stays in same group."
    )


class WordSplitRequest(BaseModel):
    """Request to split a segment at specific word boundaries."""
    source: str = Field(..., description="Diarization source (pyannote, gpt4o)")
    speaker_group_id: str = Field(..., description="Speaker group ID")
    segment_id: str = Field(..., description="Segment ID to split")
    split_word_indices: List[int] = Field(
        ...,
        description="Word indices AFTER which to split (0-based). E.g., [5] splits after word 5."
    )
    rebuild_downstream: bool = Field(
        True,
        description="Rebuild voice clusters and transcript after split"
    )


class WordSplitResponse(BaseModel):
    """Response after word-level smart split."""
    status: str = Field(..., description="Status (ok, error)")
    ep_id: str = Field(..., description="Episode ID")
    source: str = Field(..., description="Diarization source")
    original_segment_id: str = Field(..., description="Original segment that was split")
    new_segments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="New segments created from the split"
    )
    message: Optional[str] = Field(None, description="Status message")


class SegmentWordsRequest(BaseModel):
    """Request to get word-level info for a segment."""
    source: str = Field(..., description="Diarization source (pyannote, gpt4o)")
    speaker_group_id: str = Field(..., description="Speaker group ID")
    segment_id: str = Field(..., description="Segment ID")


class SegmentWordsResponse(BaseModel):
    """Response with word-level info for a segment."""
    segment_id: str
    start: float
    end: float
    text: str
    words: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Word list with timing [{w, t0, t1}, ...]"
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _get_audio_paths(ep_id: str) -> dict:
    """Get paths for audio artifacts."""
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))

    audio_dir = data_root / "audio" / ep_id
    manifests_dir = data_root / "manifests" / ep_id

    return {
        "audio_original": audio_dir / "episode_original.wav",
        "audio_vocals": audio_dir / "episode_vocals.wav",
        "audio_vocals_enhanced": audio_dir / "episode_vocals_enhanced.wav",
        "audio_final": audio_dir / "episode_final_voice_only.wav",
        "diarization": manifests_dir / "audio_diarization.jsonl",
        "asr_raw": manifests_dir / "audio_asr_raw.jsonl",
        "diarization_combined": manifests_dir / "audio_diarization_combined.jsonl",
        "voice_clusters": manifests_dir / "audio_voice_clusters.json",
        "voice_clusters_gpt4o": manifests_dir / "audio_voice_clusters_gpt4o.json",
        "voice_mapping": manifests_dir / "audio_voice_mapping.json",
        "transcript_jsonl": manifests_dir / "episode_transcript.jsonl",
        "transcript_vtt": manifests_dir / "episode_transcript.vtt",
        "qc": manifests_dir / "audio_qc.json",
        "archived_segments": manifests_dir / "audio_archived_segments.json",
        "speaker_groups": manifests_dir / "audio_speaker_groups.json",
        "speaker_assignments": manifests_dir / "audio_speaker_assignments.json",
        "voiceprint_overrides": manifests_dir / "audio_voiceprint_overrides.json",
    }


def _check_artifacts_exist(ep_id: str) -> dict:
    """Check which audio artifacts exist for an episode."""
    paths = _get_audio_paths(ep_id)
    return {
        key: str(path) if path.exists() else None
        for key, path in paths.items()
    }


def _get_show_id_from_ep_id(ep_id: str) -> str:
    """Extract show_id from episode id (e.g., 'RHOSLC-S05E01' -> 'RHOSLC')."""
    if "-" in ep_id:
        return ep_id.split("-")[0].upper()
    return ep_id.upper()


def _load_qc_status(ep_id: str) -> tuple[str, dict]:
    """Load QC status from file."""
    paths = _get_audio_paths(ep_id)
    qc_path = paths["qc"]

    if not qc_path.exists():
        return "unknown", {}

    try:
        with qc_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("status", "unknown"), {
            "duration_drift_pct": data.get("duration_drift_pct"),
            "snr_db": data.get("snr_db"),
            "mean_asr_conf": data.get("mean_asr_conf"),
            "mean_diarization_conf": data.get("mean_diarization_conf"),
            "voice_cluster_count": data.get("voice_cluster_count"),
            "labeled_voices": data.get("labeled_voices"),
            "unlabeled_voices": data.get("unlabeled_voices"),
        }
    except Exception as e:
        LOGGER.warning(f"Failed to load QC status for {ep_id}: {e}")
        return "error", {}


def _normalize_run_mode(run_mode: Optional[str]) -> str:
    """Normalize run mode values (treat 'redis' as 'queue')."""
    if not run_mode:
        return "local"
    if run_mode == "redis":
        return "queue"
    return run_mode


@edit_router.post("/episodes/{ep_id}/audio/smart_split")
async def smart_split_segment(ep_id: str, req: SmartSplitRequest) -> dict:
    """Smart split a diarization segment into subsegments and reassign speakers."""
    if not req.segment_id and (req.start is None or req.end is None):
        raise HTTPException(status_code=400, detail="Provide segment_id or start/end to split")

    try:
        from py_screenalytics.audio.speaker_edit import smart_split_segment as _smart_split

        result = _smart_split(
            ep_id=ep_id,
            source=req.source,
            speaker_group_id=req.speaker_group_id,
            segment_id=req.segment_id,
            start=req.start,
            end=req.end,
            expected_voices=req.expected_voices,
        )
        return result.model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Smart split failed for %s: %s", ep_id, exc)
        raise HTTPException(status_code=500, detail=f"Smart split failed: {exc}")


# =============================================================================
# API Endpoints
# =============================================================================


@router.post("/episode_audio_pipeline", response_model=AudioPipelineResponse, status_code=202)
async def start_audio_pipeline(req: AudioPipelineRequest) -> AudioPipelineResponse:
    """Start the audio pipeline for an episode.

    This endpoint kicks off the complete audio processing pipeline:
    1. Extract audio from video
    2. Separate vocals (MDX-Extra)
    3. Enhance audio (Resemble API)
    4. Speaker diarization (Pyannote)
    5. Transcription (OpenAI Whisper / Gemini)
    6. Voice clustering and bank mapping
    7. Transcript generation (JSONL + VTT)
    8. Quality control

    Args:
        req: Pipeline request with ep_id, run_mode, and options

    Returns:
        AudioPipelineResponse with job_id and status
    """
    run_mode = _normalize_run_mode(req.run_mode)

    # Validate episode exists
    video_path = get_path(req.ep_id, "video")
    if not video_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Video not found for episode {req.ep_id}. Mirror from S3 first.",
        )

    if run_mode == "queue":
        # Normalize provider before Celery chain (gemini → gemini_3)
        asr_provider = req.asr_provider
        if asr_provider == "gemini":
            asr_provider = "gemini_3"

        # Queue via Celery
        try:
            if episode_audio_pipeline_task is None:
                raise HTTPException(status_code=503, detail="Audio pipeline task unavailable")

            if hasattr(episode_audio_pipeline_task, "delay"):
                result = episode_audio_pipeline_task.delay(
                    req.ep_id,
                    overwrite=req.overwrite,
                    asr_provider=asr_provider,
                )
                job_id = getattr(result, "id", None)
                return AudioPipelineResponse(
                    job_id=job_id,
                    job_type="audio_pipeline",
                    status="queued",
                    ep_id=req.ep_id,
                    run_mode=run_mode,
                )

            result = episode_audio_pipeline_task(
                req.ep_id,
                overwrite=req.overwrite,
                asr_provider=asr_provider,
            )

            if result.get("status") == "error":
                return AudioPipelineResponse(
                    status="error",
                    ep_id=req.ep_id,
                    error=result.get("error"),
                )

            return AudioPipelineResponse(
                job_id=result.get("job_id"),
                job_type="audio_pipeline",
                status="queued",
                ep_id=req.ep_id,
                run_mode=run_mode,
            )

        except Exception as e:
            LOGGER.exception(f"Failed to queue audio pipeline: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to queue audio pipeline: {e}",
            )

    else:
        # Run locally via streaming subprocess for real-time logs
        # Uses shared infrastructure that:
        # - Registers job with PID for cancel support
        # - Kills subprocess on client disconnect (page refresh)
        # - Properly cleans up on completion
        from apps.api.routers.celery_jobs import _stream_local_subprocess

        # Normalize ASR provider name (API uses gemini_3, CLI uses gemini)
        asr_provider = req.asr_provider or "openai_whisper"
        if asr_provider == "gemini_3":
            asr_provider = "gemini"

        # Build command for audio pipeline
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "audio_pipeline_run.py"),
            "--ep-id", req.ep_id,
            "--asr-provider", asr_provider,
        ]
        if req.overwrite:
            command.append("--overwrite")
        if req.min_speakers is not None:
            command.extend(["--min-speakers", str(req.min_speakers)])
        if req.max_speakers is not None:
            command.extend(["--max-speakers", str(req.max_speakers)])

        # Progress file for UI polling
        data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
        manifests_dir = data_root / "manifests" / req.ep_id
        manifests_dir.mkdir(parents=True, exist_ok=True)
        progress_file = manifests_dir / "audio_progress.json"
        command.extend(["--progress-file", str(progress_file)])

        # Options for job tracking
        options = {
            "asr_provider": asr_provider,
            "overwrite": req.overwrite,
        }

        # Return streaming response for live log updates
        # Uses _stream_local_subprocess which:
        # - Registers job with PID (enables cancel button)
        # - Handles GeneratorExit on client disconnect
        # - Uses process groups to kill child processes
        return StreamingResponse(
            _stream_local_subprocess(command, req.ep_id, "audio_pipeline", options, timeout=7200),
            media_type="application/x-ndjson",
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache",
            },
        )


@router.get("/episode_audio_status", response_model=AudioStatusResponse)
async def get_audio_status(
    ep_id: str = Query(..., description="Episode identifier"),
) -> AudioStatusResponse:
    """Get audio pipeline status for an episode.

    Returns the current status, QC results, and available artifacts.

    Args:
        ep_id: Episode identifier

    Returns:
        AudioStatusResponse with status and summary
    """
    # Check which artifacts exist
    artifacts = _check_artifacts_exist(ep_id)

    # Determine overall status based on artifacts
    if artifacts.get("transcript_jsonl"):
        status = "succeeded"
    elif artifacts.get("audio_original"):
        status = "running"
    else:
        status = "not_started"

    # Load QC status if available
    qc_status, summary = _load_qc_status(ep_id)

    return AudioStatusResponse(
        ep_id=ep_id,
        status=status,
        qc_status=qc_status,
        summary=summary,
        artifacts=artifacts,
    )


@router.get("/episodes/{ep_id}/audio/transcript.vtt")
async def download_transcript_vtt(ep_id: str) -> Response:
    """Download WebVTT transcript for an episode.

    Args:
        ep_id: Episode identifier

    Returns:
        VTT file response
    """
    paths = _get_audio_paths(ep_id)
    vtt_path = paths["transcript_vtt"]

    if not vtt_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"VTT transcript not found for episode {ep_id}",
        )

    content = vtt_path.read_text(encoding="utf-8")
    return Response(
        content=content,
        media_type="text/vtt",
        headers={
            "Content-Disposition": f'attachment; filename="{ep_id}_transcript.vtt"'
        },
    )


@router.get("/episodes/{ep_id}/audio/transcript.jsonl")
async def download_transcript_jsonl(ep_id: str) -> Response:
    """Download JSONL transcript for an episode.

    Args:
        ep_id: Episode identifier

    Returns:
        JSONL file response
    """
    paths = _get_audio_paths(ep_id)
    jsonl_path = paths["transcript_jsonl"]

    if not jsonl_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"JSONL transcript not found for episode {ep_id}",
        )

    content = jsonl_path.read_text(encoding="utf-8")
    return Response(
        content=content,
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="{ep_id}_transcript.jsonl"'
        },
    )


@router.get("/episodes/{ep_id}/audio/qc.json")
async def download_audio_qc(ep_id: str) -> dict:
    """Get audio QC report for an episode.

    Args:
        ep_id: Episode identifier

    Returns:
        QC report JSON
    """
    paths = _get_audio_paths(ep_id)
    qc_path = paths["qc"]

    if not qc_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio QC report not found for episode {ep_id}",
        )

    with qc_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/audio/prerequisites")
@edit_router.get("/audio/prerequisites")
async def check_audio_prerequisites() -> dict:
    """Check if audio pipeline prerequisites are met.

    Returns status of required dependencies and API keys.

    Returns:
        Dict with status of each prerequisite
    """
    try:
        from py_screenalytics.audio.episode_audio_pipeline import check_pipeline_prerequisites
        return check_pipeline_prerequisites()
    except ImportError as e:
        return {
            "error": f"Audio module not available: {e}",
            "ffmpeg": False,
            "soundfile": False,
            "demucs": False,
            "nemo": False,  # NeMo MSDD diarization
            "openai": bool(os.environ.get("OPENAI_API_KEY")),
            "gemini": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
            "resemble": bool(os.environ.get("RESEMBLE_API_KEY")),
        }


@router.post("/episodes/{ep_id}/audio/voices/assign", response_model=VoiceAssignResponse)
async def assign_voice_cluster(
    ep_id: str,
    req: VoiceAssignRequest,
) -> VoiceAssignResponse:
    """Assign a voice cluster to a cast member or custom label.

    Updates the voice mapping file with the new assignment.

    Args:
        ep_id: Episode identifier
        req: Voice assignment request

    Returns:
        VoiceAssignResponse with assignment details
    """
    paths = _get_audio_paths(ep_id)
    mapping_path = paths["voice_mapping"]

    if not mapping_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Voice mapping not found for episode {ep_id}",
        )

    # Validate request - must have either cast_id or custom_label
    if not req.cast_id and not req.custom_label:
        raise HTTPException(
            status_code=400,
            detail="Must provide either cast_id or custom_label",
        )

    try:
        # Load current mapping
        with mapping_path.open("r", encoding="utf-8") as f:
            mapping_data = json.load(f)

        # Find the cluster in mapping
        cluster_entry = None
        cluster_idx = -1
        for i, entry in enumerate(mapping_data):
            if entry.get("voice_cluster_id") == req.voice_cluster_id:
                cluster_entry = entry
                cluster_idx = i
                break

        if cluster_entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"Voice cluster {req.voice_cluster_id} not found in mapping",
            )

        # Determine speaker info
        if req.cast_id:
            # Assign to cast member
            speaker_id = f"SPK_{req.cast_id.upper()}"
            speaker_display_name = req.cast_id.replace("_", " ").title()
            voice_bank_id = f"voice_{req.cast_id.lower()}"

            # Try to get cast member name from API
            try:
                from apps.api.routers.cast import get_cast_member
                show_id = ep_id.split("-")[0].upper()
                cast_info = await get_cast_member(show_id, req.cast_id)
                if cast_info and cast_info.get("name"):
                    speaker_display_name = cast_info["name"]
            except Exception:
                pass  # Use default name if cast lookup fails
        else:
            # Custom label
            speaker_id = f"SPK_CUSTOM_{req.custom_label.upper().replace(' ', '_')}"
            speaker_display_name = req.custom_label
            voice_bank_id = f"voice_custom_{req.custom_label.lower().replace(' ', '_')}"

        # Update the entry
        mapping_data[cluster_idx]["speaker_id"] = speaker_id
        mapping_data[cluster_idx]["speaker_display_name"] = speaker_display_name
        mapping_data[cluster_idx]["voice_bank_id"] = voice_bank_id
        mapping_data[cluster_idx]["similarity"] = 1.0  # Manual assignment = perfect match

        # Save updated mapping
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(mapping_data, f, indent=2)

        LOGGER.info(f"Assigned voice cluster {req.voice_cluster_id} to {speaker_display_name}")

        return VoiceAssignResponse(
            voice_cluster_id=req.voice_cluster_id,
            speaker_id=speaker_id,
            speaker_display_name=speaker_display_name,
            voice_bank_id=voice_bank_id,
            success=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception(f"Failed to assign voice cluster: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to assign voice cluster: {e}",
        )


# =============================================================================
# Speaker Group Assignment Endpoints
# =============================================================================


@router.get("/episodes/{ep_id}/audio/speaker-assignments", response_model=SpeakerAssignmentsResponse)
async def get_speaker_assignments(ep_id: str) -> SpeakerAssignmentsResponse:
    """Get current speaker group assignments and available cast members.

    Returns:
        - Current assignments from audio_speaker_assignments.json
        - Available cast members from show's cast profiles
        - Speaker groups from audio_speaker_groups.json with utterance analysis
    """
    import json
    from datetime import datetime

    from apps.api.services.cast import CastService
    from py_screenalytics.audio.models import ASRSegment
    from py_screenalytics.audio.speaker_groups import load_speaker_groups_manifest
    from py_screenalytics.audio.voiceprint_selection import (
        load_speaker_assignments,
        load_voiceprint_overrides,
        analyze_segment_utterances,
        analyze_speaker_group_purity,
    )

    paths = _get_audio_paths(ep_id)

    # Load ASR segments for utterance analysis
    asr_segments = []
    asr_path = paths.get("asr_raw")
    if asr_path and asr_path.exists():
        try:
            with open(asr_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        seg_data = json.loads(line.strip())
                        asr_segments.append(ASRSegment(**seg_data))
        except Exception as e:
            LOGGER.warning(f"Failed to load ASR for {ep_id}: {e}")

    # Load voiceprint overrides
    override_lookup = {}
    overrides_manifest = load_voiceprint_overrides(paths["voiceprint_overrides"])
    if overrides_manifest:
        override_lookup = overrides_manifest.get_override_lookup()

    # Load current assignments
    assignments_list = []
    assignments_manifest = load_speaker_assignments(paths["speaker_assignments"])
    if assignments_manifest:
        assignments_list = [a.to_dict() for a in assignments_manifest.assignments]

    # Load cast members for this show
    show_id = _get_show_id_from_ep_id(ep_id)
    cast_service = CastService()
    try:
        cast_members = cast_service.list_cast(show_id)
        # Simplify to just what UI needs
        cast_list = [
            {
                "cast_id": m.get("cast_id"),
                "name": m.get("name"),
                "role": m.get("role"),
                "status": m.get("status"),
                "has_voiceprint": bool(m.get("voiceprint_blob")),
            }
            for m in cast_members
        ]
    except Exception as e:
        LOGGER.warning(f"Failed to load cast for {show_id}: {e}")
        cast_list = []

    # Load speaker groups
    speaker_groups_list = []
    sg_path = paths["speaker_groups"]
    if sg_path.exists():
        try:
            sg_manifest = load_speaker_groups_manifest(sg_path)
            for source in sg_manifest.sources:
                for group in source.speakers:
                    # Calculate mean diar_confidence for the group
                    confidences = [
                        seg.diar_confidence for seg in group.segments
                        if seg.diar_confidence is not None
                    ]
                    mean_conf = sum(confidences) / len(confidences) if confidences else None

                    # Include segment details with utterance analysis for UI display
                    segments_data = []
                    for seg in sorted(group.segments, key=lambda s: s.start):
                        seg_data = {
                            "segment_id": seg.segment_id,
                            "start": seg.start,
                            "end": seg.end,
                            "duration": round(seg.end - seg.start, 2),
                            "diar_confidence": seg.diar_confidence,
                        }

                        # Add utterance analysis if ASR available
                        is_clean = True
                        if asr_segments:
                            analysis = analyze_segment_utterances(
                                asr_segments=asr_segments,
                                seg_start=seg.start,
                                seg_end=seg.end,
                            )
                            seg_data["utterance_count"] = analysis.utterance_count
                            seg_data["dialogue_risk"] = analysis.dialogue_risk
                            seg_data["dialogue_reasons"] = analysis.dialogue_reasons
                            seg_data["has_narrator_speech"] = analysis.has_narrator_speech
                            seg_data["is_clean_for_voiceprint"] = analysis.is_clean_for_voiceprint
                            seg_data["rejection_reason"] = analysis.rejection_reason
                            is_clean = analysis.is_clean_for_voiceprint

                        # Add voiceprint override and computed selected value
                        override = override_lookup.get(seg.segment_id)
                        seg_data["voiceprint_override"] = override

                        # Compute effective voiceprint_selected based on override precedence
                        if override == "force_exclude":
                            voiceprint_selected = False
                        elif override == "force_include":
                            voiceprint_selected = True
                        else:
                            voiceprint_selected = is_clean
                        seg_data["voiceprint_selected"] = voiceprint_selected

                        # Find applicable assignment for this segment (time-range > group-level)
                        seg_assignment = None
                        if assignments_manifest:
                            assignment = assignments_manifest.find_assignment_for_segment(
                                speaker_group_id=group.speaker_group_id,
                                seg_start=seg.start,
                                seg_end=seg.end,
                            )
                            if assignment:
                                seg_assignment = {
                                    "cast_id": assignment.cast_id,
                                    "cast_display_name": assignment.cast_display_name,
                                    "is_time_range": assignment.is_time_range_assignment(),
                                    "start": assignment.start,
                                    "end": assignment.end,
                                }
                        seg_data["segment_assignment"] = seg_assignment

                        segments_data.append(seg_data)

                    # Analyze group purity (cross-segment speaker mixing)
                    group_purity = "high"
                    group_purity_warnings = []
                    if asr_segments:
                        group_segs = [{"start": s.start, "end": s.end} for s in group.segments]
                        group_purity, group_purity_warnings = analyze_speaker_group_purity(
                            group_segs, asr_segments
                        )

                    speaker_groups_list.append({
                        "speaker_group_id": group.speaker_group_id,
                        "speaker_label": group.speaker_label,
                        "source": source.source,
                        "segment_count": group.segment_count,
                        "total_duration": round(group.total_duration, 2),
                        "mean_diar_confidence": round(mean_conf, 3) if mean_conf else None,
                        "segments": segments_data,
                        "group_purity": group_purity,
                        "group_purity_warnings": group_purity_warnings,
                    })
        except Exception as e:
            LOGGER.warning(f"Failed to load speaker groups for {ep_id}: {e}")

    return SpeakerAssignmentsResponse(
        ep_id=ep_id,
        assignments=assignments_list,
        cast_members=cast_list,
        speaker_groups=speaker_groups_list,
    )


@router.post("/episodes/{ep_id}/audio/speaker-assignments", response_model=SpeakerAssignmentsResponse)
async def set_speaker_assignments(
    ep_id: str,
    req: SpeakerAssignmentsRequest,
) -> SpeakerAssignmentsResponse:
    """Set or update speaker group assignments.

    Supports both group-level and time-range-level assignments:
    - Group-level: start and end are omitted → applies to entire speaker group
    - Time-range: start and end are provided → applies only to that time window

    Time-range assignments take precedence over group-level assignments.

    Replaces all assignments with the provided list.
    To remove an assignment, simply omit it from the list.

    Args:
        ep_id: Episode identifier
        req: List of speaker group to cast member assignments (supports time-range)

    Returns:
        Updated assignments and available options
    """
    from datetime import datetime, timezone

    from apps.api.services.cast import CastService
    from py_screenalytics.audio.voiceprint_selection import (
        SpeakerAssignment,
        SpeakerAssignmentsManifest,
        save_speaker_assignments,
    )

    paths = _get_audio_paths(ep_id)

    # Build new assignments
    new_assignments = []
    for item in req.assignments:
        # Resolve cast display name if not provided
        cast_display_name = item.cast_display_name
        if not cast_display_name:
            show_id = _get_show_id_from_ep_id(ep_id)
            try:
                cast_service = CastService()
                cast_member = cast_service.get_cast_member(show_id, item.cast_id)
                if cast_member:
                    cast_display_name = cast_member.get("name")
            except Exception as exc:
                LOGGER.debug("[cast-lookup] Failed to get cast member %s: %s", item.cast_id, exc)

        new_assignments.append(SpeakerAssignment(
            source=item.source,
            speaker_group_id=item.speaker_group_id,
            cast_id=item.cast_id,
            cast_display_name=cast_display_name,
            assigned_by="user",
            assigned_at=datetime.now(timezone.utc).isoformat(),
            start=item.start,  # Time-range support
            end=item.end,  # Time-range support
        ))

    # Create and save manifest
    manifest = SpeakerAssignmentsManifest(
        ep_id=ep_id,
        assignments=new_assignments,
    )
    save_speaker_assignments(manifest, paths["speaker_assignments"])

    # Log assignment breakdown
    group_level = sum(1 for a in new_assignments if a.start is None)
    time_range = len(new_assignments) - group_level
    LOGGER.info(
        f"Saved {len(new_assignments)} speaker assignments for {ep_id}: "
        f"{group_level} group-level, {time_range} time-range"
    )

    # Return updated state
    return await get_speaker_assignments(ep_id)


class SingleAssignmentRequest(BaseModel):
    """Request to upsert a single speaker assignment."""
    source: str = Field(..., description="Diarization source (pyannote, gpt4o)")
    speaker_group_id: str = Field(..., description="Speaker group ID")
    cast_id: Optional[str] = Field(None, description="Cast member ID (null to remove)")
    start: Optional[float] = Field(None, description="Segment start time (for time-range)")
    end: Optional[float] = Field(None, description="Segment end time (for time-range)")


class SingleAssignmentResponse(BaseModel):
    """Response after upserting a single assignment."""
    status: str = Field(..., description="Status (ok, removed, error)")
    assignment: Optional[Dict[str, Any]] = Field(None, description="Created/updated assignment")
    message: Optional[str] = Field(None, description="Optional message")


@router.post("/episodes/{ep_id}/audio/speaker-assignment")
async def upsert_single_speaker_assignment(
    ep_id: str,
    req: SingleAssignmentRequest,
) -> SingleAssignmentResponse:
    """Upsert or remove a single speaker assignment.

    This is a convenience endpoint for per-segment cast assignment in the UI.
    It handles both group-level and time-range assignments.

    - If cast_id is provided: creates/updates an assignment
    - If cast_id is null/empty: removes the matching assignment

    For time-range assignments (start/end provided):
    - Creates a segment-specific assignment that takes precedence over group-level

    Args:
        ep_id: Episode identifier
        req: Assignment details

    Returns:
        Status and the created/updated assignment
    """
    from datetime import datetime, timezone

    from apps.api.services.cast import CastService
    from py_screenalytics.audio.voiceprint_selection import (
        SpeakerAssignmentsManifest,
        load_speaker_assignments,
        save_speaker_assignments,
    )

    paths = _get_audio_paths(ep_id)

    # Load or create manifest
    manifest = load_speaker_assignments(paths["speaker_assignments"])
    if not manifest:
        manifest = SpeakerAssignmentsManifest(ep_id=ep_id)

    # Handle removal
    if not req.cast_id:
        removed = manifest.remove_assignment(
            speaker_group_id=req.speaker_group_id,
            start=req.start,
            end=req.end,
        )
        if removed:
            save_speaker_assignments(manifest, paths["speaker_assignments"])
            LOGGER.info(
                f"Removed assignment for {req.speaker_group_id} "
                f"[{req.start}-{req.end}] in {ep_id}"
            )
            return SingleAssignmentResponse(
                status="removed",
                message=f"Assignment removed for {req.speaker_group_id}"
            )
        else:
            return SingleAssignmentResponse(
                status="ok",
                message="No matching assignment found to remove"
            )

    # Resolve cast display name
    cast_display_name = None
    show_id = _get_show_id_from_ep_id(ep_id)
    try:
        cast_service = CastService()
        cast_member = cast_service.get_cast_member(show_id, req.cast_id)
        if cast_member:
            cast_display_name = cast_member.get("name")
    except Exception as exc:
        LOGGER.debug("[cast-lookup] Failed to get cast member %s: %s", req.cast_id, exc)

    # Upsert the assignment
    new_assignment = manifest.upsert_assignment(
        source=req.source,
        speaker_group_id=req.speaker_group_id,
        cast_id=req.cast_id,
        cast_display_name=cast_display_name,
        start=req.start,
        end=req.end,
    )

    save_speaker_assignments(manifest, paths["speaker_assignments"])

    assignment_type = "time-range" if req.start is not None else "group-level"
    LOGGER.info(
        f"Upserted {assignment_type} assignment: {req.speaker_group_id} -> {req.cast_id} "
        f"[{req.start}-{req.end}] in {ep_id}"
    )

    return SingleAssignmentResponse(
        status="ok",
        assignment=new_assignment.to_dict(),
        message=f"Assignment saved: {req.speaker_group_id} -> {cast_display_name or req.cast_id}"
    )


@router.post("/episodes/{ep_id}/audio/voiceprint-overrides", response_model=VoiceprintOverrideResponse)
async def set_voiceprint_override(
    ep_id: str,
    req: VoiceprintOverrideRequest,
) -> VoiceprintOverrideResponse:
    """Set or clear a voiceprint override for a specific segment.

    Overrides control whether a segment is used for voiceprint creation:
    - force_include: Always use this segment (overrides heuristics)
    - force_exclude: Never use this segment
    - null/None: Clear override, use heuristic auto-selection

    Args:
        ep_id: Episode identifier
        req: Override request with source, speaker_group_id, segment_id, and override value

    Returns:
        Status and updated segment state
    """
    from py_screenalytics.audio.models import ASRSegment
    from py_screenalytics.audio.voiceprint_selection import (
        VoiceprintOverridesManifest,
        load_voiceprint_overrides,
        save_voiceprint_overrides,
        analyze_segment_utterances,
    )
    from py_screenalytics.audio.speaker_groups import load_speaker_groups_manifest
    import json

    paths = _get_audio_paths(ep_id)

    # Validate override value
    if req.voiceprint_override not in (None, "force_include", "force_exclude"):
        return VoiceprintOverrideResponse(
            status="error",
            segment={},
            message=f"Invalid override value: {req.voiceprint_override}. Must be 'force_include', 'force_exclude', or null."
        )

    # Load or create overrides manifest
    overrides_manifest = load_voiceprint_overrides(paths["voiceprint_overrides"])
    if overrides_manifest is None:
        overrides_manifest = VoiceprintOverridesManifest(ep_id=ep_id)

    # Set the override
    overrides_manifest.set_override(
        source=req.source,
        speaker_group_id=req.speaker_group_id,
        segment_id=req.segment_id,
        override=req.voiceprint_override,
        reason="Manual override from Voices Review UI",
    )

    # Save the manifest
    save_voiceprint_overrides(overrides_manifest, paths["voiceprint_overrides"])

    # Find the segment to return its updated state
    segment_data = {
        "source": req.source,
        "speaker_group_id": req.speaker_group_id,
        "segment_id": req.segment_id,
        "voiceprint_override": req.voiceprint_override,
        "voiceprint_selected": None,  # Will be computed below
    }

    # Load speaker groups to get segment details
    sg_path = paths["speaker_groups"]
    if sg_path.exists():
        sg_manifest = load_speaker_groups_manifest(sg_path)
        for source in sg_manifest.sources:
            for group in source.speakers:
                if group.speaker_group_id == req.speaker_group_id:
                    for seg in group.segments:
                        if seg.segment_id == req.segment_id:
                            segment_data["start"] = seg.start
                            segment_data["end"] = seg.end
                            segment_data["duration"] = round(seg.end - seg.start, 2)
                            segment_data["diar_confidence"] = seg.diar_confidence
                            break

    # Load ASR for analysis
    asr_segments = []
    asr_path = paths.get("asr_raw")
    if asr_path and asr_path.exists():
        try:
            with open(asr_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        seg_data = json.loads(line.strip())
                        asr_segments.append(ASRSegment(**seg_data))
        except Exception as exc:
            LOGGER.debug("[asr-load] Failed to load ASR segments from %s: %s", asr_path, exc)

    # Compute is_clean and voiceprint_selected
    is_clean = True
    if asr_segments and "start" in segment_data:
        analysis = analyze_segment_utterances(
            asr_segments=asr_segments,
            seg_start=segment_data["start"],
            seg_end=segment_data["end"],
        )
        is_clean = analysis.is_clean_for_voiceprint
        segment_data["utterance_count"] = analysis.utterance_count
        segment_data["dialogue_risk"] = analysis.dialogue_risk
        segment_data["is_clean_for_voiceprint"] = is_clean

    # Compute voiceprint_selected based on override
    if req.voiceprint_override == "force_exclude":
        segment_data["voiceprint_selected"] = False
    elif req.voiceprint_override == "force_include":
        segment_data["voiceprint_selected"] = True
    else:
        segment_data["voiceprint_selected"] = is_clean

    LOGGER.info(
        f"Set voiceprint override for {ep_id}/{req.segment_id}: "
        f"override={req.voiceprint_override}, selected={segment_data['voiceprint_selected']}"
    )

    return VoiceprintOverrideResponse(
        status="ok",
        segment=segment_data,
        message=f"Override {'set' if req.voiceprint_override else 'cleared'} successfully"
    )


@router.post("/episodes/{ep_id}/audio/speaker-groups/split-at-utterance")
async def split_segment_at_utterance(ep_id: str, req: SplitAtUtteranceRequest) -> dict:
    """Split a speaker group segment at a specific time (utterance boundary).

    This allows splitting a diarization segment when it contains multiple speakers.
    The original segment is split into two parts at the specified time.

    Use case: A segment shows 3 utterances, but utterance 2 is actually a different
    speaker. Split at utterance 2's start time, then reassign the second part.

    Args:
        ep_id: Episode identifier
        req: Split request with source, speaker_group_id, segment_id, and split_time

    Returns:
        Dict with new segment details
    """
    from py_screenalytics.audio.speaker_groups import (
        load_speaker_groups_manifest,
        save_speaker_groups_manifest,
        update_summaries,
    )
    from py_screenalytics.audio.models import SpeakerSegment

    paths = _get_audio_paths(ep_id)
    speaker_groups_path = paths.get("speaker_groups")

    if not speaker_groups_path or not speaker_groups_path.exists():
        raise HTTPException(status_code=404, detail="Speaker groups manifest not found")

    try:
        manifest = load_speaker_groups_manifest(speaker_groups_path)

        # Find the source and speaker group
        target_source = None
        target_group = None
        target_segment = None
        segment_idx = -1

        for source in manifest.sources:
            if source.source == req.source:
                target_source = source
                for group in source.speakers:
                    if group.speaker_group_id == req.speaker_group_id:
                        target_group = group
                        for idx, seg in enumerate(group.segments):
                            if seg.segment_id == req.segment_id:
                                target_segment = seg
                                segment_idx = idx
                                break
                        break
                break

        if target_source is None:
            raise HTTPException(status_code=404, detail=f"Source '{req.source}' not found")
        if target_group is None:
            raise HTTPException(status_code=404, detail=f"Speaker group '{req.speaker_group_id}' not found")
        if target_segment is None:
            raise HTTPException(status_code=404, detail=f"Segment '{req.segment_id}' not found in group")

        # Validate split time is within segment bounds
        if req.split_time <= target_segment.start or req.split_time >= target_segment.end:
            raise HTTPException(
                status_code=400,
                detail=f"Split time {req.split_time} must be between segment bounds ({target_segment.start}, {target_segment.end})"
            )

        # Create two new segments from the split
        # Generate new segment IDs - use original ID with suffixes
        base_id = req.segment_id.rsplit("_", 1)[0] if "_" in req.segment_id else req.segment_id
        # Find max existing segment number for this prefix to avoid collisions
        existing_nums = []
        for group in target_source.speakers:
            for seg in group.segments:
                if seg.segment_id.startswith(base_id):
                    try:
                        num_part = seg.segment_id.split("_")[-1]
                        existing_nums.append(int(num_part))
                    except (ValueError, IndexError):
                        pass

        next_num = max(existing_nums, default=0) + 1

        seg1 = SpeakerSegment(
            segment_id=f"{base_id}_{next_num:04d}",
            start=target_segment.start,
            end=req.split_time,
            diar_confidence=target_segment.diar_confidence,
        )

        seg2 = SpeakerSegment(
            segment_id=f"{base_id}_{next_num + 1:04d}",
            start=req.split_time,
            end=target_segment.end,
            diar_confidence=target_segment.diar_confidence,
        )

        # Remove original segment
        target_group.segments.pop(segment_idx)

        # Add first part to original group
        target_group.segments.insert(segment_idx, seg1)

        # Handle second part - either same group or move to target
        if req.target_speaker_group_id and req.target_speaker_group_id != req.speaker_group_id:
            # Find target group and add segment there
            moved_to_group = None
            for group in target_source.speakers:
                if group.speaker_group_id == req.target_speaker_group_id:
                    moved_to_group = group
                    group.segments.append(seg2)
                    group.segments.sort(key=lambda s: s.start)
                    break

            if moved_to_group is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Target speaker group '{req.target_speaker_group_id}' not found"
                )
        else:
            # Keep in same group
            target_group.segments.insert(segment_idx + 1, seg2)

        # Re-sort segments in both groups
        target_group.segments.sort(key=lambda s: s.start)

        # Update summaries
        manifest = update_summaries(manifest)

        # Save manifest
        save_speaker_groups_manifest(manifest, speaker_groups_path)

        LOGGER.info(
            f"Split segment {req.segment_id} at {req.split_time}: "
            f"created {seg1.segment_id} ({seg1.start:.2f}-{seg1.end:.2f}) and "
            f"{seg2.segment_id} ({seg2.start:.2f}-{seg2.end:.2f})"
        )

        return {
            "status": "ok",
            "original_segment": {
                "segment_id": req.segment_id,
                "start": target_segment.start,
                "end": target_segment.end,
            },
            "new_segments": [
                {
                    "segment_id": seg1.segment_id,
                    "start": seg1.start,
                    "end": seg1.end,
                    "speaker_group_id": req.speaker_group_id,
                },
                {
                    "segment_id": seg2.segment_id,
                    "start": seg2.start,
                    "end": seg2.end,
                    "speaker_group_id": req.target_speaker_group_id or req.speaker_group_id,
                },
            ],
            "message": f"Split at {req.split_time:.2f}s - segment divided into 2 parts"
        }

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception(f"Failed to split segment at utterance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/episodes/{ep_id}/audio/segments/{segment_id}/words", response_model=SegmentWordsResponse)
async def get_segment_words(
    ep_id: str,
    segment_id: str,
    source: str = "nemo",
    speaker_group_id: Optional[str] = None,
) -> SegmentWordsResponse:
    """Get word-level timing information for a segment.

    This is useful for the Smart Split UI to show word boundaries
    and let users select where to split.

    Args:
        ep_id: Episode identifier
        segment_id: Segment ID to get words for
        source: Diarization source (nemo, pyannote, gpt4o - legacy)
        speaker_group_id: Optional speaker group ID (inferred if not provided)

    Returns:
        SegmentWordsResponse with word list and timing
    """
    from py_screenalytics.audio.speaker_groups import (
        load_speaker_groups_manifest,
        speaker_group_lookup,
    )
    from py_screenalytics.audio.speaker_edit import _load_asr_words_for_segment

    paths = _get_audio_paths(ep_id)
    speaker_groups_path = paths.get("speaker_groups")
    asr_path = paths.get("asr_raw")

    if not speaker_groups_path or not speaker_groups_path.exists():
        raise HTTPException(status_code=404, detail="Speaker groups manifest not found")
    if not asr_path or not asr_path.exists():
        raise HTTPException(status_code=404, detail="ASR data not found")

    try:
        manifest = load_speaker_groups_manifest(speaker_groups_path)
        groups = speaker_group_lookup(manifest)

        # Find the segment
        target_segment = None
        found_group_id = speaker_group_id

        for gid, group in groups.items():
            if speaker_group_id and gid != speaker_group_id:
                continue
            for seg in group.segments:
                if seg.segment_id == segment_id:
                    target_segment = seg
                    found_group_id = gid
                    break
            if target_segment:
                break

        if target_segment is None:
            raise HTTPException(
                status_code=404,
                detail=f"Segment '{segment_id}' not found"
            )

        # Load words
        words = _load_asr_words_for_segment(
            asr_path, target_segment.start, target_segment.end
        )
        text = " ".join(w.get("w", "") for w in words)

        return SegmentWordsResponse(
            segment_id=segment_id,
            start=target_segment.start,
            end=target_segment.end,
            text=text,
            words=words,
        )

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception(f"Failed to get segment words: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/episodes/{ep_id}/audio/segments/word-split", response_model=WordSplitResponse)
async def word_level_smart_split(ep_id: str, req: WordSplitRequest) -> WordSplitResponse:
    """Split a diarization segment at specific word boundaries.

    This is a user-driven split that creates subsegments at precise word
    boundaries. Use this when a single diarization segment contains multiple
    speakers (e.g., "I didn't make out with someone and neither did Todd."
    where two different people are speaking).

    Workflow:
    1. Call GET /episodes/{ep_id}/audio/segments/{segment_id}/words to see words
    2. User selects split point(s) in UI (e.g., after word 5 "someone")
    3. Call this endpoint with split_word_indices=[5]
    4. Original segment is replaced with 2+ new segments
    5. User can then assign each segment to different cast members

    Args:
        ep_id: Episode identifier
        req: WordSplitRequest with source, speaker_group_id, segment_id, split_word_indices

    Returns:
        WordSplitResponse with new segment details
    """
    from py_screenalytics.audio.speaker_edit import smart_split_segment_by_words

    try:
        result = smart_split_segment_by_words(
            ep_id=ep_id,
            source=req.source,
            speaker_group_id=req.speaker_group_id,
            segment_id=req.segment_id,
            split_word_indices=req.split_word_indices,
            rebuild_downstream=req.rebuild_downstream,
        )

        return WordSplitResponse(
            status="ok",
            ep_id=result.ep_id,
            source=result.source,
            original_segment_id=result.original_segment_id,
            new_segments=[
                {
                    "segment_id": s.segment_id,
                    "start": s.start,
                    "end": s.end,
                    "speaker_group_id": s.speaker_group_id,
                }
                for s in result.new_segments
            ],
            message=f"Split into {len(result.new_segments)} segments at word boundaries"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        LOGGER.exception(f"Failed to word-split segment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Phased Pipeline Endpoints
# =============================================================================


@router.post("/episode_audio_files", response_model=AudioPipelineResponse)
async def start_audio_files(req: AudioFilesRequest) -> AudioPipelineResponse:
    """Phase 1: Create audio files (extract, separate, enhance).

    Args:
        req: Request with ep_id and options

    Returns:
        AudioPipelineResponse with job_id and status
    """
    # Validate episode exists
    video_path = get_path(req.ep_id, "video")
    if not video_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Video not found for episode {req.ep_id}. Mirror from S3 first.",
        )

    try:
        from apps.api.jobs_audio import episode_audio_files_async

        result = episode_audio_files_async(
            req.ep_id,
            overwrite=req.overwrite,
        )

        if result.get("status") == "error":
            return AudioPipelineResponse(
                status="error",
                ep_id=req.ep_id,
                error=result.get("error"),
            )

        return AudioPipelineResponse(
            job_id=result.get("job_id"),
            status="queued",
            ep_id=req.ep_id,
            run_mode="queue",
        )

    except Exception as e:
        LOGGER.exception(f"Failed to queue audio files job: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue audio files job: {e}",
        )


@router.post("/episode_audio_diarize_transcribe", response_model=AudioPipelineResponse)
async def start_diarize_transcribe(req: DiarizeTranscribeRequest) -> AudioPipelineResponse:
    """Phase 2: Diarization + Transcription + Initial Clustering.

    Args:
        req: Request with ep_id, asr_provider, and options

    Returns:
        AudioPipelineResponse with job_id and status
    """
    # Validate audio files exist
    paths = _get_audio_paths(req.ep_id)
    if not paths["audio_vocals"].exists():
        raise HTTPException(
            status_code=400,
            detail=f"Audio files not found for episode {req.ep_id}. Run 'Create Audio Files' first.",
        )

    try:
        from apps.api.jobs_audio import episode_audio_diarize_transcribe_async

        result = episode_audio_diarize_transcribe_async(
            req.ep_id,
            asr_provider=req.asr_provider,
            overwrite=req.overwrite,
        )

        if result.get("status") == "error":
            return AudioPipelineResponse(
                status="error",
                ep_id=req.ep_id,
                error=result.get("error"),
            )

        return AudioPipelineResponse(
            job_id=result.get("job_id"),
            status="queued",
            ep_id=req.ep_id,
            run_mode="queue",
        )

    except Exception as e:
        LOGGER.exception(f"Failed to queue diarize/transcribe job: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue diarize/transcribe job: {e}",
        )


@router.post("/episode_audio_finalize", response_model=AudioPipelineResponse)
async def start_finalize_transcript(req: FinalizeTranscriptRequest) -> AudioPipelineResponse:
    """Phase 4: Finalize transcript (align -> export -> qc).

    Args:
        req: Request with ep_id and options

    Returns:
        AudioPipelineResponse with job_id and status
    """
    # Validate prerequisites exist
    paths = _get_audio_paths(req.ep_id)
    if not paths["diarization"].exists() or not paths["asr_raw"].exists():
        raise HTTPException(
            status_code=400,
            detail=f"Diarization or ASR not found for episode {req.ep_id}. Run 'Diarization + Transcription' first.",
        )

    try:
        from apps.api.jobs_audio import episode_audio_finalize_async

        result = episode_audio_finalize_async(
            req.ep_id,
            overwrite=req.overwrite,
        )

        if result.get("status") == "error":
            return AudioPipelineResponse(
                status="error",
                ep_id=req.ep_id,
                error=result.get("error"),
            )

        return AudioPipelineResponse(
            job_id=result.get("job_id"),
            status="queued",
            ep_id=req.ep_id,
            run_mode="queue",
        )

    except Exception as e:
        LOGGER.exception(f"Failed to queue finalize job: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue finalize job: {e}",
        )


# =============================================================================
# Voice Cluster Management Endpoints
# =============================================================================


class SegmentMoveRequest(BaseModel):
    """Request to move a segment to a different cluster."""
    segment_start: float = Field(..., description="Segment start time in seconds")
    segment_end: float = Field(..., description="Segment end time in seconds")
    from_cluster_id: str = Field(..., description="Current cluster ID")
    to_cluster_id: str = Field(..., description="Target cluster ID")


class ClusterMergeRequest(BaseModel):
    """Request to merge two clusters."""
    source_cluster_id: str = Field(..., description="Cluster to merge from (will be deleted)")
    target_cluster_id: str = Field(..., description="Cluster to merge into")


class ClusterCreateRequest(BaseModel):
    """Request to create a new cluster from segments."""
    segments: list = Field(..., description="List of segments [{start, end, from_cluster_id}]")
    new_cluster_label: Optional[str] = Field(None, description="Optional label for new cluster")


class SegmentSplitRequest(BaseModel):
    """Request to split a segment into multiple based on ASR boundaries."""
    cluster_id: str = Field(..., description="Cluster containing the segment")
    segment_start: float = Field(..., description="Original segment start time")
    segment_end: float = Field(..., description="Original segment end time")
    split_points: List[Dict] = Field(
        ...,
        description="List of new segment boundaries [{start, end}, ...]",
    )


class SmartSplitRequest(BaseModel):
    """Request for smart split that uses voice embeddings to assign segments."""
    cluster_id: str = Field(..., description="Cluster containing the segment")
    segment_start: float = Field(..., description="Original segment start time")
    segment_end: float = Field(..., description="Original segment end time")
    split_points: List[Dict] = Field(
        ...,
        description="List of new segment boundaries [{start, end}, ...]",
    )
    auto_assign: bool = Field(
        True,
        description="If True, automatically assign to best matching cluster; if False, keep all in original cluster",
    )
    min_similarity: float = Field(
        0.65,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold for auto-assignment",
    )


class ClusterReclusterRequest(BaseModel):
    """Request to re-run clustering with different parameters."""
    similarity_threshold: float = Field(0.35, ge=0.1, le=0.9, description="Similarity threshold")
    min_segments_per_cluster: int = Field(1, ge=1, description="Minimum segments per cluster")
    run_mode: Literal["queue", "local", "redis"] = Field(
        "local",
        description="Execution mode: 'queue' for Celery background job, 'local' for streaming subprocess",
    )


class SegmentAssignCastRequest(BaseModel):
    """Request to move a segment to a new cluster and assign it to a cast member."""
    segment_start: float = Field(..., description="Segment start time in seconds")
    segment_end: float = Field(..., description="Segment end time in seconds")
    from_cluster_id: str = Field(..., description="Current cluster ID")
    cast_id: str = Field(..., description="Cast member ID to assign the new cluster to")


@router.post("/episodes/{ep_id}/audio/segments/move")
async def move_segment(ep_id: str, req: SegmentMoveRequest) -> dict:
    """Move a segment from one cluster to another.

    Args:
        ep_id: Episode identifier
        req: Move request with segment times and cluster IDs

    Returns:
        Success status and updated cluster info
    """
    paths = _get_audio_paths(ep_id)
    clusters_path = paths["voice_clusters"]

    if not clusters_path.exists():
        raise HTTPException(status_code=404, detail="Voice clusters not found")

    try:
        with clusters_path.open("r", encoding="utf-8") as f:
            clusters = json.load(f)

        # Find source and target clusters
        source_cluster = None
        target_cluster = None
        segment_to_move = None

        for cluster in clusters:
            if cluster["voice_cluster_id"] == req.from_cluster_id:
                source_cluster = cluster
                # Find the segment
                for seg in cluster["segments"]:
                    if abs(seg["start"] - req.segment_start) < 0.1 and abs(seg["end"] - req.segment_end) < 0.1:
                        segment_to_move = seg
                        break
            if cluster["voice_cluster_id"] == req.to_cluster_id:
                target_cluster = cluster

        if not source_cluster:
            raise HTTPException(status_code=404, detail=f"Source cluster {req.from_cluster_id} not found")
        if not target_cluster:
            raise HTTPException(status_code=404, detail=f"Target cluster {req.to_cluster_id} not found")
        if not segment_to_move:
            raise HTTPException(status_code=404, detail="Segment not found in source cluster")

        # Move segment
        source_cluster["segments"].remove(segment_to_move)
        target_cluster["segments"].append(segment_to_move)

        # Update totals
        seg_duration = segment_to_move["end"] - segment_to_move["start"]
        source_cluster["segment_count"] = len(source_cluster["segments"])
        source_cluster["total_duration"] = sum(s["end"] - s["start"] for s in source_cluster["segments"])
        target_cluster["segment_count"] = len(target_cluster["segments"])
        target_cluster["total_duration"] = sum(s["end"] - s["start"] for s in target_cluster["segments"])

        # Save
        with clusters_path.open("w", encoding="utf-8") as f:
            json.dump(clusters, f, indent=2)

        return {
            "success": True,
            "source_cluster": {
                "id": source_cluster["voice_cluster_id"],
                "segment_count": source_cluster["segment_count"],
            },
            "target_cluster": {
                "id": target_cluster["voice_cluster_id"],
                "segment_count": target_cluster["segment_count"],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception(f"Failed to move segment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/episodes/{ep_id}/audio/segments/split")
async def split_segment(ep_id: str, req: SegmentSplitRequest) -> dict:
    """Split a segment into multiple segments based on ASR boundaries.

    This is useful when a diarization segment contains multiple speakers
    and needs to be split so each speaker's portion can be assigned correctly.

    Args:
        ep_id: Episode identifier
        req: Split request with cluster_id, original segment bounds, and split points

    Returns:
        Success status and new segment details
    """
    paths = _get_audio_paths(ep_id)
    clusters_path = paths["voice_clusters"]

    if not clusters_path.exists():
        raise HTTPException(status_code=404, detail="Voice clusters not found")

    if not req.split_points or len(req.split_points) < 1:
        raise HTTPException(status_code=400, detail="At least one split point is required")

    try:
        with clusters_path.open("r", encoding="utf-8") as f:
            clusters = json.load(f)

        # Find the cluster
        target_cluster = None
        for cluster in clusters:
            if cluster["voice_cluster_id"] == req.cluster_id:
                target_cluster = cluster
                break

        if not target_cluster:
            raise HTTPException(status_code=404, detail=f"Cluster {req.cluster_id} not found")

        # Find the segment to split
        segment_to_split = None
        segment_idx = -1
        for idx, seg in enumerate(target_cluster["segments"]):
            if abs(seg["start"] - req.segment_start) < 0.1 and abs(seg["end"] - req.segment_end) < 0.1:
                segment_to_split = seg
                segment_idx = idx
                break

        if segment_to_split is None:
            raise HTTPException(
                status_code=404,
                detail=f"Segment at {req.segment_start}-{req.segment_end} not found in cluster {req.cluster_id}",
            )

        # Get diar_speaker from original segment
        diar_speaker = segment_to_split.get("diar_speaker", "SPEAKER_00")

        # Create new segments from split points
        new_segments = []
        for sp in req.split_points:
            new_seg = {
                "start": sp["start"],
                "end": sp["end"],
                "diar_speaker": diar_speaker,
            }
            new_segments.append(new_seg)

        # Remove old segment and insert new ones at the same position
        target_cluster["segments"].pop(segment_idx)
        for i, new_seg in enumerate(new_segments):
            target_cluster["segments"].insert(segment_idx + i, new_seg)

        # Update cluster stats
        target_cluster["segment_count"] = len(target_cluster["segments"])
        target_cluster["total_duration"] = sum(s["end"] - s["start"] for s in target_cluster["segments"])

        # Save
        with clusters_path.open("w", encoding="utf-8") as f:
            json.dump(clusters, f, indent=2)

        return {
            "success": True,
            "cluster_id": req.cluster_id,
            "original_segment": {
                "start": req.segment_start,
                "end": req.segment_end,
            },
            "new_segments": new_segments,
            "new_segment_count": target_cluster["segment_count"],
        }

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception(f"Failed to split segment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/episodes/{ep_id}/audio/segments/smart_split")
async def smart_split_segment(ep_id: str, req: SmartSplitRequest) -> dict:
    """Smart split that extracts voice embeddings and assigns segments to best matching clusters.

    This is useful when a segment contains multiple speakers. For each sub-segment:
    1. Extract voice embedding using pyannote
    2. Compare against all cluster centroids
    3. Assign to best matching cluster (if above min_similarity threshold)

    Args:
        ep_id: Episode identifier
        req: Smart split request with cluster_id, segment bounds, split points, and options

    Returns:
        Success status, assignments made, and any segments that stayed in original cluster
    """
    paths = _get_audio_paths(ep_id)
    clusters_path = paths["voice_clusters"]

    if not clusters_path.exists():
        raise HTTPException(status_code=404, detail="Voice clusters not found")

    if not req.split_points or len(req.split_points) < 1:
        raise HTTPException(status_code=400, detail="At least one split point is required")

    # Find audio file for embedding extraction
    audio_path = paths["audio_vocals_enhanced"]
    if not audio_path.exists():
        audio_path = paths["audio_vocals"]
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio files not found for embedding extraction")

    try:
        with clusters_path.open("r", encoding="utf-8") as f:
            clusters = json.load(f)

        # Find the source cluster
        source_cluster = None
        for cluster in clusters:
            if cluster["voice_cluster_id"] == req.cluster_id:
                source_cluster = cluster
                break

        if not source_cluster:
            raise HTTPException(status_code=404, detail=f"Cluster {req.cluster_id} not found")

        # Find the segment to split
        segment_to_split = None
        segment_idx = -1
        for idx, seg in enumerate(source_cluster["segments"]):
            if abs(seg["start"] - req.segment_start) < 0.1 and abs(seg["end"] - req.segment_end) < 0.1:
                segment_to_split = seg
                segment_idx = idx
                break

        if segment_to_split is None:
            raise HTTPException(
                status_code=404,
                detail=f"Segment at {req.segment_start}-{req.segment_end} not found in cluster {req.cluster_id}",
            )

        # Get diar_speaker from original segment
        diar_speaker = segment_to_split.get("diar_speaker", "SPEAKER_00")

        # Remove the original segment from source cluster
        source_cluster["segments"].pop(segment_idx)

        # Build lookup of cluster centroids (cluster_id -> normalized centroid)
        cluster_centroids = {}
        for cluster in clusters:
            if cluster.get("centroid"):
                import numpy as np
                centroid = np.array(cluster["centroid"])
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                cluster_centroids[cluster["voice_cluster_id"]] = centroid

        # Extract embeddings for each sub-segment and determine assignments
        assignments = []  # List of {split_point, assigned_cluster_id, similarity}
        segments_by_cluster = {}  # cluster_id -> list of new segments

        if req.auto_assign and cluster_centroids:
            # NOTE: Auto-assign with embedding extraction is temporarily disabled
            # after migration from pyannote to NeMo MSDD.
            # TODO: Implement NeMo TitaNet-based embedding extraction for Smart Split
            try:
                import numpy as np
                import soundfile as sf

                # Load audio
                audio_data, sample_rate = sf.read(audio_path)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)

                # NeMo embedding extraction for Smart Split - not yet implemented
                # For now, fall back to basic split (no auto-assignment)
                inference = None
                LOGGER.warning("Smart Split auto-assign is temporarily unavailable (NeMo migration pending)")

                if inference:
                    import torch

                    for sp in req.split_points:
                        sp_start = sp["start"]
                        sp_end = sp["end"]

                        # Extract audio for this sub-segment
                        start_sample = int(sp_start * sample_rate)
                        end_sample = int(sp_end * sample_rate)
                        segment_audio = audio_data[start_sample:end_sample]

                        # Skip very short segments (< 0.3s)
                        if len(segment_audio) < sample_rate * 0.3:
                            LOGGER.debug(f"Segment {sp_start:.2f}-{sp_end:.2f} too short for embedding")
                            # Keep in original cluster
                            if req.cluster_id not in segments_by_cluster:
                                segments_by_cluster[req.cluster_id] = []
                            segments_by_cluster[req.cluster_id].append({
                                "start": sp_start,
                                "end": sp_end,
                                "diar_speaker": diar_speaker,
                            })
                            assignments.append({
                                "start": sp_start,
                                "end": sp_end,
                                "assigned_cluster_id": req.cluster_id,
                                "similarity": None,
                                "reason": "segment_too_short",
                            })
                            continue

                        # Get embedding
                        try:
                            segment_tensor = torch.from_numpy(segment_audio).float().unsqueeze(0)
                            embedding = inference({"waveform": segment_tensor, "sample_rate": sample_rate})
                            embedding = np.array(embedding)

                            # Normalize embedding
                            norm = np.linalg.norm(embedding)
                            if norm > 0:
                                embedding = embedding / norm

                            # Find best matching cluster
                            best_cluster_id = req.cluster_id
                            best_similarity = 0.0

                            for cid, centroid in cluster_centroids.items():
                                sim = float(np.dot(embedding, centroid))
                                if sim > best_similarity:
                                    best_similarity = sim
                                    best_cluster_id = cid

                            # Only assign if above threshold
                            if best_similarity >= req.min_similarity:
                                assigned_cluster_id = best_cluster_id
                                reason = "embedding_match"
                            else:
                                assigned_cluster_id = req.cluster_id
                                reason = "below_threshold"

                            if assigned_cluster_id not in segments_by_cluster:
                                segments_by_cluster[assigned_cluster_id] = []
                            segments_by_cluster[assigned_cluster_id].append({
                                "start": sp_start,
                                "end": sp_end,
                                "diar_speaker": diar_speaker,
                            })
                            assignments.append({
                                "start": sp_start,
                                "end": sp_end,
                                "assigned_cluster_id": assigned_cluster_id,
                                "similarity": round(best_similarity, 3),
                                "reason": reason,
                            })

                        except Exception as emb_err:
                            LOGGER.warning(f"Failed to extract embedding for {sp_start:.2f}-{sp_end:.2f}: {emb_err}")
                            # Keep in original cluster
                            if req.cluster_id not in segments_by_cluster:
                                segments_by_cluster[req.cluster_id] = []
                            segments_by_cluster[req.cluster_id].append({
                                "start": sp_start,
                                "end": sp_end,
                                "diar_speaker": diar_speaker,
                            })
                            assignments.append({
                                "start": sp_start,
                                "end": sp_end,
                                "assigned_cluster_id": req.cluster_id,
                                "similarity": None,
                                "reason": "embedding_error",
                            })
                else:
                    # No inference available, keep all in original
                    for sp in req.split_points:
                        if req.cluster_id not in segments_by_cluster:
                            segments_by_cluster[req.cluster_id] = []
                        segments_by_cluster[req.cluster_id].append({
                            "start": sp["start"],
                            "end": sp["end"],
                            "diar_speaker": diar_speaker,
                        })
                        assignments.append({
                            "start": sp["start"],
                            "end": sp["end"],
                            "assigned_cluster_id": req.cluster_id,
                            "similarity": None,
                            "reason": "no_inference",
                        })

            except ImportError as ie:
                LOGGER.warning(f"Cannot import embedding dependencies: {ie}")
                # Fall back to basic split (all in original cluster)
                for sp in req.split_points:
                    if req.cluster_id not in segments_by_cluster:
                        segments_by_cluster[req.cluster_id] = []
                    segments_by_cluster[req.cluster_id].append({
                        "start": sp["start"],
                        "end": sp["end"],
                        "diar_speaker": diar_speaker,
                    })
                    assignments.append({
                        "start": sp["start"],
                        "end": sp["end"],
                        "assigned_cluster_id": req.cluster_id,
                        "similarity": None,
                        "reason": "import_error",
                    })
        else:
            # No auto-assign or no centroids - keep all in original cluster
            for sp in req.split_points:
                if req.cluster_id not in segments_by_cluster:
                    segments_by_cluster[req.cluster_id] = []
                segments_by_cluster[req.cluster_id].append({
                    "start": sp["start"],
                    "end": sp["end"],
                    "diar_speaker": diar_speaker,
                })
                assignments.append({
                    "start": sp["start"],
                    "end": sp["end"],
                    "assigned_cluster_id": req.cluster_id,
                    "similarity": None,
                    "reason": "auto_assign_disabled" if not req.auto_assign else "no_centroids",
                })

        # Add new segments to their assigned clusters
        for cluster in clusters:
            cid = cluster["voice_cluster_id"]
            if cid in segments_by_cluster:
                cluster["segments"].extend(segments_by_cluster[cid])
                cluster["segment_count"] = len(cluster["segments"])
                cluster["total_duration"] = sum(s["end"] - s["start"] for s in cluster["segments"])

        # Save updated clusters
        with clusters_path.open("w", encoding="utf-8") as f:
            json.dump(clusters, f, indent=2)

        # Update voice mapping to reflect new cluster segment counts and suggest cast
        mapping_path = paths["voice_mapping"]
        voice_bank_suggestions = []

        if mapping_path.exists():
            try:
                with mapping_path.open("r", encoding="utf-8") as f:
                    mappings = json.load(f)

                # Update segment counts in mappings
                cluster_info = {c["voice_cluster_id"]: c for c in clusters}
                for m in mappings:
                    cid = m.get("voice_cluster_id", "")
                    if cid in cluster_info:
                        m["segment_count"] = cluster_info[cid].get("segment_count", 0)
                        m["total_duration"] = cluster_info[cid].get("total_duration", 0)

                # Save updated mappings
                with mapping_path.open("w", encoding="utf-8") as f:
                    json.dump(mappings, f, indent=2)

                # Check voice bank for cast suggestions on newly assigned clusters
                data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
                voice_bank_dir = data_root / "voice_bank"

                if voice_bank_dir.exists():
                    # Extract show_id from ep_id (e.g., rhoslc-s06e02 -> rhoslc)
                    show_id = ep_id.split("-")[0].upper() if "-" in ep_id else ep_id.upper()
                    show_bank_path = voice_bank_dir / show_id / "voice_bank.json"

                    if show_bank_path.exists():
                        with show_bank_path.open("r", encoding="utf-8") as f:
                            voice_bank = json.load(f)

                        # For each cluster that received new segments, suggest cast matches
                        for cid in segments_by_cluster.keys():
                            cluster = cluster_info.get(cid, {})
                            centroid = cluster.get("centroid")
                            if not centroid:
                                continue

                            import numpy as np
                            cluster_emb = np.array(centroid)
                            norm = np.linalg.norm(cluster_emb)
                            if norm > 0:
                                cluster_emb = cluster_emb / norm

                            # Compare against voice bank entries
                            best_match = None
                            best_sim = 0.0

                            for entry in voice_bank.get("entries", []):
                                bank_emb = entry.get("embedding")
                                if not bank_emb:
                                    continue
                                bank_emb = np.array(bank_emb)
                                bnorm = np.linalg.norm(bank_emb)
                                if bnorm > 0:
                                    bank_emb = bank_emb / bnorm
                                sim = float(np.dot(cluster_emb, bank_emb))
                                if sim > best_sim and sim >= 0.7:
                                    best_sim = sim
                                    best_match = entry

                            if best_match:
                                voice_bank_suggestions.append({
                                    "cluster_id": cid,
                                    "suggested_cast_id": best_match.get("cast_id"),
                                    "suggested_name": best_match.get("name", best_match.get("cast_id")),
                                    "similarity": round(best_sim, 3),
                                })
            except Exception as mapping_err:
                LOGGER.warning(f"Failed to update mappings after smart split: {mapping_err}")

        # Count how many went to other clusters vs stayed
        moved_count = sum(1 for a in assignments if a["assigned_cluster_id"] != req.cluster_id)
        stayed_count = len(assignments) - moved_count

        return {
            "success": True,
            "original_cluster_id": req.cluster_id,
            "original_segment": {
                "start": req.segment_start,
                "end": req.segment_end,
            },
            "assignments": assignments,
            "moved_to_other_clusters": moved_count,
            "stayed_in_original": stayed_count,
            "total_new_segments": len(assignments),
            "voice_bank_suggestions": voice_bank_suggestions,
        }

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception(f"Failed to smart split segment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/episodes/{ep_id}/audio/clusters/merge")
async def merge_clusters(ep_id: str, req: ClusterMergeRequest) -> dict:
    """Merge one cluster into another.

    Args:
        ep_id: Episode identifier
        req: Merge request with source and target cluster IDs

    Returns:
        Success status and merged cluster info
    """
    paths = _get_audio_paths(ep_id)
    clusters_path = paths["voice_clusters"]

    if not clusters_path.exists():
        raise HTTPException(status_code=404, detail="Voice clusters not found")

    try:
        with clusters_path.open("r", encoding="utf-8") as f:
            clusters = json.load(f)

        source_cluster = None
        target_cluster = None
        source_idx = -1

        for i, cluster in enumerate(clusters):
            if cluster["voice_cluster_id"] == req.source_cluster_id:
                source_cluster = cluster
                source_idx = i
            if cluster["voice_cluster_id"] == req.target_cluster_id:
                target_cluster = cluster

        if not source_cluster:
            raise HTTPException(status_code=404, detail=f"Source cluster {req.source_cluster_id} not found")
        if not target_cluster:
            raise HTTPException(status_code=404, detail=f"Target cluster {req.target_cluster_id} not found")

        # Merge segments
        target_cluster["segments"].extend(source_cluster["segments"])
        target_cluster["segment_count"] = len(target_cluster["segments"])
        target_cluster["total_duration"] = sum(s["end"] - s["start"] for s in target_cluster["segments"])

        # Remove source cluster
        clusters.pop(source_idx)

        # Save
        with clusters_path.open("w", encoding="utf-8") as f:
            json.dump(clusters, f, indent=2)

        return {
            "success": True,
            "merged_cluster": {
                "id": target_cluster["voice_cluster_id"],
                "segment_count": target_cluster["segment_count"],
                "total_duration": target_cluster["total_duration"],
            },
            "deleted_cluster_id": req.source_cluster_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception(f"Failed to merge clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TranscribeOnlyRequest(BaseModel):
    """Request to re-run transcription only."""
    asr_provider: Optional[Literal["openai_whisper", "gemini_3"]] = Field(
        None, description="ASR provider"
    )
    run_mode: Literal["queue", "local", "redis"] = Field(
        "local",
        description="Execution mode: 'queue' for Celery background job, 'local' for streaming subprocess",
    )


@router.post("/episodes/{ep_id}/audio/transcribe_only")
async def transcribe_only(ep_id: str, req: TranscribeOnlyRequest):
    """Re-run only the transcription stage without re-doing diarization.

    Useful when you want to try a different ASR provider or fix transcription
    issues without losing existing diarization work.

    Supports two modes:
    - local: Runs as streaming subprocess with real-time logs (default)
    - queue: Queues as Celery task (not yet implemented, falls back to sync)

    Args:
        ep_id: Episode identifier
        req: Transcription options

    Returns:
        StreamingResponse for local mode, or result dict for queue mode
    """
    paths = _get_audio_paths(ep_id)

    if not paths["diarization"].exists():
        raise HTTPException(
            status_code=400,
            detail="Diarization not found. Run full diarization first.",
        )

    audio_path = paths["audio_vocals_enhanced"]
    if not audio_path.exists():
        audio_path = paths["audio_vocals"]
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio files not found")

    asr_provider = req.asr_provider or "openai_whisper"
    # Normalize provider alias for downstream components
    provider = "gemini_3" if asr_provider in ("gemini", "gemini_3") else "openai_whisper"
    run_mode = _normalize_run_mode(req.run_mode or "local")

    if run_mode == "local":
        # Run locally via streaming subprocess for real-time logs
        from apps.api.routers.celery_jobs import _stream_local_subprocess

        # Normalize provider name for CLI
        cli_provider = "gemini" if asr_provider == "gemini_3" else asr_provider

        # Build command for transcribe-only
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "audio_pipeline_run.py"),
            "--ep-id", ep_id,
            "--transcribe-only",
            "--asr-provider", cli_provider,
        ]

        # Options for job tracking
        options = {
            "asr_provider": asr_provider,
            "operation_type": "transcribe_only",
        }

        # Return streaming response for live log updates
        return StreamingResponse(
            _stream_local_subprocess(command, ep_id, "transcribe_only", options, timeout=3600),
            media_type="application/x-ndjson",
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache",
            },
        )

    # Queue mode (Celery) - fall back to sync for now
    try:
        from py_screenalytics.audio.asr_openai import transcribe_audio as transcribe_openai
        from py_screenalytics.audio.asr_gemini import transcribe_audio as transcribe_gemini
        from py_screenalytics.audio.episode_audio_pipeline import _load_config

        # Use YAML-backed config so chunking/model options stay in sync with pipeline
        asr_config = _load_config().asr
        asr_config = asr_config.model_copy(update={"provider": provider})

        LOGGER.info(f"Re-running transcription for {ep_id} with {provider}")

        if provider == "gemini_3":
            segments = transcribe_gemini(
                audio_path,
                paths["asr_raw"],
                asr_config,
                overwrite=True,
            )
        else:
            segments = transcribe_openai(
                audio_path,
                paths["asr_raw"],
                asr_config,
                overwrite=True,
            )

        return {
            "success": True,
            "ep_id": ep_id,
            "provider": provider,
            "segment_count": len(segments),
            "output_path": str(paths["asr_raw"]),
        }

    except Exception as e:
        LOGGER.exception(f"Failed to re-run transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class DiarizeOnlyRequest(BaseModel):
    """Request body for re-running diarization."""
    num_speakers: Optional[int] = Field(
        None,
        description="Force exact speaker count. If None, uses min/max hints.",
    )
    min_speakers: Optional[int] = Field(
        None,
        description="Minimum expected speakers (hint to diarization model).",
    )
    max_speakers: Optional[int] = Field(
        None,
        description="Maximum expected speakers (hint to diarization model).",
    )
    run_mode: Literal["queue", "local", "redis"] = Field(
        "local",
        description="Execution mode: 'queue' for Celery background job, 'local' for streaming subprocess",
    )


@router.post("/episodes/{ep_id}/audio/diarize_only")
async def diarize_only(ep_id: str, req: Optional[DiarizeOnlyRequest] = None):
    """Re-run only the diarization stage without re-doing separation/enhancement.

    Useful for fixing speaker segmentation issues.

    Supports two modes:
    - local: Runs as streaming subprocess with real-time logs (default)
    - queue: Queues as Celery task (not yet implemented, falls back to local)

    Args:
        ep_id: Episode identifier
        req: Optional request body with num_speakers override and run_mode

    Returns:
        StreamingResponse for local mode, or result dict for queue mode
    """
    paths = _get_audio_paths(ep_id)

    audio_path = paths["audio_vocals_enhanced"]
    if not audio_path.exists():
        audio_path = paths["audio_vocals"]
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio files not found")

    # Determine run mode (default to local for real-time feedback)
    run_mode = _normalize_run_mode((req and req.run_mode) or "local")
    num_speakers = (req and req.num_speakers) or None
    min_speakers = (req and req.min_speakers) or None
    max_speakers = (req and req.max_speakers) or None

    if run_mode == "local":
        # Run locally via streaming subprocess for real-time logs
        from apps.api.routers.celery_jobs import _stream_local_subprocess

        # Build command for diarize-only
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "audio_pipeline_run.py"),
            "--ep-id", ep_id,
            "--diarize-only",
        ]
        if num_speakers is not None:
            command.extend(["--num-speakers", str(num_speakers)])
        if min_speakers is not None:
            command.extend(["--min-speakers", str(min_speakers)])
        if max_speakers is not None:
            command.extend(["--max-speakers", str(max_speakers)])

        # Options for job tracking
        options = {
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "operation_type": "diarize_only",
        }

        # Return streaming response for live log updates
        return StreamingResponse(
            _stream_local_subprocess(command, ep_id, "diarize_only", options, timeout=3600),
            media_type="application/x-ndjson",
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache",
            },
        )

    # Queue mode (Celery) - fall back to sync for now
    # TODO: Implement Celery task for diarize_only
    from apps.api.jobs_audio import _write_progress, _write_progress_complete, _write_progress_error

    try:
        _write_progress(ep_id, "diarize", "Starting re-diarization...", 0.0)

        from py_screenalytics.audio.diarization_nemo import run_diarization_nemo, NeMoDiarizationConfig
        from py_screenalytics.audio.episode_audio_pipeline import _load_config

        # Load config from yaml for proper defaults
        pipeline_config = _load_config()
        config = pipeline_config.diarization

        # Build NeMo config from pipeline config
        nemo_config = NeMoDiarizationConfig(
            max_num_speakers=config.max_speakers,
            min_num_speakers=config.min_speakers,
            num_speakers=num_speakers if num_speakers is not None else config.num_speakers,
            overlap_threshold=getattr(config, 'overlap_threshold', 0.5),
        )

        if num_speakers is not None:
            LOGGER.info(f"Re-running diarization for {ep_id} with forced num_speakers={num_speakers}")
            _write_progress(ep_id, "diarize", f"Running diarization (forcing {num_speakers} speakers)...", 0.2)
        else:
            LOGGER.info(f"Re-running diarization for {ep_id} with auto speaker detection")
            _write_progress(ep_id, "diarize", "Running diarization (auto speaker detection)...", 0.2)

        result = run_diarization_nemo(
            audio_path,
            paths["diarization"],
            nemo_config,
            overwrite=True,
        )
        segments = result.segments

        speakers = set(s.speaker for s in segments)

        _write_progress_complete(ep_id, f"Re-diarization complete: {len(speakers)} speakers, {len(segments)} segments")

        return {
            "success": True,
            "ep_id": ep_id,
            "segment_count": len(segments),
            "speaker_count": len(speakers),
            "output_path": str(paths["diarization"]),
        }

    except Exception as e:
        LOGGER.exception(f"Failed to re-run diarization: {e}")
        _write_progress_error(ep_id, "diarize", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/episodes/{ep_id}/audio/voices_only")
async def voices_only(ep_id: str, req: ClusterReclusterRequest) -> dict:
    """Re-run only the voice clustering stage.

    Same as recluster but with a clearer name for the incremental pipeline.

    Args:
        ep_id: Episode identifier
        req: Clustering parameters

    Returns:
        Result summary
    """
    # Delegate to existing recluster endpoint
    return await recluster_voices(ep_id, req)


@router.post("/episodes/{ep_id}/audio/clusters/preview")
async def preview_clustering(ep_id: str, req: ClusterReclusterRequest) -> dict:
    """Preview clustering with new parameters WITHOUT saving.

    Runs clustering in-memory and returns proposed cluster count and
    representative samples. Useful for real-time threshold adjustment.

    Args:
        ep_id: Episode identifier
        req: Clustering parameters to preview

    Returns:
        Preview of clustering results
    """
    paths = _get_audio_paths(ep_id)

    if not paths["diarization"].exists():
        raise HTTPException(status_code=404, detail="Diarization not found")

    audio_path = paths["audio_vocals_enhanced"]
    if not audio_path.exists():
        audio_path = paths["audio_vocals"]
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio files not found")

    try:
        from py_screenalytics.audio.models import DiarizationSegment, VoiceClusteringConfig
        from py_screenalytics.audio.diarization_nemo import extract_speaker_embeddings

        import numpy as np
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        # Load diarization segments
        segments = []
        with paths["diarization"].open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    segments.append(DiarizationSegment(**data))

        if not segments:
            return {"error": "No diarization segments"}

        # Extract embeddings
        config = VoiceClusteringConfig(
            similarity_threshold=req.similarity_threshold,
            min_segments_per_cluster=req.min_segments_per_cluster,
        )

        segment_embeddings = extract_speaker_embeddings(
            audio_path,
            segments,
            config.embedding_model,
        )

        if len(segment_embeddings) < 2:
            return {
                "ep_id": ep_id,
                "similarity_threshold": req.similarity_threshold,
                "proposed_cluster_count": 1,
                "segments_analyzed": len(segment_embeddings),
                "clusters": [],
            }

        # Run clustering in memory
        embeddings = np.array([emb for _, emb in segment_embeddings])

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        # Compute pairwise cosine distances
        distances = pdist(embeddings, metric='cosine')

        # Hierarchical clustering
        Z = linkage(distances, method='average')

        # Cut at threshold
        distance_threshold = 1 - req.similarity_threshold
        cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')

        # Group segments by cluster
        clusters_dict = {}
        for i, (segment, embedding) in enumerate(segment_embeddings):
            cluster_id = int(cluster_labels[i])
            if cluster_id not in clusters_dict:
                clusters_dict[cluster_id] = []
            clusters_dict[cluster_id].append({
                "start": segment.start,
                "end": segment.end,
                "duration": segment.end - segment.start,
            })

        # Filter by min segments and build preview
        preview_clusters = []
        for cluster_id, segs in sorted(clusters_dict.items()):
            if len(segs) < req.min_segments_per_cluster:
                continue

            total_duration = sum(s["duration"] for s in segs)

            # Get representative samples (first 3 segments)
            samples = sorted(segs, key=lambda x: x["start"])[:3]

            preview_clusters.append({
                "cluster_id": f"VC_{len(preview_clusters) + 1:02d}",
                "segment_count": len(segs),
                "total_duration": round(total_duration, 1),
                "samples": samples,
            })

        # Sort by total duration
        preview_clusters.sort(key=lambda x: x["total_duration"], reverse=True)

        # Compare with current
        current_count = 0
        if paths["voice_clusters"].exists():
            with paths["voice_clusters"].open("r", encoding="utf-8") as f:
                current_clusters = json.load(f)
                current_count = len(current_clusters)

        return {
            "ep_id": ep_id,
            "similarity_threshold": req.similarity_threshold,
            "min_segments_per_cluster": req.min_segments_per_cluster,
            "segments_analyzed": len(segment_embeddings),
            "proposed_cluster_count": len(preview_clusters),
            "current_cluster_count": current_count,
            "change": len(preview_clusters) - current_count,
            "clusters": preview_clusters,
        }

    except Exception as e:
        LOGGER.exception(f"Failed to preview clustering: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/episodes/{ep_id}/audio/clusters/recluster")
async def recluster_voices(ep_id: str, req: ClusterReclusterRequest):
    """Re-run voice clustering with new parameters.

    Supports two modes:
    - local: Runs as streaming subprocess with real-time logs (default)
    - queue: Runs synchronously (Celery not yet implemented for this operation)

    Args:
        ep_id: Episode identifier
        req: Parameters for reclustering

    Returns:
        StreamingResponse for local mode, or result dict for queue mode
    """
    paths = _get_audio_paths(ep_id)

    if not paths["diarization"].exists():
        raise HTTPException(status_code=404, detail="Diarization not found")

    run_mode = _normalize_run_mode(req.run_mode or "local")

    if run_mode == "local":
        # Run locally via streaming subprocess for real-time logs
        from apps.api.routers.celery_jobs import _stream_local_subprocess

        # Build command for voices-only
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "audio_pipeline_run.py"),
            "--ep-id", ep_id,
            "--voices-only",
            "--similarity-threshold", str(req.similarity_threshold),
        ]

        # Options for job tracking
        options = {
            "similarity_threshold": req.similarity_threshold,
            "min_segments_per_cluster": req.min_segments_per_cluster,
            "operation_type": "voices_only",
        }

        # Return streaming response for live log updates
        return StreamingResponse(
            _stream_local_subprocess(command, ep_id, "voices_only", options, timeout=1800),
            media_type="application/x-ndjson",
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache",
            },
        )

    # Queue mode - fall back to sync execution
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from py_screenalytics.audio.voice_clusters import cluster_episode_voices
        from py_screenalytics.audio.voice_bank import match_voice_clusters_to_bank
        from py_screenalytics.audio.diarization_nemo import load_diarization_manifest
        from py_screenalytics.audio.models import VoiceClusteringConfig, VoiceBankConfig

        # Get show_id
        show_id = ep_id.rsplit("-", 1)[0] if "-" in ep_id else ep_id

        # Load diarization
        diarization_segments = load_diarization_manifest(paths["diarization"])

        # Get audio path
        audio_path = paths["audio_vocals_enhanced"]
        if not audio_path.exists():
            audio_path = paths["audio_vocals"]
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio files not found")

        # Create config with new parameters
        clustering_config = VoiceClusteringConfig(
            similarity_threshold=req.similarity_threshold,
            min_segments_per_cluster=req.min_segments_per_cluster,
        )

        # Re-run clustering
        voice_clusters = cluster_episode_voices(
            audio_path,
            diarization_segments,
            paths["voice_clusters"],
            clustering_config,
            overwrite=True,
        )

        # Re-run voice mapping
        voice_bank_config = VoiceBankConfig()
        match_voice_clusters_to_bank(
            show_id,
            voice_clusters,
            paths["voice_mapping"],
            voice_bank_config,
            req.similarity_threshold,
            overwrite=True,
        )

        return {
            "success": True,
            "cluster_count": len(voice_clusters),
            "parameters": {
                "similarity_threshold": req.similarity_threshold,
                "min_segments_per_cluster": req.min_segments_per_cluster,
            },
            "clusters": [
                {
                    "id": c.voice_cluster_id,
                    "segment_count": c.segment_count,
                    "total_duration": c.total_duration,
                }
                for c in voice_clusters
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception(f"Failed to recluster: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/episodes/{ep_id}/audio/clusters/suggest_merges")
async def suggest_cluster_merges(
    ep_id: str,
    min_similarity: float = Query(0.85, ge=0.5, le=1.0, description="Minimum similarity for merge suggestion"),
) -> dict:
    """Get suggested cluster merges based on centroid similarity.

    Analyzes pairwise cosine similarity between all voice cluster centroids
    and returns pairs that exceed the min_similarity threshold.

    Args:
        ep_id: Episode identifier
        min_similarity: Minimum similarity threshold (default 0.85)

    Returns:
        List of suggested merges with similarity scores
    """
    paths = _get_audio_paths(ep_id)
    clusters_path = paths["voice_clusters"]
    mapping_path = paths["voice_mapping"]

    if not clusters_path.exists():
        raise HTTPException(status_code=404, detail="Voice clusters not found")

    try:
        import numpy as np

        with clusters_path.open("r", encoding="utf-8") as f:
            clusters = json.load(f)

        # Load mapping for display names
        mapping_lookup = {}
        if mapping_path.exists():
            with mapping_path.open("r", encoding="utf-8") as f:
                mapping_data = json.load(f)
            mapping_lookup = {m.get("voice_cluster_id", ""): m for m in mapping_data}

        # Extract clusters with centroids
        clusters_with_centroids = []
        for c in clusters:
            if c.get("centroid"):
                centroid = np.array(c["centroid"])
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                clusters_with_centroids.append({
                    "voice_cluster_id": c["voice_cluster_id"],
                    "centroid": centroid,
                    "segment_count": c.get("segment_count", 0),
                    "total_duration": c.get("total_duration", 0),
                })

        # Compute pairwise similarities
        suggestions = []
        n = len(clusters_with_centroids)

        for i in range(n):
            for j in range(i + 1, n):
                c1 = clusters_with_centroids[i]
                c2 = clusters_with_centroids[j]

                similarity = float(np.dot(c1["centroid"], c2["centroid"]))

                if similarity >= min_similarity:
                    # Get display names from mapping
                    m1 = mapping_lookup.get(c1["voice_cluster_id"], {})
                    m2 = mapping_lookup.get(c2["voice_cluster_id"], {})

                    suggestions.append({
                        "cluster_a": {
                            "voice_cluster_id": c1["voice_cluster_id"],
                            "display_name": m1.get("speaker_display_name", c1["voice_cluster_id"]),
                            "segment_count": c1["segment_count"],
                            "total_duration": round(c1["total_duration"], 1),
                            "is_labeled": m1.get("similarity") is not None,
                        },
                        "cluster_b": {
                            "voice_cluster_id": c2["voice_cluster_id"],
                            "display_name": m2.get("speaker_display_name", c2["voice_cluster_id"]),
                            "segment_count": c2["segment_count"],
                            "total_duration": round(c2["total_duration"], 1),
                            "is_labeled": m2.get("similarity") is not None,
                        },
                        "similarity": round(similarity, 3),
                    })

        # Sort by similarity descending
        suggestions.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "ep_id": ep_id,
            "min_similarity": min_similarity,
            "suggestions": suggestions,
            "total_clusters": len(clusters),
            "clusters_with_centroids": n,
        }

    except Exception as e:
        LOGGER.exception(f"Failed to compute merge suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/episodes/{ep_id}/audio/clusters/bulk_merge")
async def bulk_merge_clusters(ep_id: str, merges: List[Dict]) -> dict:
    """Merge multiple cluster pairs at once.

    Args:
        ep_id: Episode identifier
        merges: List of merge operations [{source_cluster_id, target_cluster_id}, ...]

    Returns:
        Success status and summary
    """
    paths = _get_audio_paths(ep_id)
    clusters_path = paths["voice_clusters"]

    if not clusters_path.exists():
        raise HTTPException(status_code=404, detail="Voice clusters not found")

    try:
        with clusters_path.open("r", encoding="utf-8") as f:
            clusters = json.load(f)

        merged_count = 0
        errors = []

        for merge in merges:
            source_id = merge.get("source_cluster_id")
            target_id = merge.get("target_cluster_id")

            if not source_id or not target_id:
                errors.append(f"Invalid merge spec: {merge}")
                continue

            # Find clusters
            source_cluster = None
            target_cluster = None
            source_idx = -1

            for i, c in enumerate(clusters):
                if c["voice_cluster_id"] == source_id:
                    source_cluster = c
                    source_idx = i
                if c["voice_cluster_id"] == target_id:
                    target_cluster = c

            if not source_cluster:
                errors.append(f"Source cluster {source_id} not found")
                continue
            if not target_cluster:
                errors.append(f"Target cluster {target_id} not found")
                continue

            # Merge segments
            target_cluster["segments"].extend(source_cluster["segments"])
            target_cluster["segment_count"] = len(target_cluster["segments"])
            target_cluster["total_duration"] = sum(
                s["end"] - s["start"] for s in target_cluster["segments"]
            )

            # Mark source for deletion (we'll remove after loop)
            source_cluster["_deleted"] = True
            merged_count += 1

        # Remove deleted clusters
        clusters = [c for c in clusters if not c.get("_deleted")]

        # Save
        with clusters_path.open("w", encoding="utf-8") as f:
            json.dump(clusters, f, indent=2)

        return {
            "success": True,
            "merged_count": merged_count,
            "remaining_clusters": len(clusters),
            "errors": errors if errors else None,
        }

    except Exception as e:
        LOGGER.exception(f"Failed to bulk merge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/episodes/{ep_id}/audio/segments/assign_cast")
async def assign_segment_to_cast(ep_id: str, req: SegmentAssignCastRequest) -> dict:
    """Move a segment to a new cluster and assign it to a cast member.

    This creates a new voice cluster containing the specified segment,
    removes the segment from the source cluster, and assigns the new
    cluster to the specified cast member.

    Args:
        ep_id: Episode identifier
        req: Request with segment info and cast_id

    Returns:
        Success status and new cluster details
    """
    paths = _get_audio_paths(ep_id)
    clusters_path = paths["voice_clusters"]
    mapping_path = paths["voice_mapping"]

    if not clusters_path.exists():
        raise HTTPException(status_code=404, detail="Voice clusters not found")

    try:
        with clusters_path.open("r", encoding="utf-8") as f:
            clusters = json.load(f)

        # Find source cluster and segment
        source_cluster = None
        segment_to_move = None
        segment_idx = -1

        for cluster in clusters:
            if cluster["voice_cluster_id"] == req.from_cluster_id:
                source_cluster = cluster
                for idx, seg in enumerate(cluster["segments"]):
                    if abs(seg["start"] - req.segment_start) < 0.1 and abs(seg["end"] - req.segment_end) < 0.1:
                        segment_to_move = seg
                        segment_idx = idx
                        break
                break

        if not source_cluster:
            raise HTTPException(status_code=404, detail=f"Source cluster {req.from_cluster_id} not found")
        if segment_to_move is None:
            raise HTTPException(status_code=404, detail="Segment not found in source cluster")

        # Generate new cluster ID
        existing_ids = [c["voice_cluster_id"] for c in clusters]
        new_id_num = 1
        while f"VC_{new_id_num:02d}" in existing_ids:
            new_id_num += 1
        new_cluster_id = f"VC_{new_id_num:02d}"

        # Remove segment from source cluster
        source_cluster["segments"].pop(segment_idx)
        source_cluster["segment_count"] = len(source_cluster["segments"])
        source_cluster["total_duration"] = sum(s["end"] - s["start"] for s in source_cluster["segments"])

        # Create new cluster
        seg_duration = segment_to_move["end"] - segment_to_move["start"]
        new_cluster = {
            "voice_cluster_id": new_cluster_id,
            "segments": [segment_to_move],
            "total_duration": seg_duration,
            "segment_count": 1,
            "centroid": None,  # Will be computed on next recluster
        }
        clusters.append(new_cluster)

        # Save updated clusters
        with clusters_path.open("w", encoding="utf-8") as f:
            json.dump(clusters, f, indent=2)

        # Now assign the new cluster to the cast member
        # Load mapping
        if mapping_path.exists():
            with mapping_path.open("r", encoding="utf-8") as f:
                mapping_data = json.load(f)
        else:
            mapping_data = []

        # Get cast member name
        speaker_id = f"SPK_{req.cast_id.upper()}"
        speaker_display_name = req.cast_id.replace("_", " ").title()
        voice_bank_id = f"voice_{req.cast_id.lower()}"

        # Try to get actual cast member name
        try:
            from apps.api.routers.cast import get_cast_member
            show_id = ep_id.split("-")[0].upper()
            cast_info = await get_cast_member(show_id, req.cast_id)
            if cast_info and cast_info.get("name"):
                speaker_display_name = cast_info["name"]
        except Exception as exc:
            LOGGER.debug("[cast-lookup] Failed to get cast member %s: %s", req.cast_id, exc)

        # Add new mapping entry
        new_mapping = {
            "voice_cluster_id": new_cluster_id,
            "speaker_id": speaker_id,
            "speaker_display_name": speaker_display_name,
            "voice_bank_id": voice_bank_id,
            "similarity": 1.0,  # Manual assignment
        }
        mapping_data.append(new_mapping)

        # Save mapping
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(mapping_data, f, indent=2)

        LOGGER.info(f"Created cluster {new_cluster_id} and assigned to {speaker_display_name}")

        return {
            "success": True,
            "new_cluster_id": new_cluster_id,
            "cast_id": req.cast_id,
            "speaker_display_name": speaker_display_name,
            "segment_moved": {
                "start": req.segment_start,
                "end": req.segment_end,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception(f"Failed to assign segment to cast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Voice Reference Upload Endpoints (Feature #1)
# =============================================================================


def _get_voice_reference_path(show_id: str, cast_id: str) -> Path:
    """Get path for voice reference audio file."""
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
    return data_root / "voice_bank" / show_id.lower() / cast_id.lower() / "reference.wav"


@router.post("/shows/{show_id}/cast/{cast_id}/voice_reference")
async def upload_voice_reference(
    show_id: str,
    cast_id: str,
    file: UploadFile = File(..., description="Audio file (WAV, MP3, M4A)"),
) -> dict:
    """Upload a voice reference audio clip for a cast member.

    This reference can be used by the ASR pipeline to improve speaker
    identification accuracy for known cast members.

    Args:
        show_id: Show identifier
        cast_id: Cast member identifier
        file: Audio file upload

    Returns:
        Success status and file path
    """
    # Validate file type
    allowed_types = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/mp4", "audio/m4a"}
    content_type = file.content_type or ""
    if content_type not in allowed_types and not file.filename.lower().endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {content_type}. Use WAV, MP3, or M4A.",
        )

    try:
        import subprocess
        import tempfile

        # Read uploaded file
        content = await file.read()

        # Save to temp file for conversion
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Output path
        ref_path = _get_voice_reference_path(show_id, cast_id)
        ref_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to standardized WAV format (16kHz mono for embeddings)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", tmp_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(ref_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

        # Get file info
        file_size = ref_path.stat().st_size if ref_path.exists() else 0

        LOGGER.info(f"Uploaded voice reference for {show_id}/{cast_id}: {ref_path}")

        return {
            "success": True,
            "show_id": show_id,
            "cast_id": cast_id,
            "file_path": str(ref_path),
            "file_size_bytes": file_size,
        }

    except subprocess.CalledProcessError as e:
        LOGGER.error(f"FFmpeg conversion failed: {e.stderr.decode() if e.stderr else e}")
        raise HTTPException(status_code=500, detail="Failed to convert audio file")
    except Exception as e:
        LOGGER.exception(f"Failed to upload voice reference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/shows/{show_id}/cast/{cast_id}/voice_reference")
async def get_voice_reference_status(show_id: str, cast_id: str) -> dict:
    """Check if a voice reference exists for a cast member.

    Args:
        show_id: Show identifier
        cast_id: Cast member identifier

    Returns:
        Status and file info if exists
    """
    ref_path = _get_voice_reference_path(show_id, cast_id)

    if ref_path.exists():
        import subprocess

        # Get audio duration using ffprobe
        duration = None
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(ref_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                duration = float(result.stdout.strip())
        except Exception as exc:
            LOGGER.debug("[ffprobe] Failed to get audio duration for %s: %s", ref_path, exc)

        return {
            "exists": True,
            "show_id": show_id,
            "cast_id": cast_id,
            "file_path": str(ref_path),
            "file_size_bytes": ref_path.stat().st_size,
            "duration_seconds": duration,
        }

    return {
        "exists": False,
        "show_id": show_id,
        "cast_id": cast_id,
    }


@router.delete("/shows/{show_id}/cast/{cast_id}/voice_reference")
async def delete_voice_reference(show_id: str, cast_id: str) -> dict:
    """Delete a voice reference for a cast member.

    Args:
        show_id: Show identifier
        cast_id: Cast member identifier

    Returns:
        Success status
    """
    ref_path = _get_voice_reference_path(show_id, cast_id)

    if not ref_path.exists():
        raise HTTPException(status_code=404, detail="Voice reference not found")

    try:
        ref_path.unlink()
        LOGGER.info(f"Deleted voice reference for {show_id}/{cast_id}")

        return {
            "success": True,
            "show_id": show_id,
            "cast_id": cast_id,
            "deleted": True,
        }

    except Exception as e:
        LOGGER.exception(f"Failed to delete voice reference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/episodes/{ep_id}/audio/waveform")
async def get_audio_waveform(
    ep_id: str,
    start: float = Query(0.0, description="Start time in seconds"),
    end: float = Query(60.0, description="End time in seconds"),
    resolution: int = Query(1000, description="Number of data points"),
) -> dict:
    """Get waveform data for audio visualization.

    Returns amplitude data sampled at the specified resolution for
    rendering interactive waveform displays with speaker lanes.

    Args:
        ep_id: Episode identifier
        start: Start time in seconds
        end: End time in seconds (max 60s window)
        resolution: Number of amplitude samples to return

    Returns:
        Waveform data with speaker annotations
    """
    paths = _get_audio_paths(ep_id)

    audio_path = paths["audio_vocals_enhanced"]
    if not audio_path.exists():
        audio_path = paths["audio_vocals"]
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio files not found")

    try:
        import numpy as np
        import soundfile as sf

        # Limit window size to prevent memory issues
        max_window = 120  # 2 minutes max
        end = min(end, start + max_window)

        # Load audio segment
        info = sf.info(audio_path)
        sample_rate = info.samplerate
        total_duration = info.duration

        start = max(0, min(start, total_duration))
        end = min(end, total_duration)

        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)

        audio_data, _ = sf.read(audio_path, start=start_sample, stop=end_sample)

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Downsample for visualization
        n_samples = len(audio_data)
        samples_per_point = max(1, n_samples // resolution)

        # Get min/max for each window (creates envelope)
        n_points = min(resolution, n_samples)
        waveform_data = []

        for i in range(n_points):
            chunk_start = i * samples_per_point
            chunk_end = min(chunk_start + samples_per_point, n_samples)
            chunk = audio_data[chunk_start:chunk_end]

            if len(chunk) > 0:
                waveform_data.append({
                    "time": round(start + (i * (end - start) / n_points), 3),
                    "min": round(float(np.min(chunk)), 4),
                    "max": round(float(np.max(chunk)), 4),
                    "rms": round(float(np.sqrt(np.mean(chunk ** 2))), 4),
                })

        # Load speaker annotations
        speaker_regions = []
        mapping_lookup = {}

        # Load voice mapping for display names
        if paths["voice_mapping"].exists():
            with paths["voice_mapping"].open("r", encoding="utf-8") as f:
                for m in json.load(f):
                    mapping_lookup[m.get("voice_cluster_id", "")] = m.get("speaker_display_name", "")

        # Load voice clusters
        if paths["voice_clusters"].exists():
            with paths["voice_clusters"].open("r", encoding="utf-8") as f:
                clusters = json.load(f)

            for cluster in clusters:
                cluster_id = cluster.get("voice_cluster_id", "")
                display_name = mapping_lookup.get(cluster_id, cluster_id)

                for seg in cluster.get("segments", []):
                    seg_start = seg.get("start", 0)
                    seg_end = seg.get("end", 0)

                    # Check if segment overlaps with requested window
                    if seg_start < end and seg_end > start:
                        speaker_regions.append({
                            "start": max(seg_start, start),
                            "end": min(seg_end, end),
                            "cluster_id": cluster_id,
                            "speaker": display_name,
                        })

        # Sort regions by start time
        speaker_regions.sort(key=lambda x: x["start"])

        return {
            "ep_id": ep_id,
            "start": start,
            "end": end,
            "duration": round(end - start, 2),
            "total_duration": round(total_duration, 2),
            "sample_rate": sample_rate,
            "resolution": len(waveform_data),
            "waveform": waveform_data,
            "speaker_regions": speaker_regions,
        }

    except Exception as e:
        LOGGER.exception(f"Failed to get waveform: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/episodes/{ep_id}/audio/diarization/comparison")
async def get_diarization_comparison(ep_id: str) -> dict:
    """DEPRECATED: Get A/B comparison of pyannote vs GPT-4o diarization.

    This endpoint is deprecated as the system now uses NeMo MSDD exclusively
    for diarization. The endpoint remains for backward compatibility with
    episodes processed before the migration.

    For new episodes, use the main diarization endpoint instead.

    Args:
        ep_id: Episode identifier

    Returns:
        Comparison data for UI visualization (legacy episodes only)
    """
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
    manifests_dir = data_root / "manifests" / ep_id

    comparison_path = manifests_dir / "audio_diarization_comparison.json"
    pyannote_path = manifests_dir / "audio_diarization_pyannote.jsonl"
    gpt4o_path = manifests_dir / "audio_diarization_gpt4o.jsonl"
    transcript_path = manifests_dir / "episode_transcript.jsonl"

    result = {
        "ep_id": ep_id,
        "has_comparison": comparison_path.exists(),
        "has_pyannote": pyannote_path.exists(),
        "has_gpt4o": gpt4o_path.exists(),
    }

    # Load comparison summary if available
    if comparison_path.exists():
        try:
            with comparison_path.open("r", encoding="utf-8") as f:
                comparison_data = json.load(f)
            # Optionally augment with canonical text + mixed-speaker flags if transcript exists
            if transcript_path.exists():
                try:
                    from py_screenalytics.audio.diarization_comparison import augment_diarization_comparison

                    comparison_data = augment_diarization_comparison(comparison_path, transcript_path) or comparison_data
                except Exception as exc:
                    LOGGER.warning("Could not augment diarization comparison: %s", exc)
            result["summary"] = comparison_data
        except Exception as e:
            LOGGER.warning(f"Failed to load comparison: {e}")

    # Load pyannote segments for timeline
    if pyannote_path.exists():
        try:
            segments = []
            with pyannote_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        seg = json.loads(line)
                        segments.append({
                            "start": seg.get("start", 0),
                            "end": seg.get("end", 0),
                            "speaker": seg.get("speaker", ""),
                            "provider": "pyannote",
                        })
            # First try to inject canonical_text from comparison if available
            comp_segments = (result.get("summary") or {}).get("segments", {}).get("pyannote", [])
            if comp_segments:
                for seg in segments:
                    for cseg in comp_segments:
                        if abs(seg["start"] - cseg.get("start", 0)) < 1e-3 and abs(seg["end"] - cseg.get("end", 0)) < 1e-3:
                            if cseg.get("canonical_text"):
                                seg["canonical_text"] = cseg.get("canonical_text")
                            break
            # Fallback: compute canonical_text directly from transcript for segments without it
            if transcript_path.exists():
                try:
                    from py_screenalytics.audio.diarization_comparison import get_canonical_text_for_segment
                    for seg in segments:
                        if not seg.get("canonical_text"):
                            text = get_canonical_text_for_segment(
                                transcript_path,
                                seg["start"],
                                seg["end"],
                                min_words=1,  # Allow single-word segments
                            )
                            if text:
                                seg["canonical_text"] = text
                except Exception as exc:
                    LOGGER.warning(f"Could not compute canonical text for pyannote segments: {exc}")
            result["pyannote_segments"] = segments
        except Exception as e:
            LOGGER.warning(f"Failed to load pyannote segments: {e}")

    # Load GPT-4o segments for timeline
    if gpt4o_path.exists():
        try:
            segments = []
            with gpt4o_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        seg = json.loads(line)
                        seg_data = {
                            "start": seg.get("start", 0),
                            "end": seg.get("end", 0),
                            "speaker": seg.get("speaker", ""),
                            "text": seg.get("text", "")[:100],  # First 100 chars
                            "provider": "gpt4o",
                        }
                        segments.append(seg_data)
            for idx, seg in enumerate(segments):
                seg.setdefault("segment_id", f"gpt4o_{idx+1:04d}")
            # Inject canonical_text/mixed_speaker from comparison if available
            comp_segments = (result.get("summary") or {}).get("segments", {}).get("gpt4o", [])
            if comp_segments:
                for seg in segments:
                    for cseg in comp_segments:
                        if abs(seg["start"] - cseg.get("start", 0)) < 1e-3 and abs(seg["end"] - cseg.get("end", 0)) < 1e-3:
                            if cseg.get("canonical_text"):
                                seg["canonical_text"] = cseg.get("canonical_text")
                            if cseg.get("raw_text"):
                                seg["raw_text"] = cseg.get("raw_text")
                            if cseg.get("mixed_speaker") is not None:
                                seg["mixed_speaker"] = cseg.get("mixed_speaker")
                            if cseg.get("speakers") is not None:
                                seg["speakers"] = cseg.get("speakers")
                            break
            result["gpt4o_segments"] = segments
        except Exception as e:
            LOGGER.warning(f"Failed to load gpt4o segments: {e}")

    return result


@router.get("/episodes/{ep_id}/audio/segments/quality")
async def get_segment_quality_scores(ep_id: str) -> dict:
    """Get quality scores for each audio segment.

    Computes SNR (signal-to-noise ratio), clarity, and overlap detection
    for each diarization segment. Useful for identifying segments that
    need manual review.

    Args:
        ep_id: Episode identifier

    Returns:
        Quality scores per segment
    """
    paths = _get_audio_paths(ep_id)

    if not paths["diarization"].exists():
        raise HTTPException(status_code=404, detail="Diarization not found")

    audio_path = paths["audio_vocals_enhanced"]
    if not audio_path.exists():
        audio_path = paths["audio_vocals"]
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio files not found")

    try:
        import numpy as np
        import soundfile as sf

        # Load audio
        audio_data, sample_rate = sf.read(audio_path)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Load diarization segments
        segments = []
        with paths["diarization"].open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    segments.append(json.loads(line))

        # Load ASR for overlap detection
        asr_segments = []
        if paths["asr_raw"].exists():
            with paths["asr_raw"].open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        asr_segments.append(json.loads(line))

        quality_scores = []

        for seg in segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            speaker = seg.get("speaker", "")

            # Extract segment audio
            start_sample = int(seg_start * sample_rate)
            end_sample = int(seg_end * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]

            if len(segment_audio) < sample_rate * 0.1:
                # Skip very short segments
                continue

            # Compute RMS (as proxy for signal level)
            rms = np.sqrt(np.mean(segment_audio ** 2))

            # Compute SNR estimate (ratio of RMS to noise floor)
            # Noise floor estimated from lowest 10% of samples
            sorted_abs = np.sort(np.abs(segment_audio))
            noise_floor = np.mean(sorted_abs[:len(sorted_abs) // 10]) + 1e-10
            snr_estimate = 20 * np.log10(rms / noise_floor) if noise_floor > 0 else 0

            # Compute zero-crossing rate (high = noisy)
            zero_crossings = np.sum(np.abs(np.diff(np.sign(segment_audio)))) / (2 * len(segment_audio))

            # Check for overlapping speech (multiple ASR segments in this time range)
            overlap_count = 0
            for asr in asr_segments:
                asr_start = asr.get("start", 0)
                asr_end = asr.get("end", 0)
                # Check if ASR segment overlaps with diarization segment
                if asr_start < seg_end and asr_end > seg_start:
                    overlap_count += 1

            has_overlap = bool(overlap_count > 1)

            # Compute clipping detection
            clip_threshold = 0.95
            clipping_ratio = np.sum(np.abs(segment_audio) > clip_threshold) / len(segment_audio)
            has_clipping = bool(clipping_ratio > 0.01)

            # Determine quality status
            if snr_estimate < 10 or has_clipping:
                status = "poor"
                badge = "🔴"
            elif snr_estimate < 15 or has_overlap:
                status = "fair"
                badge = "⚠️"
            else:
                status = "good"
                badge = "✅"

            quality_scores.append({
                "start": round(float(seg_start), 2),
                "end": round(float(seg_end), 2),
                "speaker": speaker,
                "duration": round(float(seg_end - seg_start), 2),
                "snr_db": round(float(snr_estimate), 1),
                "zero_crossing_rate": round(float(zero_crossings), 4),
                "has_overlap": has_overlap,
                "overlap_count": int(overlap_count),
                "has_clipping": has_clipping,
                "clipping_ratio": round(float(clipping_ratio), 4),
                "status": status,
                "badge": badge,
            })

        # Summary stats
        poor_count = sum(1 for q in quality_scores if q["status"] == "poor")
        fair_count = sum(1 for q in quality_scores if q["status"] == "fair")
        good_count = sum(1 for q in quality_scores if q["status"] == "good")

        return {
            "ep_id": ep_id,
            "segment_count": len(quality_scores),
            "summary": {
                "good": good_count,
                "fair": fair_count,
                "poor": poor_count,
                "poor_pct": round(poor_count / len(quality_scores) * 100, 1) if quality_scores else 0,
            },
            "segments": quality_scores,
        }

    except Exception as e:
        LOGGER.exception(f"Failed to compute segment quality: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Segment Archive Endpoints
# =============================================================================


class ArchiveSegmentRequest(BaseModel):
    """Request to archive a transcript segment."""
    start: float = Field(..., description="Segment start time")
    end: float = Field(..., description="Segment end time")
    text: str = Field(..., description="Segment text")
    voice_cluster_id: Optional[str] = Field(None, description="Voice cluster the segment belongs to")
    reason: Optional[str] = Field(None, description="Reason for archiving")


@router.post("/episodes/{ep_id}/audio/segments/archive")
async def archive_transcript_segment(ep_id: str, body: ArchiveSegmentRequest) -> dict:
    """Archive a transcript segment so it won't be used in voicebank.

    Archived segments are stored separately and excluded from cast voice profiles.
    They can be restored later if needed.

    Args:
        ep_id: Episode identifier
        body: Segment to archive

    Returns:
        Success status
    """
    paths = _get_audio_paths(ep_id)
    archive_path = paths["archived_segments"]

    try:
        # Load existing archived segments
        archived = []
        if archive_path.exists():
            with archive_path.open("r", encoding="utf-8") as f:
                archived = json.load(f)

        # Check if already archived (by time range)
        already_archived = any(
            abs(s.get("start", 0) - body.start) < 0.1 and abs(s.get("end", 0) - body.end) < 0.1
            for s in archived
        )
        if already_archived:
            return {"success": True, "message": "Segment already archived", "archived_count": len(archived)}

        # Add to archive
        from datetime import datetime
        archived.append({
            "start": body.start,
            "end": body.end,
            "text": body.text,
            "voice_cluster_id": body.voice_cluster_id,
            "reason": body.reason,
            "archived_at": datetime.now().isoformat(),
        })

        # Save
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with archive_path.open("w", encoding="utf-8") as f:
            json.dump(archived, f, indent=2)

        LOGGER.info(f"Archived segment [{body.start:.1f}-{body.end:.1f}] for {ep_id}")

        return {
            "success": True,
            "message": f"Archived segment [{body.start:.1f}s - {body.end:.1f}s]",
            "archived_count": len(archived),
        }

    except Exception as e:
        LOGGER.exception(f"Failed to archive segment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/episodes/{ep_id}/audio/segments/archived")
async def list_archived_segments(ep_id: str) -> dict:
    """List all archived transcript segments for an episode.

    Args:
        ep_id: Episode identifier

    Returns:
        List of archived segments
    """
    paths = _get_audio_paths(ep_id)
    archive_path = paths["archived_segments"]

    archived = []
    if archive_path.exists():
        try:
            with archive_path.open("r", encoding="utf-8") as f:
                archived = json.load(f)
        except Exception as e:
            LOGGER.warning(f"Failed to load archived segments: {e}")

    return {
        "ep_id": ep_id,
        "archived_count": len(archived),
        "segments": archived,
    }


@router.post("/episodes/{ep_id}/audio/segments/restore")
async def restore_archived_segment(ep_id: str, body: ArchiveSegmentRequest) -> dict:
    """Restore an archived segment (remove from archive).

    Args:
        ep_id: Episode identifier
        body: Segment to restore (matched by start/end time)

    Returns:
        Success status
    """
    paths = _get_audio_paths(ep_id)
    archive_path = paths["archived_segments"]

    if not archive_path.exists():
        return {"success": False, "message": "No archived segments found"}

    try:
        with archive_path.open("r", encoding="utf-8") as f:
            archived = json.load(f)

        # Find and remove the segment
        original_count = len(archived)
        archived = [
            s for s in archived
            if not (abs(s.get("start", 0) - body.start) < 0.1 and abs(s.get("end", 0) - body.end) < 0.1)
        ]

        if len(archived) == original_count:
            return {"success": False, "message": "Segment not found in archive"}

        # Save updated archive
        with archive_path.open("w", encoding="utf-8") as f:
            json.dump(archived, f, indent=2)

        LOGGER.info(f"Restored segment [{body.start:.1f}-{body.end:.1f}] for {ep_id}")

        return {
            "success": True,
            "message": f"Restored segment [{body.start:.1f}s - {body.end:.1f}s]",
            "archived_count": len(archived),
        }

    except Exception as e:
        LOGGER.exception(f"Failed to restore segment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/episodes/{ep_id}/audio/clusters/similarity_matrix")
async def get_cluster_similarity_matrix(ep_id: str) -> dict:
    """Get pairwise similarity matrix between all voice clusters.

    Returns a matrix of cosine similarities between cluster centroids,
    useful for visualizing which voices are easily confused.

    Args:
        ep_id: Episode identifier

    Returns:
        Similarity matrix data for heatmap visualization
    """
    paths = _get_audio_paths(ep_id)
    clusters_path = paths["voice_clusters"]
    mapping_path = paths["voice_mapping"]

    if not clusters_path.exists():
        raise HTTPException(status_code=404, detail="Voice clusters not found")

    try:
        import numpy as np

        with clusters_path.open("r", encoding="utf-8") as f:
            clusters = json.load(f)

        # Load mapping for display names
        mapping_lookup = {}
        if mapping_path.exists():
            with mapping_path.open("r", encoding="utf-8") as f:
                mapping_data = json.load(f)
            mapping_lookup = {m.get("voice_cluster_id", ""): m for m in mapping_data}

        # Extract clusters with centroids
        cluster_info = []
        centroids = []

        for c in clusters:
            if c.get("centroid"):
                centroid = np.array(c["centroid"])
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                    centroids.append(centroid)

                    m = mapping_lookup.get(c["voice_cluster_id"], {})
                    cluster_info.append({
                        "voice_cluster_id": c["voice_cluster_id"],
                        "display_name": m.get("speaker_display_name", c["voice_cluster_id"]),
                        "is_labeled": m.get("similarity") is not None,
                        "segment_count": c.get("segment_count", 0),
                        "total_duration": round(c.get("total_duration", 0), 1),
                    })

        if len(centroids) < 2:
            return {
                "ep_id": ep_id,
                "cluster_count": len(cluster_info),
                "matrix": [],
                "labels": [],
                "message": "Need at least 2 clusters with centroids for similarity matrix",
            }

        # Compute pairwise similarity matrix
        n = len(centroids)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = float(np.dot(centroids[i], centroids[j]))

        # Format for JSON (list of lists)
        similarity_matrix = matrix.tolist()

        # Find most confusable pairs (for highlighting)
        confusable_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = matrix[i, j]
                if sim >= 0.7:
                    confusable_pairs.append({
                        "cluster_a": cluster_info[i]["voice_cluster_id"],
                        "cluster_b": cluster_info[j]["voice_cluster_id"],
                        "similarity": round(sim, 3),
                    })

        confusable_pairs.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "ep_id": ep_id,
            "cluster_count": n,
            "clusters": cluster_info,
            "labels": [c["display_name"] for c in cluster_info],
            "matrix": similarity_matrix,
            "confusable_pairs": confusable_pairs[:10],  # Top 10
        }

    except Exception as e:
        LOGGER.exception(f"Failed to compute similarity matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/shows/{show_id}/voice_references")
async def list_voice_references(show_id: str) -> dict:
    """List all voice references for a show.

    Args:
        show_id: Show identifier

    Returns:
        List of cast members with voice references
    """
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
    voice_bank_dir = data_root / "voice_bank" / show_id.lower()

    references = []

    if voice_bank_dir.exists():
        for cast_dir in voice_bank_dir.iterdir():
            if cast_dir.is_dir():
                ref_path = cast_dir / "reference.wav"
                if ref_path.exists():
                    references.append({
                        "cast_id": cast_dir.name,
                        "file_path": str(ref_path),
                        "file_size_bytes": ref_path.stat().st_size,
                    })

    return {
        "show_id": show_id,
        "references": references,
        "count": len(references),
    }


@router.get("/shows/{show_id}/voice_analytics")
async def get_voice_analytics(show_id: str) -> dict:
    """Get cross-episode voice tracking analytics for a show.

    Aggregates speaking time per cast member across all episodes,
    showing patterns like total speaking time, episode appearances,
    and first/last appearances.

    Args:
        show_id: Show identifier

    Returns:
        Voice analytics data for dashboard visualization
    """
    data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
    manifests_dir = data_root / "manifests"

    # Find all episode directories for this show
    show_prefix = show_id.lower()
    episode_dirs = []

    if manifests_dir.exists():
        for ep_dir in manifests_dir.iterdir():
            if ep_dir.is_dir() and ep_dir.name.lower().startswith(show_prefix):
                episode_dirs.append(ep_dir)

    if not episode_dirs:
        return {
            "show_id": show_id,
            "episode_count": 0,
            "message": "No episodes found for this show",
        }

    # Sort episodes by name (which gives chronological order for format show-sXXeYY)
    episode_dirs.sort(key=lambda d: d.name)

    # Aggregate data
    cast_stats = {}  # cast_id -> stats
    episode_summaries = []

    for ep_dir in episode_dirs:
        ep_id = ep_dir.name
        mapping_path = ep_dir / "audio_voice_mapping.json"
        clusters_path = ep_dir / "audio_voice_clusters.json"

        if not mapping_path.exists() or not clusters_path.exists():
            continue

        try:
            with mapping_path.open("r", encoding="utf-8") as f:
                mappings = json.load(f)

            with clusters_path.open("r", encoding="utf-8") as f:
                clusters = json.load(f)

            # Build cluster duration lookup
            cluster_durations = {}
            for c in clusters:
                cluster_durations[c["voice_cluster_id"]] = c.get("total_duration", 0)

            episode_cast = {}

            for m in mappings:
                cluster_id = m.get("voice_cluster_id", "")
                speaker_name = m.get("speaker_display_name", "Unknown")
                duration = cluster_durations.get(cluster_id, 0)

                # Skip unlabeled/unknown
                if speaker_name.startswith("Unlabeled") or speaker_name.lower() == "unknown":
                    continue

                # Use speaker name as key (normalized)
                cast_key = speaker_name.lower().replace(" ", "_")

                # Update cast stats
                if cast_key not in cast_stats:
                    cast_stats[cast_key] = {
                        "name": speaker_name,
                        "total_speaking_time": 0,
                        "episode_count": 0,
                        "episodes": [],
                        "first_appearance": ep_id,
                        "last_appearance": ep_id,
                    }

                cast_stats[cast_key]["total_speaking_time"] += duration
                cast_stats[cast_key]["last_appearance"] = ep_id

                if ep_id not in cast_stats[cast_key]["episodes"]:
                    cast_stats[cast_key]["episodes"].append(ep_id)
                    cast_stats[cast_key]["episode_count"] += 1

                episode_cast[cast_key] = {
                    "name": speaker_name,
                    "speaking_time": duration,
                }

            if episode_cast:
                episode_summaries.append({
                    "ep_id": ep_id,
                    "cast_speaking_times": episode_cast,
                })

        except Exception as e:
            LOGGER.warning(f"Failed to process {ep_id}: {e}")
            continue

    # Sort cast by total speaking time
    sorted_cast = sorted(
        cast_stats.values(),
        key=lambda x: x["total_speaking_time"],
        reverse=True,
    )

    # Build chart data (speaking time per cast per episode)
    chart_data = []
    for ep_summary in episode_summaries:
        ep_id = ep_summary["ep_id"]
        # Extract season/episode for display
        ep_label = ep_id.split("-")[-1] if "-" in ep_id else ep_id

        for cast_key, cast_data in ep_summary["cast_speaking_times"].items():
            chart_data.append({
                "episode": ep_label,
                "ep_id": ep_id,
                "cast_member": cast_data["name"],
                "speaking_time": round(cast_data["speaking_time"], 1),
            })

    return {
        "show_id": show_id,
        "episode_count": len(episode_summaries),
        "cast_count": len(cast_stats),
        "cast_stats": sorted_cast,
        "episode_summaries": episode_summaries,
        "chart_data": chart_data,
    }


# =============================================================================
# Voiceprint Identification Refresh Endpoint
# =============================================================================


class VoiceprintRefreshRequest(BaseModel):
    """Request to refresh voiceprint identification for an episode."""

    show_id: Optional[str] = Field(
        None,
        description="Show identifier. Auto-detected from ep_id if not provided.",
    )
    overwrite_voiceprints: bool = Field(
        False,
        description="Force recreation of voiceprints even if they exist.",
    )
    ident_threshold: int = Field(
        60,
        ge=0,
        le=100,
        description="Confidence threshold for identification matching (0-100).",
    )
    run_mode: Literal["queue", "local"] = Field(
        "local",
        description="Execution mode: 'queue' for Celery background job, 'local' for streaming subprocess.",
    )


class VoiceprintRefreshResponse(BaseModel):
    """Response from voiceprint refresh job."""

    success: bool
    ep_id: str
    job_id: Optional[str] = None
    status: Optional[str] = None
    message: Optional[str] = None
    voiceprints_created: Optional[int] = None
    voiceprints_skipped: Optional[int] = None
    segments_processed: Optional[int] = None
    review_queue_count: Optional[int] = None


@router.post("/episodes/{ep_id}/audio/voiceprint_refresh", response_model=VoiceprintRefreshResponse)
async def refresh_voiceprint_identification(
    ep_id: str,
    req: Optional[VoiceprintRefreshRequest] = None,
) -> VoiceprintRefreshResponse:
    """Refresh voiceprint identification for an episode.

    This endpoint triggers the voiceprint identification pipeline which:
    1. Selects clean segments from manually assigned clusters for each cast member
    2. Creates voiceprints using Pyannote API (if needed)
    3. Runs identification pass on full episode audio
    4. Regenerates speaker transcript with cast names
    5. Generates review queue for low-confidence segments

    Prerequisites:
    - Episode must be diarized
    - Manual cluster assignments must exist (cast members need 10s+ of clean speech)

    Args:
        ep_id: Episode identifier
        req: Optional request body with configuration options

    Returns:
        VoiceprintRefreshResponse with job status and results
    """
    # Default request values
    show_id = (req and req.show_id) or None
    overwrite_voiceprints = (req and req.overwrite_voiceprints) or False
    ident_threshold = (req and req.ident_threshold) or 60
    run_mode = _normalize_run_mode((req and req.run_mode) or "local")

    # Auto-detect show_id from ep_id if not provided
    if not show_id:
        from py_screenalytics.artifacts import parse_ep_id

        try:
            parsed = parse_ep_id(ep_id)
            show_id = parsed.get("show_id")
        except Exception as exc:
            LOGGER.debug("[parse-ep-id] Failed to parse ep_id '%s': %s", ep_id, exc)

    if not show_id:
        raise HTTPException(
            status_code=400,
            detail="Could not determine show_id from ep_id. Please provide show_id explicitly.",
        )

    # Validate prerequisites
    paths = _get_audio_paths(ep_id)
    if not paths["diarization"].exists():
        raise HTTPException(
            status_code=404,
            detail="Diarization not found. Run audio pipeline first.",
        )

    if run_mode == "local":
        # Run locally with streaming subprocess for real-time logs
        from apps.api.routers.celery_jobs import _stream_local_subprocess

        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "audio_pipeline_run.py"),
            "--ep-id", ep_id,
            "--voiceprint-refresh",
        ]
        if overwrite_voiceprints:
            command.append("--overwrite-voiceprints")
        if ident_threshold != 60:
            command.extend(["--ident-threshold", str(ident_threshold)])

        options = {
            "show_id": show_id,
            "overwrite_voiceprints": overwrite_voiceprints,
            "ident_threshold": ident_threshold,
            "operation_type": "voiceprint_refresh",
        }

        return StreamingResponse(
            _stream_local_subprocess(command, ep_id, "voiceprint_refresh", options, timeout=3600),
            media_type="application/x-ndjson",
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache",
            },
        )

    # Queue mode (Celery)
    from apps.api.jobs_audio import episode_voiceprint_refresh_task

    try:
        task = episode_voiceprint_refresh_task.delay(
            ep_id=ep_id,
            show_id=show_id,
            overwrite_voiceprints=overwrite_voiceprints,
            ident_threshold=ident_threshold,
        )

        return VoiceprintRefreshResponse(
            success=True,
            ep_id=ep_id,
            job_id=task.id,
            status="queued",
            message="Voiceprint refresh job queued",
        )

    except Exception as e:
        LOGGER.exception(f"Failed to queue voiceprint refresh: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PyannoteAI Webhook Endpoints
# =============================================================================


class PyannoteWebhookPayload(BaseModel):
    """Webhook payload from PyannoteAI on job completion.

    Per official docs, PyannoteAI sends:
    - jobId: The job identifier
    - status: succeeded, failed, or canceled
    - output: Diarization results (only on success)

    Webhook retry policy: immediate, 1 min, 5 min
    Expected response: HTTP 200 within 10 seconds
    """
    jobId: str = Field(..., description="PyannoteAI job identifier")
    status: str = Field(..., description="Job status: succeeded, failed, canceled")
    output: Optional[dict] = Field(None, description="Diarization output (on success)")


class PyannoteWebhookResponse(BaseModel):
    """Response to PyannoteAI webhook."""
    received: bool = True
    status: str
    segments_saved: Optional[int] = None
    error: Optional[str] = None


# In-memory registry for webhook job correlation
# In production, this should use Redis or a database
_webhook_job_registry: Dict[str, dict] = {}


def register_pyannote_job(job_id: str, ep_id: str, output_path: str) -> None:
    """Register a job for webhook correlation.

    Call this when submitting a diarization job with webhook enabled.

    Args:
        job_id: PyannoteAI job ID
        ep_id: Episode identifier
        output_path: Path to save diarization output
    """
    _webhook_job_registry[job_id] = {
        "ep_id": ep_id,
        "output_path": output_path,
        "registered_at": __import__("datetime").datetime.utcnow().isoformat(),
    }
    LOGGER.info(f"Registered PyannoteAI job {job_id} for {ep_id}")


def get_pyannote_job_metadata(job_id: str) -> Optional[dict]:
    """Get metadata for a registered job.

    Args:
        job_id: PyannoteAI job ID

    Returns:
        Job metadata dict or None if not found
    """
    return _webhook_job_registry.get(job_id)


def mark_pyannote_job_complete(job_id: str) -> None:
    """Mark a job as complete and remove from registry.

    Args:
        job_id: PyannoteAI job ID
    """
    if job_id in _webhook_job_registry:
        del _webhook_job_registry[job_id]
        LOGGER.info(f"Completed PyannoteAI job {job_id}")


@router.post("/webhooks/pyannote/diarization", response_model=PyannoteWebhookResponse)
async def receive_pyannote_webhook(payload: PyannoteWebhookPayload) -> PyannoteWebhookResponse:
    """Receive webhook from PyannoteAI on diarization job completion.

    This endpoint receives async notifications when a diarization job completes.
    Persists output immediately since Pyannote deletes results after 24 hours.

    Per official docs:
    - Webhook retry policy: immediate, 1 min, 5 min
    - Expected response: HTTP 200 within 10 seconds

    Args:
        payload: Webhook payload from PyannoteAI

    Returns:
        Acknowledgment response
    """
    LOGGER.info(f"Received PyannoteAI webhook: jobId={payload.jobId}, status={payload.status}")

    # Handle non-success status
    if payload.status != "succeeded":
        LOGGER.warning(f"PyannoteAI job {payload.jobId} {payload.status}")
        return PyannoteWebhookResponse(
            received=True,
            status=payload.status,
            error=f"Job {payload.status}",
        )

    # Look up job metadata for correlation
    job_metadata = get_pyannote_job_metadata(payload.jobId)
    if not job_metadata:
        LOGGER.warning(f"PyannoteAI job {payload.jobId} not found in registry")
        return PyannoteWebhookResponse(
            received=True,
            status=payload.status,
            error="job_not_found_in_registry",
        )

    ep_id = job_metadata.get("ep_id", "unknown")
    output_path = job_metadata.get("output_path")

    # Parse and save diarization output
    try:
        output = payload.output or {}

        # Prefer exclusiveDiarization for clean segment merging
        diar_data = output.get("exclusiveDiarization") or output.get("diarization", [])

        if not diar_data:
            LOGGER.warning(f"No diarization data in webhook for {payload.jobId}")
            return PyannoteWebhookResponse(
                received=True,
                status=payload.status,
                error="no_diarization_data",
            )

        # Save to JSONL manifest
        from py_screenalytics.audio.models import DiarizationSegment

        segments = []
        for entry in diar_data:
            if isinstance(entry, dict):
                segments.append(DiarizationSegment(
                    start=entry.get("start", 0),
                    end=entry.get("end", 0),
                    speaker=entry.get("speaker", "SPEAKER_00"),
                    confidence=entry.get("confidence"),
                ))

        # Save JSONL manifest
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with output_file.open("w", encoding="utf-8") as f:
                for segment in segments:
                    f.write(segment.model_dump_json() + "\n")

            LOGGER.info(f"PyannoteAI diarization saved: {output_path} ({len(segments)} segments)")

            # Also save raw response for debugging
            raw_path = output_file.parent / "audio_diarization_pyannote_raw.json"
            with raw_path.open("w", encoding="utf-8") as f:
                json.dump({"jobId": payload.jobId, "status": payload.status, "output": output}, f, indent=2)

        # Mark job complete
        mark_pyannote_job_complete(payload.jobId)

        return PyannoteWebhookResponse(
            received=True,
            status=payload.status,
            segments_saved=len(segments),
        )

    except Exception as e:
        LOGGER.exception(f"Failed to process PyannoteAI webhook: {e}")
        return PyannoteWebhookResponse(
            received=True,
            status=payload.status,
            error=str(e),
        )
