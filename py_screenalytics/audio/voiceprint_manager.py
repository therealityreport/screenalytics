"""Voiceprint creation and management for cast members.

Creates voiceprints using the Pyannote API from selected audio segments.

Process:
1. Slice audio segments using ffmpeg
2. Upload to S3 with presigned URL
3. POST /v1/voiceprint to Pyannote
4. Poll for result
5. Update cast record with voiceprint_blob
6. Save per-cast artifact
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import VoiceprintIdentificationConfig
from .pyannote_api import PyannoteAPIClient, PyannoteAPIError, VoiceprintJobResult
from .voiceprint_selection import CastVoiceprintSelection, VoiceprintCandidate

LOGGER = logging.getLogger(__name__)


@dataclass
class VoiceprintCreationResult:
    """Result of creating a voiceprint for a cast member."""

    cast_id: str
    cast_name: Optional[str]
    status: str  # "success" | "failed" | "skipped"
    voiceprint_blob: Optional[str] = None
    job_id: Optional[str] = None
    segments_used: List[str] = field(default_factory=list)
    total_duration_s: float = 0.0
    mean_confidence: Optional[float] = None
    score: Optional[float] = None
    error: Optional[str] = None
    skipped_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cast_id": self.cast_id,
            "cast_name": self.cast_name,
            "status": self.status,
            "voiceprint_blob": self.voiceprint_blob[:50] + "..." if self.voiceprint_blob else None,
            "job_id": self.job_id,
            "segments_used": self.segments_used,
            "total_duration_s": self.total_duration_s,
            "mean_confidence": self.mean_confidence,
            "score": self.score,
            "error": self.error,
            "skipped_reason": self.skipped_reason,
        }


def slice_audio_segment(
    audio_path: Path,
    start: float,
    end: float,
    output_path: Path,
) -> Path:
    """Extract audio segment using ffmpeg.

    Args:
        audio_path: Source audio file
        start: Start time in seconds
        end: End time in seconds
        output_path: Output file path

    Returns:
        Path to the extracted segment

    Raises:
        subprocess.CalledProcessError: If ffmpeg fails
    """
    duration = end - start

    # Ensure we don't exceed 30s (Pyannote limit)
    if duration > 30.0:
        LOGGER.warning(f"Segment duration {duration:.1f}s exceeds 30s limit, truncating")
        duration = 30.0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", str(audio_path),
        "-ss", str(start),
        "-t", str(duration),
        "-c:a", "pcm_s16le",  # WAV format
        "-ar", "16000",  # 16kHz sample rate (standard for speech)
        "-ac", "1",  # Mono
        str(output_path),
    ]

    LOGGER.debug(f"Slicing audio: {start:.2f}s - {end:.2f}s -> {output_path}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        check=True,
    )

    if not output_path.exists():
        raise RuntimeError(f"ffmpeg did not create output file: {output_path}")

    return output_path


def concatenate_segments(
    segments: List[Path],
    output_path: Path,
) -> Path:
    """Concatenate multiple audio segments into one file.

    Args:
        segments: List of segment file paths
        output_path: Output file path

    Returns:
        Path to concatenated file
    """
    if len(segments) == 1:
        # Just copy the single segment
        import shutil
        shutil.copy(segments[0], output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create concat list file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for seg_path in segments:
            f.write(f"file '{seg_path}'\n")
        concat_list = f.name

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list,
            "-c:a", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_path),
        ]

        subprocess.run(cmd, capture_output=True, check=True)

        if not output_path.exists():
            raise RuntimeError(f"ffmpeg did not create concatenated file: {output_path}")

        return output_path
    finally:
        Path(concat_list).unlink(missing_ok=True)


def create_voiceprint_for_cast(
    cast_id: str,
    cast_name: Optional[str],
    selection: CastVoiceprintSelection,
    audio_path: Path,
    show_id: str,
    ep_id: str,
    client: PyannoteAPIClient,
    config: VoiceprintIdentificationConfig,
    artifacts_dir: Path,
) -> VoiceprintCreationResult:
    """Create a voiceprint for a single cast member.

    Args:
        cast_id: Cast member ID
        cast_name: Cast member display name
        selection: Voiceprint segment selection
        audio_path: Path to episode audio file
        show_id: Show identifier
        ep_id: Episode identifier
        client: Pyannote API client
        config: Configuration
        artifacts_dir: Directory to save artifacts

    Returns:
        VoiceprintCreationResult
    """
    if selection.status != "ready":
        return VoiceprintCreationResult(
            cast_id=cast_id,
            cast_name=cast_name,
            status="skipped",
            skipped_reason=selection.reason or f"Selection status: {selection.status}",
            total_duration_s=selection.total_duration_s,
        )

    # Check overwrite policy
    from apps.api.services.cast import CastService

    cast_service = CastService()
    new_score = selection.score or 0.0

    should_update = cast_service.should_update_voiceprint(
        show_id=show_id,
        cast_id=cast_id,
        new_score=new_score,
        policy=config.voiceprint_overwrite_policy,
        improvement_threshold=config.if_better_improvement_threshold,
    )

    if not should_update:
        existing = cast_service.get_cast_voiceprint(show_id, cast_id)
        old_score = existing.get("voiceprint_metadata", {}).get("score", 0) if existing else 0
        LOGGER.info(
            f"[{ep_id}] Skipping voiceprint for {cast_name} ({cast_id}): "
            f"policy={config.voiceprint_overwrite_policy}, "
            f"old_score={old_score:.2f}, new_score={new_score:.2f}"
        )
        return VoiceprintCreationResult(
            cast_id=cast_id,
            cast_name=cast_name,
            status="skipped",
            skipped_reason=f"Overwrite policy '{config.voiceprint_overwrite_policy}' - existing score {old_score:.2f} >= new score {new_score:.2f}",
            total_duration_s=selection.total_duration_s,
            mean_confidence=selection.mean_confidence,
            score=selection.score,
        )

    # Slice and concatenate segments
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        segment_files: List[Path] = []

        for i, candidate in enumerate(selection.candidates):
            seg_path = tmp_path / f"segment_{i}.wav"
            try:
                slice_audio_segment(
                    audio_path=audio_path,
                    start=candidate.start,
                    end=candidate.end,
                    output_path=seg_path,
                )
                segment_files.append(seg_path)
            except subprocess.CalledProcessError as e:
                LOGGER.error(f"[{ep_id}] Failed to slice segment {candidate.segment_id}: {e}")
                continue

        if not segment_files:
            return VoiceprintCreationResult(
                cast_id=cast_id,
                cast_name=cast_name,
                status="failed",
                error="Failed to slice any segments",
                total_duration_s=selection.total_duration_s,
            )

        # Concatenate if multiple segments
        combined_path = tmp_path / f"combined_{cast_id}.wav"
        try:
            concatenate_segments(segment_files, combined_path)
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"[{ep_id}] Failed to concatenate segments for {cast_name}: {e}")
            return VoiceprintCreationResult(
                cast_id=cast_id,
                cast_name=cast_name,
                status="failed",
                error=f"Failed to concatenate segments: {e}",
                total_duration_s=selection.total_duration_s,
            )

        # Upload to S3 and get presigned URL
        try:
            media_url = client.upload_and_get_url(combined_path, expiry_seconds=3600)
        except PyannoteAPIError as e:
            LOGGER.error(f"[{ep_id}] Failed to upload audio for {cast_name}: {e}")
            return VoiceprintCreationResult(
                cast_id=cast_id,
                cast_name=cast_name,
                status="failed",
                error=f"Failed to upload audio: {e}",
                total_duration_s=selection.total_duration_s,
            )

        # Submit voiceprint job
        try:
            job_id = client.submit_voiceprint(media_url)
            LOGGER.info(f"[{ep_id}] Submitted voiceprint job for {cast_name}: {job_id}")
        except PyannoteAPIError as e:
            LOGGER.error(f"[{ep_id}] Failed to submit voiceprint job for {cast_name}: {e}")
            return VoiceprintCreationResult(
                cast_id=cast_id,
                cast_name=cast_name,
                status="failed",
                error=f"Failed to submit voiceprint job: {e}",
                total_duration_s=selection.total_duration_s,
            )

        # Poll for result
        try:
            result: VoiceprintJobResult = client.poll_voiceprint_job(job_id, max_wait=300.0)
        except PyannoteAPIError as e:
            LOGGER.error(f"[{ep_id}] Voiceprint job failed for {cast_name}: {e}")
            return VoiceprintCreationResult(
                cast_id=cast_id,
                cast_name=cast_name,
                status="failed",
                job_id=job_id,
                error=f"Voiceprint job failed: {e}",
                total_duration_s=selection.total_duration_s,
            )

        if result.status != "succeeded" or not result.voiceprint:
            return VoiceprintCreationResult(
                cast_id=cast_id,
                cast_name=cast_name,
                status="failed",
                job_id=job_id,
                error=result.error or "No voiceprint in result",
                total_duration_s=selection.total_duration_s,
            )

        # Update cast record with voiceprint
        metadata = {
            "segments_used": [c.segment_id for c in selection.candidates],
            "total_duration_s": selection.total_duration_s,
            "mean_confidence": selection.mean_confidence,
            "score": selection.score,
            "source_episode_id": ep_id,
            "job_id": job_id,
        }

        cast_service.update_cast_voiceprint(
            show_id=show_id,
            cast_id=cast_id,
            voiceprint_blob=result.voiceprint,
            source_ep_id=ep_id,
            metadata=metadata,
        )

        LOGGER.info(f"[{ep_id}] Created voiceprint for {cast_name}: score={selection.score:.2f}")

        # Save per-cast artifact
        artifact_path = artifacts_dir / f"{cast_id}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cast_id": cast_id,
                    "cast_name": cast_name,
                    "ep_id": ep_id,
                    "job_id": job_id,
                    "status": "success",
                    "metadata": metadata,
                    "raw_response": result.raw_response,
                },
                f,
                indent=2,
            )

        return VoiceprintCreationResult(
            cast_id=cast_id,
            cast_name=cast_name,
            status="success",
            voiceprint_blob=result.voiceprint,
            job_id=job_id,
            segments_used=[c.segment_id for c in selection.candidates],
            total_duration_s=selection.total_duration_s,
            mean_confidence=selection.mean_confidence,
            score=selection.score,
        )


def create_voiceprints_for_episode(
    ep_id: str,
    show_id: str,
    selections: Dict[str, CastVoiceprintSelection],
    audio_path: Path,
    config: Optional[VoiceprintIdentificationConfig] = None,
    artifacts_dir: Optional[Path] = None,
) -> Dict[str, VoiceprintCreationResult]:
    """Create voiceprints for all eligible cast members.

    Args:
        ep_id: Episode identifier
        show_id: Show identifier
        selections: Voiceprint selections from select_voiceprint_segments
        audio_path: Path to episode audio file
        config: Optional configuration
        artifacts_dir: Directory to save artifacts (default: data/manifests/{ep_id}/voiceprints/)

    Returns:
        Dict mapping cast_id -> VoiceprintCreationResult
    """
    if config is None:
        config = VoiceprintIdentificationConfig()

    if artifacts_dir is None:
        from pathlib import Path
        import os
        data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data"))
        artifacts_dir = data_root / "manifests" / ep_id / "voiceprints"

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Filter to ready selections only
    ready_selections = {
        cast_id: sel
        for cast_id, sel in selections.items()
        if sel.status == "ready"
    }

    LOGGER.info(
        f"[{ep_id}] Creating voiceprints for {len(ready_selections)} cast members "
        f"(skipping {len(selections) - len(ready_selections)} with insufficient data)"
    )

    results: Dict[str, VoiceprintCreationResult] = {}

    # Create Pyannote client
    try:
        client = PyannoteAPIClient()
    except PyannoteAPIError as e:
        LOGGER.error(f"[{ep_id}] Failed to create Pyannote client: {e}")
        # Return skipped results for all
        for cast_id, sel in selections.items():
            results[cast_id] = VoiceprintCreationResult(
                cast_id=cast_id,
                cast_name=sel.cast_name,
                status="failed",
                error=f"Failed to create Pyannote client: {e}",
            )
        return results

    try:
        # Process each cast member
        for cast_id, selection in selections.items():
            result = create_voiceprint_for_cast(
                cast_id=cast_id,
                cast_name=selection.cast_name,
                selection=selection,
                audio_path=audio_path,
                show_id=show_id,
                ep_id=ep_id,
                client=client,
                config=config,
                artifacts_dir=artifacts_dir,
            )
            results[cast_id] = result
    finally:
        client.close()

    # Log summary
    success_count = sum(1 for r in results.values() if r.status == "success")
    failed_count = sum(1 for r in results.values() if r.status == "failed")
    skipped_count = sum(1 for r in results.values() if r.status == "skipped")

    LOGGER.info(
        f"[{ep_id}] Voiceprint creation complete: "
        f"{success_count} success, {failed_count} failed, {skipped_count} skipped"
    )

    return results


def save_voiceprint_summary(
    ep_id: str,
    results: Dict[str, VoiceprintCreationResult],
    output_path: Path,
) -> None:
    """Save summary of voiceprint creation results.

    Args:
        ep_id: Episode identifier
        results: Voiceprint creation results
        output_path: Path to save JSON summary
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "ep_id": ep_id,
        "schema_version": "voiceprint_creation_v1",
        "summary": {
            "total": len(results),
            "success": sum(1 for r in results.values() if r.status == "success"),
            "failed": sum(1 for r in results.values() if r.status == "failed"),
            "skipped": sum(1 for r in results.values() if r.status == "skipped"),
        },
        "results": {cast_id: r.to_dict() for cast_id, r in results.items()},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info(f"[{ep_id}] Saved voiceprint creation summary to {output_path}")
