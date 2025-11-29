"""Quality control checks for the audio pipeline.

Handles:
- Duration drift validation
- SNR checks
- Diarization confidence validation
- ASR confidence validation
- Voice cluster validation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from .models import (
    ASRSegment,
    DiarizationSegment,
    QCConfig,
    QCMetric,
    QCReport,
    QCStatus,
    TranscriptRow,
    VoiceBankMatchResult,
    VoiceCluster,
)

LOGGER = logging.getLogger(__name__)


def run_qc_checks(
    ep_id: str,
    config: QCConfig,
    duration_original_s: float,
    duration_final_s: float,
    snr_db: Optional[float],
    diarization_segments: List[DiarizationSegment],
    asr_segments: List[ASRSegment],
    voice_clusters: List[VoiceCluster],
    voice_mapping: List[VoiceBankMatchResult],
    transcript_rows: List[TranscriptRow],
    output_path: Path,
    overwrite: bool = False,
) -> QCReport:
    """Run all QC checks and generate report.

    Args:
        ep_id: Episode identifier
        config: QC configuration
        duration_original_s: Duration of original audio
        duration_final_s: Duration of final audio
        snr_db: Estimated SNR in dB
        diarization_segments: Diarization results
        asr_segments: ASR results
        voice_clusters: Voice cluster data
        voice_mapping: Voice bank mapping
        transcript_rows: Final transcript rows
        output_path: Path for QC report JSON
        overwrite: Whether to overwrite existing report

    Returns:
        QCReport object
    """
    if output_path.exists() and not overwrite:
        LOGGER.info(f"QC report already exists: {output_path}")
        return _load_qc_report(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = []
    warnings = []
    errors = []

    # Duration drift check
    if duration_original_s > 0 and duration_final_s > 0:
        drift_pct = abs(duration_final_s - duration_original_s) / duration_original_s * 100
        passed = drift_pct <= config.max_duration_drift_pct

        metrics.append(QCMetric(
            name="duration_drift_pct",
            value=drift_pct,
            threshold=config.max_duration_drift_pct,
            passed=passed,
            severity="error" if not passed else "info",
        ))

        if not passed:
            errors.append(f"Duration drift {drift_pct:.2f}% exceeds threshold {config.max_duration_drift_pct}%")
    else:
        drift_pct = None

    # SNR check
    if snr_db is not None:
        passed = snr_db >= config.min_snr_db
        warn = snr_db < config.warn_snr_db

        metrics.append(QCMetric(
            name="snr_db",
            value=snr_db,
            threshold=config.min_snr_db,
            passed=passed,
            severity="error" if not passed else ("warn" if warn else "info"),
        ))

        if not passed:
            errors.append(f"SNR {snr_db:.1f}dB below minimum {config.min_snr_db}dB")
        elif snr_db < config.warn_snr_db:
            warnings.append(f"SNR {snr_db:.1f}dB below warning threshold {config.warn_snr_db}dB")

    # Diarization confidence check
    if diarization_segments:
        confidences = [s.confidence for s in diarization_segments if s.confidence is not None]
        if confidences:
            mean_conf = sum(confidences) / len(confidences)
            passed = mean_conf >= config.min_diarization_conf

            metrics.append(QCMetric(
                name="mean_diarization_conf",
                value=mean_conf,
                threshold=config.min_diarization_conf,
                passed=passed,
                severity="warn" if not passed else "info",
            ))

            if not passed:
                warnings.append(f"Mean diarization confidence {mean_conf:.2f} below {config.min_diarization_conf}")
        else:
            mean_conf = None
    else:
        mean_conf = None

    # ASR confidence check
    if asr_segments:
        confidences = [s.confidence for s in asr_segments if s.confidence is not None]
        if confidences:
            mean_asr_conf = sum(confidences) / len(confidences)
            passed = mean_asr_conf >= config.min_asr_conf

            metrics.append(QCMetric(
                name="mean_asr_conf",
                value=mean_asr_conf,
                threshold=config.min_asr_conf,
                passed=passed,
                severity="warn" if not passed else "info",
            ))

            if not passed:
                warnings.append(f"Mean ASR confidence {mean_asr_conf:.2f} below {config.min_asr_conf}")
        else:
            mean_asr_conf = None
    else:
        mean_asr_conf = None

    # Voice cluster validation
    labeled_voices = sum(1 for m in voice_mapping if m.similarity is not None)
    unlabeled_voices = len(voice_mapping) - labeled_voices

    # Check minimum cluster duration
    short_clusters = []
    for cluster in voice_clusters:
        if cluster.total_duration < config.min_cluster_duration_s:
            short_clusters.append(cluster.voice_cluster_id)

    if short_clusters:
        warnings.append(f"{len(short_clusters)} clusters below minimum duration {config.min_cluster_duration_s}s")

    # Transcript speaker fields validation
    if config.require_all_speaker_fields and transcript_rows:
        missing_fields = []
        for i, row in enumerate(transcript_rows):
            if not row.speaker_id:
                missing_fields.append(f"row {i}: missing speaker_id")
            if not row.speaker_display_name:
                missing_fields.append(f"row {i}: missing speaker_display_name")
            if not row.voice_cluster_id:
                missing_fields.append(f"row {i}: missing voice_cluster_id")
            if not row.voice_bank_id:
                missing_fields.append(f"row {i}: missing voice_bank_id")

        if missing_fields:
            warnings.append(f"{len(missing_fields)} transcript rows have missing speaker fields")
            metrics.append(QCMetric(
                name="transcript_speaker_fields",
                value=len(transcript_rows) - len(missing_fields),
                threshold=len(transcript_rows),
                passed=False,
                severity="warn",
            ))

    # Voice cluster consistency check
    transcript_cluster_ids = set(r.voice_cluster_id for r in transcript_rows)
    cluster_ids = set(c.voice_cluster_id for c in voice_clusters)
    mapping_cluster_ids = set(m.voice_cluster_id for m in voice_mapping)

    if transcript_cluster_ids - cluster_ids:
        warnings.append(f"Transcript references {len(transcript_cluster_ids - cluster_ids)} unknown cluster IDs")

    if transcript_cluster_ids - mapping_cluster_ids:
        warnings.append(f"Transcript references {len(transcript_cluster_ids - mapping_cluster_ids)} unmapped cluster IDs")

    # Determine overall status
    if errors:
        status = QCStatus.NEEDS_REVIEW
    elif warnings:
        status = QCStatus.WARN
    else:
        status = QCStatus.OK

    # Build report
    report = QCReport(
        ep_id=ep_id,
        status=status,
        metrics=metrics,
        duration_original_s=duration_original_s,
        duration_final_s=duration_final_s,
        duration_drift_pct=drift_pct,
        snr_db=snr_db,
        mean_diarization_conf=mean_conf,
        mean_asr_conf=mean_asr_conf,
        voice_cluster_count=len(voice_clusters),
        labeled_voices=labeled_voices,
        unlabeled_voices=unlabeled_voices,
        transcript_row_count=len(transcript_rows),
        warnings=warnings,
        errors=errors,
    )

    # Save report
    _save_qc_report(report, output_path)

    LOGGER.info(f"QC report generated: status={status.value}, {len(warnings)} warnings, {len(errors)} errors")

    return report


def _save_qc_report(report: QCReport, output_path: Path):
    """Save QC report to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = report.model_dump()
    # Convert QCStatus enum to string
    data["status"] = report.status.value
    data["metrics"] = [m.model_dump() for m in report.metrics]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_qc_report(report_path: Path) -> QCReport:
    """Load QC report from JSON file."""
    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert status string back to enum
    data["status"] = QCStatus(data["status"])
    data["metrics"] = [QCMetric(**m) for m in data.get("metrics", [])]

    return QCReport(**data)


def get_qc_summary(report: QCReport) -> dict:
    """Get a summary dictionary from QC report."""
    return {
        "status": report.status.value,
        "duration_drift_pct": report.duration_drift_pct,
        "snr_db": report.snr_db,
        "mean_diarization_conf": report.mean_diarization_conf,
        "mean_asr_conf": report.mean_asr_conf,
        "voice_cluster_count": report.voice_cluster_count,
        "labeled_voices": report.labeled_voices,
        "unlabeled_voices": report.unlabeled_voices,
        "transcript_row_count": report.transcript_row_count,
        "warning_count": len(report.warnings),
        "error_count": len(report.errors),
    }
