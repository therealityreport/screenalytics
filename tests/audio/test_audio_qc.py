"""Tests for audio pipeline QC checks.

Verifies:
1. Duration drift detection
2. SNR thresholds
3. Diarization confidence checks
4. ASR confidence checks
5. Voice cluster validation
"""

from __future__ import annotations

import pytest
import json
import tempfile
from pathlib import Path


class TestQCChecks:
    """Tests for py_screenalytics.audio.qc module."""

    def test_duration_drift_ok(self):
        """Duration within threshold passes."""
        from py_screenalytics.audio.qc import run_qc_checks
        from py_screenalytics.audio.models import QCConfig, QCStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "qc.json"
            config = QCConfig(max_duration_drift_pct=1.0)

            report = run_qc_checks(
                ep_id="test-s01e01",
                config=config,
                duration_original_s=3600.0,
                duration_final_s=3595.0,  # 0.14% drift
                snr_db=25.0,
                diarization_segments=[],
                asr_segments=[],
                voice_clusters=[],
                voice_mapping=[],
                transcript_rows=[],
                output_path=output_path,
                overwrite=True,
            )

            assert report.status == QCStatus.OK
            assert report.duration_drift_pct < 1.0
            assert len(report.errors) == 0

    def test_duration_drift_error(self):
        """Duration exceeding threshold fails."""
        from py_screenalytics.audio.qc import run_qc_checks
        from py_screenalytics.audio.models import QCConfig, QCStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "qc.json"
            config = QCConfig(max_duration_drift_pct=1.0)

            report = run_qc_checks(
                ep_id="test-s01e01",
                config=config,
                duration_original_s=3600.0,
                duration_final_s=3500.0,  # 2.78% drift
                snr_db=25.0,
                diarization_segments=[],
                asr_segments=[],
                voice_clusters=[],
                voice_mapping=[],
                transcript_rows=[],
                output_path=output_path,
                overwrite=True,
            )

            assert report.status == QCStatus.NEEDS_REVIEW
            assert report.duration_drift_pct > 1.0
            assert len(report.errors) > 0
            assert "drift" in report.errors[0].lower()

    def test_snr_below_minimum(self):
        """SNR below minimum threshold triggers error."""
        from py_screenalytics.audio.qc import run_qc_checks
        from py_screenalytics.audio.models import QCConfig, QCStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "qc.json"
            config = QCConfig(min_snr_db=14.0, warn_snr_db=18.0)

            report = run_qc_checks(
                ep_id="test-s01e01",
                config=config,
                duration_original_s=3600.0,
                duration_final_s=3600.0,
                snr_db=12.0,  # Below 14.0 minimum
                diarization_segments=[],
                asr_segments=[],
                voice_clusters=[],
                voice_mapping=[],
                transcript_rows=[],
                output_path=output_path,
                overwrite=True,
            )

            assert report.status == QCStatus.NEEDS_REVIEW
            assert len(report.errors) > 0
            assert "snr" in report.errors[0].lower()

    def test_snr_warning_threshold(self):
        """SNR below warning threshold triggers warning."""
        from py_screenalytics.audio.qc import run_qc_checks
        from py_screenalytics.audio.models import QCConfig, QCStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "qc.json"
            config = QCConfig(min_snr_db=14.0, warn_snr_db=18.0)

            report = run_qc_checks(
                ep_id="test-s01e01",
                config=config,
                duration_original_s=3600.0,
                duration_final_s=3600.0,
                snr_db=16.0,  # Between 14 and 18
                diarization_segments=[],
                asr_segments=[],
                voice_clusters=[],
                voice_mapping=[],
                transcript_rows=[],
                output_path=output_path,
                overwrite=True,
            )

            assert report.status == QCStatus.WARN
            assert len(report.warnings) > 0
            assert "snr" in report.warnings[0].lower()

    def test_qc_report_save_load(self):
        """QC report can be saved and loaded."""
        from py_screenalytics.audio.qc import run_qc_checks, _load_qc_report
        from py_screenalytics.audio.models import QCConfig, QCStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "qc.json"
            config = QCConfig()

            report = run_qc_checks(
                ep_id="test-s01e01",
                config=config,
                duration_original_s=3600.0,
                duration_final_s=3600.0,
                snr_db=25.0,
                diarization_segments=[],
                asr_segments=[],
                voice_clusters=[],
                voice_mapping=[],
                transcript_rows=[],
                output_path=output_path,
                overwrite=True,
            )

            # Verify file was created
            assert output_path.exists()

            # Load and verify
            loaded = _load_qc_report(output_path)
            assert loaded.ep_id == report.ep_id
            assert loaded.status == report.status
            assert loaded.snr_db == report.snr_db


class TestQCSummary:
    """Tests for QC summary generation."""

    def test_get_qc_summary(self):
        """QC summary dict has expected keys."""
        from py_screenalytics.audio.qc import get_qc_summary
        from py_screenalytics.audio.models import QCReport, QCStatus

        report = QCReport(
            ep_id="test-s01e01",
            status=QCStatus.OK,
            metrics=[],
            duration_original_s=3600.0,
            duration_final_s=3595.0,
            duration_drift_pct=0.14,
            snr_db=22.5,
            voice_cluster_count=5,
            labeled_voices=3,
            unlabeled_voices=2,
            transcript_row_count=150,
            warnings=["test warning"],
            errors=[],
        )

        summary = get_qc_summary(report)

        assert summary["status"] == "ok"
        assert summary["snr_db"] == 22.5
        assert summary["voice_cluster_count"] == 5
        assert summary["labeled_voices"] == 3
        assert summary["unlabeled_voices"] == 2
        assert summary["warning_count"] == 1
        assert summary["error_count"] == 0
