"""Tests for audio pipeline Pydantic models.

Verifies:
1. Model instantiation with valid data
2. Validation of required fields
3. Enum serialization
4. Config loading from YAML
"""

from __future__ import annotations

import pytest


class TestAudioModels:
    """Tests for py_screenalytics.audio.models"""

    def test_diarization_segment_creation(self):
        """DiarizationSegment can be created with valid data."""
        from py_screenalytics.audio.models import DiarizationSegment

        seg = DiarizationSegment(
            speaker_label="SPEAKER_00",
            start_ms=0,
            end_ms=5000,
            confidence=0.95,
        )
        assert seg.speaker_label == "SPEAKER_00"
        assert seg.start_ms == 0
        assert seg.end_ms == 5000
        assert seg.duration_ms == 5000
        assert seg.confidence == 0.95

    def test_asr_segment_creation(self):
        """ASRSegment can be created with word-level timestamps."""
        from py_screenalytics.audio.models import ASRSegment, WordTiming

        words = [
            WordTiming(word="Hello", start_ms=0, end_ms=500),
            WordTiming(word="world", start_ms=550, end_ms=1000),
        ]
        seg = ASRSegment(
            text="Hello world",
            start_ms=0,
            end_ms=1000,
            confidence=0.98,
            words=words,
        )
        assert seg.text == "Hello world"
        assert len(seg.words) == 2
        assert seg.words[0].word == "Hello"

    def test_voice_cluster_creation(self):
        """VoiceCluster can be created with embedding."""
        from py_screenalytics.audio.models import VoiceCluster

        cluster = VoiceCluster(
            voice_cluster_id="vc_001",
            total_duration=120.5,
            segment_count=15,
            centroid=[0.1] * 256,
        )
        assert cluster.voice_cluster_id == "vc_001"
        assert cluster.total_duration == 120.5
        assert len(cluster.centroid) == 256

    def test_transcript_row_creation(self):
        """TranscriptRow can be created with all fields."""
        from py_screenalytics.audio.models import TranscriptRow

        row = TranscriptRow(
            idx=0,
            start_ms=0,
            end_ms=5000,
            text="Hello, how are you?",
            speaker_id="cast_001",
            speaker_display_name="John Doe",
            voice_cluster_id="vc_001",
            voice_bank_id="vb_001",
            diarization_confidence=0.92,
            asr_confidence=0.97,
        )
        assert row.idx == 0
        assert row.speaker_display_name == "John Doe"
        assert row.voice_bank_id == "vb_001"

    def test_qc_status_enum(self):
        """QCStatus enum values are correct."""
        from py_screenalytics.audio.models import QCStatus

        assert QCStatus.OK.value == "ok"
        assert QCStatus.WARN.value == "warn"
        assert QCStatus.NEEDS_REVIEW.value == "needs_review"

    def test_qc_config_defaults(self):
        """QCConfig has sensible defaults."""
        from py_screenalytics.audio.models import QCConfig

        config = QCConfig()
        assert config.max_duration_drift_pct == 1.0
        assert config.min_snr_db == 14.0
        assert config.warn_snr_db == 18.0
        assert config.min_diarization_conf == 0.65
        assert config.min_asr_conf == 0.70

    def test_qc_report_creation(self):
        """QCReport can be created with metrics."""
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
            warnings=[],
            errors=[],
        )
        assert report.status == QCStatus.OK
        assert report.snr_db == 22.5
        assert report.voice_cluster_count == 5


class TestAudioConfig:
    """Tests for audio pipeline configuration loading."""

    def test_audio_pipeline_config_creation(self):
        """AudioPipelineConfig can be created with nested configs."""
        from py_screanalytics.audio.models import (
            AudioPipelineConfig,
            SeparationConfig,
            EnhanceConfig,
            DiarizationConfig,
            ASRConfig,
            VoiceClusteringConfig,
            QCConfig,
            ExportConfig,
        )

        config = AudioPipelineConfig(
            separation=SeparationConfig(provider="mdx_extra"),
            enhance=EnhanceConfig(provider="resemble"),
            diarization=DiarizationConfig(provider="pyannote"),
            asr=ASRConfig(provider="openai_whisper"),
            voice_clustering=VoiceClusteringConfig(),
            qc=QCConfig(),
            export=ExportConfig(),
        )
        assert config.separation.provider == "mdx_extra"
        assert config.asr.provider == "openai_whisper"

    def test_load_audio_config_from_yaml(self):
        """Config can be loaded from YAML file."""
        from pathlib import Path
        import yaml

        config_path = Path(__file__).parents[2] / "config" / "pipeline" / "audio.yaml"

        if not config_path.exists():
            pytest.skip("audio.yaml not found")

        with config_path.open("r") as f:
            data = yaml.safe_load(f)

        assert "audio_pipeline" in data
        pipeline_data = data["audio_pipeline"]
        assert "separation" in pipeline_data
        assert "asr" in pipeline_data
