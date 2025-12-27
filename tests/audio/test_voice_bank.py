"""Tests for voice bank management.

Verifies:
1. Matching against existing voice bank entry (similarity above threshold)
2. Handling of below-threshold (creating unlabeled voice entry)
3. Writing audio_voice_mapping.json with required fields
4. Persistent voice bank storage
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestVoiceBank:
    """Tests for py_screenalytics.audio.voice_bank module."""

    def test_voice_bank_empty_show(self):
        """Empty voice bank returns empty list."""
        from py_screenalytics.audio.voice_bank import VoiceBank
        from py_screenalytics.audio.models import VoiceBankConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = VoiceBankConfig(data_dir=tmpdir)
            bank = VoiceBank(config)

            entries = bank.get_entries("nonexistent_show")
            assert entries == []

    def test_voice_bank_add_entry(self):
        """Adding entry persists to JSON file."""
        from py_screenalytics.audio.voice_bank import VoiceBank
        from py_screenalytics.audio.models import VoiceBankConfig, VoiceBankEntry

        with tempfile.TemporaryDirectory() as tmpdir:
            config = VoiceBankConfig(data_dir=tmpdir)
            bank = VoiceBank(config)

            entry = VoiceBankEntry(
                voice_bank_id="voice_lisa_barlow",
                show_id="rhoslc",
                cast_id="lisa_barlow",
                display_name="Lisa Barlow",
                embeddings=[[0.1] * 256],
                is_labeled=True,
            )

            bank.add_entry(entry, "rhoslc")

            # Verify file exists
            bank_path = Path(tmpdir) / "rhoslc.json"
            assert bank_path.exists()

            # Verify content
            with bank_path.open("r") as f:
                data = json.load(f)

            assert len(data) == 1
            assert data[0]["voice_bank_id"] == "voice_lisa_barlow"
            assert data[0]["display_name"] == "Lisa Barlow"
            assert data[0]["is_labeled"] is True

    def test_voice_bank_match_above_threshold(self):
        """Matching with similarity above threshold returns match."""
        from py_screenalytics.audio.voice_bank import VoiceBank
        from py_screenalytics.audio.models import (
            VoiceBankConfig,
            VoiceBankEntry,
            VoiceCluster,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = VoiceBankConfig(data_dir=tmpdir, auto_create_unlabeled=False)
            bank = VoiceBank(config)

            # Create a voice bank entry with a known embedding
            base_embedding = np.random.randn(256)
            base_embedding = base_embedding / np.linalg.norm(base_embedding)

            entry = VoiceBankEntry(
                voice_bank_id="voice_lisa_barlow",
                show_id="rhoslc",
                cast_id="lisa_barlow",
                display_name="Lisa Barlow",
                embeddings=[base_embedding.tolist()],
                is_labeled=True,
            )
            bank.add_entry(entry, "rhoslc")

            # Create cluster with matching embedding
            similar_embedding = base_embedding.copy()

            cluster = VoiceCluster(
                voice_cluster_id="VC_01",
                segments=[],
                total_duration=30.0,
                segment_count=5,
                centroid=similar_embedding.tolist(),
            )

            # Match should succeed
            result = bank.match_cluster("rhoslc", cluster, threshold=0.78)

            assert result.voice_bank_id == "voice_lisa_barlow"
            assert result.speaker_id == "SPK_LISA_BARLOW"
            assert result.speaker_display_name == "Lisa Barlow"
            assert result.similarity is not None
            assert result.similarity >= 0.78
            assert result.is_new_entry is False

    def test_voice_bank_match_below_threshold_creates_unlabeled(self):
        """Matching below threshold creates unlabeled entry."""
        from py_screenalytics.audio.voice_bank import VoiceBank
        from py_screenalytics.audio.models import (
            VoiceBankConfig,
            VoiceBankEntry,
            VoiceCluster,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = VoiceBankConfig(data_dir=tmpdir, auto_create_unlabeled=True)
            bank = VoiceBank(config)

            # Create a voice bank entry with a known embedding
            base_embedding = np.random.randn(256)
            base_embedding = base_embedding / np.linalg.norm(base_embedding)

            entry = VoiceBankEntry(
                voice_bank_id="voice_lisa_barlow",
                show_id="rhoslc",
                cast_id="lisa_barlow",
                display_name="Lisa Barlow",
                embeddings=[base_embedding.tolist()],
                is_labeled=True,
            )
            bank.add_entry(entry, "rhoslc")

            # Create cluster with very different embedding
            different_embedding = np.random.randn(256)
            different_embedding = different_embedding / np.linalg.norm(different_embedding)

            cluster = VoiceCluster(
                voice_cluster_id="VC_02",
                segments=[],
                total_duration=15.0,
                segment_count=3,
                centroid=different_embedding.tolist(),
            )

            # Match should fail and create unlabeled
            result = bank.match_cluster("rhoslc", cluster, threshold=0.78)

            assert result.voice_bank_id.startswith("voice_unlabeled")
            assert result.speaker_id.startswith("SPK_UNLABELED")
            assert "Unlabeled Voice" in result.speaker_display_name
            assert result.similarity is None
            assert result.is_new_entry is True

    def test_voice_bank_match_no_centroid(self):
        """Cluster without centroid returns unlabeled match."""
        from py_screenalytics.audio.voice_bank import VoiceBank
        from py_screenalytics.audio.models import VoiceBankConfig, VoiceCluster

        with tempfile.TemporaryDirectory() as tmpdir:
            config = VoiceBankConfig(data_dir=tmpdir, auto_create_unlabeled=True)
            bank = VoiceBank(config)

            cluster = VoiceCluster(
                voice_cluster_id="VC_01",
                segments=[],
                total_duration=10.0,
                segment_count=2,
                centroid=None,  # No centroid
            )

            result = bank.match_cluster("rhoslc", cluster, threshold=0.78)

            assert result.voice_bank_id.startswith("voice_unlabeled")


class TestVoiceBankMatchAll:
    """Tests for matching all clusters to voice bank."""

    def test_match_voice_clusters_to_bank(self):
        """All clusters get mapped with required fields."""
        from py_screenalytics.audio.voice_bank import (
            VoiceBank,
            match_voice_clusters_to_bank,
        )
        from py_screenalytics.audio.models import (
            VoiceBankConfig,
            VoiceBankEntry,
            VoiceCluster,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = VoiceBankConfig(data_dir=tmpdir, auto_create_unlabeled=True)
            bank = VoiceBank(config)

            # Add a known voice
            base_embedding = np.random.randn(256)
            base_embedding = base_embedding / np.linalg.norm(base_embedding)

            entry = VoiceBankEntry(
                voice_bank_id="voice_lisa_barlow",
                show_id="rhoslc",
                cast_id="lisa_barlow",
                display_name="Lisa Barlow",
                embeddings=[base_embedding.tolist()],
                is_labeled=True,
            )
            bank.add_entry(entry, "rhoslc")

            # Create clusters - one matching, one not
            similar_embedding = base_embedding + np.random.randn(256) * 0.05
            similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)

            different_embedding = np.random.randn(256)
            different_embedding = different_embedding / np.linalg.norm(different_embedding)

            clusters = [
                VoiceCluster(
                    voice_cluster_id="VC_01",
                    segments=[],
                    total_duration=30.0,
                    segment_count=5,
                    centroid=similar_embedding.tolist(),
                ),
                VoiceCluster(
                    voice_cluster_id="VC_02",
                    segments=[],
                    total_duration=15.0,
                    segment_count=3,
                    centroid=different_embedding.tolist(),
                ),
            ]

            output_path = Path(tmpdir) / "voice_mapping.json"

            results = match_voice_clusters_to_bank(
                "rhoslc",
                clusters,
                output_path,
                config,
                similarity_threshold=0.78,
                overwrite=True,
            )

            # Should have mappings for both clusters
            assert len(results) == 2

            # Verify each result has required fields
            for result in results:
                assert result.voice_cluster_id
                assert result.voice_bank_id
                assert result.speaker_id
                assert result.speaker_display_name

            # Verify file was created
            assert output_path.exists()

            with output_path.open("r") as f:
                data = json.load(f)

            assert len(data) == 2
            assert all("voice_cluster_id" in d for d in data)
            assert all("voice_bank_id" in d for d in data)
            assert all("speaker_id" in d for d in data)
            assert all("speaker_display_name" in d for d in data)

    def test_create_voice_bank_entry_from_cast(self):
        """Helper creates properly formatted entry."""
        from py_screenalytics.audio.voice_bank import create_voice_bank_entry_from_cast

        entry = create_voice_bank_entry_from_cast(
            show_id="rhoslc",
            cast_id="lisa_barlow",
            display_name="Lisa Barlow",
            embeddings=[[0.1] * 256],
            metadata={"season": 5},
        )

        assert entry.voice_bank_id == "voice_lisa_barlow"
        assert entry.show_id == "rhoslc"
        assert entry.cast_id == "lisa_barlow"
        assert entry.display_name == "Lisa Barlow"
        assert entry.is_labeled is True
        assert entry.metadata == {"season": 5}
