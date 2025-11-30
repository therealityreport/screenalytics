"""Tests for voice clustering.

Verifies:
1. Clusters are formed correctly from diarization segments
2. audio_voice_clusters.json has expected structure
3. Cluster IDs are stable (sorted by duration)
4. Segment-to-cluster lookup works
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


class TestVoiceClustering:
    """Tests for py_screenalytics.audio.voice_clusters module."""

    def test_clusters_from_diarization_labels(self):
        """Clusters created from diarization labels without embeddings."""
        from py_screenalytics.audio.voice_clusters import _clusters_from_diarization_labels
        from py_screanalytics.audio.models import DiarizationSegment

        segments = [
            DiarizationSegment(start=0.0, end=5.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=5.5, end=10.0, speaker="SPEAKER_01"),
            DiarizationSegment(start=10.5, end=15.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=15.5, end=20.0, speaker="SPEAKER_01"),
        ]

        clusters = _clusters_from_diarization_labels(segments)

        assert len(clusters) == 2
        assert clusters[0].voice_cluster_id == "VC_01"
        assert clusters[1].voice_cluster_id == "VC_02"

        # Check segment counts
        speaker_00_cluster = next(c for c in clusters if any(s.diar_speaker == "SPEAKER_00" for s in c.segments))
        speaker_01_cluster = next(c for c in clusters if any(s.diar_speaker == "SPEAKER_01" for s in c.segments))

        assert speaker_00_cluster.segment_count == 2
        assert speaker_01_cluster.segment_count == 2

    def test_clusters_saved_to_json(self):
        """Clusters are saved correctly to JSON."""
        from py_screenalytics.audio.voice_clusters import _save_voice_clusters, _load_voice_clusters
        from py_screenalytics.audio.models import VoiceCluster, VoiceClusterSegment

        clusters = [
            VoiceCluster(
                voice_cluster_id="VC_01",
                segments=[
                    VoiceClusterSegment(start=0.0, end=5.0, diar_speaker="SPEAKER_00"),
                    VoiceClusterSegment(start=10.0, end=15.0, diar_speaker="SPEAKER_00"),
                ],
                total_duration=10.0,
                segment_count=2,
                centroid=[0.1] * 256,
            ),
            VoiceCluster(
                voice_cluster_id="VC_02",
                segments=[
                    VoiceClusterSegment(start=5.0, end=10.0, diar_speaker="SPEAKER_01"),
                ],
                total_duration=5.0,
                segment_count=1,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "voice_clusters.json"

            _save_voice_clusters(clusters, output_path)

            assert output_path.exists()

            # Verify JSON structure
            with output_path.open("r") as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]["voice_cluster_id"] == "VC_01"
            assert data[0]["segment_count"] == 2
            assert data[0]["total_duration"] == 10.0
            assert len(data[0]["segments"]) == 2
            assert data[0]["centroid"] is not None

            # Load back
            loaded = _load_voice_clusters(output_path)

            assert len(loaded) == 2
            assert loaded[0].voice_cluster_id == "VC_01"
            assert loaded[0].segment_count == 2

    def test_get_cluster_for_timestamp(self):
        """Cluster lookup by timestamp works."""
        from py_screenalytics.audio.voice_clusters import get_cluster_for_timestamp
        from py_screenalytics.audio.models import VoiceCluster, VoiceClusterSegment

        clusters = [
            VoiceCluster(
                voice_cluster_id="VC_01",
                segments=[
                    VoiceClusterSegment(start=0.0, end=5.0, diar_speaker="SPEAKER_00"),
                    VoiceClusterSegment(start=10.0, end=15.0, diar_speaker="SPEAKER_00"),
                ],
                total_duration=10.0,
                segment_count=2,
            ),
            VoiceCluster(
                voice_cluster_id="VC_02",
                segments=[
                    VoiceClusterSegment(start=5.0, end=10.0, diar_speaker="SPEAKER_01"),
                ],
                total_duration=5.0,
                segment_count=1,
            ),
        ]

        # Timestamp in VC_01 first segment
        cluster = get_cluster_for_timestamp(clusters, 2.5)
        assert cluster is not None
        assert cluster.voice_cluster_id == "VC_01"

        # Timestamp in VC_02
        cluster = get_cluster_for_timestamp(clusters, 7.0)
        assert cluster is not None
        assert cluster.voice_cluster_id == "VC_02"

        # Timestamp in VC_01 second segment
        cluster = get_cluster_for_timestamp(clusters, 12.0)
        assert cluster is not None
        assert cluster.voice_cluster_id == "VC_01"

        # Timestamp outside all segments
        cluster = get_cluster_for_timestamp(clusters, 20.0)
        assert cluster is None

    def test_cluster_ids_sorted_by_duration(self):
        """Cluster IDs are assigned by descending duration."""
        from py_screanalytics.audio.voice_clusters import _assign_cluster_ids
        from py_screenalytics.audio.models import VoiceCluster

        clusters = [
            VoiceCluster(
                voice_cluster_id="",
                segments=[],
                total_duration=5.0,
                segment_count=1,
            ),
            VoiceCluster(
                voice_cluster_id="",
                segments=[],
                total_duration=20.0,
                segment_count=4,
            ),
            VoiceCluster(
                voice_cluster_id="",
                segments=[],
                total_duration=10.0,
                segment_count=2,
            ),
        ]

        sorted_clusters = _assign_cluster_ids(clusters)

        # VC_01 should have the longest duration
        assert sorted_clusters[0].voice_cluster_id == "VC_01"
        assert sorted_clusters[0].total_duration == 20.0

        # VC_02 should have second longest
        assert sorted_clusters[1].voice_cluster_id == "VC_02"
        assert sorted_clusters[1].total_duration == 10.0

        # VC_03 should have shortest
        assert sorted_clusters[2].voice_cluster_id == "VC_03"
        assert sorted_clusters[2].total_duration == 5.0

    def test_compute_cluster_centroid(self):
        """Centroid computation is normalized."""
        from py_screanalytics.audio.voice_clusters import compute_cluster_centroid

        embeddings = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        centroid = compute_cluster_centroid(embeddings, method="mean")

        # Mean of unit vectors along each axis
        assert centroid.shape == (3,)

        # Should be normalized
        norm = np.linalg.norm(centroid)
        assert abs(norm - 1.0) < 0.01

    def test_compute_cluster_centroid_median(self):
        """Median centroid is also normalized."""
        from py_screanalytics.audio.voice_clusters import compute_cluster_centroid

        embeddings = [
            np.array([1.0, 0.0]),
            np.array([0.5, 0.5]),
            np.array([0.0, 1.0]),
        ]

        centroid = compute_cluster_centroid(embeddings, method="median")

        norm = np.linalg.norm(centroid)
        assert abs(norm - 1.0) < 0.01


class TestVoiceClusteringIntegration:
    """Integration tests for voice clustering with mocked embeddings."""

    @patch("py_screanalytics.audio.voice_clusters.extract_speaker_embeddings")
    def test_cluster_episode_voices_with_embeddings(self, mock_extract):
        """Clustering with embeddings merges similar speakers."""
        from py_screenalytics.audio.voice_clusters import cluster_episode_voices
        from py_screanalytics.audio.models import DiarizationSegment, VoiceClusteringConfig

        segments = [
            DiarizationSegment(start=0.0, end=5.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=5.0, end=10.0, speaker="SPEAKER_01"),
            DiarizationSegment(start=10.0, end=15.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=15.0, end=20.0, speaker="SPEAKER_02"),
        ]

        # Mock embeddings: SPEAKER_00 and SPEAKER_02 are similar
        base_embedding = np.random.randn(256)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)

        similar_embedding = base_embedding + np.random.randn(256) * 0.1
        similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)

        different_embedding = np.random.randn(256)
        different_embedding = different_embedding / np.linalg.norm(different_embedding)

        mock_extract.return_value = [
            (segments[0], base_embedding),
            (segments[1], different_embedding),
            (segments[2], base_embedding),
            (segments[3], similar_embedding),
        ]

        config = VoiceClusteringConfig(
            similarity_threshold=0.90,  # High threshold - only very similar merge
            min_segments_per_cluster=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "voice_clusters.json"

            clusters = cluster_episode_voices(
                Path("/fake/audio.wav"),
                segments,
                output_path,
                config,
                overwrite=True,
            )

            # With high threshold, we should have 3 clusters
            # (SPEAKER_00, SPEAKER_01, SPEAKER_02 stay separate)
            assert len(clusters) >= 2

            # Verify file was created
            assert output_path.exists()
