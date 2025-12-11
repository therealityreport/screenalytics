"""
Tests for face alignment evaluation harness.

Tests the metric computation functions and CLI behavior using synthetic fixtures.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

# Import functions from eval script
import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.experiments.face_alignment_eval import (
    EpisodeMetrics,
    ComparisonResult,
    TrackMetrics,
    compute_embedding_jitter,
    compute_id_switch_rate,
    compute_screen_time,
    load_tracks,
    load_identities,
    load_embeddings,
    run_evaluation,
    run_comparison,
    save_results,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_manifest_dir():
    """Create temporary manifest directory with synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_dir = Path(tmpdir)

        # Create tracks.jsonl
        tracks = [
            {
                "track_id": 1,
                "frame_count": 100,
                "duration": 4.0,
                "detections": [
                    {"frame_idx": i, "timestamp": i / 24.0}
                    for i in range(0, 100, 1)
                ]
            },
            {
                "track_id": 2,
                "frame_count": 50,
                "duration": 2.0,
                "detections": [
                    {"frame_idx": i, "timestamp": i / 24.0}
                    for i in range(100, 150, 1)
                ]
            },
            {
                "track_id": 3,
                "frame_count": 30,
                "duration": 1.25,
                "detections": [
                    {"frame_idx": i, "timestamp": i / 24.0}
                    for i in range(200, 230, 1)
                ]
            },
        ]

        with open(manifest_dir / "tracks.jsonl", "w") as f:
            for t in tracks:
                f.write(json.dumps(t) + "\n")

        # Create identities.json
        identities = {
            "identities": [
                {
                    "identity_id": "cast_1",
                    "tracks": [{"track_id": 1, "duration": 4.0}]
                },
                {
                    "identity_id": "cast_2",
                    "tracks": [
                        {"track_id": 2, "duration": 2.0},
                        {"track_id": 3, "duration": 1.25}
                    ]
                },
            ]
        }

        with open(manifest_dir / "identities.json", "w") as f:
            json.dump(identities, f)

        yield manifest_dir


@pytest.fixture
def temp_manifest_with_embeddings(temp_manifest_dir):
    """Add embeddings to manifest directory."""
    manifest_dir = temp_manifest_dir

    # Create synthetic embeddings (3 tracks, 10 samples each)
    n_samples = 30
    embedding_dim = 512

    # Create embeddings with controlled jitter
    np.random.seed(42)
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    np.save(manifest_dir / "face_embeddings.npy", embeddings)

    # Create metadata
    meta = []
    for i in range(n_samples):
        track_id = (i // 10) + 1  # 10 samples per track, tracks 1-3
        frame_idx = (i % 10) * 10 + (track_id - 1) * 100
        meta.append({
            "track_id": track_id,
            "frame_idx": frame_idx,
        })

    with open(manifest_dir / "face_embeddings_meta.json", "w") as f:
        json.dump(meta, f)

    return manifest_dir


@pytest.fixture
def tracks_with_gaps():
    """Tracks with gaps indicating potential ID switches."""
    return [
        {
            "track_id": 1,
            "duration": 10.0,
            "detections": [
                {"frame_idx": 0},
                {"frame_idx": 24},  # 1 second later
                {"frame_idx": 120},  # 4 seconds later (gap > 2s = switch)
                {"frame_idx": 144},
            ]
        },
        {
            "track_id": 2,
            "duration": 5.0,
            "detections": [
                {"frame_idx": 200},
                {"frame_idx": 224},
                {"frame_idx": 248},
            ]
        },
    ]


# ============================================================================
# Tests: Metric Computation
# ============================================================================

class TestComputeEmbeddingJitter:
    """Tests for embedding jitter computation."""

    def test_identical_embeddings_zero_jitter(self):
        """Identical consecutive embeddings should have zero jitter."""
        n_samples = 10
        embedding_dim = 512

        # Create identical embeddings
        base_emb = np.random.randn(embedding_dim).astype(np.float32)
        base_emb = base_emb / np.linalg.norm(base_emb)
        embeddings = np.tile(base_emb, (n_samples, 1))

        meta = [{"track_id": 1, "frame_idx": i * 10} for i in range(n_samples)]
        tracks = [{"track_id": 1}]

        jitter = compute_embedding_jitter(embeddings, meta, tracks)

        assert 1 in jitter
        assert jitter[1] < 0.001  # Should be ~0

    def test_orthogonal_embeddings_max_jitter(self):
        """Orthogonal consecutive embeddings should have high jitter."""
        embedding_dim = 512

        # Create orthogonal embeddings (alternating)
        emb1 = np.zeros(embedding_dim, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(embedding_dim, dtype=np.float32)
        emb2[1] = 1.0

        embeddings = np.array([emb1, emb2, emb1, emb2])
        meta = [{"track_id": 1, "frame_idx": i * 10} for i in range(4)]
        tracks = [{"track_id": 1}]

        jitter = compute_embedding_jitter(embeddings, meta, tracks)

        assert 1 in jitter
        assert jitter[1] > 0.9  # Orthogonal = cosine distance ~1.0

    def test_single_embedding_zero_jitter(self):
        """Single embedding in track should have zero jitter."""
        embeddings = np.random.randn(1, 512).astype(np.float32)
        meta = [{"track_id": 1, "frame_idx": 0}]
        tracks = [{"track_id": 1}]

        jitter = compute_embedding_jitter(embeddings, meta, tracks)

        assert jitter.get(1, 0.0) == 0.0

    def test_empty_embeddings(self):
        """Empty embeddings should return empty dict."""
        jitter = compute_embedding_jitter(None, [], [])
        assert jitter == {}


class TestComputeIdSwitchRate:
    """Tests for ID switch rate computation."""

    def test_continuous_tracks_no_switches(self):
        """Tracks with no gaps should have zero switches."""
        tracks = [
            {
                "track_id": 1,
                "duration": 4.0,
                "detections": [{"frame_idx": i} for i in range(100)]
            }
        ]

        count, rate = compute_id_switch_rate(tracks, fps=24.0)

        assert count == 0
        assert rate == 0.0

    def test_tracks_with_gaps(self, tracks_with_gaps):
        """Tracks with large gaps should count as switches."""
        count, rate = compute_id_switch_rate(tracks_with_gaps, fps=24.0)

        # Track 1 has one gap > 2 seconds
        assert count == 1
        assert rate > 0

    def test_empty_tracks(self):
        """Empty tracks list should return zeros."""
        count, rate = compute_id_switch_rate([], fps=24.0)

        assert count == 0
        assert rate == 0.0


class TestComputeScreenTime:
    """Tests for screen time computation."""

    def test_screen_time_per_identity(self):
        """Should compute correct screen time per identity."""
        identities = {
            "identities": [
                {
                    "identity_id": "person_a",
                    "tracks": [
                        {"duration": 10.0},
                        {"duration": 5.0},
                    ]
                },
                {
                    "identity_id": "person_b",
                    "tracks": [
                        {"duration": 20.0},
                    ]
                },
            ]
        }

        screen_time = compute_screen_time(identities)

        assert screen_time["person_a"] == 15.0
        assert screen_time["person_b"] == 20.0

    def test_empty_identities(self):
        """Empty identities should return empty dict."""
        screen_time = compute_screen_time({})
        assert screen_time == {}


# ============================================================================
# Tests: Data Loading
# ============================================================================

class TestLoadTracks:
    """Tests for track loading."""

    def test_load_tracks_from_jsonl(self, temp_manifest_dir):
        """Should load tracks from tracks.jsonl."""
        tracks = load_tracks(temp_manifest_dir)

        assert len(tracks) == 3
        assert tracks[0]["track_id"] == 1
        assert tracks[1]["track_id"] == 2

    def test_load_tracks_missing_file(self):
        """Should return empty list for missing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracks = load_tracks(Path(tmpdir))
            assert tracks == []


class TestLoadIdentities:
    """Tests for identity loading."""

    def test_load_identities_json(self, temp_manifest_dir):
        """Should load identities from identities.json."""
        identities = load_identities(temp_manifest_dir)

        assert "identities" in identities
        assert len(identities["identities"]) == 2

    def test_load_identities_missing_file(self):
        """Should return empty dict for missing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            identities = load_identities(Path(tmpdir))
            assert identities == {}


class TestLoadEmbeddings:
    """Tests for embedding loading."""

    def test_load_embeddings_with_meta(self, temp_manifest_with_embeddings):
        """Should load embeddings and metadata."""
        embeddings, meta = load_embeddings(temp_manifest_with_embeddings)

        assert embeddings is not None
        assert embeddings.shape == (30, 512)
        assert len(meta) == 30

    def test_load_embeddings_missing_file(self):
        """Should return None for missing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings, meta = load_embeddings(Path(tmpdir))
            assert embeddings is None
            assert meta == []


# ============================================================================
# Tests: Evaluation Pipeline
# ============================================================================

class TestRunEvaluation:
    """Tests for run_evaluation function."""

    def test_run_evaluation_basic(self, temp_manifest_dir):
        """Should compute metrics for episode."""
        metrics = run_evaluation(
            episode_id="test_ep",
            manifest_dir=temp_manifest_dir,
            alignment_enabled=True,
        )

        assert metrics.episode_id == "test_ep"
        assert metrics.alignment_enabled is True
        assert metrics.num_tracks == 3
        assert metrics.avg_track_length > 0

    def test_run_evaluation_screen_time(self, temp_manifest_dir):
        """Should compute screen time per identity."""
        metrics = run_evaluation(
            episode_id="test_ep",
            manifest_dir=temp_manifest_dir,
            alignment_enabled=True,
        )

        assert "cast_1" in metrics.screen_time_per_identity
        assert "cast_2" in metrics.screen_time_per_identity
        assert metrics.screen_time_per_identity["cast_1"] == 4.0
        assert metrics.screen_time_per_identity["cast_2"] == 3.25


class TestComparisonResult:
    """Tests for ComparisonResult."""

    def test_compute_deltas(self):
        """Should compute correct deltas."""
        baseline = EpisodeMetrics(
            episode_id="test",
            alignment_enabled=False,
            num_tracks=100,
            avg_track_length=10.0,
            mean_embedding_jitter=0.05,
            id_switch_rate_per_minute=2.0,
        )

        aligned = EpisodeMetrics(
            episode_id="test",
            alignment_enabled=True,
            num_tracks=95,
            avg_track_length=12.0,
            mean_embedding_jitter=0.03,
            id_switch_rate_per_minute=1.5,
        )

        result = ComparisonResult(
            episode_id="test",
            baseline=baseline,
            aligned=aligned,
        )
        result.compute_deltas()

        assert result.delta_num_tracks == -5
        assert result.delta_avg_track_length == 2.0
        assert result.delta_embedding_jitter == pytest.approx(-0.02)
        assert result.delta_id_switch_rate == -0.5
        assert result.jitter_improved is True
        assert result.track_length_improved is True

    def test_to_dict_structure(self):
        """Should serialize to dict with expected keys."""
        baseline = EpisodeMetrics(episode_id="test", alignment_enabled=False)
        aligned = EpisodeMetrics(episode_id="test", alignment_enabled=True)

        result = ComparisonResult(
            episode_id="test",
            baseline=baseline,
            aligned=aligned,
        )
        result.compute_deltas()

        d = result.to_dict()

        assert "episode_id" in d
        assert "baseline" in d
        assert "aligned" in d
        assert "deltas" in d
        assert "improvements" in d


# ============================================================================
# Tests: Output
# ============================================================================

class TestSaveResults:
    """Tests for saving results."""

    def test_save_results_creates_file(self):
        """Should create JSON file with results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "results.json"

            results = {
                "experiment": "test",
                "results": [{"episode_id": "ep1"}]
            }

            save_results(results, output_path)

            assert output_path.exists()
            with open(output_path) as f:
                loaded = json.load(f)
            assert loaded["experiment"] == "test"


class TestEpisodeMetricsToDict:
    """Tests for EpisodeMetrics serialization."""

    def test_to_dict_structure(self):
        """Should serialize to dict with expected structure."""
        metrics = EpisodeMetrics(
            episode_id="test_ep",
            alignment_enabled=True,
            num_tracks=50,
            avg_track_length=25.5,
            mean_embedding_jitter=0.042,
            id_switch_rate_per_minute=1.5,
            cluster_count=10,
            singleton_count=3,
        )

        d = metrics.to_dict()

        assert d["episode_id"] == "test_ep"
        assert d["alignment_enabled"] is True
        assert d["track_metrics"]["num_tracks"] == 50
        assert d["track_metrics"]["avg_track_length"] == 25.5
        assert d["embedding_quality"]["mean_embedding_jitter"] == 0.042
        assert d["id_switch_metrics"]["id_switch_rate_per_minute"] == 1.5
        assert d["clustering_metrics"]["cluster_count"] == 10
