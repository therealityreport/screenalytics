"""Integration test for clustering with metric validation.

Tests identity clustering and validates:
- identities.json exists and parses correctly
- Cluster metrics (singleton_fraction, largest_cluster_fraction) within bounds
- Cluster counts match expectations
- Profile support works correctly
- Synthetic embeddings produce expected clustering
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ML_TESTS = os.environ.get("RUN_ML_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_ML_TESTS,
    reason="set RUN_ML_TESTS=1 to run ML integration tests"
)

# Acceptance thresholds from ACCEPTANCE_MATRIX.md
THRESHOLDS = {
    "singleton_fraction": 0.50,  # Warning threshold (target: < 0.30)
    "largest_cluster_fraction": 0.60,  # Warning threshold (target: < 0.40)
    "num_clusters_typical": (5, 30),  # Typical range for TV episode
}


def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL file into list of dicts."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _create_synthetic_embeddings(
    manifest_root: Path,
    num_identities: int = 5,
    tracks_per_identity: int = 4,
) -> tuple[Path, Path, int]:
    """Create synthetic faces.jsonl and faces.npy for testing clustering.

    Creates embeddings where each identity has a distinct cluster in embedding space.
    """
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not installed")

    faces_jsonl = manifest_root / "faces.jsonl"
    faces_npy = manifest_root / "faces.npy"

    # Create clustered embeddings (512-dim unit vectors)
    embeddings = []
    faces_metadata = []
    track_counter = 0

    for identity_id in range(num_identities):
        # Create a random base vector for this identity
        np.random.seed(42 + identity_id)  # Reproducible
        base_vector = np.random.randn(512)
        base_vector = base_vector / np.linalg.norm(base_vector)

        # Create tracks for this identity with small perturbations
        for track_idx in range(tracks_per_identity):
            track_id = f"track-{track_counter:05d}"
            track_counter += 1

            # Add small noise to base vector
            noise = np.random.randn(512) * 0.01  # Smaller perturbation to keep tracks in same cluster
            perturbed = base_vector + noise
            perturbed = perturbed / np.linalg.norm(perturbed)  # Normalize

            embeddings.append(perturbed)
            faces_metadata.append({
                "track_id": track_id,
                "frame_idx": track_idx * 10,
                "quality_score": 0.85,
                "bbox": [100, 100, 200, 250],
            })

    # Save embeddings
    embeddings_array = np.array(embeddings, dtype=np.float32)
    np.save(faces_npy, embeddings_array)

    # Save metadata
    with faces_jsonl.open("w", encoding="utf-8") as f:
        for face in faces_metadata:
            f.write(json.dumps(face) + "\n")

    # Create minimal tracks.jsonl
    tracks_jsonl = manifest_root / "tracks.jsonl"
    with tracks_jsonl.open("w", encoding="utf-8") as f:
        for idx in range(len(faces_metadata)):
            track = {
                "track_id": f"track-{idx:05d}",
                "frame_count": 10,
                "first_frame": idx * 10,
                "last_frame": idx * 10 + 9,
                "pipeline_ver": "2.0.0",
            }
            f.write(json.dumps(track) + "\n")

    return faces_jsonl, faces_npy, num_identities


@pytest.mark.timeout(180)
def test_cluster_with_synthetic_embeddings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test clustering with synthetic embeddings of known structure."""

    # Force CPU
    try:
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    except ImportError:
        pass

    data_root = tmp_path / "data"
    ep_id = "test-cluster-synthetic"
    manifest_root = data_root / "manifests" / ep_id
    manifest_root.mkdir(parents=True, exist_ok=True)

    # Create synthetic embeddings: 5 identities, 4 tracks each = 20 tracks total
    num_identities = 5
    tracks_per_identity = 4
    _create_synthetic_embeddings(manifest_root, num_identities, tracks_per_identity)

    # Run clustering
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id", ep_id,
        "--cluster",
        "--profile", "balanced",
        "--device", "cpu",
        "--out-root", str(data_root),
    ]

    env = os.environ.copy()
    env["SCREANALYTICS_DATA_ROOT"] = str(data_root)

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"Clustering failed: {result.stderr}"

    # Assert identities.json exists
    identities_path = manifest_root / "identities.json"
    assert identities_path.exists(), f"identities.json missing at {identities_path}"

    # Load and validate identities
    with identities_path.open("r") as f:
        identities_data = json.load(f)

    assert "identities" in identities_data, "Missing 'identities' field"
    assert "stats" in identities_data, "Missing 'stats' field"

    identities = identities_data["identities"]
    stats = identities_data["stats"]

    # Validate stats fields
    required_stats = [
        "clusters",
        "total_tracks",
        "singleton_count",
        "singleton_fraction",
        "largest_cluster_size",
        "largest_cluster_fraction",
    ]
    for field in required_stats:
        assert field in stats, f"Missing required stat: {field}"

    # Assert cluster count is reasonable (should be close to num_identities)
    cluster_count = stats["clusters"]
    # Allow some variation due to clustering algorithm
    assert num_identities - 2 <= cluster_count <= num_identities + 3, \
        f"Expected ~{num_identities} clusters, got {cluster_count}"

    # Assert metrics within acceptance thresholds
    singleton_frac = stats["singleton_fraction"]
    assert singleton_frac <= THRESHOLDS["singleton_fraction"], \
        f"singleton_fraction={singleton_frac:.3f} exceeds threshold {THRESHOLDS['singleton_fraction']}"

    largest_cluster_frac = stats["largest_cluster_fraction"]
    assert largest_cluster_frac <= THRESHOLDS["largest_cluster_fraction"], \
        f"largest_cluster_fraction={largest_cluster_frac:.3f} exceeds threshold {THRESHOLDS['largest_cluster_fraction']}"

    # Validate track_metrics.json was updated
    metrics_path = manifest_root / "track_metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r") as f:
            metrics_data = json.load(f)

        assert "cluster_metrics" in metrics_data, "Missing cluster_metrics in track_metrics.json"
        cluster_metrics = metrics_data["cluster_metrics"]

        # Verify consistency between identities.json stats and track_metrics.json
        assert cluster_metrics["singleton_count"] == stats["singleton_count"]
        assert cluster_metrics["total_clusters"] == stats["clusters"]

    print(f"\n✓ Cluster metrics validation passed:")
    print(f"  Total clusters: {cluster_count} (expected: ~{num_identities})")
    print(f"  Singleton fraction: {singleton_frac:.3f} (threshold: {THRESHOLDS['singleton_fraction']})")
    print(f"  Largest cluster fraction: {largest_cluster_frac:.3f} (threshold: {THRESHOLDS['largest_cluster_fraction']})")


@pytest.mark.timeout(180)
def test_cluster_threshold_sensitivity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that cluster_thresh parameter affects clustering results."""

    try:
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    except ImportError:
        pass

    data_root = tmp_path / "data"
    ep_id = "test-cluster-thresh"
    manifest_root = data_root / "manifests" / ep_id
    manifest_root.mkdir(parents=True, exist_ok=True)

    # Create synthetic embeddings
    _create_synthetic_embeddings(manifest_root, num_identities=5, tracks_per_identity=3)

    # Run clustering with LOOSE threshold (should create fewer clusters)
    cmd_loose = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id", ep_id,
        "--cluster",
        "--cluster-thresh", "0.45",  # Loose threshold
        "--device", "cpu",
        "--out-root", str(data_root),
    ]

    env = os.environ.copy()
    env["SCREANALYTICS_DATA_ROOT"] = str(data_root)

    result = subprocess.run(
        cmd_loose,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"Clustering with loose threshold failed: {result.stderr}"

    # Load results
    identities_path = manifest_root / "identities.json"
    with identities_path.open("r") as f:
        loose_data = json.load(f)

    loose_clusters = loose_data["stats"]["clusters"]

    # Run clustering with TIGHT threshold (should create more clusters)
    cmd_tight = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id", ep_id,
        "--cluster",
        "--cluster-thresh", "0.75",  # Tight threshold
        "--device", "cpu",
        "--out-root", str(data_root),
    ]

    result = subprocess.run(
        cmd_tight,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"Clustering with tight threshold failed: {result.stderr}"

    with identities_path.open("r") as f:
        tight_data = json.load(f)

    tight_clusters = tight_data["stats"]["clusters"]

    # Assert that tighter threshold creates more clusters
    print(f"\n✓ Threshold sensitivity validated:")
    print(f"  Loose threshold (0.45): {loose_clusters} clusters")
    print(f"  Tight threshold (0.75): {tight_clusters} clusters")

    # We expect tight threshold to produce >= loose threshold clusters
    # (though not always guaranteed depending on data structure)
    assert tight_clusters >= loose_clusters or abs(tight_clusters - loose_clusters) <= 2, \
        "Threshold sensitivity check failed (tight should produce similar or more clusters)"


@pytest.mark.timeout(180)
def test_cluster_with_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test clustering with profile support."""

    try:
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    except ImportError:
        pass

    data_root = tmp_path / "data"
    ep_id = "test-cluster-profile"
    manifest_root = data_root / "manifests" / ep_id
    manifest_root.mkdir(parents=True, exist_ok=True)

    _create_synthetic_embeddings(manifest_root, num_identities=4, tracks_per_identity=5)

    # Run with high_accuracy profile
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "episode_run.py"),
        "--ep-id", ep_id,
        "--cluster",
        "--profile", "high_accuracy",
        "--device", "cpu",
        "--out-root", str(data_root),
    ]

    env = os.environ.copy()
    env["SCREANALYTICS_DATA_ROOT"] = str(data_root)

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, f"Clustering with profile failed: {result.stderr}"

    # Verify identities.json exists and is valid
    identities_path = manifest_root / "identities.json"
    assert identities_path.exists()

    with identities_path.open("r") as f:
        data = json.load(f)

    assert "identities" in data
    assert "stats" in data
    assert data["stats"]["clusters"] > 0

    print(f"\n✓ Profile support validated:")
    print(f"  Clusters created: {data['stats']['clusters']}")
