"""Test cluster cohesion computation with synthetic embeddings."""

import numpy as np


def test_tight_cluster_high_cohesion():
    """A tight cluster with similar embeddings should have high cohesion (>0.7)."""
    from apps.api.services.track_reps import cosine_similarity, l2_normalize

    # Create 5 very similar embeddings (small random noise)
    np.random.seed(42)
    base_vector = np.random.randn(512)
    base_vector = l2_normalize(base_vector)

    embeddings = []
    for i in range(5):
        # Add small noise
        noisy = base_vector + np.random.randn(512) * 0.05
        embeddings.append(l2_normalize(noisy))

    # Compute cluster centroid
    cluster_centroid = l2_normalize(np.mean(embeddings, axis=0))

    # Compute cohesion as mean similarity to centroid
    similarities = [cosine_similarity(emb, cluster_centroid) for emb in embeddings]
    cohesion = float(np.mean(similarities))

    assert (
        cohesion > 0.7
    ), f"Tight cluster should have cohesion > 0.7, got {cohesion:.3f}"
    assert cohesion <= 1.0, f"Cohesion should be <= 1.0, got {cohesion:.3f}"


def test_loose_cluster_low_cohesion():
    """A loose cluster with dissimilar embeddings should have low cohesion (<0.4)."""
    from apps.api.services.track_reps import cosine_similarity, l2_normalize

    # Create 5 very different embeddings (orthogonal directions)
    np.random.seed(123)
    embeddings = []
    for i in range(5):
        vec = np.random.randn(512)
        # Make them more different by scaling different dimensions
        vec[i * 100 : (i + 1) * 100] *= 10.0
        embeddings.append(l2_normalize(vec))

    # Compute cluster centroid
    cluster_centroid = l2_normalize(np.mean(embeddings, axis=0))

    # Compute cohesion
    similarities = [cosine_similarity(emb, cluster_centroid) for emb in embeddings]
    cohesion = float(np.mean(similarities))

    assert (
        cohesion < 0.4
    ), f"Loose cluster should have cohesion < 0.4, got {cohesion:.3f}"
    assert cohesion >= 0.0, f"Cohesion should be >= 0.0, got {cohesion:.3f}"


def test_two_separated_clusters():
    """Two well-separated clusters should each have high internal cohesion."""
    from apps.api.services.track_reps import cosine_similarity, l2_normalize

    np.random.seed(456)

    # Cluster 1: centered around [1, 0, 0, ...]
    cluster1_base = np.zeros(512)
    cluster1_base[0] = 10.0
    cluster1_base = l2_normalize(cluster1_base)
    cluster1_embeddings = []
    for i in range(5):
        noisy = cluster1_base + np.random.randn(512) * 0.05
        cluster1_embeddings.append(l2_normalize(noisy))

    # Cluster 2: centered around [0, 1, 0, ...]
    cluster2_base = np.zeros(512)
    cluster2_base[1] = 10.0
    cluster2_base = l2_normalize(cluster2_base)
    cluster2_embeddings = []
    for i in range(5):
        noisy = cluster2_base + np.random.randn(512) * 0.05
        cluster2_embeddings.append(l2_normalize(noisy))

    # Compute cohesion for each cluster
    def compute_cohesion(embeddings):
        centroid = l2_normalize(np.mean(embeddings, axis=0))
        similarities = [cosine_similarity(emb, centroid) for emb in embeddings]
        return float(np.mean(similarities))

    cohesion1 = compute_cohesion(cluster1_embeddings)
    cohesion2 = compute_cohesion(cluster2_embeddings)

    # Both clusters should have high internal cohesion
    assert cohesion1 > 0.7, f"Cluster 1 should have high cohesion, got {cohesion1:.3f}"
    assert cohesion2 > 0.7, f"Cluster 2 should have high cohesion, got {cohesion2:.3f}"

    # Cross-cluster similarity should be low
    cluster1_centroid = l2_normalize(np.mean(cluster1_embeddings, axis=0))
    cluster2_centroid = l2_normalize(np.mean(cluster2_embeddings, axis=0))
    cross_similarity = cosine_similarity(cluster1_centroid, cluster2_centroid)

    assert (
        cross_similarity < 0.3
    ), f"Separated clusters should have low cross-similarity, got {cross_similarity:.3f}"


def test_cohesion_single_embedding():
    """A cluster with a single embedding should have cohesion = 1.0."""
    from apps.api.services.track_reps import cosine_similarity, l2_normalize

    np.random.seed(789)
    embedding = l2_normalize(np.random.randn(512))

    # Centroid is the same as the embedding
    centroid = embedding

    similarity = cosine_similarity(embedding, centroid)
    assert (
        abs(similarity - 1.0) < 0.001
    ), f"Single embedding should have similarity 1.0 to itself, got {similarity:.3f}"
