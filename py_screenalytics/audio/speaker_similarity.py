"""Speaker embedding similarity utilities.

Provides functions for computing similarity between speaker embeddings,
useful for cross-episode speaker identification and voice bank matching.

Uses NeMo TitaNet embeddings (192-dimensional vectors).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


def speaker_similarity(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
) -> float:
    """Compute cosine similarity between two speaker embeddings.

    Args:
        embedding_a: NeMo TitaNet embedding (192-dim)
        embedding_b: NeMo TitaNet embedding (192-dim)

    Returns:
        Cosine similarity score [-1, 1], higher = more similar.
        Returns 0.0 if either embedding is zero vector.
    """
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(embedding_a, embedding_b) / (norm_a * norm_b))


def find_similar_speakers(
    query_embedding: np.ndarray,
    speaker_embeddings: Dict[str, np.ndarray],
    threshold: float = 0.7,
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Find speakers similar to query embedding.

    Args:
        query_embedding: NeMo TitaNet embedding to match
        speaker_embeddings: Dict mapping speaker_id to embedding
        threshold: Minimum similarity to include (default 0.7)
        top_k: Maximum number of results (None = no limit)

    Returns:
        List of (speaker_id, similarity) tuples, sorted by similarity descending.
        Only includes speakers with similarity >= threshold.
    """
    if not speaker_embeddings:
        return []

    results = []
    for speaker_id, embedding in speaker_embeddings.items():
        sim = speaker_similarity(query_embedding, embedding)
        if sim >= threshold:
            results.append((speaker_id, sim))

    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        results = results[:top_k]

    return results


def best_match(
    query_embedding: np.ndarray,
    speaker_embeddings: Dict[str, np.ndarray],
    threshold: float = 0.6,
) -> Optional[Tuple[str, float]]:
    """Find the best matching speaker for a query embedding.

    Args:
        query_embedding: NeMo TitaNet embedding to match
        speaker_embeddings: Dict mapping speaker_id to embedding
        threshold: Minimum similarity to consider a match (default 0.6)

    Returns:
        Tuple of (speaker_id, similarity) for best match, or None if no match
        meets the threshold.
    """
    matches = find_similar_speakers(
        query_embedding,
        speaker_embeddings,
        threshold=threshold,
        top_k=1,
    )

    if matches:
        return matches[0]
    return None


def compute_centroid(embeddings: List[np.ndarray], method: str = "mean") -> np.ndarray:
    """Compute centroid embedding from a list of embeddings.

    Args:
        embeddings: List of NeMo TitaNet embeddings
        method: Aggregation method - "mean" or "median"

    Returns:
        Centroid embedding (192-dim). Returns zero vector if input is empty.
    """
    if not embeddings:
        return np.zeros(192)

    stacked = np.stack(embeddings)

    if method == "median":
        centroid = np.median(stacked, axis=0)
    else:  # default to mean
        centroid = np.mean(stacked, axis=0)

    return centroid


def compute_cluster_similarity(
    cluster_a_embeddings: List[np.ndarray],
    cluster_b_embeddings: List[np.ndarray],
    method: str = "centroid",
) -> float:
    """Compute similarity between two clusters of embeddings.

    Args:
        cluster_a_embeddings: Embeddings from first cluster
        cluster_b_embeddings: Embeddings from second cluster
        method: Comparison method:
            - "centroid": Compare cluster centroids (default)
            - "mean_pairwise": Average of all pairwise similarities
            - "max_pairwise": Maximum pairwise similarity

    Returns:
        Similarity score. Returns 0.0 if either cluster is empty.
    """
    if not cluster_a_embeddings or not cluster_b_embeddings:
        return 0.0

    if method == "centroid":
        centroid_a = compute_centroid(cluster_a_embeddings)
        centroid_b = compute_centroid(cluster_b_embeddings)
        return speaker_similarity(centroid_a, centroid_b)

    elif method == "mean_pairwise":
        similarities = []
        for emb_a in cluster_a_embeddings:
            for emb_b in cluster_b_embeddings:
                similarities.append(speaker_similarity(emb_a, emb_b))
        return float(np.mean(similarities)) if similarities else 0.0

    elif method == "max_pairwise":
        max_sim = 0.0
        for emb_a in cluster_a_embeddings:
            for emb_b in cluster_b_embeddings:
                sim = speaker_similarity(emb_a, emb_b)
                if sim > max_sim:
                    max_sim = sim
        return max_sim

    else:
        LOGGER.warning(f"Unknown method '{method}', using centroid")
        return compute_cluster_similarity(cluster_a_embeddings, cluster_b_embeddings, "centroid")


def embedding_distance(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
    metric: str = "cosine",
) -> float:
    """Compute distance between two embeddings.

    Args:
        embedding_a: First embedding
        embedding_b: Second embedding
        metric: Distance metric - "cosine" or "euclidean"

    Returns:
        Distance value. For cosine, returns 1 - similarity (range [0, 2]).
        For euclidean, returns L2 distance.
    """
    if metric == "euclidean":
        return float(np.linalg.norm(embedding_a - embedding_b))

    # Default to cosine distance
    return 1.0 - speaker_similarity(embedding_a, embedding_b)


__all__ = [
    "speaker_similarity",
    "find_similar_speakers",
    "best_match",
    "compute_centroid",
    "compute_cluster_similarity",
    "embedding_distance",
]
