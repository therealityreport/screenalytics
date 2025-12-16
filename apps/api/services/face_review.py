"""Face Review Service.

Manages the "Improve Faces" workflow for suggesting and deciding cluster merges
and cast assignments. Persists state to disk so decisions survive session restarts.

This service handles:
1. Initial post-cluster pass for unassigned↔unassigned merge suggestions
2. On-demand "Improve Faces" for both unassigned↔unassigned and unassigned↔assigned
3. Recording user decisions (yes/no) to avoid repeated suggestions
"""

from __future__ import annotations

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from py_screenalytics.artifacts import get_path

from apps.api.services.identities import (
    load_identities,
    merge_identities,
    write_identities,
    sync_manifests,
    update_identity_stats,
)
from apps.api.services.track_reps import (
    load_cluster_centroids,
    load_track_reps,
    build_cluster_track_reps,
)
from apps.api.services.people import PeopleService
from apps.api.services.storage import (
    StorageService,
    artifact_prefixes,
    episode_context_from_id,
)
from apps.api.services.facebank import FacebankService

LOGGER = logging.getLogger(__name__)
STORAGE = StorageService()
PEOPLE_SERVICE = PeopleService()
FACEBANK_SERVICE = FacebankService()

# Thresholds for suggestion generation
# Note: cluster_thresh in config/pipeline/clustering.yaml is 0.58
# Pairs >= 0.58 were already auto-merged during clustering, so they're the same cluster
MERGE_SIMILARITY_THRESHOLD = 0.45  # Minimum sim between clusters to suggest merge
MERGE_SIMILARITY_UPPER = 0.58  # Matches cluster_thresh - above this, already merged
CAST_SUGGESTION_THRESHOLD = 0.45  # Minimum sim to suggest cast assignment


class FaceReviewService:
    """Service for managing Face Review decisions and suggestions."""

    def __init__(self):
        """Initialize the face review service."""
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _get_state_file_path(self, ep_id: str) -> Path:
        """Get the path to the face review state file for an episode."""
        manifests_dir = get_path(ep_id, "detections").parent
        return manifests_dir / "face_review_state.json"

    def _load_state(self, ep_id: str) -> Dict[str, Any]:
        """Load face review state from disk."""
        if ep_id in self._cache:
            return self._cache[ep_id]

        state_file = self._get_state_file_path(ep_id)
        if not state_file.exists():
            default_state = {
                "initial_unassigned_pass_done": False,
                "decisions": [],  # List of {pair_type, cluster_a, cluster_b, person_id, decision, timestamp}
                "updated_at": None,
            }
            self._cache[ep_id] = default_state
            return default_state

        try:
            with open(state_file, "r") as f:
                data = json.load(f)
                self._cache[ep_id] = data
                return data
        except Exception as e:
            LOGGER.warning(f"[{ep_id}] Failed to load face review state: {e}")
            default_state = {
                "initial_unassigned_pass_done": False,
                "decisions": [],
                "updated_at": None,
            }
            self._cache[ep_id] = default_state
            return default_state

    def _save_state(self, ep_id: str, state: Dict[str, Any]) -> bool:
        """Save face review state to disk."""
        state_file = self._get_state_file_path(ep_id)
        try:
            state["updated_at"] = datetime.utcnow().isoformat()
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
            self._cache[ep_id] = state
            return True
        except Exception as e:
            LOGGER.error(f"[{ep_id}] Failed to save face review state: {e}")
            return False

    def _invalidate_cache(self, ep_id: str) -> None:
        """Invalidate the cache for an episode."""
        if ep_id in self._cache:
            del self._cache[ep_id]

    def is_initial_pass_done(self, ep_id: str) -> bool:
        """Check if the initial unassigned↔unassigned pass has been completed."""
        state = self._load_state(ep_id)
        return state.get("initial_unassigned_pass_done", False)

    def mark_initial_pass_done(self, ep_id: str) -> bool:
        """Mark the initial unassigned↔unassigned pass as completed."""
        state = self._load_state(ep_id)
        state["initial_unassigned_pass_done"] = True
        return self._save_state(ep_id, state)

    def reset_initial_pass(self, ep_id: str) -> bool:
        """Reset the initial pass flag (allows re-running)."""
        state = self._load_state(ep_id)
        state["initial_unassigned_pass_done"] = False
        return self._save_state(ep_id, state)

    def reset_state(self, ep_id: str, *, archive_existing: bool = True) -> Dict[str, Any]:
        """Reset face review state (decisions + initial pass) for an episode.

        The default behavior preserves user intent by archiving the existing state file
        before writing a fresh empty state.
        """
        state_file = self._get_state_file_path(ep_id)
        archived_name: str | None = None

        if archive_existing and state_file.exists():
            archive_suffix = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            archived_path = state_file.with_name(f"{state_file.name}.{archive_suffix}.bak")
            try:
                state_file.rename(archived_path)
                archived_name = archived_path.name
                LOGGER.info("[%s] Archived face review state: %s", ep_id, archived_name)
            except OSError as exc:
                LOGGER.error("[%s] Failed to archive face review state: %s", ep_id, exc)
                return {"ok": False, "error": f"Failed to archive existing face review state: {exc}"}

        self._invalidate_cache(ep_id)
        default_state = {
            "initial_unassigned_pass_done": False,
            "decisions": [],
            "updated_at": None,
        }
        state_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._save_state(ep_id, default_state):
            return {"ok": False, "error": "Failed to write face review state"}

        return {"ok": True, "archived": archived_name}

    def get_decision(
        self,
        ep_id: str,
        cluster_a: str,
        cluster_b: Optional[str] = None,
        person_id: Optional[str] = None,
    ) -> Optional[str]:
        """Get a previous decision for a pair.

        Args:
            ep_id: Episode ID
            cluster_a: First cluster ID
            cluster_b: Second cluster ID (for unassigned↔unassigned)
            person_id: Person/cast member ID (for unassigned↔assigned)

        Returns:
            "yes", "no", or None if no decision exists
        """
        state = self._load_state(ep_id)
        decisions = state.get("decisions", [])

        for decision in decisions:
            d_cluster_a = decision.get("cluster_a")
            d_cluster_b = decision.get("cluster_b")
            d_person_id = decision.get("person_id")

            # Forward match: cluster_a matches
            if d_cluster_a == cluster_a:
                if cluster_b and d_cluster_b == cluster_b:
                    return decision.get("decision")
                if person_id and d_person_id == person_id:
                    return decision.get("decision")

            # Reverse match for cluster↔cluster pairs
            if cluster_b and d_cluster_a == cluster_b and d_cluster_b == cluster_a:
                return decision.get("decision")

        return None

    def record_decision(
        self,
        ep_id: str,
        pair_type: str,  # "unassigned_unassigned" or "unassigned_assigned"
        cluster_a: str,
        decision: str,  # "yes" or "no"
        cluster_b: Optional[str] = None,
        person_id: Optional[str] = None,
        cast_id: Optional[str] = None,
    ) -> bool:
        """Record a user decision for a face review comparison.

        Args:
            ep_id: Episode ID
            pair_type: Type of comparison
            cluster_a: First/unassigned cluster ID
            decision: User decision ("yes" or "no")
            cluster_b: Second cluster ID (for unassigned↔unassigned)
            person_id: Person ID (for unassigned↔assigned)
            cast_id: Cast member ID (for unassigned↔assigned)

        Returns:
            True if successful
        """
        state = self._load_state(ep_id)
        decisions = state.get("decisions", [])

        # Remove any existing decision for this pair (safely handles None values)
        def _should_remove_decision(d: dict) -> bool:
            """Check if existing decision should be removed (replaced by new one)."""
            d_cluster_a = d.get("cluster_a")
            d_cluster_b = d.get("cluster_b")
            d_person_id = d.get("person_id")

            if d_cluster_a != cluster_a:
                # Check reverse for cluster pairs only
                if cluster_b and d_cluster_a == cluster_b and d_cluster_b == cluster_a:
                    return True
                return False

            # cluster_a matches - check the secondary ID (only if values are provided)
            if cluster_b is not None and d_cluster_b == cluster_b:
                return True
            if person_id is not None and d_person_id == person_id:
                return True
            return False

        decisions = [d for d in decisions if not _should_remove_decision(d)]

        # Add new decision
        new_decision = {
            "pair_type": pair_type,
            "cluster_a": cluster_a,
            "cluster_b": cluster_b,
            "person_id": person_id,
            "cast_id": cast_id,
            "decision": decision,
            "timestamp": datetime.utcnow().isoformat(),
        }
        decisions.append(new_decision)
        state["decisions"] = decisions

        return self._save_state(ep_id, state)

    def get_all_decisions(self, ep_id: str) -> List[Dict[str, Any]]:
        """Get all recorded decisions for an episode."""
        state = self._load_state(ep_id)
        return state.get("decisions", [])

    def _get_cluster_size(self, identity: Dict[str, Any]) -> Tuple[int, int]:
        """Get (track_count, face_count) for sorting clusters by size."""
        track_ids = identity.get("track_ids", [])
        track_count = len(track_ids)
        # Face count stored in identity stats or size field
        face_count = identity.get("size", 0) or track_count
        return (track_count, face_count)

    def _compute_cluster_similarity(
        self,
        ep_id: str,
        cluster_a_id: str,
        cluster_b_id: str,
        centroids: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute cosine similarity between two cluster centroids."""
        if centroids is None:
            centroids = load_cluster_centroids(ep_id)

        centroid_a = centroids.get(cluster_a_id, {}).get("centroid")
        centroid_b = centroids.get(cluster_b_id, {}).get("centroid")

        if not centroid_a or not centroid_b:
            return 0.0

        a = np.array(centroid_a, dtype=np.float32)
        b = np.array(centroid_b, dtype=np.float32)

        # Compute norms and check for zero vectors (corrupted embeddings)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            LOGGER.warning(
                f"[{ep_id}] Zero/near-zero embedding detected for cluster pair "
                f"{cluster_a_id}/{cluster_b_id}: norm_a={norm_a:.2e}, norm_b={norm_b:.2e}"
            )
            return 0.0

        # Normalize and compute cosine similarity
        a_norm = a / norm_a
        b_norm = b / norm_b

        return float(np.dot(a_norm, b_norm))

    def _get_representative_crop_url(
        self,
        ep_id: str,
        cluster_id: str,
        track_reps: Optional[Dict[str, Dict[str, Any]]] = None,
        centroids: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Get the representative crop URL for a cluster (face closest to centroid).

        Uses the track with highest similarity to the cluster centroid,
        then returns the crop URL for that track's representative frame.
        """
        try:
            cluster_data = build_cluster_track_reps(ep_id, cluster_id, track_reps, centroids)
            tracks = cluster_data.get("tracks", [])

            if not tracks:
                return None

            # Sort by similarity to centroid (descending) to find best representative
            tracks_sorted = sorted(tracks, key=lambda t: t.get("similarity", 0), reverse=True)
            best_track = tracks_sorted[0]

            # Get crop URL from track rep
            crop_key = best_track.get("crop_key")
            if crop_key:
                # Construct full S3 key from relative crop_key
                try:
                    ep_ctx = episode_context_from_id(ep_id)
                    prefixes = artifact_prefixes(ep_ctx)
                    crops_prefix = prefixes.get("crops")
                    if crops_prefix:
                        crop_rel = crop_key
                        if crop_rel.startswith("crops/"):
                            crop_rel = crop_rel[6:]
                        constructed_key = f"{crops_prefix}{crop_rel}"
                        url = STORAGE.presign_get(constructed_key)
                        if url:
                            return url
                except (ValueError, KeyError) as e:
                    LOGGER.warning(f"[{ep_id}] Failed to construct S3 key for crop: {e}")

            return None
        except Exception as e:
            LOGGER.warning(f"[{ep_id}] Failed to get rep crop for {cluster_id}: {e}")
            return None

    def get_initial_unassigned_suggestions(
        self,
        ep_id: str,
        show_id: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Get cluster pairs for the initial post-cluster pass to reduce duplicates.

        Returns pairs where clusters might be the same person but were assigned
        to different people. This helps reduce the total number of people to deal with.

        Includes:
        - Unassigned↔unassigned pairs (both without person_id)
        - Assigned↔assigned pairs (different person_ids, no cast assignment yet)

        Args:
            ep_id: Episode ID
            show_id: Show ID for looking up people
            limit: Maximum number of suggestions to return

        Returns:
            {
                "ep_id": "...",
                "initial_pass_done": bool,
                "suggestions": [
                    {
                        "cluster_a": {"id": "...", "crop_url": "...", "tracks": N, "faces": M},
                        "cluster_b": {"id": "...", "crop_url": "...", "tracks": N, "faces": M},
                        "similarity": 0.72,
                    },
                    ...
                ],
                "total_candidates": N
            }
        """
        state = self._load_state(ep_id)

        if state.get("initial_unassigned_pass_done"):
            return {
                "ep_id": ep_id,
                "initial_pass_done": True,
                "suggestions": [],
                "total_candidates": 0,
            }

        identities_data = load_identities(ep_id)
        identities = identities_data.get("identities", [])

        # Bug 12 fix: Handle missing centroid/track_reps files
        try:
            centroids = load_cluster_centroids(ep_id)
        except FileNotFoundError:
            LOGGER.warning(f"[{ep_id}] Centroids not computed yet, cannot generate suggestions")
            return {
                "ep_id": ep_id,
                "initial_pass_done": False,
                "suggestions": [],
                "total_candidates": 0,
                "error": "Centroids not computed yet. Run clustering first.",
            }

        try:
            track_reps = load_track_reps(ep_id)
        except FileNotFoundError:
            LOGGER.warning(f"[{ep_id}] Track reps not computed, using empty dict")
            track_reps = {}

        # Get ALL clusters with track_ids (both assigned and unassigned)
        # to find potential duplicates that should be merged
        all_clusters = [
            ident for ident in identities
            if ident.get("track_ids")
        ]

        # Also track unassigned count for logging
        unassigned_count = sum(1 for c in all_clusters if not c.get("person_id"))
        assigned_count = len(all_clusters) - unassigned_count

        LOGGER.info(f"[{ep_id}] Found {len(all_clusters)} clusters for duplicate detection "
                    f"({unassigned_count} unassigned, {assigned_count} assigned)")

        # Build candidate pairs - compare clusters that might be duplicates
        candidates = []
        seen_pairs = set()

        for i, cluster_a in enumerate(all_clusters):
            for cluster_b in all_clusters[i + 1:]:
                id_a = cluster_a.get("identity_id")
                id_b = cluster_b.get("identity_id")
                person_a = cluster_a.get("person_id")
                person_b = cluster_b.get("person_id")

                if not id_a or not id_b:
                    continue

                # Skip if both are assigned to the SAME person (already merged)
                if person_a and person_b and person_a == person_b:
                    continue

                # Skip if we already have a decision
                if self.get_decision(ep_id, id_a, cluster_b=id_b):
                    continue

                # Compute similarity
                sim = self._compute_cluster_similarity(ep_id, id_a, id_b, centroids)

                # Only suggest borderline pairs (between thresholds)
                if MERGE_SIMILARITY_THRESHOLD <= sim < MERGE_SIMILARITY_UPPER:
                    pair_key = tuple(sorted([id_a, id_b]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        candidates.append({
                            "cluster_a": cluster_a,
                            "cluster_b": cluster_b,
                            "similarity": sim,
                            "person_a": person_a,
                            "person_b": person_b,
                        })

        # Sort by similarity (highest first - most likely matches)
        candidates.sort(key=lambda x: x["similarity"], reverse=True)

        # Build response with crop URLs
        suggestions = []
        for cand in candidates[:limit]:
            cluster_a = cand["cluster_a"]
            cluster_b = cand["cluster_b"]
            id_a = cluster_a.get("identity_id")
            id_b = cluster_b.get("identity_id")

            size_a = self._get_cluster_size(cluster_a)
            size_b = self._get_cluster_size(cluster_b)

            suggestions.append({
                "id": f"merge:{id_a}:{id_b}",
                "type": "merge",
                "cluster_a": {
                    "id": id_a,
                    "crop_url": self._get_representative_crop_url(ep_id, id_a, track_reps, centroids),
                    "tracks": size_a[0],
                    "faces": size_a[1],
                },
                "cluster_b": {
                    "id": id_b,
                    "crop_url": self._get_representative_crop_url(ep_id, id_b, track_reps, centroids),
                    "tracks": size_b[0],
                    "faces": size_b[1],
                },
                "similarity": round(cand["similarity"], 4),
            })

        return {
            "status": "success",
            "ep_id": ep_id,
            "initial_pass_done": False,
            "suggestions": suggestions,
            "total_candidates": len(candidates),
        }

    def get_improve_faces_queue(
        self,
        ep_id: str,
        show_id: Optional[str] = None,
        limit: int = 30,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Get mixed queue of suggestions for the Faces Review "Improve Faces" feature.

        OPTIMIZED VERSION: Pre-loads all decisions, caches crop URLs, and limits comparisons.

        Returns both:
        1. Unassigned↔unassigned pairs (cluster merges)
        2. Unassigned↔assigned pairs (cast assignments)

        Args:
            ep_id: Episode ID
            show_id: Show ID for looking up cast members
            limit: Maximum number of suggestions to return
            progress_callback: Optional callback(step, progress, message) for async progress

        Returns:
            {
                "ep_id": "...",
                "suggestions": [
                    {
                        "type": "unassigned_unassigned" | "unassigned_assigned",
                        "cluster_a": {"id": "...", "crop_url": "...", "tracks": N, "faces": M},
                        "cluster_b": {...} | None,  # For unassigned↔unassigned
                        "cast_member": {...} | None,  # For unassigned↔assigned
                        "similarity": 0.72,
                    },
                    ...
                ],
                "total_unassigned_pairs": N,
                "total_cast_suggestions": M
            }
        """
        def _report_progress(step: str, progress: float, message: str):
            if progress_callback:
                try:
                    progress_callback(step, progress, message)
                except Exception:
                    pass

        _report_progress("loading", 0.05, "Loading identities and embeddings...")

        identities_data = load_identities(ep_id)
        identities = identities_data.get("identities", [])

        # Bug 12 fix: Handle missing centroid/track_reps files
        try:
            centroids = load_cluster_centroids(ep_id)
        except FileNotFoundError:
            LOGGER.warning(f"[{ep_id}] Centroids not computed yet, cannot generate suggestions")
            return {
                "ep_id": ep_id,
                "suggestions": [],
                "total_unassigned_pairs": 0,
                "total_cast_suggestions": 0,
                "error": "Centroids not computed yet. Run clustering first.",
            }

        try:
            track_reps = load_track_reps(ep_id)
        except FileNotFoundError:
            LOGGER.warning(f"[{ep_id}] Track reps not computed, using empty dict")
            track_reps = {}

        _report_progress("loading", 0.10, "Building cluster lookups...")

        # Build lookup for identities
        identity_by_id = {ident.get("identity_id"): ident for ident in identities}

        # Separate unassigned and assigned clusters
        unassigned = [
            ident for ident in identities
            if not ident.get("person_id") and ident.get("track_ids")
        ]
        assigned = [
            ident for ident in identities
            if ident.get("person_id") and ident.get("track_ids")
        ]

        # OPTIMIZATION 1: Pre-load ALL decisions once into a set for O(1) lookups
        _report_progress("loading", 0.15, "Loading previous decisions...")
        all_decisions = self.get_all_decisions(ep_id)
        decided_cluster_pairs = set()  # (sorted tuple of cluster_a, cluster_b)
        decided_person_pairs = set()   # (cluster_a, person_id)

        for decision in all_decisions:
            cluster_a = decision.get("cluster_a")
            cluster_b = decision.get("cluster_b")
            person_id = decision.get("person_id")

            if cluster_a and cluster_b:
                decided_cluster_pairs.add(tuple(sorted([cluster_a, cluster_b])))
            if cluster_a and person_id:
                decided_person_pairs.add((cluster_a, person_id))

        # OPTIMIZATION 2: Cache crop URLs to avoid repeated S3 presigning
        _crop_url_cache: Dict[str, Optional[str]] = {}

        def _get_cached_crop_url(cluster_id: str) -> Optional[str]:
            if cluster_id not in _crop_url_cache:
                _crop_url_cache[cluster_id] = self._get_representative_crop_url(
                    ep_id, cluster_id, track_reps, centroids
                )
            return _crop_url_cache[cluster_id]

        suggestions = []
        unassigned_pair_count = 0
        cast_suggestion_count = 0

        # 1. Unassigned↔unassigned pairs (with early termination once we have enough)
        _report_progress("computing", 0.20, f"Computing similarities for {len(unassigned)} unassigned clusters...")

        # OPTIMIZATION 3: Limit total comparisons and collect candidates first
        max_merge_suggestions = limit * 2  # Collect more than needed, then take best
        merge_candidates = []

        total_pairs = len(unassigned) * (len(unassigned) - 1) // 2
        pairs_checked = 0

        for i, cluster_a in enumerate(unassigned):
            id_a = cluster_a.get("identity_id")
            if not id_a:
                continue

            for cluster_b in unassigned[i + 1:]:
                id_b = cluster_b.get("identity_id")
                if not id_b:
                    continue

                pairs_checked += 1

                # Report progress every 100 pairs
                if pairs_checked % 100 == 0:
                    pct = 0.20 + 0.30 * (pairs_checked / max(total_pairs, 1))
                    _report_progress("computing", pct, f"Checked {pairs_checked}/{total_pairs} cluster pairs...")

                # Skip if we already have a decision (O(1) lookup)
                pair_key = tuple(sorted([id_a, id_b]))
                if pair_key in decided_cluster_pairs:
                    continue

                sim = self._compute_cluster_similarity(ep_id, id_a, id_b, centroids)

                if sim >= MERGE_SIMILARITY_THRESHOLD:
                    unassigned_pair_count += 1
                    merge_candidates.append({
                        "id_a": id_a,
                        "id_b": id_b,
                        "cluster_a": cluster_a,
                        "cluster_b": cluster_b,
                        "similarity": sim,
                    })

            # Early termination if we have enough high-quality candidates
            if len(merge_candidates) >= max_merge_suggestions:
                break

        # Sort merge candidates by similarity and build suggestions
        _report_progress("building", 0.55, "Building merge suggestions...")
        merge_candidates.sort(key=lambda x: x["similarity"], reverse=True)

        for candidate in merge_candidates[:limit]:
            id_a, id_b = candidate["id_a"], candidate["id_b"]
            cluster_a, cluster_b = candidate["cluster_a"], candidate["cluster_b"]
            sim = candidate["similarity"]

            size_a = self._get_cluster_size(cluster_a)
            size_b = self._get_cluster_size(cluster_b)

            suggestions.append({
                "id": f"merge:{id_a}:{id_b}",
                "type": "merge",
                "cluster_a": {
                    "id": id_a,
                    "crop_url": _get_cached_crop_url(id_a),
                    "tracks": size_a[0],
                    "faces": size_a[1],
                },
                "cluster_b": {
                    "id": id_b,
                    "crop_url": _get_cached_crop_url(id_b),
                    "tracks": size_b[0],
                    "faces": size_b[1],
                },
                "cast": None,
                "similarity": round(sim, 4),
            })

        # 2. Unassigned↔assigned (cast member) suggestions
        _report_progress("computing", 0.60, "Computing cast assignment suggestions...")

        # Load people to get cast member info
        people_data = {}
        if show_id:
            try:
                people_data = PEOPLE_SERVICE.list_people(show_id) or {}
            except Exception as e:
                LOGGER.warning(f"[{ep_id}] Failed to load people: {e}")

        people_list = people_data.get("people", [])
        people_by_id = {p.get("person_id"): p for p in people_list}

        # OPTIMIZATION 4: Limit assigned clusters to check per unassigned
        max_assigned_checks = min(len(assigned), 50)  # Don't check more than 50 assigned clusters

        for idx, unassigned_cluster in enumerate(unassigned):
            unassigned_id = unassigned_cluster.get("identity_id")
            if not unassigned_id:
                continue

            if idx % 10 == 0:
                pct = 0.60 + 0.30 * (idx / max(len(unassigned), 1))
                _report_progress("computing", pct, f"Checking cast matches for cluster {idx}/{len(unassigned)}...")

            best_match = None
            best_sim = 0

            # Check only first N assigned clusters (sorted by size for better matches)
            for assigned_cluster in assigned[:max_assigned_checks]:
                assigned_id = assigned_cluster.get("identity_id")
                person_id = assigned_cluster.get("person_id")

                if not assigned_id or not person_id:
                    continue

                # Skip if we already have a decision for this pair (O(1) lookup)
                if (unassigned_id, person_id) in decided_person_pairs:
                    continue

                sim = self._compute_cluster_similarity(ep_id, unassigned_id, assigned_id, centroids)

                if sim >= CAST_SUGGESTION_THRESHOLD and sim > best_sim:
                    person = people_by_id.get(person_id, {})
                    best_match = {
                        "assigned_cluster": assigned_cluster,
                        "person_id": person_id,
                        "cast_id": person.get("cast_id"),
                        "cast_name": person.get("name", f"Person {person_id}"),
                        "similarity": sim,
                    }
                    best_sim = sim

            # Add the best match for this unassigned cluster
            if best_match:
                cast_suggestion_count += 1

                size_a = self._get_cluster_size(unassigned_cluster)
                assigned_cluster = best_match["assigned_cluster"]
                size_assigned = self._get_cluster_size(assigned_cluster)

                suggestions.append({
                    "id": f"assign:{unassigned_id}:{best_match['person_id']}",
                    "type": "assign",
                    "cluster": {
                        "id": unassigned_id,
                        "crop_url": _get_cached_crop_url(unassigned_id),
                        "tracks": size_a[0],
                        "faces": size_a[1],
                    },
                    "cluster_a": None,
                    "cluster_b": None,
                    "cast": {
                        "person_id": best_match["person_id"],
                        "cast_id": best_match["cast_id"],
                        "name": best_match["cast_name"],
                        "crop_url": _get_cached_crop_url(assigned_cluster.get("identity_id")),
                        "tracks": size_assigned[0],
                        "faces": size_assigned[1],
                    },
                    "similarity": round(best_match["similarity"], 4),
                })

        # Sort all suggestions by similarity
        _report_progress("finalizing", 0.95, "Sorting and finalizing suggestions...")
        suggestions.sort(key=lambda x: x["similarity"], reverse=True)

        _report_progress("done", 1.0, f"Found {len(suggestions[:limit])} suggestions")

        return {
            "status": "success",
            "ep_id": ep_id,
            "suggestions": suggestions[:limit],
            "total_unassigned_pairs": unassigned_pair_count,
            "total_cast_suggestions": cast_suggestion_count,
        }

    def process_decision_by_id(
        self,
        ep_id: str,
        suggestion_id: str,
        decision: str,
        show_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a decision using the suggestion ID format.

        Parses suggestion IDs like:
        - "merge:cluster_a_id:cluster_b_id" for merge suggestions
        - "assign:cluster_id:person_id" for assignment suggestions

        Args:
            ep_id: Episode ID
            suggestion_id: ID from the suggestion (e.g., "merge:abc:def")
            decision: "merge", "assign", or "reject"
            show_id: Show ID (for assignments)

        Returns:
            {
                "status": "success" | "error",
                "action": "merged" | "assigned" | "rejected",
                "details": {...}
            }
        """
        # Parse the suggestion ID
        parts = suggestion_id.split(":")
        if len(parts) < 3:
            return {
                "status": "error",
                "action": "none",
                "details": {"error": f"Invalid suggestion_id format: {suggestion_id}"},
            }

        suggestion_type = parts[0].strip()
        id_a = parts[1].strip()
        id_b = parts[2].strip()

        # Validate all parts are non-empty
        if not suggestion_type or not id_a or not id_b:
            return {
                "status": "error",
                "action": "none",
                "details": {"error": f"Empty values in suggestion_id: {suggestion_id}"},
            }

        # Validate suggestion_type before processing
        if suggestion_type not in ("merge", "assign"):
            return {
                "status": "error",
                "action": "none",
                "details": {"error": f"Unknown suggestion type: {suggestion_type}"},
            }

        # Map decision types
        if decision == "reject":
            internal_decision = "no"
        else:
            internal_decision = "yes"

        if suggestion_type == "merge":
            return self.process_decision(
                ep_id=ep_id,
                pair_type="unassigned_unassigned",
                cluster_a_id=id_a,
                decision=internal_decision,
                cluster_b_id=id_b,
                show_id=show_id,
            )
        elif suggestion_type == "assign":
            return self.process_decision(
                ep_id=ep_id,
                pair_type="unassigned_assigned",
                cluster_a_id=id_a,
                decision=internal_decision,
                person_id=id_b,
                show_id=show_id,
            )
        else:
            return {
                "status": "error",
                "action": "none",
                "details": {"error": f"Unknown suggestion type: {suggestion_type}"},
            }

    def process_decision(
        self,
        ep_id: str,
        pair_type: str,
        cluster_a_id: str,
        decision: str,
        cluster_b_id: Optional[str] = None,
        person_id: Optional[str] = None,
        cast_id: Optional[str] = None,
        show_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a user decision (yes/no) for a face review comparison.

        For "yes" decisions:
        - unassigned_unassigned: Merge clusters (smaller into larger)
        - unassigned_assigned: Assign unassigned cluster to cast member

        For "no" decisions:
        - Record the negative constraint to prevent re-suggestion

        Args:
            ep_id: Episode ID
            pair_type: "unassigned_unassigned" or "unassigned_assigned"
            cluster_a_id: Unassigned cluster ID
            decision: "yes" or "no"
            cluster_b_id: Second cluster ID (for merges)
            person_id: Person ID (for assignments)
            cast_id: Cast member ID (for assignments)
            show_id: Show ID (for assignments)

        Returns:
            {
                "status": "success" | "error",
                "action": "merged" | "assigned" | "rejected",
                "details": {...}
            }
        """
        # Record the decision first
        self.record_decision(
            ep_id=ep_id,
            pair_type=pair_type,
            cluster_a=cluster_a_id,
            decision=decision,
            cluster_b=cluster_b_id,
            person_id=person_id,
            cast_id=cast_id,
        )

        if decision == "no":
            return {
                "status": "success",
                "action": "rejected",
                "details": {
                    "cluster_a": cluster_a_id,
                    "cluster_b": cluster_b_id,
                    "person_id": person_id,
                },
            }

        # decision == "yes"
        if pair_type == "unassigned_unassigned" and cluster_b_id:
            return self._merge_clusters(ep_id, cluster_a_id, cluster_b_id)
        elif pair_type == "unassigned_assigned" and person_id:
            return self._assign_to_cast(ep_id, cluster_a_id, person_id, cast_id, show_id)
        else:
            return {
                "status": "error",
                "action": "none",
                "details": {"error": "Invalid pair_type or missing required IDs"},
            }

    def _merge_clusters(
        self,
        ep_id: str,
        cluster_a_id: str,
        cluster_b_id: str,
    ) -> Dict[str, Any]:
        """Merge two unassigned clusters (smaller into larger).

        The cluster with more tracks (then faces) becomes the target.
        """
        import time
        t0 = time.time()
        try:
            identities_data = load_identities(ep_id)
            t1 = time.time()
            LOGGER.info(f"[{ep_id}] _merge_clusters: load_identities took {t1-t0:.3f}s")
            identities = identities_data.get("identities", [])

            cluster_a = next((i for i in identities if i.get("identity_id") == cluster_a_id), None)
            cluster_b = next((i for i in identities if i.get("identity_id") == cluster_b_id), None)

            if not cluster_a or not cluster_b:
                return {
                    "status": "error",
                    "action": "merge_failed",
                    "details": {"error": "One or both clusters not found"},
                }

            # Determine which is larger (target) vs smaller (source)
            size_a = self._get_cluster_size(cluster_a)
            size_b = self._get_cluster_size(cluster_b)

            if size_a >= size_b:
                target_id, source_id = cluster_a_id, cluster_b_id
            else:
                target_id, source_id = cluster_b_id, cluster_a_id

            # Perform merge
            t2 = time.time()
            merged = merge_identities(ep_id, source_id=source_id, target_id=target_id)
            t3 = time.time()
            LOGGER.info(f"[{ep_id}] _merge_clusters: merge_identities took {t3-t2:.3f}s, total {t3-t0:.3f}s")
            LOGGER.info(f"[{ep_id}] Merged cluster {source_id} into {target_id}")

            return {
                "status": "success",
                "action": "merged",
                "details": {
                    "source_cluster": source_id,
                    "target_cluster": target_id,
                    "merged_track_ids": merged.get("track_ids", []),
                },
            }
        except Exception as e:
            LOGGER.error(f"[{ep_id}] Failed to merge clusters: {e}")
            return {
                "status": "error",
                "action": "merge_failed",
                "details": {"error": str(e)},
            }

    def _assign_to_cast(
        self,
        ep_id: str,
        cluster_id: str,
        person_id: str,
        cast_id: Optional[str],
        show_id: Optional[str],
    ) -> Dict[str, Any]:
        """Assign an unassigned cluster to a cast member.

        This links the cluster to the existing person record.
        """
        try:
            # Bug 16 fix: Validate person_id exists before assignment
            if show_id and person_id:
                try:
                    person = PEOPLE_SERVICE.get_person(show_id, person_id)
                    if not person:
                        return {
                            "status": "error",
                            "action": "assign_failed",
                            "details": {"error": f"Person {person_id} not found in show {show_id}"},
                        }
                except Exception as e:
                    LOGGER.warning(f"[{ep_id}] Could not validate person {person_id}: {e}")
                    # Continue anyway - the add_cluster_to_person call will fail if person doesn't exist

            identities_data = load_identities(ep_id)
            identities = identities_data.get("identities", [])

            cluster = next((i for i in identities if i.get("identity_id") == cluster_id), None)
            if not cluster:
                return {
                    "status": "error",
                    "action": "assign_failed",
                    "details": {"error": "Cluster not found"},
                }

            # Set the person_id on the cluster
            cluster["person_id"] = person_id

            # Update manual_assignments tracking
            manual_assignments = identities_data.setdefault("manual_assignments", {})
            manual_assignments[cluster_id] = {
                "assigned_by": "face_review",
                "timestamp": datetime.utcnow().isoformat(),
                "cast_id": cast_id,
                "person_id": person_id,
            }

            # Skip heavy stats update for quick operations - cluster count unchanged
            identities_path = write_identities(ep_id, identities_data)

            # Update the person's cluster_ids list BEFORE syncing to S3
            # This avoids a race condition where async upload could overwrite rollback data
            if show_id and person_id:
                try:
                    qualified_cluster_id = f"{ep_id}:{cluster_id}"
                    PEOPLE_SERVICE.add_cluster_to_person(show_id, person_id, qualified_cluster_id)
                except Exception as e:
                    # Rollback the identity assignment to maintain consistency
                    LOGGER.error(f"[{ep_id}] Failed to update person cluster list, rolling back: {e}")
                    cluster["person_id"] = None
                    cluster.pop("cast_id", None)
                    manual_assignments.pop(cluster_id, None)
                    write_identities(ep_id, identities_data)
                    sync_manifests(ep_id, identities_path)
                    return {
                        "status": "error",
                        "action": "assign_failed",
                        "details": {"error": f"Failed to update person record: {e}"},
                    }

            # Now sync to S3 - safe to use async since person update succeeded (or wasn't needed)
            sync_manifests(ep_id, identities_path, async_upload=True)

            LOGGER.info(f"[{ep_id}] Assigned cluster {cluster_id} to person {person_id}")

            # Auto-seed facebank with cluster centroid for future matching
            if cast_id and show_id:
                try:
                    self._auto_seed_facebank(ep_id, cluster_id, cast_id, show_id, cluster)
                except Exception as seed_err:
                    # Don't fail the assignment if seeding fails - just log
                    LOGGER.warning(f"[{ep_id}] Auto-seed facebank failed for cluster {cluster_id}: {seed_err}")

            return {
                "status": "success",
                "action": "assigned",
                "details": {
                    "cluster_id": cluster_id,
                    "person_id": person_id,
                    "cast_id": cast_id,
                },
            }
        except Exception as e:
            LOGGER.error(f"[{ep_id}] Failed to assign cluster to cast: {e}")
            return {
                "status": "error",
                "action": "assign_failed",
                "details": {"error": str(e)},
            }

    def _auto_seed_facebank(
        self,
        ep_id: str,
        cluster_id: str,
        cast_id: str,
        show_id: str,
        cluster: Dict[str, Any],
    ) -> None:
        """Auto-seed facebank with cluster centroid when assigned to cast.

        This enables future clustering/matching to recognize this cast member
        based on the assigned cluster's representative face.

        Args:
            ep_id: Episode ID
            cluster_id: Cluster being assigned
            cast_id: Cast member ID receiving the assignment
            show_id: Show ID
            cluster: Cluster identity data dict
        """
        # Load centroid embedding for this cluster
        try:
            centroids = load_cluster_centroids(ep_id)
        except FileNotFoundError:
            LOGGER.warning(f"[{ep_id}] No centroids file, skipping auto-seed for {cluster_id}")
            return

        centroid_data = centroids.get(cluster_id)
        if not centroid_data:
            LOGGER.warning(f"[{ep_id}] No centroid for cluster {cluster_id}, skipping auto-seed")
            return

        centroid_embedding = centroid_data.get("centroid")
        if not centroid_embedding:
            LOGGER.warning(f"[{ep_id}] Empty centroid for cluster {cluster_id}, skipping auto-seed")
            return

        embedding = np.array(centroid_embedding, dtype=np.float32)

        # Get representative crop URL for this cluster
        try:
            track_reps = load_track_reps(ep_id)
        except FileNotFoundError:
            track_reps = {}

        crop_url = self._get_representative_crop_url(ep_id, cluster_id, track_reps, centroids)

        # Build quality info from cluster data
        quality = {
            "source": "cluster_centroid",
            "episode_id": ep_id,
            "cluster_id": cluster_id,
            "track_count": len(cluster.get("track_ids", [])),
            "cluster_size": cluster.get("size", 0),
        }

        # Generate seed ID that encodes the source
        seed_id = f"auto_{ep_id}_{cluster_id}"

        # Add the seed to facebank
        FACEBANK_SERVICE.add_seed(
            show_id=show_id,
            cast_id=cast_id,
            image_path=crop_url or "",  # display URI
            embedding=embedding,
            quality=quality,
            seed_id=seed_id,
            display_uri=crop_url,
        )

        LOGGER.info(
            f"[{ep_id}] Auto-seeded facebank for cast {cast_id} from cluster {cluster_id} "
            f"({len(cluster.get('track_ids', []))} tracks)"
        )


# Singleton instance
face_review_service = FaceReviewService()
