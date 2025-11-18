"""Cluster grouping service for within-episode and across-episode person matching."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from sklearn.cluster import AgglomerativeClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from py_screenalytics.artifacts import get_path

from apps.api.services.facebank import FacebankService, SEED_ATTACH_SIM
from apps.api.services.cast import CastService

from .people import PeopleService, l2_normalize, cosine_distance

LOGGER = logging.getLogger(__name__)

# Config from environment
GROUP_WITHIN_EP_DISTANCE = float(os.getenv("GROUP_WITHIN_EP_DISTANCE", "0.35"))
PEOPLE_MATCH_DISTANCE = float(os.getenv("PEOPLE_MATCH_DISTANCE", "0.35"))
PEOPLE_PROTO_MOMENTUM = float(os.getenv("PEOPLE_PROTO_MOMENTUM", "0.9"))
SEED_CLUSTER_DELTA = float(os.getenv("SEED_CLUSTER_DELTA", "0.05"))

DEFAULT_DATA_ROOT = Path("data").expanduser()


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _parse_ep_id(ep_id: str) -> Optional[Dict[str, Any]]:
    """Parse episode ID like rhobh-s05e14 into components."""
    import re
    pattern = r"^(?P<show>.+)-s(?P<season>\d{2})e(?P<episode>\d{2})$"
    match = re.match(pattern, ep_id, re.IGNORECASE)
    if not match:
        return None
    return {
        "show": match.group("show").upper(),
        "season": int(match.group("season")),
        "episode": int(match.group("episode")),
    }


class GroupingService:
    """Handle cluster centroid computation and grouping."""

    def __init__(self, data_root: Path | str | None = None):
        self.data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
        self.people_service = PeopleService(data_root)
        self.facebank_service = FacebankService(data_root)
        self.cast_service = CastService(data_root)

    def _cluster_centroids_path(self, ep_id: str) -> Path:
        """Get path to cluster_centroids.json for an episode."""
        manifests_dir = get_path(ep_id, "detections").parent
        return manifests_dir / "cluster_centroids.json"

    def _identities_path(self, ep_id: str) -> Path:
        """Get path to identities.json for an episode."""
        manifests_dir = get_path(ep_id, "detections").parent
        return manifests_dir / "identities.json"

    def _faces_path(self, ep_id: str) -> Path:
        """Get path to faces.jsonl for an episode."""
        manifests_dir = get_path(ep_id, "detections").parent
        return manifests_dir / "faces.jsonl"

    def _group_log_path(self, ep_id: str) -> Path:
        """Get path to group_log.json for an episode."""
        manifests_dir = get_path(ep_id, "detections").parent
        return manifests_dir / "group_log.json"

    def compute_cluster_centroids(self, ep_id: str, *, progress_callback=None) -> Dict[str, Any]:
        """Compute centroids for all clusters in an episode.

        Args:
            ep_id: Episode identifier
            progress_callback: Optional function(current, total, status) for progress updates

        Returns: {"centroids": [{cluster_id, centroid, num_faces}, ...]}
        """
        import logging
        LOGGER = logging.getLogger(__name__)

        faces_path = self._faces_path(ep_id)
        if not faces_path.exists():
            raise FileNotFoundError(f"faces.jsonl not found for {ep_id}")

        identities_path = self._identities_path(ep_id)
        if not identities_path.exists():
            raise FileNotFoundError(f"identities.json not found for {ep_id}")

        LOGGER.info(f"[cluster_cleanup] Computing centroids for {ep_id}")
        if progress_callback:
            progress_callback(0, 1, "Loading identities")

        # Load identities to get cluster assignments
        identities_data = json.loads(identities_path.read_text(encoding="utf-8"))
        identities = identities_data.get("identities", [])

        # Build map: cluster_id -> track_ids
        cluster_to_tracks: Dict[str, List[int]] = {}
        for identity in identities:
            cluster_id = identity["identity_id"]
            track_ids = identity.get("track_ids", [])
            cluster_to_tracks[cluster_id] = track_ids

        # Load faces and group by cluster
        cluster_embeddings: Dict[str, List[np.ndarray]] = {cid: [] for cid in cluster_to_tracks}
        cluster_counts: Dict[str, int] = {cid: 0 for cid in cluster_to_tracks}
        cluster_seed_matches: Dict[str, List[str]] = {cid: [] for cid in cluster_to_tracks}

        LOGGER.info(f"[cluster_cleanup] Processing faces from {faces_path}")
        if progress_callback:
            progress_callback(1, 3, "Loading face embeddings")

        with faces_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    face = json.loads(line)
                except json.JSONDecodeError:
                    continue

                track_id = face.get("track_id")
                embedding = face.get("embedding")
                seed_cast_id = face.get("seed_cast_id")

                if track_id is None or not embedding:
                    continue

                # Find which cluster this track belongs to
                for cluster_id, track_ids in cluster_to_tracks.items():
                    if track_id in track_ids:
                        emb_vec = np.array(embedding, dtype=np.float32)
                        cluster_embeddings[cluster_id].append(emb_vec)
                        cluster_counts[cluster_id] += 1
                        if seed_cast_id:
                            cluster_seed_matches[cluster_id].append(seed_cast_id)
                        break

        # Compute centroids
        LOGGER.info(f"[cluster_cleanup] Computing centroids for {len(cluster_to_tracks)} clusters")
        if progress_callback:
            progress_callback(2, 3, f"Computing {len(cluster_to_tracks)} centroids")

        # Use dict format (cluster_id -> centroid data) instead of list
        centroids = {}
        for cluster_id in sorted(cluster_to_tracks.keys()):
            embs = cluster_embeddings.get(cluster_id, [])
            if not embs:
                continue

            # Mean and L2-normalize
            mean_emb = np.mean(embs, axis=0)
            centroid = l2_normalize(mean_emb)

            # Get track IDs for this cluster (convert to track_XXXX format)
            track_ids = cluster_to_tracks.get(cluster_id, [])
            tracks_formatted = [f"track_{int(tid):04d}" for tid in track_ids]

            # Compute cohesion (mean similarity of embeddings to centroid)
            similarities = [float(np.dot(emb, centroid)) for emb in embs]
            cohesion = np.mean(similarities) if similarities else None

            # Determine primary seed (most common seed_cast_id)
            seed_matches = cluster_seed_matches.get(cluster_id, [])
            primary_seed = None
            seed_confidence = 0.0
            if seed_matches:
                from collections import Counter
                seed_counts = Counter(seed_matches)
                most_common_seed, count = seed_counts.most_common(1)[0]
                seed_confidence = count / len(seed_matches)
                # Only use seed if >50% of faces match
                if seed_confidence > 0.5:
                    primary_seed = most_common_seed

            centroid_entry = {
                "centroid": centroid.tolist(),
                "tracks": tracks_formatted,
                "cohesion": round(float(cohesion), 3) if cohesion is not None else None,
                "num_faces": cluster_counts[cluster_id],
            }
            if primary_seed:
                centroid_entry["seed_cast_id"] = primary_seed
                centroid_entry["seed_confidence"] = round(float(seed_confidence), 3)

            centroids[cluster_id] = centroid_entry

        # Save to file in new dict format
        output = {"ep_id": ep_id, "centroids": centroids, "computed_at": _now_iso()}
        centroids_path = self._cluster_centroids_path(ep_id)
        centroids_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

        LOGGER.info(f"[cluster_cleanup] Saved {len(centroids)} centroids to {centroids_path}")
        if progress_callback:
            progress_callback(3, 3, f"Saved {len(centroids)} centroids")

        return output

    def load_cluster_centroids(self, ep_id: str) -> Dict[str, Any]:
        """Load cluster centroids from file."""
        path = self._cluster_centroids_path(ep_id)
        if not path.exists():
            raise FileNotFoundError(f"cluster_centroids.json not found for {ep_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def group_within_episode(
        self,
        ep_id: str,
        distance_threshold: float = GROUP_WITHIN_EP_DISTANCE,
        *,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Perform agglomerative clustering on cluster centroids within an episode.

        Args:
            ep_id: Episode identifier
            distance_threshold: Maximum distance for grouping clusters
            progress_callback: Optional function(current, total, status) for progress updates

        Returns: {"groups": [{person_id, cluster_ids}, ...], "merged_count": int}
        """
        import logging
        LOGGER = logging.getLogger(__name__)

        if not HAS_SKLEARN:
            raise RuntimeError("sklearn not available; install with: pip install scikit-learn")

        LOGGER.info(f"[cluster_cleanup] Grouping clusters within {ep_id}")
        if progress_callback:
            progress_callback(0, 2, "Loading cluster centroids")

        # Load centroids
        centroids_data = self.load_cluster_centroids(ep_id)
        centroids = centroids_data.get("centroids", {})

        # Handle both dict (new) and list (legacy) formats
        if isinstance(centroids, list):
            # Legacy format: list of {cluster_id, centroid, ...}
            centroids_list = centroids
        elif isinstance(centroids, dict):
            # New format: dict keyed by cluster_id
            centroids_list = [
                {"cluster_id": cid, **data}
                for cid, data in centroids.items()
            ]
        else:
            return {"groups": [], "merged_count": 0}

        if len(centroids_list) <= 1:
            # Nothing to group
            return {"groups": [], "merged_count": 0}

        # Extract cluster IDs, centroid vectors, and seed information
        cluster_ids = [c["cluster_id"] for c in centroids_list]
        vectors = np.array([c["centroid"] for c in centroids_list], dtype=np.float32)
        seed_cast_ids = [c.get("seed_cast_id") for c in centroids_list]

        # Compute pairwise cosine distances with seed-based adjustment
        n = len(vectors)
        distance_matrix = np.zeros((n, n), dtype=np.float32)
        seed_adjustments = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = cosine_distance(vectors[i], vectors[j])

                # Apply seed-based distance reduction if both clusters match same seed
                if seed_cast_ids[i] and seed_cast_ids[j] and seed_cast_ids[i] == seed_cast_ids[j]:
                    dist = max(0.0, dist - SEED_CLUSTER_DELTA)
                    seed_adjustments += 1

                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        # Agglomerative clustering
        LOGGER.info(f"[cluster_cleanup] Running agglomerative clustering on {n} centroids (threshold={distance_threshold})")
        if progress_callback:
            progress_callback(1, 2, f"Clustering {n} centroids")

        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='complete',
        )
        labels = model.fit_predict(distance_matrix)

        # Group clusters by label
        groups_map: Dict[int, List[str]] = {}
        for cluster_id, label in zip(cluster_ids, labels):
            groups_map.setdefault(int(label), []).append(cluster_id)

        # Filter groups with more than one cluster (actual merges)
        merged_groups = [cids for cids in groups_map.values() if len(cids) > 1]

        LOGGER.info(f"[cluster_cleanup] Found {len(merged_groups)} merged groups (seed adjustments: {seed_adjustments})")
        if progress_callback:
            progress_callback(2, 2, f"Found {len(merged_groups)} merged groups")

        return {
            "groups": [{"cluster_ids": cids} for cids in merged_groups],
            "merged_count": len(merged_groups),
            "all_labels": labels.tolist(),
            "seed_adjustments": seed_adjustments,
        }

    def group_across_episodes(
        self,
        ep_id: str,
        max_distance: float = PEOPLE_MATCH_DISTANCE,
        momentum: float = PEOPLE_PROTO_MOMENTUM,
        auto_assign: bool = False,
    ) -> Dict[str, Any]:
        """Match episode clusters to show-level people (optionally without assigning).

        Args:
            ep_id: Episode ID
            max_distance: Maximum distance for matching
            momentum: Momentum for prototype updates
            auto_assign: If True, automatically assign clusters to people.
                        If False, only compute suggestions without assigning.

        Returns: {"assigned": [{cluster_id, person_id, suggested}, ...], "new_people": [...]}
        """
        parsed = _parse_ep_id(ep_id)
        if not parsed:
            raise ValueError(f"Invalid episode ID: {ep_id}")
        show_id = parsed["show"]

        # Load centroids
        centroids_data = self.load_cluster_centroids(ep_id)
        centroids = centroids_data.get("centroids", {})

        # Handle both dict (new) and list (legacy) formats
        if isinstance(centroids, list):
            centroids_list = centroids
        elif isinstance(centroids, dict):
            centroids_list = [
                {"cluster_id": cid, **data}
                for cid, data in centroids.items()
            ]
        else:
            centroids_list = []

        assigned = []
        new_people = []
        suggestions = []

        for centroid_info in centroids_list:
            cluster_id = centroid_info["cluster_id"]
            centroid = np.array(centroid_info["centroid"], dtype=np.float32)

            # Try to find matching person
            match = self.people_service.find_matching_person(show_id, centroid, max_distance)

            if match:
                person_id, distance = match
                
                if auto_assign:
                    # Assign to existing person
                    full_cluster_id = f"{ep_id}:{cluster_id}"
                    self.people_service.add_cluster_to_person(
                        show_id,
                        person_id,
                        full_cluster_id,
                        update_prototype=True,
                        cluster_centroid=centroid,
                        momentum=momentum,
                    )
                    assigned.append({
                        "cluster_id": cluster_id,
                        "person_id": person_id,
                        "distance": distance,
                        "suggested": False,
                    })
                else:
                    # Just store suggestion without assigning
                    suggestions.append({
                        "cluster_id": cluster_id,
                        "suggested_person_id": person_id,
                        "distance": distance,
                    })
            else:
                if auto_assign:
                    # Create new person
                    full_cluster_id = f"{ep_id}:{cluster_id}"
                    person = self.people_service.create_person(
                        show_id,
                        prototype=centroid.tolist(),
                        cluster_ids=[full_cluster_id],
                    )
                    new_people.append(person)
                    assigned.append({
                        "cluster_id": cluster_id,
                        "person_id": person["person_id"],
                        "distance": None,
                        "suggested": False,
                    })

        # Update identities.json with person_id assignments (only if auto_assign=True)
        if auto_assign:
            self._update_identities_with_people(ep_id, assigned)

        return {
            "assigned": assigned if auto_assign else [],
            "suggestions": suggestions if not auto_assign else [],
            "new_people_count": len(new_people),
            "new_people": new_people,
        }

    def _clear_person_assignments(self, ep_id: str) -> int:
        """Clear all person_id assignments from identities.json AND remove episode clusters from people.json.
        
        This ensures a clean state before clustering - no stale assignments remain.
        
        Returns: Number of assignments cleared from identities.json.
        """
        # Step 1: Remove episode clusters from people.json FIRST
        # This prevents the UI from showing cast members with stale clusters
        parsed = _parse_ep_id(ep_id)
        removed_clusters = 0
        if parsed:
            show_id = parsed["show"]
            try:
                result = self.people_service.remove_episode_clusters(show_id, ep_id)
                removed_clusters = result.get("removed_clusters_count", 0)
                removed_people = result.get("removed_people_count", 0)
                LOGGER.info(f"[{ep_id}] Removed {removed_clusters} cluster(s) from people.json ({removed_people} empty people deleted)")
            except Exception as e:
                LOGGER.warning(f"[{ep_id}] Failed to remove clusters from people.json: {e}")
        
        # Step 2: Clear person_id from identities.json
        identities_path = self._identities_path(ep_id)
        cleared_identities = 0
        if identities_path.exists():
            data = json.loads(identities_path.read_text(encoding="utf-8"))
            identities = data.get("identities", [])

            # Remove all person_id assignments
            for identity in identities:
                if "person_id" in identity:
                    del identity["person_id"]
                    cleared_identities += 1

            data["identities"] = identities
            identities_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            LOGGER.info(f"[{ep_id}] Cleared {cleared_identities} stale person_id assignment(s) from identities.json")
        
        # Return total cleaned: identities.json assignments + people.json clusters
        return cleared_identities + removed_clusters

    def _update_identities_with_people(
        self,
        ep_id: str,
        assignments: List[Dict[str, Any]],
    ) -> None:
        """Update identities.json with person_id assignments."""
        identities_path = self._identities_path(ep_id)
        if not identities_path.exists():
            return

        data = json.loads(identities_path.read_text(encoding="utf-8"))
        identities = data.get("identities", [])

        # Build assignment map
        assignment_map = {a["cluster_id"]: a["person_id"] for a in assignments}

        # Update identities
        for identity in identities:
            cluster_id = identity["identity_id"]
            if cluster_id in assignment_map:
                identity["person_id"] = assignment_map[cluster_id]

        data["identities"] = identities
        identities_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def group_clusters_auto(self, ep_id: str, *, progress_callback=None) -> Dict[str, Any]:
        """Run full auto grouping: compute centroids, within-episode, across-episode.

        Args:
            ep_id: Episode ID
            progress_callback: Optional callback(step: str, progress: float, message: str)

        Returns combined result with audit log.
        """
        def _progress(step: str, pct: float, msg: str):
            if progress_callback:
                progress_callback(step, pct, msg)
            LOGGER.info(f"[{ep_id}] {step}: {msg} ({int(pct*100)}%)")

        log = {
            "ep_id": ep_id,
            "started_at": _now_iso(),
            "steps": [],
        }

        # Step 0: Clear stale person_id assignments from previous runs
        _progress("clear_assignments", 0.0, "Clearing stale person assignments...")
        try:
            cleared = self._clear_person_assignments(ep_id)
            log["steps"].append({
                "step": "clear_assignments",
                "status": "success",
                "cleared_count": cleared,
            })
            _progress("clear_assignments", 0.1, f"Cleared {cleared} stale assignment(s)")
        except Exception as e:
            log["steps"].append({"step": "clear_assignments", "status": "error", "error": str(e)})
            LOGGER.warning(f"[{ep_id}] Failed to clear assignments: {e}")

        # Step 1: Compute centroids
        _progress("compute_centroids", 0.1, "Computing cluster centroids...")
        try:
            centroids_result = self.compute_cluster_centroids(ep_id)
            centroids_count = len(centroids_result.get("centroids", []))
            log["steps"].append({
                "step": "compute_centroids",
                "status": "success",
                "centroids_count": centroids_count,
            })
            _progress("compute_centroids", 0.4, f"Computed {centroids_count} centroid(s)")
        except Exception as e:
            log["steps"].append({"step": "compute_centroids", "status": "error", "error": str(e)})
            log["finished_at"] = _now_iso()
            self._save_group_log(ep_id, log)
            _progress("compute_centroids", 0.4, f"ERROR: {str(e)}")
            raise

        # Step 2: Within-episode grouping
        _progress("group_within_episode", 0.4, "Grouping similar clusters within episode...")
        try:
            within_result = self.group_within_episode(ep_id)
            merged = within_result.get("merged_count", 0)
            log["steps"].append({
                "step": "group_within_episode",
                "status": "success",
                "merged_count": merged,
            })
            _progress("group_within_episode", 0.7, f"Merged {merged} cluster group(s)")
        except Exception as e:
            log["steps"].append({"step": "group_within_episode", "status": "error", "error": str(e)})
            _progress("group_within_episode", 0.7, f"WARNING: {str(e)}")
            # Continue even if within-episode grouping fails

        # Step 3: Across-episode matching to people (compute suggestions only, don't auto-assign)
        _progress("group_across_episodes", 0.7, "Computing similarity suggestions to cast members...")
        try:
            across_result = self.group_across_episodes(ep_id, auto_assign=False)
            suggestions_count = len(across_result.get("suggestions", []))
            log["steps"].append({
                "step": "group_across_episodes",
                "status": "success",
                "suggestions_count": suggestions_count,
                "assigned_count": len(across_result.get("assigned", [])),
            })
            _progress("group_across_episodes", 1.0, f"Computed {suggestions_count} suggestion(s)")
        except Exception as e:
            log["steps"].append({"step": "group_across_episodes", "status": "error", "error": str(e)})
            log["finished_at"] = _now_iso()
            self._save_group_log(ep_id, log)
            _progress("group_across_episodes", 1.0, f"ERROR: {str(e)}")
            raise

        log["finished_at"] = _now_iso()
        self._save_group_log(ep_id, log)

        return {
            "ep_id": ep_id,
            "centroids": centroids_result,
            "within_episode": within_result,
            "across_episodes": across_result,
            "log": log,
        }

    def _save_group_log(self, ep_id: str, log: Dict[str, Any]) -> None:
        """Save grouping audit log."""
        log_path = self._group_log_path(ep_id)
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    def manual_assign_clusters(
        self,
        ep_id: str,
        cluster_ids: List[str],
        target_person_id: Optional[str] = None,
        cast_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Manually assign clusters to a person (new or existing)."""
        parsed = _parse_ep_id(ep_id)
        if not parsed:
            raise ValueError(f"Invalid episode ID: {ep_id}")
        show_id = parsed["show"]

        # Load centroids
        centroids_data = self.load_cluster_centroids(ep_id)
        centroids = centroids_data.get("centroids", {})

        # Handle both dict (new) and list (legacy) formats
        if isinstance(centroids, list):
            centroids_map = {c["cluster_id"]: np.array(c["centroid"], dtype=np.float32)
                           for c in centroids}
        elif isinstance(centroids, dict):
            centroids_map = {cid: np.array(data["centroid"], dtype=np.float32)
                           for cid, data in centroids.items()}
        else:
            centroids_map = {}

        # If no target person, create new
        if not target_person_id:
            # Use mean of selected clusters as prototype
            centroids_to_merge = [centroids_map[cid] for cid in cluster_ids if cid in centroids_map]
            if not centroids_to_merge:
                raise ValueError("No valid clusters found")

            proto = l2_normalize(np.mean(centroids_to_merge, axis=0))
            person = self.people_service.create_person(
                show_id,
                prototype=proto.tolist(),
                cluster_ids=[f"{ep_id}:{cid}" for cid in cluster_ids],
                cast_id=cast_id,
                name=name,
            )
            target_person_id = person["person_id"]
        else:
            # Assign to existing person
            for cluster_id in cluster_ids:
                full_cluster_id = f"{ep_id}:{cluster_id}"
                centroid = centroids_map.get(cluster_id)
                self.people_service.add_cluster_to_person(
                    show_id,
                    target_person_id,
                    full_cluster_id,
                    update_prototype=True,
                    cluster_centroid=centroid,
                )

        # Update identities.json
        assignments = [{"cluster_id": cid, "person_id": target_person_id} for cid in cluster_ids]
        self._update_identities_with_people(ep_id, assignments)

        return {
            "person_id": target_person_id,
            "cluster_ids": cluster_ids,
            "ep_id": ep_id,
        }

    def group_using_facebank(
        self,
        ep_id: str,
        min_similarity: float = SEED_ATTACH_SIM,
    ) -> Dict[str, Any]:
        """Assign clusters to known cast members using facebank seeds."""
        parsed = _parse_ep_id(ep_id)
        if not parsed:
            raise ValueError(f"Invalid episode ID: {ep_id}")
        show_id = parsed["show"]

        seeds = self.facebank_service.get_all_seeds_for_show(show_id)
        if not seeds:
            raise ValueError(f"No facebank seeds available for show {show_id}")

        centroids_data = self.load_cluster_centroids(ep_id)
        centroids = centroids_data.get("centroids", {})

        # Handle both dict (new) and list (legacy) formats
        if isinstance(centroids, list):
            centroids_list = centroids
        elif isinstance(centroids, dict):
            centroids_list = [
                {"cluster_id": cid, **data}
                for cid, data in centroids.items()
            ]
        else:
            centroids_list = []

        if not centroids_list:
            raise ValueError(f"No cluster centroids found for {ep_id}")

        cast_members = self.cast_service.list_cast(show_id)
        cast_lookup = {member["cast_id"]: member for member in cast_members if member.get("cast_id")}
        people = self.people_service.list_people(show_id)
        cast_person_map = {person.get("cast_id"): person for person in people if person.get("cast_id")}

        assignments = []
        matched_clusters: List[Dict[str, Any]] = []
        unmatched_clusters: List[str] = []

        for centroid_entry in centroids_list:
            cluster_id = centroid_entry["cluster_id"]
            centroid_vec = np.array(centroid_entry["centroid"], dtype=np.float32)
            match = self.facebank_service.find_matching_seed(show_id, centroid_vec, min_similarity)
            if not match:
                unmatched_clusters.append(cluster_id)
                continue

            cast_id, seed_id, similarity = match
            person = cast_person_map.get(cast_id)
            if not person:
                cast_meta = cast_lookup.get(cast_id, {})
                name = cast_meta.get("name")
                person = self.people_service.create_person(
                    show_id,
                    name=name,
                    prototype=centroid_vec.tolist(),
                    cluster_ids=[],
                    cast_id=cast_id,
                )
                cast_person_map[cast_id] = person

            person_id = person["person_id"]
            full_cluster_id = f"{ep_id}:{cluster_id}"
            updated = self.people_service.add_cluster_to_person(
                show_id,
                person_id,
                full_cluster_id,
                update_prototype=True,
                cluster_centroid=centroid_vec,
                momentum=PEOPLE_PROTO_MOMENTUM,
            )
            if updated:
                cast_person_map[cast_id] = updated

            assignments.append({"cluster_id": cluster_id, "person_id": person_id})
            matched_clusters.append({
                "cluster_id": cluster_id,
                "person_id": person_id,
                "cast_id": cast_id,
                "seed_id": seed_id,
                "similarity": round(float(similarity), 4) if similarity is not None else None,
            })

        if assignments:
            self._update_identities_with_people(ep_id, assignments)

        log = {
            "ep_id": ep_id,
            "strategy": "facebank",
            "started_at": _now_iso(),
            "steps": [
                {"step": "load_seeds", "status": "success", "seeds": len(seeds)},
                {
                    "step": "match_clusters",
                    "status": "success",
                    "matched": len(matched_clusters),
                    "unmatched": len(unmatched_clusters),
                },
            ],
            "finished_at": _now_iso(),
        }
        self._save_group_log(ep_id, log)

        return {
            "ep_id": ep_id,
            "matched_clusters": len(matched_clusters),
            "unmatched_clusters": unmatched_clusters,
            "assigned": matched_clusters,
            "log": log,
        }

    def suggest_from_assigned_clusters(
        self,
        ep_id: str,
        max_distance: float = PEOPLE_MATCH_DISTANCE,
    ) -> Dict[str, Any]:
        """Suggest matches for unassigned clusters by comparing with assigned clusters.
        
        For each unassigned cluster, find the most similar assigned cluster and suggest
        that person. This uses actual episode data rather than facebank prototypes.
        
        Args:
            ep_id: Episode ID
            max_distance: Maximum distance threshold for suggestions
            
        Returns: {"suggestions": [{cluster_id, suggested_person_id, distance}, ...]}
        """
        # Load identities to see which clusters are assigned vs unassigned
        identities_path = self._identities_path(ep_id)
        if not identities_path.exists():
            raise FileNotFoundError(f"identities.json not found for {ep_id}")
        
        identities_data = json.loads(identities_path.read_text(encoding="utf-8"))
        identities = identities_data.get("identities", [])
        
        # Load centroids
        centroids_data = self.load_cluster_centroids(ep_id)
        centroids = centroids_data.get("centroids", {})
        
        # Handle both dict (new) and list (legacy) formats
        if isinstance(centroids, list):
            centroids_map = {c["cluster_id"]: np.array(c["centroid"], dtype=np.float32)
                           for c in centroids}
        elif isinstance(centroids, dict):
            centroids_map = {cid: np.array(data["centroid"], dtype=np.float32)
                           for cid, data in centroids.items()}
        else:
            centroids_map = {}
        
        # Separate assigned and unassigned clusters
        assigned_clusters = {}  # cluster_id -> person_id
        unassigned_clusters = []
        
        for identity in identities:
            cluster_id = identity.get("identity_id")
            person_id = identity.get("person_id")
            
            if person_id:
                assigned_clusters[cluster_id] = person_id
            else:
                unassigned_clusters.append(cluster_id)
        
        if not assigned_clusters:
            LOGGER.warning(f"[{ep_id}] No assigned clusters found for comparison")
            return {"suggestions": []}
        
        if not unassigned_clusters:
            LOGGER.info(f"[{ep_id}] No unassigned clusters to suggest")
            return {"suggestions": []}
        
        LOGGER.info(f"[{ep_id}] Comparing {len(unassigned_clusters)} unassigned vs {len(assigned_clusters)} assigned clusters")
        
        suggestions = []
        
        # For each unassigned cluster, find best match among assigned clusters
        for unassigned_id in unassigned_clusters:
            unassigned_centroid = centroids_map.get(unassigned_id)
            if unassigned_centroid is None:
                continue
            
            best_match_person = None
            best_distance = float('inf')
            
            # Compare against all assigned clusters
            for assigned_id, person_id in assigned_clusters.items():
                assigned_centroid = centroids_map.get(assigned_id)
                if assigned_centroid is None:
                    continue
                
                distance = cosine_distance(unassigned_centroid, assigned_centroid)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match_person = person_id
            
            # Only suggest if distance is below threshold
            if best_match_person and best_distance <= max_distance:
                suggestions.append({
                    "cluster_id": unassigned_id,
                    "suggested_person_id": best_match_person,
                    "distance": float(best_distance),
                })
        
        LOGGER.info(f"[{ep_id}] Generated {len(suggestions)} suggestions from assigned clusters")
        return {"suggestions": suggestions}

    def save_current_assignments(self, ep_id: str) -> Dict[str, Any]:
        """Save all current cluster->person assignments to people.json.
        
        Reads identities.json to get all cluster->person_id mappings, then ensures
        they're all properly saved in people.json with updated prototypes.
        
        Returns: {"saved_count": int}
        """
        parsed = _parse_ep_id(ep_id)
        if not parsed:
            raise ValueError(f"Invalid episode ID: {ep_id}")
        show_id = parsed["show"]
        
        # Load identities
        identities_path = self._identities_path(ep_id)
        if not identities_path.exists():
            raise FileNotFoundError(f"identities.json not found for {ep_id}")
        
        identities_data = json.loads(identities_path.read_text(encoding="utf-8"))
        identities = identities_data.get("identities", [])
        
        # Load centroids
        centroids_data = self.load_cluster_centroids(ep_id)
        centroids = centroids_data.get("centroids", {})
        
        # Handle both dict (new) and list (legacy) formats
        if isinstance(centroids, list):
            centroids_map = {c["cluster_id"]: np.array(c["centroid"], dtype=np.float32)
                           for c in centroids}
        elif isinstance(centroids, dict):
            centroids_map = {cid: np.array(data["centroid"], dtype=np.float32)
                           for cid, data in centroids.items()}
        else:
            centroids_map = {}
        
        # Collect all assignments
        saved_count = 0
        for identity in identities:
            cluster_id = identity.get("identity_id")
            person_id = identity.get("person_id")
            
            if not person_id or not cluster_id:
                continue
            
            # Ensure this cluster is in the person's cluster list
            full_cluster_id = f"{ep_id}:{cluster_id}"
            centroid = centroids_map.get(cluster_id)
            
            # Add to person (this is idempotent - won't duplicate if already there)
            self.people_service.add_cluster_to_person(
                show_id,
                person_id,
                full_cluster_id,
                update_prototype=True,
                cluster_centroid=centroid,
            )
            saved_count += 1
        
        LOGGER.info(f"[{ep_id}] Saved {saved_count} cluster assignments to people.json")
        return {"saved_count": saved_count}


__all__ = ["GroupingService", "GROUP_WITHIN_EP_DISTANCE", "PEOPLE_MATCH_DISTANCE"]

