"""Cluster grouping service for within-episode and across-episode person matching."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.cluster import AgglomerativeClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from py_screenalytics.artifacts import get_path

from .people import PeopleService, l2_normalize, cosine_distance

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

    def compute_cluster_centroids(self, ep_id: str) -> Dict[str, Any]:
        """Compute centroids for all clusters in an episode.

        Returns: {"centroids": [{cluster_id, centroid, num_faces}, ...]}
        """
        faces_path = self._faces_path(ep_id)
        if not faces_path.exists():
            raise FileNotFoundError(f"faces.jsonl not found for {ep_id}")

        identities_path = self._identities_path(ep_id)
        if not identities_path.exists():
            raise FileNotFoundError(f"identities.json not found for {ep_id}")

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
        centroids = []
        for cluster_id in sorted(cluster_to_tracks.keys()):
            embs = cluster_embeddings.get(cluster_id, [])
            if not embs:
                continue

            # Mean and L2-normalize
            mean_emb = np.mean(embs, axis=0)
            centroid = l2_normalize(mean_emb)

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
                "cluster_id": cluster_id,
                "centroid": centroid.tolist(),
                "num_faces": cluster_counts[cluster_id],
            }
            if primary_seed:
                centroid_entry["seed_cast_id"] = primary_seed
                centroid_entry["seed_confidence"] = round(float(seed_confidence), 3)

            centroids.append(centroid_entry)

        # Save to file
        output = {"ep_id": ep_id, "centroids": centroids, "computed_at": _now_iso()}
        centroids_path = self._cluster_centroids_path(ep_id)
        centroids_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

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
    ) -> Dict[str, Any]:
        """Perform agglomerative clustering on cluster centroids within an episode.

        Returns: {"groups": [{person_id, cluster_ids}, ...], "merged_count": int}
        """
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn not available; install with: pip install scikit-learn")

        # Load centroids
        centroids_data = self.load_cluster_centroids(ep_id)
        centroids_list = centroids_data.get("centroids", [])

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
    ) -> Dict[str, Any]:
        """Match episode clusters to show-level people and update prototypes.

        Returns: {"assigned": [{cluster_id, person_id}, ...], "new_people": [...]}
        """
        parsed = _parse_ep_id(ep_id)
        if not parsed:
            raise ValueError(f"Invalid episode ID: {ep_id}")
        show_id = parsed["show"]

        # Load centroids
        centroids_data = self.load_cluster_centroids(ep_id)
        centroids_list = centroids_data.get("centroids", [])

        assigned = []
        new_people = []

        for centroid_info in centroids_list:
            cluster_id = centroid_info["cluster_id"]
            centroid = np.array(centroid_info["centroid"], dtype=np.float32)

            # Try to find matching person
            match = self.people_service.find_matching_person(show_id, centroid, max_distance)

            if match:
                person_id, distance = match
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
                })
            else:
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
                })

        # Update identities.json with person_id assignments
        self._update_identities_with_people(ep_id, assigned)

        return {
            "assigned": assigned,
            "new_people_count": len(new_people),
            "new_people": new_people,
        }

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

    def group_clusters_auto(self, ep_id: str) -> Dict[str, Any]:
        """Run full auto grouping: compute centroids, within-episode, across-episode.

        Returns combined result with audit log.
        """
        log = {
            "ep_id": ep_id,
            "started_at": _now_iso(),
            "steps": [],
        }

        # Step 1: Compute centroids
        try:
            centroids_result = self.compute_cluster_centroids(ep_id)
            log["steps"].append({
                "step": "compute_centroids",
                "status": "success",
                "centroids_count": len(centroids_result.get("centroids", [])),
            })
        except Exception as e:
            log["steps"].append({"step": "compute_centroids", "status": "error", "error": str(e)})
            log["finished_at"] = _now_iso()
            self._save_group_log(ep_id, log)
            raise

        # Step 2: Within-episode grouping
        try:
            within_result = self.group_within_episode(ep_id)
            log["steps"].append({
                "step": "group_within_episode",
                "status": "success",
                "merged_count": within_result.get("merged_count", 0),
            })
        except Exception as e:
            log["steps"].append({"step": "group_within_episode", "status": "error", "error": str(e)})
            # Continue even if within-episode grouping fails

        # Step 3: Across-episode matching to people
        try:
            across_result = self.group_across_episodes(ep_id)
            log["steps"].append({
                "step": "group_across_episodes",
                "status": "success",
                "assigned_count": len(across_result.get("assigned", [])),
                "new_people_count": across_result.get("new_people_count", 0),
            })
        except Exception as e:
            log["steps"].append({"step": "group_across_episodes", "status": "error", "error": str(e)})
            log["finished_at"] = _now_iso()
            self._save_group_log(ep_id, log)
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
    ) -> Dict[str, Any]:
        """Manually assign clusters to a person (new or existing)."""
        parsed = _parse_ep_id(ep_id)
        if not parsed:
            raise ValueError(f"Invalid episode ID: {ep_id}")
        show_id = parsed["show"]

        # Load centroids
        centroids_data = self.load_cluster_centroids(ep_id)
        centroids_map = {c["cluster_id"]: np.array(c["centroid"], dtype=np.float32)
                         for c in centroids_data.get("centroids", [])}

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


__all__ = ["GroupingService", "GROUP_WITHIN_EP_DISTANCE", "PEOPLE_MATCH_DISTANCE"]
