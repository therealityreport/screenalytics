"""Cluster grouping service for within-episode and across-episode person matching."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

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
GROUP_WITHIN_EP_DISTANCE = float(os.getenv("GROUP_WITHIN_EP_DISTANCE", "0.40"))  # Increased from 0.35 for better recall
PEOPLE_MATCH_DISTANCE = float(os.getenv("PEOPLE_MATCH_DISTANCE", "0.40"))  # Increased from 0.35 for better matching
PEOPLE_PROTO_MOMENTUM = float(os.getenv("PEOPLE_PROTO_MOMENTUM", "0.9"))
SEED_CLUSTER_DELTA = float(os.getenv("SEED_CLUSTER_DELTA", "0.08"))  # Increased from 0.05 for stronger seed preference
# Apply seed delta in across-episode matching (consistent with within-episode)
USE_SEED_IN_ACROSS_EPISODE = os.getenv("USE_SEED_IN_ACROSS_EPISODE", "1").lower() in ("1", "true", "yes")
# Minimum embedding dimension for validation
MIN_EMBEDDING_DIM = 128

# Enhancement: Cohesion-weighted distance adjustment
# Clusters with high cohesion (faces are very similar) get tighter matching
COHESION_WEIGHT_ENABLED = os.getenv("COHESION_WEIGHT_ENABLED", "1").lower() in ("1", "true", "yes")
COHESION_BONUS_MAX = float(os.getenv("COHESION_BONUS_MAX", "0.05"))  # Max distance reduction for high-cohesion clusters

# Enhancement: Facebank-first matching (try facebank before people prototypes)
FACEBANK_FIRST_MATCHING = os.getenv("FACEBANK_FIRST_MATCHING", "1").lower() in ("1", "true", "yes")
FACEBANK_MATCH_SIMILARITY = float(os.getenv("FACEBANK_MATCH_SIMILARITY", "0.55"))  # Min similarity for facebank match

# Enhancement: Protect manual assignments by default
PROTECT_MANUAL_DEFAULT = os.getenv("PROTECT_MANUAL_DEFAULT", "1").lower() in ("1", "true", "yes")

# Enhancement: Minimum cluster size for reliable matching
MIN_FACES_FOR_RELIABLE_MATCH = int(os.getenv("MIN_FACES_FOR_RELIABLE_MATCH", "3"))
SMALL_CLUSTER_DISTANCE_PENALTY = float(os.getenv("SMALL_CLUSTER_DISTANCE_PENALTY", "0.05"))

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


def _normalize_centroids_to_map(
    centroids: Any,
    validate: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Normalize centroid data (dict or list format) into a unified dict format.

    Args:
        centroids: Centroid data (dict keyed by cluster_id or list of dicts)
        validate: If True, validate vector dimensions and skip invalid entries

    Returns:
        Dict mapping cluster_id -> {"centroid": np.ndarray, "seed_cast_id": str|None, ...}
    """
    result: Dict[str, Dict[str, Any]] = {}

    if isinstance(centroids, dict):
        # New format: dict keyed by cluster_id
        for cid, data in centroids.items():
            if not isinstance(data, dict):
                continue
            centroid = data.get("centroid")
            if centroid is None:
                continue
            vec = np.array(centroid, dtype=np.float32)
            # Validate vector
            if validate and (len(vec) < MIN_EMBEDDING_DIM or not np.isfinite(vec).all()):
                LOGGER.warning(f"Skipping invalid centroid for cluster {cid}: dim={len(vec)}")
                continue
            result[str(cid)] = {
                "centroid": vec,
                "seed_cast_id": data.get("seed_cast_id"),
                "seed_confidence": data.get("seed_confidence"),
                "cohesion": data.get("cohesion"),
                "num_faces": data.get("num_faces"),
            }
    elif isinstance(centroids, list):
        # Legacy format: list of dicts with cluster_id field
        for entry in centroids:
            if not isinstance(entry, dict):
                continue
            cid = entry.get("cluster_id")
            centroid = entry.get("centroid")
            if not cid or centroid is None:
                continue
            vec = np.array(centroid, dtype=np.float32)
            # Validate vector
            if validate and (len(vec) < MIN_EMBEDDING_DIM or not np.isfinite(vec).all()):
                LOGGER.warning(f"Skipping invalid centroid for cluster {cid}: dim={len(vec)}")
                continue
            result[str(cid)] = {
                "centroid": vec,
                "seed_cast_id": entry.get("seed_cast_id"),
                "seed_confidence": entry.get("seed_confidence"),
                "cohesion": entry.get("cohesion"),
                "num_faces": entry.get("num_faces"),
            }

    return result


def _compute_cohesion_bonus(
    cohesion_i: Optional[float],
    cohesion_j: Optional[float],
) -> float:
    """Compute distance bonus based on cluster cohesion.

    High cohesion clusters are more reliable, so we reduce the distance
    threshold when matching them (making it easier to merge).

    Args:
        cohesion_i: Cohesion score of first cluster (0.0-1.0)
        cohesion_j: Cohesion score of second cluster (0.0-1.0)

    Returns:
        Distance reduction bonus (0.0 to COHESION_BONUS_MAX)
    """
    if not COHESION_WEIGHT_ENABLED:
        return 0.0

    if cohesion_i is None or cohesion_j is None:
        return 0.0

    # Average cohesion of both clusters
    avg_cohesion = (cohesion_i + cohesion_j) / 2.0

    # Scale bonus: higher cohesion = bigger bonus (more likely to match)
    # cohesion of 0.9+ gets full bonus, 0.7 gets half, below 0.5 gets none
    if avg_cohesion >= 0.9:
        return COHESION_BONUS_MAX
    elif avg_cohesion >= 0.7:
        # Linear interpolation between 0.7 and 0.9
        return COHESION_BONUS_MAX * (avg_cohesion - 0.7) / 0.2
    else:
        return 0.0


def _compute_size_penalty(
    num_faces_i: Optional[int],
    num_faces_j: Optional[int],
) -> float:
    """Compute distance penalty for small clusters.

    Small clusters are less reliable, so we increase the distance
    threshold (making it harder to merge them).

    Args:
        num_faces_i: Number of faces in first cluster
        num_faces_j: Number of faces in second cluster

    Returns:
        Distance penalty (0.0 to SMALL_CLUSTER_DISTANCE_PENALTY)
    """
    if num_faces_i is None or num_faces_j is None:
        return 0.0

    min_faces = min(num_faces_i, num_faces_j)

    if min_faces >= MIN_FACES_FOR_RELIABLE_MATCH:
        return 0.0
    elif min_faces <= 1:
        return SMALL_CLUSTER_DISTANCE_PENALTY
    else:
        # Linear interpolation
        return SMALL_CLUSTER_DISTANCE_PENALTY * (1.0 - (min_faces - 1) / (MIN_FACES_FOR_RELIABLE_MATCH - 1))


def _compute_group_centroid(
    cluster_ids: List[str],
    centroids_map: Dict[str, Dict[str, Any]],
) -> Optional[np.ndarray]:
    """Compute the combined centroid for a group of clusters.

    This avoids the momentum race condition by computing the mean of all
    cluster centroids in a group at once, rather than sequentially updating
    with momentum.

    Args:
        cluster_ids: List of cluster IDs in the group
        centroids_map: Map of cluster_id -> centroid data

    Returns:
        L2-normalized mean centroid, or None if no valid centroids
    """
    vectors = []
    for cid in cluster_ids:
        data = centroids_map.get(cid)
        if data is not None and data.get("centroid") is not None:
            vectors.append(data["centroid"])

    if not vectors:
        return None

    mean_vec = np.mean(vectors, axis=0)
    return l2_normalize(mean_vec)


class GroupingService:
    """Handle cluster centroid computation and grouping."""

    def __init__(self, data_root: Path | str | None = None):
        self.data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
        self.people_service = PeopleService(data_root)
        self.facebank_service = FacebankService(data_root)
        self.cast_service = CastService(data_root)

    def backup_before_cleanup(self, ep_id: str) -> Dict[str, Any]:
        """Create backup of identities.json, people.json, cluster_centroids.json (Enhancement #7).

        Returns: {"backup_id": str, "files": [path, ...], "timestamp": str}
        """
        from datetime import datetime
        import shutil

        parsed = _parse_ep_id(ep_id)
        if not parsed:
            raise ValueError(f"Invalid episode ID: {ep_id}")
        show_id = parsed["show"]

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_id = f"cleanup_{timestamp}"

        # Create backup directory
        backup_dir = self._identities_path(ep_id).parent / "backups" / backup_id
        backup_dir.mkdir(parents=True, exist_ok=True)

        backed_up_files = []

        # Backup identities.json
        identities_path = self._identities_path(ep_id)
        if identities_path.exists():
            backup_path = backup_dir / "identities.json"
            shutil.copy2(identities_path, backup_path)
            backed_up_files.append(str(backup_path))

        # Backup cluster_centroids.json
        centroids_path = self._cluster_centroids_path(ep_id)
        if centroids_path.exists():
            backup_path = backup_dir / "cluster_centroids.json"
            shutil.copy2(centroids_path, backup_path)
            backed_up_files.append(str(backup_path))

        # Backup people.json
        people_path = self.data_root / "shows" / show_id.lower() / "people.json"
        if people_path.exists():
            backup_path = backup_dir / "people.json"
            shutil.copy2(people_path, backup_path)
            backed_up_files.append(str(backup_path))

        # Save backup metadata
        metadata = {
            "backup_id": backup_id,
            "ep_id": ep_id,
            "show_id": show_id,
            "timestamp": _now_iso(),
            "files": backed_up_files,
        }
        metadata_path = backup_dir / "backup_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        LOGGER.info(f"[{ep_id}] Created cleanup backup {backup_id} with {len(backed_up_files)} file(s)")
        return metadata

    def restore_from_backup(self, ep_id: str, backup_id: str) -> Dict[str, Any]:
        """Restore from a cleanup backup (Enhancement #7).

        Returns: {"restored": int, "files": [path, ...]}
        """
        import shutil

        parsed = _parse_ep_id(ep_id)
        if not parsed:
            raise ValueError(f"Invalid episode ID: {ep_id}")
        show_id = parsed["show"]

        backup_dir = self._identities_path(ep_id).parent / "backups" / backup_id
        if not backup_dir.exists():
            raise FileNotFoundError(f"Backup not found: {backup_id}")

        restored_files = []

        # Restore identities.json
        backup_identities = backup_dir / "identities.json"
        if backup_identities.exists():
            target = self._identities_path(ep_id)
            shutil.copy2(backup_identities, target)
            restored_files.append(str(target))

        # Restore cluster_centroids.json
        backup_centroids = backup_dir / "cluster_centroids.json"
        if backup_centroids.exists():
            target = self._cluster_centroids_path(ep_id)
            shutil.copy2(backup_centroids, target)
            restored_files.append(str(target))

        # Restore people.json
        backup_people = backup_dir / "people.json"
        if backup_people.exists():
            target = self.data_root / "shows" / show_id.lower() / "people.json"
            shutil.copy2(backup_people, target)
            restored_files.append(str(target))

        LOGGER.info(f"[{ep_id}] Restored {len(restored_files)} file(s) from backup {backup_id}")
        return {
            "restored": len(restored_files),
            "files": restored_files,
            "backup_id": backup_id,
        }

    def list_backups(self, ep_id: str) -> List[Dict[str, Any]]:
        """List available cleanup backups for an episode."""
        backup_parent = self._identities_path(ep_id).parent / "backups"
        if not backup_parent.exists():
            return []

        backups = []
        for backup_dir in sorted(backup_parent.iterdir(), reverse=True):
            if not backup_dir.is_dir():
                continue
            metadata_path = backup_dir / "backup_metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                backups.append(metadata)
            else:
                backups.append({
                    "backup_id": backup_dir.name,
                    "ep_id": ep_id,
                    "timestamp": None,
                })
        return backups[:10]  # Return last 10 backups

    def _load_manual_assignments(self, ep_id: str) -> Dict[str, Dict[str, Any]]:
        """Load manual assignment metadata for an episode.

        Returns: Dict mapping cluster_id -> {"assigned_by": "user"|"auto", "timestamp": "...", "cast_id": "..."}
        """
        identities_path = self._identities_path(ep_id)
        if not identities_path.exists():
            return {}

        data = json.loads(identities_path.read_text(encoding="utf-8"))
        return data.get("manual_assignments", {})

    def _save_manual_assignments(
        self,
        ep_id: str,
        manual_assignments: Dict[str, Dict[str, Any]],
    ) -> None:
        """Save manual assignment metadata for an episode."""
        identities_path = self._identities_path(ep_id)
        if not identities_path.exists():
            return

        data = json.loads(identities_path.read_text(encoding="utf-8"))
        data["manual_assignments"] = manual_assignments
        identities_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def mark_assignment_manual(
        self,
        ep_id: str,
        cluster_id: str,
        cast_id: Optional[str] = None,
    ) -> None:
        """Mark a cluster assignment as manually made by user (Enhancement #2)."""
        manual_assignments = self._load_manual_assignments(ep_id)
        manual_assignments[cluster_id] = {
            "assigned_by": "user",
            "timestamp": _now_iso(),
            "cast_id": cast_id,
        }
        self._save_manual_assignments(ep_id, manual_assignments)

    def is_manually_assigned(self, ep_id: str, cluster_id: str) -> bool:
        """Check if a cluster was manually assigned by user."""
        manual_assignments = self._load_manual_assignments(ep_id)
        entry = manual_assignments.get(cluster_id, {})
        return entry.get("assigned_by") == "user"

    def get_manual_cluster_ids(self, ep_id: str) -> List[str]:
        """Get all cluster IDs that were manually assigned."""
        manual_assignments = self._load_manual_assignments(ep_id)
        return [
            cid for cid, data in manual_assignments.items()
            if data.get("assigned_by") == "user"
        ]

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

    def _group_progress_path(self, ep_id: str) -> Path:
        """Get path to group_progress.json for in-flight progress."""
        manifests_dir = get_path(ep_id, "detections").parent
        return manifests_dir / "group_progress.json"

    def _write_progress(
        self,
        ep_id: str,
        entries: List[Dict[str, Any]],
        *,
        started_at: str,
        finished: bool = False,
        error: str | None = None,
    ) -> None:
        """Persist incremental progress so UI can poll while the API runs."""
        payload: Dict[str, Any] = {
            "ep_id": ep_id,
            "started_at": started_at,
            "updated_at": _now_iso(),
            "finished": finished,
            "entries": entries,
        }
        if error:
            payload["error"] = error
        path = self._group_progress_path(ep_id)
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as e:
            LOGGER.error(f"[{ep_id}] Failed to write progress file: {e}")
            # Don't raise - progress updates are non-critical

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

        # Build reverse map: track_id -> cluster_id for O(1) lookups (fixes N+1 pattern)
        track_to_cluster: Dict[int, str] = {}
        for cluster_id, track_ids in cluster_to_tracks.items():
            for tid in track_ids:
                track_to_cluster[int(tid)] = cluster_id

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

                # O(1) lookup for cluster using reverse map
                cluster_id = track_to_cluster.get(int(track_id))
                if cluster_id:
                    emb_vec = np.array(embedding, dtype=np.float32)
                    cluster_embeddings[cluster_id].append(emb_vec)
                    cluster_counts[cluster_id] += 1
                    if seed_cast_id:
                        cluster_seed_matches[cluster_id].append(seed_cast_id)

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
        protect_manual: bool = False,
    ) -> Dict[str, Any]:
        """Perform agglomerative clustering on cluster centroids within an episode.

        Args:
            ep_id: Episode identifier
            distance_threshold: Maximum distance for grouping clusters
            progress_callback: Optional function(current, total, status) for progress updates
            protect_manual: If True, don't merge clusters that have been manually assigned
                           to different cast members (Enhancement #2)

        Returns: {"groups": [{person_id, cluster_ids}, ...], "merged_count": int}
        """
        import logging

        LOGGER = logging.getLogger(__name__)

        if not HAS_SKLEARN:
            raise RuntimeError("sklearn not available; install with: pip install scikit-learn")

        LOGGER.info(f"[cluster_cleanup] Grouping clusters within {ep_id} (protect_manual={protect_manual})")
        if progress_callback:
            progress_callback(0, 2, "Loading cluster centroids")

        # Load centroids and normalize to unified format with validation
        centroids_data = self.load_cluster_centroids(ep_id)
        centroids_map = _normalize_centroids_to_map(centroids_data.get("centroids", {}), validate=True)

        if len(centroids_map) <= 1:
            # Nothing to group
            return {"groups": [], "merged_count": 0}

        # Load manual assignments if protection is enabled
        manual_assignments = {}
        manual_cluster_ids = set()
        if protect_manual:
            manual_assignments = self._load_manual_assignments(ep_id)
            manual_cluster_ids = {
                cid for cid, data in manual_assignments.items()
                if data.get("assigned_by") == "user"
            }
            LOGGER.info(f"[cluster_cleanup] Protecting {len(manual_cluster_ids)} manually assigned cluster(s)")

        # Load identities to get person_id assignments
        identities_path = self._identities_path(ep_id)
        cluster_to_person: Dict[str, str] = {}
        if identities_path.exists():
            identities_data = json.loads(identities_path.read_text(encoding="utf-8"))
            for identity in identities_data.get("identities", []):
                cid = identity.get("identity_id")
                pid = identity.get("person_id")
                if cid and pid:
                    cluster_to_person[cid] = pid

        # Extract cluster IDs, centroid vectors, and metadata for smart distance calculation
        cluster_ids = list(centroids_map.keys())
        vectors = np.array([centroids_map[cid]["centroid"] for cid in cluster_ids], dtype=np.float32)
        seed_cast_ids = [centroids_map[cid].get("seed_cast_id") for cid in cluster_ids]
        cohesions = [centroids_map[cid].get("cohesion") for cid in cluster_ids]
        num_faces_list = [centroids_map[cid].get("num_faces") for cid in cluster_ids]

        # Compute pairwise cosine distances with smart adjustments
        n = len(vectors)
        distance_matrix = np.zeros((n, n), dtype=np.float32)
        seed_adjustments = 0
        cohesion_adjustments = 0
        size_adjustments = 0
        protected_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                dist = cosine_distance(vectors[i], vectors[j])

                # Enhancement: Apply cohesion-based distance reduction
                # High cohesion clusters are more reliable and should merge more easily
                cohesion_bonus = _compute_cohesion_bonus(cohesions[i], cohesions[j])
                if cohesion_bonus > 0:
                    dist = max(0.0, dist - cohesion_bonus)
                    cohesion_adjustments += 1

                # Enhancement: Apply size-based distance penalty
                # Small clusters are less reliable and should be harder to merge
                size_penalty = _compute_size_penalty(num_faces_list[i], num_faces_list[j])
                if size_penalty > 0:
                    dist = dist + size_penalty
                    size_adjustments += 1

                # Apply seed-based distance reduction if both clusters match same seed
                if seed_cast_ids[i] and seed_cast_ids[j] and seed_cast_ids[i] == seed_cast_ids[j]:
                    dist = max(0.0, dist - SEED_CLUSTER_DELTA)
                    seed_adjustments += 1

                # Enhancement #2: Protect manual assignments
                # If both clusters are manually assigned to DIFFERENT people, set max distance
                cid_i, cid_j = cluster_ids[i], cluster_ids[j]
                if protect_manual:
                    both_manual = cid_i in manual_cluster_ids and cid_j in manual_cluster_ids
                    if both_manual:
                        person_i = cluster_to_person.get(cid_i)
                        person_j = cluster_to_person.get(cid_j)
                        # If assigned to different people, prevent merging
                        if person_i and person_j and person_i != person_j:
                            dist = float("inf")  # Prevent merging
                            protected_pairs += 1

                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        LOGGER.info(
            f"[cluster_cleanup] Distance adjustments: seed={seed_adjustments}, "
            f"cohesion={cohesion_adjustments}, size={size_adjustments}"
        )

        # Agglomerative clustering
        LOGGER.info(
            f"[cluster_cleanup] Running agglomerative clustering on {n} centroids "
            f"(threshold={distance_threshold}, protected_pairs={protected_pairs})"
        )
        if progress_callback:
            progress_callback(1, 2, f"Clustering {n} centroids")

        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="complete",
        )
        labels = model.fit_predict(distance_matrix)

        # Group clusters by label
        groups_map: Dict[int, List[str]] = {}
        for cluster_id, label in zip(cluster_ids, labels):
            groups_map.setdefault(int(label), []).append(cluster_id)

        # Filter groups with more than one cluster (actual merges)
        merged_groups = [cids for cids in groups_map.values() if len(cids) > 1]

        LOGGER.info(
            f"[cluster_cleanup] Found {len(merged_groups)} merged groups "
            f"(seed={seed_adjustments}, cohesion={cohesion_adjustments}, size={size_adjustments}, protected={protected_pairs})"
        )
        if progress_callback:
            progress_callback(2, 2, f"Found {len(merged_groups)} merged groups")

        return {
            "groups": [{"cluster_ids": cids} for cids in merged_groups],
            "merged_count": len(merged_groups),
            "all_labels": labels.tolist(),
            "seed_adjustments": seed_adjustments,
            "cohesion_adjustments": cohesion_adjustments,
            "size_adjustments": size_adjustments,
            "protected_pairs": protected_pairs,
        }

    def group_across_episodes(
        self,
        ep_id: str,
        max_distance: float = PEOPLE_MATCH_DISTANCE,
        momentum: float = PEOPLE_PROTO_MOMENTUM,
        auto_assign: bool = False,
        progress_callback: Optional[Callable[[str, float, str, Optional[Dict[str, Any]]], None]] = None,
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

        # Load centroids with validation
        centroids_data = self.load_cluster_centroids(ep_id)
        centroids_map = _normalize_centroids_to_map(centroids_data.get("centroids", {}), validate=True)

        assigned = []
        new_people = []
        suggestions = []
        total_clusters = len(centroids_map)
        processed_clusters = 0

        # Pre-load people and their cast links for seed-based matching
        people = self.people_service.list_people(show_id)
        person_cast_map: Dict[str, str] = {}  # person_id -> cast_id
        cast_person_map: Dict[str, str] = {}  # cast_id -> person_id
        for person in people:
            cast_id = person.get("cast_id")
            if cast_id:
                person_cast_map[person["person_id"]] = cast_id
                cast_person_map[cast_id] = person["person_id"]

        for cluster_id, centroid_data in centroids_map.items():
            centroid = centroid_data["centroid"]
            seed_cast_id = centroid_data.get("seed_cast_id")

            # Try to find matching person
            match = self.people_service.find_matching_person(show_id, centroid, max_distance)

            # Apply seed-based matching: if cluster has a seed, prefer matching to a person
            # linked to that cast member (with reduced distance threshold)
            if USE_SEED_IN_ACROSS_EPISODE and seed_cast_id:
                # Check if there's a person already linked to this cast
                seed_person_id = cast_person_map.get(seed_cast_id)
                if seed_person_id:
                    # Check distance to this person specifically
                    seed_person = self.people_service.get_person(show_id, seed_person_id)
                    if seed_person and seed_person.get("prototype"):
                        proto_vec = np.array(seed_person["prototype"], dtype=np.float32)
                        seed_distance = cosine_distance(centroid, proto_vec)
                        # Apply seed delta bonus - reduce threshold for seed matches
                        adjusted_threshold = max_distance + SEED_CLUSTER_DELTA
                        if seed_distance <= adjusted_threshold:
                            # Prefer seed match over regular match
                            if not match or seed_distance < match[1]:
                                match = (seed_person_id, seed_distance)
                                LOGGER.debug(
                                    f"Seed match: cluster {cluster_id} -> person {seed_person_id} "
                                    f"(seed={seed_cast_id}, dist={seed_distance:.3f})"
                                )

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
                    assigned.append(
                        {
                            "cluster_id": cluster_id,
                            "person_id": person_id,
                            "distance": float(distance),
                            "suggested": False,
                            "seed_match": seed_cast_id is not None and cast_person_map.get(seed_cast_id) == person_id,
                        }
                    )
                    # Update cast_person_map if this person now has a cast_id
                    if person_id not in person_cast_map and seed_cast_id:
                        # Consider linking this person to the cast member
                        pass  # Done separately via manual assignment
                else:
                    # Just store suggestion without assigning
                    suggestions.append(
                        {
                            "cluster_id": cluster_id,
                            "suggested_person_id": person_id,
                            "distance": float(distance),
                        }
                    )
            else:
                if auto_assign:
                    # Create new person with the cluster's seed_cast_id if available
                    full_cluster_id = f"{ep_id}:{cluster_id}"
                    person = self.people_service.create_person(
                        show_id,
                        prototype=centroid.tolist(),
                        cluster_ids=[full_cluster_id],
                        cast_id=seed_cast_id if seed_cast_id and seed_cast_id not in cast_person_map else None,
                    )
                    new_people.append(person)
                    assigned.append(
                        {
                            "cluster_id": cluster_id,
                            "person_id": person["person_id"],
                            "distance": None,
                            "suggested": False,
                            "seed_match": False,
                        }
                    )
                    # Track the new person's cast link
                    if seed_cast_id and seed_cast_id not in cast_person_map:
                        cast_person_map[seed_cast_id] = person["person_id"]
                        person_cast_map[person["person_id"]] = seed_cast_id

            processed_clusters += 1
            if progress_callback and total_clusters:
                pct = processed_clusters / max(total_clusters, 1)
                progress_callback(
                    "group_across_episodes",
                    0.7 + 0.2 * pct,
                    f"Processed {processed_clusters}/{total_clusters} clusters; assigned {len(assigned)}",
                    {
                        "total_clusters": total_clusters,
                        "processed_clusters": processed_clusters,
                        "assigned_clusters": len(assigned),
                        "new_people": len(new_people),
                    },
                )

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
                removed_clusters = result.get("clusters_removed", 0)
                removed_people = result.get("empty_people_removed", 0)
                LOGGER.info(
                    f"[{ep_id}] Removed {removed_clusters} cluster(s) from people.json ({removed_people} empty people deleted)"
                )
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

    def _prune_empty_people(self, show_id: str) -> int:
        """Delete auto-created people that no longer own any clusters."""
        removed = 0
        people = self.people_service.list_people(show_id)
        for person in people:
            if person.get("cluster_ids") or person.get("cast_id") or person.get("name"):
                continue
            if self.people_service.delete_person(show_id, person["person_id"]):
                removed += 1
        return removed

    def group_clusters_auto(
        self,
        ep_id: str,
        *,
        progress_callback=None,
        protect_manual: bool | None = None,
        facebank_first: bool | None = None,
    ) -> Dict[str, Any]:
        """Run full auto grouping: compute centroids, within-episode, across-episode.

        Enhanced algorithm:
        1. Compute cluster centroids with cohesion metrics
        2. Try facebank matching first (if enabled) - matches clusters to known cast members
        3. Within-episode grouping with cohesion-weighted distances
        4. Across-episode matching to show-level people

        Args:
            ep_id: Episode ID
            progress_callback: Optional callback(step: str, progress: float, message: str)
            protect_manual: If True, don't merge clusters that have been manually assigned
                           to different cast members. Defaults to PROTECT_MANUAL_DEFAULT.
            facebank_first: If True, try matching clusters to facebank seeds before
                           using people prototypes. Defaults to FACEBANK_FIRST_MATCHING.

        Returns combined result with audit log.
        """
        # Apply defaults from environment config
        if protect_manual is None:
            protect_manual = PROTECT_MANUAL_DEFAULT
        if facebank_first is None:
            facebank_first = FACEBANK_FIRST_MATCHING
        parsed = _parse_ep_id(ep_id)
        if not parsed:
            raise ValueError(f"Invalid episode ID: {ep_id}")
        show_id = parsed["show"]

        progress_entries: List[Dict[str, Any]] = []

        def _progress(step: str, pct: float, msg: str, meta: Optional[Dict[str, Any]] = None):
            entry: Dict[str, Any] = {
                "step": step,
                "progress": pct,
                "message": msg,
            }
            if meta:
                entry.update(meta)
            progress_entries.append(entry)
            if progress_callback:
                progress_callback(step, pct, msg)
            LOGGER.info(f"[{ep_id}] {step}: {msg} ({int(pct*100)}%)")
            self._write_progress(
                ep_id,
                progress_entries,
                started_at=log["started_at"],
                finished=False,
            )

        log = {
            "ep_id": ep_id,
            "started_at": _now_iso(),
            "steps": [],
        }
        # Initialize progress file so UI polling can show immediate state
        self._write_progress(ep_id, progress_entries, started_at=log["started_at"], finished=False)

        # Step 0: Clear stale person_id assignments from previous runs
        _progress("clear_assignments", 0.0, "Clearing stale person assignments...")
        try:
            cleared = self._clear_person_assignments(ep_id)
            log["steps"].append(
                {
                    "step": "clear_assignments",
                    "status": "success",
                    "cleared_count": cleared,
                }
            )
            _progress("clear_assignments", 0.1, f"Cleared {cleared} stale assignment(s)")
        except Exception as e:
            log["steps"].append({"step": "clear_assignments", "status": "error", "error": str(e)})
            LOGGER.warning(f"[{ep_id}] Failed to clear assignments: {e}")

        # Step 1: Compute centroids
        _progress("compute_centroids", 0.1, "Computing cluster centroids...")
        try:
            centroids_result = self.compute_cluster_centroids(ep_id)
            centroids_count = len(centroids_result.get("centroids", []))
            log["steps"].append(
                {
                    "step": "compute_centroids",
                    "status": "success",
                    "centroids_count": centroids_count,
                }
            )
            _progress(
                "compute_centroids",
                0.4,
                f"Computed {centroids_count} centroid(s)",
                {"total_clusters": centroids_count, "processed_clusters": 0},
            )
        except Exception as e:
            log["steps"].append({"step": "compute_centroids", "status": "error", "error": str(e)})
            log["finished_at"] = _now_iso()
            self._save_group_log(ep_id, log)
            _progress("compute_centroids", 0.4, f"ERROR: {str(e)}")
            self._write_progress(ep_id, progress_entries, started_at=log["started_at"], finished=True, error=str(e))
            raise

        # Normalize centroids into a unified format with validation
        centroids_map_full = _normalize_centroids_to_map(centroids_result.get("centroids", {}), validate=True)
        # Also create a simple centroid-only map for backward compatibility
        centroids_map: Dict[str, np.ndarray] = {
            cid: data["centroid"] for cid, data in centroids_map_full.items()
        }

        # Step 1.5: Facebank-first matching (if enabled)
        # Try to match clusters to known cast members via their facebank seeds
        # This provides stronger identity signals than cross-episode matching
        facebank_matches: Dict[str, str] = {}  # cluster_id -> cast_id
        facebank_result: Dict[str, Any] = {}
        if facebank_first:
            _progress("facebank_match", 0.35, "Matching clusters to cast facebank seeds...")
            try:
                from apps.api.services.facebank import cosine_similarity as fb_cosine_similarity

                seeds = self.facebank_service.get_all_seeds_for_show(show_id)
                if seeds:
                    # Group seeds by cast_id
                    seeds_by_cast: Dict[str, List[Dict[str, Any]]] = {}
                    for seed in seeds:
                        cast_id = seed.get("cast_id")
                        if cast_id and seed.get("embedding"):
                            seeds_by_cast.setdefault(cast_id, []).append(seed)

                    # For each cluster, find best matching cast member
                    matched_count = 0
                    for cluster_id, centroid_data in centroids_map_full.items():
                        centroid_vec = centroid_data["centroid"]
                        best_cast_id = None
                        best_similarity = 0.0

                        for cast_id, cast_seeds in seeds_by_cast.items():
                            for seed in cast_seeds:
                                seed_emb = np.array(seed["embedding"], dtype=np.float32)
                                sim = float(fb_cosine_similarity(centroid_vec, seed_emb))
                                if sim > best_similarity:
                                    best_similarity = sim
                                    best_cast_id = cast_id

                        # Only accept high-confidence matches
                        if best_cast_id and best_similarity >= FACEBANK_MATCH_SIMILARITY:
                            facebank_matches[cluster_id] = best_cast_id
                            matched_count += 1
                            LOGGER.debug(
                                f"[{ep_id}] Facebank match: {cluster_id} -> {best_cast_id} "
                                f"(similarity={best_similarity:.3f})"
                            )

                    facebank_result = {
                        "matched_count": matched_count,
                        "total_seeds": len(seeds),
                        "cast_members_with_seeds": len(seeds_by_cast),
                    }
                    log["steps"].append({
                        "step": "facebank_match",
                        "status": "success",
                        "matched_count": matched_count,
                        "total_seeds": len(seeds),
                    })
                    _progress(
                        "facebank_match",
                        0.4,
                        f"Matched {matched_count} cluster(s) to cast via facebank",
                    )
                else:
                    log["steps"].append({
                        "step": "facebank_match",
                        "status": "skipped",
                        "reason": "No facebank seeds available",
                    })
                    _progress("facebank_match", 0.4, "No facebank seeds available - skipping")
            except Exception as e:
                LOGGER.warning(f"[{ep_id}] Facebank matching failed: {e}")
                log["steps"].append({
                    "step": "facebank_match",
                    "status": "error",
                    "error": str(e),
                })
                _progress("facebank_match", 0.4, f"Facebank matching failed: {e}")

        # Step 2: Within-episode grouping
        _progress("group_within_episode", 0.4, "Grouping similar clusters within episode...")
        within_result: Dict[str, Any] = {}
        within_episode_failed = False
        try:
            within_result = self.group_within_episode(ep_id, protect_manual=protect_manual)
            merged = within_result.get("merged_count", 0)
            seed_adjustments = within_result.get("seed_adjustments", 0)
            log["steps"].append(
                {
                    "step": "group_within_episode",
                    "status": "success",
                    "merged_count": merged,
                    "seed_adjustments": seed_adjustments,
                }
            )
            _progress("group_within_episode", 0.7, f"Merged {merged} cluster group(s) (seed adjustments: {seed_adjustments})")
        except Exception as e:
            within_episode_failed = True
            LOGGER.warning(f"[{ep_id}] Within-episode grouping failed: {e}", exc_info=True)
            log["steps"].append({
                "step": "group_within_episode",
                "status": "error",
                "error": str(e),
                "warning": "Continuing with across-episode matching only",
            })
            _progress("group_within_episode", 0.7, f"WARNING: Within-episode grouping failed ({str(e)}) - continuing with cross-episode matching")
            self._write_progress(ep_id, progress_entries, started_at=log["started_at"], finished=False)
            # Continue even if within-episode grouping fails - clusters can still be matched across episodes

        # Step 3: Across-episode matching to people (auto-assign + create)
        _progress(
            "group_across_episodes",
            0.7,
            "Assigning clusters to show-level people...",
            {"total_clusters": len(centroids_map), "processed_clusters": 0},
        )
        assignments: List[Dict[str, Any]] = []
        new_people_count = 0
        try:
            across_result = self.group_across_episodes(
                ep_id,
                auto_assign=True,
                progress_callback=lambda step, pct, msg, meta=None: _progress(step, pct, msg, meta),
            )
            assignments = across_result.get("assigned", []) if isinstance(across_result, dict) else []
            new_people_count = across_result.get("new_people_count", 0) if isinstance(across_result, dict) else 0
            _progress(
                "group_across_episodes",
                0.9,
                f"Assigned {len(assignments)} cluster(s); created {new_people_count} new people",
            )
        except Exception as e:
            log["steps"].append({"step": "group_across_episodes", "status": "error", "error": str(e)})
            log["finished_at"] = _now_iso()
            self._save_group_log(ep_id, log)
            _progress("group_across_episodes", 0.9, f"ERROR: {str(e)}")
            self._write_progress(ep_id, progress_entries, started_at=log["started_at"], finished=True, error=str(e))
            raise

        # Step 4: Apply within-episode merges to the assigned people
        # Build groupings from within_result (if any)
        groups: List[List[str]] = []
        if isinstance(within_result, dict):
            for group_entry in within_result.get("groups", []) or []:
                if not isinstance(group_entry, dict):
                    continue
                cluster_ids = [str(cid) for cid in group_entry.get("cluster_ids", []) if cid]
                if len(cluster_ids) > 1:
                    groups.append(cluster_ids)

        # Ensure singletons are tracked so all clusters get assignments
        assigned_map: Dict[str, str] = {
            str(entry.get("cluster_id")): str(entry.get("person_id"))
            for entry in assignments
            if entry.get("cluster_id") and entry.get("person_id")
        }
        all_clusters = set(centroids_map.keys())
        grouped_clusters = set()
        for group in groups:
            grouped_clusters.update(group)
        for cid in sorted(all_clusters - grouped_clusters):
            groups.append([cid])

        merged_clusters = 0
        facebank_assigned = 0

        # Pre-load cast-person mappings for facebank matching
        people = self.people_service.list_people(show_id)
        cast_person_map: Dict[str, Dict[str, Any]] = {}
        for person in people:
            cast_id = person.get("cast_id")
            if cast_id:
                cast_person_map[cast_id] = person

        # Ensure all clusters that were grouped within the episode point at the same person
        # FIX: Use batch centroid computation to avoid momentum race condition
        for group in groups:
            if not group:
                continue
            base_cluster = group[0]
            base_person = assigned_map.get(base_cluster)

            # Enhancement: Check facebank matches first for any cluster in the group
            # If any cluster in the group has a facebank match, use that cast member's person
            facebank_cast_id = None
            for cid in group:
                if cid in facebank_matches:
                    facebank_cast_id = facebank_matches[cid]
                    break

            # Compute the combined centroid for the entire group ONCE
            # This avoids the momentum race condition from sequential updates
            group_centroid = _compute_group_centroid(group, centroids_map_full)

            # If we have a facebank match, prioritize it
            if facebank_cast_id and not base_person:
                cast_person = cast_person_map.get(facebank_cast_id)
                if cast_person:
                    # Use existing person for this cast member
                    base_person = cast_person["person_id"]
                    facebank_assigned += 1
                    LOGGER.info(f"[{ep_id}] Facebank assigned group {group} to cast {facebank_cast_id}")
                else:
                    # Create person for this cast member
                    if group_centroid is not None:
                        cast_meta = self.cast_service.get_cast_member(show_id, facebank_cast_id)
                        person = self.people_service.create_person(
                            show_id,
                            prototype=group_centroid.tolist(),
                            cluster_ids=[],
                            cast_id=facebank_cast_id,
                            name=cast_meta.get("name") if cast_meta else None,
                        )
                        base_person = person["person_id"]
                        cast_person_map[facebank_cast_id] = person
                        new_people_count += 1
                        facebank_assigned += 1
                        LOGGER.info(f"[{ep_id}] Created person for cast {facebank_cast_id} via facebank match")

            if not base_person:
                # No assignment for the lead cluster yet - create a new person for this group
                # FIX: Always use a valid prototype (group centroid) - never empty
                if group_centroid is None:
                    LOGGER.warning(f"[{ep_id}] Cannot create person for group {group}: no valid centroids")
                    continue  # Skip this group - can't create a person without prototype

                person = self.people_service.create_person(
                    show_id,
                    prototype=group_centroid.tolist(),
                    cluster_ids=[],
                )
                base_person = person["person_id"]
                new_people_count += 1
                LOGGER.info(f"[{ep_id}] Created new person {base_person} for group of {len(group)} clusters")

            # Add all clusters in the group to the same person
            # Use update_prototype=False to avoid momentum bias - we set the prototype directly
            for cid in group:
                previous_person = assigned_map.get(cid)
                if previous_person and previous_person != base_person:
                    merged_clusters += 1
                self.people_service.add_cluster_to_person(
                    show_id,
                    base_person,
                    f"{ep_id}:{cid}",
                    update_prototype=False,  # Don't use momentum - we set prototype directly
                    cluster_centroid=None,
                )
                assigned_map[cid] = base_person

            # Now set the prototype directly to the group centroid
            # This ensures consistent representation regardless of processing order
            if group_centroid is not None and len(group) > 1:
                self.people_service.update_person(
                    show_id,
                    base_person,
                    prototype=group_centroid.tolist(),
                )

        # Update identities.json with the final assignments and clean up empty people
        final_assignments = [{"cluster_id": cid, "person_id": pid} for cid, pid in assigned_map.items()]
        self._update_identities_with_people(ep_id, final_assignments)
        pruned_people = self._prune_empty_people(show_id)

        if isinstance(across_result, dict):
            across_result["assigned"] = final_assignments
            across_result["new_people_count"] = new_people_count
            across_result["pruned_people_count"] = pruned_people

        log["steps"].append(
            {
                "step": "group_across_episodes",
                "status": "success",
                "assigned_count": len(final_assignments),
                "new_people_count": new_people_count,
            }
        )
        log["steps"].append(
            {
                "step": "apply_within_groups",
                "status": "success",
                "groups": len(groups),
                "merged_clusters": merged_clusters,
                "facebank_assigned": facebank_assigned,
                "pruned_people": pruned_people,
            }
        )
        _progress(
            "apply_within_groups",
            1.0,
            f"Completed grouping for {len(centroids_map)} clusters",
            {
                "total_clusters": len(centroids_map),
                "processed_clusters": len(centroids_map),
                "assigned_clusters": len(final_assignments),
                "merged_clusters": merged_clusters,
                "facebank_assigned": facebank_assigned,
                "new_people": new_people_count,
            },
        )
        _progress(
            "apply_within_groups",
            1.0,
            f"Applied {len(groups)} group(s); merged {merged_clusters} cluster(s); facebank matched {facebank_assigned}",
        )

        log["finished_at"] = _now_iso()
        self._save_group_log(ep_id, log)
        self._write_progress(ep_id, progress_entries, started_at=log["started_at"], finished=True)

        return {
            "ep_id": ep_id,
            "centroids": centroids_result,
            "facebank_matching": facebank_result,
            "facebank_assigned": facebank_assigned,
            "within_episode": within_result,
            "across_episodes": across_result,
            "assignments": final_assignments,
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

        # Load centroids with validation using helper
        centroids_data = self.load_cluster_centroids(ep_id)
        centroids_map_full = _normalize_centroids_to_map(centroids_data.get("centroids", {}), validate=True)
        centroids_map = {cid: data["centroid"] for cid, data in centroids_map_full.items()}

        # If no target person, create new
        if not target_person_id:
            # Use mean of selected clusters as prototype
            centroids_to_merge = [centroids_map[cid] for cid in cluster_ids if cid in centroids_map]
            if not centroids_to_merge:
                raise ValueError("No valid clusters found")

            proto = l2_normalize(np.mean(centroids_to_merge, axis=0))
            LOGGER.info(
                "manual_assign_clusters creating person for cast",
                extra={"ep_id": ep_id, "show_id": show_id, "cast_id": cast_id, "cluster_ids": cluster_ids},
            )
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
            if cast_id:
                LOGGER.info(
                    "manual_assign_clusters updating existing person cast link",
                    extra={
                        "ep_id": ep_id,
                        "show_id": show_id,
                        "target_person_id": target_person_id,
                        "cast_id": cast_id,
                        "cluster_ids": cluster_ids,
                    },
                )
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

            # Ensure cast_id is persisted for existing person when provided
            if cast_id:
                self.people_service.update_person(show_id, target_person_id, cast_id=cast_id)

        # Update identities.json
        assignments = [{"cluster_id": cid, "person_id": target_person_id} for cid in cluster_ids]
        self._update_identities_with_people(ep_id, assignments)

        # Enhancement #2: Mark clusters as manually assigned for protection
        for cluster_id in cluster_ids:
            self.mark_assignment_manual(ep_id, cluster_id, cast_id=cast_id)

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

        # Load centroids with validation using helper
        centroids_data = self.load_cluster_centroids(ep_id)
        centroids_map_full = _normalize_centroids_to_map(centroids_data.get("centroids", {}), validate=True)

        if not centroids_map_full:
            raise ValueError(f"No valid cluster centroids found for {ep_id}")

        cast_members = self.cast_service.list_cast(show_id)
        cast_lookup = {member["cast_id"]: member for member in cast_members if member.get("cast_id")}
        people = self.people_service.list_people(show_id)
        cast_person_map = {person.get("cast_id"): person for person in people if person.get("cast_id")}

        assignments = []
        matched_clusters: List[Dict[str, Any]] = []
        unmatched_clusters: List[str] = []

        for cluster_id, centroid_data in centroids_map_full.items():
            centroid_vec = centroid_data["centroid"]
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
            matched_clusters.append(
                {
                    "cluster_id": cluster_id,
                    "person_id": person_id,
                    "cast_id": cast_id,
                    "seed_id": seed_id,
                    "similarity": (round(float(similarity), 4) if similarity is not None else None),
                }
            )

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

        # Load centroids with validation using helper
        centroids_data = self.load_cluster_centroids(ep_id)
        centroids_map_full = _normalize_centroids_to_map(centroids_data.get("centroids", {}), validate=True)
        centroids_map = {cid: data["centroid"] for cid, data in centroids_map_full.items()}

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

        LOGGER.info(
            f"[{ep_id}] Comparing {len(unassigned_clusters)} unassigned vs {len(assigned_clusters)} assigned clusters"
        )

        suggestions = []

        # For each unassigned cluster, find best match among assigned clusters
        for unassigned_id in unassigned_clusters:
            unassigned_centroid = centroids_map.get(unassigned_id)
            if unassigned_centroid is None:
                continue

            best_match_person = None
            best_distance = float("inf")

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
                suggestions.append(
                    {
                        "cluster_id": unassigned_id,
                        "suggested_person_id": best_match_person,
                        "distance": float(best_distance),
                    }
                )

        LOGGER.info(f"[{ep_id}] Generated {len(suggestions)} suggestions from assigned clusters")
        return {"suggestions": suggestions}

    def suggest_cast_for_unassigned_clusters(
        self,
        ep_id: str,
        min_similarity: float = 0.50,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Suggest cast members for unassigned clusters based on facebank similarity.

        For each unassigned cluster:
        1. Load all cast member facebanks for the show
        2. Compute cosine similarity: cluster_centroid vs each seed embedding
        3. Take best match per cast member (highest similarity seed)
        4. Rank cast members by best match score
        5. Return top_k suggestions with confidence %

        Args:
            ep_id: Episode ID
            min_similarity: Minimum similarity threshold for suggestions
            top_k: Number of top suggestions to return per cluster

        Returns:
            {
                "suggestions": [
                    {
                        "cluster_id": "id_001",
                        "cast_suggestions": [
                            {"cast_id": "...", "name": "Kyle", "similarity": 0.87, "confidence": "high"},
                            {"cast_id": "...", "name": "Lisa", "similarity": 0.72, "confidence": "medium"},
                        ]
                    },
                    ...
                ]
            }
        """
        from apps.api.services.facebank import cosine_similarity

        parsed = _parse_ep_id(ep_id)
        if not parsed:
            raise ValueError(f"Invalid episode ID: {ep_id}")
        show_id = parsed["show"]

        # Load identities to find unassigned clusters
        identities_path = self._identities_path(ep_id)
        if not identities_path.exists():
            raise FileNotFoundError(f"identities.json not found for {ep_id}")

        identities_data = json.loads(identities_path.read_text(encoding="utf-8"))
        identities = identities_data.get("identities", [])

        # Load centroids with validation
        try:
            centroids_data = self.load_cluster_centroids(ep_id)
            centroids_map = _normalize_centroids_to_map(centroids_data.get("centroids", {}), validate=True)
        except FileNotFoundError:
            return {"suggestions": [], "message": "No centroids found. Run clustering first."}

        # Get all seeds for the show
        seeds = self.facebank_service.get_all_seeds_for_show(show_id)
        if not seeds:
            return {"suggestions": [], "message": f"No facebank seeds available for show {show_id}"}

        # Get cast member names for display
        cast_members = self.cast_service.list_cast(show_id)
        cast_lookup = {member["cast_id"]: member for member in cast_members if member.get("cast_id")}

        # Group seeds by cast_id
        seeds_by_cast: Dict[str, List[Dict[str, Any]]] = {}
        for seed in seeds:
            cast_id = seed.get("cast_id")
            if cast_id and seed.get("embedding"):
                seeds_by_cast.setdefault(cast_id, []).append(seed)

        # Find unassigned clusters
        unassigned_clusters = []
        for identity in identities:
            cluster_id = identity.get("identity_id")
            person_id = identity.get("person_id")
            if not person_id and cluster_id in centroids_map:
                unassigned_clusters.append(cluster_id)

        suggestions = []
        for cluster_id in unassigned_clusters:
            centroid_data = centroids_map.get(cluster_id)
            if not centroid_data:
                continue

            centroid_vec = centroid_data["centroid"]

            # Find best similarity per cast member
            cast_matches: List[Dict[str, Any]] = []
            for cast_id, cast_seeds in seeds_by_cast.items():
                best_sim = -1.0
                best_seed_id = None

                for seed in cast_seeds:
                    seed_emb = np.array(seed["embedding"], dtype=np.float32)
                    sim = cosine_similarity(centroid_vec, seed_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_seed_id = seed.get("fb_id")

                if best_sim >= min_similarity:
                    cast_meta = cast_lookup.get(cast_id, {})
                    cast_name = cast_meta.get("name", cast_id)

                    # Determine confidence level
                    if best_sim >= 0.80:
                        confidence = "high"
                    elif best_sim >= 0.65:
                        confidence = "medium"
                    else:
                        confidence = "low"

                    cast_matches.append({
                        "cast_id": cast_id,
                        "name": cast_name,
                        "similarity": round(float(best_sim), 3),
                        "confidence": confidence,
                        "best_seed_id": best_seed_id,
                    })

            # Sort by similarity (descending) and take top_k
            cast_matches.sort(key=lambda x: x["similarity"], reverse=True)
            top_matches = cast_matches[:top_k]

            suggestions.append({
                "cluster_id": cluster_id,
                "cast_suggestions": top_matches,
            })

        LOGGER.info(f"[{ep_id}] Generated cast suggestions for {len(suggestions)} unassigned clusters")
        return {"suggestions": suggestions}

    def auto_link_high_confidence_matches(
        self,
        ep_id: str,
        min_confidence: float = 0.85,
    ) -> Dict[str, Any]:
        """Auto-assign unassigned clusters to cast members when confidence is high (Enhancement #8).

        Only assigns when similarity is >= min_confidence (default 85%).
        Returns summary of auto-assignments made.
        """
        # First, get all cast suggestions
        suggestions_result = self.suggest_cast_for_unassigned_clusters(
            ep_id,
            min_similarity=min_confidence,
            top_k=1,  # Only need top match
        )

        suggestions = suggestions_result.get("suggestions", [])
        if not suggestions:
            return {"auto_assigned": 0, "assignments": []}

        auto_assigned = []
        for suggestion in suggestions:
            cluster_id = suggestion.get("cluster_id")
            cast_suggestions = suggestion.get("cast_suggestions", [])

            if not cast_suggestions:
                continue

            top_match = cast_suggestions[0]
            similarity = top_match.get("similarity", 0)
            cast_id = top_match.get("cast_id")
            cast_name = top_match.get("name")

            # Only auto-assign if confidence is high enough
            if similarity >= min_confidence and cast_id:
                LOGGER.info(
                    f"[{ep_id}] Auto-linking cluster {cluster_id} to {cast_name} "
                    f"(similarity: {similarity:.2%})"
                )

                # Assign the cluster
                try:
                    result = self.manual_assign_clusters(
                        ep_id,
                        cluster_ids=[cluster_id],
                        cast_id=cast_id,
                    )
                    auto_assigned.append({
                        "cluster_id": cluster_id,
                        "cast_id": cast_id,
                        "cast_name": cast_name,
                        "similarity": similarity,
                        "person_id": result.get("person_id"),
                    })
                except Exception as e:
                    LOGGER.warning(f"Failed to auto-assign {cluster_id}: {e}")

        LOGGER.info(f"[{ep_id}] Auto-linked {len(auto_assigned)} cluster(s) to cast members")
        return {
            "auto_assigned": len(auto_assigned),
            "assignments": auto_assigned,
        }

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

        # Load centroids with validation using helper
        centroids_data = self.load_cluster_centroids(ep_id)
        centroids_map_full = _normalize_centroids_to_map(centroids_data.get("centroids", {}), validate=True)
        centroids_map = {cid: data["centroid"] for cid, data in centroids_map_full.items()}

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
