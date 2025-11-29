"""Archive service for storing deleted/rejected face identities.

Archived items can be used to:
1. View history of deleted items
2. Auto-archive matching faces in future episodes
3. Restore previously deleted items
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_ROOT = Path("data").expanduser()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    norm = np.linalg.norm(vector) + 1e-12
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_norm = l2_normalize(a)
    b_norm = l2_normalize(b)
    return float(np.dot(a_norm, b_norm))


class ArchiveService:
    """Manage archived (deleted) face identities at the show level.

    Archive stores:
    - People: Auto-clustered people that were deleted
    - Clusters: Individual clusters that were deleted
    - Tracks: Tracks that were explicitly deleted

    Each archived item includes:
    - Original data (name, cluster_ids, etc.)
    - Centroid embedding (for future matching)
    - Metadata (when deleted, from which episode, reason)
    """

    def __init__(self, data_root: Path | str | None = None):
        self.data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
        self.shows_dir = self.data_root / "shows"
        self.shows_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_show_id(show_id: str) -> str:
        """Normalize show_id to uppercase."""
        return show_id.upper()

    def _archive_path(self, show_id: str) -> Path:
        """Get path to archived.json for a show."""
        show_id = self.normalize_show_id(show_id)
        show_dir = self.shows_dir / show_id
        show_dir.mkdir(parents=True, exist_ok=True)
        return show_dir / "archived.json"

    def _load_archive(self, show_id: str) -> Dict[str, Any]:
        """Load archived.json or create empty structure."""
        normalized = self.normalize_show_id(show_id)
        path = self._archive_path(show_id)
        if not path.exists():
            return {
                "show_id": normalized,
                "archived_people": [],
                "archived_clusters": [],
                "archived_tracks": [],
                "stats": {
                    "total_archived": 0,
                    "last_updated": None,
                },
            }
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Ensure all required fields exist
            data.setdefault("archived_people", [])
            data.setdefault("archived_clusters", [])
            data.setdefault("archived_tracks", [])
            data.setdefault("stats", {"total_archived": 0, "last_updated": None})
            return data
        except json.JSONDecodeError:
            LOGGER.warning(f"Invalid archived.json for {show_id}, returning empty")
            return {
                "show_id": normalized,
                "archived_people": [],
                "archived_clusters": [],
                "archived_tracks": [],
                "stats": {"total_archived": 0, "last_updated": None},
            }

    def _save_archive(self, show_id: str, data: Dict[str, Any]) -> None:
        """Save archived.json."""
        # Update stats
        total = (
            len(data.get("archived_people", []))
            + len(data.get("archived_clusters", []))
            + len(data.get("archived_tracks", []))
        )
        data["stats"] = {
            "total_archived": total,
            "last_updated": _now_iso(),
        }
        path = self._archive_path(show_id)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def archive_person(
        self,
        show_id: str,
        person_data: Dict[str, Any],
        *,
        episode_id: Optional[str] = None,
        reason: str = "user_deleted",
        centroid: Optional[List[float]] = None,
        rep_crop_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Archive a deleted person.

        Args:
            show_id: Show identifier
            person_data: Original person data (person_id, name, cluster_ids, etc.)
            episode_id: Episode where deletion occurred
            reason: Reason for archiving (user_deleted, rejected, etc.)
            centroid: Face centroid embedding for future matching
            rep_crop_url: Representative crop URL

        Returns:
            The archived person record
        """
        data = self._load_archive(show_id)

        archive_id = f"arch_{uuid4().hex[:12]}"
        archived_record = {
            "archive_id": archive_id,
            "type": "person",
            "original_id": person_data.get("person_id"),
            "name": person_data.get("name"),
            "cluster_ids": person_data.get("cluster_ids", []),
            "episode_id": episode_id,
            "reason": reason,
            "archived_at": _now_iso(),
            "rep_crop_url": rep_crop_url or person_data.get("rep_crop_url"),
            "rep_crop_s3_key": person_data.get("rep_crop_s3_key"),
            "centroid": centroid,
            "original_data": person_data,
        }

        data["archived_people"].append(archived_record)
        self._save_archive(show_id, data)

        LOGGER.info(f"Archived person {person_data.get('person_id')} as {archive_id}")
        return archived_record

    def archive_cluster(
        self,
        show_id: str,
        episode_id: str,
        cluster_id: str,
        *,
        reason: str = "user_deleted",
        centroid: Optional[List[float]] = None,
        rep_crop_url: Optional[str] = None,
        track_ids: Optional[List[int]] = None,
        face_count: int = 0,
    ) -> Dict[str, Any]:
        """Archive a deleted cluster.

        Args:
            show_id: Show identifier
            episode_id: Episode identifier
            cluster_id: Cluster/identity ID
            reason: Reason for archiving
            centroid: Cluster centroid embedding
            rep_crop_url: Representative crop URL
            track_ids: List of track IDs in this cluster
            face_count: Number of faces in this cluster

        Returns:
            The archived cluster record
        """
        data = self._load_archive(show_id)

        archive_id = f"arch_{uuid4().hex[:12]}"
        archived_record = {
            "archive_id": archive_id,
            "type": "cluster",
            "original_id": cluster_id,
            "episode_id": episode_id,
            "reason": reason,
            "archived_at": _now_iso(),
            "rep_crop_url": rep_crop_url,
            "centroid": centroid,
            "track_ids": track_ids or [],
            "face_count": face_count,
        }

        data["archived_clusters"].append(archived_record)
        self._save_archive(show_id, data)

        LOGGER.info(f"Archived cluster {cluster_id} from {episode_id} as {archive_id}")
        return archived_record

    def archive_track(
        self,
        show_id: str,
        episode_id: str,
        track_id: int,
        *,
        reason: str = "user_deleted",
        centroid: Optional[List[float]] = None,
        rep_crop_url: Optional[str] = None,
        frame_count: int = 0,
        cluster_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Archive a deleted track.

        Args:
            show_id: Show identifier
            episode_id: Episode identifier
            track_id: Track ID
            reason: Reason for archiving
            centroid: Track centroid embedding
            rep_crop_url: Representative crop URL
            frame_count: Number of frames in this track
            cluster_id: Parent cluster ID if any

        Returns:
            The archived track record
        """
        data = self._load_archive(show_id)

        archive_id = f"arch_{uuid4().hex[:12]}"
        archived_record = {
            "archive_id": archive_id,
            "type": "track",
            "original_id": track_id,
            "episode_id": episode_id,
            "cluster_id": cluster_id,
            "reason": reason,
            "archived_at": _now_iso(),
            "rep_crop_url": rep_crop_url,
            "centroid": centroid,
            "frame_count": frame_count,
        }

        data["archived_tracks"].append(archived_record)
        self._save_archive(show_id, data)

        LOGGER.info(f"Archived track {track_id} from {episode_id} as {archive_id}")
        return archived_record

    def list_archived(
        self,
        show_id: str,
        *,
        item_type: Optional[str] = None,
        episode_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List archived items.

        Args:
            show_id: Show identifier
            item_type: Filter by type (person, cluster, track) or None for all
            episode_id: Filter by episode or None for all
            limit: Maximum items to return
            offset: Offset for pagination

        Returns:
            Dict with items, total counts, and pagination info
        """
        data = self._load_archive(show_id)

        # Collect items based on type filter
        items = []
        if item_type is None or item_type == "person":
            items.extend(data.get("archived_people", []))
        if item_type is None or item_type == "cluster":
            items.extend(data.get("archived_clusters", []))
        if item_type is None or item_type == "track":
            items.extend(data.get("archived_tracks", []))

        # Filter by episode if specified
        if episode_id:
            items = [i for i in items if i.get("episode_id") == episode_id]

        # Sort by archived_at descending (newest first)
        items.sort(key=lambda x: x.get("archived_at", ""), reverse=True)

        total = len(items)
        paginated = items[offset : offset + limit]

        return {
            "items": paginated,
            "total": total,
            "limit": limit,
            "offset": offset,
            "counts": {
                "people": len(data.get("archived_people", [])),
                "clusters": len(data.get("archived_clusters", [])),
                "tracks": len(data.get("archived_tracks", [])),
            },
        }

    def get_archived_centroids(
        self,
        show_id: str,
        *,
        item_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all archived items with their centroids for matching.

        Returns list of {archive_id, type, centroid, ...} for items with centroids.
        Used to check if new faces match archived (rejected) faces.
        """
        data = self._load_archive(show_id)

        results = []
        if item_type is None or item_type == "person":
            for item in data.get("archived_people", []):
                if item.get("centroid"):
                    results.append({
                        "archive_id": item["archive_id"],
                        "type": "person",
                        "name": item.get("name"),
                        "centroid": item["centroid"],
                        "rep_crop_url": item.get("rep_crop_url"),
                    })

        if item_type is None or item_type == "cluster":
            for item in data.get("archived_clusters", []):
                if item.get("centroid"):
                    results.append({
                        "archive_id": item["archive_id"],
                        "type": "cluster",
                        "original_id": item.get("original_id"),
                        "episode_id": item.get("episode_id"),
                        "centroid": item["centroid"],
                        "rep_crop_url": item.get("rep_crop_url"),
                    })

        if item_type is None or item_type == "track":
            for item in data.get("archived_tracks", []):
                if item.get("centroid"):
                    results.append({
                        "archive_id": item["archive_id"],
                        "type": "track",
                        "original_id": item.get("original_id"),
                        "episode_id": item.get("episode_id"),
                        "centroid": item["centroid"],
                        "rep_crop_url": item.get("rep_crop_url"),
                    })

        return results

    def find_matching_archived(
        self,
        show_id: str,
        centroid: List[float],
        *,
        threshold: float = 0.70,
        item_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find archived items that match a given centroid.

        Args:
            show_id: Show identifier
            centroid: Face centroid to match
            threshold: Minimum cosine similarity (0.70 = 70% similar)
            item_type: Filter by type or None for all

        Returns:
            List of matching archived items with similarity scores
        """
        archived = self.get_archived_centroids(show_id, item_type=item_type)
        if not archived:
            return []

        query = np.array(centroid)
        matches = []

        for item in archived:
            item_centroid = np.array(item["centroid"])
            similarity = cosine_similarity(query, item_centroid)
            if similarity >= threshold:
                matches.append({
                    **item,
                    "similarity": round(similarity, 4),
                })

        # Sort by similarity descending
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches

    def restore_person(
        self,
        show_id: str,
        archive_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Restore an archived person (remove from archive).

        Returns the original person data that can be re-created.
        """
        data = self._load_archive(show_id)
        people = data.get("archived_people", [])

        for i, item in enumerate(people):
            if item.get("archive_id") == archive_id:
                restored = people.pop(i)
                self._save_archive(show_id, data)
                LOGGER.info(f"Restored archived person {archive_id}")
                return restored.get("original_data", restored)

        return None

    def delete_archived(
        self,
        show_id: str,
        archive_id: str,
    ) -> bool:
        """Permanently delete an archived item.

        Returns True if deleted, False if not found.
        """
        data = self._load_archive(show_id)

        for collection in ["archived_people", "archived_clusters", "archived_tracks"]:
            items = data.get(collection, [])
            for i, item in enumerate(items):
                if item.get("archive_id") == archive_id:
                    items.pop(i)
                    self._save_archive(show_id, data)
                    LOGGER.info(f"Permanently deleted archived item {archive_id}")
                    return True

        return False

    def get_stats(self, show_id: str) -> Dict[str, Any]:
        """Get archive statistics for a show."""
        data = self._load_archive(show_id)
        return {
            "show_id": self.normalize_show_id(show_id),
            "people_count": len(data.get("archived_people", [])),
            "clusters_count": len(data.get("archived_clusters", [])),
            "tracks_count": len(data.get("archived_tracks", [])),
            "total_archived": data.get("stats", {}).get("total_archived", 0),
            "last_updated": data.get("stats", {}).get("last_updated"),
        }


# Singleton instance
archive_service = ArchiveService()
