"""People management service for show-level person entities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

DEFAULT_DATA_ROOT = Path("data").expanduser()


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    norm = np.linalg.norm(vector) + 1e-12
    return vector / norm


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two L2-normalized vectors."""
    return 1.0 - float(np.dot(a, b))


class PeopleService:
    """Manage show-level person entities and cluster grouping."""

    def __init__(self, data_root: Path | str | None = None):
        self.data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
        self.shows_dir = self.data_root / "shows"
        self.shows_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_show_id(show_id: str) -> str:
        """Normalize show_id to uppercase for consistent directory structure.

        All show-level data (people.json, cast.json, facebank/) uses uppercase
        show slugs (e.g., RHOBH, DEMO) regardless of how episode IDs are specified.
        """
        return show_id.upper()

    def _people_path(self, show_id: str) -> Path:
        """Get path to people.json for a show."""
        show_id = self.normalize_show_id(show_id)  # Normalize to uppercase
        show_dir = self.shows_dir / show_id
        show_dir.mkdir(parents=True, exist_ok=True)
        return show_dir / "people.json"

    def _load_people(self, show_id: str) -> Dict[str, Any]:
        """Load people.json or create empty structure."""
        path = self._people_path(show_id)
        if not path.exists():
            return {"show_id": show_id, "people": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"show_id": show_id, "people": []}
        people = data.get("people")
        if isinstance(people, list):
            for person in people:
                if isinstance(person, dict) and "rep_crop_s3_key" not in person:
                    person["rep_crop_s3_key"] = None
        return data

    def _save_people(self, show_id: str, data: Dict[str, Any]) -> None:
        """Save people.json."""
        path = self._people_path(show_id)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list_people(self, show_id: str) -> List[Dict[str, Any]]:
        """Get all people for a show."""
        data = self._load_people(show_id)
        return data.get("people", [])

    def get_person(self, show_id: str, person_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific person."""
        people = self.list_people(show_id)
        for person in people:
            if person["person_id"] == person_id:
                return person
        return None

    def create_person(
        self,
        show_id: str,
        name: Optional[str] = None,
        prototype: Optional[List[float]] = None,
        cluster_ids: Optional[List[str]] = None,
        rep_crop: Optional[str] = None,
        rep_crop_s3_key: Optional[str] = None,
        cast_id: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a new person."""
        data = self._load_people(show_id)
        people = data.get("people", [])

        # Generate person_id
        next_id = 1
        existing_ids = {p["person_id"] for p in people}
        while f"p_{next_id:04d}" in existing_ids:
            next_id += 1
        person_id = f"p_{next_id:04d}"

        person = {
            "person_id": person_id,
            "show_id": show_id,
            "name": name,
            "aliases": aliases or [],
            "prototype": prototype or [],
            "cluster_ids": cluster_ids or [],
            "rep_crop": rep_crop,
            "rep_crop_s3_key": rep_crop_s3_key,
            "created_at": _now_iso(),
            "cast_id": cast_id,
        }

        people.append(person)
        data["people"] = people
        self._save_people(show_id, data)
        return person

    def update_person(
        self,
        show_id: str,
        person_id: str,
        name: Optional[str] = None,
        prototype: Optional[List[float]] = None,
        cluster_ids: Optional[List[str]] = None,
        rep_crop: Optional[str] = None,
        rep_crop_s3_key: Optional[str] = None,
        cast_id: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update an existing person."""
        data = self._load_people(show_id)
        people = data.get("people", [])

        for person in people:
            if person["person_id"] == person_id:
                if name is not None:
                    person["name"] = name
                if prototype is not None:
                    person["prototype"] = prototype
                if cluster_ids is not None:
                    person["cluster_ids"] = cluster_ids
                if rep_crop is not None:
                    person["rep_crop"] = rep_crop
                if rep_crop_s3_key is not None:
                    person["rep_crop_s3_key"] = rep_crop_s3_key
                if cast_id is not None:
                    person["cast_id"] = cast_id
                if aliases is not None:
                    person["aliases"] = aliases

                data["people"] = people
                self._save_people(show_id, data)
                return person

        return None

    def add_cluster_to_person(
        self,
        show_id: str,
        person_id: str,
        cluster_id: str,
        update_prototype: bool = True,
        cluster_centroid: Optional[np.ndarray] = None,
        momentum: float = 0.9,
    ) -> Optional[Dict[str, Any]]:
        """Add a cluster to a person and optionally update prototype."""
        person = self.get_person(show_id, person_id)
        if not person:
            return None

        cluster_ids = person.get("cluster_ids", [])
        if cluster_id not in cluster_ids:
            cluster_ids.append(cluster_id)

        # Update prototype if requested
        prototype = person.get("prototype", [])
        if update_prototype and cluster_centroid is not None and len(prototype) > 0:
            old_proto = np.array(prototype, dtype=np.float32)
            new_centroid = np.array(cluster_centroid, dtype=np.float32)

            # Momentum update
            updated = momentum * old_proto + (1.0 - momentum) * new_centroid
            updated = l2_normalize(updated)
            prototype = updated.tolist()
        elif update_prototype and cluster_centroid is not None:
            # First cluster for this person
            prototype = l2_normalize(np.array(cluster_centroid, dtype=np.float32)).tolist()

        return self.update_person(show_id, person_id, cluster_ids=cluster_ids, prototype=prototype)

    def find_matching_person(
        self,
        show_id: str,
        cluster_centroid: np.ndarray,
        max_distance: float = 0.35,
    ) -> Optional[tuple[str, float]]:
        """Find the best matching person for a cluster centroid.

        Returns (person_id, distance) if match found, else None.
        """
        people = self.list_people(show_id)

        best_person_id = None
        best_distance = float('inf')

        for person in people:
            prototype = person.get("prototype", [])
            if not prototype:
                continue

            proto_vec = np.array(prototype, dtype=np.float32)
            distance = cosine_distance(cluster_centroid, proto_vec)

            if distance < best_distance:
                best_distance = distance
                best_person_id = person["person_id"]

        if best_person_id and best_distance <= max_distance:
            return (best_person_id, best_distance)

        return None

    def find_person_by_cast_id(self, show_id: str, cast_id: str) -> Optional[Dict[str, Any]]:
        """Find a person record linked to a given cast member."""
        people = self.list_people(show_id)
        for person in people:
            if person.get("cast_id") == cast_id:
                return person
        return None

    def delete_person(self, show_id: str, person_id: str) -> bool:
        """Delete a person."""
        data = self._load_people(show_id)
        people = data.get("people", [])

        original_count = len(people)
        people = [p for p in people if p["person_id"] != person_id]

        if len(people) < original_count:
            data["people"] = people
            self._save_people(show_id, data)
            return True

        return False

    def normalize_name(self, name: str) -> str:
        """Normalize a name for matching: lowercase, strip whitespace, remove extra spaces."""
        if not name:
            return ""
        return " ".join(name.strip().lower().split())

    def find_person_by_name_or_alias(
        self,
        show_id: str,
        name: str,
        *,
        cast_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find a person by name or alias (case-insensitive, normalized).

        If cast_id is provided, prefer people with matching cast_id.
        """
        normalized = self.normalize_name(name)
        if not normalized:
            return None

        people = self.list_people(show_id)

        # First pass: try to match by cast_id if provided
        if cast_id:
            for person in people:
                if person.get("cast_id") == cast_id:
                    return person

        # Second pass: match by name or aliases
        matches = []
        for person in people:
            # Ensure backwards compatibility: add aliases field if missing
            if "aliases" not in person:
                person["aliases"] = []

            # Check primary name
            person_name = person.get("name")
            if person_name and self.normalize_name(person_name) == normalized:
                matches.append(person)
                continue

            # Check aliases
            aliases = person.get("aliases") or []
            for alias in aliases:
                if self.normalize_name(alias) == normalized:
                    matches.append(person)
                    break

        if not matches:
            return None

        # If multiple matches, prefer the one with a cast_id
        for match in matches:
            if match.get("cast_id"):
                return match

        return matches[0]

    def add_alias_to_person(
        self,
        show_id: str,
        person_id: str,
        alias: str,
    ) -> Optional[Dict[str, Any]]:
        """Add an alias to a person (if not already present)."""
        person = self.get_person(show_id, person_id)
        if not person:
            return None

        normalized_alias = alias.strip()
        if not normalized_alias:
            return person

        # Ensure backwards compatibility
        aliases = person.get("aliases") or []

        # Check if alias already exists (normalized comparison)
        normalized_new = self.normalize_name(normalized_alias)
        for existing in aliases:
            if self.normalize_name(existing) == normalized_new:
                return person  # Already exists

        # Add the alias
        aliases.append(normalized_alias)
        return self.update_person(show_id, person_id, aliases=aliases)

    def merge_people(
        self,
        show_id: str,
        source_person_id: str,
        target_person_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Merge source person into target person.

        Moves all clusters, merges aliases, and deletes the source person.
        """
        source = self.get_person(show_id, source_person_id)
        target = self.get_person(show_id, target_person_id)

        if not source or not target:
            return None

        # Merge cluster_ids
        target_clusters = set(target.get("cluster_ids") or [])
        source_clusters = source.get("cluster_ids") or []
        for cluster_id in source_clusters:
            target_clusters.add(cluster_id)

        # Merge aliases
        target_aliases = list(target.get("aliases") or [])
        source_aliases = source.get("aliases") or []

        # Add source's primary name as alias if different from target's name
        source_name = source.get("name")
        target_name = target.get("name")
        if source_name and self.normalize_name(source_name) != self.normalize_name(target_name or ""):
            source_aliases.append(source_name)

        # Merge unique aliases (normalized comparison)
        existing_normalized = {self.normalize_name(a) for a in target_aliases}
        for alias in source_aliases:
            normalized = self.normalize_name(alias)
            if normalized and normalized not in existing_normalized:
                target_aliases.append(alias)
                existing_normalized.add(normalized)

        # Update target with merged data
        updated = self.update_person(
            show_id,
            target_person_id,
            cluster_ids=sorted(target_clusters),
            aliases=target_aliases,
        )

        # Delete source person
        self.delete_person(show_id, source_person_id)

        return updated


__all__ = ["PeopleService", "l2_normalize", "cosine_distance"]
