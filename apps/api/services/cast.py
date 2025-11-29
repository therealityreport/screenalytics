"""Cast management service for show/season cast members."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DATA_ROOT = Path("data").expanduser()


class CastRole(str, Enum):
    MAIN = "main"
    FRIEND = "friend"
    GUEST = "guest"
    OTHER = "other"


class CastStatus(str, Enum):
    ACTIVE = "active"
    PAST = "past"
    INACTIVE = "inactive"


class FacebankEntryType(str, Enum):
    SEED = "seed"
    EXEMPLAR = "exemplar"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class CastService:
    """Manage show-level cast members and their metadata."""

    def __init__(self, data_root: Path | str | None = None):
        self.data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
        self.cast_dir = self.data_root / "cast"
        self.cast_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_show_id(show_id: str) -> str:
        """Normalize show_id to uppercase for consistent directory structure.

        Matches PeopleService.normalize_show_id for consistent storage paths.
        """
        if not isinstance(show_id, str):
            raise ValueError("show_id must be a string")
        cleaned = show_id.strip().upper()
        if not cleaned:
            raise ValueError("show_id cannot be empty")
        return cleaned

    @staticmethod
    def _validate_show_id(show_id: str) -> str:
        normalized = CastService._normalize_show_id(show_id)
        if not re.match(r"^[A-Za-z0-9_-]+$", normalized):
            raise ValueError("show_id must be alphanumeric (plus -/_) ")
        return normalized

    def _cast_file_path(self, show_id: str) -> Path:
        """Get path to cast.json for a show."""
        normalized = self._normalize_show_id(show_id)
        show_dir = self.cast_dir / normalized
        show_dir.mkdir(parents=True, exist_ok=True)
        return show_dir / "cast.json"

    def _load_cast(self, show_id: str) -> Dict[str, Any]:
        """Load cast.json or create empty structure."""
        normalized_show_id = self._normalize_show_id(show_id)
        path = self._cast_file_path(show_id)
        if not path.exists():
            return {"show_id": normalized_show_id, "cast": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"show_id": normalized_show_id, "cast": []}

    def _save_cast(self, show_id: str, data: Dict[str, Any]) -> None:
        """Save cast.json."""
        path = self._cast_file_path(show_id)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list_registered_shows(self) -> List[Dict[str, Any]]:
        """Return all shows that have cast metadata on disk.

        Show IDs are normalized to UPPERCASE and deduplicated case-insensitively.
        This ensures 'rhoslc' and 'RHOSLC' are treated as the same show.
        """
        if not self.cast_dir.exists():
            return []

        # Use a dict keyed by uppercase show_id for case-insensitive deduplication
        shows_by_id: Dict[str, Dict[str, Any]] = {}

        for entry in sorted(self.cast_dir.iterdir(), key=lambda p: p.name.lower()):
            if not entry.is_dir():
                continue
            cast_file = entry / "cast.json"
            if not cast_file.exists():
                continue
            try:
                payload = json.loads(cast_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {"cast": []}

            # Normalize show_id to uppercase
            raw_show_id = payload.get("show_id") or entry.name
            show_id = raw_show_id.upper() if isinstance(raw_show_id, str) else str(raw_show_id).upper()

            # Skip if we already have this show (case-insensitive dedup)
            if show_id in shows_by_id:
                continue

            shows_by_id[show_id] = {
                "show_id": show_id,
                "title": payload.get("show_title"),
                "full_name": payload.get("full_name"),
                "imdb_series_id": payload.get("imdb_series_id"),
                "cast_count": len(payload.get("cast", []) or []),
            }

        # Return sorted by show_id
        return sorted(shows_by_id.values(), key=lambda s: s["show_id"])

    def register_show(
        self,
        show_id: str,
        title: Optional[str] = None,
        full_name: Optional[str] = None,
        imdb_series_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create or update show metadata for Cast management."""
        normalized = self._validate_show_id(show_id)
        cast_file = self._cast_file_path(normalized)
        created = not cast_file.exists()

        if created:
            payload = {
                "show_id": normalized,
                "show_title": title or full_name or normalized,
                "full_name": full_name or title,
                "imdb_series_id": imdb_series_id,
                "cast": [],
            }
        else:
            payload = self._load_cast(normalized)
            payload.setdefault("cast", [])
            if title is not None:
                payload["show_title"] = title or None
            elif "show_title" not in payload:
                payload["show_title"] = payload.get("full_name") or normalized
            if full_name is not None:
                payload["full_name"] = full_name or None
            if imdb_series_id is not None:
                payload["imdb_series_id"] = imdb_series_id or None

        if "full_name" not in payload or not payload["full_name"]:
            payload["full_name"] = payload.get("show_title")
        if "show_title" not in payload or not payload["show_title"]:
            payload["show_title"] = payload.get("full_name") or normalized

        self._save_cast(normalized, payload)
        return {
            "show_id": normalized,
            "title": payload.get("show_title"),
            "full_name": payload.get("full_name"),
            "imdb_series_id": payload.get("imdb_series_id"),
            "cast_count": len(payload.get("cast", []) or []),
            "created": created,
        }

    def list_cast(self, show_id: str, season: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all cast members for a show, optionally filtered by season."""
        data = self._load_cast(show_id)
        cast_members = data.get("cast", [])

        if season:
            # Filter by season (case-insensitive). Empty seasons list = all seasons.
            season_lower = season.lower()
            cast_members = [
                member
                for member in cast_members
                if not member.get("seasons")  # blank seasons means all seasons
                or any(s.lower() == season_lower for s in member.get("seasons", []))
            ]

        return cast_members

    def get_cast_member(self, show_id: str, cast_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific cast member."""
        cast_members = self.list_cast(show_id)
        for member in cast_members:
            if member["cast_id"] == cast_id:
                return member
        return None

    def create_cast_member(
        self,
        show_id: str,
        name: str,
        role: str = "other",
        status: str = "active",
        aliases: Optional[List[str]] = None,
        seasons: Optional[List[str]] = None,
        social: Optional[Dict[str, str]] = None,
        full_name: Optional[str] = None,
        imdb_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new cast member."""
        data = self._load_cast(show_id)
        cast_members = data.get("cast", [])

        # Generate cast_id
        cast_id = str(uuid.uuid4())

        member = {
            "cast_id": cast_id,
            "show_id": show_id,
            "name": name,
            "role": role,
            "status": status,
            "aliases": aliases or [],
            "seasons": seasons or [],
            "social": social or {},
            "full_name": full_name,
            "imdb_id": imdb_id,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }

        cast_members.append(member)
        data["cast"] = cast_members
        self._save_cast(show_id, data)
        return member

    def update_cast_member(
        self,
        show_id: str,
        cast_id: str,
        name: Optional[str] = None,
        role: Optional[str] = None,
        status: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        seasons: Optional[List[str]] = None,
        social: Optional[Dict[str, str]] = None,
        full_name: Optional[str] = None,
        imdb_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update an existing cast member."""
        data = self._load_cast(show_id)
        cast_members = data.get("cast", [])

        for member in cast_members:
            if member["cast_id"] == cast_id:
                if name is not None:
                    member["name"] = name
                if role is not None:
                    member["role"] = role
                if status is not None:
                    member["status"] = status
                if aliases is not None:
                    member["aliases"] = aliases
                if seasons is not None:
                    member["seasons"] = seasons
                if social is not None:
                    member["social"] = social
                if full_name is not None:
                    member["full_name"] = full_name
                if imdb_id is not None:
                    member["imdb_id"] = imdb_id
                member["updated_at"] = _now_iso()

                data["cast"] = cast_members
                self._save_cast(show_id, data)
                return member

        return None

    def delete_cast_member(self, show_id: str, cast_id: str) -> bool:
        """Delete a cast member."""
        data = self._load_cast(show_id)
        cast_members = data.get("cast", [])

        original_count = len(cast_members)
        cast_members = [m for m in cast_members if m["cast_id"] != cast_id]

        if len(cast_members) < original_count:
            data["cast"] = cast_members
            self._save_cast(show_id, data)
            return True

        return False

    def bulk_import_cast(
        self,
        show_id: str,
        members: List[Dict[str, Any]],
        force_new: bool = False,
    ) -> Dict[str, Any]:
        """Bulk import cast members from a list.

        Args:
            show_id: Show identifier
            members: List of dicts with cast member data
            force_new: If True, always create new members; if False, merge by name

        Returns:
            Audit summary with created/updated/skipped counts
        """
        data = self._load_cast(show_id)
        cast_members = data.get("cast", [])

        # Build name lookup (case-insensitive), skip members with None/empty names
        name_to_member = {
            m["name"].lower(): m
            for m in cast_members
            if m.get("name") and isinstance(m["name"], str)
        }

        created = []
        updated = []
        skipped = []

        for entry in members:
            name = entry.get("name", "").strip()
            if not name:
                skipped.append({"name": entry.get("name"), "reason": "missing_name"})
                continue

            role = entry.get("role", "other")
            status = entry.get("status", "active")
            aliases = entry.get("aliases", [])
            seasons = entry.get("seasons", [])
            social = entry.get("social", {})
            full_name = entry.get("full_name")
            imdb_id = entry.get("imdb_id")

            # Normalize aliases and seasons
            if isinstance(aliases, str):
                aliases = [a.strip() for a in aliases.split(",") if a.strip()]
            if isinstance(seasons, str):
                seasons = [s.strip() for s in seasons.split(",") if s.strip()]

            name_lower = name.lower()

            if not force_new and name_lower in name_to_member:
                # Update existing member
                existing = name_to_member[name_lower]
                existing["role"] = role
                existing["status"] = status
                existing["aliases"] = aliases
                existing["seasons"] = seasons
                existing["social"] = social
                if full_name is not None:
                    existing["full_name"] = full_name
                if imdb_id is not None:
                    existing["imdb_id"] = imdb_id
                existing["updated_at"] = _now_iso()
                updated.append({"name": name, "cast_id": existing["cast_id"]})
            else:
                # Create new member
                cast_id = str(uuid.uuid4())
                new_member = {
                    "cast_id": cast_id,
                    "show_id": show_id,
                    "name": name,
                    "full_name": full_name,
                    "imdb_id": imdb_id,
                    "role": role,
                    "status": status,
                    "aliases": aliases,
                    "seasons": seasons,
                    "social": social,
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                }
                cast_members.append(new_member)
                name_to_member[name_lower] = new_member
                created.append({"name": name, "cast_id": cast_id})

        data["cast"] = cast_members
        self._save_cast(show_id, data)

        return {
            "show_id": show_id,
            "total": len(members),
            "created": created,
            "created_count": len(created),
            "updated": updated,
            "updated_count": len(updated),
            "skipped": skipped,
            "skipped_count": len(skipped),
        }


__all__ = ["CastService", "CastRole", "CastStatus", "FacebankEntryType"]
