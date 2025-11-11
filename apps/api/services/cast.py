"""Cast management service for show/season cast members."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
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
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class CastService:
    """Manage show-level cast members and their metadata."""

    def __init__(self, data_root: Path | str | None = None):
        self.data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
        self.cast_dir = self.data_root / "cast"
        self.cast_dir.mkdir(parents=True, exist_ok=True)

    def _cast_file_path(self, show_id: str) -> Path:
        """Get path to cast.json for a show."""
        show_dir = self.cast_dir / show_id
        show_dir.mkdir(parents=True, exist_ok=True)
        return show_dir / "cast.json"

    def _load_cast(self, show_id: str) -> Dict[str, Any]:
        """Load cast.json or create empty structure."""
        path = self._cast_file_path(show_id)
        if not path.exists():
            return {"show_id": show_id, "cast": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"show_id": show_id, "cast": []}

    def _save_cast(self, show_id: str, data: Dict[str, Any]) -> None:
        """Save cast.json."""
        path = self._cast_file_path(show_id)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list_cast(self, show_id: str, season: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all cast members for a show, optionally filtered by season."""
        data = self._load_cast(show_id)
        cast_members = data.get("cast", [])

        if season:
            # Filter by season
            cast_members = [
                member for member in cast_members
                if season in member.get("seasons", [])
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

        # Build name lookup (case-insensitive)
        name_to_member = {m["name"].lower(): m for m in cast_members}

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
                existing["updated_at"] = _now_iso()
                updated.append({"name": name, "cast_id": existing["cast_id"]})
            else:
                # Create new member
                cast_id = str(uuid.uuid4())
                new_member = {
                    "cast_id": cast_id,
                    "show_id": show_id,
                    "name": name,
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
