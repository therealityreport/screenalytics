"""Episode metadata helpers for FastAPI endpoints."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional

DATA_ROOT_ENV = "SCREENALYTICS_DATA_ROOT"


def _data_root() -> Path:
    raw = os.environ.get(DATA_ROOT_ENV, "data")
    return Path(raw).expanduser()


def _meta_path() -> Path:
    return _data_root() / "meta" / "episodes.json"


_slug_regex = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    normalized = _slug_regex.sub("-", value.strip().lower())
    normalized = normalized.strip("-")
    return normalized or "episode"


@dataclass
class EpisodeRecord:
    ep_id: str
    show_ref: str
    season_number: int
    episode_number: int
    title: Optional[str]
    air_date: Optional[str]
    created_at: str
    updated_at: str


class EpisodeStore:
    """JSON-backed metadata store for ad-hoc episode records."""

    def __init__(self) -> None:
        self._path = _meta_path()

    # File helpers -----------------------------------------------------
    def _read(self) -> Dict[str, Dict[str, object]]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
            return {}
        except json.JSONDecodeError:
            return {}

    def _write(self, data: Dict[str, Dict[str, object]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self._path)

    # Public API -------------------------------------------------------
    @staticmethod
    def make_ep_id(show_ref: str, season_number: int, episode_number: int) -> str:
        slug = _slugify(show_ref)
        return f"{slug}-s{season_number:02d}e{episode_number:02d}"

    def upsert(
        self,
        *,
        show_ref: str,
        season_number: int,
        episode_number: int,
        title: Optional[str] = None,
        air_date: Optional[date] = None,
    ) -> EpisodeRecord:
        ep_id = self.make_ep_id(show_ref, season_number, episode_number)
        content = self._read()
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        record = content.get(ep_id, {}).copy()

        created_at = record.get("created_at") if record else now
        stored = {
            "ep_id": ep_id,
            "show_ref": show_ref,
            "season_number": season_number,
            "episode_number": episode_number,
            "title": title if title is not None else record.get("title"),
            "air_date": (air_date.isoformat() if air_date else record.get("air_date")),
            "created_at": created_at,
            "updated_at": now,
        }
        content[ep_id] = stored
        self._write(content)
        return EpisodeRecord(**stored)  # type: ignore[arg-type]

    @staticmethod
    def normalize_ep_id(ep_id: str) -> str:
        """Normalize ep_id to lowercase for case-insensitive handling.

        This ensures that 'RHOSLC-s06e02' and 'rhoslc-s06e02' are treated as the same episode.
        """
        return ep_id.strip().lower()

    def upsert_ep_id(
        self,
        *,
        ep_id: str,
        show_slug: str,
        season: int,
        episode: int,
        title: Optional[str] = None,
        air_date: Optional[date | str] = None,
    ) -> tuple[EpisodeRecord, bool]:
        # Normalize ep_id to lowercase for case-insensitive handling
        ep_id = self.normalize_ep_id(ep_id)
        if not ep_id:
            raise ValueError("ep_id is required")
        expected = self.make_ep_id(show_slug, season, episode)
        if ep_id != expected:
            raise ValueError(f"ep_id '{ep_id}' does not match show/season/episode ({expected})")

        content = self._read()
        existing = content.get(ep_id)
        if existing:
            return EpisodeRecord(**existing), False  # type: ignore[arg-type]

        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        stored = {
            "ep_id": ep_id,
            "show_ref": show_slug.lower(),  # Also normalize show_slug
            "season_number": season,
            "episode_number": episode,
            "title": title,
            "air_date": (air_date.isoformat() if isinstance(air_date, date) else air_date),
            "created_at": now,
            "updated_at": now,
        }
        content[ep_id] = stored
        self._write(content)
        return EpisodeRecord(**stored), True  # type: ignore[arg-type]

    def get(self, ep_id: str) -> Optional[EpisodeRecord]:
        ep_id = self.normalize_ep_id(ep_id)
        content = self._read()
        data = content.get(ep_id)
        if not data:
            return None
        return EpisodeRecord(**data)  # type: ignore[arg-type]

    def exists(self, ep_id: str) -> bool:
        ep_id = self.normalize_ep_id(ep_id)
        return ep_id in self._read()

    def list(self) -> list[EpisodeRecord]:
        content = self._read()
        records = [EpisodeRecord(**data) for data in content.values()]  # type: ignore[arg-type]
        return sorted(records, key=lambda record: record.updated_at, reverse=True)

    def update_metadata(
        self,
        *,
        ep_id: str,
        title: Optional[str] = None,
        air_date: Optional[date | str] = None,
    ) -> EpisodeRecord:
        """Update metadata fields for an existing episode.

        Only updates fields that are explicitly provided (not None).
        Always updates updated_at timestamp.

        Args:
            ep_id: Episode identifier
            title: New title (or None to keep existing)
            air_date: New air date (or None to keep existing)

        Returns:
            Updated EpisodeRecord

        Raises:
            ValueError: If episode does not exist
        """
        ep_id = self.normalize_ep_id(ep_id)
        content = self._read()

        if ep_id not in content:
            raise ValueError(f"Episode {ep_id} not found")

        record = content[ep_id].copy()
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        # Only update fields that were explicitly provided
        if title is not None:
            record["title"] = title if title else None
        if air_date is not None:
            record["air_date"] = (
                air_date.isoformat() if isinstance(air_date, date) else air_date
            ) if air_date else None

        record["updated_at"] = now
        content[ep_id] = record
        self._write(content)

        return EpisodeRecord(**record)  # type: ignore[arg-type]

    def delete(self, ep_id: str) -> bool:
        ep_id = self.normalize_ep_id(ep_id)
        content = self._read()
        if ep_id not in content:
            return False
        content.pop(ep_id, None)
        self._write(content)
        return True


__all__ = ["EpisodeStore", "EpisodeRecord"]
