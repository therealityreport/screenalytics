"""Dismissed Suggestions Service.

Persists dismissed suggestions to disk so they survive session restarts.
Each episode has its own set of dismissed suggestions stored in a JSON file.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from apps.api.services import storage

LOGGER = logging.getLogger(__name__)


class DismissedSuggestionsService:
    """Service for managing dismissed Smart Suggestions."""

    def __init__(self):
        """Initialize the dismissed suggestions service."""
        self._cache: Dict[str, Set[str]] = {}

    def _get_dismissed_file_path(self, ep_id: str) -> Path:
        """Get the path to the dismissed suggestions file for an episode."""
        # Store alongside identities.json in the episode's working directory
        ep_path = storage.working_dir_for_ep(ep_id, create=True)
        return ep_path / "dismissed_suggestions.json"

    def _load_dismissed(self, ep_id: str) -> Set[str]:
        """Load dismissed suggestions from disk."""
        if ep_id in self._cache:
            return self._cache[ep_id]

        dismissed_file = self._get_dismissed_file_path(ep_id)
        if not dismissed_file.exists():
            self._cache[ep_id] = set()
            return self._cache[ep_id]

        try:
            with open(dismissed_file, "r") as f:
                data = json.load(f)
                dismissed = set(data.get("dismissed", []))
                self._cache[ep_id] = dismissed
                return dismissed
        except Exception as e:
            LOGGER.warning(f"[{ep_id}] Failed to load dismissed suggestions: {e}")
            self._cache[ep_id] = set()
            return self._cache[ep_id]

    def _save_dismissed(self, ep_id: str, dismissed: Set[str]) -> bool:
        """Save dismissed suggestions to disk."""
        dismissed_file = self._get_dismissed_file_path(ep_id)
        try:
            data = {
                "dismissed": list(dismissed),
                "updated_at": datetime.utcnow().isoformat(),
            }
            with open(dismissed_file, "w") as f:
                json.dump(data, f, indent=2)
            self._cache[ep_id] = dismissed
            return True
        except Exception as e:
            LOGGER.error(f"[{ep_id}] Failed to save dismissed suggestions: {e}")
            return False

    def get_dismissed(self, ep_id: str) -> List[str]:
        """Get list of dismissed suggestion IDs for an episode.

        Args:
            ep_id: Episode ID

        Returns:
            List of dismissed cluster/person IDs
        """
        return list(self._load_dismissed(ep_id))

    def dismiss(self, ep_id: str, suggestion_id: str) -> bool:
        """Dismiss a suggestion.

        Args:
            ep_id: Episode ID
            suggestion_id: ID of the suggestion to dismiss (cluster_id or "person:{person_id}")

        Returns:
            True if successful
        """
        dismissed = self._load_dismissed(ep_id)
        dismissed.add(suggestion_id)
        return self._save_dismissed(ep_id, dismissed)

    def dismiss_many(self, ep_id: str, suggestion_ids: List[str]) -> bool:
        """Dismiss multiple suggestions at once.

        Args:
            ep_id: Episode ID
            suggestion_ids: List of suggestion IDs to dismiss

        Returns:
            True if successful
        """
        dismissed = self._load_dismissed(ep_id)
        dismissed.update(suggestion_ids)
        return self._save_dismissed(ep_id, dismissed)

    def restore(self, ep_id: str, suggestion_id: str) -> bool:
        """Restore a previously dismissed suggestion.

        Args:
            ep_id: Episode ID
            suggestion_id: ID of the suggestion to restore

        Returns:
            True if successful
        """
        dismissed = self._load_dismissed(ep_id)
        dismissed.discard(suggestion_id)
        return self._save_dismissed(ep_id, dismissed)

    def clear_all(self, ep_id: str) -> bool:
        """Clear all dismissed suggestions for an episode.

        Args:
            ep_id: Episode ID

        Returns:
            True if successful
        """
        return self._save_dismissed(ep_id, set())

    def is_dismissed(self, ep_id: str, suggestion_id: str) -> bool:
        """Check if a suggestion is dismissed.

        Args:
            ep_id: Episode ID
            suggestion_id: ID to check

        Returns:
            True if dismissed
        """
        return suggestion_id in self._load_dismissed(ep_id)

    def invalidate_cache(self, ep_id: Optional[str] = None) -> None:
        """Invalidate the in-memory cache.

        Args:
            ep_id: Episode ID to invalidate, or None to clear all
        """
        if ep_id:
            self._cache.pop(ep_id, None)
        else:
            self._cache.clear()


# Singleton instance
dismissed_suggestions_service = DismissedSuggestionsService()
