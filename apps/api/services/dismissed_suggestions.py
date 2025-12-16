"""Dismissed Suggestions Service.

Persists dismissed suggestions to disk so they survive session restarts.

Run scoping:
- Legacy (no run_id): data/manifests/{ep_id}/dismissed_suggestions.json
- Run-scoped:         data/manifests/{ep_id}/runs/{run_id}/dismissed_suggestions.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from py_screenalytics.artifacts import get_path
from py_screenalytics import run_layout

LOGGER = logging.getLogger(__name__)


class DismissedSuggestionsService:
    """Service for managing dismissed Smart Suggestions."""

    def __init__(self):
        """Initialize the dismissed suggestions service."""
        # Cache keyed by "ep_id::run_id" where run_id may be "legacy".
        self._cache: Dict[str, Set[str]] = {}

    @staticmethod
    def _cache_key(ep_id: str, run_id: str | None) -> str:
        return f"{ep_id}::{run_id or 'legacy'}"

    def _get_dismissed_file_path(self, ep_id: str, *, run_id: str | None = None) -> Path:
        """Get the path to the dismissed suggestions file for an episode/run."""
        if run_id:
            return run_layout.run_root(ep_id, run_layout.normalize_run_id(run_id)) / "dismissed_suggestions.json"
        manifests_dir = get_path(ep_id, "detections").parent
        return manifests_dir / "dismissed_suggestions.json"

    def _load_dismissed(self, ep_id: str, *, run_id: str | None = None) -> Set[str]:
        """Load dismissed suggestions from disk."""
        key = self._cache_key(ep_id, run_id)
        if key in self._cache:
            return self._cache[key]

        dismissed_file = self._get_dismissed_file_path(ep_id, run_id=run_id)
        if not dismissed_file.exists():
            self._cache[key] = set()
            return self._cache[key]

        try:
            with open(dismissed_file, "r") as f:
                data = json.load(f)
                dismissed = set(data.get("dismissed", []))
                self._cache[key] = dismissed
                return dismissed
        except Exception as e:
            LOGGER.warning(f"[{ep_id}] Failed to load dismissed suggestions: {e}")
            self._cache[key] = set()
            return self._cache[key]

    def _save_dismissed(self, ep_id: str, dismissed: Set[str], *, run_id: str | None = None) -> bool:
        """Save dismissed suggestions to disk."""
        dismissed_file = self._get_dismissed_file_path(ep_id, run_id=run_id)
        try:
            data = {
                "dismissed": list(dismissed),
                "updated_at": datetime.utcnow().isoformat(),
            }
            dismissed_file.parent.mkdir(parents=True, exist_ok=True)
            with open(dismissed_file, "w") as f:
                json.dump(data, f, indent=2)
            self._cache[self._cache_key(ep_id, run_id)] = dismissed
            return True
        except Exception as e:
            LOGGER.error(f"[{ep_id}] Failed to save dismissed suggestions: {e}")
            return False

    def get_dismissed(self, ep_id: str, *, run_id: str | None = None) -> List[str]:
        """Get list of dismissed suggestion IDs for an episode/run.

        Args:
            ep_id: Episode ID
            run_id: Optional run_id scope

        Returns:
            List of dismissed cluster/person IDs
        """
        return list(self._load_dismissed(ep_id, run_id=run_id))

    def dismiss(self, ep_id: str, suggestion_id: str, *, run_id: str | None = None) -> bool:
        """Dismiss a suggestion.

        Args:
            ep_id: Episode ID
            suggestion_id: ID of the suggestion to dismiss (cluster_id or "person:{person_id}")
            run_id: Optional run_id scope

        Returns:
            True if successful
        """
        dismissed = self._load_dismissed(ep_id, run_id=run_id)
        dismissed.add(suggestion_id)
        return self._save_dismissed(ep_id, dismissed, run_id=run_id)

    def dismiss_many(self, ep_id: str, suggestion_ids: List[str], *, run_id: str | None = None) -> bool:
        """Dismiss multiple suggestions at once.

        Args:
            ep_id: Episode ID
            suggestion_ids: List of suggestion IDs to dismiss
            run_id: Optional run_id scope

        Returns:
            True if successful
        """
        dismissed = self._load_dismissed(ep_id, run_id=run_id)
        dismissed.update(suggestion_ids)
        return self._save_dismissed(ep_id, dismissed, run_id=run_id)

    def restore(self, ep_id: str, suggestion_id: str, *, run_id: str | None = None) -> bool:
        """Restore a previously dismissed suggestion.

        Args:
            ep_id: Episode ID
            suggestion_id: ID of the suggestion to restore
            run_id: Optional run_id scope

        Returns:
            True if successful
        """
        dismissed = self._load_dismissed(ep_id, run_id=run_id)
        dismissed.discard(suggestion_id)
        return self._save_dismissed(ep_id, dismissed, run_id=run_id)

    def clear_all(self, ep_id: str, *, run_id: str | None = None) -> bool:
        """Clear all dismissed suggestions for an episode/run.

        Args:
            ep_id: Episode ID
            run_id: Optional run_id scope

        Returns:
            True if successful
        """
        return self._save_dismissed(ep_id, set(), run_id=run_id)

    def reset_state(
        self, ep_id: str, *, run_id: str | None = None, archive_existing: bool = True
    ) -> Dict[str, Any]:
        """Reset dismissed suggestions state for an episode/run.

        The default behavior preserves user intent by archiving the existing file
        before clearing it.
        """
        dismissed_file = self._get_dismissed_file_path(ep_id, run_id=run_id)
        archived_name: str | None = None

        if archive_existing and dismissed_file.exists():
            archive_suffix = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            archived_path = dismissed_file.with_name(f"{dismissed_file.name}.{archive_suffix}.bak")
            try:
                dismissed_file.rename(archived_path)
                archived_name = archived_path.name
                LOGGER.info("[%s] Archived dismissed suggestions: %s", ep_id, archived_name)
            except OSError as exc:
                LOGGER.error("[%s] Failed to archive dismissed suggestions: %s", ep_id, exc)
                return {"ok": False, "error": f"Failed to archive existing dismissed suggestions: {exc}"}

        self.invalidate_cache(ep_id, run_id=run_id)
        dismissed_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._save_dismissed(ep_id, set(), run_id=run_id):
            return {"ok": False, "error": "Failed to write dismissed suggestions"}

        return {"ok": True, "archived": archived_name}

    def is_dismissed(self, ep_id: str, suggestion_id: str, *, run_id: str | None = None) -> bool:
        """Check if a suggestion is dismissed.

        Args:
            ep_id: Episode ID
            suggestion_id: ID to check
            run_id: Optional run_id scope

        Returns:
            True if dismissed
        """
        return suggestion_id in self._load_dismissed(ep_id, run_id=run_id)

    def invalidate_cache(self, ep_id: Optional[str] = None, *, run_id: str | None = None) -> None:
        """Invalidate the in-memory cache.

        Args:
            ep_id: Episode ID to invalidate, or None to clear all
            run_id: Optional run_id scope
        """
        if not ep_id:
            self._cache.clear()
            return

        if run_id is None:
            prefix = f"{ep_id}::"
            for key in list(self._cache.keys()):
                if key.startswith(prefix):
                    self._cache.pop(key, None)
            return

        self._cache.pop(self._cache_key(ep_id, run_id), None)


# Singleton instance
dismissed_suggestions_service = DismissedSuggestionsService()
