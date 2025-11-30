"""Voice bank management for speaker identification.

Handles:
- Storing and loading voice bank entries
- Matching voice clusters to known voices
- Creating unlabeled voice entries
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .models import (
    VoiceBankConfig,
    VoiceBankEntry,
    VoiceBankMatchResult,
    VoiceCluster,
)

LOGGER = logging.getLogger(__name__)


class VoiceBank:
    """Voice bank for storing and matching speaker embeddings."""

    def __init__(self, config: Optional[VoiceBankConfig] = None):
        """Initialize voice bank.

        Args:
            config: Voice bank configuration
        """
        self.config = config or VoiceBankConfig()
        self.data_dir = Path(self.config.data_dir)
        self._entries_cache: Dict[str, List[VoiceBankEntry]] = {}

    def get_entries(self, show_id: str) -> List[VoiceBankEntry]:
        """Get all voice bank entries for a show.

        Args:
            show_id: Show identifier

        Returns:
            List of VoiceBankEntry objects
        """
        if show_id in self._entries_cache:
            return self._entries_cache[show_id]

        entries = self._load_bank(show_id)
        self._entries_cache[show_id] = entries
        return entries

    def add_entry(self, entry: VoiceBankEntry, show_id: str) -> VoiceBankEntry:
        """Add an entry to the voice bank.

        Args:
            entry: Voice bank entry to add
            show_id: Show identifier

        Returns:
            Added entry
        """
        entries = self.get_entries(show_id)

        # Check for duplicate ID
        existing = next((e for e in entries if e.voice_bank_id == entry.voice_bank_id), None)
        if existing:
            LOGGER.warning(f"Voice bank entry {entry.voice_bank_id} already exists, updating")
            entries.remove(existing)

        entries.append(entry)
        self._save_bank(show_id, entries)

        return entry

    def match_cluster(
        self,
        show_id: str,
        cluster: VoiceCluster,
        threshold: float = 0.78,
    ) -> VoiceBankMatchResult:
        """Match a voice cluster to the voice bank.

        Args:
            show_id: Show identifier
            cluster: Voice cluster to match
            threshold: Similarity threshold for matching

        Returns:
            VoiceBankMatchResult with match info
        """
        if cluster.centroid is None:
            return self._create_unlabeled_match(show_id, cluster)

        cluster_embedding = np.array(cluster.centroid)

        entries = self.get_entries(show_id)

        best_match: Optional[VoiceBankEntry] = None
        best_similarity = 0.0

        for entry in entries:
            if not entry.embeddings:
                continue

            # Compute similarity with entry's embeddings
            entry_centroid = np.mean(entry.embeddings, axis=0)
            norm = np.linalg.norm(entry_centroid)
            if norm > 0:
                entry_centroid = entry_centroid / norm

            similarity = float(np.dot(cluster_embedding, entry_centroid))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        if best_match and best_similarity >= threshold:
            # Found a match
            speaker_id = self._make_speaker_id(best_match)

            return VoiceBankMatchResult(
                voice_cluster_id=cluster.voice_cluster_id,
                voice_bank_id=best_match.voice_bank_id,
                speaker_id=speaker_id,
                speaker_display_name=best_match.display_name,
                similarity=best_similarity,
                is_new_entry=False,
            )
        else:
            # No match found - create unlabeled entry
            return self._create_unlabeled_match(show_id, cluster)

    def _create_unlabeled_match(
        self,
        show_id: str,
        cluster: VoiceCluster,
    ) -> VoiceBankMatchResult:
        """Create an unlabeled voice bank entry for unmatched cluster.

        Args:
            show_id: Show identifier
            cluster: Voice cluster without match

        Returns:
            VoiceBankMatchResult for new unlabeled entry
        """
        if not self.config.auto_create_unlabeled:
            # Return a generic unlabeled result without creating an entry
            return VoiceBankMatchResult(
                voice_cluster_id=cluster.voice_cluster_id,
                voice_bank_id="voice_unlabeled",
                speaker_id="SPK_UNLABELED",
                speaker_display_name="Unlabeled Voice",
                similarity=None,
                is_new_entry=False,
            )

        entries = self.get_entries(show_id)

        # Count existing unlabeled entries
        unlabeled_count = sum(1 for e in entries if not e.is_labeled)

        if unlabeled_count >= self.config.max_unlabeled_per_episode:
            # Too many unlabeled - use generic
            return VoiceBankMatchResult(
                voice_cluster_id=cluster.voice_cluster_id,
                voice_bank_id="voice_unlabeled_overflow",
                speaker_id="SPK_UNLABELED",
                speaker_display_name="Unlabeled Voice",
                similarity=None,
                is_new_entry=False,
            )

        # Create new unlabeled entry
        unlabeled_num = unlabeled_count + 1
        voice_bank_id = f"voice_unlabeled_{unlabeled_num:02d}"
        speaker_id = f"SPK_UNLABELED_{unlabeled_num:02d}"
        display_name = f"Unlabeled Voice {unlabeled_num}"

        new_entry = VoiceBankEntry(
            voice_bank_id=voice_bank_id,
            show_id=show_id,
            display_name=display_name,
            embeddings=[cluster.centroid] if cluster.centroid else [],
            is_labeled=False,
        )

        self.add_entry(new_entry, show_id)

        return VoiceBankMatchResult(
            voice_cluster_id=cluster.voice_cluster_id,
            voice_bank_id=voice_bank_id,
            speaker_id=speaker_id,
            speaker_display_name=display_name,
            similarity=None,
            is_new_entry=True,
        )

    def _make_speaker_id(self, entry: VoiceBankEntry) -> str:
        """Generate speaker ID from voice bank entry.

        Args:
            entry: Voice bank entry

        Returns:
            Speaker ID string
        """
        if entry.cast_id:
            return f"SPK_{entry.cast_id.upper()}"
        else:
            # Use voice_bank_id as base
            base = entry.voice_bank_id.upper().replace("VOICE_", "")
            return f"SPK_{base}"

    def _bank_path(self, show_id: str) -> Path:
        """Get path to voice bank file for show."""
        return self.data_dir / f"{show_id.lower()}.json"

    def _load_bank(self, show_id: str) -> List[VoiceBankEntry]:
        """Load voice bank from file."""
        path = self._bank_path(show_id)

        if not path.exists():
            LOGGER.debug(f"Voice bank not found for show {show_id}")
            return []

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            entries = []
            for item in data:
                entries.append(VoiceBankEntry(**item))

            LOGGER.info(f"Loaded {len(entries)} voice bank entries for {show_id}")
            return entries

        except Exception as e:
            LOGGER.error(f"Failed to load voice bank for {show_id}: {e}")
            return []

    def _save_bank(self, show_id: str, entries: List[VoiceBankEntry]):
        """Save voice bank to file."""
        path = self._bank_path(show_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [entry.model_dump() for entry in entries]

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Update cache
        self._entries_cache[show_id] = entries

        LOGGER.debug(f"Saved {len(entries)} voice bank entries for {show_id}")


def match_voice_clusters_to_bank(
    show_id: str,
    clusters: List[VoiceCluster],
    output_path: Path,
    config: Optional[VoiceBankConfig] = None,
    similarity_threshold: float = 0.78,
    overwrite: bool = False,
) -> List[VoiceBankMatchResult]:
    """Match all voice clusters to the voice bank.

    Args:
        show_id: Show identifier
        clusters: List of voice clusters
        output_path: Path for voice mapping JSON
        config: Voice bank configuration
        similarity_threshold: Matching threshold
        overwrite: Whether to overwrite existing results

    Returns:
        List of VoiceBankMatchResult objects
    """
    if output_path.exists() and not overwrite:
        LOGGER.info(f"Voice mapping already exists: {output_path}")
        return _load_voice_mapping(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    bank = VoiceBank(config)

    results = []
    labeled_count = 0
    unlabeled_count = 0

    for cluster in clusters:
        match = bank.match_cluster(show_id, cluster, similarity_threshold)
        results.append(match)

        if match.similarity is not None:
            labeled_count += 1
        else:
            unlabeled_count += 1

    # Save mapping
    _save_voice_mapping(results, output_path)

    LOGGER.info(
        f"Voice mapping complete: {labeled_count} labeled, {unlabeled_count} unlabeled "
        f"from {len(clusters)} clusters"
    )

    return results


def _save_voice_mapping(results: List[VoiceBankMatchResult], output_path: Path):
    """Save voice mapping to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [r.model_dump() for r in results]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_voice_mapping(mapping_path: Path) -> List[VoiceBankMatchResult]:
    """Load voice mapping from JSON file."""
    with mapping_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return [VoiceBankMatchResult(**item) for item in data]


def create_voice_bank_entry_from_cast(
    show_id: str,
    cast_id: str,
    display_name: str,
    embeddings: Optional[List[List[float]]] = None,
    metadata: Optional[Dict] = None,
) -> VoiceBankEntry:
    """Create a labeled voice bank entry from cast information.

    Args:
        show_id: Show identifier
        cast_id: Cast member ID
        display_name: Display name for the voice
        embeddings: Optional initial embeddings
        metadata: Optional metadata

    Returns:
        VoiceBankEntry object
    """
    voice_bank_id = f"voice_{cast_id.lower()}"

    return VoiceBankEntry(
        voice_bank_id=voice_bank_id,
        show_id=show_id,
        cast_id=cast_id,
        display_name=display_name,
        embeddings=embeddings or [],
        metadata=metadata or {},
        is_labeled=True,
    )
