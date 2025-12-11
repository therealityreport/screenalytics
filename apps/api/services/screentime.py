"""Screen time analysis service using cast-linked faces."""

from __future__ import annotations

import csv
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from enum import Enum

from py_screenalytics.artifacts import get_path

LOGGER = logging.getLogger(__name__)


class OverlapPolicy(str, Enum):
    """Policy for handling overlapping speech in speaking time calculation."""
    SHARED = "shared"       # Split duration between active speakers
    FULL = "full"           # Each speaker gets full credit
    PRIMARY_ONLY = "primary"  # Only highest-probability speaker gets credit


@dataclass
class ScreenTimeConfig:
    """Configuration for screen time analysis."""

    quality_min: float = 0.7
    gap_tolerance_s: float = 0.5
    use_video_decode: bool = True
    screen_time_mode: Literal["faces", "tracks"] = "faces"
    edge_padding_s: float = 0.0
    track_coverage_min: float = 0.0
    # Minimum duration for a single-face interval to avoid zero-duration entries
    # Default: 1/30s (~33ms) which is approximately one frame at 30fps
    min_interval_duration_s: float = 0.033
    # Policy for handling overlapping speech (shared, full, primary)
    overlap_policy: OverlapPolicy = OverlapPolicy.SHARED


@dataclass
class CastMetrics:
    """Screen time metrics for a single cast member."""

    cast_id: str
    person_id: str
    visual_s: float
    speaking_s: float
    both_s: float
    confidence: float
    tracks_count: int
    faces_count: int


class ScreenTimeAnalyzer:
    """Analyze per-cast screen time from assigned faces and tracks."""

    def __init__(self, config: Optional[ScreenTimeConfig] = None):
        self.config = config or ScreenTimeConfig()

    def analyze_episode(self, ep_id: str) -> Dict[str, Any]:
        """Analyze screen time for an episode.

        Args:
            ep_id: Episode identifier (e.g., 'rhobh-s05e17')

        Returns:
            Dictionary with episode_id, generated_at, metrics list, and diagnostics
        """
        LOGGER.info(f"[screentime] Starting analysis for {ep_id}")

        # Initialize diagnostic counts
        diagnostics = {
            "faces_loaded": 0,
            "tracks_loaded": 0,
            "identities_loaded": 0,
            "identities_with_person_id": 0,
            "people_loaded": 0,
            "people_with_cast_id": 0,
            "tracks_mapped_to_identity": 0,
            "tracks_with_cast_id": 0,
            "tracks_missing_identity": 0,
            "tracks_missing_person": 0,
            "tracks_missing_cast": 0,
            "tracks_considered": 0,
            "tracks_below_quality": 0,
            "tracks_below_coverage": 0,
            "tracks_missing_interval": 0,
            "tracks_used_for_visuals": 0,
        }

        # Load artifacts
        faces = self._load_faces(ep_id)
        tracks = self._load_tracks(ep_id)
        identities = self._load_identities(ep_id)
        people, show_id = self._load_people(ep_id)
        cast_members = self._load_cast_members(show_id)

        diagnostics["faces_loaded"] = len(faces)
        diagnostics["tracks_loaded"] = len(tracks)
        diagnostics["identities_loaded"] = len(identities)
        diagnostics["people_loaded"] = len(people)
        diagnostics["cast_members_loaded"] = len(cast_members)

        LOGGER.info(
            f"[screentime] Loaded {len(faces)} faces, {len(tracks)} tracks, "
            f"{len(identities)} identities, {len(people)} people, {len(cast_members)} cast"
        )

        # Count identities with person_id
        identities_with_person = sum(1 for identity in identities if identity.get("person_id"))
        diagnostics["identities_with_person_id"] = identities_with_person
        diagnostics["identities_without_person_id"] = len(identities) - identities_with_person

        # Compute assignment status
        if len(identities) == 0:
            diagnostics["assignment_status"] = "no_clusters"
        elif identities_with_person == 0:
            diagnostics["assignment_status"] = "none"
        elif identities_with_person < len(identities):
            diagnostics["assignment_status"] = "partial"
        else:
            diagnostics["assignment_status"] = "complete"

        # Build mapping chain: track_id -> identity_id -> person_id -> cast_id
        track_to_identity = self._build_track_to_identity_map(identities)
        person_to_cast = self._build_person_to_cast_map(people)
        person_to_name = self._build_person_to_name_map(people, cast_members)
        identity_to_person = self._build_identity_to_person_map(ep_id, people, identities)
        tracks_by_id = self._index_tracks(tracks)

        diagnostics["tracks_mapped_to_identity"] = len(track_to_identity)
        diagnostics["people_with_cast_id"] = len(person_to_cast)

        stale_track_ids = [tid for tid in track_to_identity if tid not in tracks_by_id]
        if stale_track_ids:
            for tid in stale_track_ids:
                track_to_identity.pop(tid, None)
            LOGGER.info(
                "[screentime] Dropped %d stale track->identity mappings missing from tracks.jsonl",
                len(stale_track_ids),
            )

        LOGGER.info(
            f"[screentime] Built mappings: {len(track_to_identity)} track->identity, "
            f"{len(identity_to_person)} identity->person, {len(person_to_cast)} person->cast"
        )

        # Log sample mappings for debugging
        if track_to_identity:
            sample_tracks = list(track_to_identity.items())[:3]
            LOGGER.info(f"[screentime] Sample track->identity mappings: {sample_tracks}")
        if identity_to_person:
            sample_identities = list(identity_to_person.items())[:3]
            LOGGER.info(f"[screentime] Sample identity->person mappings: {sample_identities}")
        if person_to_cast:
            sample_people = list(person_to_cast.items())[:3]
            LOGGER.info(f"[screentime] Sample person->cast mappings: {sample_people}")

        # Group artifacts for fast lookup
        faces_by_track = self._group_faces_by_track(faces)

        # Analyze each cast member
        cast_metrics_map: Dict[str, CastMetrics] = {}
        cast_intervals: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        # Collect intervals for unassigned tracks (have identity but no person/cast)
        unassigned_intervals: List[Tuple[float, float]] = []

        for track_id, track_faces in faces_by_track.items():
            # Resolve track -> identity -> person -> cast
            identity_id = track_to_identity.get(track_id)
            if not identity_id:
                diagnostics["tracks_missing_identity"] += 1
                continue

            person_id = identity_to_person.get(identity_id)
            if not person_id:
                diagnostics["tracks_missing_person"] += 1
                LOGGER.debug(f"[screentime] Track {track_id} -> identity {identity_id} has no person_id mapping")
                # Collect interval for unassigned track
                track_meta = tracks_by_id.get(track_id)
                valid_faces = [face for face in track_faces if face.get("quality", 1.0) >= self.config.quality_min]
                if valid_faces:
                    intervals = self._build_track_intervals(track_id, valid_faces, track_meta)
                    unassigned_intervals.extend(intervals)
                continue

            cast_id = person_to_cast.get(person_id)
            if not cast_id:
                diagnostics["tracks_missing_cast"] += 1
                LOGGER.debug(f"[screentime] Track {track_id} -> person {person_id} has no cast_id, skipping")
                # Collect interval for track without cast assignment
                track_meta = tracks_by_id.get(track_id)
                valid_faces = [face for face in track_faces if face.get("quality", 1.0) >= self.config.quality_min]
                if valid_faces:
                    intervals = self._build_track_intervals(track_id, valid_faces, track_meta)
                    unassigned_intervals.extend(intervals)
                continue

            # This track has a full chain to cast_id
            diagnostics["tracks_with_cast_id"] += 1
            diagnostics["tracks_considered"] += 1

            # Initialize metrics for this cast member if needed
            if cast_id not in cast_metrics_map:
                cast_metrics_map[cast_id] = CastMetrics(
                    cast_id=cast_id,
                    person_id=person_id,
                    visual_s=0.0,
                    speaking_s=0.0,
                    both_s=0.0,
                    confidence=0.0,
                    tracks_count=0,
                    faces_count=0,
                )

            track_meta = tracks_by_id.get(track_id)
            metrics = cast_metrics_map[cast_id]
            metrics.tracks_count += 1
            metrics.faces_count += len(track_faces)

            if not track_faces:
                diagnostics["tracks_missing_interval"] += 1
                continue

            valid_faces = [face for face in track_faces if face.get("quality", 1.0) >= self.config.quality_min]

            if not valid_faces:
                diagnostics["tracks_below_quality"] += 1
                continue

            coverage = self._compute_track_coverage(track_faces, track_meta)
            if (
                self.config.screen_time_mode == "tracks"
                and self.config.track_coverage_min > 0.0
                and coverage < self.config.track_coverage_min
            ):
                diagnostics["tracks_below_coverage"] += 1
                LOGGER.debug(
                    "[screentime] Track %s skipped (coverage %.2f < %.2f)",
                    track_id,
                    coverage,
                    self.config.track_coverage_min,
                )
                continue

            intervals = self._build_track_intervals(track_id, valid_faces, track_meta)
            if not intervals:
                diagnostics["tracks_missing_interval"] += 1
                continue

            cast_intervals[cast_id].extend(intervals)
            diagnostics["tracks_used_for_visuals"] += 1

        # Store merged intervals for timeline visualization
        cast_merged_intervals: Dict[str, List[Tuple[float, float]]] = {}

        # Merge intervals per cast member and compute durations
        for cast_id, metrics in cast_metrics_map.items():
            intervals = cast_intervals.get(cast_id, [])
            if not intervals:
                metrics.visual_s = 0.0
                cast_merged_intervals[cast_id] = []
                continue

            padded = self._apply_edge_padding(intervals)
            merged = self._merge_intervals(padded)
            raw_duration = sum(self._interval_duration(interval) for interval in padded)
            merged_duration = sum(self._interval_duration(interval) for interval in merged)
            metrics.visual_s = merged_duration
            cast_merged_intervals[cast_id] = merged

            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "[screentime] Cast %s intervals: raw=%d merged=%d raw_duration=%.2fs merged_duration=%.2fs mode=%s",
                    cast_id,
                    len(padded),
                    len(merged),
                    raw_duration,
                    merged_duration,
                    self.config.screen_time_mode,
                )

        # Compute confidence scores based on data quality indicators
        # Confidence is derived from:
        #   - Number of tracks (more tracks = higher confidence, up to ~20 tracks for max)
        #   - Faces per track ratio (more faces per track = better tracking quality)
        # Formula: base_confidence + track_bonus + density_bonus, capped at 1.0
        for cast_id, metrics in cast_metrics_map.items():
            if metrics.faces_count > 0 and metrics.tracks_count > 0:
                # Base: 0.5 for having any data
                base = 0.5
                # Track bonus: +0.025 per track, max +0.3 (at ~12 tracks)
                track_bonus = min(0.3, metrics.tracks_count * 0.025)
                # Density bonus: faces/tracks ratio indicates tracking quality
                # Average ~3 faces/track is good; reward up to +0.2 for high density
                avg_faces_per_track = metrics.faces_count / metrics.tracks_count
                density_bonus = min(0.2, avg_faces_per_track * 0.04)
                metrics.confidence = min(1.0, base + track_bonus + density_bonus)
            else:
                metrics.confidence = 0.0

        LOGGER.info(f"[screentime] Analyzed {len(cast_metrics_map)} cast members")
        LOGGER.info(f"[screentime] Diagnostics: {diagnostics}")

        # Compute speaking time from transcript (if available)
        transcript = self._load_transcript(ep_id)
        voice_mapping = self._load_voice_mapping(ep_id)

        if transcript and voice_mapping:
            speaking_times = self._compute_speaking_time(transcript, voice_mapping)
            LOGGER.info(f"[screentime] Computed speaking times for {len(speaking_times)} cast members")

            # Apply speaking times to cast metrics
            for cast_id, metrics in cast_metrics_map.items():
                metrics.speaking_s = speaking_times.get(cast_id, 0.0)

            # Add diagnostics
            diagnostics["transcript_rows"] = len(transcript)
            diagnostics["voice_mappings"] = len(voice_mapping)
            diagnostics["speaking_time_computed"] = True
        else:
            diagnostics["transcript_rows"] = 0
            diagnostics["voice_mappings"] = 0
            diagnostics["speaking_time_computed"] = False
            if not transcript:
                LOGGER.info("[screentime] No transcript available, speaking_s will be 0")
            if not voice_mapping:
                LOGGER.info("[screentime] No voice mapping available, speaking_s will be 0")

        # Merge unassigned intervals
        merged_unassigned = self._merge_intervals(unassigned_intervals) if unassigned_intervals else []

        # Build timeline data for visualization
        timeline_data = []
        for m in sorted(cast_metrics_map.values(), key=lambda x: x.visual_s, reverse=True):
            name = person_to_name.get(m.person_id, "Unknown")
            intervals = cast_merged_intervals.get(m.cast_id, [])
            timeline_data.append({
                "name": name,
                "cast_id": m.cast_id,
                "intervals": [[round(start, 2), round(end, 2)] for start, end in intervals],
            })

        # Add unassigned track timeline entry
        if merged_unassigned:
            timeline_data.append({
                "name": "Unassigned",
                "cast_id": None,
                "intervals": [[round(start, 2), round(end, 2)] for start, end in merged_unassigned],
            })

        # Convert to output format
        return {
            "episode_id": ep_id,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "diagnostics": diagnostics,
            "metrics": [
                {
                    "name": person_to_name.get(m.person_id, "Unknown"),
                    "cast_id": m.cast_id,
                    "person_id": m.person_id,
                    "visual_s": round(m.visual_s, 2),
                    "speaking_s": round(m.speaking_s, 2),
                    "both_s": round(m.both_s, 2),
                    "confidence": round(m.confidence, 3),
                    "tracks_count": m.tracks_count,
                    "faces_count": m.faces_count,
                }
                for m in sorted(cast_metrics_map.values(), key=lambda x: x.visual_s, reverse=True)
            ],
            "timeline": timeline_data,
        }

    def _load_faces(self, ep_id: str) -> List[Dict[str, Any]]:
        """Load faces.jsonl for an episode."""
        faces_path = get_path(ep_id, "detections").parent / "faces.jsonl"
        if not faces_path.exists():
            raise FileNotFoundError(f"faces.jsonl not found: {faces_path}")

        faces = []
        with faces_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    face = json.loads(line)
                    faces.append(face)
                except json.JSONDecodeError:
                    continue
        return faces

    def _load_tracks(self, ep_id: str) -> List[Dict[str, Any]]:
        """Load tracks.jsonl for an episode."""
        tracks_path = get_path(ep_id, "tracks")
        if not tracks_path.exists():
            raise FileNotFoundError(f"tracks.jsonl not found: {tracks_path}")

        tracks = []
        with tracks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    track = json.loads(line)
                    tracks.append(track)
                except json.JSONDecodeError:
                    continue
        return tracks

    def _load_identities(self, ep_id: str) -> List[Dict[str, Any]]:
        """Load identities.json for an episode."""
        identities_path = get_path(ep_id, "detections").parent / "identities.json"
        if not identities_path.exists():
            raise FileNotFoundError(f"identities.json not found: {identities_path}")

        with identities_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("identities", [])

    def _load_transcript(self, ep_id: str) -> List[Dict[str, Any]]:
        """Load transcript JSONL for an episode.

        Returns empty list if transcript doesn't exist (audio pipeline not run).
        """
        data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
        transcript_path = data_root / "manifests" / ep_id / "transcript.jsonl"

        if not transcript_path.exists():
            LOGGER.debug(f"[screentime] No transcript found: {transcript_path}")
            return []

        rows = []
        with transcript_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows

    def _load_voice_mapping(self, ep_id: str) -> Dict[str, str]:
        """Load voice bank mapping (voice_cluster_id -> cast_id).

        Returns empty dict if voice mapping doesn't exist.
        """
        data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
        mapping_path = data_root / "manifests" / ep_id / "voice_bank_mapping.json"

        if not mapping_path.exists():
            LOGGER.debug(f"[screentime] No voice mapping found: {mapping_path}")
            return {}

        try:
            with mapping_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Build voice_cluster_id -> cast_id mapping
            mapping = {}
            for entry in data.get("mappings", []):
                voice_cluster_id = entry.get("voice_cluster_id")
                cast_id = entry.get("cast_id")
                if voice_cluster_id and cast_id:
                    mapping[voice_cluster_id] = cast_id
            return mapping
        except (json.JSONDecodeError, KeyError) as e:
            LOGGER.warning(f"[screentime] Error loading voice mapping: {e}")
            return {}

    def _compute_speaking_time(
        self,
        transcript: List[Dict[str, Any]],
        voice_mapping: Dict[str, str],
    ) -> Dict[str, float]:
        """Compute speaking time per cast member from transcript.

        Args:
            transcript: Transcript rows with speaker info and overlap fields
            voice_mapping: Mapping from voice_cluster_id to cast_id

        Returns:
            Dict mapping cast_id to speaking time in seconds
        """
        speaking_time: Dict[str, float] = defaultdict(float)

        for row in transcript:
            start = row.get("start", 0.0)
            end = row.get("end", 0.0)
            duration = max(0.0, end - start)

            if duration <= 0:
                continue

            voice_cluster_id = row.get("voice_cluster_id")
            voice_bank_id = row.get("voice_bank_id")

            # Try to map to cast_id
            cast_id = None
            if voice_cluster_id and voice_cluster_id in voice_mapping:
                cast_id = voice_mapping[voice_cluster_id]
            elif voice_bank_id and voice_bank_id.startswith("voice_"):
                # voice_bank_id format: "voice_{cast_id}"
                potential_cast = voice_bank_id.replace("voice_", "", 1)
                if potential_cast in voice_mapping.values():
                    cast_id = potential_cast

            if not cast_id:
                continue

            # Check for overlap and apply policy
            is_overlap = row.get("overlap", False)
            secondary_speakers = row.get("secondary_speakers", [])

            if is_overlap and secondary_speakers:
                if self.config.overlap_policy == OverlapPolicy.SHARED:
                    # Split duration among all active speakers
                    num_speakers = 1 + len(secondary_speakers)
                    speaking_time[cast_id] += duration / num_speakers

                    # Also credit secondary speakers if they map to cast
                    for sec_speaker in secondary_speakers:
                        sec_cast_id = voice_mapping.get(sec_speaker)
                        if sec_cast_id:
                            speaking_time[sec_cast_id] += duration / num_speakers

                elif self.config.overlap_policy == OverlapPolicy.FULL:
                    # Each speaker gets full credit
                    speaking_time[cast_id] += duration

                    for sec_speaker in secondary_speakers:
                        sec_cast_id = voice_mapping.get(sec_speaker)
                        if sec_cast_id:
                            speaking_time[sec_cast_id] += duration

                else:  # PRIMARY_ONLY
                    # Only primary speaker gets credit
                    speaking_time[cast_id] += duration
            else:
                # No overlap - full credit to primary speaker
                speaking_time[cast_id] += duration

        return dict(speaking_time)

    def _load_people(self, ep_id: str) -> Tuple[List[Dict[str, Any]], str]:
        """Load people.json for the show."""
        # Parse episode ID to get show
        parts = ep_id.split("-")
        if len(parts) < 2:
            raise ValueError(f"Invalid episode ID format: {ep_id}")
        show_id = parts[0].upper()

        data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
        people_path = data_root / "shows" / show_id / "people.json"

        if not people_path.exists():
            raise FileNotFoundError(f"people.json not found: {people_path}")

        with people_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("people", []), show_id

    def _build_track_to_identity_map(self, identities: List[Dict[str, Any]]) -> Dict[int, str]:
        """Build mapping from track_id to identity_id.

        Note: identities.json uses 'track_ids' field, not 'tracks'.
        """
        mapping = {}
        for identity in identities:
            identity_id = identity.get("identity_id") or identity.get("id")
            if not identity_id:
                continue

            # FIX: Use "track_ids" field (not "tracks")
            track_ids = identity.get("track_ids", [])
            for track_id in track_ids:
                mapping[int(track_id)] = identity_id

        return mapping

    def _build_person_to_cast_map(self, people: List[Dict[str, Any]]) -> Dict[str, str]:
        """Build mapping from person_id to cast_id."""
        mapping = {}
        for person in people:
            person_id = person.get("person_id")
            cast_id = person.get("cast_id")
            if person_id and cast_id:
                mapping[person_id] = cast_id

        return mapping

    def _load_cast_members(self, show_id: str) -> List[Dict[str, Any]]:
        """Load cast members for a show."""
        from apps.api.services.cast import CastService
        cast_service = CastService()
        try:
            return cast_service.list_cast(show_id)
        except Exception as e:
            LOGGER.warning(f"[screentime] Could not load cast members for {show_id}: {e}")
            return []

    def _build_person_to_name_map(
        self, people: List[Dict[str, Any]], cast_members: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        """Build mapping from person_id to name.

        Falls back to cast member name if person name is None or "None".
        """
        # Build cast_id -> name lookup for fallback
        cast_name_lookup = {}
        if cast_members:
            for cast in cast_members:
                cast_id = cast.get("cast_id")
                cast_name = cast.get("name")
                if cast_id and cast_name:
                    cast_name_lookup[cast_id] = cast_name

        mapping = {}
        for person in people:
            person_id = person.get("person_id")
            name = person.get("name")
            cast_id = person.get("cast_id")

            # Skip if no person_id
            if not person_id:
                continue

            # Use person name if valid (not None, not empty, not the string "None")
            if name and name != "None":
                mapping[person_id] = name
            # Fall back to cast member name
            elif cast_id and cast_id in cast_name_lookup:
                mapping[person_id] = cast_name_lookup[cast_id]
            # Use person_id as last resort
            else:
                mapping[person_id] = person_id

        return mapping

    def _build_identity_to_person_map(
        self,
        ep_id: str,
        people: List[Dict[str, Any]],
        identities: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, str]:
        """Build mapping from identity_id to person_id.

        PRIMARY: Use identity.person_id directly from identities.json (most reliable)
        FALLBACK: Use people.cluster_ids for backwards compatibility
        """
        mapping: Dict[str, str] = {}

        # PRIMARY: Use person_id directly from identities (most reliable source)
        if identities:
            for identity in identities:
                identity_id = identity.get("identity_id")
                person_id = identity.get("person_id")
                if identity_id and person_id:
                    mapping[identity_id] = person_id

        # FALLBACK: Fill gaps from people.cluster_ids (backwards compat)
        for person in people:
            person_id = person.get("person_id")
            if not person_id:
                continue

            cluster_ids = person.get("cluster_ids", [])
            for cluster_id in cluster_ids:
                # cluster_id format: "ep_id:identity_id"
                if ":" in cluster_id:
                    episode, identity = cluster_id.split(":", 1)
                    if episode == ep_id and identity not in mapping:
                        mapping[identity] = person_id
                elif cluster_id not in mapping:
                    # Legacy format without episode prefix
                    mapping[cluster_id] = person_id

        return mapping

    def _group_faces_by_track(self, faces: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Group face samples by track_id."""
        groups: Dict[int, List[Dict[str, Any]]] = {}
        for face in faces:
            track_id = face.get("track_id")
            if track_id is None:
                continue

            track_id = int(track_id)
            if track_id not in groups:
                groups[track_id] = []

            groups[track_id].append(face)

        return groups

    def _index_tracks(self, tracks: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Return a lookup from track_id to raw track metadata."""
        indexed: Dict[int, Dict[str, Any]] = {}
        for track in tracks:
            track_id = track.get("track_id")
            if track_id is None:
                continue
            indexed[int(track_id)] = track
        return indexed

    def _compute_track_coverage(
        self,
        track_faces: List[Dict[str, Any]],
        track_meta: Optional[Dict[str, Any]],
    ) -> float:
        """Estimate detection coverage for a track."""
        if not track_faces:
            return 0.0

        total_frames: Optional[float] = None
        if track_meta:
            frame_count = track_meta.get("frame_count")
            faces_count = track_meta.get("faces_count")
            start_frame = track_meta.get("first_frame_idx") or track_meta.get("start_frame")
            end_frame = track_meta.get("last_frame_idx") or track_meta.get("end_frame")
            if isinstance(frame_count, (int, float)) and frame_count > 0:
                total_frames = float(frame_count)
            elif isinstance(faces_count, (int, float)) and faces_count > 0:
                total_frames = float(faces_count)
            elif (
                isinstance(start_frame, (int, float))
                and isinstance(end_frame, (int, float))
                and end_frame > start_frame
            ):
                total_frames = float(end_frame - start_frame)

        if not total_frames or total_frames <= 0:
            total_frames = float(len(track_faces))

        coverage = len(track_faces) / max(total_frames, 1.0)
        return max(0.0, min(1.0, coverage))

    def _build_track_intervals(
        self,
        track_id: int,
        valid_faces: List[Dict[str, Any]],
        track_meta: Optional[Dict[str, Any]],
    ) -> List[Tuple[float, float]]:
        """Build intervals for a track based on the configured mode."""
        if self.config.screen_time_mode == "tracks":
            track_interval = self._track_interval_from_metadata(track_meta, valid_faces)
            if track_interval:
                return [track_interval]

        return self._faces_to_intervals(valid_faces, track_meta)

    def _faces_to_intervals(
        self,
        faces: List[Dict[str, Any]],
        track_meta: Optional[Dict[str, Any]],
    ) -> List[Tuple[float, float]]:
        """Convert per-face timestamps into contiguous intervals."""
        if not faces:
            return []

        if self.config.use_video_decode:
            sort_key = lambda f: f.get("ts", f.get("frame_idx", 0))
        else:
            sort_key = lambda f: f.get("frame_idx", f.get("ts", 0))

        sorted_faces = sorted(faces, key=sort_key)
        fps = self._estimate_fps(track_meta, sorted_faces)

        intervals: List[Tuple[float, float]] = []
        current_start: Optional[float] = None
        current_end: Optional[float] = None

        for face in sorted_faces:
            face_time = self._resolve_face_time(face, fps)
            if face_time is None:
                continue

            if current_start is None:
                current_start = face_time
                current_end = face_time
                continue

            gap = face_time - current_end
            if gap <= self.config.gap_tolerance_s:
                current_end = max(current_end, face_time)
            else:
                intervals.append((current_start, current_end))
                current_start = face_time
                current_end = face_time

        if current_start is not None and current_end is not None:
            intervals.append((current_start, current_end))

        return intervals

    def _track_interval_from_metadata(
        self,
        track_meta: Optional[Dict[str, Any]],
        faces: List[Dict[str, Any]],
    ) -> Optional[Tuple[float, float]]:
        """Derive a track span from metadata, falling back to faces if needed."""
        if not track_meta:
            return None

        start_ts: Optional[float] = None
        end_ts: Optional[float] = None

        if self.config.use_video_decode:
            start_ts = track_meta.get("first_ts") or track_meta.get("start_ts")
            end_ts = track_meta.get("last_ts") or track_meta.get("end_ts")

        fps = self._estimate_fps(track_meta, faces)
        start_frame = track_meta.get("first_frame_idx") or track_meta.get("start_frame")
        end_frame = track_meta.get("last_frame_idx") or track_meta.get("end_frame")

        if fps and start_frame is not None and end_frame is not None:
            if start_ts is None:
                start_ts = start_frame / fps
            if end_ts is None:
                end_ts = end_frame / fps

        if start_ts is None or end_ts is None:
            return None

        if end_ts < start_ts:
            start_ts, end_ts = end_ts, start_ts

        return start_ts, end_ts

    def _estimate_fps(
        self,
        track_meta: Optional[Dict[str, Any]],
        faces: List[Dict[str, Any]],
    ) -> Optional[float]:
        """Estimate FPS using track metadata or face samples."""
        if track_meta:
            fps_value = track_meta.get("fps") or track_meta.get("frame_rate")
            if fps_value:
                try:
                    return float(fps_value)
                except (TypeError, ValueError):
                    pass

            first_ts = track_meta.get("first_ts")
            last_ts = track_meta.get("last_ts")
            duration = None
            if isinstance(first_ts, (int, float)) and isinstance(last_ts, (int, float)) and last_ts > first_ts:
                duration = last_ts - first_ts

            first_frame = track_meta.get("first_frame_idx") or track_meta.get("start_frame")
            last_frame = track_meta.get("last_frame_idx") or track_meta.get("end_frame")

            if (
                duration
                and isinstance(first_frame, (int, float))
                and isinstance(last_frame, (int, float))
                and last_frame > first_frame
            ):
                return float(last_frame - first_frame) / duration

            frame_count = track_meta.get("frame_count")
            if duration and isinstance(frame_count, (int, float)) and frame_count > 1:
                return float(frame_count) / duration

        for face in faces:
            ts = face.get("ts")
            frame_idx = face.get("frame_idx")
            if isinstance(ts, (int, float)) and isinstance(frame_idx, (int, float)) and ts > 0 and frame_idx > 0:
                return frame_idx / ts

        return None

    def _resolve_face_time(self, face: Dict[str, Any], fps: Optional[float]) -> Optional[float]:
        """Resolve the timestamp for a face sample respecting config preferences."""
        ts = face.get("ts")
        if self.config.use_video_decode and isinstance(ts, (int, float)):
            return ts

        frame_idx = face.get("frame_idx")
        if fps and isinstance(frame_idx, (int, float)):
            return frame_idx / fps

        if isinstance(ts, (int, float)):
            return ts

        return None

    def _apply_edge_padding(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Pad interval edges to better match human perception."""
        pad = self.config.edge_padding_s
        if pad <= 0:
            return list(intervals)

        padded: List[Tuple[float, float]] = []
        for start, end in intervals:
            padded_start = max(0.0, start - pad)
            padded_end = end + pad
            padded.append((padded_start, padded_end))
        return padded

    def _merge_intervals(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Merge overlapping/nearby intervals using the configured gap tolerance."""
        if not intervals:
            return []

        sorted_intervals = sorted(intervals, key=lambda interval: interval[0])
        merged: List[List[float]] = [[sorted_intervals[0][0], sorted_intervals[0][1]]]

        for start, end in sorted_intervals[1:]:
            prev = merged[-1]
            if start - prev[1] <= self.config.gap_tolerance_s:
                prev[1] = max(prev[1], end)
            else:
                merged.append([start, end])

        return [(start, end) for start, end in merged]

    def _interval_duration(self, interval: Tuple[float, float]) -> float:
        """Return the duration of an interval, with a minimum to avoid zero-duration entries.

        Single-face intervals (where start == end) are given a minimum duration
        based on config.min_interval_duration_s to ensure they contribute to screen time.
        """
        start, end = interval
        raw_duration = max(0.0, end - start)
        # Apply minimum duration for zero-length intervals (e.g., single-face detections)
        if raw_duration < self.config.min_interval_duration_s:
            return self.config.min_interval_duration_s
        return raw_duration

    def write_outputs(self, ep_id: str, metrics_data: Dict[str, Any]) -> Tuple[Path, Path]:
        """Write screen time outputs to JSON and CSV.

        Returns:
            (json_path, csv_path)
        """
        data_root = Path(os.environ.get("SCREENALYTICS_DATA_ROOT", "data")).expanduser()
        analytics_dir = data_root / "analytics" / ep_id
        analytics_dir.mkdir(parents=True, exist_ok=True)

        json_path = analytics_dir / "screentime.json"
        csv_path = analytics_dir / "screentime.csv"

        # Write JSON
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)

        # Write CSV
        metrics = metrics_data.get("metrics", [])
        if metrics:
            fieldnames = [
                "name",
                "cast_id",
                "person_id",
                "visual_s",
                "speaking_s",
                "both_s",
                "confidence",
                "tracks_count",
                "faces_count",
            ]
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metrics)

        LOGGER.info(f"[screentime] Wrote outputs to {json_path} and {csv_path}")

        return json_path, csv_path


__all__ = ["ScreenTimeAnalyzer", "ScreenTimeConfig", "CastMetrics"]
