"""Screen time analysis service using cast-linked faces."""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from py_screenalytics.artifacts import get_path

LOGGER = logging.getLogger(__name__)


@dataclass
class ScreenTimeConfig:
    """Configuration for screen time analysis."""
    quality_min: float = 0.7
    gap_tolerance_s: float = 0.5
    use_video_decode: bool = True


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
        }

        # Load artifacts
        faces = self._load_faces(ep_id)
        tracks = self._load_tracks(ep_id)
        identities = self._load_identities(ep_id)
        people, show_id = self._load_people(ep_id)

        diagnostics["faces_loaded"] = len(faces)
        diagnostics["tracks_loaded"] = len(tracks)
        diagnostics["identities_loaded"] = len(identities)
        diagnostics["people_loaded"] = len(people)

        LOGGER.info(
            f"[screentime] Loaded {len(faces)} faces, {len(tracks)} tracks, "
            f"{len(identities)} identities, {len(people)} people"
        )

        # Count identities with person_id
        diagnostics["identities_with_person_id"] = sum(
            1 for identity in identities if identity.get("person_id")
        )

        # Build mapping chain: track_id -> identity_id -> person_id -> cast_id
        track_to_identity = self._build_track_to_identity_map(identities)
        person_to_cast = self._build_person_to_cast_map(people)
        person_to_name = self._build_person_to_name_map(people)
        identity_to_person = self._build_identity_to_person_map(ep_id, people)

        diagnostics["tracks_mapped_to_identity"] = len(track_to_identity)
        diagnostics["people_with_cast_id"] = len(person_to_cast)

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

        # Group faces by track_id
        faces_by_track = self._group_faces_by_track(faces)

        # Analyze each cast member
        cast_metrics_map: Dict[str, CastMetrics] = {}

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
                continue

            cast_id = person_to_cast.get(person_id)
            if not cast_id:
                diagnostics["tracks_missing_cast"] += 1
                LOGGER.debug(f"[screentime] Track {track_id} -> person {person_id} has no cast_id, skipping")
                continue

            # This track has a full chain to cast_id
            diagnostics["tracks_with_cast_id"] += 1

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

            # Compute screen time for this track
            visual_s, valid_faces = self._compute_track_screen_time(track_faces)

            # Update metrics
            metrics = cast_metrics_map[cast_id]
            metrics.visual_s += visual_s
            metrics.tracks_count += 1
            metrics.faces_count += len(track_faces)

        # Compute confidence scores
        for cast_id, metrics in cast_metrics_map.items():
            if metrics.faces_count > 0:
                # Simple confidence: could be enhanced with quality distribution analysis
                metrics.confidence = min(1.0, 0.5 + (metrics.tracks_count * 0.05))
            else:
                metrics.confidence = 0.0

        LOGGER.info(f"[screentime] Analyzed {len(cast_metrics_map)} cast members")
        LOGGER.info(f"[screentime] Diagnostics: {diagnostics}")

        # Convert to output format
        return {
            "episode_id": ep_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
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

    def _build_person_to_name_map(self, people: List[Dict[str, Any]]) -> Dict[str, str]:
        """Build mapping from person_id to name."""
        mapping = {}
        for person in people:
            person_id = person.get("person_id")
            name = person.get("name")
            if person_id and name:
                mapping[person_id] = name

        return mapping

    def _build_identity_to_person_map(
        self, ep_id: str, people: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Build mapping from identity_id (with ep_id prefix) to person_id."""
        mapping = {}
        for person in people:
            person_id = person.get("person_id")
            if not person_id:
                continue

            cluster_ids = person.get("cluster_ids", [])
            for cluster_id in cluster_ids:
                # cluster_id format: "ep_id:identity_id"
                if ":" in cluster_id:
                    episode, identity = cluster_id.split(":", 1)
                    if episode == ep_id:
                        mapping[identity] = person_id
                else:
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

    def _compute_track_screen_time(
        self, track_faces: List[Dict[str, Any]]
    ) -> Tuple[float, int]:
        """Compute screen time for a single track based on face samples.

        Returns:
            (visual_seconds, valid_faces_count)
        """
        if not track_faces:
            return 0.0, 0

        # Filter by quality threshold
        valid_faces = [
            f for f in track_faces
            if f.get("quality", 1.0) >= self.config.quality_min
        ]

        if not valid_faces:
            return 0.0, 0

        # Sort by frame index or timestamp
        sorted_faces = sorted(valid_faces, key=lambda f: f.get("frame_idx", 0))

        # Compute contiguous intervals
        intervals = []
        current_start = None
        current_end = None
        fps = None

        for face in sorted_faces:
            frame_idx = face.get("frame_idx")
            ts = face.get("ts")

            # Determine FPS from first face with both frame_idx and ts
            if fps is None and frame_idx is not None and ts is not None and frame_idx > 0:
                fps = frame_idx / ts if ts > 0 else 30.0

            # Use timestamp if available, otherwise estimate from frame_idx
            if ts is not None:
                face_time = ts
            elif frame_idx is not None and fps:
                face_time = frame_idx / fps
            else:
                continue

            if current_start is None:
                current_start = face_time
                current_end = face_time
            else:
                gap = face_time - current_end
                if gap <= self.config.gap_tolerance_s:
                    # Extend current interval
                    current_end = face_time
                else:
                    # Save current interval and start new one
                    intervals.append((current_start, current_end))
                    current_start = face_time
                    current_end = face_time

        # Save final interval
        if current_start is not None and current_end is not None:
            intervals.append((current_start, current_end))

        # Sum duration of all intervals
        total_duration = sum(end - start for start, end in intervals)

        return total_duration, len(valid_faces)

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
