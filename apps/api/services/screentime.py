"""Screen time analysis service using cast-linked faces."""

from __future__ import annotations

import csv
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from py_screenalytics.artifacts import get_path

LOGGER = logging.getLogger(__name__)


@dataclass
class ScreenTimeConfig:
    """Configuration for screen time analysis."""

    quality_min: float = 0.7
    gap_tolerance_s: float = 0.5
    use_video_decode: bool = True
    screen_time_mode: Literal["faces", "tracks"] = "faces"
    edge_padding_s: float = 0.0
    track_coverage_min: float = 0.0


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
            LOGGER.info(
                f"[screentime] Sample track->identity mappings: {sample_tracks}"
            )
        if identity_to_person:
            sample_identities = list(identity_to_person.items())[:3]
            LOGGER.info(
                f"[screentime] Sample identity->person mappings: {sample_identities}"
            )
        if person_to_cast:
            sample_people = list(person_to_cast.items())[:3]
            LOGGER.info(f"[screentime] Sample person->cast mappings: {sample_people}")

        # Group artifacts for fast lookup
        faces_by_track = self._group_faces_by_track(faces)
        tracks_by_id = self._index_tracks(tracks)

        # Analyze each cast member
        cast_metrics_map: Dict[str, CastMetrics] = {}
        cast_intervals: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        for track_id, track_faces in faces_by_track.items():
            # Resolve track -> identity -> person -> cast
            identity_id = track_to_identity.get(track_id)
            if not identity_id:
                diagnostics["tracks_missing_identity"] += 1
                continue

            person_id = identity_to_person.get(identity_id)
            if not person_id:
                diagnostics["tracks_missing_person"] += 1
                LOGGER.debug(
                    f"[screentime] Track {track_id} -> identity {identity_id} has no person_id mapping"
                )
                continue

            cast_id = person_to_cast.get(person_id)
            if not cast_id:
                diagnostics["tracks_missing_cast"] += 1
                LOGGER.debug(
                    f"[screentime] Track {track_id} -> person {person_id} has no cast_id, skipping"
                )
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

            valid_faces = [
                face
                for face in track_faces
                if face.get("quality", 1.0) >= self.config.quality_min
            ]

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

        # Merge intervals per cast member and compute durations
        for cast_id, metrics in cast_metrics_map.items():
            intervals = cast_intervals.get(cast_id, [])
            if not intervals:
                metrics.visual_s = 0.0
                continue

            padded = self._apply_edge_padding(intervals)
            merged = self._merge_intervals(padded)
            raw_duration = sum(self._interval_duration(interval) for interval in padded)
            merged_duration = sum(
                self._interval_duration(interval) for interval in merged
            )
            metrics.visual_s = merged_duration

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
                for m in sorted(
                    cast_metrics_map.values(), key=lambda x: x.visual_s, reverse=True
                )
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

    def _build_track_to_identity_map(
        self, identities: List[Dict[str, Any]]
    ) -> Dict[int, str]:
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

    def _group_faces_by_track(
        self, faces: List[Dict[str, Any]]
    ) -> Dict[int, List[Dict[str, Any]]]:
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
            start_frame = track_meta.get("first_frame_idx") or track_meta.get(
                "start_frame"
            )
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
            if (
                isinstance(first_ts, (int, float))
                and isinstance(last_ts, (int, float))
                and last_ts > first_ts
            ):
                duration = last_ts - first_ts

            first_frame = track_meta.get("first_frame_idx") or track_meta.get(
                "start_frame"
            )
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
            if (
                isinstance(ts, (int, float))
                and isinstance(frame_idx, (int, float))
                and ts > 0
                and frame_idx > 0
            ):
                return frame_idx / ts

        return None

    def _resolve_face_time(
        self, face: Dict[str, Any], fps: Optional[float]
    ) -> Optional[float]:
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

    def _apply_edge_padding(
        self, intervals: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
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

    def _merge_intervals(
        self, intervals: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
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
        """Return the non-negative duration of an interval."""
        start, end = interval
        return max(0.0, end - start)

    def write_outputs(
        self, ep_id: str, metrics_data: Dict[str, Any]
    ) -> Tuple[Path, Path]:
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
