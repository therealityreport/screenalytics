"""
Face-Body Track Fusion.

Associates face tracks with body tracks for continuous identity tracking.
"""

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class FaceBodyAssociation:
    """Association between a face track and body track."""

    face_track_id: int
    body_track_id: int
    confidence: float
    method: str  # "iou" | "reid" | "temporal"
    frame_range: Tuple[int, int]  # (start_frame, end_frame)
    best_iou: float | None = None
    best_similarity: float | None = None

    def to_dict(self) -> dict:
        payload: dict = {
            "face_track_id": self.face_track_id,
            "body_track_id": self.body_track_id,
            "confidence": self.confidence,
            "method": self.method,
            "frame_range": list(self.frame_range),
        }
        if self.best_iou is not None:
            payload["best_iou"] = round(float(self.best_iou), 6)
        if self.best_similarity is not None:
            payload["best_similarity"] = round(float(self.best_similarity), 6)
        return payload


@dataclass
class FusedIdentity:
    """Identity that may span face and body tracks."""

    identity_id: str
    face_track_ids: List[int] = field(default_factory=list)
    body_track_ids: List[int] = field(default_factory=list)
    associations: List[FaceBodyAssociation] = field(default_factory=list)

    # Screen time breakdown
    face_visible_frames: int = 0
    body_only_frames: int = 0
    total_frames: int = 0

    def to_dict(self) -> dict:
        methods = {a.method for a in self.associations}
        has_iou = "iou" in methods
        has_reid = "reid" in methods
        attribution_source: str | None
        if not (self.face_track_ids and self.body_track_ids):
            attribution_source = None
        elif has_iou and has_reid:
            attribution_source = "hybrid"
        elif has_reid:
            attribution_source = "reid"
        elif has_iou:
            attribution_source = "iou"
        else:
            attribution_source = None

        best_iou: float | None = None
        best_similarity: float | None = None
        for assoc in self.associations:
            if assoc.best_iou is not None:
                best_iou = assoc.best_iou if best_iou is None else max(best_iou, assoc.best_iou)
            if assoc.best_similarity is not None:
                best_similarity = (
                    assoc.best_similarity if best_similarity is None else max(best_similarity, assoc.best_similarity)
                )

        return {
            "identity_id": self.identity_id,
            "face_track_ids": self.face_track_ids,
            "body_track_ids": self.body_track_ids,
            "associations": [a.to_dict() for a in self.associations],
            "face_visible_frames": self.face_visible_frames,
            "body_only_frames": self.body_only_frames,
            "total_frames": self.total_frames,
            "attribution": {
                "source": attribution_source,
                "best_iou": (round(float(best_iou), 6) if best_iou is not None else None),
                "best_similarity": (round(float(best_similarity), 6) if best_similarity is not None else None),
            },
        }


class TrackFusion:
    """Face-body track fusion engine."""

    def __init__(
        self,
        iou_threshold: float = 0.02,
        min_overlap_ratio: float = 0.7,
        reid_similarity_threshold: float = 0.70,
        max_gap_seconds: float = 30.0,
        face_in_upper_body: bool = True,
        upper_body_fraction: float = 0.5,
    ):
        self.iou_threshold = iou_threshold
        self.min_overlap_ratio = min_overlap_ratio
        self.reid_similarity_threshold = reid_similarity_threshold
        self.max_gap_seconds = max_gap_seconds
        self.face_in_upper_body = face_in_upper_body
        self.upper_body_fraction = upper_body_fraction
        self.last_diagnostics: dict[str, object] = {}

    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def compute_face_in_body_score(
        self,
        face_box: List[float],
        body_box: List[float],
    ) -> float:
        """
        Compute how well a face is contained in a body box.

        Returns score between 0-1, higher means face is well-positioned.
        """
        # Check overlap
        x1 = max(face_box[0], body_box[0])
        y1 = max(face_box[1], body_box[1])
        x2 = min(face_box[2], body_box[2])
        y2 = min(face_box[3], body_box[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        face_area = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])

        if face_area == 0:
            return 0.0

        overlap_ratio = inter_area / face_area

        if overlap_ratio < self.min_overlap_ratio:
            return 0.0

        # IoU gating: note that when face is fully inside body, IoU ~= face_area / body_area.
        # This is expected to be small (e.g. 0.02-0.10), so keep iou_threshold conservative.
        body_area = (body_box[2] - body_box[0]) * (body_box[3] - body_box[1])
        union_area = face_area + body_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0
        if iou < self.iou_threshold:
            return 0.0

        # Check if face is in upper portion of body
        if self.face_in_upper_body:
            body_height = body_box[3] - body_box[1]
            upper_body_y = body_box[1] + body_height * self.upper_body_fraction
            face_center_y = (face_box[1] + face_box[3]) / 2

            if face_center_y > upper_body_y:
                # Face is in lower portion, reduce score
                return overlap_ratio * 0.5

        return overlap_ratio

    def associate_by_iou(
        self,
        face_detections: List[dict],
        body_detections: List[dict],
        frame_idx: int,
    ) -> List[Tuple[int, int, float]]:
        """
        Associate faces with bodies in a single frame using IoU.

        Returns list of (face_idx, body_idx, score) tuples.
        """
        if not face_detections or not body_detections:
            return []

        associations = []

        for face_idx, face_det in enumerate(face_detections):
            face_box = face_det["bbox"]
            best_body_idx = -1
            best_score = 0.0

            for body_idx, body_det in enumerate(body_detections):
                body_box = body_det["bbox"]
                score = self.compute_face_in_body_score(face_box, body_box)

                if score > best_score and score >= self.min_overlap_ratio:
                    best_score = score
                    best_body_idx = body_idx

            if best_body_idx >= 0:
                associations.append((face_idx, best_body_idx, best_score))

        return associations

    def associate_by_reid(
        self,
        body_embeddings: np.ndarray,
        body_meta: List[dict],
        face_track_embedding: np.ndarray,
    ) -> List[Tuple[int, float]]:
        """
        Find body tracks matching a face track using Re-ID similarity.

        Returns list of (body_track_id, similarity) tuples.
        """
        if len(body_embeddings) == 0:
            return []

        # Compute cosine similarity
        face_norm = face_track_embedding / (np.linalg.norm(face_track_embedding) + 1e-8)
        body_norms = body_embeddings / (np.linalg.norm(body_embeddings, axis=1, keepdims=True) + 1e-8)

        similarities = np.dot(body_norms, face_norm)

        # Group by body track and get max similarity
        track_similarities: Dict[int, float] = {}
        for i, meta in enumerate(body_meta):
            track_id = meta["track_id"]
            sim = similarities[i]
            if track_id not in track_similarities or sim > track_similarities[track_id]:
                track_similarities[track_id] = sim

        # Filter by threshold
        results = [
            (track_id, sim)
            for track_id, sim in track_similarities.items()
            if sim >= self.reid_similarity_threshold
        ]

        return sorted(results, key=lambda x: -x[1])

    def fuse_tracks(
        self,
        face_tracks: Dict[int, dict],
        body_tracks: Dict[int, dict],
        body_embeddings: Optional[np.ndarray] = None,
        body_embeddings_meta: Optional[List[dict]] = None,
        face_embeddings: Optional[np.ndarray] = None,
        face_embeddings_meta: Optional[List[dict]] = None,
    ) -> Dict[str, FusedIdentity]:
        """
        Fuse face and body tracks into unified identities.

        Args:
            face_tracks: Dict of face_track_id -> track data
            body_tracks: Dict of body_track_id -> track data
            body_embeddings: Re-ID embeddings for body tracks (optional)
            body_embeddings_meta: Metadata for body embeddings
            face_embeddings: Face embeddings (optional)
            face_embeddings_meta: Metadata for face embeddings

        Returns:
            Dict of identity_id -> FusedIdentity
        """
        logger.info(f"Fusing {len(face_tracks)} face tracks with {len(body_tracks)} body tracks")

        # Phase 1: IoU-based frame-by-frame association
        iou_associations, spatial_diag = self._build_iou_associations(face_tracks, body_tracks)
        logger.info(f"  IoU associations: {len(iou_associations)}")

        # Phase 2: Re-ID handoff for gaps
        reid_associations: List[FaceBodyAssociation] = []
        reid_diag: dict[str, object] = {"reid_enabled": False, "reid_comparisons": 0, "reid_pass": 0}
        reid_skip_reason: str | None = None
        if body_embeddings is None:
            reid_skip_reason = "missing_body_embeddings"
        elif not body_embeddings_meta:
            reid_skip_reason = "missing_body_embeddings_meta"
        elif face_embeddings is None:
            reid_skip_reason = "missing_face_embeddings"
        elif not face_embeddings_meta:
            reid_skip_reason = "missing_face_embeddings_meta"
        elif not isinstance(body_embeddings, np.ndarray) or body_embeddings.size == 0:
            reid_skip_reason = "empty_body_embeddings"
        elif not isinstance(face_embeddings, np.ndarray) or face_embeddings.size == 0:
            reid_skip_reason = "empty_face_embeddings"
        elif body_embeddings.ndim != 2 or face_embeddings.ndim != 2:
            reid_skip_reason = "invalid_embedding_shape"
        elif body_embeddings.shape[1] != face_embeddings.shape[1]:
            reid_skip_reason = "embedding_dim_mismatch"
        else:
            reid_diag = {
                "reid_enabled": True,
                "reid_comparisons": 0,
                "reid_pass": 0,
            }
            reid_associations, reid_comparisons = self._build_reid_associations(
                face_tracks,
                body_tracks,
                face_embeddings,
                face_embeddings_meta,
                body_embeddings,
                body_embeddings_meta,
            )
            logger.info(f"  Re-ID associations: {len(reid_associations)}")
            reid_diag["reid_comparisons"] = int(reid_comparisons)
            reid_diag["reid_pass"] = len(reid_associations)
        if reid_skip_reason:
            reid_diag["reid_skip_reason"] = reid_skip_reason

        # Phase 3: Build fused identities
        all_associations = iou_associations + reid_associations
        identities = self._build_fused_identities(face_tracks, body_tracks, all_associations)
        logger.info(f"  Fused identities: {len(identities)}")

        # Diagnostics: reconcile final_pairs with association sources.
        final_pairs = 0
        iou_pairs = 0
        reid_pairs = 0
        hybrid_pairs = 0
        for identity in identities.values():
            if not (identity.face_track_ids and identity.body_track_ids):
                continue
            final_pairs += 1
            methods = {assoc.method for assoc in identity.associations}
            has_iou = "iou" in methods
            has_reid = "reid" in methods
            if has_iou and has_reid:
                hybrid_pairs += 1
            elif has_reid:
                reid_pairs += 1
            else:
                iou_pairs += 1

        self.last_diagnostics = {
            "schema_version": 1,
            "candidate_overlaps": int(spatial_diag.get("candidate_overlaps", 0) or 0),
            "iou_pass": int(spatial_diag.get("iou_pass", 0) or 0),
            "overlap_ratio_pass": int(spatial_diag.get("overlap_ratio_pass", 0) or 0),
            "iou_distribution": spatial_diag.get("iou_distribution"),
            "overlap_ratio_distribution": spatial_diag.get("overlap_ratio_distribution"),
            "reid_enabled": bool(reid_diag.get("reid_enabled", False)),
            "reid_comparisons": int(reid_diag.get("reid_comparisons", 0) or 0),
            "reid_pass": int(reid_diag.get("reid_pass", 0) or 0),
            "reid_skip_reason": reid_diag.get("reid_skip_reason"),
            "final_pairs": final_pairs,
            "iou_pairs": iou_pairs,
            "reid_pairs": reid_pairs,
            "hybrid_pairs": hybrid_pairs,
            "config": {
                "iou_threshold": float(self.iou_threshold),
                "min_overlap_ratio": float(self.min_overlap_ratio),
                "face_in_upper_body": bool(self.face_in_upper_body),
                "upper_body_fraction": float(self.upper_body_fraction),
                "reid_similarity_threshold": float(self.reid_similarity_threshold),
                "max_gap_seconds": float(self.max_gap_seconds),
            },
        }

        return identities

    def _build_iou_associations(
        self,
        face_tracks: Dict[int, dict],
        body_tracks: Dict[int, dict],
    ) -> tuple[List[FaceBodyAssociation], dict[str, object]]:
        """Build associations using frame-by-frame IoU."""
        # Index body detections by frame
        body_by_frame: Dict[int, List[Tuple[int, dict]]] = defaultdict(list)
        for track_id, track in body_tracks.items():
            for det in track.get("detections", []):
                frame_idx = det["frame_idx"]
                body_by_frame[frame_idx].append((track_id, det))

        def _face_in_upper(face_box: list[float], body_box: list[float]) -> bool:
            if not self.face_in_upper_body:
                return True
            try:
                face_center_y = (float(face_box[1]) + float(face_box[3])) / 2.0
                body_y1 = float(body_box[1])
                body_y2 = float(body_box[3])
            except (TypeError, ValueError, IndexError):
                return False
            body_height = body_y2 - body_y1
            upper_body_y = body_y1 + body_height * float(self.upper_body_fraction)
            return face_center_y <= upper_body_y

        def _iou_and_overlap_ratio(face_box: list[float], body_box: list[float]) -> tuple[float, float]:
            try:
                fx1, fy1, fx2, fy2 = (float(v) for v in face_box)
                bx1, by1, bx2, by2 = (float(v) for v in body_box)
            except (TypeError, ValueError):
                return 0.0, 0.0
            ix1, iy1 = max(fx1, bx1), max(fy1, by1)
            ix2, iy2 = min(fx2, bx2), min(fy2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0, 0.0
            face_area = max(0.0, fx2 - fx1) * max(0.0, fy2 - fy1)
            body_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = face_area + body_area - inter
            iou = inter / union if union > 0 else 0.0
            overlap_ratio = inter / face_area if face_area > 0 else 0.0
            return iou, overlap_ratio

        # For each face track, find overlapping body tracks
        associations = []
        candidate_overlaps = 0
        overlap_ratio_pass = 0
        iou_pass = 0

        sample_limit = 50_000
        iou_samples: list[float] = []
        overlap_samples: list[float] = []
        iou_seen = 0
        overlap_seen = 0
        rng = random.Random(0)

        def _reservoir_push(sample_list: list[float], value: float, *, seen: int) -> int:
            seen += 1
            if len(sample_list) < sample_limit:
                sample_list.append(value)
                return seen
            j = rng.randrange(seen)
            if j < sample_limit:
                sample_list[j] = value
            return seen

        def _distribution(sample_list: list[float]) -> dict[str, float] | None:
            if not sample_list:
                return None
            arr = np.asarray(sample_list, dtype=np.float32)
            return {
                "min": float(np.min(arr)),
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
                "max": float(np.max(arr)),
            }

        for face_track_id, face_track in face_tracks.items():
            body_overlap_counts: Dict[int, int] = defaultdict(int)
            body_overlap_scores: Dict[int, List[float]] = defaultdict(list)
            best_iou_by_body: Dict[int, float] = defaultdict(float)
            frame_ranges: Dict[int, Tuple[int, int]] = {}

            for face_det in face_track.get("detections", []):
                frame_idx = face_det["frame_idx"]
                face_box = face_det["bbox"]

                for body_track_id, body_det in body_by_frame.get(frame_idx, []):
                    body_box = body_det["bbox"]
                    if not _face_in_upper(face_box, body_box):
                        continue

                    iou, overlap_ratio = _iou_and_overlap_ratio(face_box, body_box)
                    candidate_overlaps += 1
                    iou_seen = _reservoir_push(iou_samples, float(iou), seen=iou_seen)
                    overlap_seen = _reservoir_push(overlap_samples, float(overlap_ratio), seen=overlap_seen)

                    if overlap_ratio >= self.min_overlap_ratio:
                        overlap_ratio_pass += 1
                        if iou >= self.iou_threshold:
                            iou_pass += 1

                    score = self.compute_face_in_body_score(face_box, body_box)

                    if score >= self.min_overlap_ratio:
                        body_overlap_counts[body_track_id] += 1
                        body_overlap_scores[body_track_id].append(score)
                        best_iou_by_body[body_track_id] = max(best_iou_by_body[body_track_id], float(iou))

                        if body_track_id not in frame_ranges:
                            frame_ranges[body_track_id] = (frame_idx, frame_idx)
                        else:
                            start, end = frame_ranges[body_track_id]
                            frame_ranges[body_track_id] = (min(start, frame_idx), max(end, frame_idx))

            # Create associations for body tracks with sufficient overlap
            for body_track_id, count in body_overlap_counts.items():
                if count >= 3:  # Minimum frames for association
                    avg_score = np.mean(body_overlap_scores[body_track_id])
                    associations.append(FaceBodyAssociation(
                        face_track_id=face_track_id,
                        body_track_id=body_track_id,
                        confidence=avg_score,
                        method="iou",
                        frame_range=frame_ranges[body_track_id],
                        best_iou=best_iou_by_body.get(body_track_id),
                    ))

        diagnostics: dict[str, object] = {
            "candidate_overlaps": candidate_overlaps,
            "overlap_ratio_pass": overlap_ratio_pass,
            "iou_pass": iou_pass,
            "iou_distribution": _distribution(iou_samples),
            "overlap_ratio_distribution": _distribution(overlap_samples),
        }

        return associations, diagnostics

    def _build_reid_associations(
        self,
        face_tracks: Dict[int, dict],
        body_tracks: Dict[int, dict],
        face_embeddings: np.ndarray,
        face_embeddings_meta: List[dict],
        body_embeddings: np.ndarray,
        body_embeddings_meta: List[dict],
    ) -> tuple[list[FaceBodyAssociation], int]:
        """Build associations using Re-ID when face disappears."""
        # Group face embeddings by track
        face_emb_by_track: Dict[int, np.ndarray] = {}
        for i, meta in enumerate(face_embeddings_meta):
            track_id = meta.get("track_id")
            if track_id is not None:
                if track_id not in face_emb_by_track:
                    face_emb_by_track[track_id] = []
                face_emb_by_track[track_id].append(face_embeddings[i])

        # Average face embeddings per track
        face_avg_emb: Dict[int, np.ndarray] = {}
        for track_id, embs in face_emb_by_track.items():
            face_avg_emb[track_id] = np.mean(embs, axis=0)

        associations = []
        comparisons = 0

        unique_body_tracks: set[int] = set()
        for meta in body_embeddings_meta:
            track_id = meta.get("track_id")
            if track_id is None:
                continue
            try:
                unique_body_tracks.add(int(track_id))
            except (TypeError, ValueError):
                continue
        unique_body_track_count = len(unique_body_tracks)

        # Find body tracks that could continue a face track
        for face_track_id, face_track in face_tracks.items():
            if face_track_id not in face_avg_emb:
                continue

            face_end_frame = face_track.get("end_frame", 0)
            face_emb = face_avg_emb[face_track_id]

            # Find body tracks that start after this face track ends
            comparisons += unique_body_track_count
            candidates = self.associate_by_reid(
                body_embeddings, body_embeddings_meta, face_emb
            )

            for body_track_id, similarity in candidates:
                body_track = body_tracks.get(body_track_id)
                if body_track is None:
                    continue

                body_start_frame = body_track.get("start_frame", 0)

                # Check temporal gap
                gap_frames = body_start_frame - face_end_frame
                if gap_frames < 0:
                    continue  # Body track starts before face ends (use IoU instead)

                # TODO: Convert frames to seconds using FPS
                # For now, assume 24 fps
                gap_seconds = gap_frames / 24.0
                if gap_seconds > self.max_gap_seconds:
                    continue

                associations.append(FaceBodyAssociation(
                    face_track_id=face_track_id,
                    body_track_id=body_track_id,
                    confidence=similarity,
                    method="reid",
                    frame_range=(face_end_frame, body_track.get("end_frame", body_start_frame)),
                    best_similarity=float(similarity),
                ))

        return associations, comparisons

    def _build_fused_identities(
        self,
        face_tracks: Dict[int, dict],
        body_tracks: Dict[int, dict],
        associations: List[FaceBodyAssociation],
    ) -> Dict[str, FusedIdentity]:
        """Build fused identities from associations."""
        # Use union-find to group related tracks
        parent: Dict[str, str] = {}  # "face_123" or "body_456" -> parent

        def make_key(track_type: str, track_id: int) -> str:
            return f"{track_type}_{track_id}"

        def find(x: str) -> str:
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Initialize all tracks
        for face_id in face_tracks:
            find(make_key("face", face_id))
        for body_id in body_tracks:
            find(make_key("body", body_id))

        # Union associated tracks
        for assoc in associations:
            face_key = make_key("face", assoc.face_track_id)
            body_key = make_key("body", assoc.body_track_id)
            union(face_key, body_key)

        # Group tracks by root
        groups: Dict[str, Set[str]] = defaultdict(set)
        for key in parent:
            groups[find(key)].add(key)

        # Build identities
        identities: Dict[str, FusedIdentity] = {}
        identity_idx = 0

        for root, members in groups.items():
            face_ids = []
            body_ids = []

            for member in members:
                track_type, track_id = member.split("_", 1)
                track_id = int(track_id)
                if track_type == "face":
                    face_ids.append(track_id)
                else:
                    body_ids.append(track_id)

            # Find associations for this identity
            identity_assocs = [
                a for a in associations
                if a.face_track_id in face_ids or a.body_track_id in body_ids
            ]

            identity_id = f"fused_{identity_idx:04d}"
            identity_idx += 1

            # Calculate screen time breakdown
            face_frames = set()
            body_frames = set()

            for face_id in face_ids:
                track = face_tracks.get(face_id, {})
                for det in track.get("detections", []):
                    face_frames.add(det["frame_idx"])

            for body_id in body_ids:
                track = body_tracks.get(body_id, {})
                for det in track.get("detections", []):
                    body_frames.add(det["frame_idx"])

            body_only_frames = body_frames - face_frames
            all_frames = face_frames | body_frames

            identities[identity_id] = FusedIdentity(
                identity_id=identity_id,
                face_track_ids=sorted(face_ids),
                body_track_ids=sorted(body_ids),
                associations=identity_assocs,
                face_visible_frames=len(face_frames),
                body_only_frames=len(body_only_frames),
                total_frames=len(all_frames),
            )

        return identities


def fuse_face_body_tracks(
    fusion: TrackFusion,
    face_tracks_path: Path,
    body_tracks_path: Path,
    body_embeddings_path: Optional[Path] = None,
    embeddings_meta: Optional[dict] = None,
    output_path: Path = None,
) -> Dict[str, FusedIdentity]:
    """
    Fuse face and body tracks.

    Args:
        fusion: TrackFusion instance
        face_tracks_path: Path to faces.jsonl
        body_tracks_path: Path to body_tracks.jsonl
        body_embeddings_path: Path to body_embeddings.npy (optional)
        embeddings_meta: Metadata for embeddings
        output_path: Path to output track_fusion.json

    Returns:
        Dict of fused identities
    """
    face_tracks_path = Path(face_tracks_path)
    body_tracks_path = Path(body_tracks_path)

    # Load face tracks - aggregate individual detections into tracks
    # faces.jsonl has one line per face detection (not aggregated tracks)
    face_tracks: Dict[int, dict] = {}
    best_face_embeddings: dict[int, tuple[float, np.ndarray]] = {}
    logger.info(f"Loading face tracks from: {face_tracks_path}")

    with open(face_tracks_path) as f:
        for line in f:
            face = json.loads(line)
            track_id = face.get("track_id")
            if track_id is None:
                continue

            # Initialize track if not seen
            if track_id not in face_tracks:
                face_tracks[track_id] = {
                    "track_id": track_id,
                    "detections": [],
                    "start_frame": face.get("frame_idx"),
                    "end_frame": face.get("frame_idx"),
                    "start_time": face.get("ts", 0.0),
                    "end_time": face.get("ts", 0.0),
                }

            # Add detection
            face_tracks[track_id]["detections"].append({
                "frame_idx": face.get("frame_idx"),
                "timestamp": face.get("ts", 0.0),
                "bbox": face.get("bbox_xyxy") or face.get("bbox"),
                "score": face.get("conf", 1.0),
            })

            embedding = face.get("embedding")
            if isinstance(embedding, list) and embedding:
                try:
                    emb = np.asarray(embedding, dtype=np.float32)
                    if emb.ndim == 1 and emb.size > 0:
                        quality_raw = face.get("quality")
                        quality = float(quality_raw) if quality_raw is not None else 0.0
                        prev = best_face_embeddings.get(track_id)
                        if prev is None or quality >= prev[0]:
                            best_face_embeddings[track_id] = (quality, emb)
                except Exception:
                    pass

            # Update track bounds
            frame_idx = face.get("frame_idx")
            ts = face.get("ts", 0.0)
            if frame_idx is not None:
                if frame_idx < face_tracks[track_id]["start_frame"]:
                    face_tracks[track_id]["start_frame"] = frame_idx
                    face_tracks[track_id]["start_time"] = ts
                if frame_idx > face_tracks[track_id]["end_frame"]:
                    face_tracks[track_id]["end_frame"] = frame_idx
                    face_tracks[track_id]["end_time"] = ts

    # Calculate durations and frame counts
    for track in face_tracks.values():
        track["frame_count"] = len(track["detections"])
        track["duration"] = track["end_time"] - track["start_time"]

    logger.info(f"Loaded {len(face_tracks)} face tracks (aggregated from detections)")

    face_embeddings = None
    face_embeddings_meta = None
    if best_face_embeddings:
        dim_counts: dict[int, int] = {}
        for _track_id, (_quality, emb) in best_face_embeddings.items():
            dim = int(getattr(emb, "size", 0) or 0)
            if dim > 0:
                dim_counts[dim] = dim_counts.get(dim, 0) + 1
        target_dim = max(dim_counts.items(), key=lambda kv: kv[1])[0] if dim_counts else None
        if target_dim:
            embeddings_list: list[np.ndarray] = []
            meta_list: list[dict] = []
            for track_id in sorted(best_face_embeddings):
                emb = best_face_embeddings[track_id][1]
                if emb.ndim == 1 and emb.size == target_dim:
                    embeddings_list.append(emb)
                    meta_list.append({"track_id": track_id})
            if embeddings_list:
                face_embeddings = np.stack(embeddings_list, axis=0).astype(np.float32, copy=False)
                face_embeddings_meta = meta_list

    # Load body tracks
    body_tracks: Dict[int, dict] = {}
    logger.info(f"Loading body tracks from: {body_tracks_path}")

    with open(body_tracks_path) as f:
        for line in f:
            track = json.loads(line)
            track_id = track.get("track_id")
            if track_id is not None:
                body_tracks[track_id] = track

    logger.info(f"Loaded {len(body_tracks)} body tracks")

    # Load body embeddings if available
    body_embeddings = None
    body_embeddings_meta = None

    if body_embeddings_path and Path(body_embeddings_path).exists():
        body_embeddings = np.load(body_embeddings_path)
        if embeddings_meta:
            body_embeddings_meta = embeddings_meta.get("entries", [])
        logger.info(f"Loaded body embeddings: {body_embeddings.shape}")

    # Run fusion
    identities = fusion.fuse_tracks(
        face_tracks=face_tracks,
        body_tracks=body_tracks,
        body_embeddings=body_embeddings,
        body_embeddings_meta=body_embeddings_meta,
        face_embeddings=face_embeddings,
        face_embeddings_meta=face_embeddings_meta,
    )
    # Prefer upstream skip reason (e.g., torchreid runtime failure) over derived input gaps.
    diagnostics = getattr(fusion, "last_diagnostics", None)
    if isinstance(diagnostics, dict) and isinstance(embeddings_meta, dict):
        meta_skip_reason = embeddings_meta.get("reid_skip_reason")
        if isinstance(meta_skip_reason, str) and meta_skip_reason.strip():
            meta_skip_reason = meta_skip_reason.strip()
            existing_skip = diagnostics.get("reid_skip_reason")
            secondary: list[str] = []
            if isinstance(existing_skip, str) and existing_skip.strip() and existing_skip.strip() != meta_skip_reason:
                secondary.append(existing_skip.strip())
            diagnostics["reid_skip_reason"] = meta_skip_reason
            if secondary:
                diagnostics["reid_secondary_skip_reasons"] = secondary
            else:
                diagnostics.pop("reid_secondary_skip_reasons", None)

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "schema_version": 2,
            "num_face_tracks": len(face_tracks),
            "num_body_tracks": len(body_tracks),
            "num_fused_identities": len(identities),
            "diagnostics": getattr(fusion, "last_diagnostics", {}) or {},
            "identities": {k: v.to_dict() for k, v in identities.items()},
        }

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Saved fusion results to: {output_path}")

    return identities
