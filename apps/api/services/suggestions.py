"""Run-scoped cast suggestions for Faces Review (S3-only storage)."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from py_screenalytics import run_layout

from apps.api.config.suggestions import SUGGESTION_THRESHOLDS, get_confidence_label
from apps.api.services.assignments import load_assignment_state
from apps.api.services.run_state import run_state_service
from apps.api.services.storage import StorageService, ARTIFACT_ROOT

LOGGER = logging.getLogger(__name__)
_STORAGE = StorageService()
_EP_ID_REGEX = re.compile(r"^(?P<show>.+)-s(?P<season>\d{2})e(?P<episode>\d{2})$", re.IGNORECASE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_track_id(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.startswith("track_"):
        raw = raw.replace("track_", "")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _l2_normalize(vec: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm <= 0:
        return None
    return vec / (norm + 1e-12)


def _read_json_from_s3(key: str | None) -> Any:
    if not key:
        return None
    payload = _STORAGE.download_bytes(key)
    if not payload:
        return None
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _iter_jsonl_from_s3(key: str | None) -> Iterable[Dict[str, Any]]:
    if not key:
        return []
    payload = _STORAGE.download_bytes(key)
    if not payload:
        return []

    def _generator() -> Iterable[Dict[str, Any]]:
        for line in payload.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row

    return _generator()


def _suggestions_key(ep_id: str, run_id: str) -> str:
    return run_layout.run_artifact_s3_key(ep_id, run_id, "suggestions.json")


def _cluster_tracks_from_identities(payload: Dict[str, Any]) -> Dict[str, List[int]]:
    identities = payload.get("identities")
    if not isinstance(identities, list):
        return {}
    mapping: Dict[str, List[int]] = {}
    for entry in identities:
        if not isinstance(entry, dict):
            continue
        cluster_id = entry.get("identity_id") or entry.get("cluster_id") or entry.get("id")
        if not cluster_id:
            continue
        track_ids = []
        for raw in entry.get("track_ids", []) or []:
            track_id = _parse_track_id(raw)
            if track_id is not None:
                track_ids.append(track_id)
        mapping[str(cluster_id)] = track_ids
    return mapping


def _parse_show_id(ep_id: str) -> str | None:
    match = _EP_ID_REGEX.match(ep_id or "")
    if not match:
        return None
    return match.group("show")


def _track_embeddings_from_faces(
    faces_rows: Iterable[Dict[str, Any]],
    excluded_faces: set[str],
) -> Tuple[Dict[int, np.ndarray], set[int]]:
    by_track: Dict[int, List[np.ndarray]] = {}
    tracks_with_exclusions: set[int] = set()
    for row in faces_rows:
        if not isinstance(row, dict):
            continue
        face_id = row.get("face_id")
        track_id = _parse_track_id(row.get("track_id") or row.get("track"))
        if face_id and face_id in excluded_faces:
            if track_id is not None:
                tracks_with_exclusions.add(track_id)
            continue
        embedding = row.get("embedding")
        if embedding is None:
            continue
        try:
            vec = np.array(embedding, dtype=np.float32)
        except Exception:
            continue
        if track_id is None:
            continue
        by_track.setdefault(track_id, []).append(vec)

    track_embeddings: Dict[int, np.ndarray] = {}
    for track_id, vectors in by_track.items():
        if not vectors:
            continue
        centroid = np.mean(np.stack(vectors), axis=0)
        normalized = _l2_normalize(centroid)
        if normalized is not None:
            track_embeddings[track_id] = normalized
    return track_embeddings, tracks_with_exclusions


def _track_embeddings_from_reps(
    reps_rows: Iterable[Dict[str, Any]],
    *,
    existing: Dict[int, np.ndarray],
    skip_tracks: set[int],
) -> Dict[int, np.ndarray]:
    track_embeddings = dict(existing)
    for rep in reps_rows:
        if not isinstance(rep, dict):
            continue
        track_id = _parse_track_id(rep.get("track_id") or rep.get("track") or rep.get("track_int"))
        if track_id is None or track_id in track_embeddings or track_id in skip_tracks:
            continue
        embed = rep.get("embed") or rep.get("embedding")
        if embed is None:
            continue
        try:
            vec = np.array(embed, dtype=np.float32)
        except Exception:
            continue
        normalized = _l2_normalize(vec)
        if normalized is not None:
            track_embeddings[track_id] = normalized
    return track_embeddings


def _cluster_centroids_from_payload(payload: Any) -> Dict[str, np.ndarray]:
    if isinstance(payload, dict) and isinstance(payload.get("centroids"), (dict, list)):
        centroids = payload.get("centroids")
    else:
        centroids = payload
    result: Dict[str, np.ndarray] = {}
    if isinstance(centroids, dict):
        entries = [(cluster_id, entry) for cluster_id, entry in centroids.items()]
    elif isinstance(centroids, list):
        entries = [(entry.get("cluster_id"), entry) for entry in centroids if isinstance(entry, dict)]
    else:
        entries = []

    for cluster_id, entry in entries:
        if not cluster_id or not isinstance(entry, dict):
            continue
        centroid = entry.get("centroid")
        if centroid is None:
            continue
        try:
            vec = np.array(centroid, dtype=np.float32)
        except Exception:
            continue
        normalized = _l2_normalize(vec)
        if normalized is not None:
            result[str(cluster_id)] = normalized
    return result


def _cluster_embeddings_from_tracks(
    cluster_tracks: Dict[str, List[int]],
    track_embeddings: Dict[int, np.ndarray],
    fallback_centroids: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    cluster_embeddings: Dict[str, np.ndarray] = {}
    for cluster_id, track_ids in cluster_tracks.items():
        vectors = [track_embeddings[tid] for tid in track_ids if tid in track_embeddings]
        if vectors:
            centroid = np.mean(np.stack(vectors), axis=0)
            normalized = _l2_normalize(centroid)
            if normalized is not None:
                cluster_embeddings[str(cluster_id)] = normalized
                continue
        fallback = fallback_centroids.get(str(cluster_id))
        if fallback is not None:
            cluster_embeddings[str(cluster_id)] = fallback
    return cluster_embeddings


def _facebank_prefixes(show_id: str) -> List[str]:
    show = (show_id or "").strip()
    if not show:
        return []
    variants = {show, show.upper(), show.lower()}
    return [f"{ARTIFACT_ROOT}/facebank/{variant}/" for variant in variants]


def _load_facebank_seeds(show_id: str) -> List[Dict[str, Any]]:
    seeds: List[Dict[str, Any]] = []
    for prefix in _facebank_prefixes(show_id):
        keys = _STORAGE.list_objects(prefix, suffix="facebank.json", max_items=5000)
        for key in keys:
            payload = _read_json_from_s3(key)
            if not isinstance(payload, dict):
                continue
            rel = key[len(prefix):]
            cast_id = rel.split("/", 1)[0] if rel else None
            if not cast_id:
                continue
            for seed in payload.get("seeds", []) or []:
                if not isinstance(seed, dict):
                    continue
                embedding = seed.get("embedding")
                if embedding is None:
                    continue
                seeds.append({"cast_id": cast_id, "embedding": embedding})
    return seeds


def _cast_embeddings_from_facebank(show_id: str) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    seeds = _load_facebank_seeds(show_id)
    by_cast: Dict[str, List[np.ndarray]] = {}
    for seed in seeds:
        cast_id = seed.get("cast_id")
        embedding = seed.get("embedding")
        if not cast_id or embedding is None:
            continue
        try:
            vec = np.array(embedding, dtype=np.float32)
        except Exception:
            continue
        normalized = _l2_normalize(vec)
        if normalized is None:
            continue
        by_cast.setdefault(str(cast_id), []).append(normalized)

    cast_embeddings: Dict[str, np.ndarray] = {}
    cast_sources: Dict[str, str] = {}
    for cast_id, vectors in by_cast.items():
        if not vectors:
            continue
        centroid = np.mean(np.stack(vectors), axis=0)
        normalized = _l2_normalize(centroid)
        if normalized is not None:
            cast_embeddings[cast_id] = normalized
            cast_sources[cast_id] = "facebank"
    return cast_embeddings, cast_sources


def _cast_embeddings_from_assignments(
    cluster_embeddings: Dict[str, np.ndarray],
    cluster_assignments: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    by_cast: Dict[str, List[np.ndarray]] = {}
    for cluster_id, entry in cluster_assignments.items():
        if not isinstance(entry, dict):
            continue
        cast_id = entry.get("cast_id")
        if not cast_id:
            continue
        embedding = cluster_embeddings.get(str(cluster_id))
        if embedding is None:
            continue
        by_cast.setdefault(str(cast_id), []).append(embedding)
    cast_embeddings: Dict[str, np.ndarray] = {}
    cast_sources: Dict[str, str] = {}
    for cast_id, vectors in by_cast.items():
        if not vectors:
            continue
        centroid = np.mean(np.stack(vectors), axis=0)
        normalized = _l2_normalize(centroid)
        if normalized is not None:
            cast_embeddings[cast_id] = normalized
            cast_sources[cast_id] = "assigned"
    return cast_embeddings, cast_sources


def compute_cast_suggestions(
    cluster_embeddings: Dict[str, np.ndarray],
    cast_embeddings: Dict[str, np.ndarray],
    *,
    top_k: int,
    min_similarity: float,
    cast_sources: Dict[str, str] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    suggestions: Dict[str, List[Dict[str, Any]]] = {}
    cast_sources = cast_sources or {}
    for cluster_id, cluster_vec in cluster_embeddings.items():
        scored: List[Tuple[str, float]] = []
        for cast_id, cast_vec in cast_embeddings.items():
            score = float(np.dot(cluster_vec, cast_vec))
            scored.append((cast_id, score))
        scored.sort(key=lambda item: (-item[1], str(item[0])))
        cluster_suggestions: List[Dict[str, Any]] = []
        for idx, (cast_id, score) in enumerate(scored):
            if score < min_similarity:
                continue
            cluster_suggestions.append(
                {
                    "cast_id": cast_id,
                    "score": float(score),
                    "similarity": float(score),
                    "confidence": get_confidence_label(score),
                    "rank": idx + 1,
                    "total_suggestions": min(top_k, len(scored)),
                    "method": cast_sources.get(cast_id) or "unknown",
                    "suggestion_id": f"{cluster_id}:{cast_id}",
                }
            )
            if len(cluster_suggestions) >= top_k:
                break
        suggestions[str(cluster_id)] = cluster_suggestions
    return suggestions


def compute_suggestions(
    ep_id: str,
    run_id: str,
    *,
    top_k: int = 3,
    min_similarity: float | None = None,
) -> Dict[str, Any]:
    run_id_norm = run_layout.normalize_run_id(run_id)
    if min_similarity is None:
        min_similarity = SUGGESTION_THRESHOLDS["cast_low"]

    run_state_bundle = run_state_service.get_state(ep_id=ep_id, run_id=run_id_norm)
    run_state_payload = run_state_bundle.get("run_state") if isinstance(run_state_bundle, dict) else None
    artifacts = run_state_payload.get("artifacts") if isinstance(run_state_payload, dict) else {}
    identities_key = (
        artifacts.get("identities", {}).get("s3_key")
        or run_layout.run_artifact_s3_key(ep_id, run_id_norm, "identities.json")
    )
    tracks_key = (
        artifacts.get("tracks", {}).get("s3_key")
        or run_layout.run_artifact_s3_key(ep_id, run_id_norm, "tracks.jsonl")
    )
    faces_key = (
        artifacts.get("faces", {}).get("manifest_key")
        or artifacts.get("faces", {}).get("s3_key")
        or run_layout.run_artifact_s3_key(ep_id, run_id_norm, "faces.jsonl")
    )
    reps_key = (
        artifacts.get("track_reps", {}).get("s3_key")
        or run_layout.run_artifact_s3_key(ep_id, run_id_norm, "track_reps.jsonl")
    )
    centroids_key = run_layout.run_artifact_s3_key(ep_id, run_id_norm, "cluster_centroids.json")

    identities_payload = _read_json_from_s3(identities_key)
    if not isinstance(identities_payload, dict):
        return {
            "status": "missing_identities",
            "message": "identities.json missing or unreadable",
            "ep_id": ep_id,
            "run_id": run_id_norm,
            "generated_at": _now_iso(),
            "suggestions": {},
            "triage": {},
        }

    cluster_tracks = _cluster_tracks_from_identities(identities_payload)
    if not cluster_tracks:
        return {
            "status": "empty",
            "message": "No clusters available for suggestions",
            "ep_id": ep_id,
            "run_id": run_id_norm,
            "generated_at": _now_iso(),
            "suggestions": {},
            "triage": {},
        }

    assignment_state = load_assignment_state(ep_id, run_id_norm, include_inferred=True)
    cluster_assignments = assignment_state.get("cluster_assignments", {})
    face_exclusions = assignment_state.get("face_exclusions", {})
    excluded_faces = {
        face_id
        for face_id, entry in face_exclusions.items()
        if isinstance(entry, dict) and entry.get("excluded")
    }
    excluded_tracks = {
        _parse_track_id(entry.get("track_id"))
        for entry in face_exclusions.values()
        if isinstance(entry, dict) and entry.get("excluded") and entry.get("track_id") is not None
    }
    excluded_tracks.discard(None)

    track_embeddings, tracks_with_exclusions = _track_embeddings_from_faces(
        _iter_jsonl_from_s3(faces_key),
        excluded_faces,
    )
    skip_tracks = set(excluded_tracks) | set(tracks_with_exclusions)
    track_embeddings = _track_embeddings_from_reps(
        _iter_jsonl_from_s3(reps_key),
        existing=track_embeddings,
        skip_tracks=skip_tracks,
    )
    cluster_centroids = _cluster_centroids_from_payload(_read_json_from_s3(centroids_key))
    cluster_embeddings = _cluster_embeddings_from_tracks(cluster_tracks, track_embeddings, cluster_centroids)

    show_id = (identities_payload.get("show_id") or _parse_show_id(ep_id) or "").strip()
    cast_embeddings, cast_sources = _cast_embeddings_from_facebank(show_id)
    if not cast_embeddings:
        cast_embeddings, cast_sources = _cast_embeddings_from_assignments(cluster_embeddings, cluster_assignments)

    if not cast_embeddings:
        return {
            "status": "missing_cast_embeddings",
            "message": "No facebank seeds or assigned clusters available",
            "ep_id": ep_id,
            "run_id": run_id_norm,
            "generated_at": _now_iso(),
            "suggestions": {},
            "triage": {},
        }

    suggestions_by_cluster = compute_cast_suggestions(
        cluster_embeddings,
        cast_embeddings,
        top_k=top_k,
        min_similarity=min_similarity,
        cast_sources=cast_sources,
    )

    assigned_clusters = {
        str(cluster_id)
        for cluster_id, entry in cluster_assignments.items()
        if isinstance(entry, dict) and entry.get("cast_id")
    }
    suggestions_by_cluster = {
        cid: entries
        for cid, entries in suggestions_by_cluster.items()
        if cid not in assigned_clusters
    }

    track_frame_counts: Dict[int, int] = {}
    for row in _iter_jsonl_from_s3(tracks_key):
        track_id = _parse_track_id(row.get("track_id") or row.get("track"))
        if track_id is None:
            continue
        value = row.get("frame_count") or row.get("frames") or row.get("faces_count") or 0
        try:
            track_frame_counts[track_id] = int(value)
        except (TypeError, ValueError):
            track_frame_counts[track_id] = 0

    triage: Dict[str, Dict[str, Any]] = {}
    for cluster_id, track_ids in cluster_tracks.items():
        impact_frames = sum(track_frame_counts.get(tid, 0) for tid in track_ids)
        suggestions = suggestions_by_cluster.get(str(cluster_id), [])
        top_score = suggestions[0].get("score", 0.0) if suggestions else 0.0
        second_score = suggestions[1].get("score", 0.0) if len(suggestions) > 1 else 0.0
        margin = max(float(top_score) - float(second_score), 0.0)
        uncertainty = 1.0 - float(top_score) if top_score else 1.0
        triage[str(cluster_id)] = {
            "impact_frames": impact_frames,
            "top_score": float(top_score),
            "margin": float(margin),
            "triage_score": float(impact_frames) * float(uncertainty),
        }

    return {
        "status": "ready",
        "ep_id": ep_id,
        "run_id": run_id_norm,
        "generated_at": _now_iso(),
        "model_version": "faces_review_suggestions_v1",
        "config": {
            "top_k": top_k,
            "min_similarity": min_similarity,
        },
        "suggestions": suggestions_by_cluster,
        "triage": triage,
    }


def load_suggestions(ep_id: str, run_id: str) -> Dict[str, Any] | None:
    run_id_norm = run_layout.normalize_run_id(run_id)
    payload = _read_json_from_s3(_suggestions_key(ep_id, run_id_norm))
    return payload if isinstance(payload, dict) else None


def write_suggestions(ep_id: str, run_id: str, payload: Dict[str, Any]) -> bool:
    run_id_norm = run_layout.normalize_run_id(run_id)
    data = json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8")
    return _STORAGE.upload_bytes(data, _suggestions_key(ep_id, run_id_norm), content_type="application/json")


def get_or_compute_suggestions(
    ep_id: str,
    run_id: str,
    *,
    top_k: int = 3,
    min_similarity: float | None = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    if not refresh:
        existing = load_suggestions(ep_id, run_id)
        if existing is not None:
            return existing

    payload = compute_suggestions(ep_id, run_id, top_k=top_k, min_similarity=min_similarity)
    if payload.get("status") == "ready":
        if not write_suggestions(ep_id, run_id, payload):
            payload = dict(payload)
            payload["status"] = "write_failed"
            payload["message"] = "Failed to persist suggestions to S3"
    return payload


def dismiss_suggestions(
    ep_id: str,
    run_id: str,
    suggestion_ids: List[str],
    *,
    restore: bool = False,
) -> Dict[str, Any]:
    payload = load_suggestions(ep_id, run_id)
    if not payload:
        payload = get_or_compute_suggestions(ep_id, run_id)
    dismissed = payload.get("dismissed")
    dismissed_set = set(dismissed) if isinstance(dismissed, list) else set()
    for suggestion_id in suggestion_ids:
        if restore:
            dismissed_set.discard(suggestion_id)
        else:
            dismissed_set.add(suggestion_id)
    payload["dismissed"] = sorted(dismissed_set)
    if payload.get("status") == "ready":
        write_suggestions(ep_id, run_id, payload)
    return payload


__all__ = [
    "compute_cast_suggestions",
    "compute_suggestions",
    "dismiss_suggestions",
    "get_or_compute_suggestions",
    "load_suggestions",
    "write_suggestions",
]
