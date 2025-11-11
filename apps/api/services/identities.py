from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from py_screenalytics.artifacts import get_path

from apps.api.services.storage import StorageService, episode_context_from_id

STORAGE = StorageService()


def _manifests_dir(ep_id: str) -> Path:
    return get_path(ep_id, "detections").parent


def _faces_path(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "faces.jsonl"


def _identities_path(ep_id: str) -> Path:
    return _manifests_dir(ep_id) / "identities.json"


def _tracks_path(ep_id: str) -> Path:
    return get_path(ep_id, "tracks")


def _read_json_lines(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _write_json_lines(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def load_faces(ep_id: str) -> List[Dict[str, Any]]:
    return _read_json_lines(_faces_path(ep_id))


def write_faces(ep_id: str, rows: List[Dict[str, Any]]) -> Path:
    return _write_json_lines(_faces_path(ep_id), rows)


def load_tracks(ep_id: str) -> List[Dict[str, Any]]:
    return _read_json_lines(_tracks_path(ep_id))


def write_tracks(ep_id: str, rows: List[Dict[str, Any]]) -> Path:
    return _write_json_lines(_tracks_path(ep_id), rows)


def load_identities(ep_id: str) -> Dict[str, Any]:
    path = _identities_path(ep_id)
    if not path.exists():
        return {"ep_id": ep_id, "identities": [], "stats": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ep_id": ep_id, "identities": [], "stats": {}}


def write_identities(ep_id: str, payload: Dict[str, Any]) -> Path:
    path = _identities_path(ep_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def update_identity_stats(ep_id: str, payload: Dict[str, Any]) -> None:
    faces_count = len(load_faces(ep_id))
    payload.setdefault("stats", {})
    payload["stats"]["faces"] = faces_count
    payload["stats"]["clusters"] = len(payload.get("identities", []))


def sync_manifests(ep_id: str, *paths: Path) -> None:
    try:
        ctx = episode_context_from_id(ep_id)
    except ValueError:
        return
    for path in paths:
        if path and path.exists():
            try:
                STORAGE.put_artifact(ctx, "manifests", path, path.name)
            except Exception:
                continue


def rename_identity(ep_id: str, identity_id: str, label: str | None) -> Dict[str, Any]:
    payload = load_identities(ep_id)
    identity = next((item for item in payload.get("identities", []) if item.get("identity_id") == identity_id), None)
    if not identity:
        raise ValueError("identity_not_found")
    normalized = (label or "").strip()
    identity["label"] = normalized or None
    update_identity_stats(ep_id, payload)
    identities_path = write_identities(ep_id, payload)
    sync_manifests(ep_id, identities_path)
    return identity


def merge_identities(ep_id: str, source_id: str, target_id: str) -> Dict[str, Any]:
    payload = load_identities(ep_id)
    identities = payload.get("identities", [])
    source = next((item for item in identities if item.get("identity_id") == source_id), None)
    target = next((item for item in identities if item.get("identity_id") == target_id), None)
    if not source or not target:
        raise ValueError("identity_not_found")
    merged = set(target.get("track_ids", []) or [])
    for tid in source.get("track_ids", []) or []:
        merged.add(int(tid))
    target["track_ids"] = sorted({int(val) for val in merged})
    payload["identities"] = [item for item in identities if item.get("identity_id") != source_id]
    update_identity_stats(ep_id, payload)
    identities_path = write_identities(ep_id, payload)
    sync_manifests(ep_id, identities_path)
    return target


def move_track(ep_id: str, track_id: int, target_identity_id: str | None) -> Dict[str, Any]:
    payload = load_identities(ep_id)
    identities = payload.get("identities", [])
    source_identity = None
    target_identity = None
    for identity in identities:
        track_ids = identity.get("track_ids", []) or []
        if track_id in track_ids:
            source_identity = identity
        if target_identity_id and identity.get("identity_id") == target_identity_id:
            target_identity = identity
    if target_identity_id and target_identity is None:
        raise ValueError("target_not_found")
    if source_identity and track_id in source_identity.get("track_ids", []):
        source_identity["track_ids"] = [tid for tid in source_identity["track_ids"] if tid != track_id]
    if target_identity is not None:
        target_identity.setdefault("track_ids", [])
        if track_id not in target_identity["track_ids"]:
            target_identity["track_ids"].append(track_id)
            target_identity["track_ids"] = sorted(target_identity["track_ids"])
    update_identity_stats(ep_id, payload)
    identities_path = write_identities(ep_id, payload)
    sync_manifests(ep_id, identities_path)
    return {"identity_id": target_identity_id, "track_ids": target_identity["track_ids"] if target_identity else []}


def drop_track(ep_id: str, track_id: int) -> Dict[str, Any]:
    tracks = load_tracks(ep_id)
    kept_tracks = [row for row in tracks if int(row.get("track_id", -1)) != track_id]
    if len(kept_tracks) == len(tracks):
        raise ValueError("track_not_found")
    tracks_path = write_tracks(ep_id, kept_tracks)
    identities = load_identities(ep_id)
    for identity in identities.get("identities", []):
        identity["track_ids"] = [tid for tid in identity.get("track_ids", []) if tid != track_id]
    update_identity_stats(ep_id, identities)
    identities_path = write_identities(ep_id, identities)
    sync_manifests(ep_id, tracks_path, identities_path)
    return {"track_id": track_id, "remaining_tracks": len(kept_tracks)}


def drop_frame(ep_id: str, track_id: int, frame_idx: int, delete_assets: bool = False) -> Dict[str, Any]:
    faces = load_faces(ep_id)
    removed = [
        row
        for row in faces
        if int(row.get("track_id", -1)) == track_id and int(row.get("frame_idx", -1)) == frame_idx
    ]
    if not removed:
        raise ValueError("frame_not_found")
    faces = [row for row in faces if row not in removed]
    faces_path = write_faces(ep_id, faces)
    thumbs_root = get_path(ep_id, "frames_root") / "thumbs"
    crops_root = get_path(ep_id, "frames_root")
    if delete_assets:
        for row in removed:
            thumb_rel = row.get("thumb_rel_path")
            if isinstance(thumb_rel, str):
                thumb_abs = thumbs_root / thumb_rel
                try:
                    thumb_abs.unlink()
                except FileNotFoundError:
                    pass
            crop_rel = row.get("crop_rel_path")
            if isinstance(crop_rel, str):
                crop_abs = crops_root / crop_rel
                try:
                    crop_abs.unlink()
                except FileNotFoundError:
                    pass
    identities = load_identities(ep_id)
    update_identity_stats(ep_id, identities)
    identities_path = write_identities(ep_id, identities)
    sync_manifests(ep_id, faces_path, identities_path)
    return {"track_id": track_id, "frame_idx": frame_idx, "removed": len(removed)}
