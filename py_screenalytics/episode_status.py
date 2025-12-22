from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from py_screenalytics import run_layout

STAGE_PLAN: tuple[str, ...] = (
    "detect",
    "faces",
    "cluster",
    "body_tracking",
    "track_fusion",
    "screentime",
    "pdf",
)

_STAGE_ALIASES: dict[str, str] = {
    "detect": "detect",
    "detect_track": "detect",
    "detect/track": "detect",
    "faces": "faces",
    "faces_embed": "faces",
    "faces harvest": "faces",
    "cluster": "cluster",
    "body_tracking": "body_tracking",
    "body tracking": "body_tracking",
    "body_tracking_fusion": "track_fusion",
    "track_fusion": "track_fusion",
    "track fusion": "track_fusion",
    "screen_time": "screentime",
    "screen_time_analyze": "screentime",
    "screentime": "screentime",
    "pdf": "pdf",
    "pdf_export": "pdf",
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_stage_key(raw: str | None) -> str | None:
    if not raw:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    value = value.replace("_", " ").strip()
    value = " ".join(value.split())
    return _STAGE_ALIASES.get(value, _STAGE_ALIASES.get(value.replace(" ", "_")))


def episode_status_path(ep_id: str, run_id: str) -> Path:
    run_root = run_layout.run_root(ep_id, run_layout.normalize_run_id(run_id))
    return run_root / "episode_status.json"


def read_episode_status(ep_id: str, run_id: str) -> dict[str, Any] | None:
    path = episode_status_path(ep_id, run_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_episode_status(ep_id: str, run_id: str, payload: Mapping[str, Any]) -> Path:
    path = episode_status_path(ep_id, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _duration_s(started_at: str | None, ended_at: str | None) -> float | None:
    start = _parse_iso(started_at)
    end = _parse_iso(ended_at)
    if not start or not end:
        return None
    delta = (end - start).total_seconds()
    return delta if delta >= 0 else None


def _artifact_entry(path: Path, label: str) -> dict[str, Any]:
    scope = "run" if "runs" in path.parts else "legacy"
    return {"label": label, "path": str(path), "scope": scope, "exists": path.exists()}


def stage_artifacts(ep_id: str, run_id: str, stage_key: str) -> list[dict[str, Any]]:
    run_root = run_layout.run_root(ep_id, run_layout.normalize_run_id(run_id))
    artifacts: list[dict[str, Any]] = []
    if stage_key == "detect":
        artifacts = [
            _artifact_entry(run_root / "detections.jsonl", "detections.jsonl"),
            _artifact_entry(run_root / "tracks.jsonl", "tracks.jsonl"),
            _artifact_entry(run_root / "track_metrics.json", "track_metrics.json"),
            _artifact_entry(run_root / "detect_track.json", "detect_track.json"),
        ]
    elif stage_key == "faces":
        artifacts = [
            _artifact_entry(run_root / "faces.jsonl", "faces.jsonl"),
            _artifact_entry(run_root / "faces_embed.json", "faces_embed.json"),
            _artifact_entry(run_root / "face_alignment" / "aligned_faces.jsonl", "face_alignment/aligned_faces.jsonl"),
        ]
    elif stage_key == "cluster":
        artifacts = [
            _artifact_entry(run_root / "identities.json", "identities.json"),
            _artifact_entry(run_root / "cluster_centroids.json", "cluster_centroids.json"),
            _artifact_entry(run_root / "cluster.json", "cluster.json"),
        ]
    elif stage_key == "body_tracking":
        artifacts = [
            _artifact_entry(run_root / "body_tracking" / "body_detections.jsonl", "body_tracking/body_detections.jsonl"),
            _artifact_entry(run_root / "body_tracking" / "body_tracks.jsonl", "body_tracking/body_tracks.jsonl"),
            _artifact_entry(run_root / "body_tracking" / "body_metrics.json", "body_tracking/body_metrics.json"),
            _artifact_entry(run_root / "body_tracking.json", "body_tracking.json"),
        ]
    elif stage_key == "track_fusion":
        artifacts = [
            _artifact_entry(run_root / "body_tracking" / "track_fusion.json", "body_tracking/track_fusion.json"),
            _artifact_entry(
                run_root / "body_tracking" / "screentime_comparison.json",
                "body_tracking/screentime_comparison.json",
            ),
            _artifact_entry(run_root / "body_tracking_fusion.json", "body_tracking_fusion.json"),
        ]
    elif stage_key == "screentime":
        artifacts = [
            _artifact_entry(run_root / "analytics" / "screentime.json", "analytics/screentime.json"),
            _artifact_entry(run_root / "analytics" / "screentime.csv", "analytics/screentime.csv"),
        ]
    elif stage_key == "pdf":
        artifacts = [
            _artifact_entry(run_root / "exports" / "export_index.json", "exports/export_index.json"),
        ]
    return artifacts


def collect_git_state() -> dict[str, Any]:
    def _run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    sha = _run_git(["rev-parse", "--short", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    dirty = None
    status = _run_git(["status", "--porcelain"])
    if status is not None:
        dirty = bool(status.strip())
    payload: dict[str, Any] = {}
    if sha:
        payload["git_sha"] = sha
    if branch:
        payload["git_branch"] = branch
    if dirty is not None:
        payload["git_dirty"] = dirty
    return payload


def _merge_stage(existing: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in update.items():
        if value is None and key not in {"status"}:
            continue
        merged[key] = value
    return merged


def update_episode_status(
    ep_id: str,
    run_id: str,
    *,
    stage_key: str | None = None,
    stage_update: Mapping[str, Any] | None = None,
    stage_plan: tuple[str, ...] | None = None,
    git_info: Mapping[str, Any] | None = None,
    env: Mapping[str, Any] | None = None,
    storage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_id_norm = run_layout.normalize_run_id(run_id)
    payload = read_episode_status(ep_id, run_id_norm) or {
        "episode_id": ep_id,
        "run_id": run_id_norm,
        "stage_plan": list(stage_plan or STAGE_PLAN),
        "stages": {},
    }
    payload.setdefault("episode_id", ep_id)
    payload.setdefault("run_id", run_id_norm)
    payload.setdefault("stage_plan", list(stage_plan or STAGE_PLAN))
    payload.setdefault("stages", {})
    stages = payload.get("stages")
    if isinstance(stages, dict) and not stages:
        for name in payload["stage_plan"]:
            stages[name] = {"status": "missing"}

    if stage_key and stage_update and isinstance(payload.get("stages"), dict):
        current = payload["stages"].get(stage_key, {})
        if not isinstance(current, dict):
            current = {}
        payload["stages"][stage_key] = _merge_stage(current, stage_update)

    if git_info:
        for key, value in git_info.items():
            if value is not None:
                payload[key] = value

    if env:
        payload_env = payload.get("env")
        if not isinstance(payload_env, dict):
            payload_env = {}
        for key, value in env.items():
            if value is not None:
                payload_env[key] = value
        payload["env"] = payload_env

    if storage:
        payload_storage = payload.get("storage")
        if not isinstance(payload_storage, dict):
            payload_storage = {}
        for key, value in storage.items():
            if value is not None:
                payload_storage[key] = value
        payload["storage"] = payload_storage

    payload["generated_at"] = _utcnow_iso()
    write_episode_status(ep_id, run_id_norm, payload)
    return payload


def stage_update_from_marker(
    *,
    ep_id: str,
    run_id: str,
    phase: str,
    marker_payload: Mapping[str, Any],
    extra_metrics: Mapping[str, Any] | None = None,
    extra_artifacts: list[dict[str, Any]] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    stage_key = normalize_stage_key(phase)
    if not stage_key:
        return None, {}
    status = str(marker_payload.get("status") or "unknown").strip().lower()
    started_at = marker_payload.get("started_at")
    ended_at = marker_payload.get("finished_at") or marker_payload.get("ended_at")
    duration_s = _duration_s(started_at, ended_at)
    error_reason = None
    if status in {"error", "failed"}:
        error_reason = marker_payload.get("error") or marker_payload.get("error_reason")
    metrics: dict[str, Any] = {}
    if stage_key == "detect":
        metrics = {
            "detections": marker_payload.get("detections"),
            "tracks": marker_payload.get("tracks"),
            "rtf": marker_payload.get("rtf"),
            "effective_fps_processing": marker_payload.get("effective_fps_processing"),
            "scene_cut_count": marker_payload.get("scene_cut_count"),
            "forced_scene_warmup_ratio": marker_payload.get("forced_scene_warmup_ratio"),
            "frames_total": marker_payload.get("frames_total"),
        }
    elif stage_key == "faces":
        metrics = {
            "faces": marker_payload.get("faces"),
            "embedding_backend_actual": marker_payload.get("embedding_backend_actual"),
            "embedding_model_name": marker_payload.get("embedding_model_name"),
            "embedding_backend_fallback_reason": marker_payload.get("embedding_backend_fallback_reason"),
        }
    elif stage_key == "cluster":
        metrics = {
            "identities": marker_payload.get("identities"),
            "faces": marker_payload.get("faces"),
            "singleton_fraction_before": marker_payload.get("singleton_fraction_before"),
            "singleton_fraction_after": marker_payload.get("singleton_fraction_after"),
            "cluster_thresh": marker_payload.get("cluster_thresh"),
            "min_cluster_size": marker_payload.get("min_cluster_size"),
            "min_identity_sim": marker_payload.get("min_identity_sim"),
        }
    elif stage_key == "body_tracking":
        metrics = {
            "body_reid": marker_payload.get("body_reid"),
            "tracker_backend_actual": marker_payload.get("tracker_backend_actual"),
            "tracker_fallback_reason": marker_payload.get("tracker_fallback_reason"),
        }
    elif stage_key == "track_fusion":
        metrics = {}

    if extra_metrics:
        metrics.update({k: v for k, v in extra_metrics.items() if v is not None})

    artifacts = extra_artifacts if extra_artifacts is not None else stage_artifacts(ep_id, run_id, stage_key)
    return stage_key, {
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_s": duration_s,
        "error_reason": error_reason,
        "artifacts": artifacts,
        "metrics": metrics,
    }
