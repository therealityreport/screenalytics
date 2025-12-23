from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from py_screenalytics import run_layout

STAGE_PLAN: tuple[str, ...] = (
    "detect",
    "faces",
    "cluster",
    "body_tracking",
    "track_fusion",
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


class Stage(str, Enum):
    DETECT = "detect"
    FACES = "faces"
    CLUSTER = "cluster"
    BODY_TRACKING = "body_tracking"
    TRACK_FUSION = "track_fusion"
    SCREENTIME = "screentime"
    PDF = "pdf"

    @classmethod
    def from_key(cls, value: str | None) -> "Stage | None":
        normalized = normalize_stage_key(value)
        if not normalized:
            return None
        for stage in cls:
            if stage.value == normalized:
                return stage
        return None


class StageStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"

    @classmethod
    def from_value(cls, value: str | None) -> "StageStatus":
        if not value:
            return cls.NOT_STARTED
        normalized = str(value).strip().lower()
        if normalized in {"missing", "unknown"}:
            return cls.NOT_STARTED
        if normalized in {"error", "failed", "failure"}:
            return cls.FAILED
        if normalized in {"running", "in_progress", "in-progress"}:
            return cls.RUNNING
        if normalized in {"blocked"}:
            return cls.BLOCKED
        if normalized in {"success", "complete", "completed"}:
            return cls.SUCCESS
        return cls.NOT_STARTED


@dataclass(frozen=True)
class BlockedReason:
    code: str
    message: str
    details: dict[str, Any] | None = None


@dataclass
class StageState:
    episode_id: str
    run_id: str
    stage: Stage
    status: StageStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_s: float | None = None
    blocked_reason: BlockedReason | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)
    artifact_digests: dict[str, str] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)
    thresholds: dict[str, object] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    derived: bool = False


@dataclass
class EpisodeStatus:
    episode_id: str
    run_id: str
    stages: dict[Stage, StageState] = field(default_factory=dict)
    stage_plan: tuple[Stage, ...] = field(default_factory=tuple)
    generated_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        stage_plan = [stage.value for stage in self.stage_plan] if self.stage_plan else list(STAGE_PLAN)
        return {
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "stage_plan": stage_plan,
            "stages": {stage.value: _stage_state_to_payload(state) for stage, state in self.stages.items()},
            "generated_at": _serialize_datetime(self.generated_at) or _utcnow_iso(),
        }


_STAGE_MARKER_FILES: dict[Stage, str] = {
    Stage.DETECT: "detect_track",
    Stage.FACES: "faces_embed",
    Stage.CLUSTER: "cluster",
    Stage.BODY_TRACKING: "body_tracking",
    Stage.TRACK_FUSION: "body_tracking_fusion",
    Stage.PDF: "pdf_export",
}


def _coerce_stage(stage: Stage | str) -> Stage:
    if isinstance(stage, Stage):
        return stage
    resolved = Stage.from_key(stage)
    if not resolved:
        raise ValueError(f"Unknown stage key: {stage!r}")
    return resolved


def episode_status_path(ep_id: str, run_id: str) -> Path:
    run_root = run_layout.run_root(ep_id, run_layout.normalize_run_id(run_id))
    return run_root / "episode_status.json"


def _read_status_payload(ep_id: str, run_id: str) -> dict[str, Any] | None:
    path = episode_status_path(ep_id, run_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_episode_status(ep_id: str, run_id: str) -> EpisodeStatus:
    run_id_norm = run_layout.normalize_run_id(run_id)
    payload = _read_status_payload(ep_id, run_id_norm)
    if payload is None:
        return derive_status_from_artifacts(ep_id, run_id_norm)
    return _episode_status_from_payload(ep_id, run_id_norm, payload)


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


def _serialize_datetime(value: datetime | None) -> str | None:
    if not value:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _blocked_reason_from_payload(payload: Mapping[str, Any] | None) -> BlockedReason | None:
    if not isinstance(payload, Mapping):
        return None
    code = payload.get("code")
    message = payload.get("message")
    if not code or not message:
        return None
    details = payload.get("details")
    return BlockedReason(
        code=str(code),
        message=str(message),
        details=(details if isinstance(details, dict) else None),
    )


def _blocked_reason_to_payload(reason: BlockedReason | None) -> dict[str, Any] | None:
    if not reason:
        return None
    payload: dict[str, Any] = {"code": reason.code, "message": reason.message}
    if isinstance(reason.details, dict):
        payload["details"] = reason.details
    return payload


def _artifact_paths_from_payload(payload: Mapping[str, Any]) -> dict[str, str]:
    artifact_paths: dict[str, str] = {}
    raw_paths = payload.get("artifact_paths")
    if isinstance(raw_paths, dict):
        for key, value in raw_paths.items():
            if isinstance(key, str) and isinstance(value, str):
                artifact_paths[key] = value
    if artifact_paths:
        return artifact_paths
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list):
        for entry in artifacts:
            if not isinstance(entry, dict):
                continue
            label = entry.get("label")
            path = entry.get("path")
            if isinstance(label, str) and isinstance(path, str):
                artifact_paths[label] = path
    return artifact_paths


def _stage_state_from_payload(
    ep_id: str,
    run_id: str,
    stage: Stage,
    payload: Mapping[str, Any],
) -> StageState:
    status = StageStatus.from_value(payload.get("status"))
    timestamps = payload.get("timestamps") if isinstance(payload.get("timestamps"), dict) else {}
    started_at = _parse_iso(payload.get("started_at") or timestamps.get("started_at"))
    finished_at = _parse_iso(payload.get("ended_at") or payload.get("finished_at") or timestamps.get("ended_at"))
    duration_s = payload.get("duration_s")
    if duration_s is None:
        duration_s = _duration_s(_serialize_datetime(started_at), _serialize_datetime(finished_at))

    blocked_reason = _blocked_reason_from_payload(payload.get("blocked_reason"))
    error_reason = payload.get("error_reason") or payload.get("error")
    if not blocked_reason and error_reason and status == StageStatus.FAILED:
        blocked_reason = BlockedReason(code=str(payload.get("error_code") or "error"), message=str(error_reason))

    artifact_paths = _artifact_paths_from_payload(payload)
    artifact_digests = payload.get("artifact_digests") if isinstance(payload.get("artifact_digests"), dict) else {}
    model_versions = payload.get("model_versions") if isinstance(payload.get("model_versions"), dict) else {}
    thresholds = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, int) and key not in counts:
                counts[key] = value

    derived = bool(payload.get("derived", False))
    return StageState(
        episode_id=ep_id,
        run_id=run_id,
        stage=stage,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        duration_s=(float(duration_s) if isinstance(duration_s, (int, float)) else None),
        blocked_reason=blocked_reason,
        artifact_paths=artifact_paths,
        artifact_digests=artifact_digests,
        model_versions=model_versions,
        thresholds=thresholds,
        counts={k: int(v) for k, v in counts.items() if isinstance(v, int)},
        metrics=metrics,
        derived=derived,
    )


def _stage_state_to_payload(state: StageState) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": state.status.value,
        "started_at": _serialize_datetime(state.started_at),
        "ended_at": _serialize_datetime(state.finished_at),
        "duration_s": state.duration_s,
        "blocked_reason": _blocked_reason_to_payload(state.blocked_reason),
        "artifact_paths": state.artifact_paths or None,
        "artifact_digests": state.artifact_digests or None,
        "model_versions": state.model_versions or None,
        "thresholds": state.thresholds or None,
        "counts": state.counts or None,
        "metrics": state.metrics or None,
        "derived": state.derived,
    }
    if state.status == StageStatus.FAILED and state.blocked_reason:
        payload.setdefault("error_reason", state.blocked_reason.message)
        payload.setdefault("error_code", state.blocked_reason.code)
    return {k: v for k, v in payload.items() if v is not None}


def _episode_status_from_payload(ep_id: str, run_id: str, payload: Mapping[str, Any]) -> EpisodeStatus:
    stage_plan_raw = payload.get("stage_plan")
    stage_plan: list[Stage] = []
    if isinstance(stage_plan_raw, list):
        for entry in stage_plan_raw:
            stage = Stage.from_key(str(entry)) if entry is not None else None
            if stage and stage not in stage_plan:
                stage_plan.append(stage)
    if not stage_plan:
        stage_plan = [Stage.from_key(stage) for stage in STAGE_PLAN if Stage.from_key(stage)]

    stages_payload = payload.get("stages") if isinstance(payload.get("stages"), dict) else {}
    stages: dict[Stage, StageState] = {}
    for key, entry in stages_payload.items():
        if not isinstance(entry, Mapping):
            continue
        stage = Stage.from_key(str(key))
        if not stage:
            continue
        stages[stage] = _stage_state_from_payload(ep_id, run_id, stage, entry)

    return EpisodeStatus(
        episode_id=str(payload.get("episode_id") or ep_id),
        run_id=str(payload.get("run_id") or run_id),
        stages=stages,
        stage_plan=tuple(stage_plan),
        generated_at=_parse_iso(payload.get("generated_at")),
    )


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
    payload = _read_status_payload(ep_id, run_id_norm) or {
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


def derive_status_from_artifacts(ep_id: str, run_id: str) -> EpisodeStatus:
    run_id_norm = run_layout.normalize_run_id(run_id)
    stage_plan = [Stage.from_key(stage) for stage in STAGE_PLAN if Stage.from_key(stage)]
    stages: dict[Stage, StageState] = {}

    for stage in stage_plan:
        if stage is None:
            continue
        marker_name = _STAGE_MARKER_FILES.get(stage)
        if marker_name:
            marker_path = run_layout.run_phase_marker_path(ep_id, run_id_norm, marker_name)
            if marker_path.exists():
                try:
                    marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    marker_payload = None
                if isinstance(marker_payload, dict):
                    phase = str(marker_payload.get("phase") or marker_name)
                    stage_key, stage_update = stage_update_from_marker(
                        ep_id=ep_id,
                        run_id=run_id_norm,
                        phase=phase,
                        marker_payload=marker_payload,
                    )
                    if stage_key:
                        derived_payload = dict(stage_update)
                        derived_payload["derived"] = True
                        stages[stage] = _stage_state_from_payload(ep_id, run_id_norm, stage, derived_payload)
                        continue

        artifacts = stage_artifacts(ep_id, run_id_norm, stage.value)
        artifact_paths = {
            entry["label"]: entry["path"]
            for entry in artifacts
            if isinstance(entry, dict) and entry.get("exists") and isinstance(entry.get("label"), str)
        }
        status = StageStatus.SUCCESS if artifact_paths else StageStatus.NOT_STARTED
        stages[stage] = StageState(
            episode_id=ep_id,
            run_id=run_id_norm,
            stage=stage,
            status=status,
            artifact_paths=artifact_paths,
            derived=True,
        )

    return EpisodeStatus(
        episode_id=ep_id,
        run_id=run_id_norm,
        stages=stages,
        stage_plan=tuple(stage_plan),
        generated_at=datetime.now(timezone.utc).replace(microsecond=0),
    )


def _base_status_payload(ep_id: str, run_id: str) -> dict[str, Any]:
    return {
        "episode_id": ep_id,
        "run_id": run_id,
        "stage_plan": list(STAGE_PLAN),
        "stages": {},
    }


def _merge_stage_payload(payload: dict[str, Any], stage: Stage, update: Mapping[str, Any]) -> None:
    stages = payload.get("stages")
    if not isinstance(stages, dict):
        stages = {}
    current = stages.get(stage.value, {})
    if not isinstance(current, dict):
        current = {}
    stages[stage.value] = _merge_stage(current, update)
    payload["stages"] = stages


def _write_stage_state(ep_id: str, run_id: str, stage: Stage, state: StageState) -> EpisodeStatus:
    run_id_norm = run_layout.normalize_run_id(run_id)
    payload = _read_status_payload(ep_id, run_id_norm) or _base_status_payload(ep_id, run_id_norm)
    _merge_stage_payload(payload, stage, _stage_state_to_payload(state))
    payload["generated_at"] = _utcnow_iso()
    write_episode_status(ep_id, run_id_norm, payload)
    return _episode_status_from_payload(ep_id, run_id_norm, payload)


def write_stage_started(
    ep_id: str,
    run_id: str,
    stage: Stage | str,
    *,
    started_at: datetime | None = None,
    artifact_paths: Mapping[str, str] | None = None,
    artifact_digests: Mapping[str, str] | None = None,
    model_versions: Mapping[str, str] | None = None,
    thresholds: Mapping[str, object] | None = None,
    counts: Mapping[str, int] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> EpisodeStatus:
    stage_enum = _coerce_stage(stage)
    state = StageState(
        episode_id=ep_id,
        run_id=run_layout.normalize_run_id(run_id),
        stage=stage_enum,
        status=StageStatus.RUNNING,
        started_at=started_at or datetime.now(timezone.utc).replace(microsecond=0),
        artifact_paths=dict(artifact_paths or {}),
        artifact_digests=dict(artifact_digests or {}),
        model_versions=dict(model_versions or {}),
        thresholds=dict(thresholds or {}),
        counts=dict(counts or {}),
        metrics=dict(metrics or {}),
        derived=False,
    )
    return _write_stage_state(ep_id, run_id, stage_enum, state)


def write_stage_finished(
    ep_id: str,
    run_id: str,
    stage: Stage | str,
    *,
    finished_at: datetime | None = None,
    artifact_paths: Mapping[str, str] | None = None,
    artifact_digests: Mapping[str, str] | None = None,
    model_versions: Mapping[str, str] | None = None,
    thresholds: Mapping[str, object] | None = None,
    counts: Mapping[str, int] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> EpisodeStatus:
    stage_enum = _coerce_stage(stage)
    run_id_norm = run_layout.normalize_run_id(run_id)
    payload = _read_status_payload(ep_id, run_id_norm) or _base_status_payload(ep_id, run_id_norm)
    existing_entry = payload.get("stages", {}).get(stage_enum.value) if isinstance(payload.get("stages"), dict) else {}
    started_at = None
    if isinstance(existing_entry, Mapping):
        started_at = _parse_iso(existing_entry.get("started_at"))
    if started_at is None:
        timestamps = existing_entry.get("timestamps") if isinstance(existing_entry, Mapping) else {}
        started_at = _parse_iso(timestamps.get("started_at") if isinstance(timestamps, dict) else None)

    finished_at = finished_at or datetime.now(timezone.utc).replace(microsecond=0)
    duration_s = None
    if started_at:
        duration_s = _duration_s(_serialize_datetime(started_at), _serialize_datetime(finished_at))

    state = StageState(
        episode_id=ep_id,
        run_id=run_id_norm,
        stage=stage_enum,
        status=StageStatus.SUCCESS,
        started_at=started_at,
        finished_at=finished_at,
        duration_s=duration_s,
        artifact_paths=dict(artifact_paths or {}),
        artifact_digests=dict(artifact_digests or {}),
        model_versions=dict(model_versions or {}),
        thresholds=dict(thresholds or {}),
        counts=dict(counts or {}),
        metrics=dict(metrics or {}),
        derived=False,
    )
    _merge_stage_payload(payload, stage_enum, _stage_state_to_payload(state))
    payload["generated_at"] = _utcnow_iso()
    write_episode_status(ep_id, run_id_norm, payload)
    return _episode_status_from_payload(ep_id, run_id_norm, payload)


def write_stage_failed(
    ep_id: str,
    run_id: str,
    stage: Stage | str,
    error_code: str,
    error_message: str,
    *,
    finished_at: datetime | None = None,
    artifact_paths: Mapping[str, str] | None = None,
    artifact_digests: Mapping[str, str] | None = None,
    model_versions: Mapping[str, str] | None = None,
    thresholds: Mapping[str, object] | None = None,
    counts: Mapping[str, int] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> EpisodeStatus:
    stage_enum = _coerce_stage(stage)
    run_id_norm = run_layout.normalize_run_id(run_id)
    payload = _read_status_payload(ep_id, run_id_norm) or _base_status_payload(ep_id, run_id_norm)
    existing_entry = payload.get("stages", {}).get(stage_enum.value) if isinstance(payload.get("stages"), dict) else {}
    started_at = None
    if isinstance(existing_entry, Mapping):
        started_at = _parse_iso(existing_entry.get("started_at"))
    if started_at is None:
        timestamps = existing_entry.get("timestamps") if isinstance(existing_entry, Mapping) else {}
        started_at = _parse_iso(timestamps.get("started_at") if isinstance(timestamps, dict) else None)

    finished_at = finished_at or datetime.now(timezone.utc).replace(microsecond=0)
    duration_s = None
    if started_at:
        duration_s = _duration_s(_serialize_datetime(started_at), _serialize_datetime(finished_at))

    state = StageState(
        episode_id=ep_id,
        run_id=run_id_norm,
        stage=stage_enum,
        status=StageStatus.FAILED,
        started_at=started_at,
        finished_at=finished_at,
        duration_s=duration_s,
        blocked_reason=BlockedReason(code=error_code, message=error_message),
        artifact_paths=dict(artifact_paths or {}),
        artifact_digests=dict(artifact_digests or {}),
        model_versions=dict(model_versions or {}),
        thresholds=dict(thresholds or {}),
        counts=dict(counts or {}),
        metrics=dict(metrics or {}),
        derived=False,
    )
    _merge_stage_payload(payload, stage_enum, _stage_state_to_payload(state))
    payload["generated_at"] = _utcnow_iso()
    write_episode_status(ep_id, run_id_norm, payload)
    return _episode_status_from_payload(ep_id, run_id_norm, payload)


def write_stage_blocked(
    ep_id: str,
    run_id: str,
    stage: Stage | str,
    blocked_reason: BlockedReason,
    *,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    artifact_paths: Mapping[str, str] | None = None,
    artifact_digests: Mapping[str, str] | None = None,
    model_versions: Mapping[str, str] | None = None,
    thresholds: Mapping[str, object] | None = None,
    counts: Mapping[str, int] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> EpisodeStatus:
    stage_enum = _coerce_stage(stage)
    run_id_norm = run_layout.normalize_run_id(run_id)
    payload = _read_status_payload(ep_id, run_id_norm) or _base_status_payload(ep_id, run_id_norm)
    if started_at is None:
        existing_entry = payload.get("stages", {}).get(stage_enum.value) if isinstance(payload.get("stages"), dict) else {}
        if isinstance(existing_entry, Mapping):
            started_at = _parse_iso(existing_entry.get("started_at"))
        if started_at is None:
            timestamps = existing_entry.get("timestamps") if isinstance(existing_entry, Mapping) else {}
            started_at = _parse_iso(timestamps.get("started_at") if isinstance(timestamps, dict) else None)
    if finished_at is None:
        finished_at = datetime.now(timezone.utc).replace(microsecond=0)

    duration_s = None
    if started_at:
        duration_s = _duration_s(_serialize_datetime(started_at), _serialize_datetime(finished_at))

    state = StageState(
        episode_id=ep_id,
        run_id=run_id_norm,
        stage=stage_enum,
        status=StageStatus.BLOCKED,
        started_at=started_at,
        finished_at=finished_at,
        duration_s=duration_s,
        blocked_reason=blocked_reason,
        artifact_paths=dict(artifact_paths or {}),
        artifact_digests=dict(artifact_digests or {}),
        model_versions=dict(model_versions or {}),
        thresholds=dict(thresholds or {}),
        counts=dict(counts or {}),
        metrics=dict(metrics or {}),
        derived=False,
    )
    _merge_stage_payload(payload, stage_enum, _stage_state_to_payload(state))
    payload["generated_at"] = _utcnow_iso()
    write_episode_status(ep_id, run_id_norm, payload)
    return _episode_status_from_payload(ep_id, run_id_norm, payload)
