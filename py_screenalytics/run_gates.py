from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from py_screenalytics import run_layout
from py_screenalytics.episode_status import Stage, StageStatus, normalize_stage_key, read_episode_status

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateReason:
    code: str
    message: str
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if isinstance(self.details, dict):
            payload["details"] = self.details
        return payload


@dataclass
class GateResult:
    ok: bool
    reasons: list[GateReason] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ArtifactRequirement:
    logical_name: str
    rel_path: str
    upstream_stage: Stage | None = None


@dataclass(frozen=True)
class StagePrereqs:
    upstream: tuple[Stage, ...] = ()
    artifacts: tuple[ArtifactRequirement, ...] = ()


_STAGE_PREREQS: dict[Stage, StagePrereqs] = {
    Stage.DETECT: StagePrereqs(),
    Stage.FACES: StagePrereqs(
        upstream=(Stage.DETECT,),
        artifacts=(
            ArtifactRequirement("tracks", "tracks.jsonl", upstream_stage=Stage.DETECT),
        ),
    ),
    Stage.CLUSTER: StagePrereqs(
        upstream=(Stage.FACES,),
        artifacts=(
            ArtifactRequirement("faces", "faces.jsonl", upstream_stage=Stage.FACES),
            ArtifactRequirement("tracks", "tracks.jsonl", upstream_stage=Stage.DETECT),
        ),
    ),
    Stage.BODY_TRACKING: StagePrereqs(),
    Stage.TRACK_FUSION: StagePrereqs(
        upstream=(Stage.BODY_TRACKING, Stage.FACES),
        artifacts=(
            ArtifactRequirement("body_tracks", "body_tracking/body_tracks.jsonl", upstream_stage=Stage.BODY_TRACKING),
            ArtifactRequirement("faces", "faces.jsonl", upstream_stage=Stage.FACES),
        ),
    ),
    Stage.SCREENTIME: StagePrereqs(
        upstream=(Stage.CLUSTER,),
        artifacts=(
            ArtifactRequirement("tracks", "tracks.jsonl", upstream_stage=Stage.DETECT),
            ArtifactRequirement("faces", "faces.jsonl", upstream_stage=Stage.FACES),
            ArtifactRequirement("identities", "identities.json", upstream_stage=Stage.CLUSTER),
        ),
    ),
    Stage.PDF: StagePrereqs(),
}


def get_stage_prereqs(stage: Stage | str) -> StagePrereqs:
    stage_enum = _coerce_stage(stage)
    return _STAGE_PREREQS.get(stage_enum, StagePrereqs())


def check_prereqs(
    stage: Stage | str,
    episode_id: str,
    run_id: str,
    *,
    config: dict[str, Any] | None = None,
) -> GateResult:
    stage_enum = _coerce_stage(stage)
    run_id_norm = run_layout.normalize_run_id(run_id)
    prereqs = get_stage_prereqs(stage_enum)

    reasons: list[GateReason] = []
    if _stage_disabled(stage_enum, config=config):
        reasons.append(
            GateReason(
                code="stage_disabled",
                message=f"{stage_enum.value} is disabled by configuration",
                details={"stage": stage_enum.value},
            )
        )

    status = read_episode_status(episode_id, run_id_norm)
    for upstream in prereqs.upstream:
        state = status.stages.get(upstream)
        if state is None:
            reasons.append(
                GateReason(
                    code="upstream_not_success",
                    message=f"{upstream.value} status missing",
                    details={"stage": upstream.value, "status": "missing"},
                )
            )
            continue
        if state.status == StageStatus.FAILED:
            reasons.append(
                GateReason(
                    code="upstream_failed",
                    message=f"{upstream.value} failed",
                    details={"stage": upstream.value, "status": state.status.value},
                )
            )
        elif state.status != StageStatus.SUCCESS:
            reasons.append(
                GateReason(
                    code="upstream_not_success",
                    message=f"{upstream.value} not successful",
                    details={"stage": upstream.value, "status": state.status.value},
                )
            )

    for req in prereqs.artifacts:
        expected = run_layout.run_root(episode_id, run_id_norm) / req.rel_path
        if expected.exists():
            continue
        mismatch = _find_mismatch(episode_id, run_id_norm, req.rel_path)
        if mismatch is not None:
            found_scope, found_run, found_path = mismatch
            reasons.append(
                GateReason(
                    code="run_id_mismatch",
                    message=f"{req.logical_name} belongs to a different run",
                    details={
                        "artifact": req.logical_name,
                        "expected_path": str(expected),
                        "found_scope": found_scope,
                        "found_run_id": found_run,
                        "found_path": str(found_path),
                    },
                )
            )
        else:
            reasons.append(
                GateReason(
                    code="missing_artifact",
                    message=f"Missing {req.logical_name}",
                    details={
                        "artifact": req.logical_name,
                        "expected_path": str(expected),
                    },
                )
            )

    suggested_actions = _suggest_actions(stage_enum, reasons, prereqs)
    return GateResult(ok=not reasons, reasons=reasons, suggested_actions=suggested_actions)


def _coerce_stage(stage: Stage | str) -> Stage:
    if isinstance(stage, Stage):
        return stage
    normalized = normalize_stage_key(stage)
    if not normalized:
        raise ValueError(f"Unknown stage key: {stage!r}")
    for entry in Stage:
        if entry.value == normalized:
            return entry
    raise ValueError(f"Unknown stage key: {stage!r}")


def _suggest_actions(
    stage: Stage,
    reasons: Iterable[GateReason],
    prereqs: StagePrereqs,
) -> list[str]:
    suggestions: list[str] = []
    artifact_to_stage = {req.logical_name: req.upstream_stage for req in prereqs.artifacts}
    for reason in reasons:
        if reason.code == "missing_artifact":
            artifact = (reason.details or {}).get("artifact")
            upstream = artifact_to_stage.get(artifact) if isinstance(artifact, str) else None
            if upstream:
                suggestions.append(f"Run {upstream.value} to generate {artifact}.")
            elif isinstance(artifact, str):
                suggestions.append(f"Generate required artifact: {artifact}.")
        elif reason.code == "upstream_failed":
            stage_name = (reason.details or {}).get("stage")
            if isinstance(stage_name, str):
                suggestions.append(f"Re-run {stage_name} after addressing the failure.")
        elif reason.code == "upstream_not_success":
            stage_name = (reason.details or {}).get("stage")
            if isinstance(stage_name, str):
                suggestions.append(f"Wait for {stage_name} to finish or re-run it.")
        elif reason.code == "run_id_mismatch":
            found_run = (reason.details or {}).get("found_run_id")
            if isinstance(found_run, str):
                suggestions.append(f"Use run_id={found_run} or rerun the pipeline for {stage.value}.")
            else:
                suggestions.append(f"Rerun upstream stages for run_id to populate {stage.value} prerequisites.")
        elif reason.code == "stage_disabled":
            if stage in {Stage.BODY_TRACKING, Stage.TRACK_FUSION}:
                suggestions.append("Enable body tracking in config/pipeline/body_detection.yaml or AUTO_RUN_BODY_TRACKING=1.")
            else:
                suggestions.append(f"Enable {stage.value} in the pipeline configuration.")
    if reasons and not suggestions:
        suggestions.append(f"Resolve prerequisites for {stage.value} and retry.")
    return suggestions


def _find_mismatch(ep_id: str, run_id: str, rel_path: str) -> tuple[str, str | None, Path] | None:
    for candidate in run_layout.list_run_ids(ep_id):
        if candidate == run_id:
            continue
        candidate_path = run_layout.run_root(ep_id, candidate) / rel_path
        if candidate_path.exists():
            return ("run", candidate, candidate_path)
    legacy_path = run_layout.manifests_root(ep_id) / rel_path
    if legacy_path.exists():
        return ("legacy", None, legacy_path)
    return None


def _stage_disabled(stage: Stage, config: dict[str, Any] | None) -> bool:
    if stage not in {Stage.BODY_TRACKING, Stage.TRACK_FUSION}:
        return False
    config_payload = config if isinstance(config, dict) else _load_body_tracking_config()
    body_tracking_cfg = config_payload.get("body_tracking") if isinstance(config_payload.get("body_tracking"), dict) else {}
    enabled = body_tracking_cfg.get("enabled")
    if enabled is None:
        return False
    return not bool(enabled)


def _load_body_tracking_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "config" / "pipeline" / "body_detection.yaml"
    config: dict[str, Any] = {}
    if config_path.exists():
        try:
            import yaml

            with config_path.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
        except Exception as exc:
            LOGGER.warning("Failed to load body tracking config YAML: %s", exc)

    env_override = os.environ.get("AUTO_RUN_BODY_TRACKING", "").strip().lower()
    if env_override in ("0", "false", "no", "off"):
        config.setdefault("body_tracking", {})
        config["body_tracking"]["enabled"] = False
    elif env_override in ("1", "true", "yes", "on"):
        config.setdefault("body_tracking", {})
        config["body_tracking"]["enabled"] = True
    return config
