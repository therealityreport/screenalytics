"""Run-scoped filesystem layout helpers.

This module defines a lightweight "run_id" namespace for pipeline artifacts so
reruns can be inspected without silently mixing outputs across runs.

Conventions:
- Legacy (non-run-scoped) manifests live in: data/manifests/{ep_id}/
- Run-scoped artifacts live in:           data/manifests/{ep_id}/runs/{run_id}/
- Phase markers (legacy) live in:         data/manifests/{ep_id}/runs/{phase}.json

We intentionally keep this separate from the legacy `get_path()` resolver so
existing call sites remain backward compatible.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from py_screenalytics.artifacts import get_path


RUNS_SUBDIR = "runs"
ACTIVE_RUN_FILENAME = "active_run.json"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def generate_run_id() -> str:
    return uuid.uuid4().hex


def normalize_run_id(run_id: str) -> str:
    value = (run_id or "").strip()
    if not value:
        raise ValueError("run_id must be non-empty")
    if not _RUN_ID_RE.match(value):
        raise ValueError(
            "run_id must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ "
            f"(got {run_id!r})"
        )
    return value


def manifests_root(ep_id: str) -> Path:
    return get_path(ep_id, "detections").parent


def runs_root(ep_id: str) -> Path:
    return manifests_root(ep_id) / RUNS_SUBDIR


def run_root(ep_id: str, run_id: str) -> Path:
    return runs_root(ep_id) / normalize_run_id(run_id)


def run_manifest_path(ep_id: str, run_id: str, filename: str) -> Path:
    cleaned = (filename or "").strip().lstrip("/\\")
    if not cleaned:
        raise ValueError("filename must be non-empty")
    return run_root(ep_id, run_id) / cleaned


def phase_marker_path(ep_id: str, phase: str) -> Path:
    cleaned = (phase or "").strip()
    if not cleaned:
        raise ValueError("phase must be non-empty")
    return runs_root(ep_id) / f"{cleaned}.json"


def run_phase_marker_path(ep_id: str, run_id: str, phase: str) -> Path:
    cleaned = (phase or "").strip()
    if not cleaned:
        raise ValueError("phase must be non-empty")
    return run_root(ep_id, run_id) / f"{cleaned}.json"


def active_run_path(ep_id: str) -> Path:
    return runs_root(ep_id) / ACTIVE_RUN_FILENAME


def write_active_run_id(ep_id: str, run_id: str, *, extra: dict[str, Any] | None = None) -> Path:
    runs_root(ep_id).mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "ep_id": ep_id,
        "run_id": normalize_run_id(run_id),
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    if extra:
        payload.update(extra)
    path = active_run_path(ep_id)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_active_run_id(ep_id: str) -> str | None:
    path = active_run_path(ep_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    run_id = payload.get("run_id")
    if not isinstance(run_id, str):
        return None
    try:
        return normalize_run_id(run_id)
    except ValueError:
        return None


def list_run_ids(ep_id: str) -> list[str]:
    root = runs_root(ep_id)
    if not root.exists():
        return []
    run_ids: list[str] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        try:
            run_ids.append(normalize_run_id(entry.name))
        except ValueError:
            continue
    run_ids.sort()
    return run_ids


@dataclass(frozen=True)
class RunPaths:
    ep_id: str
    run_id: str

    @property
    def root(self) -> Path:
        return run_root(self.ep_id, self.run_id)

    def manifest(self, filename: str) -> Path:
        return run_manifest_path(self.ep_id, self.run_id, filename)

    def marker(self, phase: str) -> Path:
        return run_phase_marker_path(self.ep_id, self.run_id, phase)

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


def as_run_paths(ep_id: str, run_id: str) -> RunPaths:
    return RunPaths(ep_id=ep_id, run_id=normalize_run_id(run_id))

