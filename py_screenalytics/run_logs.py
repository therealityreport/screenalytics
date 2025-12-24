from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - non-posix fallback
    fcntl = None

from py_screenalytics import run_layout
from py_screenalytics.episode_status import Stage, normalize_stage_key

LOGGER = logging.getLogger(__name__)

_LOG_LOCK_TIMEOUT_S = 5.0
_LOG_LOCK_POLL_S = 0.1
_DEFAULT_TAIL_MAX_BYTES = 512 * 1024


@dataclass(frozen=True)
class LogEvent:
    ts: str
    level: str
    episode_id: str
    run_id: str
    stage: str
    msg: str
    progress: float | None = None
    meta: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ts": self.ts,
            "level": self.level,
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "stage": self.stage,
            "msg": self.msg,
        }
        if self.progress is not None:
            payload["progress"] = self.progress
        if isinstance(self.meta, dict):
            payload["meta"] = self.meta
        return payload


def append_log(
    episode_id: str,
    run_id: str,
    stage: Stage | str,
    level: str,
    msg: str,
    *,
    progress: float | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    run_id_norm = run_layout.normalize_run_id(run_id)
    stage_key = _coerce_stage_key(stage)
    path = _log_path(episode_id, run_id_norm, stage_key)
    path.parent.mkdir(parents=True, exist_ok=True)

    event = LogEvent(
        ts=_utcnow_iso(),
        level=str(level).upper(),
        episode_id=episode_id,
        run_id=run_id_norm,
        stage=stage_key,
        msg=str(msg),
        progress=progress,
        meta=meta,
    )
    line = json.dumps(event.as_dict(), sort_keys=True)
    with _log_lock(path):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def tail_logs(
    episode_id: str,
    run_id: str,
    stage: Stage | str,
    n: int = 200,
) -> list[dict[str, Any]]:
    run_id_norm = run_layout.normalize_run_id(run_id)
    stage_key = _coerce_stage_key(stage)
    path = _log_path(episode_id, run_id_norm, stage_key)
    if not path.exists():
        return []
    lines = _tail_lines(path, max(int(n), 0))
    events: list[dict[str, Any]] = []
    for raw in lines:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def read_stage_progress(
    episode_id: str,
    run_id: str,
    stage: Stage | str,
) -> float | None:
    events = tail_logs(episode_id, run_id, stage, n=200)
    for payload in reversed(events):
        progress = payload.get("progress")
        if isinstance(progress, (int, float)):
            return float(progress)
    return None


@contextmanager
def _log_lock(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None:
        deadline = time.time() + _LOG_LOCK_TIMEOUT_S
        fd: int | None = None
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.time() >= deadline:
                    LOGGER.warning("[run_logs] Lock wait timed out for %s", lock_path)
                    break
                time.sleep(_LOG_LOCK_POLL_S)
        try:
            yield
        finally:
            if fd is not None:
                os.close(fd)
                try:
                    lock_path.unlink()
                except OSError:
                    pass
        return
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _log_path(ep_id: str, run_id: str, stage_key: str) -> Path:
    run_root = run_layout.run_root(ep_id, run_id)
    return run_root / "logs" / f"{stage_key}.jsonl"


def _tail_lines(path: Path, n: int) -> list[str]:
    if n <= 0:
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            file_size = handle.tell()
            if file_size == 0:
                return []
            read_size = min(file_size, _DEFAULT_TAIL_MAX_BYTES)
            handle.seek(file_size - read_size)
            data = handle.read(read_size)
    except OSError:
        return []

    lines = data.splitlines()
    if len(lines) > n:
        lines = lines[-n:]
    decoded: list[str] = []
    for raw in lines:
        try:
            decoded.append(raw.decode("utf-8"))
        except UnicodeDecodeError:
            decoded.append(raw.decode("utf-8", errors="ignore"))
    return decoded


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_stage_key(stage: Stage | str) -> str:
    if isinstance(stage, Stage):
        return stage.value
    normalized = normalize_stage_key(stage)
    if normalized:
        return normalized
    raw = str(stage or "unknown").strip().lower().replace(" ", "_")
    return raw or "unknown"
