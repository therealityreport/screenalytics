"""DB-backed persistence for run-scoped debugging + UI consistency.

This module persists run/job execution traces and user-facing run-scoped state
(identity locks + smart suggestion batches) to Postgres when DB_URL is configured.

For unit-test contexts that don't provision Postgres, set SCREENALYTICS_FAKE_DB=1
to use an in-memory implementation with the same public methods.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

try:
    import psycopg2  # type: ignore
    from psycopg2.extras import Json, RealDictCursor, execute_values  # type: ignore

    _PSYCOPG2_AVAILABLE = True
except Exception:  # pragma: no cover - psycopg2 optional in some test contexts
    psycopg2 = None  # type: ignore[assignment]
    Json = None  # type: ignore[assignment]
    RealDictCursor = None  # type: ignore[assignment]
    execute_values = None  # type: ignore[assignment]
    _PSYCOPG2_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

DB_URL_ENV = "DB_URL"
FAKE_DB_ENV = "SCREENALYTICS_FAKE_DB"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json(value: Any) -> Any:
    """Wrap JSON payloads for psycopg2 when available."""
    if value is None:
        return None
    if Json is None:
        return value
    return Json(value)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    ep_id: str
    created_at: str
    label: str | None = None
    stage_state_json: dict[str, Any] | None = None
    config_json: dict[str, Any] | None = None


class RunPersistenceService:
    """Facade for run/job/suggestion persistence (Postgres or in-memory)."""

    def __init__(self) -> None:
        self._fake_lock = threading.Lock()
        self._fake_runs: dict[str, dict[str, Any]] = {}
        self._fake_job_runs: dict[str, dict[str, Any]] = {}
        self._fake_identity_locks: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._fake_batches: dict[str, dict[str, Any]] = {}
        self._fake_suggestions: dict[str, dict[str, Any]] = {}
        self._fake_applies: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _use_fake() -> bool:
        return os.getenv(FAKE_DB_ENV, "0") == "1"

    @staticmethod
    def _db_url() -> str:
        url = os.getenv(DB_URL_ENV)
        if not url:
            raise RuntimeError(f"{DB_URL_ENV} is not set")
        if not _PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 is not installed; DB persistence unavailable")
        return url

    @contextmanager
    def _conn(self):
        conn = psycopg2.connect(self._db_url(), cursor_factory=RealDictCursor, connect_timeout=3)  # type: ignore[arg-type]
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------
    def ensure_run(
        self,
        *,
        ep_id: str,
        run_id: str,
        label: str | None = None,
        config_json: dict[str, Any] | None = None,
    ) -> RunRecord:
        if self._use_fake():
            now = _utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")
            with self._fake_lock:
                existing = self._fake_runs.get(run_id)
                if existing:
                    if existing.get("ep_id") != ep_id:
                        raise ValueError("run_id_ep_id_mismatch")
                    if existing.get("label") is None and label:
                        existing["label"] = label
                    if existing.get("config_json") is None and config_json is not None:
                        existing["config_json"] = config_json
                else:
                    self._fake_runs[run_id] = {
                        "run_id": run_id,
                        "ep_id": ep_id,
                        "created_at": now,
                        "label": label,
                        "stage_state_json": None,
                        "config_json": config_json,
                    }
                row = dict(self._fake_runs[run_id])
            return RunRecord(**row)  # type: ignore[arg-type]

        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (run_id, ep_id, label, config_json)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE
                SET ep_id = EXCLUDED.ep_id,
                    label = COALESCE(runs.label, EXCLUDED.label),
                    config_json = COALESCE(runs.config_json, EXCLUDED.config_json)
                RETURNING
                    run_id,
                    ep_id,
                    created_at,
                    label,
                    stage_state_json,
                    config_json;
                """,
                (run_id, ep_id, label, _json(config_json)),
            )
            row = cur.fetchone()
        if not isinstance(row, Mapping):
            raise RuntimeError("Failed to upsert runs row")
        created_at = row.get("created_at")
        created_at_iso = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
        return RunRecord(
            run_id=str(row.get("run_id")),
            ep_id=str(row.get("ep_id")),
            created_at=created_at_iso,
            label=row.get("label"),
            stage_state_json=row.get("stage_state_json"),
            config_json=row.get("config_json"),
        )

    def list_runs(self, *, ep_id: str, limit: int = 100) -> list[dict[str, Any]]:
        if self._use_fake():
            with self._fake_lock:
                rows = [dict(r) for r in self._fake_runs.values() if r.get("ep_id") == ep_id]
            rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
            return rows[: max(1, int(limit))]

        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    run_id,
                    ep_id,
                    created_at,
                    label,
                    stage_state_json,
                    config_json
                FROM runs
                WHERE ep_id = %s
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (ep_id, max(1, int(limit))),
            )
            rows = cur.fetchall() or []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            created_at = row.get("created_at")
            created_at_iso = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
            out.append(
                {
                    "run_id": str(row.get("run_id")),
                    "ep_id": str(row.get("ep_id")),
                    "created_at": created_at_iso,
                    "label": row.get("label"),
                    "stage_state_json": row.get("stage_state_json"),
                    "config_json": row.get("config_json"),
                }
            )
        return out

    def update_run_stage_state(self, *, run_id: str, stage_state_json: dict[str, Any] | None) -> None:
        if self._use_fake():
            with self._fake_lock:
                if run_id in self._fake_runs:
                    self._fake_runs[run_id]["stage_state_json"] = stage_state_json
            return

        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE runs SET stage_state_json = %s WHERE run_id = %s;",
                (_json(stage_state_json), run_id),
            )

    def get_run(self, *, ep_id: str, run_id: str) -> dict[str, Any] | None:
        if self._use_fake():
            with self._fake_lock:
                row = self._fake_runs.get(run_id)
                if not row or row.get("ep_id") != ep_id:
                    return None
                return dict(row)

        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    run_id,
                    ep_id,
                    created_at,
                    label,
                    stage_state_json,
                    config_json
                FROM runs
                WHERE ep_id = %s AND run_id = %s
                LIMIT 1;
                """,
                (ep_id, run_id),
            )
            row = cur.fetchone()
        if not isinstance(row, Mapping):
            return None
        created_at = row.get("created_at")
        created_at_iso = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
        return {
            "run_id": str(row.get("run_id")),
            "ep_id": str(row.get("ep_id")),
            "created_at": created_at_iso,
            "label": row.get("label"),
            "stage_state_json": row.get("stage_state_json"),
            "config_json": row.get("config_json"),
        }

    # ------------------------------------------------------------------
    # Job runs
    # ------------------------------------------------------------------
    def create_job_run(
        self,
        *,
        ep_id: str,
        run_id: str,
        job_name: str,
        request_json: dict[str, Any] | None,
        status: str,
        started_at: datetime | None = None,
    ) -> str:
        if self._use_fake():
            job_run_id = str(uuid.uuid4())
            started_at_iso = (started_at or _utc_now()).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            with self._fake_lock:
                self._fake_job_runs[job_run_id] = {
                    "job_run_id": job_run_id,
                    "run_id": run_id,
                    "ep_id": ep_id,
                    "job_name": job_name,
                    "request_json": request_json or {},
                    "status": status,
                    "started_at": started_at_iso,
                    "finished_at": None,
                    "error_text": None,
                    "artifact_index_json": None,
                    "metrics_json": None,
                }
            return job_run_id

        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO job_runs (
                    run_id, ep_id, job_name, request_json, status, started_at
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING job_run_id::text AS job_run_id;
                """,
                (run_id, ep_id, job_name, _json(request_json or {}), status, started_at or _utc_now()),
            )
            row = cur.fetchone()
        if not isinstance(row, Mapping) or not row.get("job_run_id"):
            raise RuntimeError("Failed to create job_runs row")
        return str(row["job_run_id"])

    def update_job_run(
        self,
        *,
        job_run_id: str,
        status: str,
        finished_at: datetime | None = None,
        error_text: str | None = None,
        artifact_index_json: dict[str, Any] | None = None,
        metrics_json: dict[str, Any] | None = None,
    ) -> None:
        if self._use_fake():
            finished_at_iso = None
            if finished_at is not None:
                finished_at_iso = finished_at.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            with self._fake_lock:
                row = self._fake_job_runs.get(job_run_id)
                if row:
                    row["status"] = status
                    row["finished_at"] = finished_at_iso
                    row["error_text"] = error_text
                    row["artifact_index_json"] = artifact_index_json
                    row["metrics_json"] = metrics_json
            return

        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE job_runs
                SET status = %s,
                    finished_at = %s,
                    error_text = %s,
                    artifact_index_json = %s,
                    metrics_json = %s
                WHERE job_run_id = %s::uuid;
                """,
                (
                    status,
                    finished_at,
                    error_text,
                    _json(artifact_index_json),
                    _json(metrics_json),
                    job_run_id,
                ),
            )

    def list_job_runs(self, *, ep_id: str, run_id: str) -> list[dict[str, Any]]:
        if self._use_fake():
            with self._fake_lock:
                rows = [
                    dict(r)
                    for r in self._fake_job_runs.values()
                    if r.get("ep_id") == ep_id and r.get("run_id") == run_id
                ]
            rows.sort(key=lambda r: (r.get("started_at") or "", r.get("job_run_id") or ""))
            return rows

        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    job_run_id::text AS job_run_id,
                    run_id,
                    ep_id,
                    job_name,
                    request_json,
                    status,
                    started_at,
                    finished_at,
                    error_text,
                    artifact_index_json,
                    metrics_json
                FROM job_runs
                WHERE ep_id = %s AND run_id = %s
                ORDER BY started_at NULLS LAST, job_run_id;
                """,
                (ep_id, run_id),
            )
            rows = cur.fetchall() or []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            started_at = row.get("started_at")
            finished_at = row.get("finished_at")
            out.append(
                {
                    "job_run_id": str(row.get("job_run_id")),
                    "run_id": str(row.get("run_id")),
                    "ep_id": str(row.get("ep_id")),
                    "job_name": row.get("job_name"),
                    "request_json": row.get("request_json"),
                    "status": row.get("status"),
                    "started_at": started_at.isoformat() if hasattr(started_at, "isoformat") else started_at,
                    "finished_at": finished_at.isoformat() if hasattr(finished_at, "isoformat") else finished_at,
                    "error_text": row.get("error_text"),
                    "artifact_index_json": row.get("artifact_index_json"),
                    "metrics_json": row.get("metrics_json"),
                }
            )
        return out

    # ------------------------------------------------------------------
    # Identity locks
    # ------------------------------------------------------------------
    def set_identity_lock(
        self,
        *,
        ep_id: str,
        run_id: str,
        identity_id: str,
        locked: bool,
        locked_by: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        locked_at = _utc_now() if locked else None
        if self._use_fake():
            payload = {
                "ep_id": ep_id,
                "run_id": run_id,
                "identity_id": identity_id,
                "locked": bool(locked),
                "locked_at": locked_at.replace(microsecond=0).isoformat().replace("+00:00", "Z") if locked_at else None,
                "locked_by": locked_by,
                "reason": reason,
            }
            with self._fake_lock:
                self._fake_identity_locks[(ep_id, run_id, identity_id)] = payload
            return payload

        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO identity_locks (
                    ep_id, run_id, identity_id, locked, locked_at, locked_by, reason
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ep_id, run_id, identity_id) DO UPDATE
                SET locked = EXCLUDED.locked,
                    locked_at = EXCLUDED.locked_at,
                    locked_by = EXCLUDED.locked_by,
                    reason = EXCLUDED.reason
                RETURNING
                    ep_id,
                    run_id,
                    identity_id,
                    locked,
                    locked_at,
                    locked_by,
                    reason;
                """,
                (ep_id, run_id, identity_id, bool(locked), locked_at, locked_by, reason),
            )
            row = cur.fetchone()
        if not isinstance(row, Mapping):
            raise RuntimeError("Failed to upsert identity lock")
        locked_at_val = row.get("locked_at")
        return {
            "ep_id": str(row.get("ep_id")),
            "run_id": str(row.get("run_id")),
            "identity_id": str(row.get("identity_id")),
            "locked": bool(row.get("locked")),
            "locked_at": locked_at_val.isoformat() if hasattr(locked_at_val, "isoformat") else locked_at_val,
            "locked_by": row.get("locked_by"),
            "reason": row.get("reason"),
        }

    def is_identity_locked(self, *, ep_id: str, run_id: str, identity_id: str) -> bool:
        if self._use_fake():
            with self._fake_lock:
                row = self._fake_identity_locks.get((ep_id, run_id, identity_id))
                return bool(row and row.get("locked"))

        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT locked
                FROM identity_locks
                WHERE ep_id = %s AND run_id = %s AND identity_id = %s
                LIMIT 1;
                """,
                (ep_id, run_id, identity_id),
            )
            row = cur.fetchone()
        if not isinstance(row, Mapping):
            return False
        return bool(row.get("locked"))

    def list_identity_locks(self, *, ep_id: str, run_id: str) -> list[dict[str, Any]]:
        if self._use_fake():
            with self._fake_lock:
                rows = [
                    dict(v)
                    for (e, r, _), v in self._fake_identity_locks.items()
                    if e == ep_id and r == run_id
                ]
            rows.sort(key=lambda x: (x.get("identity_id") or ""))
            return rows

        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    ep_id,
                    run_id,
                    identity_id,
                    locked,
                    locked_at,
                    locked_by,
                    reason
                FROM identity_locks
                WHERE ep_id = %s AND run_id = %s
                ORDER BY identity_id;
                """,
                (ep_id, run_id),
            )
            rows = cur.fetchall() or []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            locked_at_val = row.get("locked_at")
            out.append(
                {
                    "ep_id": str(row.get("ep_id")),
                    "run_id": str(row.get("run_id")),
                    "identity_id": str(row.get("identity_id")),
                    "locked": bool(row.get("locked")),
                    "locked_at": locked_at_val.isoformat() if hasattr(locked_at_val, "isoformat") else locked_at_val,
                    "locked_by": row.get("locked_by"),
                    "reason": row.get("reason"),
                }
            )
        return out

    # ------------------------------------------------------------------
    # Smart suggestion batches
    # ------------------------------------------------------------------
    def create_suggestion_batch(
        self,
        *,
        ep_id: str,
        run_id: str,
        generator_version: str,
        generator_config_json: dict[str, Any],
        summary_json: dict[str, Any] | None = None,
    ) -> str:
        if self._use_fake():
            batch_id = str(uuid.uuid4())
            created_at = _utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")
            with self._fake_lock:
                self._fake_batches[batch_id] = {
                    "batch_id": batch_id,
                    "ep_id": ep_id,
                    "run_id": run_id,
                    "created_at": created_at,
                    "generator_version": generator_version,
                    "generator_config_json": generator_config_json,
                    "summary_json": summary_json,
                }
            return batch_id

        if execute_values is None:
            raise RuntimeError("psycopg2 execute_values unavailable")
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO suggestion_batches (
                    ep_id, run_id, generator_version, generator_config_json, summary_json
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING batch_id::text AS batch_id;
                """,
                (ep_id, run_id, generator_version, _json(generator_config_json), _json(summary_json)),
            )
            row = cur.fetchone()
        if not isinstance(row, Mapping) or not row.get("batch_id"):
            raise RuntimeError("Failed to create suggestion batch")
        return str(row["batch_id"])

    def list_suggestion_batches(self, *, ep_id: str, run_id: str, limit: int = 25) -> list[dict[str, Any]]:
        if self._use_fake():
            with self._fake_lock:
                rows = [dict(v) for v in self._fake_batches.values() if v.get("ep_id") == ep_id and v.get("run_id") == run_id]
            rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
            return rows[: max(1, int(limit))]

        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    batch_id::text AS batch_id,
                    ep_id,
                    run_id,
                    created_at,
                    generator_version,
                    generator_config_json,
                    summary_json
                FROM suggestion_batches
                WHERE ep_id = %s AND run_id = %s
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (ep_id, run_id, max(1, int(limit))),
            )
            rows = cur.fetchall() or []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            created_at = row.get("created_at")
            created_at_iso = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
            out.append(
                {
                    "batch_id": str(row.get("batch_id")),
                    "ep_id": str(row.get("ep_id")),
                    "run_id": str(row.get("run_id")),
                    "created_at": created_at_iso,
                    "generator_version": row.get("generator_version"),
                    "generator_config_json": row.get("generator_config_json"),
                    "summary_json": row.get("summary_json"),
                }
            )
        return out

    def insert_suggestions(
        self,
        *,
        batch_id: str,
        ep_id: str,
        run_id: str,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Insert suggestions and return inserted rows (including suggestion_id)."""
        if not rows:
            return []

        if self._use_fake():
            created: list[dict[str, Any]] = []
            created_at = _utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")
            with self._fake_lock:
                for row in rows:
                    suggestion_id = str(uuid.uuid4())
                    stored = {
                        "suggestion_id": suggestion_id,
                        "batch_id": batch_id,
                        "ep_id": ep_id,
                        "run_id": run_id,
                        "type": row.get("type"),
                        "target_identity_id": row.get("target_identity_id"),
                        "suggested_person_id": row.get("suggested_person_id"),
                        "confidence": float(row.get("confidence") or 0.0),
                        "evidence_json": row.get("evidence_json") or {},
                        "created_at": created_at,
                        "dismissed": False,
                        "dismissed_at": None,
                    }
                    self._fake_suggestions[suggestion_id] = stored
                    created.append(dict(stored))
            return created

        if execute_values is None:
            raise RuntimeError("psycopg2 execute_values unavailable")

        values: list[tuple[Any, ...]] = []
        for row in rows:
            values.append(
                (
                    batch_id,
                    ep_id,
                    run_id,
                    row.get("type"),
                    row.get("target_identity_id"),
                    row.get("suggested_person_id"),
                    float(row.get("confidence") or 0.0),
                    _json(row.get("evidence_json") or {}),
                )
            )

        with self._conn() as conn, conn.cursor() as cur:
            execute_values(  # type: ignore[misc]
                cur,
                """
                INSERT INTO suggestions (
                    batch_id, ep_id, run_id, type, target_identity_id, suggested_person_id, confidence, evidence_json
                )
                VALUES %s
                RETURNING
                    suggestion_id::text AS suggestion_id,
                    batch_id::text AS batch_id,
                    ep_id,
                    run_id,
                    type,
                    target_identity_id,
                    suggested_person_id,
                    confidence,
                    evidence_json,
                    created_at,
                    dismissed,
                    dismissed_at;
                """,
                values,
                template="(%s::uuid, %s, %s, %s, %s, %s, %s, %s)",
            )
            inserted = cur.fetchall() or []

        out: list[dict[str, Any]] = []
        for row in inserted:
            if not isinstance(row, Mapping):
                continue
            created_at = row.get("created_at")
            dismissed_at = row.get("dismissed_at")
            out.append(
                {
                    "suggestion_id": str(row.get("suggestion_id")),
                    "batch_id": str(row.get("batch_id")),
                    "ep_id": str(row.get("ep_id")),
                    "run_id": str(row.get("run_id")),
                    "type": row.get("type"),
                    "target_identity_id": row.get("target_identity_id"),
                    "suggested_person_id": row.get("suggested_person_id"),
                    "confidence": float(row.get("confidence") or 0.0),
                    "evidence_json": row.get("evidence_json") or {},
                    "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
                    "dismissed": bool(row.get("dismissed")),
                    "dismissed_at": dismissed_at.isoformat() if hasattr(dismissed_at, "isoformat") else dismissed_at,
                }
            )
        return out

    def list_suggestions(
        self,
        *,
        ep_id: str,
        run_id: str,
        batch_id: str,
        include_dismissed: bool = False,
    ) -> list[dict[str, Any]]:
        if self._use_fake():
            with self._fake_lock:
                rows = [
                    dict(v)
                    for v in self._fake_suggestions.values()
                    if v.get("ep_id") == ep_id and v.get("run_id") == run_id and v.get("batch_id") == batch_id
                ]
            if not include_dismissed:
                rows = [r for r in rows if not r.get("dismissed")]
            rows.sort(key=lambda r: (-(float(r.get("confidence") or 0.0)), r.get("suggestion_id") or ""))
            return rows

        where_clause = "WHERE ep_id = %s AND run_id = %s AND batch_id = %s::uuid"
        if not include_dismissed:
            where_clause += " AND dismissed = false"
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    suggestion_id::text AS suggestion_id,
                    batch_id::text AS batch_id,
                    ep_id,
                    run_id,
                    type,
                    target_identity_id,
                    suggested_person_id,
                    confidence,
                    evidence_json,
                    created_at,
                    dismissed,
                    dismissed_at
                FROM suggestions
                {where_clause}
                ORDER BY confidence DESC, suggestion_id;
                """,
                (ep_id, run_id, batch_id),
            )
            rows = cur.fetchall() or []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            created_at = row.get("created_at")
            dismissed_at = row.get("dismissed_at")
            out.append(
                {
                    "suggestion_id": str(row.get("suggestion_id")),
                    "batch_id": str(row.get("batch_id")),
                    "ep_id": str(row.get("ep_id")),
                    "run_id": str(row.get("run_id")),
                    "type": row.get("type"),
                    "target_identity_id": row.get("target_identity_id"),
                    "suggested_person_id": row.get("suggested_person_id"),
                    "confidence": float(row.get("confidence") or 0.0),
                    "evidence_json": row.get("evidence_json") or {},
                    "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
                    "dismissed": bool(row.get("dismissed")),
                    "dismissed_at": dismissed_at.isoformat() if hasattr(dismissed_at, "isoformat") else dismissed_at,
                }
            )
        return out

    def set_suggestion_dismissed(
        self,
        *,
        ep_id: str,
        run_id: str,
        batch_id: str,
        suggestion_id: str,
        dismissed: bool,
    ) -> bool:
        now = _utc_now() if dismissed else None
        if self._use_fake():
            with self._fake_lock:
                row = self._fake_suggestions.get(suggestion_id)
                if not row:
                    return False
                if row.get("ep_id") != ep_id or row.get("run_id") != run_id or row.get("batch_id") != batch_id:
                    return False
                row["dismissed"] = bool(dismissed)
                row["dismissed_at"] = now.replace(microsecond=0).isoformat().replace("+00:00", "Z") if now else None
                return True

        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE suggestions
                SET dismissed = %s,
                    dismissed_at = %s
                WHERE ep_id = %s
                  AND run_id = %s
                  AND batch_id = %s::uuid
                  AND suggestion_id = %s::uuid;
                """,
                (bool(dismissed), now, ep_id, run_id, batch_id, suggestion_id),
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Suggestion applies
    # ------------------------------------------------------------------
    def record_suggestion_apply(
        self,
        *,
        ep_id: str,
        run_id: str,
        batch_id: str,
        suggestion_id: str | None,
        applied_by: str | None,
        changes_json: dict[str, Any],
    ) -> str:
        if self._use_fake():
            apply_id = str(uuid.uuid4())
            applied_at = _utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")
            with self._fake_lock:
                self._fake_applies[apply_id] = {
                    "apply_id": apply_id,
                    "batch_id": batch_id,
                    "suggestion_id": suggestion_id,
                    "ep_id": ep_id,
                    "run_id": run_id,
                    "applied_at": applied_at,
                    "applied_by": applied_by,
                    "changes_json": changes_json,
                }
            return apply_id

        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO suggestion_applies (
                    batch_id, suggestion_id, ep_id, run_id, applied_by, changes_json
                )
                VALUES (%s::uuid, %s::uuid, %s, %s, %s, %s)
                RETURNING apply_id::text AS apply_id;
                """,
                (
                    batch_id,
                    suggestion_id,
                    ep_id,
                    run_id,
                    applied_by,
                    _json(changes_json or {}),
                ),
            )
            row = cur.fetchone()
        if not isinstance(row, Mapping) or not row.get("apply_id"):
            raise RuntimeError("Failed to record suggestion apply")
        return str(row["apply_id"])

    def list_suggestion_applies(self, *, ep_id: str, run_id: str) -> list[dict[str, Any]]:
        if self._use_fake():
            with self._fake_lock:
                rows = [dict(v) for v in self._fake_applies.values() if v.get("ep_id") == ep_id and v.get("run_id") == run_id]
            rows.sort(key=lambda r: (r.get("applied_at") or "", r.get("apply_id") or ""))
            return rows

        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    apply_id::text AS apply_id,
                    batch_id::text AS batch_id,
                    suggestion_id::text AS suggestion_id,
                    ep_id,
                    run_id,
                    applied_at,
                    applied_by,
                    changes_json
                FROM suggestion_applies
                WHERE ep_id = %s AND run_id = %s
                ORDER BY applied_at, apply_id;
                """,
                (ep_id, run_id),
            )
            rows = cur.fetchall() or []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            applied_at = row.get("applied_at")
            out.append(
                {
                    "apply_id": str(row.get("apply_id")),
                    "batch_id": str(row.get("batch_id")),
                    "suggestion_id": str(row.get("suggestion_id")) if row.get("suggestion_id") else None,
                    "ep_id": str(row.get("ep_id")),
                    "run_id": str(row.get("run_id")),
                    "applied_at": applied_at.isoformat() if hasattr(applied_at, "isoformat") else applied_at,
                    "applied_by": row.get("applied_by"),
                    "changes_json": row.get("changes_json") or {},
                }
            )
        return out


run_persistence_service = RunPersistenceService()
