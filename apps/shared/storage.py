from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

LOGGER = logging.getLogger(__name__)
_S3_PREFIXES: tuple[str, ...] = ("artifacts/",)
_STORAGE = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    raw = os.environ.get("SCREENALYTICS_DATA_ROOT")
    if raw:
        return Path(raw).expanduser()
    return project_root() / "data"


def use_s3() -> bool:
    backend = os.environ.get("STORAGE_BACKEND", "s3").lower()
    return backend in {"s3", "minio"}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_storage():
    global _STORAGE
    if _STORAGE is None:
        from apps.api.services.storage import StorageService

        _STORAGE = StorageService()
    return _STORAGE


def _is_s3_key(target: str | Path) -> tuple[bool, str]:
    if not use_s3() or isinstance(target, Path):
        return False, ""
    raw = str(target).strip().lstrip("/").replace("\\", "/")
    if any(raw.startswith(prefix) for prefix in _S3_PREFIXES):
        return True, raw
    return False, ""


def _resolve_local_path(target: str | Path) -> Path:
    if isinstance(target, Path):
        return target
    raw = str(target).strip()
    if not raw:
        raise ValueError("Path must be non-empty")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    normalized = raw.replace("\\", "/")
    if normalized.startswith("data/"):
        remainder = normalized.split("/", 1)[1] if "/" in normalized else ""
        return data_root() / remainder
    return project_root() / normalized


def exists(target: str | Path) -> bool:
    is_s3, key = _is_s3_key(target)
    if is_s3:
        storage = _ensure_storage()
        try:
            return storage.object_exists(key)
        except Exception as exc:  # pragma: no cover - best-effort existence check
            LOGGER.debug("Failed to check S3 object %s: %s", key, exc)
            return False
    path = _resolve_local_path(target)
    return path.exists()


def read_json(target: str | Path) -> Dict[str, Any]:
    is_s3, key = _is_s3_key(target)
    if is_s3:
        return _s3_read_json(key)
    path = _resolve_local_path(target)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}") from exc


def write_json(target: str | Path, payload: Dict[str, Any]) -> str | Path:
    is_s3, key = _is_s3_key(target)
    if is_s3:
        s3_write_json(key, payload)
        return key
    path = _resolve_local_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)
    return path


def _s3_read_json(key: str) -> Dict[str, Any]:
    storage = _ensure_storage()
    client = getattr(storage, "_client", None)
    if client is None:
        raise FileNotFoundError(f"S3 backend not configured for key {key}")
    try:
        response = client.get_object(Bucket=storage.bucket, Key=key)  # type: ignore[assignment]
    except Exception as exc:  # pragma: no cover - network errors
        raise FileNotFoundError(f"S3 object {key} not found") from exc
    body = response.get("Body")
    if body is None:
        raise FileNotFoundError(f"S3 object {key} missing body")
    try:
        data = body.read()
    finally:  # pragma: no cover - boto handles closing
        try:
            body.close()
        except Exception:
            # Some boto streams lack close(); ignoring keeps cleanup best-effort.
            pass
    try:
        return json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"S3 object {key} is not valid JSON") from exc


def s3_write_json(key: str, payload: Dict[str, Any]) -> None:
    storage = _ensure_storage()
    if not storage.s3_enabled():
        return
    client = getattr(storage, "_client", None)
    if client is None:
        return
    body = json.dumps(payload, indent=2).encode("utf-8")
    try:
        client.put_object(  # type: ignore[union-attr]
            Bucket=storage.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
    except Exception as exc:  # pragma: no cover - best-effort upload
        LOGGER.warning("Failed to write S3 object %s: %s", key, exc)


__all__ = [
    "data_root",
    "exists",
    "now_iso",
    "project_root",
    "read_json",
    "s3_write_json",
    "use_s3",
    "write_json",
]
