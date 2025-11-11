"""Object storage helpers for presigned uploads and artifact sync."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from py_screenalytics.artifacts import get_path

DEFAULT_BUCKET = "screenalytics"
DEFAULT_REGION = "us-east-1"
DEFAULT_EXPIRY = 900  # 15 minutes
LOCAL_UPLOAD_BASE = "http://localhost/_local-storage"
ARTIFACT_ROOT = "artifacts"
_V2_KEY_RE = re.compile(
    r"raw/videos/(?P<show>[^/]+)/s(?P<season>\d{2})/e(?P<episode>\d{2})/episode\.mp4"
)
_V1_KEY_RE = re.compile(r"raw/videos/(?P<ep_id>[^/]+)/episode\.mp4")
_EP_ID_REGEX = re.compile(r"^(?P<show>.+)-s(?P<season>\d{2})e(?P<episode>\d{2})$", re.IGNORECASE)
_FRAME_NAME_RE = re.compile(r"frame_(\d{6})\.jpg$", re.IGNORECASE)
_CURSOR_SEP = "|"
LOGGER = logging.getLogger(__name__)


def _frame_idx_from_key(key: str) -> int | None:
    match = _FRAME_NAME_RE.search(key)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _split_cursor(raw: str | None) -> tuple[str | None, int]:
    if not raw:
        return None, 0
    value = raw.strip()
    if not value:
        return None, 0
    if _CURSOR_SEP in value:
        key, _, remainder = value.partition(_CURSOR_SEP)
        try:
            cycle = int(remainder)
        except ValueError:
            cycle = 0
        return key or None, max(cycle, 0)
    return value, 0


def _encode_cursor(key: str, cycle: int) -> str:
    cycle = max(int(cycle), 0)
    return f"{key}{_CURSOR_SEP}{cycle}" if cycle else key


def _normalize_cursor_key(track_prefix: str, raw_key: str | None) -> str | None:
    if not raw_key:
        return None
    candidate = raw_key.strip()
    if not candidate:
        return None
    candidate = candidate.replace("\\", "/")
    marker = track_prefix
    idx = candidate.find(marker)
    if idx >= 0:
        candidate = candidate[idx:]
    if candidate.startswith(marker):
        return candidate
    leaf = candidate.split("/")[-1]
    if not leaf:
        return None
    return f"{marker}{leaf}"


def _boto3():
    try:
        import boto3  # type: ignore
        return boto3
    except ImportError as exc:  # pragma: no cover - only triggered in misconfig
        raise RuntimeError(
            "boto3 is required when STORAGE_BACKEND is 's3' or 'minio'"
        ) from exc


@dataclass(frozen=True)
class EpisodeContext:
    ep_id: str
    show_slug: str
    season_number: int
    episode_number: int


@dataclass
class PresignedUpload:
    ep_id: str
    bucket: str
    object_key: str
    upload_url: str | None
    expires_in: int | None
    headers: Dict[str, str]
    method: str
    path: str | None = None
    backend: str = "s3"


def episode_context_from_id(ep_id: str) -> EpisodeContext:
    match = _EP_ID_REGEX.match(ep_id)
    if not match:
        raise ValueError(f"Unable to parse episode id '{ep_id}' into show/season/episode")
    show = match.group("show")
    season = int(match.group("season"))
    episode = int(match.group("episode"))
    return EpisodeContext(ep_id=ep_id, show_slug=show, season_number=season, episode_number=episode)


def artifact_prefixes(ep_ctx: EpisodeContext) -> Dict[str, str]:
    """Return v2 S3 prefixes for frames/crops/manifests under single bucket."""

    show = ep_ctx.show_slug
    season = ep_ctx.season_number
    episode = ep_ctx.episode_number
    return {
        "frames": f"{ARTIFACT_ROOT}/frames/{show}/s{season:02d}/e{episode:02d}/frames/",
        "crops": f"{ARTIFACT_ROOT}/crops/{show}/s{season:02d}/e{episode:02d}/tracks/",
        "manifests": f"{ARTIFACT_ROOT}/manifests/{show}/s{season:02d}/e{episode:02d}/",
        "thumbs_tracks": f"{ARTIFACT_ROOT}/thumbs/{show}/s{season:02d}/e{episode:02d}/tracks/",
        "thumbs_identities": f"{ARTIFACT_ROOT}/thumbs/{show}/s{season:02d}/e{episode:02d}/identities/",
    }


def parse_v2_episode_key(key: str) -> Dict[str, object] | None:
    """Parse a v2 episode key (raw/videos/{show}/s{ss}/e{ee}/episode.mp4)."""

    match = _V2_KEY_RE.search(key)
    if not match:
        return None
    show = match.group("show")
    season = int(match.group("season"))
    episode = int(match.group("episode"))
    ep_id = f"{show.lower()}-s{season:02d}e{episode:02d}"
    return {
        "ep_id": ep_id,
        "show": show,
        "show_slug": show,
        "season": season,
        "episode": episode,
        "key_version": "v2",
    }


class StorageService:
    """Lightweight S3/MinIO helper that only presigns uploads."""

    def __init__(self) -> None:
        self.backend = os.environ.get("STORAGE_BACKEND", "s3").lower()
        self.region = os.environ.get("AWS_DEFAULT_REGION", DEFAULT_REGION)
        self.prefix = os.environ.get("AWS_S3_PREFIX", "raw/")
        if self.prefix and not self.prefix.endswith("/"):
            self.prefix += "/"
        auto_create = os.environ.get("S3_AUTO_CREATE", "0")
        self.auto_create = auto_create.lower() in {"1", "true", "yes"}
        self.bucket = DEFAULT_BUCKET
        self._client = None
        self._client_error_cls = None
        self.write_enabled = True

        if self.backend == "s3":
            boto3_mod = _boto3()
            from botocore.exceptions import ClientError  # type: ignore

            client_kwargs: Dict[str, object] = {"region_name": self.region}
            custom_endpoint = os.environ.get("SCREENALYTICS_OBJECT_STORE_ENDPOINT")
            if custom_endpoint:
                client_kwargs["endpoint_url"] = custom_endpoint
            self._client = boto3_mod.client("s3", **client_kwargs)
            configured_bucket = os.environ.get("SCREENALYTICS_S3_BUCKET") or os.environ.get("AWS_S3_BUCKET")
            self.bucket = configured_bucket or DEFAULT_BUCKET
            self._client_error_cls = ClientError
            self._ensure_s3_bucket(ClientError)
        elif self.backend == "minio":
            boto3_mod = _boto3()
            from botocore.client import Config  # type: ignore
            from botocore.exceptions import ClientError  # type: ignore

            endpoint = os.environ.get("SCREENALYTICS_OBJECT_STORE_ENDPOINT", "http://localhost:9000")
            access_key = os.environ.get("SCREENALYTICS_OBJECT_STORE_ACCESS_KEY", "minio")
            secret_key = os.environ.get("SCREENALYTICS_OBJECT_STORE_SECRET_KEY", "miniosecret")
            signature_version = os.environ.get("SCREENALYTICS_OBJECT_STORE_SIGNATURE", "s3v4")
            minio_region = os.environ.get("SCREENALYTICS_OBJECT_STORE_REGION", DEFAULT_REGION)
            self.bucket = os.environ.get("SCREENALYTICS_OBJECT_STORE_BUCKET", DEFAULT_BUCKET)
            self._client = boto3_mod.client(
                "s3",
                endpoint_url=endpoint,
                region_name=minio_region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=Config(signature_version=signature_version),
            )
            self._client_error_cls = ClientError
        elif self.backend == "local":
            self.bucket = "local"
        else:
            raise ValueError(f"Unsupported STORAGE_BACKEND '{self.backend}'")

        flag = os.environ.get("S3_WRITE")
        default_enabled = self.backend in {"s3", "minio"} and self._client is not None
        if flag is not None:
            default_enabled = flag.lower() in {"1", "true", "yes"}
        self.write_enabled = default_enabled and self.backend in {"s3", "minio"} and self._client is not None

    def s3_enabled(self) -> bool:
        return self.backend in {"s3", "minio"} and self._client is not None

    def presign_episode_video(
        self,
        ep_id: str,
        *,
        object_key: str,
        content_type: str = "video/mp4",
        expires_in: int = DEFAULT_EXPIRY,
    ) -> PresignedUpload:
        headers = {"Content-Type": content_type}

        if self.backend == "local":
            upload_url = f"{LOCAL_UPLOAD_BASE}/{object_key}"
            method = "FILE"
            path = object_key
        else:
            assert self._client is not None  # for mypy
            params = {"Bucket": self.bucket, "Key": object_key, "ContentType": content_type}
            upload_url = self._client.generate_presigned_url(
                "put_object",
                Params=params,
                ExpiresIn=expires_in,
                HttpMethod="PUT",
            )
            method = "PUT"
            path = None

        return PresignedUpload(
            ep_id=ep_id,
            bucket=self.bucket,
            object_key=object_key,
            upload_url=upload_url,
            expires_in=expires_in,
            headers=headers,
            method=method,
            path=path,
            backend=self.backend,
        )

    def presign_get(self, key: str, expires_in: int = 3600) -> str | None:
        if self.backend not in {"s3", "minio"} or self._client is None:
            return None
        params = {"Bucket": self.bucket, "Key": key}
        return self._client.generate_presigned_url(  # type: ignore[union-attr]
            "get_object",
            Params=params,
            ExpiresIn=expires_in,
        )

    def ensure_local_mirror(
        self,
        ep_id: str,
        *,
        show_ref: str | None = None,
        season_number: int | None = None,
        episode_number: int | None = None,
    ) -> Dict[str, Optional[object]]:
        local_path = get_path(ep_id, "video")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        info: Dict[str, Optional[object]] = {
            "local_video_path": str(local_path),
            "bytes": local_path.stat().st_size if local_path.exists() else None,
            "etag": None,
            "used_key_version": None,
        }
        if self.backend == "local":
            return info
        if self.backend not in {"s3", "minio"} or self._client is None or self._client_error_cls is None:
            return info
        keys_to_try: List[tuple[str, str]] = []
        if show_ref is not None and season_number is not None and episode_number is not None:
            keys_to_try.append(("v2", self.video_object_key_v2(show_ref, season_number, episode_number)))
        keys_to_try.append(("v1", self.video_object_key_v1(ep_id)))

        for version, key in keys_to_try:
            try:
                head = self._client.head_object(Bucket=self.bucket, Key=key)
                etag = head.get("ETag")
                info["etag"] = etag.strip('"') if isinstance(etag, str) else etag
                info["bytes"] = head.get("ContentLength")
                if not local_path.exists():
                    self._client.download_file(self.bucket, key, str(local_path))
                    info["bytes"] = local_path.stat().st_size
                info["used_key_version"] = version
                return info
            except self._client_error_cls as exc:  # type: ignore[misc]
                error_code = exc.response.get("Error", {}).get("Code") if hasattr(exc, "response") else None
                if error_code in {"404", "NoSuchKey", "NotFound"}:
                    continue
                raise
        raise RuntimeError("Episode video not found in S3 (checked v2 then v1)")
        return info

    def object_exists(self, key: str) -> bool:
        if self.backend not in {"s3", "minio"} or self._client is None or self._client_error_cls is None:
            return False
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self._client_error_cls as exc:  # type: ignore[misc]
            error_code = exc.response.get("Error", {}).get("Code") if hasattr(exc, "response") else None
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def video_object_key_v1(self, ep_id: str) -> str:
        prefix = self.prefix if self.backend == "s3" else ""
        return f"{prefix}videos/{ep_id}/episode.mp4"

    def video_object_key_v2(self, show_slug: str, season: int, episode: int) -> str:
        prefix = self.prefix if self.backend == "s3" else ""
        return f"{prefix}videos/{show_slug}/s{season:02d}/e{episode:02d}/episode.mp4"

    # ------------------------------------------------------------------
    def put_artifact(self, ep_ctx: EpisodeContext, kind: str, local_path: Path, key_rel: str) -> bool:
        """Upload a single artifact file under the v2 hierarchy."""

        if self.backend not in {"s3", "minio"} or self._client is None:
            return False
        if not local_path.exists():
            LOGGER.debug("Artifact %s missing; skipping upload", local_path)
            return False
        prefixes = artifact_prefixes(ep_ctx)
        prefix = prefixes.get(kind)
        if prefix is None:
            raise ValueError(f"Unknown artifact kind '{kind}'")
        key_rel = key_rel.lstrip("/\ ")
        key = f"{prefix}{key_rel}" if prefix.endswith("/") else f"{prefix}/{key_rel}"
        try:
            self._client.upload_file(str(local_path), self.bucket, key)  # type: ignore[union-attr]
            return True
        except Exception as exc:  # pragma: no cover - best-effort upload
            LOGGER.warning("Failed to upload artifact %s to s3://%s/%s: %s", local_path, self.bucket, key, exc)
            return False

    def sync_tree_to_s3(self, ep_ctx: EpisodeContext, local_dir: Path, s3_prefix: str) -> int:
        """Sync an entire directory tree to the configured bucket (best effort)."""

        return self.upload_dir(local_dir, s3_prefix)

    def upload_dir(self, local_dir: Path | str, s3_prefix: str, *, guess_mime: bool = True) -> int:
        if not self.s3_enabled() or not self.write_enabled:
            return 0
        root = Path(local_dir)
        if not root.exists() or not root.is_dir():
            return 0
        prefix = (s3_prefix or "").rstrip("/")
        if not prefix:
            LOGGER.debug("Empty S3 prefix for upload_dir; skipping")
            return 0
        prefix = prefix + "/"
        uploaded = 0
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            key = f"{prefix}{rel}" if rel else prefix.rstrip("/")
            extra = None
            if guess_mime:
                mime, _ = guess_type(str(path))
                if mime:
                    extra = {"ContentType": mime}
            try:
                if extra:
                    self._client.upload_file(str(path), self.bucket, key, ExtraArgs=extra)  # type: ignore[union-attr]
                else:
                    self._client.upload_file(str(path), self.bucket, key)  # type: ignore[union-attr]
                uploaded += 1
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to upload %s to s3://%s/%s: %s", path, self.bucket, key, exc)
        return uploaded

    def list_objects(self, prefix: str, suffix: str | None = None, max_items: int = 1000) -> List[str]:
        """Return up to `max_items` object keys under the provided prefix."""

        if self.backend not in {"s3", "minio"} or self._client is None or max_items <= 0:
            return []
        results: List[str] = []
        continuation_token: str | None = None
        while len(results) < max_items:
            kwargs: Dict[str, object] = {
                "Bucket": self.bucket,
                "Prefix": prefix,
                "MaxKeys": min(1000, max_items - len(results)),
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            response = self._client.list_objects_v2(**kwargs)
            contents: Iterable[Dict[str, object]] = response.get("Contents", [])
            if not contents:
                break
            for obj in contents:
                key = obj.get("Key")
                if not isinstance(key, str):
                    continue
                if suffix and not key.endswith(suffix):
                    continue
                results.append(key)
                if len(results) >= max_items:
                    break
            if len(results) >= max_items:
                break
            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
            if not continuation_token:
                break
        return results

    def delete_prefix(self, prefix: str, *, bucket_override: str | None = None) -> int:
        if self.backend not in {"s3", "minio"} or self._client is None:
            return 0
        normalized = (prefix or "").lstrip("/")
        if not normalized:
            LOGGER.warning("Refusing to delete empty S3 prefix")
            return 0
        bucket = bucket_override or self.bucket
        total_deleted = 0
        continuation_token: str | None = None
        while True:
            kwargs: Dict[str, Any] = {
                "Bucket": bucket,
                "Prefix": normalized,
                "MaxKeys": 1000,
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            response = self._client.list_objects_v2(**kwargs)
            contents: Iterable[Dict[str, Any]] = response.get("Contents", [])
            if not contents:
                break
            batch = [{"Key": obj.get("Key")} for obj in contents if isinstance(obj.get("Key"), str)]
            if not batch:
                break
            self._client.delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})
            total_deleted += len(batch)
            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
            if not continuation_token:
                break
        return total_deleted

    def list_episode_videos_s3(
        self,
        prefix: str = "raw/videos/",
        suffix: str = "/episode.mp4",
        limit: int = 1000,
    ) -> List[Dict[str, object]]:
        if self.backend not in {"s3", "minio"} or self._client is None:
            return []

        results: List[Dict[str, object]] = []
        continuation_token: str | None = None
        fetched = 0
        while True:
            kwargs: Dict[str, object] = {
                "Bucket": self.bucket,
                "Prefix": prefix,
                "MaxKeys": min(1000, limit - fetched),
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            response = self._client.list_objects_v2(**kwargs)
            contents: Iterable[Dict[str, object]] = response.get("Contents", [])
            for obj in contents:
                key = obj.get("Key")
                if not isinstance(key, str):
                    continue
                if not key.startswith(prefix) or not key.endswith(suffix):
                    continue
                meta = self._parse_s3_key_metadata(key)
                results.append(
                    {
                        "bucket": self.bucket,
                        "key": key,
                        **meta,
                        "size": obj.get("Size"),
                        "last_modified": obj.get("LastModified"),
                        "etag": (obj.get("ETag") or '').strip('"'),
                    }
                )
                fetched += 1
                if fetched >= limit:
                    break
            if fetched >= limit:
                break
            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
            if not continuation_token:
                break
        return results

    # ------------------------------------------------------------------
    def list_track_crops(
        self,
        ep_ctx: EpisodeContext,
        track_id: int,
        *,
        sample: int = 1,
        max_keys: int = 500,
        start_after: str | None = None,
    ) -> Dict[str, Any]:
        sample = max(1, int(sample or 1))
        max_keys = max(1, min(int(max_keys or 1), 1000))
        track_prefix = f"track_{max(int(track_id), 0):04d}/"
        cursor_key, cursor_cycle = _split_cursor(start_after)
        cursor_cycle = cursor_cycle % sample
        normalized_cursor = _normalize_cursor_key(track_prefix, cursor_key)
        if self.backend in {"s3", "minio"} and self._client is not None:
            return self._list_remote_track_crops(
                ep_ctx,
                track_prefix,
                sample,
                max_keys,
                normalized_cursor,
                cursor_cycle,
            )
        return self._list_local_track_crops(
            ep_ctx.ep_id,
            track_prefix,
            sample,
            max_keys,
            normalized_cursor,
            cursor_cycle,
        )

    def _list_local_track_crops(
        self,
        ep_id: str,
        track_prefix: str,
        sample: int,
        max_keys: int,
        cursor_key: str | None,
        cursor_cycle: int,
    ) -> Dict[str, Any]:
        crops_root = get_path(ep_id, "frames_root") / "crops"
        track_dir = crops_root / track_prefix.rstrip("/")
        if not track_dir.exists() or not track_dir.is_dir():
            return {"items": [], "next_start_after": None}
        entries = self._load_track_index_from_path(track_dir, track_prefix)
        if not entries:
            entries = self._entries_from_files(track_dir, track_prefix)
        filtered: List[Dict[str, Any]] = []
        for entry in entries:
            rel_key = entry["key"]
            abs_path = crops_root / rel_key
            if not abs_path.exists():
                continue
            filtered.append({**entry, "_abs_path": abs_path})
        selected, cursor = self._slice_entries(filtered, sample, max_keys, cursor_key, cursor_cycle)
        items: List[Dict[str, Any]] = []
        for entry in selected:
            abs_path: Path = entry.get("_abs_path")  # type: ignore[assignment]
            items.append(
                {
                    "key": entry["key"],
                    "frame_idx": entry["frame_idx"],
                    "ts": entry.get("ts"),
                    "url": abs_path.as_posix() if abs_path else None,
                }
            )
        return {"items": items, "next_start_after": cursor}

    def _list_remote_track_crops(
        self,
        ep_ctx: EpisodeContext,
        track_prefix: str,
        sample: int,
        max_keys: int,
        cursor_key: str | None,
        cursor_cycle: int,
    ) -> Dict[str, Any]:
        prefixes = artifact_prefixes(ep_ctx)
        crops_prefix = prefixes.get("crops")
        if not crops_prefix:
            return {"items": [], "next_start_after": None}
        base_prefix = crops_prefix.rstrip("/") + "/"
        track_prefix_full = f"{base_prefix}{track_prefix}"
        index_key = f"{track_prefix_full}index.json"
        entries = self._load_remote_track_index(index_key, track_prefix)
        if entries:
            selected, cursor = self._slice_entries(entries, sample, max_keys, cursor_key, cursor_cycle)
            items: List[Dict[str, Any]] = []
            for entry in selected:
                s3_key = f"{base_prefix}{entry['key']}"
                url = self.presign_get(s3_key)
                if not url:
                    continue
                items.append(
                    {
                        "key": entry["key"],
                        "frame_idx": entry["frame_idx"],
                        "ts": entry.get("ts"),
                        "url": url,
                    }
                )
            return {"items": items, "next_start_after": cursor}
        return self._list_remote_without_index(
            base_prefix,
            track_prefix,
            sample,
            max_keys,
            cursor_key,
            cursor_cycle,
        )

    def _load_track_index_from_path(self, track_dir: Path, track_prefix: str) -> List[Dict[str, Any]]:
        index_path = track_dir / "index.json"
        if not index_path.exists():
            return []
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            LOGGER.warning("Invalid track index at %s", index_path)
            return []
        if not isinstance(raw, list):
            return []
        return self._normalize_track_index_entries(raw, track_prefix)

    def _entries_from_files(self, track_dir: Path, track_prefix: str) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        if not track_dir.exists():
            return entries
        for path in sorted(track_dir.glob("frame_*.jpg")):
            key = f"{track_prefix}{path.name}"
            frame_idx = _frame_idx_from_key(key)
            if frame_idx is None:
                continue
            entries.append({"key": key, "frame_idx": frame_idx, "ts": None})
        return entries

    def _normalize_track_index_entries(self, raw_entries: Iterable[Any], track_prefix: str) -> List[Dict[str, Any]]:
        normalized: Dict[str, Dict[str, Any]] = {}
        marker = track_prefix
        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            raw_key = entry.get("key")
            if not isinstance(raw_key, str):
                continue
            candidate = raw_key.strip().replace("\\", "/")
            idx = candidate.find(marker)
            if idx >= 0:
                candidate = candidate[idx:]
            elif candidate:
                leaf = candidate.split("/")[-1]
                candidate = f"{marker}{leaf}" if leaf else candidate
            frame_idx = entry.get("frame_idx")
            if not isinstance(frame_idx, int):
                frame_idx = _frame_idx_from_key(candidate)
                if frame_idx is None:
                    continue
            ts_val = entry.get("ts")
            if ts_val is not None:
                try:
                    ts_val = float(ts_val)
                except (TypeError, ValueError):
                    ts_val = None
            normalized[candidate] = {"key": candidate, "frame_idx": frame_idx, "ts": ts_val}
        ordered = sorted(normalized.values(), key=lambda item: item["frame_idx"])
        return ordered

    def _slice_entries(
        self,
        entries: List[Dict[str, Any]],
        sample: int,
        max_keys: int,
        cursor_key: str | None,
        cursor_cycle: int,
    ) -> tuple[List[Dict[str, Any]], str | None]:
        if not entries:
            return [], None
        start_index = 0
        if cursor_key:
            for idx, entry in enumerate(entries):
                if entry.get("key") == cursor_key:
                    start_index = idx + 1
                    break
        cycle = cursor_cycle % sample
        selected: List[Dict[str, Any]] = []
        next_cursor: str | None = None
        for idx in range(start_index, len(entries)):
            entry = entries[idx]
            include = cycle == 0
            cycle = (cycle + 1) % sample
            if not include:
                continue
            selected.append(entry)
            if len(selected) >= max_keys:
                next_cursor = _encode_cursor(entry["key"], cycle)
                break
        return selected, next_cursor

    def _load_remote_track_index(self, index_key: str, track_prefix: str) -> List[Dict[str, Any]]:
        if self.backend not in {"s3", "minio"} or self._client is None:
            return []
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=index_key)
        except self._client_error_cls as exc:  # type: ignore[misc]
            error_code = exc.response.get("Error", {}).get("Code") if hasattr(exc, "response") else None
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                return []
            raise
        body = response.get("Body")
        if body is None:
            return []
        data = body.read()
        if isinstance(data, bytes):
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                LOGGER.warning("Unable to decode %s as UTF-8", index_key)
                return []
        else:
            text = str(data)
        try:
            raw_entries = json.loads(text)
        except json.JSONDecodeError:
            LOGGER.warning("Invalid JSON in %s", index_key)
            return []
        if not isinstance(raw_entries, list):
            return []
        return self._normalize_track_index_entries(raw_entries, track_prefix)

    def _list_remote_without_index(
        self,
        crops_prefix: str,
        track_prefix: str,
        sample: int,
        max_keys: int,
        cursor_key: str | None,
        cursor_cycle: int,
    ) -> Dict[str, Any]:
        if self.backend not in {"s3", "minio"} or self._client is None:
            return {"items": [], "next_start_after": None}
        normalized_prefix = crops_prefix.rstrip("/") + "/"
        track_root = f"{normalized_prefix}{track_prefix}"
        start_after_key = f"{normalized_prefix}{cursor_key}" if cursor_key else None
        items: List[Dict[str, Any]] = []
        next_cursor: str | None = None
        cycle = cursor_cycle % sample
        continuation_token: str | None = None
        more_available = False
        while len(items) < max_keys:
            remaining = max(max_keys - len(items), 1)
            fetch_budget = min(1000, max(remaining * sample, sample))
            kwargs: Dict[str, Any] = {"Bucket": self.bucket, "Prefix": track_root, "MaxKeys": fetch_budget}
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            elif start_after_key:
                kwargs["StartAfter"] = start_after_key
            response = self._client.list_objects_v2(**kwargs)
            contents: Iterable[Dict[str, Any]] = response.get("Contents", [])
            if not contents:
                break
            for obj in contents:
                key = obj.get("Key")
                if not isinstance(key, str) or not key.endswith(".jpg"):
                    continue
                rel = key[len(normalized_prefix) :]
                if not rel.startswith(track_prefix):
                    continue
                frame_idx = _frame_idx_from_key(rel)
                if frame_idx is None:
                    continue
                include = cycle == 0
                cycle = (cycle + 1) % sample
                if not include:
                    continue
                url = self.presign_get(key)
                if not url:
                    continue
                items.append({"key": rel, "frame_idx": frame_idx, "ts": None, "url": url})
                if len(items) >= max_keys:
                    next_cursor = _encode_cursor(rel, cycle)
                    more_available = True
                    break
            if len(items) >= max_keys or not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
            start_after_key = None
        return {"items": items, "next_start_after": next_cursor if more_available else None}

    def _parse_s3_key_metadata(self, key: str) -> Dict[str, object]:
        parsed_v2 = parse_v2_episode_key(key)
        if parsed_v2:
            return parsed_v2
        match_v1 = _V1_KEY_RE.search(key)
        if match_v1:
            ep_id = match_v1.group("ep_id")
            return {
                "ep_id": ep_id,
                "show": None,
                "season": None,
                "episode": None,
                "key_version": "v1",
            }
        return {"ep_id": "unknown", "key_version": "unknown", "show": None, "season": None, "episode": None}

    def _ensure_s3_bucket(self, client_error_cls) -> None:
        assert self._client is not None
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except client_error_cls as exc:  # pragma: no cover - network interaction
            if self.auto_create:
                create_kwargs = {"Bucket": self.bucket}
                if self.region != "us-east-1":
                    create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self.region}
                self._client.create_bucket(**create_kwargs)
            else:
                raise RuntimeError(
                    f"Bucket {self.bucket} does not exist. Run scripts/s3_bootstrap.sh or set S3_AUTO_CREATE=1"
                ) from exc


def delete_s3_prefix(bucket: str, prefix: str, storage: StorageService | None = None) -> int:
    service = storage or StorageService()
    target_bucket = bucket or service.bucket
    return service.delete_prefix(prefix, bucket_override=target_bucket)


def delete_local_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink()
        return
    shutil.rmtree(path)


def v2_artifact_prefixes(ep_ctx: EpisodeContext) -> Dict[str, str]:
    base = artifact_prefixes(ep_ctx).copy()
    show = ep_ctx.show_slug
    season = ep_ctx.season_number
    episode = ep_ctx.episode_number
    base.setdefault("analytics", f"{ARTIFACT_ROOT}/analytics/{show}/s{season:02d}/e{episode:02d}/")
    base.setdefault("raw_v2", f"raw/videos/{show}/s{season:02d}/e{episode:02d}/")
    base.setdefault("raw_v1", f"raw/videos/{ep_ctx.ep_id}/")
    return base


__all__ = [
    "EpisodeContext",
    "PresignedUpload",
    "StorageService",
    "artifact_prefixes",
    "delete_local_tree",
    "delete_s3_prefix",
    "episode_context_from_id",
    "parse_v2_episode_key",
    "v2_artifact_prefixes",
]
