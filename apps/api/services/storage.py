"""Object storage helpers for presigned uploads and artifact sync."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
LOGGER = logging.getLogger(__name__)


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

        if self.backend == "s3":
            boto3_mod = _boto3()
            from botocore.exceptions import ClientError  # type: ignore

            client_kwargs: Dict[str, object] = {"region_name": self.region}
            custom_endpoint = os.environ.get("SCREENALYTICS_OBJECT_STORE_ENDPOINT")
            if custom_endpoint:
                client_kwargs["endpoint_url"] = custom_endpoint
            self._client = boto3_mod.client("s3", **client_kwargs)
            self.bucket = os.environ.get("AWS_S3_BUCKET", DEFAULT_BUCKET)
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

        if self.backend not in {"s3", "minio"} or self._client is None:
            return 0
        if not local_dir.exists() or not local_dir.is_dir():
            return 0
        s3_prefix = s3_prefix.rstrip("/") + "/"
        uploaded = 0
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            key = f"{s3_prefix}{rel}" if rel else s3_prefix.rstrip("/")
            try:
                self._client.upload_file(str(path), self.bucket, key)  # type: ignore[union-attr]
                uploaded += 1
            except Exception as exc:  # pragma: no cover - best-effort upload
                LOGGER.warning(
                    "Failed to sync artifact %s to s3://%s/%s: %s",
                    path,
                    self.bucket,
                    key,
                    exc,
                )
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


__all__ = [
    "EpisodeContext",
    "PresignedUpload",
    "StorageService",
    "artifact_prefixes",
    "episode_context_from_id",
    "parse_v2_episode_key",
]
