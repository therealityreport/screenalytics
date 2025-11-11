"""Object storage helpers for presigned uploads."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

DEFAULT_BUCKET = "screenalytics"
DEFAULT_ENDPOINT = "http://localhost:9000"
DEFAULT_REGION = "us-east-1"
DEFAULT_EXPIRY = 900  # 15 minutes
LOCAL_UPLOAD_BASE = "http://localhost/_local-storage"


def _boto3():
    try:
        import boto3  # type: ignore
        return boto3
    except ImportError as exc:  # pragma: no cover - only triggered in misconfig
        raise RuntimeError(
            "boto3 is required when STORAGE_BACKEND is 's3' or 'minio'"
        ) from exc


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


class StorageService:
    """Lightweight S3/MinIO helper that only presigns uploads."""

    def __init__(self) -> None:
        self.backend = os.environ.get("STORAGE_BACKEND", "s3").lower()
        self.endpoint = os.environ.get("SCREENALYTICS_OBJECT_STORE_ENDPOINT", DEFAULT_ENDPOINT)
        self.region = os.environ.get("SCREENALYTICS_OBJECT_STORE_REGION", DEFAULT_REGION)
        self.env_name = os.environ.get("SCREENALYTICS_ENV", "dev")
        self.prefix = os.environ.get("AWS_S3_PREFIX", "raw/")
        if self.prefix and not self.prefix.endswith("/"):
            self.prefix += "/"
        auto_create = os.environ.get("S3_AUTO_CREATE", "0")
        self.auto_create = auto_create.lower() in {"1", "true", "yes"}
        self._client = None

        if self.backend in {"s3", "minio"}:
            boto3_mod = _boto3()
            from botocore.client import Config  # type: ignore
            from botocore.exceptions import ClientError  # type: ignore

            access_key = os.environ.get("SCREENALYTICS_OBJECT_STORE_ACCESS_KEY", "minio")
            secret_key = os.environ.get("SCREENALYTICS_OBJECT_STORE_SECRET_KEY", "miniosecret")
            signature_version = os.environ.get("SCREENALYTICS_OBJECT_STORE_SIGNATURE", "s3v4")
            self._client = boto3_mod.client(
                "s3",
                endpoint_url=self.endpoint,
                region_name=self.region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=Config(signature_version=signature_version),
            )

            if self.backend == "s3":
                env_bucket = os.environ.get("AWS_S3_BUCKET")
                if env_bucket:
                    self.bucket = env_bucket
                else:
                    sts_client = boto3_mod.client("sts")
                    account_id = sts_client.get_caller_identity().get("Account", "dev")
                    self.bucket = f"screenalytics-{self.env_name}-{account_id}"
                self._ensure_s3_bucket(ClientError)
            else:  # minio
                self.bucket = os.environ.get("SCREENALYTICS_OBJECT_STORE_BUCKET", DEFAULT_BUCKET)
        elif self.backend == "local":
            self.bucket = "local"
        else:
            raise ValueError(f"Unsupported STORAGE_BACKEND '{self.backend}'")

    def presign_episode_video(
        self,
        ep_id: str,
        *,
        content_type: str = "video/mp4",
        expires_in: int = DEFAULT_EXPIRY,
    ) -> PresignedUpload:
        prefix = self.prefix if self.backend == "s3" else ""
        object_key = f"{prefix}videos/{ep_id}/episode.mp4"
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
        )

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


__all__ = ["PresignedUpload", "StorageService"]
