"""
Artifact Storage Helper for S3/MinIO.

Provides unified interface for downloading and uploading artifacts
to S3-compatible storage (AWS S3, MinIO, etc.).

Environment Variables:
    ARTIFACTS_ENDPOINT: S3 endpoint URL (optional, default: AWS S3)
    ARTIFACTS_ACCESS_KEY: AWS access key ID
    ARTIFACTS_SECRET_KEY: AWS secret access key
    ARTIFACTS_REGION: AWS region (default: us-east-1)
    ARTIFACTS_BUCKET: Default bucket name
"""

import logging
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Base exception for storage operations."""
    pass


class CredentialsError(StorageError):
    """Raised when credentials are missing or invalid."""
    pass


class ArtifactsStore:
    """S3/MinIO compatible artifact storage."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
        bucket: Optional[str] = None,
        use_ssl: bool = True,
    ):
        self.endpoint = endpoint or os.environ.get("ARTIFACTS_ENDPOINT")
        self.access_key = access_key or os.environ.get("ARTIFACTS_ACCESS_KEY")
        self.secret_key = secret_key or os.environ.get("ARTIFACTS_SECRET_KEY")
        self.region = region or os.environ.get("ARTIFACTS_REGION", "us-east-1")
        self.bucket = bucket or os.environ.get("ARTIFACTS_BUCKET", "screenalytics-artifacts")
        self.use_ssl = use_ssl if use_ssl is not None else (
            os.environ.get("ARTIFACTS_USE_SSL", "true").lower() == "true"
        )

        self._client = None
        self._available = None

    @property
    def is_available(self) -> bool:
        """Check if storage is available (credentials configured)."""
        if self._available is not None:
            return self._available

        if not self.access_key or not self.secret_key:
            logger.debug("Storage credentials not configured")
            self._available = False
            return False

        try:
            self._get_client()
            self._available = True
        except Exception as e:
            logger.warning(f"Storage not available: {e}")
            self._available = False

        return self._available

    def _get_client(self):
        """Get or create the boto3 S3 client."""
        if self._client is not None:
            return self._client

        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise StorageError("boto3 not installed. Install with: pip install boto3")

        config = Config(
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=5,
            read_timeout=30,
        )

        client_kwargs = {
            "service_name": "s3",
            "aws_access_key_id": self.access_key,
            "aws_secret_access_key": self.secret_key,
            "region_name": self.region,
            "config": config,
        }

        if self.endpoint:
            client_kwargs["endpoint_url"] = self.endpoint
            client_kwargs["use_ssl"] = self.use_ssl

        self._client = boto3.client(**client_kwargs)
        return self._client

    def parse_uri(self, uri: str) -> tuple:
        """Parse an S3 URI into bucket and key."""
        if uri.startswith("s3://") or uri.startswith("minio://"):
            parsed = urlparse(uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
        elif "/" in uri:
            parts = uri.split("/", 1)
            if len(parts) == 2 and not parts[0].startswith("."):
                bucket, key = parts
            else:
                bucket = self.bucket
                key = uri
        else:
            bucket = self.bucket
            key = uri

        return bucket, key

    def exists(self, remote_uri: str) -> bool:
        """Check if an object exists in storage."""
        if not self.is_available:
            return False

        bucket, key = self.parse_uri(remote_uri)

        try:
            client = self._get_client()
            client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    def download_if_exists(
        self,
        remote_uri: str,
        local_path: str,
        verify_hash: bool = False,
    ) -> bool:
        """Download artifact if it exists in remote storage."""
        if not self.is_available:
            logger.debug("Storage not available, skipping download")
            return False

        bucket, key = self.parse_uri(remote_uri)
        local_path = Path(local_path)

        try:
            client = self._get_client()

            try:
                client.head_object(Bucket=bucket, Key=key)
            except Exception:
                logger.debug(f"Object not found: {remote_uri}")
                return False

            local_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading {remote_uri} to {local_path}")
            client.download_file(bucket, key, str(local_path))

            logger.info(f"Downloaded successfully: {local_path}")
            return True

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def upload_file(
        self,
        remote_uri: str,
        local_path: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Upload a file to remote storage."""
        if not self.is_available:
            raise CredentialsError("Storage credentials not configured.")

        bucket, key = self.parse_uri(remote_uri)
        local_path = Path(local_path)

        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        try:
            client = self._get_client()

            extra_args = {}
            if metadata:
                extra_args["Metadata"] = {k: str(v) for k, v in metadata.items()}

            logger.info(f"Uploading {local_path} to {remote_uri}")
            client.upload_file(
                str(local_path),
                bucket,
                key,
                ExtraArgs=extra_args if extra_args else None,
            )

            logger.info(f"Uploaded successfully: {remote_uri}")
            return True

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise StorageError(f"Upload failed: {e}") from e

    def delete(self, remote_uri: str) -> bool:
        """Delete an object from storage."""
        if not self.is_available:
            return False

        bucket, key = self.parse_uri(remote_uri)

        try:
            client = self._get_client()
            client.delete_object(Bucket=bucket, Key=key)
            logger.info(f"Deleted: {remote_uri}")
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False


# Module-level convenience
_default_store: Optional[ArtifactsStore] = None


def get_store() -> ArtifactsStore:
    """Get the default artifact store instance."""
    global _default_store
    if _default_store is None:
        _default_store = ArtifactsStore()
    return _default_store


def download_if_exists(remote_uri: str, local_path: str) -> bool:
    """Download artifact if it exists. Convenience wrapper."""
    return get_store().download_if_exists(remote_uri, local_path)


def upload_file(remote_uri: str, local_path: str) -> bool:
    """Upload file to storage. Convenience wrapper."""
    return get_store().upload_file(remote_uri, local_path)


def storage_available() -> bool:
    """Check if storage is available. Convenience wrapper."""
    return get_store().is_available
