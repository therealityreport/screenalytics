# Artifacts Store

S3/MinIO compatible artifact storage for Screenalytics.

## Overview

The `ArtifactsStore` class provides a unified interface for:
- Downloading artifacts from remote storage
- Uploading artifacts to remote storage
- Checking artifact existence

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARTIFACTS_ENDPOINT` | S3 endpoint URL | AWS S3 |
| `ARTIFACTS_ACCESS_KEY` | AWS access key ID | Required |
| `ARTIFACTS_SECRET_KEY` | AWS secret access key | Required |
| `ARTIFACTS_REGION` | AWS region | `us-east-1` |
| `ARTIFACTS_BUCKET` | Default bucket name | `screenalytics-artifacts` |
| `ARTIFACTS_USE_SSL` | Use SSL for connections | `true` |

### AWS S3 Configuration

```bash
export ARTIFACTS_ACCESS_KEY="AKIAIOSFODNN7EXAMPLE"
export ARTIFACTS_SECRET_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
export ARTIFACTS_REGION="us-east-1"
export ARTIFACTS_BUCKET="my-screenalytics-bucket"
```

### MinIO Configuration

```bash
export ARTIFACTS_ENDPOINT="http://localhost:9000"
export ARTIFACTS_ACCESS_KEY="minioadmin"
export ARTIFACTS_SECRET_KEY="minioadmin"
export ARTIFACTS_USE_SSL="false"
```

## Usage

### Python API

```python
from tools.storage.artifacts_store import ArtifactsStore, get_store

# Using the default store (configured via env vars)
store = get_store()

# Check if storage is available
if store.is_available:
    # Download if exists
    success = store.download_if_exists(
        "models/arcface_r100.onnx",
        "/local/path/arcface_r100.onnx"
    )

    # Upload a file
    store.upload_file(
        "models/my_model.onnx",
        "/local/path/my_model.onnx",
        metadata={"version": "1.0", "precision": "fp16"}
    )
```

### Convenience Functions

```python
from tools.storage.artifacts_store import (
    download_if_exists,
    upload_file,
    storage_available,
)

if storage_available():
    download_if_exists("engines/arcface.trt", "/local/engines/arcface.trt")
```

## URI Formats

Supported URI formats:
- `s3://bucket/key/path` - Full S3 URI
- `minio://bucket/key/path` - MinIO URI (treated same as S3)
- `bucket/key/path` - Bucket and key
- `key/path` - Uses default bucket

## Integration with TensorRT Builder

The TensorRT builder uses the artifacts store for engine caching:

```python
from tools.storage.artifacts_store import ArtifactsStore

store = ArtifactsStore()

# Try to download cached engine
if store.download_if_exists(remote_key, local_path):
    return load_engine(local_path)

# Build engine and upload
engine = build_engine(...)
store.upload_file(remote_key, local_path)
```

## Error Handling

```python
from tools.storage.artifacts_store import (
    ArtifactsStore,
    StorageError,
    CredentialsError,
)

store = ArtifactsStore()

try:
    store.upload_file("key", "/path/to/file")
except CredentialsError:
    print("Storage not configured")
except StorageError as e:
    print(f"Storage error: {e}")
```
