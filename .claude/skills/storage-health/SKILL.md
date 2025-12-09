# Storage & S3 Health Skill

Use this skill for storage diagnostics and S3 sync issues.

## When to Use

- Thumbnails not loading in UI
- S3 sync failures
- Presigned URL errors
- Disk space concerns
- Missing artifacts

## Storage Architecture

```
STORAGE_BACKEND env var
    ↓
┌─────────────────────────────────────┐
│  local  │  s3    │  minio          │
├─────────┼────────┼─────────────────┤
│ data/   │ S3     │ MinIO (S3-compat)│
└─────────┴────────┴─────────────────┘
```

## Diagnostic Steps

### Step 1: Check Storage Backend

```python
import os
backend = os.environ.get("STORAGE_BACKEND", "local")
print(f"Storage backend: {backend}")
```

### Step 2: Verify Path Structure

S3/MinIO path format:
```
artifacts/{kind}/{show}/s{season}/e{episode}/
```

Example:
```
artifacts/thumbnails/rhobh/s05/e02/track_001_frame_1234.jpg
artifacts/embeddings/rhobh/s05/e02/embeddings.npy
```

### Step 3: Check Presigned URLs

```python
from apps.api.services.storage import get_presigned_url

url = get_presigned_url(
    bucket="screenalytics",
    key="artifacts/thumbnails/rhobh/s05/e02/track_001.jpg"
)

if url:
    print(f"URL generated: {url[:50]}...")
else:
    print("ERROR: Failed to generate presigned URL")
```

### Step 4: Test S3 Connectivity

```python
import boto3

s3 = boto3.client('s3')
try:
    s3.head_bucket(Bucket='screenalytics')
    print("S3 connection OK")
except Exception as e:
    print(f"S3 connection failed: {e}")
```

### Step 5: Check Local Artifacts

```bash
# List episode artifacts
ls -la data/artifacts/{show}/s{season}/e{episode}/

# Check disk usage
du -sh data/artifacts/
```

## Key Files

| File | Purpose |
|------|---------|
| `apps/api/services/storage.py` | Storage abstraction |
| `py_screenalytics/artifacts.py` | Path resolution |
| `config/storage.yaml` | Storage configuration |

## Common Issues

### Thumbnails Not Loading

1. Check presigned URL generation
2. Verify CORS settings on S3 bucket
3. Check file exists at expected path

```python
# Debug thumbnail path
from py_screenalytics.artifacts import get_thumbnail_path

path = get_thumbnail_path(ep_id, track_id, frame_num)
print(f"Expected path: {path}")
```

### S3 Sync Failures

1. Check AWS credentials
2. Verify bucket permissions
3. Check network connectivity

```bash
# Test AWS CLI
aws s3 ls s3://screenalytics/

# Check credentials
aws sts get-caller-identity
```

### MinIO Connection Issues

```python
# MinIO-specific config
import os
os.environ["AWS_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
```

### Disk Space

```bash
# Check available space
df -h data/

# Find large files
find data/ -type f -size +100M -exec ls -lh {} \;

# Clean old artifacts
find data/artifacts -mtime +30 -type f -delete  # CAUTION
```

## S3 Bucket Policy

Required permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::screenalytics",
                "arn:aws:s3:::screenalytics/*"
            ]
        }
    ]
}
```

## CORS Configuration

For presigned URL access from browser:
```json
[
    {
        "AllowedHeaders": ["*"],
        "AllowedMethods": ["GET", "PUT"],
        "AllowedOrigins": ["*"],
        "ExposeHeaders": []
    }
]
```

## Checklist

- [ ] Storage backend identified
- [ ] Path structure verified
- [ ] Presigned URLs working
- [ ] S3/MinIO connectivity tested
- [ ] Local disk space adequate
- [ ] Permissions configured
