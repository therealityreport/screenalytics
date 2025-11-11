# 2025-11-11 — S3 bucket structure & prefixes

## Summary
- Standardized S3 layout: `raw/videos/`, `artifacts/manifests/`, and `artifacts/faces/` per environment-specific bucket (`screenalytics-<env>-<account>`).
- Added `scripts/s3_bootstrap.sh` to create dev/stg/prod buckets with encryption, versioning, and lifecycle policies.
- `apps/api.services.storage.StorageService` now derives the bucket name, optional prefix, and auto-creates when `S3_AUTO_CREATE=1`.
- Streamlit surfaces the active storage backend and bucket and warns when uploads fail due to `NoSuchBucket`.

## Switching environments
- Local dev: `STORAGE_BACKEND=local` (writes to `data/`).
- MinIO: `STORAGE_BACKEND=minio` with `SCREENALYTICS_OBJECT_STORE_*` env vars.
- AWS S3: `STORAGE_BACKEND=s3`, optional `AWS_S3_BUCKET`, or let it auto-derive from `SCREENALYTICS_ENV` + account ID.
