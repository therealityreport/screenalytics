# 2025-11-11 — Single S3 bucket (`screenalytics`)

## Summary
- Consolidated uploads into one AWS S3 bucket named `screenalytics` with `raw/` and `artifacts/` prefixes.
- `StorageService` now defaults to that bucket (with optional auto-create) and still supports MinIO/local backends.
- Streamlit sidebar shows the active backend + bucket and success messages reference “object storage”.
- README/.env updated so the bootstrap + verification steps describe the single bucket flow.
