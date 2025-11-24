# 2025-11-11 — Backend-aware presign tests

## Highlights
- `tests/api/test_episodes_presign.py` and `tests/api/test_presign_headers.py` now branch their expectations based on the response method so local FILE uploads assert local paths while PUT uploads pin the S3 bucket/prefix.
- Added `tests/api/test_presign_matrix.py` to exercise both local and S3 configurations via monkeypatched storage, ensuring the API surfaces the right metadata no matter the backend.

## Why it matters
Previous tests assumed every environment used the S3 bucket `screenalytics`, which broke under the default local-storage profile. Updating the assertions keeps CI green while still verifying that presign responses include the correct bucket, prefix, path, and headers for each backend.

## How to use it
`pytest tests/api/test_episodes_presign.py tests/api/test_presign_headers.py tests/api/test_presign_matrix.py -q`
