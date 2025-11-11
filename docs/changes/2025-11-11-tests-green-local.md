# 2025-11-11 — Tests green on local backend

## Summary
- Added `tests/conftest.py` to default `STORAGE_BACKEND=local`, so API suites mock S3/MinIO and no longer require `boto3`.
- Lightweight requirements mean `pytest -q` completes in ~3s on Apple Silicon (15 tests covering the new episodes + jobs flows).

## Repro
```bash
source .venv/bin/activate
pip install -r requirements-core.txt
pytest -q
```
Expect: `15 passed`.
