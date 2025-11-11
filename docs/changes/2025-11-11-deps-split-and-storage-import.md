# 2025-11-11 — Dependency split & storage import guard

## What changed
- Added `requirements-core.txt` and `requirements-ml.txt`, with `requirements.txt` now referencing the lightweight core profile by default.
- Storage service lazily imports `boto3`/botocore only when `STORAGE_BACKEND` is `s3|minio`; added a `local` backend for tests and offline mode.
- Tests default to `STORAGE_BACKEND=local` so CI and dev boxes no longer need the object-store SDK when running the API suite.

## Why
PyAV builds pulled in by `faster-whisper` frequently fail on macOS because Homebrew ships FFmpeg 8.x headers. Core API + upload UI + stub flows do not need the ML stack, so splitting dependencies and gating boto3 keeps fresh installs and `pytest` green without heavy/native wheels.

## How to run now
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
pytest -q
uvicorn apps.api.main:app --reload
streamlit run apps/workspace-ui/streamlit_app.py
```

Optional ML stack (not required for tests/UI):
```bash
pip install -r requirements-ml.txt
```
