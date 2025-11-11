# 2025-11-11 — CORS for Streamlit & stub import guard

## Summary
- FastAPI now enables CORS for `http://localhost:8501` (override via `UI_ORIGIN`) so the Streamlit uploader can call the API without proxying.
- `tools/episode_run.py` skips all heavy OpenCV/ML imports when `--stub` is set, emitting placeholder detections/tracks instead of loading RetinaFace/ByteTrack.

## Notes
- Customize origins by exporting `UI_ORIGIN=https://workspace.localhost` before launching the API.
- Stub mode simply mirrors the uploaded video and writes deterministic JSONL manifests; run without `--stub` (and install `requirements-ml.txt`) for the full detection/track pipeline.
