# 2025-11-11 — YOLOv8 + ByteTrack detect/track path

## Highlights
- `tools/episode_run.py` now runs Ultralytics YOLOv8 + ByteTrack when `--stub` is omitted, writing detections/tracks with the public schema (`ts`, `frame_idx`, `class`, `conf`, `bbox_xyxy`, `track_id`, `model`, `tracker`, `pipeline_ver`).
- `POST /jobs/detect_track` and the Streamlit upload UI call the same path; the checkbox now only controls whether to stay on the stub for quick smoke tests.
- Added `tests/ml/test_detect_track_real.py`, skipped unless `RUN_ML_TESTS=1`, to assert that the real pipeline generates >0 detections and tracks for a synthetic clip.

## Why it matters
Teams can run the first production-grade stage (detect + track) right after uploading footage instead of relying on the stub. The manifests produced via CLI, API, or UI now share a stable schema suitable for downstream automation or diffs against the stub output.

## How to use it
```bash
# Install ML extras only when you need the real pipeline
pip install -r requirements-ml.txt

# Run detection+tracking locally
python tools/episode_run.py --ep-id ep_demo --video samples/demo.mp4 --stride 3 --fps 8
# Add --stub for the classic stub path if you just need smoke-test manifests

# Optional: gate the slow ML test behind RUN_ML_TESTS
RUN_ML_TESTS=1 pytest tests/ml/test_detect_track_real.py -q
```

Tweak `--stride` and `--fps` for the speed/recall trade-off (lower stride or higher FPS → more compute, but better recall). The UI surfaces the same toggle via **Use stub (fast, no ML)**.
