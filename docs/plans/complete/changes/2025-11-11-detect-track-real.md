# 2025-11-11 — YOLOv8 + ByteTrack detect/track path

## Highlights
- `tools/episode_run.py` runs Ultralytics YOLOv8 + ByteTrack end to end, writing detections/tracks with the public schema (`ts`, `frame_idx`, `class`, `conf`, `bbox_xyxy`, `track_id`, `model`, `tracker`, `pipeline_ver`).
- `POST /jobs/detect_track` and the Streamlit upload UI call the same path so every run executes the real pipeline.
- Added `tests/ml/test_detect_track_real.py`, skipped unless `RUN_ML_TESTS=1`, to assert that the real pipeline generates >0 detections and tracks for a synthetic clip.

## Why it matters
Teams can run the first production-grade stage (detect + track) right after uploading footage. The manifests produced via CLI, API, or UI now share a stable schema suitable for downstream automation or diffs across runs.

## How to use it
```bash
# Install ML extras only when you need the real pipeline
pip install -r requirements-ml.txt

# Run detection+tracking locally
python tools/episode_run.py --ep-id ep_demo --video samples/demo.mp4 --stride 3 --fps 8
# Optional: gate the slow ML test behind RUN_ML_TESTS
RUN_ML_TESTS=1 pytest tests/ml/test_detect_track_real.py -q
```

## Runtime diagnostics (marker fields)
`runs/detect_track.json` now records wall-clock vs video runtime (`detect_wall_time_s`, `rtf`, `effective_fps_processing`) plus stride accounting (`frames_scanned_total`, `face_detect_frames_processed`, `stride_effective`, `stride_observed_median`). If `face_detect_frames_processed` exceeds `ceil(frames_scanned_total/stride_effective)`, check `face_detect_frames_processed_forced_scene_warmup` (extra frames sampled after scene cuts). Tracker execution is also auditable via `tracker_backend_configured`, `tracker_backend_actual`, and `tracker_fallback_reason` (same pattern in `runs/body_tracking.json`).

Tweak `--stride` and `--fps` for the speed/recall trade-off (lower stride or higher FPS → more compute, but better recall). The UI now always runs the real pipeline.
