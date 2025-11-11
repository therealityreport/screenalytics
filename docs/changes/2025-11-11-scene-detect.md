# 2025-11-11 — Scene-cut prepass for detect/track

## Summary
- Added an OpenCV HSV histogram prepass to `tools/episode_run.py` that finds hard scene cuts, emits `scene_detect:*` SSE events, resets ByteTrack/StrongSORT on each cut, and forces `SCENE_WARMUP_DETS` consecutive detections after every reset.
- Plumbed `scene_detect`, `scene_threshold`, `scene_min_len`, and `scene_warmup_dets` through the CLI, job service/router payloads, and the Episode Detail UI (advanced toggle + per-run badge) with env defaults: `SCENE_DETECT=1`, `SCENE_THRESHOLD=0.30`, `SCENE_MIN_LEN=12`, `SCENE_WARMUP_DETS=3`.
- Added coverage: histogram unit test (`tests/ml/test_scene_detect_hist.py`), tracker reset test (`tests/api/test_track_resets_on_cuts.py`), and a UI helper test for the Scene cuts badge.

## Details
1. `detect_scene_cuts()` scans the video once before detection, reporting progress via `scene_detect:start/cut/done` events. `_run_full_pipeline()` now consumes those cut indices, resets the tracker, and forces early detections via the `scene_warmup_dets` window. Summary payloads include `scene_cuts.count` and the CLI exposes `--scene-*` switches (mapped from the new env vars in `.env.example`).
2. API: `DetectTrackRequest` gained the four scene fields; JobService always appends the corresponding flags when launching `episode_run`. The UI payloads inherit the defaults from `ui_helpers` so Quick Run keeps scene detection enabled even without touching the advanced toggle.
3. Episode Detail: the Advanced expander surfaces the toggle + numeric fields, and the completion toast includes a scene-cut badge (driven by `helpers.scene_cuts_badge_text`). Tests ensure the badge string stays stable even under the mocked Streamlit runtime.
4. Docs: README now documents the new CLI flags + UI controls, and `.env.example` lists the env knobs so ops can tune thresholds globally.
