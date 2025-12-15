# NOV-17 Zero Tracks + ByteTrack Diagnostics

## Root cause & immediate fix
- ByteTrack rejected otherwise valid detections because the default `track_high_thresh`/`new_track_thresh` (0.6) exceeded the RetinaFace detection threshold (≈0.5), so no tracks could form even when detections existed.
- The detect/track stage now lowers the default ByteTrack gates to 0.5, scales `track_buffer` by stride, and validates runs that yield detections but no tracks. When `tracks_total == 0` the CLI logs a high-severity diagnostic with the effective ByteTrack config (ep id, counts, stride, thresholds, match/min-box/track-buffer) so zero-track runs are obvious without silently failing.

## Diagnostics & observability
- `tools/episode_run.py`
  - Emits per-frame `detect_track_stats` in `progress.json`/SSE (frames_seen, detections_seen, detections/tracks per frame, trackers born/alive/lost, tracker config) and logs a digest every 100 sampled frames.
  - Adds detection confidence histograms (`detection_confidence_histogram`), `scene_cuts_total`, and `scene_cuts_per_1k_frames` to run summaries.
  - Includes `detect_track_stats`, `detections_total`, `tracks_total`, and `track_to_detection_ratio` in job summaries so downstream clients can render ratios/warnings.
  - Final progress completion events now carry the latest `detect_track_stats` snapshot for API/UI consumers.
- `apps/workspace-ui/pages/2_Episode_Detail.py`
  - Surface the track/detection ratio in the pipeline status panel and in post-run flash messages. shows a warning when ratio < 0.10 with guidance to lower thresholds or rerun.

## Configurability (API + env + UI)
- `apps/api/routers/jobs.py`, `apps/api/services/jobs.py`
  - Detect/track requests accept optional `track_high_thresh`, `new_track_thresh`, `track_buffer`, and `min_box_area`. JobService resolves config using the request → `SCREENALYTICS_*` env vars → defaults chain, and always forwards explicit CLI flags so async jobs persist the actual values in `requested` metadata.
- `apps/workspace-ui/pages/2_Episode_Detail.py`
  - Adds an “Advanced tracking” section (ByteTrack-only) exposing sliders for `track_high_thresh` and `new_track_thresh` (defaults loaded from the last job/env). Values flow into job payloads so operators can tune gates per episode.
- `apps/workspace-ui/ui_helpers.py`
  - New helpers/constants (`TRACK_HIGH_THRESH_DEFAULT`, `TRACK_NEW_THRESH_DEFAULT`, `coerce_float`, etc.) keep UI defaults aligned with server env.

## ByteTrack runtime changes
- `tools/episode_run.py`
  - Introduces `ByteTrackRuntimeConfig` to hold gates, min box area, and base track buffer. Buffer now scales with stride (≈constant temporal window) and values can be overridden via CLI/API/env.
  - Zero-track validation happens before manifest writes; counts remain true-to-life but the log clearly states when ByteTrack produced no tracks.

## Progress artifacts & scene cuts
- `tools/episode_run.py` now writes `scene_cuts_total`/`scene_cuts_per_1k_frames` and attaches the tracker config + diagnostics path to the metrics artifact so post-run tooling can inspect threshold usage.

## Episode Detail manifest-aware completion
- The Episode Detail page no longer depends solely on `/episodes/{ep_id}/status` for Detect/Track completion. A new helper evaluates the status payload plus the local manifests (`detections.jsonl` + `tracks.jsonl`):
  - When the status API reports `success`, the UI trusts it and Faces Harvest is enabled.
  - If the status entry is missing/stale but both manifests exist with data, the UI infers completion, enables Faces Harvest, and shows the caption “Detect/Track completion inferred from manifests (status API missing/stale).”
  - If manifests are missing/empty, Detect/Track remains “not started/unknown” and Faces Harvest stays disabled.
- Manifest fallback never triggers unless valid tracks exist; zero-track runs are still treated as failures. The ByteTrack zero-track logs plus the `track_to_detection_ratio` (<0.10 warning) continue to surface pathological runs even when manifests exist.
- Inferred completion is purely a UI affordance; operators still see the diagnostic caption and can rerun Detect/Track if something looks wrong.

## Files touched
- `tools/episode_run.py`
- `apps/api/routers/jobs.py`
- `apps/api/services/jobs.py`
- `apps/workspace-ui/ui_helpers.py`
- `apps/workspace-ui/pages/2_Episode_Detail.py`
