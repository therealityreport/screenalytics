# Run-scoped status groundwork (repo map)

## Branching + mainline
- Mainline branch: `origin/main` (no `nov-18` branch present).
- Feature branch: `screentime-improvements/run-scoped-observability`.
- Base branch: `screentime-improvements/episode-details-downstream-stage`.

## run_id generation + format
- `py_screenalytics/run_layout.py`: `generate_run_id()` uses UUID4 hex (32 chars).
- `py_screenalytics/run_layout.py`: `generate_attempt_run_id()` returns `AttemptN_YYYY-MM-DD_HHMMSS_EST` with attempt number based on existing runs.
- `py_screenalytics/run_layout.py`: `normalize_run_id()` enforces `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`.

## Run-scoped artifact layout
- Local run-scoped artifacts live under `data/manifests/{ep_id}/runs/{run_id}/` (`py_screenalytics/run_layout.py`).
- Legacy (non-run-scoped) manifests live under `data/manifests/{ep_id}/` (`py_screenalytics/run_layout.py`).
- Run-scoped markers (phase JSON) live under `data/manifests/{ep_id}/runs/{run_id}/{phase}.json`.
- Legacy markers live under `data/manifests/{ep_id}/runs/{phase}.json`.
- S3 run layout lives under `runs/{show}/s{ss}/e{ee}/{run_id}/` (canonical) or `runs/{ep_id}/{run_id}/` (legacy), resolved in `py_screenalytics/run_layout.py`.

## Current stage plan + naming keys
- Stage plan used in UI/autorun/pipeline:
  - `apps/workspace-ui/episode_detail_layout.py`: `PIPELINE_STAGE_PLAN = (detect, faces, cluster, body_tracking, track_fusion, pdf)`
  - `apps/workspace-ui/pages/2_Episode_Detail.py`: `_SETUP_STAGE_PLAN = (detect, faces, cluster, body_tracking, track_fusion, pdf)`
  - `py_screenalytics/autorun_plan.py`: `build_autorun_stage_plan()` returns `detect → faces → cluster → body_tracking → track_fusion → pdf`
  - `py_screenalytics/episode_status.py`: `STAGE_PLAN` matches the same keys
- Common aliases across UI/workers:
  - Detect: `detect`, `detect_track`, `detect/track`.
  - Faces: `faces`, `faces_embed`, `faces harvest`.
  - Track fusion: `track_fusion`, `body_tracking_fusion`.
  - PDF: `pdf`, `pdf_export`.
  - Optional in export contexts: `screentime` / `screen_time` (see `apps/api/services/run_export.py`).

## Current status sources (writers)
- `tools/episode_run.py` writes phase markers via `_write_run_marker(...)` for detect/track, faces_embed, cluster, body_tracking, track_fusion.
- `_write_run_marker(...)` writes **both** legacy and run-scoped markers and calls `_update_episode_status_from_marker(...)`.
- `_update_episode_status_from_marker(...)` uses `py_screenalytics/episode_status.stage_update_from_marker()` + `update_episode_status()` to update `episode_status.json`.
- `tools/episode_run.py` `StageStatusHeartbeat` writes progress + timestamps into `episode_status.json` during long-running stages.

## Current status sources (readers)
- Episode Detail UI reads API status (`/episodes/{ep_id}/status`) via `apps/workspace-ui/ui_helpers.py:get_episode_status()`.
- Episode Detail UI also reads the run-scoped status file directly via `apps/workspace-ui/pages/2_Episode_Detail.py:_cached_episode_status_file()` → `{run_root}/episode_status.json`.
- Export/diagnostics read the file in `apps/api/services/run_export.py`.

## Stage cards + stage plan rendering
- Stage plan + labels + aliases are defined in `apps/workspace-ui/episode_detail_layout.py`.
- Stage cards and downstream UI wiring live in `apps/workspace-ui/pages/2_Episode_Detail.py`.
