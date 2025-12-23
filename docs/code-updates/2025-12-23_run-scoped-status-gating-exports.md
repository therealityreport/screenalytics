# Run-scoped status groundwork (repo map)

## Branching + mainline
- Mainline branch: `origin/main` (no `nov-18` branch present).
- Feature branch: `screentime-improvements/run-scoped-observability`.
- Base branch: `screentime-improvements/episode-details-downstream-stage`.

## run_id generation + format
- `py_screenalytics/run_layout.py`: `generate_run_id()` uses UUID4 hex (32 chars).
- `py_screenalytics/run_layout.py`: `generate_attempt_run_id()` returns `AttemptN_YYYY-MM-DD_HHMMSS_EST` with attempt number based on existing runs.
- `py_screenalytics/run_layout.py`: `normalize_run_id()` enforces `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`.
- `py_screenalytics/run_layout.py`: `get_or_create_run_id(...)` normalizes or generates run_id and reserves the run directory.

## Run ID propagation (DONE)
- CLI entrypoints generate run_id when omitted: `tools/episode_run.py`, `tools/run_pipeline.py`.
- API entrypoint surfaces run_id: `apps/api/routers/jobs.py` → `/jobs/episode-run`.
- Run_id generation lives in `apps/api/services/jobs.py` (`start_episode_run_job`).
- Pipeline config/result carry run_id: `py_screenalytics/pipeline/episode_engine.py` (`EpisodeRunConfig`, `EpisodeRunResult`).
- Stage runners receive run_id: `py_screenalytics/pipeline/stages.py` (`_config_to_args_namespace`).

Acceptance:
- No supported pipeline invocation path results in missing run_id inside stage execution.
- Status updates never no-op due to missing run_id.

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

## Consistency notes
- jobs.py disambiguation:
  - API entrypoint: `apps/api/routers/jobs.py` (returns run_id in `/jobs/episode-run`).
  - Service implementation: `apps/api/services/jobs.py` (generates run_id).
- Naming: `run_id` is the only canonical identifier; the `AttemptN_...` format is a run_id, not a separate attempt_id concept.

## Verification + cleanup tasks
- DONE: CLI regression test for missing run_id (status file created with generated run_id).
- DONE: API regression test for missing run_id (response includes run_id; run dir uses it).
- DONE: Broader unit suite executed (minimum: `python -m pytest -q tests/unit`).
- Re-verify canonical status hardening still holds after episode_status changes:
  - Lost-update protection (lock or merge/retry).
  - Monotonic transitions (no SUCCESS→RUNNING without force).
  - Derived-status labeling (`is_derived`, `derived_from`).
- Next milestone after verification: wire canonical status transitions for remaining stages.

## PR checklist (when opening PR)
- Include commits: `01b9842` and `774a83d`.
- Describe run_id propagation + compatibility notes.
- List full paths for jobs.py changes.
- Tests run (deduplicated; include broader unit suite).

## Tests run
- `python -m pytest -q tests/unit tests/api/test_jobs_episode_run_run_id.py` (fails: `test_faces_embed_limits`, `test_run_export_s3`, `test_scene_fallback`, `test_track_reps_run_scoped_crops`).
- `python -m pytest -q tests/unit/test_run_id_cli_status.py tests/api/test_jobs_episode_run_run_id.py`
