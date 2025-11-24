# Codex PR #4 — Review resolutions

Each review thread from PR #4 now has an explicit resolution. ✅ denotes a code change landed locally; ℹ️ denotes a no-op with rationale.

## FEATURES/tracking/src/bytetrack_runner.py
- r2515565161 — ✅ Removed the unused `dataclasses.field` import (FEATURES/tracking/src/bytetrack_runner.py:15).

## FEATURES/tracking/tests/test_tracking_io.py
- r2515565401 — ✅ Dropped the unused `load_config` import (FEATURES/tracking/tests/test_tracking_io.py:12) so the test matches the current CLI surface.

## apps/api/routers/episodes.py
- r2515565539 — ✅ Added comments on the face-asset cleanup to explain why missing thumbnails/crops are ignored during deletion (apps/api/routers/episodes.py:82-101, 1433-1447).
- r2515565580 — ✅ Documented the roster dedupe `ValueError` handling so duplicate names don’t fail the request (apps/api/routers/episodes.py:1034-1047).
- r2515565611 — ✅ Explained why we ignore missing crop files inside `delete_frame` (apps/api/routers/episodes.py:1433-1447).
- r2515565517 — ✅ Same `_remove_face_assets` comment coverage as r2515565539 (apps/api/routers/episodes.py:82-101).
- r2515565596 — ✅ Same `delete_frame` comment coverage as r2515565611 (apps/api/routers/episodes.py:1433-1447).

## apps/api/routers/jobs.py
- r2515565729 — ✅ Added context for swallowing `FileNotFoundError` when clearing stale progress files (apps/api/routers/jobs.py:205-214).
- r2515565789 — ✅ Added context for ignoring JSON decode failures when cluster stats are partially written (apps/api/routers/jobs.py:527-545).

## apps/api/routers/people.py
- r2515565269 — ✅ Removed the unused `Field` import (apps/api/routers/people.py:11).

## apps/api/services/cast.py
- r2515565191 — ℹ️ No action: this module no longer imports `numpy`, so the warning is already obsolete.

## apps/api/services/grouping.py
- r2515565242 — ✅ Removed the unused `Tuple` import (apps/api/services/grouping.py:9).

## apps/api/services/identities.py
- r2515565626 / r2515565636 / r2515565647 / r2515565658 / r2515565674 / r2515565701 / r2515565752 / r2515565768 — ✅ Added inline comments for every pass-only `except` so asset cleanup, roster seeding, and file moves are clearly documented as best-effort (apps/api/services/identities.py:360-642).
- r2515565903 — ✅ Deleted the unreachable second `return` at the end of the face-transfer handler (apps/api/services/identities.py:684-699).

## apps/api/services/jobs.py
- r2515565811 — ✅ Documented the benign `FileNotFoundError` when clearing progress files (apps/api/services/jobs.py:136-147).
- r2515565825 — ✅ Documented why `ProcessLookupError` is ignored when canceling already-exited jobs (apps/api/services/jobs.py:402-417).
- r2515565849 — ✅ Documented the benign `PermissionError` branch in job cancelation (apps/api/services/jobs.py:402-417).

## apps/api/services/people.py
- r2515565257 — ✅ Removed the unused `uuid` import (apps/api/services/people.py:6).

## apps/api/services/storage.py
- r2515565427 — ✅ Guarded `_load_remote_track_index` so the `except self._client_error_cls` clause never binds `None` (apps/api/services/storage.py:287-305).
- r2515565923 — ✅ Removed the unreachable `return info` that followed the final `raise RuntimeError` (apps/api/services/storage.py:300-305).

## apps/shared/storage.py
- r2515565883 — ✅ Added context for ignoring close-time exceptions on boto bodies (apps/shared/storage.py:83-86).

## apps/workspace-ui/pages/2_Episode_Detail.py
- r2515565125 — ✅ Removed the unused `time` import (apps/workspace-ui/pages/2_Episode_Detail.py:3).

## apps/workspace-ui/pages/4_Screentime.py
- r2515565108 — ✅ Added `_stop_forever()` so `_require_episode()` never mixes implicit/explicit returns and Streamlit stops are typed (apps/workspace-ui/pages/4_Screentime.py:12-55).

## apps/workspace-ui/ui_helpers.py
- r2515565054 — ✅ Simplified the progress time-prefix logic so the condition is meaningful and no longer always false (apps/workspace-ui/ui_helpers.py:623-647).
- r2515565074 — ✅ Removed the unused `show_time` flag entirely (apps/workspace-ui/ui_helpers.py:623-647).

## py_screenalytics/artifacts.py
- r2515565135 — ✅ Removed the unused `ModuleType` import (py_screenalytics/artifacts.py:5).
- r2515565151 — ✅ Removed the unused `Any` import (py_screenalytics/artifacts.py:5).

## tests/api/test_episode_status.py
- r2515565286 — ✅ Dropped the unused `pytest` import (tests/api/test_episode_status.py:6).

## tests/api/test_frames_reassign.py
- r2515565300 — ✅ Dropped the unused `os` import (tests/api/test_frames_reassign.py:2).

## tests/api/test_mirror_v1_v2_fallback.py
- r2515565116 — ✅ Removed the stray `*** End Patch` line so the test file compiles again (tests/api/test_mirror_v1_v2_fallback.py:1-61).

## tests/api/test_roster_endpoints.py
- r2515565353 — ✅ Dropped the unused `os` import (tests/api/test_roster_endpoints.py:1).

## tests/api/test_video_meta_fps.py
- r2515565376 — ✅ Removed the unused `os` import (tests/api/test_video_meta_fps.py:1-4).

## tests/ml/test_retinaface_init_missing.py
- r2515565330 — ✅ Swapped the dummy `import numpy` for `pytest.importorskip(\"numpy\")` so there’s no unused import while still gating on the dependency (tests/ml/test_retinaface_init_missing.py:7-9).

## tests/ui/test_helpers_compile.py
- r2515565311 — ✅ Removed the unused `pytest` import (tests/ui/test_helpers_compile.py:4).

## tools/_img_utils.py
- r2515565445 / r2515565463 — ✅ Added comments documenting why the post-write cleanup ignores `OSError` (tools/_img_utils.py:102-118).

## tools/debug_thumbs.py
- r2515565489 — ✅ Added a note explaining why `JsonlLogger.close()` swallows close errors (tools/debug_thumbs.py:24-33).

## tools/episode_run.py
- r2515565091 / r2515565101 — ✅ Removed the redundant `tracker_choice` reassignments so `_run_detect_track` normalizes the value once (tools/episode_run.py:1923-1950).
- r2515565209 — ✅ Removed the unused `Callable` import (tools/episode_run.py:17).
- r2515565216 — ✅ Removed the unused `lru_cache` import (tools/episode_run.py:20), eliminating the perf-style nit.
- r2515565231 — ✅ Dropped the unused `clip_bbox` import and tightened the generic `except Exception` handlers (`pick_device`, `_prepare_face_crop`, `_emit_debug`, `FrameDecoder.__del__`) with explanatory comments (tools/episode_run.py:35, 270-311, 750-775, 1502-1546).
- r2515565563 — ✅ Same debug-logger comment coverage as r2515565231.
