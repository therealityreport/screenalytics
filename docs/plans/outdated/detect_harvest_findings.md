# Detect/Track & Harvest Faces review

> Status: outdated — early review notes; for current behavior see `docs/pipeline/detect_track_faces.md` and `docs/pipeline/faces_harvest.md`.

## Bugs (ranked high → low)
1. **Faces thumbnail size dropped on synchronous harvest runs** – The `/jobs/faces_embed` path builds the command without `--thumb-size`, so UI selections are ignored and harvests always default to 256px, unlike the async path. (apps/api/routers/jobs.py)
   - **Suggested fix**: Pass `thumb_size` through `_build_faces_command` and include it in the synchronous `/faces_embed` execution to mirror the async launcher.
2. **Harvest/cluster blocked for legacy tracks without detector metadata** – `detector_is_face_only` returns `False` whenever `tracks.jsonl` exists but lacks a `detector` field, which disables Harvest Faces and clustering even if the tracks are valid face detections. (apps/workspace-ui/ui_helpers.py & pages/2_Episode_Detail.py)
   - **Suggested fix**: Treat missing detector metadata as “unknown/face-safe” (or derive from defaults) instead of auto-blocking when tracks exist.
3. **Faces-ready state ignores existing manifests** – The Episode Detail page only trusts the status API for faces readiness; if the API is stale but `faces.jsonl` exists locally, clustering remains disabled. (apps/workspace-ui/pages/2_Episode_Detail.py)
   - **Suggested fix**: Add a manifest-based fallback (like detect/track) that marks faces as ready when `faces.jsonl` is present/readable.
4. **Detect/Track configuration panel misstates stride impact** – The help text claims the run processes “every frame for 24fps episodes” even when the stride is greater than 1, giving users a false sense of frame coverage. (apps/workspace-ui/pages/2_Episode_Detail.py)
   - **Suggested fix**: Dynamically describe sampling based on the chosen stride (e.g., “every Nth frame”).
5. **Faces harvest UI allows disabled button without actionable recovery** – When legacy detectors are detected, the button is disabled but only a warning is shown, leaving users without a way to retry from the page. (apps/workspace-ui/pages/2_Episode_Detail.py)
   - **Suggested fix**: Provide a CTA (link/button) to rerun Detect/Track with a supported detector directly from the warning state.
6. **Mirroring requirement not enforced server-side for async harvest enqueue** – The async `/faces_embed_async` route doesn’t validate local video presence before starting jobs, so jobs can fail later when frames are needed. (apps/api/routers/jobs.py)
   - **Suggested fix**: Reuse `_validate_episode_ready` (or a lighter check) before queueing async harvest jobs to ensure the episode is mirrored.

## Improvements (ranked high → low)
1. **Auto-mirror before Detect/Track** – The Detect/Track button is disabled when the video isn’t local, forcing a separate mirror click. Allow auto-mirroring via `_ensure_local_artifacts` on run to streamline the flow. (apps/workspace-ui/pages/2_Episode_Detail.py)
2. **Surface detector/tracker labels consistently** – The Faces Harvest pipeline banner shows raw values; using the same human-readable labels as the Detect/Track section would reduce confusion. (apps/workspace-ui/pages/2_Episode_Detail.py)
3. **Expose detection threshold control** – The UI fixes `det_thresh` to a default even though the backend supports configurable thresholds. Adding a slider would help tune recall vs. precision. (apps/workspace-ui/pages/2_Episode_Detail.py)
4. **Warn when detect/track used scene-only detector** – If the last run used the scene detector fallback, highlight that Harvest Faces will be blocked or ineffective, and propose rerunning Detect/Track. (apps/workspace-ui/pages/2_Episode_Detail.py)
5. **Persist advanced detect/track choices** – Remember scene detector/threshold/max-gap selections in session state or user defaults so reruns don’t reset expert settings. (apps/workspace-ui/pages/2_Episode_Detail.py)
6. **Show expected storage impact** – When enabling “Save frames/crops,” display estimated S3/local footprint based on episode length to avoid accidental large uploads. (apps/workspace-ui/pages/2_Episode_Detail.py)
