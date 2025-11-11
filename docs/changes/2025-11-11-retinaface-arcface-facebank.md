# 2025-11-11 · RetinaFace/ArcFace face-first pipeline + S3-first Facebank

## Summary

- Default face pipeline runs RetinaFace + ArcFace + ByteTrack, with StrongSORT available for tough occlusions.
- `tools/episode_run.py` now ships ArcFace embeddings/indices to S3, records tracker metadata in progress payloads, and writes per-track `index.json` files when exporting crops.
- Storage/API layers expose `GET /episodes/{ep_id}/tracks/{track_id}/crops` backed by S3 presigned URLs plus a new `StorageService.list_track_crops`.
- Episode/job routers validate tracked episodes + mirrored video early, include detector/tracker details in job payloads, and expose new deletion endpoints (`POST /episodes/{ep_id}/delete`, `POST /episodes/delete_all`).
- Faces Review UI streams rows of 4:5 crops straight from S3, adds tracker controls, and lets moderators re-run Faces Harvest from the page when crops are missing.
- Streamlit Episode Detail shows detector+tracker selections/progress, trackers propagate through the pipeline (jobs/tests/docs), and `scripts/dev.sh` logs the storage backend/bucket before waiting for `/healthz`.

## Notable changes

1. **Detector/tracker CLI + embeddings**
   - `tools/episode_run.py` accepts `--tracker {bytetrack,strongsort}` (default ByteTrack), instantiates BoT-SORT for StrongSORT-style ReID, and feeds tracker labels into detections, tracks, JSON summaries, and SSE progress.
   - ArcFace embeddings now annotate `faces.jsonl` with `embedding_model="arcface_r100_v1"`, while `TrackRecorder` exports thumbnails/face vectors and writes S3-ready `index.json` files for each track whenever crops are saved.
   - Progress payloads and job summaries always include `detector` + `tracker`, so Episode Detail/Faces Harvest/Cluster pages can render accurate status bars.

2. **Storage + crop listing APIs**
   - `StorageService` adds `list_track_crops()` with local + S3 implementations (index-driven when `index.json` exists, cursor-driven fallback otherwise).
   - New `GET /episodes/{ep_id}/tracks/{track_id}/crops` route returns `{items:[{key,url,frame_idx,ts}], next_start_after}`, enriching items with face metadata (ts/width/height) and presigned URLs for remote artifacts.

3. **Jobs + validation**
   - `/jobs/detect_track( _async )` payloads include detector/tracker/device/stride/fps/save flags. The router checks that the `ep_id` exists in the EpisodeStore and that `data/videos/{ep_id}/episode.mp4` is readable before spawning `episode_run.py`.
   - JobService stores tracker metadata, appends `--tracker ...` to commands, and records requested settings in `data/jobs/{job_id}.json`.
   - API tests cover detector/tracker validation, stub job behavior, remote crop listing, and the new deletion endpoints.

4. **Episodes + deletion**
   - New `POST /episodes/{ep_id}/delete` always wipes local caches, with `include_s3` toggling S3 artifact deletion for that episode. `POST /episodes/delete_all` performs the same sweep across every tracked episode.
   - Episodes page confirmation modals now surface the simplified options, call the new POST endpoints, and echo how many local dirs / S3 objects were removed.

5. **Streamlit UI**
   - Episode Detail exposes Detector + Tracker dropdowns (RetinaFace/YOLOv8-face, ByteTrack/StrongSORT), pipes tracker choices into job payloads, and adds tracker info to progress + completion banners.
   - Faces Review defaults cluster rows to sample=3 and track view to sample=1, adds Delete Track buttons, and shows a “Run Faces Harvest (save crops to S3)” action whenever a track lacks crops.
   - Track rails now consume the new `/tracks/{track_id}/crops` endpoint, so thumbnails render even when crops exist only on S3.
   - Streamlit S3 picker keeps its “Create episode in store” toast, but Episode + Facebank pages no longer reference legacy “Person” detectors.

6. **Dev ergonomics**
   - `scripts/dev.sh` defaults `STORAGE_BACKEND=s3`, prints `[dev.sh] STORAGE_BACKEND=... BUCKET=... API=...`, and reports whether the `/healthz` wait loop succeeded before launching Streamlit.
