# 2025-11-11 · Face-first tracking + StrongSORT/ReID

## Summary

- Added face-native detection + tracking flow in `tools/episode_run.py`
  - New CLI switches: `--detector {face,person}`, `--tracker {auto,strongsort,bytetrack}`, `--conf-face`, `--min-face`, `--ratio-face min,max`, and `--max-gap`.
  - Face runs default to YOLOv8-face + StrongSORT (OSNet ReID) on CUDA/MPS; CPU hosts default back to ByteTrack.
  - Detections are filtered by confidence, minimum bounding box size, aspect ratio, and inter-frame gap to avoid over-merged IDs.
  - StrongSORT uses cached OSNet embeddings (`~/.cache/screenalytics/reid/osnet_x0_25_msmt17.pt`) via `torchreid`; ByteTrack runs also inherit the new gap splitter.
  - Structured progress now publishes track-quality metrics (`tracks_born`, `tracks_lost`, `id_switches`, `longest_tracks`) in addition to detection + track counts.

- Episode Detail UI
  - Detector/Tracker dropdowns plus an “Advanced gates” expander drive the new CLI switches.
  - Max gap slider controls the per-track splitter.
  - Status panel surfaces track metrics, flagging any track that exceeds 500 frames.

- New moderation endpoints (`apps/api/routers/identities.py`)
  - `POST /identities/{ep_id}/rename`
  - `POST /identities/{ep_id}/merge`
  - `POST /identities/{ep_id}/move_track`
  - `POST /identities/{ep_id}/drop_track`
  - `POST /identities/{ep_id}/drop_frame`
  - All mutations persist `identities.json`, `tracks.jsonl`, and `faces.jsonl` plus the synced manifests under `artifacts/manifests/...`.

- Facebank UI overhaul (page 3)
  - Identity cards now expose inline Rename/Delete, **Merge into…**, and “View cluster” actions.
  - Cluster view surfaces track moves/removals; track view surfaces frame removal + delete track controls wired to the new `/identities/*` endpoints.

- Requirements
  - `requirements-ml.txt` now pins `ultralytics>=8.2.70` and adds `torchreid`, `gdown`, and `tensorboard` for the ReID stack.

## Acceptance

1. Install the ML extras and run the CLI without `--stub`:

   ```bash
   pip install -r requirements-ml.txt
   python tools/episode_run.py --ep-id demo-s01e01 --video samples/demo.mp4 --stride 3 --detector face --tracker strongsort --max-gap 30
   ```

2. Confirm `data/manifests/demo-s01e01/tracks.jsonl` only reports `"class": "face"` and that `summary.metrics` reports reasonable `tracks_born/lost` with `id_switches <= 1`.

3. In Episode Detail → Run detect/track with detector=Face, tracker=StrongSORT, min face 96 px, ratio 0.75–1.5. Verify the result banner shows track metrics and warns if any track exceeds 500 frames.

4. Navigate to “Faces & Tracks Review”, pick the same episode, rename an identity, merge it into another, move a track, drop a track, and remove a frame. Each action should toast success and the counts should refresh without reloading.

5. Smoke the API directly:

   ```bash
   curl -X POST http://localhost:8000/identities/demo-s01e01/drop_track -d '{"track_id": 2}' -H 'Content-Type: application/json'
   ```

   The response returns the track id and `tracks.jsonl` no longer includes it.
