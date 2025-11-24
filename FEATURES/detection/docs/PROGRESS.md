# Detection Progress

## Current status
- `run_retinaface.py` now samples frames (frame plan or OpenCV stride) and calls a RetinaFace-backed detector with a deterministic simulation fallback.
- Emits `detections.jsonl` in `det_v1` with frame_idx/ts metadata driven by config.
- Added `tests/FEATURES/detection/test_detection_emit.py` to run the full pipeline on a dummy clip plus resolver-aware defaults in the CLI.
- PROGRESS + TODOs synced with README/SETUP instructions.

## Next steps
- Swap the simulation fallback for GPU-backed RetinaFace in CI (pre-warm weights, add caching).
- Persist per-frame artifacts (chips, quality metrics) alongside detections.
- Provide a CLI entry that consumes pipeline config + manifests directly from the DAG.
