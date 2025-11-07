# Detection Progress

## Current status
- Scaffolded `run_retinaface.py` with `load_model` + `detect_frames`.
- JSONL manifest → `detections.jsonl` schema stub implemented.
- Config validation test ensures thresholds exist.

## Next steps
- Integrate actual RetinaFace weights and GPU inference.
- Add MediaPipe fallback branch.
- Emit quality metrics (magface, serfiq) into detections.
- Wire detections stage into orchestration CLI.
