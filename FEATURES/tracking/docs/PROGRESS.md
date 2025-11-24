# Tracking Progress

## Current status
- `bytetrack_runner.py` upgraded to ByteTrack-lite: filters detections, matches by IoU, honors `track_thresh/match_thresh/track_buffer`, and now defaults to the artifact resolver.
- Outputs full `track_v1` rows (frame span + stats) and exposes `run_tracking`.
- New regression test (`tests/FEATURES/tracking/test_tracking_tracks.py`) validates deterministic IDs + schema.

## Next steps
- Swap IoU-only association with full ByteTrack (high/low confidence pools + velocity gating).
- Surface QA metrics (ID switches, FPS) and optionally emit thumbnails.
- Integrate multi-episode CLI entry and logging hooks for orchestration.
