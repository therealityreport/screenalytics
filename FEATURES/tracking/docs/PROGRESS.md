# Tracking Progress

## Current status
- Stubbed `bytetrack_runner.py` with CLI that reads detections and emits `tracks.jsonl`.
- Added unit tests for track building + CLI I/O expectations.
- CLI adheres to `track_v1` schema for downstream stages.

## Next steps
- Integrate real ByteTrack association + scene boundary handling.
- Carry over detector confidence to track-level scoring.
- Emit thumbnails/chip references for QA tooling.
