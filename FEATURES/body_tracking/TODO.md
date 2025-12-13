# TODO: body-tracking

**Status:** IMPLEMENTED
**Owner:** Engineering
**Created:** 2025-12-11
**TTL:** 2026-01-10

---

## Overview

Body tracking feature using YOLO for person detection, ByteTrack for tracking,
OSNet for Re-ID embeddings, and track fusion for face↔body association.

**Full Documentation:** [docs/todo/feature_body_tracking_reid_fusion.md](../../docs/todo/feature_body_tracking_reid_fusion.md)

---

## Usage

```bash
# Run full pipeline on episode
python -m FEATURES.body_tracking --episode-id rhoslc-s06e01

# Detection only
python -m FEATURES.body_tracking --episode-id rhoslc-s06e01 --stage detect

# Tracking only
python -m FEATURES.body_tracking --episode-id rhoslc-s06e01 --stage track

# Track fusion (requires face tracks)
python -m FEATURES.body_tracking --episode-id rhoslc-s06e01 --stage fuse

# Screen-time comparison
python -m FEATURES.body_tracking --episode-id rhoslc-s06e01 --stage compare
```

**Output artifacts** (in `data/manifests/{ep_id}/body_tracking/`):
- `body_detections.jsonl` - Frame-by-frame person detections
- `body_tracks.jsonl` - Tracked persons over time
- `body_embeddings.npy` + `body_embeddings_meta.json` - Re-ID embeddings
- `track_fusion.json` - Face↔body associations
- `screentime_comparison.json` - Face-only vs face+body metrics
- `body_metrics.json` - Pipeline summary

---

## Tasks

### Phase A: Person Detection
- [x] Create `src/detect_bodies.py` - YOLO person detection
- [x] Implement COCO class filtering
- [x] Add config: `config/pipeline/body_detection.yaml`
- [x] Write tests: `tests/test_body_tracking.py`

### Phase B: Person Tracking
- [x] Create `src/track_bodies.py` - ByteTrack for body boxes
- [x] Implement `SimpleIoUTracker` fallback
- [x] Add `body_track_id` with ID offset (100000+)
- [x] Write tests for tracking

### Phase C: Person Re-ID
- [x] Create `src/body_embeddings.py` - OSNet embeddings via torchreid
- [x] Implement body crop extraction with margin
- [x] Add quality gating for minimum crop size
- [x] Batch embedding computation

### Phase D: Track Fusion
- [x] Create `src/track_fusion.py` - Face↔body association
- [x] Implement IoU-based frame-by-frame association
- [x] Implement Re-ID handoff for gaps
- [x] Handle ambiguous cases with union-find
- [x] Add config: `config/pipeline/track_fusion.yaml`
- [x] Write tests for fusion

### Phase E: Screen Time Comparison
- [x] Create `src/screentime_compare.py` - Face-only vs combined
- [x] Implement segment merging
- [x] Calculate duration gain metrics
- [x] Per-identity breakdown

### Phase F: CLI and Runner
- [x] Create `__main__.py` - CLI entrypoint
- [x] Create `body_tracking_runner.py` - Pipeline orchestration
- [x] Config loading from YAML
- [x] Skip existing artifacts option

---

## Promotion Checklist

- [x] Tests present (`pytest FEATURES/body_tracking/tests/ -v`)
- [ ] Tests passing (requires ultralytics, torchreid dependencies)
- [ ] Lint clean (`black`, `ruff`, `mypy`)
- [x] Docs complete (`FEATURES/body_tracking/docs/`)
- [x] Config-driven (no hardcoded thresholds)
- [ ] Integration tests with real video
- [x] Row added to `ACCEPTANCE_MATRIX.md` (sections 3.10, 3.11, 3.12)

---

## Acceptance Criteria

| Metric | Target | Status |
|--------|--------|--------|
| `person_recall` | ≥ 90% | Pending eval |
| `body_track_fragmentation` | < 0.15 | Pending eval |
| `body_id_switch_rate` | < 0.05 | Pending eval |
| `reid_mAP` | ≥ 0.80 | Pending eval |
| `association_accuracy` | ≥ 95% | Pending eval |
| `screen_time_gap_reduction` | ≥ 30% | Pending eval |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/__init__.py` | Package exports |
| `src/__main__.py` | CLI entrypoint |
| `src/body_tracking_runner.py` | Pipeline orchestration |
| `src/detect_bodies.py` | YOLO person detection |
| `src/track_bodies.py` | ByteTrack tracking + fallback |
| `src/body_embeddings.py` | OSNet Re-ID embeddings |
| `src/track_fusion.py` | Face↔body association |
| `src/screentime_compare.py` | Screen time comparison |
| `tests/test_body_tracking.py` | Unit and integration tests |

---

## Dependencies

```
ultralytics>=8.2.0      # YOLO
torchreid>=1.4.0        # OSNet Re-ID
supervision>=0.20.0     # ByteTrack (optional, has fallback)
numpy
opencv-python
pyyaml
```
