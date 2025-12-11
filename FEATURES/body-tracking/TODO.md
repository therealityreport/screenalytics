# TODO: body-tracking

**Status:** IN_PROGRESS
**Owner:** Engineering
**Created:** 2025-12-11
**TTL:** 2026-01-10

---

## Overview

Body tracking feature using YOLO for person detection, ByteTrack for tracking,
OSNet for Re-ID embeddings, and track fusion for face↔body association.

**Full Documentation:** [docs/todo/feature_body_tracking_reid_fusion.md](../../docs/todo/feature_body_tracking_reid_fusion.md)

---

## Tasks

### Phase A: Person Detection
- [ ] Create `src/person_detector.py` - YOLO person detection
- [ ] Implement COCO class filtering
- [ ] Add config: `config/pipeline/body_detection.yaml`
- [ ] Write tests: `tests/test_person_detection.py`

### Phase B: Person Tracking
- [ ] Extend ByteTrack for body boxes
- [ ] Add `body_track_id` to detection schema
- [ ] Maintain separate track buffers
- [ ] Write tests: `tests/test_body_tracking.py`

### Phase C: Person Re-ID
- [ ] Create `src/person_embedder.py` - OSNet embeddings
- [ ] Implement body crop extraction
- [ ] Add quality gating for body crops
- [ ] Write tests: `tests/test_person_reid.py`

### Phase D: Track Fusion
- [ ] Create `src/track_fusion.py` - Face↔body association
- [ ] Implement IoU-based association
- [ ] Implement Re-ID handoff
- [ ] Handle ambiguous cases
- [ ] Add config: `config/pipeline/track_fusion.yaml`
- [ ] Write tests: `tests/test_track_fusion.py`

### Phase E: Screen Time Fusion
- [ ] Extend `identities.json` schema
- [ ] Implement timeline fusion
- [ ] Add UI support for visibility breakdown

---

## Promotion Checklist

- [ ] Tests present and passing (`pytest FEATURES/body-tracking/tests/ -v`)
- [ ] Lint clean (`black`, `ruff`, `mypy`)
- [ ] Docs complete (`FEATURES/body-tracking/docs/`)
- [ ] Config-driven (no hardcoded thresholds)
- [ ] Integration tests passing
- [ ] Row added to `ACCEPTANCE_MATRIX.md` (sections 3.10, 3.11, 3.12)

---

## Acceptance Criteria

| Metric | Target |
|--------|--------|
| `person_recall` | ≥ 90% |
| `body_track_fragmentation` | < 0.15 |
| `body_id_switch_rate` | < 0.05 |
| `reid_mAP` | ≥ 0.80 |
| `association_accuracy` | ≥ 95% |
| `screen_time_gap_reduction` | ≥ 30% |

---

## Key Files

- `src/person_detector.py` - YOLO person detection
- `src/person_embedder.py` - OSNet Re-ID embeddings
- `src/track_fusion.py` - Face↔body association
- `tests/test_body_tracking.py` - Tracking tests
- `tests/test_person_reid.py` - Re-ID tests
- `tests/test_track_fusion.py` - Fusion tests
