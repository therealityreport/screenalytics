# TODO: face-alignment

**Status:** PARTIAL (MVP implementations integrated; full models pending)
**Owner:** Engineering
**Created:** 2025-12-11
**Updated:** 2025-12-23
**TTL:** 2026-01-10

---

## Overview

Face alignment feature using FAN (Face Alignment Network) for 68-point landmarks,
quality gating (heuristic-based MVP), and head pose estimation (OpenCV solvePnP MVP).

**Full Documentation:** [docs/plans/in_progress/feature_face_alignment_fan_luvli_3ddfa.md](../../docs/plans/in_progress/feature_face_alignment_fan_luvli_3ddfa.md)

**Production Code:** `py_screenalytics/face_alignment/` (integrated into harvest job)

---

## Tasks

### Phase A: FAN 2D Integration ✅ COMPLETE
- [x] Create `py_screenalytics/face_alignment/fan2d.py` - FAN 2D landmark extraction
- [x] Implement alignment transform (68-point → aligned crop)
- [x] Add config: `config/pipeline/face_alignment.yaml`
- [x] Write tests: `tests/test_face_alignment.py`
- [x] Add `face-alignment>=1.4.0` to `requirements-ml.txt`
- [x] Implement real FAN model loading with lazy initialization
- [x] Wire into pipeline (`tools/episode_run.py` → harvest job)
- [x] Write `aligned_faces.jsonl` artifacts

### Phase B: Quality Gate ✅ MVP IMPLEMENTED (heuristic-based)
- [x] Implement `alignment_quality.compute_alignment_quality` heuristic in `py_screenalytics/face_alignment/fan2d.py`
- [x] Wire heuristic quality score into `aligned_faces.jsonl` and `faces.jsonl`
- [x] Add quality gating config: `face_alignment.quality_gate.enabled`
- [x] Support `drop` and `downweight` modes via config
- [ ] **FUTURE**: Replace heuristic with LUVLi model outputs (per-landmark uncertainty)
- [ ] **FUTURE**: Add per-landmark visibility fields

**Note:** MVP uses deterministic heuristics from FAN landmarks. True LUVLi model integration is future work.

### Phase C: Head Pose ✅ MVP IMPLEMENTED (OpenCV solvePnP)
- [x] Create `py_screenalytics/face_alignment/head_pose.py` - 3D pose extraction
- [x] Implement yaw/pitch/roll extraction via OpenCV solvePnP
- [x] Selective execution (every N frames, low-quality faces)
- [x] Write pose fields into `aligned_faces.jsonl` and `faces.jsonl`
- [x] Add config: `head_pose_3d.enabled`, `head_pose_3d.run_every_n_frames`
- [ ] **FUTURE**: Replace with 3DDFA_V2 model for higher accuracy

**Note:** MVP uses OpenCV solvePnP from FAN 68-point landmarks. True 3DDFA_V2 model integration is future work.

---

**Phase Status:**
- Phase A (FAN 2D): ✅ COMPLETE - production code in `py_screenalytics/face_alignment/fan2d.py`
- Phase B (Quality Gate): ✅ MVP - heuristic-based, integrated in harvest job
- Phase C (Head Pose): ✅ MVP - OpenCV-based, integrated in harvest job

---

## Future Feature: LUVLi Quality Gate

### Problem
Current approach has no per-face uncertainty measure. Low-quality alignments
(extreme pose, occlusion, blur) produce poor embeddings that pollute clusters.

### Solution: LUVLi Uncertainty Estimation
LUVLi (Look Up and Learn Uncertainty in Face Alignment) provides per-landmark
uncertainty scores that can be aggregated into a face-level quality metric.

### Dependencies
| Library | Version | Purpose |
|---------|---------|---------|
| luvli | custom | Per-landmark uncertainty |

### Implementation Plan

**Step 1: Model Integration**
```python
# src/luvli_quality.py
class LUVLiQualityEstimator:
    """Estimate alignment quality using LUVLi uncertainty."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = load_luvli_model(model_path)
        self.device = device

    def estimate_quality(
        self,
        image: np.ndarray,
        landmarks_68: List[List[float]],
    ) -> Tuple[float, List[float]]:
        """
        Returns:
            quality_score: Aggregated quality (0-1, higher=better)
            per_landmark_uncertainty: 68 uncertainty values
        """
        # Run LUVLi model
        uncertainties = self.model.predict(image, landmarks_68)

        # Aggregate: lower uncertainty = higher quality
        mean_uncertainty = np.mean(uncertainties)
        quality_score = 1.0 - np.clip(mean_uncertainty / MAX_UNCERTAINTY, 0, 1)

        return quality_score, uncertainties.tolist()
```

**Step 2: Integration Points**
- Add `alignment_quality` field to `AlignedFace` dataclass
- Filter in `run_fan_alignment()` before embedding
- Config: `quality.threshold: 0.6` (skip faces below this)

**Step 3: Quality Gating in Embedding**
```yaml
# config/pipeline/face_alignment.yaml
quality:
  enabled: true
  threshold: 0.6
  action: skip  # skip | embed_with_flag | warn
```

### Acceptance Criteria
| Metric | Target |
|--------|--------|
| `faces_gated_pct` | 10-30% |
| `false_rejection_rate` | < 5% |
| `recognition_accuracy_delta` | ≥ 2% improvement |

---

## Future Feature: 3DDFA_V2 Selective 3D Pose

### Problem
Need accurate yaw/pitch/roll for:
- Profile detection (skip embedding for extreme poses)
- Visibility analytics (face-visible vs body-only)
- Temporal consistency validation

### Solution: 3DDFA_V2 Adaptive Execution
Run 3D pose estimation selectively to bound compute:
- Every Nth frame per track (N=10-15)
- On "uncertain" faces (low quality, extreme 2D pose hints)
- On track start/end for handoff validation

### Dependencies
| Library | Version | Purpose |
|---------|---------|---------|
| 3DDFA_V2 | custom | Dense 3D face alignment |

### Implementation Plan

**Step 1: Model Integration**
```python
# src/ddfa_v2.py
class DDFA_V2Estimator:
    """3D head pose estimation using 3DDFA_V2."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = load_3ddfa_model(model_path)
        self.device = device

    def estimate_pose(
        self,
        image: np.ndarray,
        bbox: List[float],
    ) -> Dict[str, float]:
        """
        Returns:
            yaw: Left/right rotation (-90 to 90)
            pitch: Up/down rotation (-90 to 90)
            roll: Tilt rotation (-90 to 90)
            visibility: Estimated face visibility fraction
        """
        result = self.model.predict(image, bbox)
        return {
            "yaw": result.yaw,
            "pitch": result.pitch,
            "roll": result.roll,
            "visibility": self._compute_visibility(result),
        }
```

**Step 2: Adaptive Execution Strategy**
```python
def should_run_3d_pose(
    frame_idx: int,
    track_id: int,
    alignment_quality: float,
    config: dict,
) -> bool:
    """Decide if 3D pose should run for this face."""

    # Always run on track boundaries
    if is_track_boundary(frame_idx, track_id):
        return True

    # Run every N frames
    if frame_idx % config["run_every_n_frames"] == 0:
        return True

    # Run on uncertain faces
    if alignment_quality < config["uncertain_threshold"]:
        return True

    return False
```

**Step 3: Config**
```yaml
# config/pipeline/face_alignment.yaml
3ddfa_v2:
  enabled: false  # Future
  model_path: "models/3ddfa_v2.pth"
  run_every_n_frames: 10
  uncertain_threshold: 0.7
  max_yaw_for_embedding: 75  # Skip profiles beyond this
```

### Acceptance Criteria
| Metric | Target |
|--------|--------|
| `pose_yaw_accuracy` | ≤ 5° MAE |
| `pose_pitch_accuracy` | ≤ 5° MAE |
| `pose_roll_accuracy` | ≤ 3° MAE |
| Runtime (per face) | ≤ 10ms GPU |

---

## Promotion Checklist

- [ ] Tests present and passing (`pytest FEATURES/face_alignment/tests/ -v`)
- [ ] Lint clean (`black`, `ruff`, `mypy`)
- [ ] Docs complete (`FEATURES/face_alignment/docs/`)
- [ ] Config-driven (no hardcoded thresholds)
- [ ] Integration tests passing
- [ ] Row added to `ACCEPTANCE_MATRIX.md` (sections 3.7, 3.8, 3.9)

---

## Acceptance Criteria

| Metric | Target |
|--------|--------|
| `landmark_jitter_px` | < 2.0 px |
| `alignment_quality_mean` | ≥ 0.75 |
| `pose_accuracy_degrees` | ≤ 5° MAE |
| `track_fragmentation_delta` | ≥ 10% reduction |
| Runtime (1hr episode) | ≤ 5 min |

---

## Key Files

- `src/run_fan_alignment.py` - FAN landmark extraction + crop alignment
- `src/alignment_quality.py` - Heuristic `alignment_quality` scoring
- `src/run_luvli_quality.py` - LUVLi-style scaffolding (not true LUVLi yet)
- `tests/test_face_alignment.py` - Unit + synthetic tests
