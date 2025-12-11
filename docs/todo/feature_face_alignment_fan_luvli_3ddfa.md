# TODO: Face Alignment (FAN, LUVLi, 3DDFA_V2)

Version: 1.0
Status: IN_PROGRESS
Owner: Engineering
Created: 2025-12-11
TTL: 2026-01-10

**Feature Sandbox:** `FEATURES/face-alignment/`

---

## Problem Statement

Current alignment uses InsightFace's 5-point landmarks which limits:
- **Pose normalization accuracy** - Only 5 points (eyes, nose, mouth corners) means imprecise alignment for non-frontal faces
- **Quality assessment granularity** - No per-landmark uncertainty, can't identify which parts of face are occluded/unreliable
- **Profile/occlusion handling** - 5-point fails silently on profiles and partial occlusions

**Impact:** Identity fragmentation, low-quality embeddings entering clustering, inconsistent recognition.

---

## Dependencies

| Library | Version | License | Purpose |
|---------|---------|---------|---------|
| face-alignment | 1.3.5+ | BSD-3 | FAN 2D/3D landmarks |
| torch | 2.0+ | BSD-3 | FAN backbone |
| LUVLi | custom | MIT | Uncertainty estimation (port from paper) |
| 3DDFA_V2 | custom | MIT | 3D dense alignment |
| numpy | 1.24+ | BSD-3 | Array operations |

**Installation:**
```bash
pip install face-alignment>=1.3.5
# LUVLi and 3DDFA_V2 require custom integration
```

---

## Implementation Tasks

### Phase A: FAN 2D Integration (MVP)

**Goal:** Replace InsightFace 5-point with FAN 68-point for better alignment.

- [ ] **A1.** Create `FEATURES/face-alignment/src/fan_aligner.py`
  - Implement `FAN2DAligner` class
  - Load pretrained FAN model (s3fd detector + 2D landmarks)
  - Input: BGR image + bbox → Output: 68 landmarks + confidence

- [ ] **A2.** Implement landmark-to-alignment transform
  - Define 5-point subset from 68 for ArcFace alignment (eyes, nose)
  - Implement similarity transform to 112x112 aligned crop
  - Preserve compatibility with InsightFace `face_align.norm_crop()`

- [ ] **A3.** Modify `_prepare_face_crop()` in `tools/episode_run.py`
  - Add `aligner` config switch: `fan_2d` | `insightface`
  - Default to `fan_2d`, fallback to `insightface` if FAN fails
  - Log alignment source for debugging

- [ ] **A4.** Create config: `config/pipeline/alignment.yaml`
  ```yaml
  aligner: fan_2d              # fan_2d | fan_3d | insightface
  fan_2d:
    detector: s3fd             # s3fd | blazeface
    device: auto               # auto | cuda | cpu
    flip_input: false          # horizontal flip for augmentation

  fallback_on_failure: true    # Use insightface if FAN fails
  ```

- [ ] **A5.** Write unit tests: `FEATURES/face-alignment/tests/test_fan_alignment.py`
  - Test landmark detection on frontal/profile faces
  - Test alignment transform produces 112x112 crops
  - Test fallback behavior when FAN fails
  - Benchmark: FAN vs InsightFace on 100 test faces

**Acceptance Criteria (Phase A):**
- [ ] FAN detects 68 landmarks on ≥95% of faces where InsightFace succeeds
- [ ] Aligned crops visually match InsightFace quality on frontals
- [ ] Runtime overhead: ≤20ms per face (GPU)

---

### Phase B: LUVLi Quality Gate

**Goal:** Use per-landmark uncertainty to filter low-quality faces before embedding.

**Reference:** "LUVLi Face Alignment: Estimating Landmarks' Location, Uncertainty, and Visibility Likelihood" (CVPR 2020)

- [ ] **B1.** Create `FEATURES/face-alignment/src/luvli_quality.py`
  - Port LUVLi uncertainty estimation from paper/reference implementation
  - Input: image + landmarks → Output: per-landmark uncertainty + visibility

- [ ] **B2.** Define `alignment_quality` metric
  ```python
  def compute_alignment_quality(landmarks, uncertainties, visibilities):
      """
      Aggregate per-landmark scores into single quality metric.

      Returns: float in [0, 1], higher = better quality
      """
      # Weight by landmark importance (eyes > nose > chin)
      # Penalize high uncertainty and low visibility
      # Normalize to [0, 1]
  ```

- [ ] **B3.** Add quality gating to embedding pipeline
  - In `tools/episode_run.py`, check `alignment_quality >= min_alignment_quality`
  - If below threshold, skip embedding (log skip reason)
  - Store `alignment_quality` in face metadata

- [ ] **B4.** Update config with quality thresholds
  ```yaml
  # config/pipeline/alignment.yaml
  quality_gating:
    enabled: true
    min_alignment_quality: 0.60  # [0, 1], higher = stricter

    # Per-landmark weights for quality computation
    landmark_weights:
      eyes: 1.5
      nose: 1.0
      mouth: 0.8
      chin: 0.5
      ears: 0.3
  ```

- [ ] **B5.** Write tests: `FEATURES/face-alignment/tests/test_alignment_quality.py`
  - Test quality computation on known good/bad faces
  - Test gating correctly filters low-quality faces
  - Test skip reasons logged correctly

**Acceptance Criteria (Phase B):**
- [ ] Quality gate filters 10-30% of faces (tunable via threshold)
- [ ] False rejection rate: <5% of good faces filtered
- [ ] Recognition accuracy improvement: ≥2% on eval set

---

### Phase C: 3DDFA_V2 Selective 3D

**Goal:** Extract accurate head pose and 3D visibility for challenging faces.

**Reference:** cleardusk/3DDFA_V2 - "Towards Fast, Accurate and Stable 3D Dense Face Alignment"

- [ ] **C1.** Create `FEATURES/face-alignment/src/ddfa_v2.py`
  - Integrate 3DDFA_V2 model (ONNX or PyTorch)
  - Input: image + bbox → Output: 3DMM params, pose, visibility

- [ ] **C2.** Implement adaptive execution strategy
  ```python
  def should_run_3d(face, track_context, config):
      """
      Decide whether to run expensive 3D alignment.

      Run 3D if:
      - Every Nth frame per track (sampling)
      - Low alignment_quality from LUVLi
      - Extreme pose estimate from 2D landmarks
      - Track quality metrics degraded
      """
  ```

- [ ] **C3.** Extract and store head pose
  ```python
  @dataclass
  class HeadPose:
      yaw: float    # Left/right rotation (-90 to +90)
      pitch: float  # Up/down rotation (-90 to +90)
      roll: float   # Head tilt (-180 to +180)
      confidence: float
  ```

- [ ] **C4.** Add pose to track metrics
  - Compute per-track pose statistics (mean, std, range)
  - Detect pose jumps (track QA signal)
  - Store in `tracks.jsonl` as `head_pose_stats`

- [ ] **C5.** Update config for 3D alignment
  ```yaml
  # config/pipeline/alignment.yaml
  ddfa_v2:
    enabled: true
    run_every_n_frames: 10        # Sampling rate
    run_on_uncertain: true        # Run if alignment_quality < threshold
    uncertainty_threshold: 0.50   # Trigger 3D below this quality

    pose_limits:
      max_yaw_for_embedding: 75   # Skip near-profile
      max_pitch_for_embedding: 60
  ```

- [ ] **C6.** Write tests: `FEATURES/face-alignment/tests/test_3ddfa.py`
  - Test pose extraction accuracy on labeled set
  - Test adaptive execution triggers correctly
  - Test pose-based filtering

**Acceptance Criteria (Phase C):**
- [ ] Pose accuracy: ≤5° MAE for yaw/pitch on eval set
- [ ] Pose jump detection: ≥95% of synthetic jumps detected
- [ ] Runtime: ≤10ms per face (GPU)

---

## Integration Checklist

### Code Integration

- [ ] Update `tools/episode_run.py`:
  - Import alignment modules
  - Add aligner initialization in `_init_models()`
  - Modify `_prepare_face_crop()` to use configurable aligner
  - Add alignment_quality to face metadata

- [ ] Update `py_screenalytics/pipeline/constants.py`:
  - Add alignment artifact paths
  - Add default config values

- [ ] Update data schemas:
  - `faces.jsonl`: Add `alignment_quality`, `head_pose` fields
  - `tracks.jsonl`: Add `alignment_quality_mean`, `head_pose_mean`

### Config Integration

- [ ] Create `config/pipeline/alignment.yaml`
- [ ] Update `EpisodeRunConfig` dataclass with alignment fields
- [ ] Add alignment section to `config/pipeline/README.md`

### Testing Integration

- [ ] Move tests from `FEATURES/face-alignment/tests/` to `tests/ml/`
- [ ] Add alignment benchmarks to CI
- [ ] Add alignment metrics to acceptance checks

---

## Data Needs

### Evaluation Set

| Dataset | Size | Purpose |
|---------|------|---------|
| Internal TV faces | 1000 faces | Reality TV domain validation |
| AFLW2000-3D | 2000 faces | 3D pose ground truth |
| WFLW | 10000 faces | Landmark accuracy benchmark |

### Labeling Tasks

- [ ] Label 100 faces from production episodes with:
  - Ground truth 68-point landmarks
  - Pose angles (yaw, pitch, roll)
  - Quality rating (good/marginal/bad)
  - Occlusion annotations

---

## Milestones

| Phase | Target Date | Deliverable |
|-------|-------------|-------------|
| Phase A | +2 weeks | FAN 2D integration, config, basic tests |
| Phase B | +3 weeks | LUVLi quality gate, gating in pipeline |
| Phase C | +4 weeks | 3DDFA_V2 selective 3D, full integration |
| Promotion | +5 weeks | Move to `py_screenalytics/alignment/` |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| FAN slower than InsightFace | Medium | Medium | Batch inference, GPU optimization |
| LUVLi port complexity | Medium | Low | Start with simplified uncertainty |
| 3DDFA accuracy on TV faces | Low | Medium | Fine-tune on domain data |
| Memory increase | Medium | Low | Lazy model loading, unload when done |

---

## References

- [face-alignment GitHub](https://github.com/1adrianb/face-alignment)
- [LUVLi Paper (CVPR 2020)](https://arxiv.org/abs/2004.02980)
- [3DDFA_V2 GitHub](https://github.com/cleardusk/3DDFA_V2)
- [InsightFace norm_crop](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py)

---

## Related Documents

- [Feature Overview](../features/vision_alignment_and_body_tracking.md)
- [ACCEPTANCE_MATRIX.md](../../ACCEPTANCE_MATRIX.md) - Sections 3.7, 3.8, 3.9
- [Skill: face-alignment](../../.claude/skills/face-alignment/SKILL.md)
