# TODO: Face Mesh + Advanced Visibility Analytics

Version: 1.0
Status: PLANNED (docs/config only)
Owner: Engineering
Created: 2025-12-11
TTL: 2026-01-10

**Feature Sandbox:** `FEATURES/vision_analytics/` (docs-only today; `src/` not implemented yet)

---

## Problem Statement

Current visibility metrics are coarse:
- **Binary presence** - Face either visible or not, no gradation
- **No gaze information** - Can't tell if cast is looking at camera or away
- **Limited occlusion handling** - Partial faces counted same as full faces
- **No expression data** - Missing rich reaction signals for future analytics

**Impact:** Imprecise screen time, missed analytics opportunities.

**Goal:** Fine-grained visibility metrics via face mesh, plus gaze direction for extended analytics.

---

## Dependencies

| Library | Version | License | Purpose |
|---------|---------|---------|---------|
| mediapipe | 0.10+ | Apache-2.0 | Face mesh (468 landmarks) |
| opencv-python | 4.8+ | Apache-2.0 | Image processing |
| numpy | 1.24+ | BSD-3 | Array operations |

**Installation:**
```bash
pip install mediapipe>=0.10.0
```

---

## Implementation Tasks

### Phase A: Face Mesh Integration

**Goal:** Extract 468-point face mesh on close-up faces.

- [ ] **A1.** Create `FEATURES/vision_analytics/src/face_mesh.py`
  ```python
  class FaceMeshExtractor:
      """
      Extract 468-point face mesh using MediaPipe.

      Run selectively on close-ups to limit compute.
      """
      def __init__(
          self,
          static_image_mode: bool = True,
          max_num_faces: int = 1,
          min_detection_confidence: float = 0.5,
          min_tracking_confidence: float = 0.5
      ):
          import mediapipe as mp
          self.face_mesh = mp.solutions.face_mesh.FaceMesh(
              static_image_mode=static_image_mode,
              max_num_faces=max_num_faces,
              refine_landmarks=True,  # Include iris landmarks
              min_detection_confidence=min_detection_confidence,
              min_tracking_confidence=min_tracking_confidence
          )

      def extract(self, face_crop: np.ndarray) -> Optional[FaceMeshResult]:
          """
          Extract mesh from aligned face crop.

          Returns: 468 landmarks with x, y, z coordinates
          """
  ```

- [ ] **A2.** Define mesh result structure
  ```python
  @dataclass
  class FaceMeshResult:
      landmarks: np.ndarray        # (468, 3) normalized coords
      face_oval_landmarks: np.ndarray  # Subset for face boundary
      left_eye_landmarks: np.ndarray
      right_eye_landmarks: np.ndarray
      lips_landmarks: np.ndarray

      # Derived metrics
      visibility_fraction: float  # 0-1, how much face visible
      mesh_confidence: float
  ```

- [ ] **A3.** Implement selective execution
  ```python
  def should_run_mesh(face: FaceDetection, config: MeshConfig) -> bool:
      """
      Decide whether to run expensive mesh extraction.

      Run mesh if:
      - Face is close-up (large bbox relative to frame)
      - Mesh explicitly requested for analytics
      - Sampling rate triggers (every Nth frame)
      """
      face_area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
      frame_area = config.frame_width * config.frame_height

      is_closeup = (face_area / frame_area) >= config.closeup_threshold

      return is_closeup or (face.frame_idx % config.sample_rate == 0)
  ```

- [ ] **A4.** Create config section
  ```yaml
  # config/pipeline/analytics.yaml
  face_mesh:
    enabled: true

    execution:
      closeup_threshold: 0.05    # Face area / frame area
      sample_rate: 30            # Every Nth frame

    confidence:
      min_detection: 0.5
      min_tracking: 0.5
  ```

- [ ] **A5.** Write tests: `FEATURES/vision_analytics/tests/test_face_mesh.py`
  - Test mesh extraction on sample faces
  - Test selective execution logic
  - Test landmark subset extraction

**Acceptance Criteria (Phase A):**
- [ ] Mesh extracts 468 landmarks on close-ups
- [ ] Selective execution reduces compute by ≥80%
- [ ] Runtime: ≤15ms per face

---

### Phase B: Visibility Fraction

**Goal:** Compute what fraction of face is visible in frame.

- [ ] **B1.** Create `FEATURES/vision_analytics/src/visibility.py`
  ```python
  def compute_visibility_fraction(
      mesh: FaceMeshResult,
      face_bbox: Tuple[int, int, int, int],
      frame_shape: Tuple[int, int]
  ) -> VisibilityResult:
      """
      Compute visibility fraction from face mesh.

      Methods:
      1. Landmark-based: Count visible landmarks / total landmarks
      2. Area-based: Face oval area in frame / full face area
      3. Occlusion: Check which regions have missing/uncertain landmarks
      """
  ```

- [ ] **B2.** Define visibility result
  ```python
  @dataclass
  class VisibilityResult:
      overall_fraction: float      # 0-1, overall visibility

      # Regional breakdown
      left_eye_visible: float
      right_eye_visible: float
      nose_visible: float
      mouth_visible: float
      forehead_visible: float
      chin_visible: float

      # Occlusion detection
      occluded_regions: List[str]  # e.g., ["left_eye", "forehead"]
      occlusion_type: str          # "none", "partial", "significant"
  ```

- [ ] **B3.** Implement region-based visibility
  ```python
  # MediaPipe face mesh landmark indices for regions
  FACE_REGIONS = {
      "left_eye": [33, 133, 160, 144, 145, 153, ...],
      "right_eye": [362, 263, 387, 373, 374, 380, ...],
      "nose": [1, 2, 98, 327, ...],
      "mouth": [13, 14, 78, 308, ...],
      "forehead": [10, 67, 109, 338, ...],
      "chin": [152, 377, 400, 378, ...]
  }

  def compute_region_visibility(
      landmarks: np.ndarray,
      region_indices: List[int],
      confidence_threshold: float = 0.5
  ) -> float:
      """
      Compute visibility for a face region.

      Returns fraction of landmarks in region that are confident.
      """
  ```

- [ ] **B4.** Add visibility to track metrics
  - Aggregate visibility over track frames
  - Store mean, min, max visibility
  - Add to `tracks.jsonl` schema

- [ ] **B5.** Write tests: `FEATURES/vision_analytics/tests/test_visibility.py`
  - Test visibility computation on known cases
  - Test regional breakdown accuracy
  - Test occlusion detection

**Acceptance Criteria (Phase B):**
- [ ] Visibility fraction accuracy: ≥90% on labeled test set
- [ ] Regional breakdown correlates with actual occlusion
- [ ] Metrics aggregate correctly to track level

---

### Phase C: Gaze Direction

**Goal:** Estimate coarse gaze direction from face mesh.

- [ ] **C1.** Create `FEATURES/vision_analytics/src/gaze.py`
  ```python
  class GazeEstimator:
      """
      Estimate gaze direction from face mesh.

      Coarse categories: left, center, right, up, down
      """
      def __init__(self, use_iris: bool = True):
          """
          Args:
              use_iris: Use refined iris landmarks for better accuracy
          """
          self.use_iris = use_iris

      def estimate(self, mesh: FaceMeshResult) -> GazeResult:
          """
          Estimate gaze from eye landmarks.

          Uses:
          - Iris position relative to eye corners
          - Head pose from face mesh
          - Combined gaze vector
          """
  ```

- [ ] **C2.** Define gaze result
  ```python
  @dataclass
  class GazeResult:
      # Coarse categories
      horizontal: str    # "left", "center", "right"
      vertical: str      # "up", "center", "down"

      # Continuous values (optional)
      yaw_degrees: float   # Horizontal angle
      pitch_degrees: float # Vertical angle

      # Confidence
      confidence: float

      # Derived
      looking_at_camera: bool  # Both center
  ```

- [ ] **C3.** Implement iris-based gaze
  ```python
  def estimate_gaze_from_iris(
      left_iris_landmarks: np.ndarray,
      right_iris_landmarks: np.ndarray,
      left_eye_landmarks: np.ndarray,
      right_eye_landmarks: np.ndarray
  ) -> Tuple[float, float]:
      """
      Estimate gaze angles from iris position.

      Returns: (yaw_degrees, pitch_degrees)
      """
      # Compute iris center relative to eye corners
      # Map to angle using calibration
  ```

- [ ] **C4.** Add config for gaze
  ```yaml
  # config/pipeline/analytics.yaml
  gaze:
    enabled: true
    use_iris: true

    thresholds:
      # Degrees from center for each category
      center_threshold: 15
      extreme_threshold: 45

    confidence:
      min_confidence: 0.6
  ```

- [ ] **C5.** Store gaze in track metrics
  - Aggregate gaze over frames
  - Compute "looking at camera" percentage
  - Add to visibility breakdown

**Acceptance Criteria (Phase C):**
- [ ] Gaze categories correct ≥85% of time
- [ ] "Looking at camera" detection useful for analytics
- [ ] Gaze adds ≤5ms overhead

---

### Phase D: CenterFace Detector (Stub)

**Goal:** Stub interface for CPU-friendly detector (future work).

- [ ] **D1.** Create `FEATURES/vision_analytics/src/centerface.py`
  ```python
  class CenterFaceDetector:
      """
      CPU-friendly face detector.

      STATUS: STUB - Implementation deferred.

      Features:
      - ONNX model for cross-platform support
      - 5-point alignment included
      - Lighter than RetinaFace

      Use cases:
      - CPU-only environments
      - Lightweight preview modes
      - Edge deployment (future)
      """

      def __init__(self):
          raise NotImplementedError(
              "CenterFace not yet implemented. "
              "Use RetinaFace with 'device: cpu' for now."
          )

      def detect(self, frame: np.ndarray) -> List[FaceDetection]:
          """Detect faces and return detections with 5-point landmarks."""
          ...
  ```

- [ ] **D2.** Add config stub
  ```yaml
  # config/pipeline/detection.yaml
  detector: retinaface          # retinaface | centerface (future)

  centerface:
    # Future implementation
    enabled: false
    model_path: null
    confidence_threshold: 0.5
  ```

- [ ] **D3.** Document CenterFace roadmap
  - Integration plan
  - Expected accuracy vs RetinaFace
  - Deployment scenarios

**Acceptance Criteria (Phase D):**
- [ ] Stub interface defined
- [ ] Config structure ready
- [ ] Roadmap documented

---

## Integration Checklist

### Code Integration

- [ ] Update `tools/episode_run.py`:
  - Add mesh extraction after face detection (optional)
  - Add visibility computation to face metadata
  - Add gaze to track metrics

- [ ] Update data schemas:
  - `faces.jsonl`: Add `visibility_fraction`, `gaze` fields
  - `tracks.jsonl`: Add visibility/gaze statistics

### Config Integration

- [ ] Create `config/pipeline/analytics.yaml`
- [ ] Update detection config with CenterFace stub

### UI Integration

- [ ] Add visibility breakdown to Episode Detail
- [ ] Show gaze distribution chart (optional)
- [ ] Color-code screen time by visibility level

---

## Future Work (Expression Analytics)

**Note:** Not in current scope, but architecture supports extension.

### Potential Expression Features

| Feature | Implementation | Use Case |
|---------|---------------|----------|
| Mouth open/closed | Lip landmarks distance | Speaking detection |
| Eye open/closed | Eye aspect ratio | Blink detection |
| Smile detection | Mouth corner angles | Reaction analytics |
| Eyebrow raise | Forehead landmarks | Surprise detection |

### Extension Points

```python
# Future: Expression extractor
class ExpressionAnalyzer:
    """
    Analyze facial expressions from mesh.

    NOT YET IMPLEMENTED - architecture ready.
    """
    def analyze(self, mesh: FaceMeshResult) -> ExpressionResult:
        ...

@dataclass
class ExpressionResult:
    mouth_open: float          # 0-1 openness
    eyes_open: float           # 0-1 average
    smile_intensity: float     # 0-1
    eyebrow_raise: float       # 0-1

    dominant_expression: str   # "neutral", "happy", "surprised", etc.
    confidence: float
```

---

## Milestones

| Phase | Target Date | Deliverable |
|-------|-------------|-------------|
| Phase A | +1 week | Face mesh integration |
| Phase B | +2 weeks | Visibility fraction computation |
| Phase C | +3 weeks | Gaze direction estimation |
| Phase D | +3 weeks | CenterFace stub (minimal) |
| Promotion | +4 weeks | Move to `py_screenalytics/analytics/` |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| MediaPipe accuracy on TV faces | Medium | Medium | Validate on domain data |
| Compute overhead too high | Medium | Medium | Strict selective execution |
| Gaze estimation unreliable | Medium | Low | Use as supplementary metric only |
| CenterFace quality insufficient | Unknown | Low | Defer until validated |

---

## References

- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [MediaPipe Face Landmark Indices](https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)
- [CenterFace GitHub](https://github.com/Star-Clouds/CenterFace)

---

## Related Documents

- [Feature Overview](../features/vision_alignment_and_body_tracking.md)
- [ACCEPTANCE_MATRIX.md](../../ACCEPTANCE_MATRIX.md) - Sections 3.14, 3.15
- [Skill: vision-analytics](../../.claude/skills/vision-analytics/SKILL.md)
