# TODO: Body Tracking + Person Re-ID + Track Fusion

Version: 1.0
Status: IN_PROGRESS
Owner: Engineering
Created: 2025-12-11
TTL: 2026-01-10

**Feature Sandbox:** `FEATURES/body-tracking/`

---

## Problem Statement

Screen time is lost when cast members turn away from camera, are partially occluded, or their faces aren't visible. Current face-only tracking:

- **Loses identity** when face is not visible (back of head, profile beyond threshold)
- **Creates gaps** in screen time timelines that don't reflect actual presence
- **Fragments tracks** when faces reappear after occlusion

**Impact:** Underreported screen time, fragmented identities, incomplete analytics.

**Goal:** Maintain identity continuity via body tracking when faces aren't visible.

---

## Dependencies

| Library | Version | License | Purpose |
|---------|---------|---------|---------|
| ultralytics | 8.2+ | AGPL-3.0 | YOLOv8 person detection |
| torchreid | 1.4+ | MIT | Person Re-ID embeddings |
| scipy | 1.10+ | BSD-3 | Hungarian algorithm for association |
| numpy | 1.24+ | BSD-3 | Array operations |

**Installation:**
```bash
pip install ultralytics>=8.2.0
pip install torchreid>=1.4.0
```

**Note:** YOLOv8 is AGPL-3.0. For commercial use, consider:
- YOLO license exception for inference
- Alternative: DETR (Apache-2.0) or RT-DETR

---

## Implementation Tasks

### Phase A: Person Detection

**Goal:** Detect all persons in frame, producing bounding boxes.

- [ ] **A1.** Create `FEATURES/body-tracking/src/person_detector.py`
  ```python
  class YOLOPersonDetector:
      """
      Detect persons using YOLOv8.

      Filters COCO detections to only "person" class (class_id=0).
      """
      def __init__(self, model_name: str = "yolov8n.pt", device: str = "auto"):
          ...

      def detect(self, frame: np.ndarray) -> List[PersonDetection]:
          """
          Returns list of PersonDetection with bbox, confidence.
          """
  ```

- [ ] **A2.** Define `PersonDetection` dataclass
  ```python
  @dataclass
  class PersonDetection:
      bbox_xyxy: Tuple[int, int, int, int]
      confidence: float
      frame_idx: int
      timestamp: float
      # Optional: keypoints if using pose model
      keypoints: Optional[np.ndarray] = None
  ```

- [ ] **A3.** Create config: `config/pipeline/body_detection.yaml`
  ```yaml
  body_tracking:
    enabled: true

  person_detection:
    model: yolov8n              # yolov8n | yolov8s | yolov8m | rtdetr
    confidence_threshold: 0.50
    nms_iou_threshold: 0.45
    min_height_px: 50           # Ignore very small persons
    device: auto
  ```

- [ ] **A4.** Write tests: `FEATURES/body-tracking/tests/test_person_detection.py`
  - Test detection on sample frames
  - Test confidence filtering
  - Test COCO class filtering (only persons)

**Acceptance Criteria (Phase A):**
- [ ] Person recall: ≥90% on test frames
- [ ] Person precision: ≥85% on test frames
- [ ] Runtime: ≤50ms per frame (GPU)

---

### Phase B: Person Tracking

**Goal:** Track persons across frames, maintaining temporal consistency.

- [ ] **B1.** Extend ByteTrack for body tracking
  - Reuse existing `ByteTrackAdapter` from face tracking
  - Create separate tracker instance for body tracks
  - Maintain separate `body_track_id` namespace

- [ ] **B2.** Add body track management
  ```python
  class BodyTracker:
      """
      Manage body tracks using ByteTrack.

      Separate from face tracker to avoid ID conflicts.
      """
      def __init__(self, config: BodyTrackingConfig):
          self.tracker = BYTETracker(...)
          self.track_id_offset = 100000  # Avoid collision with face IDs

      def update(self, detections: List[PersonDetection]) -> List[BodyTrack]:
          ...
  ```

- [ ] **B3.** Define `BodyTrack` dataclass
  ```python
  @dataclass
  class BodyTrack:
      body_track_id: int
      first_frame: int
      last_frame: int
      bbox_history: List[Tuple[int, int, int, int]]

      # Association
      associated_face_track_id: Optional[int] = None
      association_confidence: float = 0.0
  ```

- [ ] **B4.** Update tracking config
  ```yaml
  # config/pipeline/body_detection.yaml
  person_tracking:
    track_thresh: 0.50
    new_track_thresh: 0.55
    match_thresh: 0.70
    track_buffer: 120           # Keep lost tracks 5 seconds
  ```

- [ ] **B5.** Write tests: `FEATURES/body-tracking/tests/test_body_tracking.py`
  - Test track continuity across frames
  - Test track buffer behavior
  - Test ID consistency

**Acceptance Criteria (Phase B):**
- [ ] Body track fragmentation: <0.15
- [ ] Body ID switch rate: <0.05
- [ ] Track buffer: Correctly maintains tracks through short occlusions

---

### Phase C: Person Re-ID

**Goal:** Generate embeddings for body crops to enable re-identification.

- [ ] **C1.** Create `FEATURES/body-tracking/src/person_embedder.py`
  ```python
  class OSNetEmbedder:
      """
      Generate Re-ID embeddings using OSNet.

      Default: osnet_x1_0 (256-d embeddings)
      """
      def __init__(self, model_name: str = "osnet_x1_0", device: str = "auto"):
          self.model = torchreid.models.build_model(
              name=model_name,
              num_classes=1,  # Feature extraction only
              pretrained=True
          )

      def encode(self, crops: List[np.ndarray]) -> np.ndarray:
          """
          Input: List of body crops (BGR, any size)
          Output: (N, 256) normalized embeddings
          """
  ```

- [ ] **C2.** Implement body crop extraction
  ```python
  def extract_body_crop(frame: np.ndarray, bbox: Tuple, config: CropConfig) -> np.ndarray:
      """
      Extract and preprocess body crop for Re-ID.

      - Expand bbox by margin (capture more context)
      - Resize to 256x128 (standard Re-ID size)
      - Normalize colors
      """
  ```

- [ ] **C3.** Add quality gating for body crops
  - Check minimum size (width, height)
  - Check occlusion ratio (if keypoints available)
  - Check aspect ratio validity

- [ ] **C4.** Update config for Re-ID
  ```yaml
  # config/pipeline/body_detection.yaml
  person_reid:
    model: osnet_x1_0           # osnet_x1_0 | osnet_ain_x1_0
    embedding_dim: 256
    device: auto

    crop:
      margin: 0.1               # Expand bbox by 10%
      target_size: [256, 128]   # Height x Width
      min_height_px: 64
      min_width_px: 32

    # Pluggable architecture for future models
    backend: torchreid          # torchreid | custom
  ```

- [ ] **C5.** Write tests: `FEATURES/body-tracking/tests/test_person_reid.py`
  - Test embedding generation
  - Test embedding normalization (L2 norm = 1)
  - Test same-person similarity > different-person similarity

**Acceptance Criteria (Phase C):**
- [ ] Re-ID mAP: ≥0.80 on Market-1501 subset
- [ ] Re-ID Rank-1: ≥0.90
- [ ] Runtime: ≤5ms per crop (GPU)

---

### Phase D: Face↔Body Association

**Goal:** Link face tracks with body tracks to create unified identities.

- [ ] **D1.** Create `FEATURES/body-tracking/src/track_fusion.py`
  ```python
  class TrackFusion:
      """
      Associate face tracks with body tracks.

      Strategy:
      1. When face visible: Use IoU to link face/body boxes
      2. When face disappears: Maintain body track as identity proxy
      3. When face reappears: Match via Re-ID + temporal continuity
      """
  ```

- [ ] **D2.** Implement IoU-based association
  ```python
  def associate_by_iou(
      face_boxes: List[BBox],
      body_boxes: List[BBox],
      iou_threshold: float = 0.50
  ) -> List[Tuple[int, int, float]]:
      """
      Associate faces with bodies using spatial IoU.

      Returns: List of (face_idx, body_idx, iou_score)
      """
      # Face should be inside upper portion of body box
      # Use modified IoU that checks containment
  ```

- [ ] **D3.** Implement Re-ID handoff
  ```python
  def handoff_via_reid(
      face_track: FaceTrack,
      candidate_body_tracks: List[BodyTrack],
      reid_embeddings: Dict[int, np.ndarray],
      threshold: float = 0.70
  ) -> Optional[int]:
      """
      When face disappears, find matching body track via Re-ID.

      Use last known face-body association + Re-ID similarity.
      """
  ```

- [ ] **D4.** Handle ambiguous cases
  ```python
  class AssociationResolver:
      """
      Resolve ambiguous face↔body associations.

      Strategies:
      - Multiple faces in one body: Use face positions (spatial clustering)
      - Similar clothing: Use temporal continuity + motion prediction
      - Long occlusions: Decay confidence, require higher match threshold
      """
  ```

- [ ] **D5.** Create config: `config/pipeline/track_fusion.yaml`
  ```yaml
  track_fusion:
    association_iou_thresh: 0.50    # Face inside body
    reid_similarity_thresh: 0.70    # Re-ID match threshold

    handoff:
      max_gap_seconds: 30           # Max time to maintain association
      confidence_decay_rate: 0.95   # Per-second decay
      min_confidence: 0.30          # Drop association below this

    ambiguity:
      similar_clothing_penalty: 0.10    # Reduce score if clothes similar
      motion_consistency_weight: 0.30   # Use motion for disambiguation
  ```

- [ ] **D6.** Write tests: `FEATURES/body-tracking/tests/test_track_fusion.py`
  - Test IoU association on synthetic data
  - Test Re-ID handoff when face disappears
  - Test ambiguity resolution
  - Test confidence decay over time

**Acceptance Criteria (Phase D):**
- [ ] Association accuracy: ≥95% on annotated test set
- [ ] False association rate: <2%
- [ ] Handoff latency: ≤3 frames

---

### Phase E: Screen Time Fusion

**Goal:** Combine face and body screen time into unified timeline.

- [ ] **E1.** Extend `identities.json` schema
  ```python
  @dataclass
  class UnifiedIdentity:
      identity_id: str
      track_ids: List[int]           # Face tracks
      body_track_ids: List[int]      # Associated body tracks

      # Screen time breakdown
      face_visible_duration: float   # Seconds with face visible
      body_only_duration: float      # Seconds body-only
      total_duration: float          # face + body

      # Visibility breakdown
      visibility: VisibilityStats
  ```

- [ ] **E2.** Implement timeline fusion
  ```python
  def fuse_timelines(
      face_timeline: List[TimeSegment],
      body_timeline: List[TimeSegment],
      associations: List[Association]
  ) -> UnifiedTimeline:
      """
      Merge face and body segments into unified timeline.

      Priority: face > body (when both visible, count face only)
      """
  ```

- [ ] **E3.** Add UI support for visibility breakdown
  - Display face vs body screen time in Episode Detail
  - Show visibility timeline (frontal/profile/body-only)
  - Color-code segments by visibility type

- [ ] **E4.** Update screen time calculations
  - Modify `apps/api/services/screentime.py`
  - Add `face_visible_duration` and `body_only_duration` fields
  - Update totals to include body-only time

**Acceptance Criteria (Phase E):**
- [ ] Screen time gap reduction: ≥30%
- [ ] UI displays visibility breakdown correctly
- [ ] Total duration = face_visible + body_only (no double-counting)

---

## Integration Checklist

### Code Integration

- [ ] Update `tools/episode_run.py`:
  - Add person detection stage (optional, configurable)
  - Add body tracking after face tracking
  - Add track fusion before clustering

- [ ] Update `py_screenalytics/pipeline/constants.py`:
  - Add body detection artifact paths
  - Add default config values

- [ ] Update data schemas:
  - `detections.jsonl`: Add `body_bbox`, `body_track_id` fields
  - `tracks.jsonl`: Add `body_track_id`, `track_type` fields
  - `identities.json`: Add body track IDs and visibility stats

### Config Integration

- [ ] Create `config/pipeline/body_detection.yaml`
- [ ] Create `config/pipeline/track_fusion.yaml`
- [ ] Update `EpisodeRunConfig` dataclass

### Testing Integration

- [ ] Move tests from `FEATURES/body-tracking/tests/` to `tests/ml/`
- [ ] Add body tracking benchmarks to CI
- [ ] Create annotated test videos with ground truth

---

## Ambiguity Handling

### Similar Clothing

| Scenario | Strategy |
|----------|----------|
| Two people, same outfit | Use temporal continuity (don't switch IDs) |
| Outfit change mid-episode | Re-ID score will drop, but face re-links |

### Long Occlusions

| Duration | Action |
|----------|--------|
| <5 seconds | Maintain association at full confidence |
| 5-30 seconds | Decay confidence, require higher Re-ID match |
| >30 seconds | Drop association, treat as new track |

### Multiple Faces in Body Box

| Scenario | Strategy |
|----------|----------|
| Group hug | Associate based on face position within body box |
| Overlapping bodies | Use body center, not full box |

---

## Milestones

| Phase | Target Date | Deliverable |
|-------|-------------|-------------|
| Phase A | +1 week | Person detection with YOLO |
| Phase B | +2 weeks | Body tracking with ByteTrack |
| Phase C | +3 weeks | Person Re-ID with OSNet |
| Phase D | +4 weeks | Track fusion implementation |
| Phase E | +5 weeks | Screen time fusion, UI updates |
| Promotion | +6 weeks | Move to `py_screenalytics/body/` |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| YOLO license concerns | Medium | High | Use RT-DETR (Apache) as alternative |
| Re-ID fails on TV domain | Medium | Medium | Fine-tune OSNet on TV dataset |
| Association errors compound | Medium | High | Conservative thresholds, human review |
| Compute overhead significant | Medium | Medium | Skip body tracking on "face-rich" segments |

---

## References

- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [torchreid GitHub](https://github.com/KaiyangZhou/deep-person-reid)
- [OSNet Paper](https://arxiv.org/abs/1905.00953)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)

---

## Related Documents

- [Feature Overview](../features/vision_alignment_and_body_tracking.md)
- [ACCEPTANCE_MATRIX.md](../../ACCEPTANCE_MATRIX.md) - Sections 3.10, 3.11, 3.12
- [Skill: body-tracking](../../.claude/skills/body-tracking/SKILL.md)
