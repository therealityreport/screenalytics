# Cluster Min Similarity, Thumbnail Improvements & Track Representatives

**Date:** 2025-11-17
**Branch:** `nov-17`
**Files Modified:**
- `tools/episode_run.py`
- `tools/episode_cleanup.py`
- `apps/api/routers/jobs.py`
- `apps/api/services/jobs.py`
- `apps/api/services/track_reps.py`
- `apps/api/routers/episodes.py`
- `apps/api/services/people.py`
- `apps/workspace-ui/pages/3_Faces_Review.py`

## Summary

Implemented minimum identity similarity enforcement and thumbnail similarity gating to prevent mixed identities (e.g., Rinna + Kyle in the same cluster) and weak/partial thumbnails from being selected. Tracks with low similarity to their identity centroid are now split out into separate identities, and thumbnail selection now filters by both quality AND similarity to the track/identity centroid.

**Thresholds (aligned with UI badges):**
- Identity/cluster cohesion: ≥0.75 strong, ≥0.60 good (flagged when below 0.60)
- Track consistency: ≥0.85 strong, ≥0.70 good
- Frame similarity: ≥0.80 strong, ≥0.65 good

## Problem Statement

The clustering pass was producing mixed identities where visually distinct cast members (e.g., Rinna + Kyle) were incorrectly merged into the same cluster. Additionally, thumbnail selection sometimes picked weak or partial face frames even though quality-based selection was in place. These issues occurred because:

1. **No minimum similarity check after clustering**: Once AgglomerativeClustering assigned tracks to a cluster, ALL tracks remained in that cluster regardless of how similar they actually were to the cluster centroid
2. **Thumbnail selection ignored similarity**: Thumbnails were selected based on quality metrics (sharpness, frontalness, detection confidence) but not on how well the frame matched the track/identity it was representing

## Changes

### 1. Minimum Identity Similarity Enforcement

#### Configuration

Added new environment variable and parameter:
- **`SCREENALYTICS_MIN_IDENTITY_SIM`** (recommended: ≥0.75 to match UI “strong” badge; remains configurable)
- CLI flag: `--min-identity-sim`
- API field: `min_identity_sim` in `ClusterRequest` and `CleanupJobRequest`

#### Implementation ([tools/episode_run.py](../../tools/episode_run.py))

**Constant Added (line 153):**
```python
MIN_IDENTITY_SIMILARITY = float(os.environ.get("SCREENALYTICS_MIN_IDENTITY_SIM", "0.50"))
```

**CLI Argument (lines 2309-2314):**
```python
parser.add_argument(
    "--min-identity-sim",
    type=float,
    default=MIN_IDENTITY_SIMILARITY,
    help="Minimum cosine similarity for a track to remain in an identity cluster (outliers are split out)",
)
```

**Outlier Removal Function (lines 4418-4472):**

Added `_remove_low_similarity_outliers()` which:
1. For each identity cluster (with 2+ tracks):
   - Computes the cluster centroid (mean of track embeddings)
   - Checks similarity of each track embedding to the centroid
   - Removes tracks with similarity < `min_identity_sim`
   - Returns updated clusters + list of outlier tracks

2. Single-track identities always pass (no centroid to compare against)

**Integration into Clustering Pipeline (lines 4047-4081):**

After initial AgglomerativeClustering:
1. Build track embeddings index
2. Call `_remove_low_similarity_outliers()` to filter each cluster
3. Create separate single-track identities for outliers
4. Log outlier removals

**Identity Metadata Enhancement (lines 4100-4143):**

Each identity in `identities.json` now includes:
- `cohesion`: Average similarity of tracks to identity centroid (for multi-track identities)
- `min_identity_sim`: Minimum similarity across all tracks in the identity
- `low_cohesion`: Boolean flag when cohesion < 0.80 OR min_identity_sim < threshold
- `outlier_reason`: For singleton outliers, the reason they were split (e.g., `"low_identity_similarity_0.423"`)

**Manifest Stats (lines 4169-4186):**

Updated `identities.json` payload to include:
```json
{
  "config": {
    "cluster_thresh": 0.7,
    "min_cluster_size": 2,
    "min_identity_sim": 0.5
  },
  "stats": {
    "faces": 1234,
    "clusters": 45,
    "mixed_tracks": 2,
    "outlier_tracks": 3,
    "low_cohesion_identities": 5
  },
  "identities": [...]
}
```

### 2. Thumbnail Similarity Gating

#### Configuration ([apps/api/services/track_reps.py](../../apps/api/services/track_reps.py))

**New Environment Variable (line 22):**
```python
REP_MIN_SIM_TO_CENTROID = float(os.getenv("REP_MIN_SIM_TO_CENTROID", "0.50"))  # Recommend ≥0.70 (good) / 0.85 (strong)
```

#### Implementation

**Three-Pass Selection Strategy (lines 158-314):**

1. **Pass 1: High-quality AND high-similarity**
   - Compute track centroid from all quality-passing frames
   - Select first frame that passes BOTH:
     - Quality gates (detection confidence, sharpness, crop exists)
     - Similarity threshold (sim_to_centroid >= 0.50)

2. **Pass 2: Quality-only fallback**
   - If no high-similarity frame found, select any quality-passing frame
   - Mark as `low_confidence: true` in manifest
   - Log warning with actual similarity value

3. **Pass 3: Best-available fallback**
   - If no quality-passing frames exist, use first available crop
   - Also marked as `low_confidence: true`

**Enhanced Track Rep Metadata:**

Each track_rep in `track_reps.jsonl` now includes:
```json
{
  "track_id": "track_0042",
  "rep_frame": 123,
  "crop_key": "crops/track_0042/frame_000123.jpg",
  "embed": [...],
  "quality": {"det": 0.82, "std": 15.3},
  "sim_to_centroid": 0.95,
  "low_confidence": false
}
```

- `sim_to_centroid`: Actual similarity of representative frame to track centroid
- `low_confidence`: True when rep frame has sim < 0.50

### 3. API Layer Integration

#### Request Models ([apps/api/routers/jobs.py](../../apps/api/routers/jobs.py))

**ClusterRequest (lines 177-182):**
```python
min_identity_sim: float = Field(
    MIN_IDENTITY_SIMILARITY,
    ge=0.0,
    le=0.99,
    description="Minimum cosine similarity for a track to remain in an identity cluster",
)
```

**CleanupJobRequest (line 204):**
```python
min_identity_sim: float = Field(MIN_IDENTITY_SIMILARITY, ge=0.0, le=0.99)
```

#### Command Builders

**`_build_cluster_command()` (line 386):**
```python
command += ["--min-identity-sim", str(req.min_identity_sim)]
```

#### Job Service ([apps/api/services/jobs.py](../../apps/api/services/jobs.py))

**`start_cluster_job()` (lines 416-456):**
- Added `min_identity_sim` parameter to function signature
- Passed through to command
- Included in `requested` metadata

**`start_episode_cleanup_job()` (lines 458-575):**
- Added `min_identity_sim` parameter
- Passed to `episode_cleanup.py` command
- Included in `requested` metadata

#### Episode Cleanup Tool ([tools/episode_cleanup.py](../../tools/episode_cleanup.py))

**Argument (line 186):**
```python
parser.add_argument("--min-identity-sim", type=float, default=0.5)
```

**Cluster Command (lines 138-139):**
```python
"--min-identity-sim",
str(args.min_identity_sim),
```

## Behavior

### Identity Clustering

**Before:**
- AgglomerativeClustering with `cluster_thresh=0.7` merges tracks
- ALL tracks assigned to a cluster remain in that cluster
- Mixed identities (Rinna + Kyle) possible when embeddings are close enough to be clustered but not actually the same person

**After:**
- AgglomerativeClustering with `cluster_thresh=0.7` performs initial merge
- Post-clustering filter removes tracks with `sim_to_centroid < min_identity_sim`
- Outlier tracks become single-track identities with `outlier_reason` metadata
- Cohesion metrics computed and stored for all identities

**Example:**

Initial cluster from Agglomerative: `[track_42, track_51, track_67]`

After outlier removal (min_identity_sim=0.50):
- Identity `id_0012`: `[track_42, track_51]` (cohesion=0.85, min_identity_sim=0.78)
- Identity `id_0099`: `[track_67]` (outlier_reason="low_identity_similarity_0.423")

### Thumbnail Selection

**Before:**
- Select first frame passing quality gates (det >= 0.60, std >= 1.0, crop exists)
- No similarity check → could pick frames that don't match the track

**After:**
- Compute track centroid from quality-passing frames
- Select first frame passing quality gates AND sim_to_centroid >= 0.50
- If no high-sim frame exists, fall back to quality-only but mark `low_confidence: true`
- Low-confidence thumbnails can be filtered or highlighted in UI

**Example:**

Track with 10 frames, centroid computed from frames 1-8 (quality-passing):
- Frame 3: quality ✅, sim=0.92 → **selected as representative**
- Frame 6: quality ✅, sim=0.45 → skipped (low similarity)
- Result: `sim_to_centroid: 0.92, low_confidence: false`

If NO frames have sim >= 0.50:
- Frame 6: quality ✅, sim=0.45 → **selected (best available)**
- Result: `sim_to_centroid: 0.45, low_confidence: true`

## Configuration

### Environment Variables

```bash
# Clustering minimum identity similarity (default: 0.50)
export SCREENALYTICS_MIN_IDENTITY_SIM=0.50

# Thumbnail representative minimum similarity to centroid (default: 0.50)
export REP_MIN_SIM_TO_CENTROID=0.50

# Existing clustering threshold (default: 0.70)
export SCREENALYTICS_CLUSTER_SIM=0.70
```

### CLI Usage

```bash
# Cluster with custom min_identity_sim
python tools/episode_run.py \
  --ep-id rhobh-s05e17 \
  --cluster \
  --cluster-thresh 0.7 \
  --min-cluster-size 2 \
  --min-identity-sim 0.55

# Full cleanup pipeline with custom thresholds
python tools/episode_cleanup.py \
  --ep-id rhobh-s05e17 \
  --video data/videos/rhobh-s05e17.mp4 \
  --cluster-thresh 0.65 \
  --min-identity-sim 0.50 \
  --actions split_tracks reembed recluster group_clusters
```

### API Usage

```python
# Cluster request with min_identity_sim
POST /jobs/cluster_async
{
  "ep_id": "rhobh-s05e17",
  "cluster_thresh": 0.7,
  "min_cluster_size": 2,
  "min_identity_sim": 0.50
}

# Cleanup request
POST /jobs/episode_cleanup_async
{
  "ep_id": "rhobh-s05e17",
  "cluster_thresh": 0.7,
  "min_cluster_size": 2,
  "min_identity_sim": 0.55,
  "actions": ["split_tracks", "reembed", "recluster", "group_clusters"]
}
```

## Manifest Schema

### identities.json

```json
{
  "ep_id": "rhobh-s05e17",
  "pipeline_ver": 11,
  "config": {
    "cluster_thresh": 0.7,
    "min_cluster_size": 2,
    "min_identity_sim": 0.5
  },
  "stats": {
    "faces": 1234,
    "clusters": 45,
    "mixed_tracks": 2,
    "outlier_tracks": 3,
    "low_cohesion_identities": 5
  },
  "identities": [
    {
      "identity_id": "id_0012",
      "label": null,
      "track_ids": [42, 51],
      "size": 127,
      "rep_thumb_rel_path": "identities/id_0012/rep.jpg",
      "rep_thumb_s3_key": "artifacts/thumbs/.../identities/id_0012/rep.jpg",
      "cohesion": 0.8542,
      "min_identity_sim": 0.7823
    },
    {
      "identity_id": "id_0099",
      "label": null,
      "track_ids": [67],
      "size": 23,
      "rep_thumb_rel_path": "identities/id_0099/rep.jpg",
      "rep_thumb_s3_key": null,
      "outlier_reason": "low_identity_similarity_0.423",
      "low_cohesion": true
    }
  ]
}
```

### track_reps.jsonl

```jsonl
{"track_id": "track_0042", "rep_frame": 123, "crop_key": "crops/track_0042/frame_000123.jpg", "embed": [...], "quality": {"det": 0.82, "std": 15.3}, "sim_to_centroid": 0.95}
{"track_id": "track_0067", "rep_frame": 456, "crop_key": "crops/track_0067/frame_000456.jpg", "embed": [...], "quality": {"det": 0.65, "std": 8.2}, "sim_to_centroid": 0.42, "low_confidence": true}
```

## Testing

### Manual Testing

1. **Cluster an episode with mixed identities:**
   ```bash
   python tools/episode_run.py --ep-id rhobh-s05e17 --cluster --min-identity-sim 0.50
   ```

2. **Check identities.json for outliers:**
   ```bash
   cat data/manifests/rhobh-s05e17/identities.json | jq '.stats'
   # Should show outlier_tracks count

   cat data/manifests/rhobh-s05e17/identities.json | jq '.identities[] | select(.outlier_reason != null)'
   # Should list singleton outliers
   ```

3. **Check track_reps.jsonl for low-confidence thumbnails:**
   ```bash
   cat data/manifests/rhobh-s05e17/track_reps.jsonl | jq 'select(.low_confidence == true)'
   ```

4. **Verify cohesion metrics:**
   ```bash
   cat data/manifests/rhobh-s05e17/identities.json | jq '.identities[] | {id: .identity_id, cohesion, min_sim: .min_identity_sim, low_cohesion}'
   ```

### Expected Results

- **Before fix:** Rinna + Kyle might appear in same identity (`track_ids: [42, 51, 67]`)
- **After fix:**
  - High-cohesion identity: `id_0012` with tracks 42, 51 (cohesion=0.85)
  - Outlier identity: `id_0099` with track 67 (outlier_reason="low_identity_similarity_0.423")

- **Thumbnails:** No frames with sim < 0.50 selected unless NO high-sim frames exist (then marked `low_confidence: true`)

## Future Enhancements

### UI Improvements

1. **Low-cohesion warning pills** in cluster detail view:
   - Red pill: "Low identity cohesion (min sim < 0.50)" when `low_cohesion: true`
   - Show `cohesion` and `min_identity_sim` metrics in identity header

2. **Outlier highlighting:**
   - Border tracks marked with `outlier_reason` in cluster detail rail
   - Show outlier reason on hover

3. **Low-confidence thumbnail indicators:**
   - Small "!" badge on thumbnails where `low_confidence: true`
   - Tooltip showing `sim_to_centroid` value

### Adaptive Thresholds

- Auto-adjust `min_identity_sim` based on episode characteristics:
  - Lower threshold (0.40) for episodes with many similar cast members
  - Higher threshold (0.60) for episodes with visually distinct cast

### Incremental Clustering

- Allow adding new tracks to existing identities without full recluster
- Check new track similarity against existing centroid before assignment

## Related Files

- [tools/episode_run.py](../../tools/episode_run.py) - Clustering pipeline with outlier removal
- [tools/episode_cleanup.py](../../tools/episode_cleanup.py) - Full pipeline orchestration
- [apps/api/routers/jobs.py](../../apps/api/routers/jobs.py) - API request models and endpoints
- [apps/api/services/jobs.py](../../apps/api/services/jobs.py) - Job service with command building
- [apps/api/services/track_reps.py](../../apps/api/services/track_reps.py) - Thumbnail selection with similarity gating
- [docs/code-updates/nov-17-episode-detail-detect-track-fallback.md](nov-17-episode-detail-detect-track-fallback.md) - Related manifest fallback changes
- [docs/code-updates/nov-17-zero-tracks-and-bytetrack-thresholds.md](nov-17-zero-tracks-and-bytetrack-thresholds.md) - ByteTrack threshold improvements

## Technical Details

### Cosine Similarity Computation

```python
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized vectors."""
    norm_a = np.linalg.norm(a) + 1e-12
    norm_b = np.linalg.norm(b) + 1e-12
    return float(np.dot(a / norm_a, b / norm_b))
```

- Returns value in range [0, 1] where 1 = identical, 0 = orthogonal
- Threshold of 0.50 means vectors must be at least 60° apart to be considered different identities

### Cluster Centroid

```python
centroid = np.mean(cluster_embeds, axis=0)
norm_centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
```

- Centroid = mean of all track embeddings in the cluster
- L2-normalized before similarity comparison
- Represents the "average face" of the identity

### Cohesion Metric

```python
sims = [_cosine_similarity(emb, norm_centroid) for emb in bucket_embeds]
cohesion_score = float(np.mean(sims))
```

- Cohesion = average similarity of all tracks to the identity centroid
- Range [0, 1] where 1 = perfect cohesion (all tracks identical)
- Low cohesion (<0.80) suggests mixed identity or high intra-person variation

## Migration Notes

### Backward Compatibility

- Default `min_identity_sim=0.50` is conservative and should not split valid identities
- Existing `identities.json` files without `cohesion`/`min_identity_sim` fields remain valid
- New fields are optional and only present when clustering runs with this version

### Reprocessing Existing Episodes

To apply outlier removal to existing clustered episodes:

```bash
# Recluster with new threshold
python tools/episode_run.py \
  --ep-id rhobh-s05e17 \
  --cluster \
  --min-identity-sim 0.50

# Or run full cleanup pipeline
python tools/episode_cleanup.py \
  --ep-id rhobh-s05e17 \
  --video data/videos/rhobh-s05e17.mp4 \
  --actions recluster group_clusters
```

---

## Track Representatives & Quality-Based Frame Selection

### Problem Statement

The original track representative selection used a "first-pass" strategy: it selected the **first frame** that passed quality gates (detection confidence ≥ 0.60, sharpness ≥ 1.0), rather than the **best-quality frame**. This led to several issues:

1. **Suboptimal representatives**: If the first passing frame was only marginally acceptable (e.g., slightly blurry, small face box), it would be selected even if much clearer frames existed later in the track
2. **No visibility into frame quality**: The track view UI showed frames in chronological order with no indication of which frames were highest quality
3. **Cluster reassignment copied instead of moved**: When clusters were reassigned to a different cast member in the People service, they were added to the new person but **not removed** from the original person, causing duplicate cluster ownership

### Changes

### 1. Quality-Score-Based Representative Selection

#### Implementation ([apps/api/services/track_reps.py](../../apps/api/services/track_reps.py))

**Added Quality Metric Extraction (lines 127-160):**

```python
def _extract_quality_metrics(face: Dict[str, Any]) -> Tuple[float, float, float]:
    """Extract quality metrics from a face record.

    Returns:
        (det_score, crop_std, box_area)
    """
    # Detection confidence
    det_score = face.get("det_score") or face.get("conf") or 0.0
    if not det_score:
        quality = face.get("quality")
        if isinstance(quality, dict):
            det_score = quality.get("det") or 0.0

    # Crop standard deviation (sharpness)
    std = face.get("crop_std") or 0.0
    if not std:
        quality = face.get("quality")
        if isinstance(quality, dict):
            std = quality.get("std") or 0.0

    # Face box area (if available)
    box = face.get("box") or face.get("bbox")
    box_area = 0.0
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            width = abs(float(box[2]) - float(box[0]))
            height = abs(float(box[3]) - float(box[1]))
            box_area = width * height
        except (TypeError, ValueError, IndexError):
            box_area = 0.0

    return float(det_score), float(std), float(box_area)
```

**Added Quality Scoring Function (lines 163-189):**

```python
def _compute_quality_score(
    det_score: float,
    crop_std: float,
    box_area: float,
) -> float:
    """Compute a weighted quality score for representative frame selection.

    Higher scores = better quality.

    Weights:
    - Detection confidence: 40%
    - Sharpness (crop_std): 35%
    - Face box area: 25%
    """
    # Normalize scores to 0-1 range
    det_norm = min(max(det_score, 0.0), 1.0)
    std_norm = min(crop_std / 100.0, 1.0)
    area_norm = min(box_area / 100000.0, 1.0)

    # Weighted combination
    score = (0.40 * det_norm) + (0.35 * std_norm) + (0.25 * area_norm)

    return float(score)
```

**Rewrote Track Representative Selection (lines 213-379):**

**OLD behavior:**
- Selected the **first frame** that passed quality gates (det ≥ 0.60, std ≥ 1.0)
- No scoring or comparison between candidate frames

**NEW behavior:**
- Scores **ALL candidate frames** using the quality scoring function
- Selects the **BEST frame** by quality score (not just the first passing frame)
- Three-pass selection with quality-based ranking:
  1. **Best frame with quality gates + similarity ≥ 0.50**: Sort by quality score, select highest
  2. **Best frame with quality gates only** (low_confidence): Sort by quality score, select highest
  3. **Best available frame by score** (low_confidence + rep_low_quality): Select highest-scoring frame regardless of gates

**Enhanced Return Metadata:**

```python
{
    "track_id": "track_0001",
    "rep_frame": 123,
    "crop_key": "crops/track_0001/frame_000123.jpg",
    "embed": [...512-d L2-norm...],
    "quality": {
        "det": 0.82,
        "std": 15.3,
        "box_area": 12345,  # NEW
        "score": 0.87       # NEW (composite quality score 0-1)
    },
    "sim_to_centroid": 0.95,
    "low_confidence": false,
    "rep_low_quality": false  # NEW (true when had to use low-quality fallback)
}
```

**Key Differences:**
- **`quality.box_area`**: Face bounding box area in pixels
- **`quality.score`**: Composite quality score (0-1) based on weighted combination of det, std, and area
- **`rep_low_quality`**: Flag indicating the representative frame had to fall back to low-quality selection (didn't pass minimum gates)

### 2. Per-Frame Quality Metadata in Track API

#### Implementation ([apps/api/routers/episodes.py](../../apps/api/routers/episodes.py))

**Enhanced Track Frames Endpoint (lines 836-1034):**

Modified `_list_track_frame_media()` to:
1. Import quality extraction and scoring functions from `track_reps.py` (lines 847-851)
2. Compute quality scores for **ALL frames** to identify the best frame (lines 878-890, 950-963)
3. Include per-frame quality metadata in each item (lines 910-920, 996-1006)
4. Return `best_frame_idx` in the response payload (lines 947, 1033)

**Enhanced API Response:**

```json
{
  "items": [
    {
      "track_id": 42,
      "frame_idx": 1523,
      "media_url": "...",
      "similarity": 0.94,
      "quality": {
        "det_score": 0.87,
        "crop_std": 18.2,
        "box_area": 15234.5,
        "score": 0.91
      }
    },
    ...
  ],
  "best_frame_idx": 1523,
  "total": 156,
  "page": 1,
  "page_size": 50
}
```

**Key Additions:**
- **`quality`**: Per-frame quality metadata (det_score, crop_std, box_area, composite score)
- **`best_frame_idx`**: The frame index with the highest quality score across the entire track (not just the current page)

### 3. Best-Quality Frame First in Track View UI

#### Implementation ([apps/workspace-ui/pages/3_Faces_Review.py](../../apps/workspace-ui/pages/3_Faces_Review.py))

**Frame Reordering (lines 1290-1307):**

After fetching frames from the API:
1. Extract `best_frame_idx` from the response
2. If the best frame is present in the current page, move it to the front of the frames list
3. Display frames with the best-quality frame shown first, followed by chronological order

**Visual Indicators (lines 1396-1419):**

Added quality badges to each frame:
1. **Best Quality Badge** (line 1397-1402): Green "★ BEST QUALITY" badge appears on the frame with the highest quality score
2. **Quality Score Badge** (lines 1408-1419): All frames show their quality score as a percentage with color coding:
   - **Green** (≥70%): High quality
   - **Orange** (40-70%): Medium quality
   - **Red** (<40%): Low quality

**Example UI Output:**

```
Frame 1523
★ BEST QUALITY
Similarity: 94%
Q: 91%
[Select checkbox]

Frame 1455
Similarity: 89%
Q: 76%
[Select checkbox]
```

### 4. Cluster Reassignment: Move Not Copy

#### Problem

When a cluster was reassigned to a new cast member using the People service (`add_cluster_to_person()`), it was **added** to the target person but **not removed** from the source person. This caused:
- Duplicate cluster ownership (clusters appearing under multiple people)
- Incorrect cluster/track counts in the UI
- Confusing user experience (same cluster shown in multiple places)

#### Implementation ([apps/api/services/people.py](../../apps/api/services/people.py))

**Added Removal Helper (lines 167-192):**

```python
def remove_cluster_from_all_people(
    self,
    show_id: str,
    cluster_id: str,
) -> List[str]:
    """Remove a cluster from all people in a show.

    Returns list of person_ids that had the cluster removed.

    This ensures single ownership when moving clusters between people.
    """
    data = self._load_people(show_id)
    people = data.get("people", [])
    modified_person_ids: List[str] = []

    for person in people:
        cluster_ids = person.get("cluster_ids", [])
        if cluster_id in cluster_ids:
            person["cluster_ids"] = [cid for cid in cluster_ids if cid != cluster_id]
            modified_person_ids.append(person["person_id"])

    if modified_person_ids:
        data["people"] = people
        self._save_people(show_id, data)

    return modified_person_ids
```

**Modified Assignment Function (lines 194-238):**

```python
def add_cluster_to_person(
    self,
    show_id: str,
    person_id: str,
    cluster_id: str,
    update_prototype: bool = True,
    cluster_centroid: Optional[np.ndarray] = None,
    momentum: float = 0.9,
) -> Optional[Dict[str, Any]]:
    """Add a cluster to a person and optionally update prototype.

    IMPORTANT: This operation ensures single ownership by first removing
    the cluster from all other people before adding it to the target person.
    The operation is idempotent - running it multiple times yields the same result.
    """
    # First, remove cluster from all people to ensure single ownership
    removed_from = self.remove_cluster_from_all_people(show_id, cluster_id)
    if removed_from:
        import logging
        LOGGER = logging.getLogger(__name__)
        LOGGER.info(f"Removed cluster {cluster_id} from people {removed_from} before reassigning to {person_id}")

    # ... rest of function adds cluster to target person
```

**Key Changes:**
1. **Single ownership enforced**: Before adding a cluster to the target person, it's removed from all other people
2. **Idempotent operation**: Running the same reassignment multiple times yields the same result
3. **Audit logging**: Logs when clusters are moved from other people

**Track Reassignment:**

The existing `move_track()` function in [apps/api/services/identities.py](../../apps/api/services/identities.py:425-448) already implemented move semantics correctly:

```python
def move_track(ep_id: str, track_id: int, target_identity_id: str | None) -> Dict[str, Any]:
    # ... find source and target identities ...

    # Remove track from source identity
    if source_identity and track_id in source_identity.get("track_ids", []):
        source_identity["track_ids"] = [tid for tid in source_identity["track_ids"] if tid != track_id]

    # Add track to target identity (if not already present)
    if target_identity is not None:
        target_identity.setdefault("track_ids", [])
        if track_id not in target_identity["track_ids"]:
            target_identity["track_ids"].append(track_id)
            target_identity["track_ids"] = sorted(target_identity["track_ids"])

    # ... save and return ...
```

This function already removed tracks from the source identity before adding to the target, ensuring move semantics. The cluster reassignment fix brings the People service into parity with this behavior.

### Benefits

1. **Higher-quality representatives**: Tracks now use the best-quality frame as their representative, not just the first acceptable frame
2. **Transparent quality visibility**: Users can see quality scores for all frames in the track view and identify the best frame at a glance
3. **Better user experience**: Best-quality frames are prioritized in the UI, making it easier to review and curate track galleries
4. **Cleaner ownership model**: Clusters and tracks are now truly moved (not copied) when reassigned, preventing confusion and duplicate entries
5. **Idempotent operations**: Reassignment can be retried safely without creating duplicate cluster associations

### Testing

**Quality-based representative selection:**
1. Run cluster pipeline on an episode with existing track representatives
2. Verify `track_reps.jsonl` now includes `quality.box_area` and `quality.score` fields
3. Verify `rep_low_quality: true` flag appears on tracks that fell back to low-quality frames

**Per-frame quality metadata:**
1. Fetch track frames via `GET /episodes/{ep_id}/tracks/{track_id}/frames`
2. Verify response includes `best_frame_idx` and each item has a `quality` object
3. Verify quality scores are computed correctly (composite of det, std, area)

**Best-quality frame UI:**
1. Open track view in Faces Review page
2. Verify best-quality frame appears first (when on first page)
3. Verify green "★ BEST QUALITY" badge appears on the correct frame
4. Verify quality score badges (Q: N%) appear on all frames with color coding

**Cluster reassignment:**
1. Assign a cluster to Person A
2. Reassign the same cluster to Person B
3. Verify cluster no longer appears under Person A
4. Verify cluster appears only under Person B
5. Repeat step 2 (idempotency test) - verify no errors and same result
