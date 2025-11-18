# Cluster Min Similarity & Thumbnail Improvements

**Date:** 2025-11-17
**Branch:** `nov-17`
**Files Modified:**
- `tools/episode_run.py`
- `tools/episode_cleanup.py`
- `apps/api/routers/jobs.py`
- `apps/api/services/jobs.py`
- `apps/api/services/track_reps.py`

## Summary

Implemented minimum identity similarity enforcement and thumbnail similarity gating to prevent mixed identities (e.g., Rinna + Kyle in the same cluster) and weak/partial thumbnails from being selected. Tracks with low similarity to their identity centroid are now split out into separate identities, and thumbnail selection now filters by both quality AND similarity to the track/identity centroid.

## Problem Statement

The clustering pass was producing mixed identities where visually distinct cast members (e.g., Rinna + Kyle) were incorrectly merged into the same cluster. Additionally, thumbnail selection sometimes picked weak or partial face frames even though quality-based selection was in place. These issues occurred because:

1. **No minimum similarity check after clustering**: Once AgglomerativeClustering assigned tracks to a cluster, ALL tracks remained in that cluster regardless of how similar they actually were to the cluster centroid
2. **Thumbnail selection ignored similarity**: Thumbnails were selected based on quality metrics (sharpness, frontalness, detection confidence) but not on how well the frame matched the track/identity it was representing

## Changes

### 1. Minimum Identity Similarity Enforcement

#### Configuration

Added new environment variable and parameter:
- **`SCREENALYTICS_MIN_IDENTITY_SIM`** (default: `0.50`)
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
REP_MIN_SIM_TO_CENTROID = float(os.getenv("REP_MIN_SIM_TO_CENTROID", "0.50"))
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
