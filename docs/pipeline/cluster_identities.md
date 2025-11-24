# Cluster Identities — Screenalytics Pipeline

**Version:** 2.0
**Last Updated:** 2025-11-18

---

## 1. Overview

The **Cluster Identities** stage groups face tracks by visual similarity using **track-level embeddings** and **agglomerative clustering**. This produces `identities.json` and Facebank thumbnails, enabling:

1. Automatic identity assignment per track
2. Manual moderation (merge, split, move, delete identities)
3. Cross-episode identity recognition via Facebank

**Key Principle:** Clustering operates on **track-level embeddings** (pooled from frame-level embeddings), not per-frame embeddings, to reduce noise and improve stability.

---

## 2. Architecture

```
faces.jsonl + faces.npy
    ↓
[TRACK-LEVEL POOLING]
- Group embeddings by track_id
- Pool (mean) unit-norm frame embeddings → track embedding
- Compute track stats (num_faces, mean_quality, time_span)
    ↓
[AGGLOMERATIVE CLUSTERING]
- Compute pairwise cosine similarity matrix
- Cluster tracks using linkage (ward/average/complete)
- Apply cluster_thresh, min_cluster_size
- Handle outliers (singletons → cluster_id=null or noise label)
    ↓
[IDENTITY FORMATION]
- Assign identity_id to each cluster
- Select canonical face (highest quality) per identity
- Generate Facebank thumbnail
    ↓
[ARTIFACTS]
- identities.json (identity metadata, track_ids, labels, locked)
- thumbs/{identity_id}/rep.jpg (canonical thumbnail)
- Cluster metrics (num_clusters, singleton_fraction, etc.)
```

---

## 3. CLI Usage

### 3.1 Basic Clustering
```bash
python tools/episode_run.py --ep-id <ep_id> --cluster
```

### 3.2 With Custom Thresholds
```bash
python tools/episode_run.py --ep-id <ep_id> --cluster \
  --cluster-thresh 0.58 \
  --min-cluster-size 2
```

### 3.3 With Cleanup (Episode Cleanup Flow)
```bash
python tools/episode_cleanup.py --ep-id <ep_id> \
  --actions split_tracks reembed recluster group_clusters
```

---

## 4. API Usage

```bash
POST /jobs/cluster
Content-Type: application/json

{
  "ep_id": "rhobh-s05e02",
  "cluster_thresh": 0.58,
  "min_cluster_size": 2
}

# Response: SSE stream or async job_id
```

---

## 5. Configuration

### 5.1 Clustering Config (TBD: `config/pipeline/recognition.yaml`)
```yaml
# Clustering algorithm
algorithm: agglomerative  # "agglomerative" | "dbscan" | "hdbscan"
linkage: average          # "ward" | "average" | "complete"

# Thresholds
cluster_thresh: 0.58      # Cosine similarity threshold (1 - distance)
min_cluster_size: 2       # Minimum tracks per cluster

# Outlier handling
outlier_mode: singleton   # "singleton" (cluster_id=null) | "noise" (label=-1)

# Track-level pooling
pooling_method: mean      # "mean" | "median" | "max"
```

**Note:** Currently, clustering params are passed via CLI/API args; move to config file for consistency.

---

## 6. Artifacts

### 6.1 `identities.json`
```json
{
  "identities": [
    {
      "identity_id": "identity-00001",
      "cluster_id": 1,
      "track_ids": ["track-00001", "track-00005", "track-00012"],
      "canonical_face_path": "thumbs/identity-00001/rep.jpg",
      "embedding_stats": {
        "centroid_norm": 1.0,
        "variance": 0.023,
        "num_tracks": 3,
        "num_faces_total": 142
      },
      "labels": {
        "person_id": "lisa-vanderpump",
        "name": "Lisa Vanderpump"
      },
      "locked": true,
      "created_at": "2025-11-18T12:34:56Z",
      "updated_at": "2025-11-18T12:34:56Z"
    },
    {
      "identity_id": "identity-00002",
      "cluster_id": null,
      "track_ids": ["track-00042"],
      "canonical_face_path": "thumbs/identity-00002/rep.jpg",
      "embedding_stats": {
        "centroid_norm": 1.0,
        "variance": 0.0,
        "num_tracks": 1,
        "num_faces_total": 12
      },
      "labels": {},
      "locked": false
    }
  ],
  "metadata": {
    "ep_id": "rhobh-s05e02",
    "num_tracks": 42,
    "num_clusters": 8,
    "num_singletons": 6,
    "singleton_fraction": 0.14,
    "largest_cluster_size": 12,
    "largest_cluster_fraction": 0.29,
    "cluster_thresh": 0.58,
    "min_cluster_size": 2,
    "algorithm": "agglomerative",
    "linkage": "average",
    "created_at": "2025-11-18T12:34:56Z"
  }
}
```

**Fields (per identity):**
- `identity_id`: Unique identity identifier (e.g., `"identity-00001"`)
- `cluster_id`: Cluster number (null for singletons/outliers)
- `track_ids`: List of track IDs assigned to this identity
- `canonical_face_path`: Path to representative thumbnail
- `embedding_stats`:
  - `centroid_norm`: Norm of cluster centroid (should be ≈ 1.0)
  - `variance`: Intra-cluster variance (lower = tighter cluster)
  - `num_tracks`: Number of tracks in this cluster
  - `num_faces_total`: Total faces across all tracks
- `labels`: Human labels (person_id, name, etc.)
- `locked`: Boolean (true = confirmed by human, protected from auto-merge/split)

**Metadata:**
- `num_tracks`: Total tracks clustered
- `num_clusters`: Number of clusters formed
- `num_singletons`: Tracks in singleton clusters (cluster_id=null)
- `singleton_fraction`: `num_singletons / num_tracks`
- `largest_cluster_size`: Tracks in largest cluster
- `largest_cluster_fraction`: `largest_cluster_size / num_tracks`

### 6.2 Facebank Thumbnails (`thumbs/{identity_id}/rep.jpg`)
Canonical thumbnail for each identity, stored at:
```
data/frames/{ep_id}/thumbs/identity-00001/rep.jpg
```

**Selection criteria:**
1. Highest `quality_score` among all faces in cluster
2. Frontal pose (yaw ≈ 0°)
3. Largest face size

---

## 7. Track-Level Embedding Pooling

### 7.1 Why Track-Level?
Clustering per-frame embeddings leads to:
- High noise (transient expressions, lighting changes)
- Millions of pairwise comparisons (slow)
- Over-segmentation (same person split into many clusters)

**Solution:** Pool frame embeddings per track → single representative embedding.

### 7.2 Pooling Methods
```yaml
pooling_method: mean  # Default
```

**Options:**
- **mean:** Average of unit-norm frame embeddings (re-normalized to unit norm)
- **median:** Median embedding (robust to outliers)
- **max:** Max-pooling (not recommended for cosine similarity)

**Implementation:**
```python
# Pseudocode
track_embeddings = {}
for track_id in unique_track_ids:
    frame_embeds = faces_npy[faces_jsonl["track_id"] == track_id]
    pooled = np.mean(frame_embeds, axis=0)
    pooled /= np.linalg.norm(pooled)  # Re-normalize to unit norm
    track_embeddings[track_id] = pooled
```

### 7.3 Track-Level Artifact (Optional)
For caching and debugging, write track embeddings:
```
data/embeds/{ep_id}/tracks.npy  # Shape: (num_tracks, 512)
data/manifests/{ep_id}/track_embeddings_meta.jsonl
```

Example entry:
```jsonl
{"track_id": "track-00001", "num_faces": 48, "mean_quality": 0.85, "time_span_sec": 12.3}
```

---

## 8. Clustering Algorithm

### 8.1 Agglomerative Clustering (Default)
```python
from scipy.cluster.hierarchy import linkage, fcluster

# Compute pairwise distances (1 - cosine_similarity)
dist_matrix = 1 - cosine_similarity(track_embeddings)

# Hierarchical clustering
Z = linkage(dist_matrix, method='average')  # or 'ward', 'complete'

# Form flat clusters
cluster_labels = fcluster(Z, t=1-cluster_thresh, criterion='distance')
```

**Linkage methods:**
- **average:** Average distance between all pairs (balanced)
- **ward:** Minimize within-cluster variance (good for spherical clusters)
- **complete:** Maximum distance between clusters (strict separation)

### 8.2 Outlier Handling
Tracks that don't fit well into any cluster:

**Singleton mode (default):**
- Tracks in clusters with < `min_cluster_size` tracks → `cluster_id = null`
- Treated as "unidentified" or "low-confidence"

**Noise mode:**
- Same as singleton, but labeled with `cluster_id = -1`

---

## 9. Cluster Metrics

### 9.1 Quality Metrics
| Metric | Target | Warning Threshold | Indicates |
|--------|--------|-------------------|-----------|
| **num_clusters** | 5–15 (typical TV episode) | > 30 | Over-segmentation; decrease `cluster_thresh` |
| **singleton_fraction** | < 0.30 | > 0.50 | Too many outliers; poor embedding quality or threshold too high |
| **largest_cluster_fraction** | < 0.40 | > 0.60 | Over-merging; increase `cluster_thresh` |
| **avg_cluster_size** | 2–5 | > 10 | Likely over-merging |

### 9.2 Guardrails
The pipeline emits **warnings** if:
- `singleton_fraction > 0.5` → "High fraction of singleton clusters; review embedding quality or adjust cluster_thresh"
- `largest_cluster_fraction > 0.6` → "Largest cluster contains >60% of tracks; may be over-merged; increase cluster_thresh"

---

## 10. Identity Moderation (Facebank UI)

### 10.1 Merge Identities
**API:** `POST /episodes/{ep_id}/identities/merge`

```json
{
  "source_ids": ["identity-00002", "identity-00003"],
  "target_id": "identity-00001"
}
```

**Behavior:**
- All `track_ids` from source identities → moved to target identity
- Source identities deleted
- `identities.json` updated
- Facebank thumbnail regenerated (pick best canonical face)

### 10.2 Split Identity
**API:** `POST /episodes/{ep_id}/identities/split`

```json
{
  "identity_id": "identity-00001",
  "track_ids_to_split": ["track-00005", "track-00012"]
}
```

**Behavior:**
- Specified tracks removed from original identity
- New identity created for split tracks
- `identities.json` updated
- New Facebank thumbnail generated

### 10.3 Move Track
**API:** `POST /episodes/{ep_id}/tracks/{track_id}/move`

```json
{
  "from_identity_id": "identity-00001",
  "to_identity_id": "identity-00002"
}
```

**Behavior:**
- Track removed from source identity
- Track added to target identity
- `identities.json` updated
- Thumbnails regenerated if canonical face changed

### 10.4 Delete Track
**API:** `DELETE /episodes/{ep_id}/tracks/{track_id}`

**Behavior:**
- Track removed from all identities
- `tracks.jsonl` entry marked `deleted: true` (soft delete)
- `faces.jsonl` entries for this track marked deleted
- Crops optionally purged from S3/FS

### 10.5 Lock Identity
**API:** `PATCH /episodes/{ep_id}/identities/{identity_id}`

```json
{
  "locked": true,
  "labels": {"person_id": "lisa-vanderpump", "name": "Lisa Vanderpump"}
}
```

**Behavior:**
- `locked: true` prevents automatic re-clustering/merging
- Human-confirmed identities preserved across cleanup runs

---

## 11. Facebank Integration

### 11.1 Facebank Seed Upload
**API:** `POST /cast/{cast_id}/seeds/upload`

**Purpose:** Upload reference face images for known cast members (outside of episode detection).

**Flow:**
1. User uploads face image
2. RetinaFace detects face, extracts landmarks
3. Crop + align face
4. ArcFace embedding extracted
5. Store in `facebank/{person_id}/{seed_id}_d.png` (display) and `{seed_id}_e.png` (embed)
6. Embedding stored in DB (pgvector) for cross-episode matching

### 11.2 Cross-Episode Matching
**Algorithm:**
1. For each episode identity (cluster), compute centroid embedding
2. Query pgvector for nearest Facebank seed embeddings
3. If similarity > threshold (e.g., 0.70), auto-assign `person_id` label
4. If ambiguous (multiple seeds > threshold), flag for manual review

### 11.3 Facebank Thumbnails Layout (S3/FS)
```
artifacts/thumbs/{show_slug}/s{season}/e{episode}/identities/identity-00001/rep.jpg
facebank/{person_id}/{seed_id}_d.png  # Display derivative (512x512)
facebank/{person_id}/{seed_id}_e.png  # Embed derivative (112x112)
facebank/{person_id}/{seed_id}_orig.png  # Original upload (if FACEBANK_KEEP_ORIG=1)
```

---

## 12. Common Issues

### 12.1 Too Many Clusters (Over-Segmentation)
**Symptom:** `num_clusters > 30`, many small clusters

**Causes:**
- `cluster_thresh` too high (e.g., 0.75+)
- Poor embedding quality (blur, occlusion, extreme pose)
- Too few crops per track (under-sampling)

**Fixes:**
1. Decrease `cluster_thresh` from 0.58 → 0.50
2. Increase embedding quality (higher `min_quality` in faces_embed)
3. Increase `max_crops_per_track` for better track-level pooling
4. Review `variance` in cluster metadata (high variance = loose cluster)

### 12.2 Over-Merged Cluster (One Mega-Cluster)
**Symptom:** `largest_cluster_fraction > 0.6`, single cluster dominates

**Causes:**
- `cluster_thresh` too low (e.g., 0.40)
- Uniform embeddings (all faces look similar; poor ArcFace discrimination)

**Fixes:**
1. Increase `cluster_thresh` from 0.58 → 0.65
2. Review embedding distribution (plot t-SNE/UMAP)
3. Check for duplicate tracks (same person tracked multiple times → increase `track_buffer`)

### 12.3 High Singleton Fraction
**Symptom:** `singleton_fraction > 0.5`, many unidentified tracks

**Causes:**
- `min_cluster_size` too high (e.g., 5+)
- Poor embedding quality (rejected by clustering)
- Short tracks with few faces

**Fixes:**
1. Decrease `min_cluster_size` from 2 → 1 (allow singleton clusters)
2. Increase embedding quality gating
3. Review track lengths (filter short tracks in cleanup stage)

### 12.4 Canonical Thumbnail is Poor Quality
**Symptom:** Representative thumbnail is blurry/occluded

**Causes:**
- Canonical face selection logic prioritizes wrong metric

**Fixes:**
1. Adjust canonical selection to prioritize `quality_score` > `face_size`
2. Manually replace thumbnail via Facebank UI
3. Re-cluster after improving embedding quality

---

## 13. Integration Tests

### 13.1 Run Clustering Test
```bash
RUN_ML_TESTS=1 pytest tests/ml/test_cluster.py -v
```

**What it tests:**
- Track-level pooling
- Agglomerative clustering
- Artifact schemas (`identities.json`)
- Cluster metrics within acceptable ranges
- Canonical thumbnail selection

### 13.2 Expected Assertions
```python
assert identities_json_exists
assert len(identities) > 0
assert metadata["singleton_fraction"] < 0.5
assert metadata["largest_cluster_fraction"] < 0.6
assert all(identity["embedding_stats"]["centroid_norm"] ≈ 1.0 for identity in identities)
```

---

## 14. References

- [Pipeline Overview](overview.md) — Full pipeline stages
- [Detect & Track](detect_track_faces.md) — Detection and tracking
- [Faces Harvest](faces_harvest.md) — Embedding extraction
- [Episode Cleanup](episode_cleanup.md) — Post-processing cleanup
- [Artifact Schemas](../reference/artifacts_faces_tracks_identities.md) — Complete schema reference
- [Facebank Reference](../reference/facebank.md) — Facebank layout and seed management
- [Config Reference](../reference/config/pipeline_configs.md) — Key-by-key config docs
- [Troubleshooting](../ops/troubleshooting_faces_pipeline.md) — Common issues

---

**Maintained by:** Screenalytics Engineering
**Next Review:** After Phase 2 completion
