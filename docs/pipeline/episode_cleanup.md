# Episode Cleanup — Screenalytics Pipeline

**Version:** 2.0
**Last Updated:** 2025-11-18

---

## 1. Overview

The **Episode Cleanup** workflow refines initial detect/track/embed/cluster results through iterative post-processing. It addresses common issues like:

- **Long tracks contaminated by multiple people** (split via appearance gate)
- **Low-quality embeddings** (re-embed with stricter quality gating)
- **Over-/under-segmented clusters** (re-cluster with tuned thresholds)
- **Fragmented identities** (group related clusters)

**Entry point:** `tools/episode_cleanup.py` or `POST /jobs/episode_cleanup_async`

---

## 2. Cleanup Phases

```
1. SPLIT_TRACKS
   - Apply AppearanceGate to long tracks
   - Split tracks where appearance changes significantly
   - Update tracks.jsonl

2. REEMBED
   - Re-run faces_embed with stricter quality gating
   - Filter low-quality crops
   - Update faces.jsonl, faces.npy

3. RECLUSTER
   - Re-cluster with adjusted thresholds
   - Update identities.json

4. GROUP_CLUSTERS
   - Merge related clusters based on cross-episode Facebank matching
   - Update identities.json with person_id labels
```

---

## 3. CLI Usage

```bash
python tools/episode_cleanup.py \
  --ep-id rhobh-s05e02 \
  --video data/videos/rhobh-s05e02/episode.mp4 \
  --profile balanced \
  --actions split_tracks reembed recluster group_clusters \
  --write-back
```

**Profile Support:**

Cleanup now supports performance profiles (same as detect/track/faces/cluster):
- `--profile fast_cpu` (or `low_power`) - Lower quality, faster processing
- `--profile balanced` - Default balanced settings
- `--profile high_accuracy` - Slower, higher quality

Profiles set defaults for `stride`, `fps`, and clustering thresholds. Explicit CLI parameters override profile defaults.

**Example with Profile:**
```bash
python tools/episode_cleanup.py \
  --ep-id rhobh-s05e02 \
  --profile high_accuracy \
  --stride 2 \  # Override profile default
  --write-back
```

**API:**
```bash
POST /jobs/episode_cleanup_async
{
  "ep_id": "rhobh-s05e02",
  "profile": "balanced",
  "actions": ["split_tracks", "reembed", "recluster", "group_clusters"],
  "write_back": true
}
```

---

## 4. Configuration

Cleanup reuses configs from detect, track, embed, cluster stages to ensure consistency:

**Performance Profiles:**
- `config/pipeline/performance_profiles.yaml` - Hardware-aware presets (fast_cpu, low_power, balanced, high_accuracy)

**Stage Configs:**
- `config/pipeline/detection.yaml`
- `config/pipeline/tracking.yaml`
- `config/pipeline/faces_embed_sampling.yaml`
- `config/pipeline/clustering.yaml` - Clustering thresholds (cluster_thresh, min_cluster_size, min_identity_sim)

**Resolution Order:**
1. Explicit CLI/API parameters (highest priority)
2. Environment variables (e.g., `SCREENALYTICS_PERF_PROFILE`)
3. Profile preset values (from performance_profiles.yaml)
4. Stage config defaults (from detection.yaml, clustering.yaml, etc.)
5. Hardcoded fallbacks (lowest priority)

**No new config file**—intentionally shares configs to ensure consistency across detect/track/faces/cluster/cleanup.

---

## 5. Artifacts

### 5.1 `cleanup_report.json`
```json
{
  "ep_id": "rhobh-s05e02",
  "actions_completed": ["split_tracks", "reembed", "recluster"],
  "runtime_sec": 245.3,

  "tracks_before": 42,
  "tracks_after": 58,
  "faces_before": 2048,
  "faces_after": 1876,
  "clusters_before": 18,
  "clusters_after": 12,

  "metrics_before": {
    "tracks_per_minute": 12.5,
    "short_track_fraction": 0.25,
    "id_switch_rate": 0.08,
    "singleton_fraction": 0.38,
    "largest_cluster_fraction": 0.45
  },
  "metrics_after": {
    "tracks_per_minute": 18.2,
    "short_track_fraction": 0.08,
    "id_switch_rate": 0.03,
    "singleton_fraction": 0.15,
    "largest_cluster_fraction": 0.32
  },

  "splits": {
    "hard": 5,
    "streak": 2,
    "iou": 1,
    "total": 8
  },
  "grouping": {
    "centroids": {...},
    "within_episode": {...},
    "across_episodes": {...}
  },

  "actions": ["split_tracks", "reembed", "recluster"]
}
```

**Field Definitions:**

**Core Fields:**
- `actions_completed` - List of cleanup phases executed
- `runtime_sec` - Total cleanup runtime in seconds

**Before/After Counts:**
- `tracks_before`, `tracks_after` - Track count before/after split_tracks
- `faces_before`, `faces_after` - Face count before/after reembed
- `clusters_before`, `clusters_after` - Cluster count before/after recluster

**Before/After Metrics:**
- `metrics_before` - Key metrics from track_metrics.json before cleanup
  - `tracks_per_minute` - Track generation rate (target: < 30)
  - `short_track_fraction` - Fraction of short tracks (target: < 0.20)
  - `id_switch_rate` - Identity fragmentation rate (target: < 0.05)
  - `singleton_fraction` - Fraction of singleton clusters (target: < 0.30)
  - `largest_cluster_fraction` - Largest cluster as fraction of total (target: < 0.40)
- `metrics_after` - Same metrics after cleanup

**Legacy Fields:**
- `actions` - Duplicate of `actions_completed` (for compatibility)
- `splits` - Appearance gate split breakdown (if split_tracks ran)
- `grouping` - Grouping service results (if group_clusters ran)

### 5.2 `cleanup_progress.json`

Cleanup writes progress updates at the start/end of each phase:

```json
{
  "stage": "episode_cleanup",
  "ep_id": "rhobh-s05e02",
  "phase": "reembed",
  "phase_index": 2,
  "phase_total": 4,
  "phase_progress": 0.5,
  "total_elapsed_seconds": 127.8
}
```

**Field Definitions:**
- `stage` - Always `"episode_cleanup"`
- `ep_id` - Episode identifier
- `phase` - Current phase (`"split_tracks"`, `"reembed"`, `"recluster"`, `"group_clusters"`)
- `phase_index` - Current phase number (1-indexed)
- `phase_total` - Total number of phases to execute
- `phase_progress` - Phase completion (0.0 = starting, 1.0 = complete)
- `total_elapsed_seconds` - Elapsed time since cleanup started

**API Usage:**

Poll `GET /jobs/{job_id}/progress` to monitor cleanup status:

```bash
GET /jobs/{job_id}/progress
{
  "state": "running",
  "track_metrics": {...},
  "phase": "reembed",
  "phase_index": 2,
  "phase_total": 4,
  "phase_progress": 0.5,
  "total_elapsed_seconds": 127.8
}
```

### 5.3 Updated Artifacts
- `tracks.jsonl` (with split tracks)
- `faces.jsonl`, `faces.npy` (with re-embedded faces)
- `identities.json` (with re-clustered identities)
- `track_metrics.json` (updated metrics)

---

## 6. Phase Details

### 6.1 Split Tracks
**Purpose:** Break contaminated tracks (multiple people in one track)

**Algorithm:**
1. Load existing tracks
2. For each track, extract frame embeddings
3. Compute pairwise cosine similarity
4. Split track where similarity drops below `TRACK_GATE_APPEAR_HARD` threshold
5. Write updated `tracks.jsonl`

**Metrics:**
- `tracks_split`: Number of original tracks split
- `new_tracks_created`: Number of new tracks created

### 6.2 Re-Embed
**Purpose:** Re-extract embeddings with stricter quality gating

**Algorithm:**
1. Load existing `faces.jsonl`
2. Re-run faces_embed with higher `min_quality` threshold
3. Reject low-quality crops (blur, occlusion, extreme pose)
4. Update `faces.jsonl`, `faces.npy`

**Metrics:**
- `faces_rejected`: Number of faces dropped
- `mean_quality_before`, `mean_quality_after`

### 6.3 Re-Cluster
**Purpose:** Re-cluster tracks with adjusted thresholds

**Algorithm:**
1. Load track embeddings (pooled from updated `faces.npy`)
2. Re-run agglomerative clustering with new `cluster_thresh`
3. Update `identities.json`

**Metrics:**
- `clusters_merged`: Number of clusters combined
- `singleton_fraction_before`, `singleton_fraction_after`
- `largest_cluster_fraction_before`, `largest_cluster_fraction_after`

### 6.4 Group Clusters
**Purpose:** Merge clusters across episodes using Facebank

**Algorithm:**
1. For each cluster, compute centroid embedding
2. Query pgvector for nearest Facebank seed embeddings
3. If similarity > threshold, assign `person_id` label
4. Merge clusters with same `person_id`
5. Update `identities.json`

**Metrics:**
- `clusters_grouped`: Number of clusters merged via Facebank matching
- `auto_assigned_identities`: Number of identities auto-labeled

---

## 7. Common Workflows

### 7.1 Fix Over-Segmented Clusters
```bash
python tools/episode_cleanup.py \
  --ep-id <ep_id> \
  --actions recluster \
  --cluster-thresh 0.50 \
  --write-back
```

### 7.2 Fix Low-Quality Embeddings
```bash
python tools/episode_cleanup.py \
  --ep-id <ep_id> \
  --actions reembed recluster \
  --min-quality 0.8 \
  --write-back
```

### 7.3 Full Cleanup (All Phases)
```bash
python tools/episode_cleanup.py \
  --ep-id <ep_id> \
  --video <path> \
  --actions split_tracks reembed recluster group_clusters \
  --write-back
```

---

## 8. References

- [Pipeline Overview](overview.md)
- [Detect & Track](detect_track_faces.md)
- [Faces Harvest](faces_harvest.md)
- [Cluster Identities](cluster_identities.md)
- [Artifact Schemas](../reference/artifacts_faces_tracks_identities.md)

---

**Maintained by:** Screenalytics Engineering
