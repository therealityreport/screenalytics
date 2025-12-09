# Cluster Quality Analysis Skill

Use this skill for metrics interpretation and quality diagnosis.

## When to Use

- Interpreting cluster quality metrics
- Diagnosing poor clustering results
- Understanding why suggestions are made
- Tuning quality thresholds

## Metrics Reference

### Core Metrics

| Metric | What It Measures | Target | Warning | Critical |
|--------|-----------------|--------|---------|----------|
| Identity Similarity | Cluster cohesion | ≥ 0.85 | 0.70-0.85 | < 0.70 |
| Cast Similarity | Match to facebank | ≥ 0.75 | 0.65-0.75 | < 0.65 |
| Track Similarity | Frame consistency | ≥ 0.80 | 0.70-0.80 | < 0.70 |
| Ambiguity | 1st vs 2nd match gap | ≥ 0.15 | 0.08-0.15 | < 0.08 |

### Cluster-Level Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| Singleton Fraction | % clusters with 1 track | < 30% |
| Cluster Count | Total clusters | Varies by episode |
| Avg Tracks/Cluster | Tracks per cluster | 3-10 |

### Temporal Metrics

| Metric | Description |
|--------|-------------|
| Frame Range | First to last frame appearance |
| Temporal Density | How continuously person appears |
| Overlap Score | Co-occurrence with other clusters |

## Diagnostic Steps

### Step 1: Load Cluster Overview

```python
# Read identities.json
import json
with open(f"data/manifests/{ep_id}/identities.json") as f:
    identities = json.load(f)

print(f"Total clusters: {len(identities)}")
singletons = sum(1 for c in identities if len(c.get('tracks', [])) == 1)
print(f"Singletons: {singletons} ({singletons/len(identities)*100:.1f}%)")
```

### Step 2: Check Centroid Quality

```python
# Read cluster_centroids.json
with open(f"data/manifests/{ep_id}/cluster_centroids.json") as f:
    centroids = json.load(f)

# Verify embedding dimensions
for cid, centroid in centroids.items():
    if len(centroid) != 512:
        print(f"WARNING: Cluster {cid} has {len(centroid)}-d centroid")
```

### Step 3: Review Track Metrics

```python
# Read track_metrics.json
with open(f"data/manifests/{ep_id}/track_metrics.json") as f:
    metrics = json.load(f)

# Find low-quality tracks
for track_id, m in metrics.items():
    if m.get("similarity", 1.0) < 0.70:
        print(f"Low quality track: {track_id} = {m['similarity']:.3f}")
```

### Step 4: Compare to Thresholds

Check against `config/pipeline/thresholds.yaml`:

```yaml
clustering:
  min_identity_similarity: 0.70
  min_cast_similarity: 0.65
  min_ambiguity: 0.08

quality_rescue:
  enabled: true
  singleton_threshold: 0.30
```

## Quality Rescue Logic

When singleton fraction > threshold, quality rescue activates:

1. Finds singletons with cast similarity ≥ threshold
2. Suggests merging into existing clusters
3. Uses temporal overlap as secondary signal

```python
# apps/api/services/identities.py
def get_rescued_clusters(ep_id: str) -> List[Dict]:
    """Find singletons eligible for rescue."""
    ...
```

## Key Services

| File | Purpose |
|------|---------|
| `apps/api/services/grouping.py` | Clustering algorithms |
| `apps/api/services/track_reps.py` | Representative selection |
| `apps/api/services/identities.py` | Cast linking |

## Interpretation Guide

### High Singleton Fraction (>50%)

Possible causes:
- Detection threshold too sensitive
- Tracking lost faces frequently
- Embedding quality issues

Fixes:
- Adjust detection confidence threshold
- Review tracking parameters
- Check embedding model

### Low Identity Similarity (<0.70)

Possible causes:
- Diverse angles/lighting in cluster
- Wrong faces grouped together
- Embedding dimension mismatch

Fixes:
- Review cluster manually
- Check for merge errors
- Validate embedding dimensions

### Low Ambiguity (<0.08)

Possible causes:
- Similar-looking cast members
- Facebank embeddings too similar
- Insufficient reference images

Fixes:
- Add more diverse facebank images
- Manual disambiguation
- Adjust matching threshold

## Checklist

- [ ] Identities.json loaded and analyzed
- [ ] Singleton fraction calculated
- [ ] Centroid dimensions verified
- [ ] Track metrics reviewed
- [ ] Thresholds compared
- [ ] Quality rescue eligibility checked
