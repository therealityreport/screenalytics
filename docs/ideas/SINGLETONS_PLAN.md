# Plan for Single-Track Clusters and Single-Frame Tracks

---

## THE PROBLEM

### What Are They?

| Type | Definition | Why They Exist |
|------|------------|----------------|
| **Single-Track Cluster** | Cluster containing exactly 1 track | Outlier removal, mixed identity detection, no accepted embeddings, clustering edge cases |
| **Single-Frame Track** | Track with only 1 face/frame | Brief appearances, detection gaps, quality filtering removed other frames |

### Current Pain Points

1. **High Singleton Fraction** - Target is <30%, warning at 50%. Many episodes exceed this.
2. **Unreliable Matching** - Single-track clusters can't compute reliable centroids
3. **Quality Metrics Fail** - Cohesion, similarity scores meaningless for 1 face
4. **Manual Work Explosion** - Each singleton requires individual review
5. **Silent Filtering** - Single-frame tracks in multi-track clusters are silently dropped
6. **No Good Merge Path** - Singleton merge exists but is opt-in and limited

---

## PROPOSED PLAN

### Phase 1: Better Detection & Visibility

#### 1.1 Add Singleton Dashboard
Create an at-a-glance view showing:
```
Episode Singleton Health
━━━━━━━━━━━━━━━━━━━━━━━━
Total Clusters: 156
Single-Track:   47 (30%) ⚠️ Above target
Single-Frame:   12 (8%)  ✓ OK

Top Causes:
• Outlier removal: 23
• Mixed identity:  15
• No embeddings:   9
```

#### 1.2 Tag Singleton Origins
Track WHY each singleton was created:
- `origin: "outlier_removal"` - Split from cluster due to low similarity
- `origin: "mixed_identity"` - High face_embedding_spread detected
- `origin: "no_embeddings"` - Faces seen but none accepted
- `origin: "clustering_edge"` - Natural clustering result
- `origin: "manual_split"` - User action

#### 1.3 Visual Indicators in UI
- Red badge: Single-track + single-frame (highest risk)
- Orange badge: Single-track + multi-frame (medium risk)
- Yellow badge: Multi-track but has single-frame tracks filtered
- Show origin reason on hover

---

### Phase 2: Smarter Matching for Singletons

#### 2.1 Multi-Strategy Matching
Instead of just per-frame max similarity, use ensemble:

```python
def match_single_track_cluster(cluster):
    strategies = []

    # Strategy 1: Best frame match
    best_frame_sim = max(cosine_sim(face, cast_emb) for face in faces)
    strategies.append(("best_frame", best_frame_sim))

    # Strategy 2: Median frame match (less sensitive to outliers)
    median_sim = median([cosine_sim(face, cast_emb) for face in faces])
    strategies.append(("median_frame", median_sim))

    # Strategy 3: Temporal neighbor matching
    # Look at tracks before/after in timeline - are they assigned?
    neighbor_cast = get_temporal_neighbors(track)
    if neighbor_cast:
        neighbor_sim = cosine_sim(cluster_centroid, neighbor_cast.centroid)
        strategies.append(("temporal", neighbor_sim))

    # Strategy 4: Visual similarity to assigned clusters
    visual_match = find_visually_similar_assigned_cluster(cluster)
    if visual_match:
        strategies.append(("visual", visual_match.similarity))

    # Combine with weighted voting
    return weighted_ensemble(strategies)
```

#### 2.2 Temporal Consistency Bonus
If a singleton appears between two tracks assigned to same person, boost confidence:

```
Timeline: [Kyle Track A] → [Singleton X] → [Kyle Track B]
                              ↑
                    +0.10 confidence bonus for Kyle match
```

---

### Phase 3: Automated Singleton Reduction

#### 3.1 Pre-Clustering Singleton Prevention

Before clustering, identify tracks likely to become singletons:
```python
def predict_singleton_risk(track):
    risk = 0.0
    if track.face_count == 1:
        risk += 0.3
    if track.avg_quality < 0.5:
        risk += 0.2
    if track.embedding_spread > 0.3:
        risk += 0.2
    if track.duration_seconds < 1.0:
        risk += 0.2
    return risk

# Adjust clustering threshold for high-risk tracks
if predict_singleton_risk(track) > 0.5:
    use_relaxed_threshold = True
```

#### 3.2 Aggressive Singleton Merge (Opt-In)

Enhanced merge algorithm:
```python
singleton_merge_config = {
    "enabled": True,
    "mode": "aggressive",  # NEW: aggressive | conservative | disabled

    # Aggressive mode settings
    "trigger_singleton_frac": 0.25,  # Lower trigger (was 0.4)
    "secondary_cluster_thresh": 0.70,  # Lower threshold (was 0.8)
    "max_pairs_per_track": 10,  # More neighbors (was 5)
    "allow_merge_to_multitrack": True,  # NEW: Merge singleton into existing multi-track
    "temporal_window_frames": 300,  # NEW: Prefer merging temporally close tracks
    "max_iterations": 3,  # Multiple passes (was 1)
}
```

#### 3.3 "Likely Same Person" Auto-Grouping

Group singletons that are probably the same person:
```
Before: [Singleton A] [Singleton B] [Singleton C] [Singleton D]
After:  [Group: A+B+C] (likely same person) [Singleton D] (distinct)
```

Algorithm:
1. Build similarity matrix for all singletons
2. Find connected components with sim > 0.65
3. Create "candidate groups" for user review
4. One-click merge for each group

---

### Phase 4: UI Workflow Improvements

#### 4.1 Singleton Triage Mode

Dedicated view for processing singletons efficiently:
```
┌─────────────────────────────────────────────────────────┐
│ SINGLETON TRIAGE                          [Exit Triage] │
├─────────────────────────────────────────────────────────┤
│ 47 singletons to review                                 │
│                                                         │
│ ┌─────────┐  Best Match: Kyle (78%)     [✓ Accept]     │
│ │  Face   │  Alt Match:  Mike (65%)     [→ Assign Alt] │
│ │  Crop   │  Temporal:   Kyle before/after             │
│ └─────────┘  Origin:     outlier_removal               │
│              Frames: 3   Quality: 0.72                 │
│                                                         │
│  [← Previous]  [Skip]  [New Person]  [Next →]          │
│                                                         │
│  Progress: ████████░░░░░░░░░░ 17/47 (36%)              │
└─────────────────────────────────────────────────────────┘
```

Features:
- Keyboard navigation (J/K, A for accept, S for skip)
- Show temporal context (what's before/after in timeline)
- Show suggested similar singletons to batch merge
- Progress bar with session stats

#### 4.2 Batch Singleton Actions

New bulk operations:
- **Auto-Accept High Confidence** - Accept all singletons with >80% match
- **Group Similar** - Find and propose groups of similar singletons
- **Merge to Nearest** - Merge each singleton into its closest multi-track cluster
- **Create Catch-All** - Group remaining singletons into "Unknown Person #N" clusters
- **Delete Low Quality** - Remove singletons with <3 frames AND <50% quality

#### 4.3 Singleton Merge Wizard

Step-by-step guided merge:
```
Step 1: Select singletons to merge
        [x] Singleton A (3 frames)
        [x] Singleton B (1 frame)
        [ ] Singleton C (5 frames)

Step 2: Preview merged cluster
        Combined: 4 frames
        New centroid similarity to Kyle: 0.82

Step 3: Confirm
        [Merge into Kyle] [Merge as New Person] [Cancel]
```

---

### Phase 5: Quality Gates & Guardrails

#### 5.1 Singleton Fraction Gate

Block progression if singleton fraction too high:
```
⚠️ Singleton fraction is 52% (target: <30%)

Before proceeding to Smart Suggestions, consider:
• [Run Singleton Merge] - Automatically merge similar singletons
• [Adjust Clustering] - Re-cluster with relaxed threshold
• [Triage Mode] - Manually review singletons

[Proceed Anyway] [Fix First]
```

#### 5.2 Single-Frame Warning

When reviewing a single-frame cluster:
```
⚠️ This cluster has only 1 frame

Low confidence assignment. Consider:
• Finding more frames of this person
• Merging with similar cluster
• Marking as "Unknown"

Confidence penalty: -15%
```

#### 5.3 Export Warnings

Before exporting screen time:
```
⚠️ 12 singletons will be exported as separate identities

This may inflate "unique person" counts.
[Review Singletons] [Export Anyway]
```

---

### Phase 6: Analytics & Learning

#### 6.1 Singleton Analytics Dashboard

Track singleton metrics over time:
- Singleton fraction trend across episodes
- Most common singleton origins
- Auto-merge success rate
- Manual merge patterns

#### 6.2 Clustering Parameter Recommendations

Based on episode characteristics:
```
Episode Analysis:
• Average track length: 45 frames
• Face quality: 0.68 avg
• Temporal spread: clustered (not sparse)

Recommended Clustering:
• cluster_thresh: 0.72 (slightly relaxed)
• min_cluster_size: 1
• singleton_merge: enabled (aggressive)

Expected singleton fraction: ~20%
```

#### 6.3 Feedback Loop

Track which singletons get merged/assigned and learn:
- Which singleton origins are usually correct (don't auto-merge)
- Which origins are usually wrong (auto-merge candidates)
- Threshold adjustments based on episode type

---

## IMPLEMENTATION PRIORITY

### Quick Wins (1-2 days each)
1. Singleton Dashboard - visibility into the problem
2. Visual indicators with origin reason
3. Batch "Accept High Confidence" action
4. Singleton fraction gate before export

### Medium Effort (3-5 days each)
5. Singleton Triage Mode UI
6. Enhanced singleton merge (aggressive mode)
7. Temporal consistency bonus in matching
8. Batch merge wizard

### Strategic (1-2 weeks each)
9. Multi-strategy matching ensemble
10. Pre-clustering singleton prediction
11. "Likely Same Person" auto-grouping
12. Analytics and learning system

---

## SUCCESS METRICS

| Metric | Current | Target |
|--------|---------|--------|
| Singleton fraction | ~40% | <25% |
| Manual singleton reviews per episode | ~50 | <15 |
| Singleton auto-merge success rate | N/A | >80% |
| Time spent on singleton triage | ~30 min | <10 min |
| False merges (wrong identity) | N/A | <5% |

---

## CONFIGURATION ADDITIONS

```yaml
# config/pipeline/clustering.yaml
singleton_handling:
  # Detection
  warn_singleton_fraction: 0.30
  block_singleton_fraction: 0.50

  # Auto-merge
  merge:
    enabled: true
    mode: aggressive  # aggressive | conservative | disabled
    trigger_fraction: 0.25
    similarity_threshold: 0.70
    temporal_window_frames: 300
    max_iterations: 3
    allow_merge_to_multitrack: true

  # Matching
  matching:
    use_ensemble: true
    temporal_bonus: 0.10
    confidence_penalty_single_frame: 0.15
    min_frames_for_high_confidence: 3

  # UI
  ui:
    show_origin_badges: true
    enable_triage_mode: true
    batch_accept_threshold: 0.80
```

---

## SUMMARY

The plan addresses single-track clusters and single-frame tracks through:

1. **Visibility** - Dashboard, badges, origin tracking
2. **Smarter Matching** - Ensemble strategies, temporal context
3. **Automation** - Aggressive merge, pre-clustering prevention, auto-grouping
4. **UI Efficiency** - Triage mode, batch actions, merge wizard
5. **Guardrails** - Gates before progression, export warnings
6. **Learning** - Analytics, recommendations, feedback loop

The goal is to reduce singleton fraction from ~40% to <25% and cut manual review time from ~30 minutes to <10 minutes per episode.
