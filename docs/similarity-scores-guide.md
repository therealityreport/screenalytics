# Similarity Scores Guide

**Color-Coded Reference for Face Recognition Similarity Metrics**

*Last Updated: November 2024*

---

## Overview

The system uses **cosine similarity** (0.0-1.0 scale) to compare face embeddings at different stages of the pipeline. Each type of similarity comparison has a specific purpose and color code.

**November 2024 Updates:**
- Added 4 new metrics: Temporal Consistency, Ambiguity Score, Cluster Isolation, Confidence Trend
- Renamed CAST_TRACK to PERSON_COHESION for clarity
- Raised IDENTITY threshold from 60% to 70% (60-70% often contains errors)
- Standardized weak thresholds to 50% across all types
- FRAME similarity now only shown on outliers (not every frame)
- Enhanced badges: Cast shows rank, Cluster shows min-max range, Track shows excluded frames, Quality shows breakdown on hover

---

## Core Similarity Types

### 🔵 Identity Similarity (Blue #2196F3)
**What:** How similar auto-generated people/clusters are to their centroid (person-level cohesion)
**When:** Cluster/people review, cohesion checks
**Range:** 0.0 - 1.0
**Thresholds:**
- ≥ 0.75: Strong match (dark blue)
- 0.70 - 0.74: Good match (light blue) *[RAISED from 0.60]*
- < 0.70: Needs review (very light blue)

### 🟣 Cast Similarity (Purple #9C27B0)
**What:** Similarity between a cluster/person and a cast member's seed faces
**When:** During identity assignment and cast linking
**Range:** 0.0 - 1.0
**Thresholds:**
- ≥ 0.68: Strong match (auto-assign safe)
- 0.50 - 0.67: Possible match, needs review
- < 0.50: Weak match

**Enhancement:** Now shows rank context: `CAST: 68% (#1 of 5)` indicating this is the best match among 5 suggestions.

### 🟠 Track Similarity (Orange #FF9800)
**What:** Consistency of frames within a track (frame-to-track centroid)
**When:** Track review/outlier detection
**Range:** 0.0 - 1.0
**Thresholds:**
- ≥ 0.85: Strong consistency
- 0.70 - 0.84: Good consistency
- < 0.70: Weak consistency

**Enhancement:** Now shows excluded frames: `TRK: 85% (3 excl)` indicating 3 frames were excluded from centroid due to quality gates.

### 🔷 Person Cohesion (Teal #00ACC1)
**What:** How well a track fits with other tracks assigned to the same person
**When:** Reviewing if tracks belong to the correct person
**Range:** 0.0 - 1.0
**Thresholds:**
- ≥ 0.70: Strong fit
- 0.50 - 0.69: Good fit
- < 0.50: Poor fit (may be misassigned)

*Note: Renamed from "Cast Track Score" for clarity - applies to both cast members and auto-generated people.*

### 🟢 Cluster Cohesion (Green #8BC34A)
**What:** How cohesive/similar all tracks in a cluster are
**When:** Cluster review/merge decisions
**Range:** 0.0 - 1.0
**Thresholds:**
- ≥ 0.80: Tight cluster
- 0.60 - 0.79: Moderate cohesion
- < 0.60: Loose cluster (may need splitting)

**Enhancement:** Now shows min-max range: `CLU: 72% (58-89%)` showing the range of track similarities within the cluster.

---

## New Metrics (November 2024)

### 🔹 Temporal Consistency (Cyan #00BCD4)
**What:** How consistent a person's appearance is across time within an episode
**When:** Detecting problematic assignments due to lighting/costume/angle changes
**Range:** 0.0 - 1.0
**Thresholds:**
- ≥ 0.80: Consistent appearance over time
- 0.60 - 0.79: Variable appearance
- < 0.60: Significantly different across time (may indicate misassignment)

**Use case:** A person who looks different at the beginning vs end of an episode may have tracks from different people mixed in, or may have significant costume/lighting changes that need special handling.

### ⚠️ Ambiguity Score (Red/Amber/Green)
**What:** The gap between the 1st and 2nd best match
**When:** Identifying risky assignments where the system can't clearly distinguish between candidates
**Range:** 0.0 - 1.0 (margin)
**Thresholds:**
- ≥ 0.15 (15%): Clear winner (green) - confident assignment
- 0.08 - 0.14 (8-14%): OK (amber) - probably correct but verify
- < 0.08 (<8%): Risky (red) - could easily be either person

**Use case:** When ambiguity is "Risky", the track could reasonably be assigned to either the 1st or 2nd best match. These should be manually reviewed first.

### 🔮 Cluster Isolation (Indigo #3F51B5)
**What:** Distance to the nearest neighbor cluster
**When:** Identifying merge candidates
**Range:** 0.0 - 1.0 (distance)
**Thresholds:**
- ≥ 0.40: Well isolated (indigo) - distinct cluster
- 0.25 - 0.39: Moderate (light indigo)
- < 0.25: Close (very light) - consider merging

**Use case:** When isolation is "Close", this cluster is very similar to another cluster and may actually be the same person. Suggests a merge operation.

### 📈 Confidence Trend (Green/Gray/Red)
**What:** Whether assignment confidence is improving or degrading over time
**When:** Early warning for problematic assignments
**Values:**
- ↑ Improving (green): Confidence increasing as more data added
- → Stable (gray): Confidence staying consistent
- ↓ Degrading (red): Confidence decreasing - warning sign

**Use case:** A degrading trend means new tracks being added are lowering the overall confidence - possibly incorrect assignments are accumulating.

---

## Frame-Level Indicators

### 🟠 Outlier Badge (Light Orange #FFA726)
**What:** Identifies frames that differ significantly from the rest of the track
**When:** Only shown on frames below the outlier threshold (< 50%)
**Display:** `OUTLIER: 45% ⚠️` with severity indicators

**Severity levels:**
- ⚠️ (one): Mild outlier (1-2 standard deviations)
- ⚠️⚠️ (two): Moderate outlier (2-3 standard deviations)
- ⚠️⚠️⚠️ (three): Severe outlier (3+ standard deviations)

*Note: Frame similarity is no longer shown on every frame - only on outliers. This reduces visual noise while preserving the important signal.*

---

## Quality Score

### 🟢 Quality (Q: XX%)
**What:** Composite score of detection confidence + sharpness + face area
**Range:** 0 - 100%
**Thresholds:**
- ≥ 85%: High quality (green)
- 60-84%: Medium quality (amber)
- < 60%: Low quality (red)

**Enhancement:** Hover over the badge to see component breakdown: `Det: 95%, Sharp: 78%, Area: 74%`

This helps diagnose WHY a frame has low quality - is it the detection confidence, sharpness (blur), or face size?

---

## Quality Indicators

These status badges appear on clusters to provide quick visual cues:

| Badge | Color | Meaning |
|-------|-------|---------|
| ⚠️ Low Quality | Red | Cluster contains low-quality frames |
| 🔀 Mixed Identity | Orange | Faces may belong to different people |
| 👁️ Review Needed | Amber | Borderline confidence, verify |
| 🤖 Auto | Blue | Automatically assigned |
| ✋ Manual | Green | Manually verified |
| ✓ High Confidence | Green | Strong match, reliable |
| 📉 Few Faces | Gray | Not enough data for reliable matching |

---

## Sort Options

### Track Sorting
- Track Similarity (Low to High / High to Low)
- Frame Count (Low to High / High to Low)
- Average Frame Similarity (Low to High / High to Low)
- Track ID (Low to High / High to Low)
- Person Cohesion (Low to High / High to Low) *[renamed from Cast Track Score]*

### Cluster Sorting
- Cluster Similarity/Cohesion (High to Low / Low to High)
- Track Count (High to Low / Low to High)
- Face Count (High to Low / Low to High)
- Cast Match Score (High to Low / Low to High)
- Cluster ID (A-Z / Z-A)

### Person Sorting
- Impact (Clusters × Faces)
- Identity Similarity (High to Low / Low to High)
- Face/Track/Cluster Count (High to Low / Low to High)
- Average Cluster Similarity (High to Low / Low to High)
- Name (A-Z / Z-A)

---

## Threshold Standardization

All metrics now use standardized weak thresholds for consistency:

| Metric | Strong | Good | Weak (Review) |
|--------|--------|------|---------------|
| Identity | ≥75% | ≥70% | <70% |
| Cast | ≥68% | ≥50% | <50% |
| Track | ≥85% | ≥70% | <70% |
| Person Cohesion | ≥70% | ≥50% | <50% |
| Cluster | ≥80% | ≥60% | <60% |
| Temporal | ≥80% | ≥60% | <60% |
| Ambiguity | ≥15% | ≥8% | <8% |
| Isolation | ≥40% | ≥25% | <25% |

---

## Best Practices

1. **Start with Ambiguity**: Sort by ambiguity score to review the most uncertain assignments first
2. **Check Isolation**: Look for "Close" clusters that may need merging
3. **Monitor Trends**: Watch for degrading confidence trends - early warning of problems
4. **Use Temporal**: Low temporal scores may indicate lighting/costume issues OR misassignments
5. **Trust High Confidence**: Focus review time on uncertain cases, not high-confidence ones
6. **Check the Range**: A cluster with wide min-max range (e.g., 72% (30-95%)) has outliers to investigate
