# Similarity Scores Guide

**Color-Coded Reference for Face Recognition Similarity Metrics**

---

## Overview

The system uses **cosine similarity** (0.0-1.0 scale) to compare face embeddings at different stages of the pipeline. Each type of similarity comparison has a specific purpose and color code.

---

## Similarity Types & Color Codes

### 🟣 Cast Similarity (Purple #9C27B0)
**What:** Similarity between a cluster/person and a cast member's seed faces
**When:** During identity assignment and cast linking
**Range:** 0.0 - 1.0
**Thresholds:**
- ≥ 0.68: Strong match (auto-assign safe)
- 0.50 - 0.67: Possible match, needs review
- < 0.50: Weak match

### 🔵 Identity Similarity (Blue #2196F3)
**What:** How similar auto-generated people/clusters are to their centroid (person-level cohesion)
**When:** Cluster/people review, cohesion checks
**Range:** 0.0 - 1.0
**Thresholds:**
- ≥ 0.75: Strong match
- 0.60 - 0.74: Good match
- < 0.60: Weak match

### 🟢 Cluster Similarity (Green #4CAF50)
**What:** Cohesion of tracks inside a cluster (track-to-centroid similarity)
**When:** Cluster review/merge decisions
**Range:** 0.0 - 1.0
**Thresholds:**
- ≥ 0.80: Tight cluster
- 0.60 - 0.79: Moderate cohesion
- < 0.60: Loose cluster

### 🟠 Track Similarity (Orange #FF9800)
**What:** Consistency of frames within a track (frame-to-track centroid)
**When:** Track review/outlier detection
**Range:** 0.0 - 1.0
**Thresholds:**
- ≥ 0.85: Strong consistency
- 0.70 - 0.84: Good consistency
- < 0.70: Weak consistency

### 🟠 Frame Similarity (Light Orange #FFA726)
**What:** How similar a specific frame is to the rest of the track
**When:** Frame-level outlier checks
**Range:** 0.0 - 1.0
**Thresholds:**
- ≥ 0.80: Strong match
- 0.65 - 0.79: Good match
- < 0.65: Potential outlier

### 🟢 Quality Score (Green)
**What:** Detection confidence + sharpness + face area (displayed as `Q: XX%`)
**Thresholds:**
- ≥ 0.85: High quality
- 0.60 - 0.84: Medium quality
- < 0.60: Low quality
