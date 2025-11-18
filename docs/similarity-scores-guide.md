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
- ≥ 0.68: Strong match, can auto-assign
- 0.50 - 0.67: Possible match, needs review
- < 0.50: Weak match, likely different person

### 🔵 Identity Similarity (Blue #2196F3)
**What:** Similarity between a single frame and its track's identity centroid
**When:** During track representative selection and quality scoring
**Range:** 0.0 - 1.0
**Thresholds:**
- ≥ 0.75: Excellent identity match
- 0.60 - 0.74: Good match
- < 0.60: Poor match, may not be same person

### 🟢 Cluster Similarity (Green #4CAF50)
**What:** Similarity between cluster centroids during clustering
**When:** DBSCAN clustering phase, merging similar faces
**Range:** 0.0 - 1.0
**Thresholds:**
- ≥ 0.35: Same cluster (default CLUSTER_MIN_SIM)
- < 0.35: Different clusters

### 🟠 Track Similarity (Orange #FF9800)
**What:** Similarity between consecutive frames in a track (appearance gate)
**When:** Real-time during detect/track, frame-by-frame identity verification
**Range:** 0.0 - 1.0
**Thresholds:**
- ≥ 0.82: Continue track (soft threshold)
- 0.75 - 0.81: Low similarity streak, may split
- < 0.75: Force split track (hard threshold)

### 🔴 Assignment Confidence (Red-Amber Spectrum)
**What:** Meta-score combining similarity + ambiguity margin for auto-assignment
**When:** Auto-assigning clusters to people/cast
**Colors:**
- 🟢 **Strong** (#4CAF50): sim ≥ 0.68 AND margin ≥ 0.10 → Auto-assign
- 🟡 **Ambiguous** (#FFC107): sim ≥ 0.68 BUT margin < 0.10 → Manual review
- 🔴 **Weak** (#F44336): sim < 0.68 → Do not assign
