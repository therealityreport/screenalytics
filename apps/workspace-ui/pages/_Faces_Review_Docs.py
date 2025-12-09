"""Faces Review Documentation Page.

This page provides comprehensive documentation for the Faces Review workflow,
including all metrics, thresholds, and best practices.

URL: /Faces_Review_Docs
"""

from __future__ import annotations

import sys
from pathlib import Path

# Setup path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ui_helpers as helpers  # noqa: E402
from metrics_strip import (  # noqa: E402
    METRIC_HELP,
    render_metrics_documentation,
    get_all_metric_keys,
)

# Page config
cfg = helpers.init_page("Faces Review Docs")
st.title("📖 Faces Review Documentation")
st.caption("Complete guide to Faces Review metrics, workflows, and best practices")

# ============================================================================
# OVERVIEW SECTION
# ============================================================================

st.header("Overview")
st.markdown("""
The **Faces Review** page is your central hub for reviewing and assigning face tracks
to cast members. The system automatically clusters detected faces into groups that
likely represent the same person, and you review/assign these clusters.

### Key Concepts

- **Track**: A sequence of face detections following one person through consecutive frames
- **Cluster**: A group of tracks that appear to be the same person
- **Person**: An auto-generated identity that may span multiple clusters
- **Cast Member**: A known person from the show's roster (linked from cast database)

### Workflow

1. **Review Smart Suggestions**: Start with clusters that have high-confidence suggestions
2. **Assign to Cast**: Link clusters to known cast members
3. **Handle Unknowns**: Review and potentially merge/split unknown clusters
4. **QC Low Confidence**: Review clusters with low similarity or ambiguous assignments
""")

# ============================================================================
# METRICS REFERENCE
# ============================================================================

st.header("Metrics Reference")
st.markdown("""
All metrics are displayed as color-coded badges. Hover over any badge to see the
full description. Click the ❓ icon on any metric for detailed help.
""")

# Core metrics first
st.subheader("Core Similarity Metrics")
CORE_METRICS = ["identity", "cast", "track", "cluster", "person_cohesion", "quality"]
render_metrics_documentation(CORE_METRICS, show_thresholds_table=False)

# New metrics section
st.subheader("New Metrics (November 2024)")
st.markdown("""
These advanced metrics provide deeper insights into assignment confidence and
potential issues:
""")
NEW_METRICS = ["temporal", "ambiguity", "isolation", "trend"]
render_metrics_documentation(NEW_METRICS, show_thresholds_table=False)

# ============================================================================
# THRESHOLD QUICK REFERENCE
# ============================================================================

st.header("Quick Reference: All Thresholds")

threshold_data = []
for key in get_all_metric_keys():
    data = METRIC_HELP.get(key, {})
    if data:
        threshold_data.append({
            "Metric": data.get("title", key.title()),
            "Thresholds": data.get("thresholds", "N/A"),
        })

st.table(threshold_data)

# ============================================================================
# COMMON WORKFLOWS
# ============================================================================

st.header("Common Workflows")

with st.expander("🎯 Assigning Clusters to Cast Members", expanded=False):
    st.markdown("""
    ### Steps:
    1. Find a cluster with a **Smart Suggestion** (blue banner with cast name)
    2. Check the **Cast Similarity** score:
       - ≥68%: High confidence, safe to assign
       - 50-67%: Review the thumbnails carefully
       - <50%: Probably wrong suggestion, verify manually
    3. Check the **Ambiguity Score**:
       - ≥15%: Clear match, go ahead
       - <8%: Risky! The 2nd best match is almost as good
    4. Click "Assign" or select a different cast member

    ### Tips:
    - Use "View" to see all frames in a track before assigning
    - Low temporal consistency might indicate costume changes (normal) or mismerged identities (bad)
    - Clusters with low isolation may be the same person as a nearby cluster
    """)

with st.expander("🔀 Splitting Clusters", expanded=False):
    st.markdown("""
    ### When to Split:
    - **Low Cluster Cohesion** (<60%): Tracks don't look alike
    - **Variable Temporal Consistency**: Appearance changes dramatically
    - **Visual inspection**: Different people in the thumbnails

    ### How to Split:
    1. Enter the Cluster View (click "View")
    2. Identify which tracks don't belong
    3. Use "Move Track" to reassign them to a different cluster
    """)

with st.expander("🔗 Merging Clusters", expanded=False):
    st.markdown("""
    ### When to Merge:
    - **Low Isolation** (<25%): Clusters are very similar
    - **Same person**: Visual inspection confirms they're the same
    - **Duplicate clusters**: Created during separate processing runs

    ### How to Merge:
    1. Assign both clusters to the same cast member
    2. Or use the "Merge" action in Person View
    """)

with st.expander("❓ Handling Unknown Clusters", expanded=False):
    st.markdown("""
    ### For clusters without suggestions:
    1. Check if they match any existing cast members manually
    2. If it's a recurring background person, create a new "Unknown" person
    3. If it's a one-off appearance, consider archiving

    ### For clusters with poor quality:
    - Low quality tracks may have unreliable face embeddings
    - Consider re-running detection with different settings
    - Archive truly unusable clusters
    """)

# ============================================================================
# VIEWS REFERENCE
# ============================================================================

st.header("Views Reference")

views_data = [
    {
        "View": "👥 People View",
        "Purpose": "Overview of all auto-detected people and unassigned clusters",
        "Key Metrics": "Identity Similarity, Cluster Cohesion, Temporal Consistency",
    },
    {
        "View": "👤 Person View",
        "Purpose": "Details of a single person's clusters across the episode",
        "Key Metrics": "Cluster Cohesion (per cluster), Isolation, Ambiguity",
    },
    {
        "View": "📦 Cluster View",
        "Purpose": "All tracks within a cluster with similarity scores",
        "Key Metrics": "Track Similarity, Person Cohesion, Quality",
    },
    {
        "View": "🖼️ Frames View",
        "Purpose": "All frames within a track for detailed QC",
        "Key Metrics": "Frame Quality (Det/Sharp/Area), Outlier detection",
    },
    {
        "View": "🎭 Cast Tracks View",
        "Purpose": "All tracks assigned to a cast member",
        "Key Metrics": "Person Cohesion, Temporal Consistency",
    },
]

st.table(views_data)

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

st.header("Troubleshooting")

with st.expander("⚠️ 'Crops missing' warning", expanded=False):
    st.markdown("""
    **Symptom:** Cluster preview shows an image, but "View frames" shows 0 crops.

    **Cause:** All faces in the track were marked as "skipped" due to quality filters.

    **Solution:**
    1. Enable "Show skipped faces" toggle
    2. Review the skipped faces - they may be usable
    3. Click "Unskip all" to include them

    See `docs/debugging/faces-manifest-crops-mismatch.md` for details.
    """)

with st.expander("🐌 Slow page loading", expanded=False):
    st.markdown("""
    **Possible causes:**
    - Large number of clusters (>500)
    - S3 thumbnail loading
    - Metrics computation for many clusters

    **Solutions:**
    1. Use the search/filter to narrow down clusters
    2. Start with "Smart Suggestions" tab (pre-filtered)
    3. Process episode in batches
    """)

with st.expander("🔄 Assignments not reflecting", expanded=False):
    st.markdown("""
    **Cache refresh needed:**
    1. Click the refresh button in the sidebar
    2. Or change the episode and change back
    3. Clear browser cache if persistent

    **If still not working:**
    - Check the API logs for errors
    - Verify the identities.json file was updated
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("""
📚 **Additional Resources:**
- `docs/similarity-scores-guide.md` - Full similarity scoring details
- `docs/debugging/faces-manifest-crops-mismatch.md` - Troubleshooting guide
- `docs/ops/faces_review_guide.md` - Operator's handbook
""")

st.caption("Last updated: December 2024 | SCREENALYTICS v2.0")
