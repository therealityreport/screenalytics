"""Centralized similarity badge rendering for SCREENALYTICS workspace UI.

This module provides consistent color-coded badges for all similarity metrics:
- IDENTITY (Blue): How similar clusters are for AUTO-GENERATED PEOPLE
- CAST (Purple): How similar clusters are for CAST MEMBERS (facebank)
- TRACK (Orange): How similar FRAMES within a TRACK are to each other
- PERSON_COHESION (Teal): How similar a track is to other tracks assigned to same cast/identity
- CLUSTER (Green): How cohesive/similar all tracks in a cluster are

NEW METRICS (Nov 2024):
- TEMPORAL (Cyan): Consistency of appearance over time within episode
- AMBIGUITY (Red): How close 2nd-best match is to 1st (lower = risky)
- ISOLATION (Indigo): How far apart from nearest cluster (merge candidates)
- CONFIDENCE_TREND (Gray): Is confidence improving or degrading over time
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class SimilarityType(str, Enum):
    """Enumeration of all similarity metric types."""

    IDENTITY = "identity"  # Blue - auto-generated people's clusters
    CAST = "cast"  # Purple - cast member clusters
    TRACK = "track"  # Orange - track internal consistency (frames within track)
    PERSON_COHESION = "person_cohesion"  # Teal - track vs other tracks in same person (renamed from CAST_TRACK)
    CLUSTER = "cluster"  # Green - cluster cohesion (tracks within cluster)
    # NEW METRICS
    TEMPORAL = "temporal"  # Cyan - consistency over time within episode
    AMBIGUITY = "ambiguity"  # Red - margin between 1st and 2nd best match
    ISOLATION = "isolation"  # Indigo - distance to nearest cluster
    CONFIDENCE_TREND = "confidence_trend"  # Gray - trend of confidence over time

    # DEPRECATED - kept for backwards compatibility, only shown on outliers
    FRAME = "frame"  # Light Orange - individual frame vs track centroid (DEPRECATED)
    CAST_TRACK = "cast_track"  # DEPRECATED alias for PERSON_COHESION


class ColorScheme(NamedTuple):
    """Color configuration for a similarity type."""

    strong: str  # High confidence color
    good: str  # Medium confidence color
    weak: str  # Low confidence color
    strong_threshold: float  # Value >= this gets strong color
    good_threshold: float  # Value >= this gets good color (else weak)


# Standardized weak threshold: 50% across all types for consistency
# Strong/Good thresholds vary by metric type based on empirical observations

SIMILARITY_COLORS: dict[SimilarityType, ColorScheme] = {
    # CORE METRICS
    SimilarityType.IDENTITY: ColorScheme(
        strong="#2196F3",  # Blue
        good="#64B5F6",  # Light Blue
        weak="#BBDEFB",  # Very Light Blue
        strong_threshold=0.75,
        good_threshold=0.70,  # RAISED from 0.60 - 60-70% often has errors
    ),
    SimilarityType.CAST: ColorScheme(
        strong="#9C27B0",  # Purple
        good="#CE93D8",  # Light Purple
        weak="#F3E5F5",  # Very Light Purple
        strong_threshold=0.68,
        good_threshold=0.50,  # Standardized
    ),
    SimilarityType.TRACK: ColorScheme(
        strong="#FF9800",  # Orange
        good="#FFB74D",  # Light Orange
        weak="#FFE0B2",  # Very Light Orange
        strong_threshold=0.85,
        good_threshold=0.70,
    ),
    SimilarityType.PERSON_COHESION: ColorScheme(  # Renamed from CAST_TRACK
        strong="#00ACC1",  # Teal
        good="#4DD0E1",  # Light Teal
        weak="#B2EBF2",  # Very Light Teal
        strong_threshold=0.70,
        good_threshold=0.50,  # Standardized from 0.55
    ),
    SimilarityType.CLUSTER: ColorScheme(
        strong="#8BC34A",  # Green
        good="#C5E1A5",  # Light Green
        weak="#E0E0E0",  # Gray
        strong_threshold=0.80,
        good_threshold=0.60,
    ),
    # NEW METRICS (Nov 2024)
    SimilarityType.TEMPORAL: ColorScheme(
        strong="#00BCD4",  # Cyan
        good="#4DD0E1",  # Light Cyan
        weak="#B2EBF2",  # Very Light Cyan
        strong_threshold=0.80,  # Appearance consistent over time
        good_threshold=0.60,
    ),
    SimilarityType.AMBIGUITY: ColorScheme(
        strong="#4CAF50",  # Green (HIGH margin = GOOD = green)
        good="#FFC107",  # Amber (medium margin = caution)
        weak="#F44336",  # Red (LOW margin = ambiguous = danger)
        strong_threshold=0.15,  # Margin >= 15% = clear winner
        good_threshold=0.08,  # Margin >= 8% = acceptable
    ),
    SimilarityType.ISOLATION: ColorScheme(
        strong="#3F51B5",  # Indigo (well separated)
        good="#7986CB",  # Light Indigo
        weak="#C5CAE9",  # Very Light Indigo (too close, merge candidate)
        strong_threshold=0.40,  # Distance >= 0.40 = well isolated
        good_threshold=0.25,  # Distance >= 0.25 = moderately isolated
    ),
    SimilarityType.CONFIDENCE_TREND: ColorScheme(
        strong="#4CAF50",  # Green (improving)
        good="#607D8B",  # Gray (stable)
        weak="#F44336",  # Red (degrading)
        strong_threshold=0.02,  # Trend >= +2% = improving
        good_threshold=-0.02,  # Trend >= -2% = stable
    ),
    # DEPRECATED - kept for backwards compatibility
    SimilarityType.FRAME: ColorScheme(
        strong="#FFA726",  # Light Orange (distinct from Track)
        good="#FFCC80",  # Lighter Orange
        weak="#FFE0B2",  # Very Light Orange
        strong_threshold=0.80,
        good_threshold=0.50,  # Standardized from 0.65
    ),
    SimilarityType.CAST_TRACK: ColorScheme(  # DEPRECATED alias
        strong="#00ACC1",  # Teal
        good="#4DD0E1",  # Light Teal
        weak="#B2EBF2",  # Very Light Teal
        strong_threshold=0.70,
        good_threshold=0.50,
    ),
}

# Badge label abbreviations for compact display
SIMILARITY_LABELS: dict[SimilarityType, str] = {
    # Core metrics
    SimilarityType.IDENTITY: "ID",
    SimilarityType.CAST: "CAST",
    SimilarityType.TRACK: "TRK",
    SimilarityType.PERSON_COHESION: "FIT",  # Renamed from CAST_TRACK/MATCH
    SimilarityType.CLUSTER: "CLU",
    # New metrics (Nov 2024)
    SimilarityType.TEMPORAL: "TIME",
    SimilarityType.AMBIGUITY: "AMB",
    SimilarityType.ISOLATION: "ISO",
    SimilarityType.CONFIDENCE_TREND: "TREND",
    # Deprecated
    SimilarityType.FRAME: "FRM",
    SimilarityType.CAST_TRACK: "MATCH",  # Deprecated alias
}

# Colors that need dark text instead of white for readability
_LIGHT_COLORS = {
    "#E0E0E0", "#F3E5F5", "#BBDEFB", "#FFE0B2", "#C5E1A5", "#B2EBF2", "#FFCC80",
    "#C5CAE9",  # Very Light Indigo (ISOLATION weak)
    "#FFC107",  # Amber (AMBIGUITY good) - needs dark text
}


def get_badge_color(value: float, similarity_type: SimilarityType) -> str:
    """Get the appropriate color for a similarity value based on thresholds.

    Args:
        value: Similarity value between 0.0 and 1.0
        similarity_type: The type of similarity metric

    Returns:
        Hex color code string
    """
    scheme = SIMILARITY_COLORS[similarity_type]
    if value >= scheme.strong_threshold:
        return scheme.strong
    elif value >= scheme.good_threshold:
        return scheme.good
    return scheme.weak


def render_similarity_badge(
    similarity: float | None,
    similarity_type: SimilarityType = SimilarityType.IDENTITY,
    *,
    show_label: bool = False,
    custom_label: str | None = None,
) -> str:
    """Render a similarity score as a color-coded HTML badge.

    Args:
        similarity: Value between 0.0 and 1.0, or None
        similarity_type: The type of similarity metric (determines color scheme)
        show_label: If True, prefix badge with type abbreviation (e.g., "ID: 85%")
        custom_label: Override the default label abbreviation

    Returns:
        HTML string for the badge, or empty string if similarity is None
    """
    if similarity is None:
        return ""

    value = max(0.0, min(float(similarity), 1.0))
    pct = int(round(value * 100))
    color = get_badge_color(value, similarity_type)

    # Determine text color based on background brightness
    text_color = "#333333" if color in _LIGHT_COLORS else "#FFFFFF"

    # Build label prefix
    label_prefix = ""
    if show_label or custom_label:
        label = custom_label or SIMILARITY_LABELS.get(similarity_type, "")
        if label:
            label_prefix = f"{label}: "

    return (
        f'<span class="sim-badge sim-badge-{similarity_type.value}" '
        f'style="background-color: {color}; color: {text_color}; '
        f'padding: 2px 6px; border-radius: 3px; font-size: 0.8em; font-weight: bold;">'
        f"{label_prefix}{pct}%</span>"
    )


def render_similarity_badge_compat(
    similarity: float | None,
    metric: str = "identity",
) -> str:
    """Backwards-compatible wrapper for existing code.

    Maps old string metrics to new SimilarityType enum.

    Args:
        similarity: Value between 0.0 and 1.0, or None
        metric: One of "identity", "cast", "track", "frame", "cluster", "cast_track"

    Returns:
        HTML string for the badge
    """
    metric_map = {
        "identity": SimilarityType.IDENTITY,
        "cast": SimilarityType.CAST,
        "track": SimilarityType.TRACK,
        "frame": SimilarityType.FRAME,
        "cluster": SimilarityType.CLUSTER,
        "cast_track": SimilarityType.CAST_TRACK,
    }
    sim_type = metric_map.get((metric or "identity").lower(), SimilarityType.IDENTITY)
    return render_similarity_badge(similarity, sim_type)


def inject_similarity_badge_css() -> None:
    """Inject CSS styles for similarity badges (call once per page).

    This provides consistent styling and hover effects for badges.
    """
    import streamlit as st

    css = """
    <style>
    .sim-badge {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.8em;
        font-weight: bold;
        margin: 0 2px;
        vertical-align: middle;
    }
    .sim-badge-identity { /* Blue family */ }
    .sim-badge-cast { /* Purple family */ }
    .sim-badge-track { /* Orange family */ }
    .sim-badge-frame { /* Orange family */ }
    .sim-badge-cluster { /* Green family */ }
    .sim-badge-cast_track { /* Teal family */ }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def get_similarity_key_data(include_deprecated: bool = False) -> list[dict]:
    """Return data for rendering the similarity scores legend/key.

    Args:
        include_deprecated: If True, include deprecated metrics like FRAME

    Returns:
        List of dicts with keys: type, color, label, description, thresholds
    """
    core_metrics = [
        {
            "type": SimilarityType.IDENTITY,
            "color": SIMILARITY_COLORS[SimilarityType.IDENTITY].strong,
            "label": "Identity Similarity",
            "emoji": "🔵",
            "description": "How similar clusters are for AUTO-GENERATED PEOPLE",
            "thresholds": f"≥{int(SIMILARITY_COLORS[SimilarityType.IDENTITY].strong_threshold*100)}%: Strong, "
            f"≥{int(SIMILARITY_COLORS[SimilarityType.IDENTITY].good_threshold*100)}%: Good, "
            f"<{int(SIMILARITY_COLORS[SimilarityType.IDENTITY].good_threshold*100)}%: Review",
        },
        {
            "type": SimilarityType.CAST,
            "color": SIMILARITY_COLORS[SimilarityType.CAST].strong,
            "label": "Cast Similarity",
            "emoji": "🟣",
            "description": "How similar clusters are for CAST MEMBERS (facebank). Shows rank: #1 of N",
            "thresholds": f"≥{int(SIMILARITY_COLORS[SimilarityType.CAST].strong_threshold*100)}%: Strong, "
            f"≥{int(SIMILARITY_COLORS[SimilarityType.CAST].good_threshold*100)}%: Good",
        },
        {
            "type": SimilarityType.TRACK,
            "color": SIMILARITY_COLORS[SimilarityType.TRACK].strong,
            "label": "Track Similarity",
            "emoji": "🟠",
            "description": "How consistent FRAMES within a TRACK are. Shows excluded frames count",
            "thresholds": f"≥{int(SIMILARITY_COLORS[SimilarityType.TRACK].strong_threshold*100)}%: Strong, "
            f"≥{int(SIMILARITY_COLORS[SimilarityType.TRACK].good_threshold*100)}%: Good",
        },
        {
            "type": SimilarityType.PERSON_COHESION,
            "color": SIMILARITY_COLORS[SimilarityType.PERSON_COHESION].strong,
            "label": "Person Cohesion",
            "emoji": "🔷",
            "description": "How well a track fits with other tracks assigned to the same person",
            "thresholds": f"≥{int(SIMILARITY_COLORS[SimilarityType.PERSON_COHESION].strong_threshold*100)}%: Strong, "
            f"≥{int(SIMILARITY_COLORS[SimilarityType.PERSON_COHESION].good_threshold*100)}%: Good",
        },
        {
            "type": SimilarityType.CLUSTER,
            "color": SIMILARITY_COLORS[SimilarityType.CLUSTER].strong,
            "label": "Cluster Cohesion",
            "emoji": "🟢",
            "description": "How cohesive/similar all tracks in a cluster are. Shows min-max range",
            "thresholds": f"≥{int(SIMILARITY_COLORS[SimilarityType.CLUSTER].strong_threshold*100)}%: Strong, "
            f"≥{int(SIMILARITY_COLORS[SimilarityType.CLUSTER].good_threshold*100)}%: Good",
        },
    ]

    new_metrics = [
        {
            "type": SimilarityType.TEMPORAL,
            "color": SIMILARITY_COLORS[SimilarityType.TEMPORAL].strong,
            "label": "Temporal Consistency",
            "emoji": "🔹",
            "description": "How consistent person's appearance is across time within episode",
            "thresholds": f"≥{int(SIMILARITY_COLORS[SimilarityType.TEMPORAL].strong_threshold*100)}%: Consistent, "
            f"≥{int(SIMILARITY_COLORS[SimilarityType.TEMPORAL].good_threshold*100)}%: Variable",
        },
        {
            "type": SimilarityType.AMBIGUITY,
            "color": SIMILARITY_COLORS[SimilarityType.AMBIGUITY].strong,
            "label": "Ambiguity Score",
            "emoji": "⚠️",
            "description": "Gap between 1st and 2nd best match. LOW = risky, HIGH = clear winner",
            "thresholds": f"≥{int(SIMILARITY_COLORS[SimilarityType.AMBIGUITY].strong_threshold*100)}%: Clear, "
            f"≥{int(SIMILARITY_COLORS[SimilarityType.AMBIGUITY].good_threshold*100)}%: OK, "
            f"<{int(SIMILARITY_COLORS[SimilarityType.AMBIGUITY].good_threshold*100)}%: Risky",
        },
        {
            "type": SimilarityType.ISOLATION,
            "color": SIMILARITY_COLORS[SimilarityType.ISOLATION].strong,
            "label": "Cluster Isolation",
            "emoji": "🔮",
            "description": "Distance to nearest cluster. LOW = merge candidate, HIGH = distinct",
            "thresholds": f"≥{int(SIMILARITY_COLORS[SimilarityType.ISOLATION].strong_threshold*100)}%: Isolated, "
            f"≥{int(SIMILARITY_COLORS[SimilarityType.ISOLATION].good_threshold*100)}%: Moderate",
        },
        {
            "type": SimilarityType.CONFIDENCE_TREND,
            "color": SIMILARITY_COLORS[SimilarityType.CONFIDENCE_TREND].strong,
            "label": "Confidence Trend",
            "emoji": "📈",
            "description": "Is assignment confidence improving (↑), stable (→), or degrading (↓)?",
            "thresholds": "↑ Improving, → Stable, ↓ Degrading",
        },
    ]

    result = core_metrics + new_metrics

    if include_deprecated:
        result.append({
            "type": SimilarityType.FRAME,
            "color": SIMILARITY_COLORS[SimilarityType.FRAME].strong,
            "label": "Frame Outlier (Deprecated)",
            "emoji": "🟠",
            "description": "Only shown on outlier frames that differ significantly from track",
            "thresholds": f"Shown when <{int(SIMILARITY_COLORS[SimilarityType.FRAME].good_threshold*100)}% (outlier)",
        })

    return result


# ============================================================================
# SORT OPTIONS
# ============================================================================

# Track sort options for track listings
TRACK_SORT_OPTIONS = [
    "Track Similarity (Low to High)",
    "Track Similarity (High to Low)",
    "Frame Count (Low to High)",
    "Frame Count (High to Low)",
    "Average Frame Similarity (Low to High)",
    "Average Frame Similarity (High to Low)",
    "Track ID (Low to High)",
    "Track ID (High to Low)",
]

# Cluster sort options for cluster listings
CLUSTER_SORT_OPTIONS = [
    "Cluster Similarity (High to Low)",
    "Cluster Similarity (Low to High)",
    "Track Count (High to Low)",
    "Track Count (Low to High)",
    "Face Count (High to Low)",
    "Face Count (Low to High)",
    "Cluster ID (A-Z)",
    "Cluster ID (Z-A)",
]

# Unassigned cluster sort options (includes Cast Similarity from suggestions)
UNASSIGNED_CLUSTER_SORT_OPTIONS = [
    "Face Count (High to Low)",
    "Face Count (Low to High)",
    "Track Count (High to Low)",
    "Track Count (Low to High)",
    "Cast Match Score (High to Low)",
    "Cast Match Score (Low to High)",
    "Cluster Similarity (High to Low)",
    "Cluster Similarity (Low to High)",
    "Cluster ID (A-Z)",
    "Cluster ID (Z-A)",
]

# Person sort options for auto-clustered people listings
PERSON_SORT_OPTIONS = [
    "Impact (Clusters × Faces)",
    "Identity Similarity (High to Low)",
    "Identity Similarity (Low to High)",
    "Face Count (High to Low)",
    "Face Count (Low to High)",
    "Track Count (High to Low)",
    "Track Count (Low to High)",
    "Cluster Count (High to Low)",
    "Cluster Count (Low to High)",
    "Average Cluster Similarity (High to Low)",
    "Average Cluster Similarity (Low to High)",
    "Name (A-Z)",
    "Name (Z-A)",
]

# Cast member all-tracks sort options (for outlier detection)
CAST_TRACKS_SORT_OPTIONS = [
    "Person Cohesion (Low to High)",  # Renamed from Cast Track Score - outliers first
    "Person Cohesion (High to Low)",
    "Track Similarity (Low to High)",  # Outliers first
    "Track Similarity (High to Low)",
    "Frame Count (Low to High)",
    "Frame Count (High to Low)",
    "Cluster ID (A-Z)",
    "Cluster ID (Z-A)",
    "Track ID (Low to High)",
    "Track ID (High to Low)",
]


def sort_tracks(
    tracks: list[dict],
    sort_option: str,
    track_meta_getter: callable = None,
) -> list[dict]:
    """Sort a list of track dictionaries based on the selected sort option.

    Args:
        tracks: List of track dictionaries
        sort_option: One of TRACK_SORT_OPTIONS
        track_meta_getter: Optional callable to fetch track metadata (for frame counts/similarities)

    Returns:
        Sorted list of tracks (sorts in place and returns)
    """
    if sort_option == "Track Similarity (Low to High)":
        tracks.sort(key=lambda t: (t.get("similarity") if t.get("similarity") is not None else 999.0))
    elif sort_option == "Track Similarity (High to Low)":
        tracks.sort(
            key=lambda t: (t.get("similarity") if t.get("similarity") is not None else -999.0),
            reverse=True,
        )
    elif sort_option == "Frame Count (Low to High)":
        if track_meta_getter:
            for track in tracks:
                meta = track_meta_getter(track.get("track_int") or track.get("track_id"))
                frame_count = meta.get("faces_count") or meta.get("frames_count") or len(meta.get("frames", []) or [])
                track["_frame_count"] = int(frame_count) if frame_count else 0
        tracks.sort(key=lambda t: t.get("_frame_count", t.get("frames", 0)))
    elif sort_option == "Frame Count (High to Low)":
        if track_meta_getter:
            for track in tracks:
                meta = track_meta_getter(track.get("track_int") or track.get("track_id"))
                frame_count = meta.get("faces_count") or meta.get("frames_count") or len(meta.get("frames", []) or [])
                track["_frame_count"] = int(frame_count) if frame_count else 0
        tracks.sort(key=lambda t: t.get("_frame_count", t.get("frames", 0)), reverse=True)
    elif sort_option == "Average Frame Similarity (Low to High)":
        if track_meta_getter:
            for track in tracks:
                meta = track_meta_getter(track.get("track_int") or track.get("track_id"))
                frames = meta.get("frames", []) or []
                similarities = [f.get("similarity") for f in frames if f.get("similarity") is not None]
                track["_avg_frame_sim"] = sum(similarities) / len(similarities) if similarities else 999.0
        tracks.sort(key=lambda t: t.get("_avg_frame_sim", 999.0))
    elif sort_option == "Average Frame Similarity (High to Low)":
        if track_meta_getter:
            for track in tracks:
                meta = track_meta_getter(track.get("track_int") or track.get("track_id"))
                frames = meta.get("frames", []) or []
                similarities = [f.get("similarity") for f in frames if f.get("similarity") is not None]
                track["_avg_frame_sim"] = sum(similarities) / len(similarities) if similarities else -999.0
        tracks.sort(key=lambda t: t.get("_avg_frame_sim", -999.0), reverse=True)
    elif sort_option == "Track ID (Low to High)":
        tracks.sort(
            key=lambda t: (
                int(str(t.get("track_id", "0")).replace("track_", ""))
                if t.get("track_id")
                else 0
            )
        )
    elif sort_option == "Track ID (High to Low)":
        tracks.sort(
            key=lambda t: (
                int(str(t.get("track_id", "0")).replace("track_", ""))
                if t.get("track_id")
                else 0
            ),
            reverse=True,
        )
    elif sort_option in ("Person Cohesion (Low to High)", "Cast Track Score (Low to High)"):
        # Sort by person_cohesion/cast_track_score - how similar track is to other tracks of same person
        tracks.sort(
            key=lambda t: (
                t.get("person_cohesion") or t.get("cast_track_score")
                if (t.get("person_cohesion") or t.get("cast_track_score")) is not None
                else 999.0
            )
        )
    elif sort_option in ("Person Cohesion (High to Low)", "Cast Track Score (High to Low)"):
        tracks.sort(
            key=lambda t: (
                t.get("person_cohesion") or t.get("cast_track_score")
                if (t.get("person_cohesion") or t.get("cast_track_score")) is not None
                else -999.0
            ),
            reverse=True,
        )
    elif sort_option == "Cluster ID (A-Z)":
        tracks.sort(key=lambda t: t.get("cluster_id", ""))
    elif sort_option == "Cluster ID (Z-A)":
        tracks.sort(key=lambda t: t.get("cluster_id", ""), reverse=True)
    return tracks


def sort_clusters(
    clusters: list[dict],
    sort_option: str,
    cast_suggestions: dict = None,
) -> list[dict]:
    """Sort a list of cluster dictionaries based on the selected sort option.

    Args:
        clusters: List of cluster dictionaries
        sort_option: One of CLUSTER_SORT_OPTIONS or UNASSIGNED_CLUSTER_SORT_OPTIONS
        cast_suggestions: Optional dict mapping cluster_id to suggestion data (for Cast Match Score)

    Returns:
        Sorted list of clusters (sorts in place and returns)
    """
    cast_suggestions = cast_suggestions or {}

    if sort_option == "Cluster Similarity (High to Low)":
        clusters.sort(
            key=lambda c: (c.get("cohesion") if c.get("cohesion") is not None else -999.0),
            reverse=True,
        )
    elif sort_option == "Cluster Similarity (Low to High)":
        clusters.sort(
            key=lambda c: (c.get("cohesion") if c.get("cohesion") is not None else 999.0)
        )
    elif sort_option == "Track Count (High to Low)":
        clusters.sort(key=lambda c: c.get("tracks", 0), reverse=True)
    elif sort_option == "Track Count (Low to High)":
        clusters.sort(key=lambda c: c.get("tracks", 0))
    elif sort_option == "Face Count (High to Low)":
        clusters.sort(key=lambda c: c.get("faces", 0), reverse=True)
    elif sort_option == "Face Count (Low to High)":
        clusters.sort(key=lambda c: c.get("faces", 0))
    elif sort_option == "Cast Match Score (High to Low)":
        def get_cast_score(c):
            cid = c.get("cluster_id")
            sugg = cast_suggestions.get(cid)
            if sugg and isinstance(sugg, dict):
                return sugg.get("similarity", sugg.get("score", -999.0))
            return -999.0
        clusters.sort(key=get_cast_score, reverse=True)
    elif sort_option == "Cast Match Score (Low to High)":
        def get_cast_score(c):
            cid = c.get("cluster_id")
            sugg = cast_suggestions.get(cid)
            if sugg and isinstance(sugg, dict):
                return sugg.get("similarity", sugg.get("score", 999.0))
            return 999.0
        clusters.sort(key=get_cast_score)
    elif sort_option == "Cluster ID (A-Z)":
        clusters.sort(key=lambda c: c.get("cluster_id", ""))
    elif sort_option == "Cluster ID (Z-A)":
        clusters.sort(key=lambda c: c.get("cluster_id", ""), reverse=True)
    return clusters


def sort_people(
    people: list[dict],
    sort_option: str,
) -> list[dict]:
    """Sort a list of auto-clustered people entries based on the selected sort option.

    Args:
        people: List of people entry dicts with structure:
            {
                "person": {...},
                "episode_clusters": [...],
                "counts": {"clusters": N, "tracks": N, "faces": N},
                "avg_cohesion": float or None
            }
        sort_option: One of PERSON_SORT_OPTIONS

    Returns:
        Sorted list of people (sorts in place and returns)
    """
    if sort_option == "Impact (Clusters × Faces)":
        # Sort by clusters (desc), then tracks (desc), then faces (desc), then name
        people.sort(
            key=lambda entry: (
                -(entry.get("counts", {}).get("clusters") or 0),
                -(entry.get("counts", {}).get("tracks") or 0),
                -(entry.get("counts", {}).get("faces") or 0),
                (entry.get("person", {}).get("name") or "").lower(),
            )
        )
    elif sort_option == "Identity Similarity (High to Low)":
        # Sort by avg_cohesion (identity similarity) - higher is better
        people.sort(
            key=lambda e: (e.get("avg_cohesion") if e.get("avg_cohesion") is not None else -999.0),
            reverse=True,
        )
    elif sort_option == "Identity Similarity (Low to High)":
        # Sort by avg_cohesion - lower first (potential outliers)
        people.sort(
            key=lambda e: (e.get("avg_cohesion") if e.get("avg_cohesion") is not None else 999.0)
        )
    elif sort_option == "Face Count (High to Low)":
        people.sort(key=lambda e: e.get("counts", {}).get("faces", 0), reverse=True)
    elif sort_option == "Face Count (Low to High)":
        people.sort(key=lambda e: e.get("counts", {}).get("faces", 0))
    elif sort_option == "Track Count (High to Low)":
        people.sort(key=lambda e: e.get("counts", {}).get("tracks", 0), reverse=True)
    elif sort_option == "Track Count (Low to High)":
        people.sort(key=lambda e: e.get("counts", {}).get("tracks", 0))
    elif sort_option == "Cluster Count (High to Low)":
        people.sort(key=lambda e: e.get("counts", {}).get("clusters", 0), reverse=True)
    elif sort_option == "Cluster Count (Low to High)":
        people.sort(key=lambda e: e.get("counts", {}).get("clusters", 0))
    elif sort_option == "Average Cluster Similarity (High to Low)":
        people.sort(
            key=lambda e: (e.get("avg_cohesion") if e.get("avg_cohesion") is not None else -999.0),
            reverse=True,
        )
    elif sort_option == "Average Cluster Similarity (Low to High)":
        people.sort(
            key=lambda e: (e.get("avg_cohesion") if e.get("avg_cohesion") is not None else 999.0)
        )
    elif sort_option == "Name (A-Z)":
        people.sort(key=lambda e: (e.get("person", {}).get("name") or "").lower())
    elif sort_option == "Name (Z-A)":
        people.sort(key=lambda e: (e.get("person", {}).get("name") or "").lower(), reverse=True)
    return people


# ============================================================================
# CLUSTER QUALITY INDICATORS (Feature 10)
# ============================================================================

class QualityIndicator(str, Enum):
    """Quality indicator types for clusters."""

    LOW_QUALITY = "low_quality"  # Cluster has poor frame quality
    MIXED_IDENTITY = "mixed_identity"  # Faces in cluster look different (low cohesion)
    REVIEW_NEEDED = "review_needed"  # Borderline confidence assignment
    AUTO_ASSIGNED = "auto_assigned"  # Cluster was auto-assigned
    MANUALLY_ASSIGNED = "manually_assigned"  # Cluster was manually assigned
    HIGH_CONFIDENCE = "high_confidence"  # Strong match confidence
    NEEDS_MORE_FACES = "needs_more_faces"  # Too few faces for reliable matching


# Quality indicator colors and icons
QUALITY_INDICATORS = {
    QualityIndicator.LOW_QUALITY: {
        "color": "#F44336",  # Red
        "icon": "⚠️",
        "label": "Low Quality",
        "description": "This cluster contains low-quality frames (blurry, partial faces)",
    },
    QualityIndicator.MIXED_IDENTITY: {
        "color": "#FF9800",  # Orange
        "icon": "🔀",
        "label": "Mixed Identity",
        "description": "Faces in this cluster may belong to different people (low cohesion)",
    },
    QualityIndicator.REVIEW_NEEDED: {
        "color": "#FFC107",  # Amber
        "icon": "👁️",
        "label": "Review Needed",
        "description": "Borderline confidence - manual review recommended",
    },
    QualityIndicator.AUTO_ASSIGNED: {
        "color": "#2196F3",  # Blue
        "icon": "🤖",
        "label": "Auto",
        "description": "This cluster was automatically assigned",
    },
    QualityIndicator.MANUALLY_ASSIGNED: {
        "color": "#4CAF50",  # Green
        "icon": "✋",
        "label": "Manual",
        "description": "This cluster was manually assigned",
    },
    QualityIndicator.HIGH_CONFIDENCE: {
        "color": "#4CAF50",  # Green
        "icon": "✓",
        "label": "High Confidence",
        "description": "Strong match with high confidence",
    },
    QualityIndicator.NEEDS_MORE_FACES: {
        "color": "#9E9E9E",  # Gray
        "icon": "📉",
        "label": "Few Faces",
        "description": "Too few faces for reliable matching",
    },
}


def render_quality_indicator(
    indicator: QualityIndicator,
    *,
    show_icon: bool = True,
    show_label: bool = True,
    compact: bool = False,
) -> str:
    """Render a quality indicator badge.

    Args:
        indicator: The quality indicator type
        show_icon: Include emoji icon
        show_label: Include text label
        compact: Use smaller padding

    Returns:
        HTML string for the indicator badge
    """
    config = QUALITY_INDICATORS.get(indicator)
    if not config:
        return ""

    color = config["color"]
    icon = config["icon"] if show_icon else ""
    label = config["label"] if show_label else ""

    # Determine text color
    text_color = "#FFFFFF" if color not in {"#FFC107", "#E0E0E0"} else "#333333"

    # Build content
    content_parts = []
    if icon:
        content_parts.append(icon)
    if label:
        content_parts.append(label)
    content = " ".join(content_parts)

    padding = "1px 4px" if compact else "2px 6px"
    font_size = "0.75em" if compact else "0.8em"

    return (
        f'<span class="quality-indicator quality-{indicator.value}" '
        f'title="{config["description"]}" '
        f'style="background-color: {color}; color: {text_color}; '
        f'padding: {padding}; border-radius: 3px; font-size: {font_size}; '
        f'font-weight: bold; cursor: help; white-space: nowrap;">'
        f"{content}</span>"
    )


def get_cluster_quality_indicators(
    cluster: dict,
    *,
    cohesion_threshold_low: float = 0.50,
    cohesion_threshold_mixed: float = 0.60,
    quality_threshold_low: float = 0.50,
    min_faces_reliable: int = 3,
    confidence_threshold_review: float = 0.65,
    confidence_threshold_high: float = 0.80,
) -> list[QualityIndicator]:
    """Determine which quality indicators apply to a cluster.

    Args:
        cluster: Cluster dictionary with keys like:
            - cohesion: float (0-1)
            - avg_quality: float (0-1)
            - faces: int
            - assignment_type: "auto" or "manual"
            - confidence: float (0-1) or None
        cohesion_threshold_low: Below this triggers MIXED_IDENTITY
        cohesion_threshold_mixed: Below this suggests review
        quality_threshold_low: Below this triggers LOW_QUALITY
        min_faces_reliable: Fewer faces triggers NEEDS_MORE_FACES
        confidence_threshold_review: Below this triggers REVIEW_NEEDED
        confidence_threshold_high: Above this triggers HIGH_CONFIDENCE

    Returns:
        List of applicable QualityIndicator values
    """
    indicators = []

    # Check cohesion for mixed identity
    cohesion = cluster.get("cohesion")
    if cohesion is not None:
        if cohesion < cohesion_threshold_low:
            indicators.append(QualityIndicator.MIXED_IDENTITY)
        elif cohesion < cohesion_threshold_mixed:
            indicators.append(QualityIndicator.REVIEW_NEEDED)

    # Check average quality
    avg_quality = cluster.get("avg_quality") or cluster.get("quality")
    if avg_quality is not None and avg_quality < quality_threshold_low:
        indicators.append(QualityIndicator.LOW_QUALITY)

    # Check face count
    faces = cluster.get("faces", 0)
    if faces < min_faces_reliable:
        indicators.append(QualityIndicator.NEEDS_MORE_FACES)

    # Check assignment type
    assignment_type = cluster.get("assignment_type") or cluster.get("assigned_by")
    if assignment_type == "manual":
        indicators.append(QualityIndicator.MANUALLY_ASSIGNED)
    elif assignment_type == "auto":
        indicators.append(QualityIndicator.AUTO_ASSIGNED)

    # Check confidence
    confidence = cluster.get("confidence") or cluster.get("similarity")
    if confidence is not None:
        if confidence >= confidence_threshold_high:
            indicators.append(QualityIndicator.HIGH_CONFIDENCE)
        elif confidence < confidence_threshold_review:
            if QualityIndicator.REVIEW_NEEDED not in indicators:
                indicators.append(QualityIndicator.REVIEW_NEEDED)

    return indicators


def render_cluster_quality_badges(
    cluster: dict,
    *,
    max_badges: int = 3,
    compact: bool = False,
    show_icons: bool = True,
) -> str:
    """Render all applicable quality indicator badges for a cluster.

    Args:
        cluster: Cluster dictionary
        max_badges: Maximum number of badges to show
        compact: Use compact badge style
        show_icons: Show emoji icons

    Returns:
        HTML string with all applicable badges
    """
    indicators = get_cluster_quality_indicators(cluster)

    # Priority order for display (most important first)
    priority_order = [
        QualityIndicator.MIXED_IDENTITY,
        QualityIndicator.LOW_QUALITY,
        QualityIndicator.REVIEW_NEEDED,
        QualityIndicator.NEEDS_MORE_FACES,
        QualityIndicator.HIGH_CONFIDENCE,
        QualityIndicator.MANUALLY_ASSIGNED,
        QualityIndicator.AUTO_ASSIGNED,
    ]

    # Sort by priority and limit
    sorted_indicators = [ind for ind in priority_order if ind in indicators][:max_badges]

    if not sorted_indicators:
        return ""

    badges = [
        render_quality_indicator(ind, show_icon=show_icons, compact=compact)
        for ind in sorted_indicators
    ]
    return " ".join(badges)


# ============================================================================
# ENHANCED RENDERING FUNCTIONS (Nov 2024)
# ============================================================================


def render_cluster_range_badge(
    cohesion: float | None,
    min_sim: float | None = None,
    max_sim: float | None = None,
) -> str:
    """Render cluster cohesion with min-max range.

    Example output: "72% (58-89%)" showing cohesion with range

    Args:
        cohesion: Average cohesion value (0-1)
        min_sim: Minimum similarity in cluster
        max_sim: Maximum similarity in cluster

    Returns:
        HTML badge showing cohesion with optional range
    """
    if cohesion is None:
        return ""

    pct = int(round(cohesion * 100))
    color = get_badge_color(cohesion, SimilarityType.CLUSTER)
    text_color = "#333333" if color in _LIGHT_COLORS else "#FFFFFF"

    # Build range text
    range_text = ""
    if min_sim is not None and max_sim is not None:
        min_pct = int(round(min_sim * 100))
        max_pct = int(round(max_sim * 100))
        range_text = f" ({min_pct}-{max_pct}%)"

    return (
        f'<span class="sim-badge sim-badge-cluster-range" '
        f'style="background-color: {color}; color: {text_color}; '
        f'padding: 2px 6px; border-radius: 3px; font-size: 0.8em; font-weight: bold;" '
        f'title="Cluster cohesion with min-max range">'
        f"CLU: {pct}%{range_text}</span>"
    )


def render_quality_breakdown_badge(
    quality: float | None,
    detection: float | None = None,
    sharpness: float | None = None,
    area: float | None = None,
) -> str:
    """Render quality score with component breakdown on hover.

    Example: "Q: 82%" with tooltip "Det: 95%, Sharp: 78%, Area: 74%"

    Args:
        quality: Composite quality score (0-1)
        detection: Detection confidence component
        sharpness: Sharpness component
        area: Face area component

    Returns:
        HTML badge with quality and tooltip breakdown
    """
    if quality is None:
        return ""

    pct = int(round(quality * 100))

    # Determine color based on quality
    if pct >= 85:
        color = "#4CAF50"  # Green
    elif pct >= 60:
        color = "#FFC107"  # Amber
    else:
        color = "#F44336"  # Red

    text_color = "#333333" if color == "#FFC107" else "#FFFFFF"

    # Build tooltip with breakdown
    breakdown_parts = []
    if detection is not None:
        breakdown_parts.append(f"Det: {int(round(detection * 100))}%")
    if sharpness is not None:
        breakdown_parts.append(f"Sharp: {int(round(sharpness * 100))}%")
    if area is not None:
        breakdown_parts.append(f"Area: {int(round(area * 100))}%")

    tooltip = ", ".join(breakdown_parts) if breakdown_parts else "Quality score"

    return (
        f'<span class="sim-badge sim-badge-quality" '
        f'style="background-color: {color}; color: {text_color}; '
        f'padding: 2px 6px; border-radius: 3px; font-size: 0.8em; font-weight: bold; cursor: help;" '
        f'title="{tooltip}">'
        f"Q: {pct}%</span>"
    )


def render_cast_rank_badge(
    similarity: float | None,
    rank: int = 1,
    total_suggestions: int = 1,
    cast_name: str | None = None,
) -> str:
    """Render cast similarity with rank context.

    Example: "68% (#1 of 5)" showing this is the top suggestion

    Args:
        similarity: Similarity to cast member (0-1)
        rank: This suggestion's rank (1 = best match)
        total_suggestions: Total number of cast suggestions
        cast_name: Optional cast member name for tooltip

    Returns:
        HTML badge showing similarity with rank
    """
    if similarity is None:
        return ""

    pct = int(round(similarity * 100))
    color = get_badge_color(similarity, SimilarityType.CAST)
    text_color = "#333333" if color in _LIGHT_COLORS else "#FFFFFF"

    rank_text = f" (#{rank}" if total_suggestions > 1 else ""
    if total_suggestions > 1:
        rank_text += f" of {total_suggestions})"

    tooltip = f"Match to {cast_name}" if cast_name else "Cast match similarity"
    if rank == 1 and total_suggestions > 1:
        tooltip += " - Best match"

    return (
        f'<span class="sim-badge sim-badge-cast-rank" '
        f'style="background-color: {color}; color: {text_color}; '
        f'padding: 2px 6px; border-radius: 3px; font-size: 0.8em; font-weight: bold; cursor: help;" '
        f'title="{tooltip}">'
        f"CAST: {pct}%{rank_text}</span>"
    )


def render_track_with_dropout(
    similarity: float | None,
    excluded_frames: int = 0,
    total_frames: int = 0,
) -> str:
    """Render track similarity with frame dropout indicator.

    Example: "TRK: 85% (3 excluded)" showing frames excluded from centroid

    Args:
        similarity: Track similarity (0-1)
        excluded_frames: Number of frames excluded from centroid calculation
        total_frames: Total frames in track (for context)

    Returns:
        HTML badge showing similarity with exclusion count
    """
    if similarity is None:
        return ""

    pct = int(round(similarity * 100))
    color = get_badge_color(similarity, SimilarityType.TRACK)
    text_color = "#333333" if color in _LIGHT_COLORS else "#FFFFFF"

    exclusion_text = ""
    if excluded_frames > 0:
        exclusion_text = f" ({excluded_frames} excl)"

    tooltip = f"Track consistency"
    if excluded_frames > 0:
        tooltip += f" - {excluded_frames} frame(s) excluded due to quality"

    return (
        f'<span class="sim-badge sim-badge-track-dropout" '
        f'style="background-color: {color}; color: {text_color}; '
        f'padding: 2px 6px; border-radius: 3px; font-size: 0.8em; font-weight: bold; cursor: help;" '
        f'title="{tooltip}">'
        f"TRK: {pct}%{exclusion_text}</span>"
    )


def render_outlier_severity_badge(
    similarity: float | None,
    track_mean: float | None = None,
    track_std: float | None = None,
) -> str:
    """Render frame outlier with severity indicator.

    Only shown when frame is an outlier (below threshold). Shows how many
    standard deviations from mean.

    Args:
        similarity: Frame similarity to track centroid (0-1)
        track_mean: Mean similarity of frames in track
        track_std: Standard deviation of frame similarities

    Returns:
        HTML badge showing outlier severity, or empty if not an outlier
    """
    if similarity is None:
        return ""

    # Check if this is actually an outlier
    outlier_threshold = SIMILARITY_COLORS[SimilarityType.FRAME].good_threshold
    if similarity >= outlier_threshold:
        return ""  # Not an outlier, don't show

    pct = int(round(similarity * 100))
    color = get_badge_color(similarity, SimilarityType.FRAME)
    text_color = "#333333" if color in _LIGHT_COLORS else "#FFFFFF"

    # Calculate severity if stats available
    severity_text = ""
    if track_mean is not None and track_std is not None and track_std > 0:
        z_score = abs(track_mean - similarity) / track_std
        if z_score >= 3:
            severity_text = " ⚠️⚠️⚠️"  # Severe
        elif z_score >= 2:
            severity_text = " ⚠️⚠️"  # Moderate
        else:
            severity_text = " ⚠️"  # Mild

    tooltip = f"Frame differs from track - potential outlier"
    if track_mean is not None and track_std is not None and track_std > 0:
        z_score = abs(track_mean - similarity) / track_std
        tooltip += f" ({z_score:.1f}σ from mean)"

    return (
        f'<span class="sim-badge sim-badge-outlier" '
        f'style="background-color: {color}; color: {text_color}; '
        f'padding: 2px 6px; border-radius: 3px; font-size: 0.8em; font-weight: bold; cursor: help;" '
        f'title="{tooltip}">'
        f"OUTLIER: {pct}%{severity_text}</span>"
    )


def render_ambiguity_badge(
    margin: float | None,
    first_match_name: str | None = None,
    second_match_name: str | None = None,
) -> str:
    """Render ambiguity score (gap between 1st and 2nd best match).

    LOW margin = risky (red), HIGH margin = clear winner (green)

    Args:
        margin: Gap between 1st and 2nd best similarity (0-1)
        first_match_name: Name of best match for tooltip
        second_match_name: Name of 2nd best match for tooltip

    Returns:
        HTML badge showing ambiguity level
    """
    if margin is None:
        return ""

    pct = int(round(margin * 100))
    color = get_badge_color(margin, SimilarityType.AMBIGUITY)
    text_color = "#333333" if color in _LIGHT_COLORS else "#FFFFFF"

    # Determine status text
    if margin >= 0.15:
        status = "Clear"
    elif margin >= 0.08:
        status = "OK"
    else:
        status = "Risky"

    tooltip = f"Gap between 1st and 2nd match: {pct}%"
    if first_match_name and second_match_name:
        tooltip += f" ({first_match_name} vs {second_match_name})"

    return (
        f'<span class="sim-badge sim-badge-ambiguity" '
        f'style="background-color: {color}; color: {text_color}; '
        f'padding: 2px 6px; border-radius: 3px; font-size: 0.8em; font-weight: bold; cursor: help;" '
        f'title="{tooltip}">'
        f"AMB: {status} ({pct}%)</span>"
    )


def render_isolation_badge(distance: float | None) -> str:
    """Render cluster isolation score (distance to nearest cluster).

    LOW = merge candidate, HIGH = well isolated

    Args:
        distance: Distance to nearest cluster centroid (0-1)

    Returns:
        HTML badge showing isolation level
    """
    if distance is None:
        return ""

    pct = int(round(distance * 100))
    color = get_badge_color(distance, SimilarityType.ISOLATION)
    text_color = "#333333" if color in _LIGHT_COLORS else "#FFFFFF"

    # Determine status
    if distance >= 0.40:
        status = "Isolated"
    elif distance >= 0.25:
        status = "Moderate"
    else:
        status = "Close"

    tooltip = f"Distance to nearest cluster: {pct}%"
    if distance < 0.25:
        tooltip += " - Consider merging"

    return (
        f'<span class="sim-badge sim-badge-isolation" '
        f'style="background-color: {color}; color: {text_color}; '
        f'padding: 2px 6px; border-radius: 3px; font-size: 0.8em; font-weight: bold; cursor: help;" '
        f'title="{tooltip}">'
        f"ISO: {status}</span>"
    )


def render_confidence_trend_badge(trend: float | None) -> str:
    """Render confidence trend indicator.

    Shows if assignment confidence is improving, stable, or degrading.

    Args:
        trend: Trend value (+ve = improving, -ve = degrading)

    Returns:
        HTML badge showing trend direction
    """
    if trend is None:
        return ""

    color = get_badge_color(trend, SimilarityType.CONFIDENCE_TREND)
    text_color = "#FFFFFF"

    # Determine arrow and status
    if trend >= 0.02:
        arrow = "↑"
        status = "Improving"
    elif trend <= -0.02:
        arrow = "↓"
        status = "Degrading"
    else:
        arrow = "→"
        status = "Stable"

    pct = int(round(abs(trend) * 100))
    tooltip = f"Confidence trend: {status} ({'+' if trend > 0 else ''}{pct}%)"

    return (
        f'<span class="sim-badge sim-badge-trend" '
        f'style="background-color: {color}; color: {text_color}; '
        f'padding: 2px 6px; border-radius: 3px; font-size: 0.8em; font-weight: bold; cursor: help;" '
        f'title="{tooltip}">'
        f"{arrow} {status}</span>"
    )


def render_temporal_badge(consistency: float | None) -> str:
    """Render temporal consistency badge.

    Shows how consistent a person's appearance is across time in episode.

    Args:
        consistency: Temporal consistency score (0-1)

    Returns:
        HTML badge showing temporal consistency
    """
    if consistency is None:
        return ""

    pct = int(round(consistency * 100))
    color = get_badge_color(consistency, SimilarityType.TEMPORAL)
    text_color = "#333333" if color in _LIGHT_COLORS else "#FFFFFF"

    tooltip = f"Appearance consistency over time: {pct}%"
    if consistency < 0.60:
        tooltip += " - Appearance varies significantly (lighting, costume, angle changes?)"

    return (
        f'<span class="sim-badge sim-badge-temporal" '
        f'style="background-color: {color}; color: {text_color}; '
        f'padding: 2px 6px; border-radius: 3px; font-size: 0.8em; font-weight: bold; cursor: help;" '
        f'title="{tooltip}">'
        f"TIME: {pct}%</span>"
    )


def render_singleton_risk_badge(
    track_count: int,
    face_count: int,
    origin: str | None = None,
) -> str:
    """Render singleton risk badge based on track and frame counts.

    Risk levels:
    - HIGH (red): Single-track cluster with single frame - unreliable
    - MEDIUM (orange): Single-track cluster with multiple frames - limited
    - LOW (green): Multi-track cluster - reliable

    Args:
        track_count: Number of tracks in the cluster
        face_count: Total number of faces/frames across all tracks
        origin: Optional origin reason (outlier_removal, mixed_identity, etc.)

    Returns:
        HTML badge showing singleton risk level
    """
    # Determine risk level
    if track_count == 1 and face_count == 1:
        risk_level = "HIGH"
        color = "#F44336"  # Red
        tooltip = "Single-track, single-frame cluster - unreliable embedding"
    elif track_count == 1:
        risk_level = "MEDIUM"
        color = "#FF9800"  # Orange
        tooltip = f"Single-track cluster ({face_count} frames) - limited matching confidence"
    else:
        risk_level = "LOW"
        color = "#4CAF50"  # Green
        tooltip = f"Multi-track cluster ({track_count} tracks) - reliable"

    # Add origin info to tooltip
    if origin:
        origin_labels = {
            "outlier_removal": "Split due to outlier detection",
            "mixed_identity": "Split due to high embedding variance",
            "no_embeddings": "No accepted embeddings",
            "clustering_edge": "Natural clustering result",
            "manual_split": "User action",
        }
        origin_text = origin_labels.get(origin, origin)
        tooltip += f" | Origin: {origin_text}"

    return (
        f'<span class="sim-badge sim-badge-singleton-risk" '
        f'style="background-color: {color}; color: white; '
        f'padding: 2px 6px; border-radius: 3px; font-size: 0.75em; font-weight: bold; cursor: help;" '
        f'title="{tooltip}">'
        f"🎯 {risk_level}</span>"
    )


def render_singleton_fraction_badge(
    singleton_count: int,
    total_count: int,
    single_frame_count: int | None = None,
) -> str:
    """Render singleton fraction badge for episode health.

    Thresholds:
    - <25%: Healthy (green)
    - 25-40%: Warning (orange)
    - >40%: High (red)

    Args:
        singleton_count: Number of single-track clusters
        total_count: Total number of clusters
        single_frame_count: Optional count of single-frame tracks

    Returns:
        HTML badge showing singleton fraction health
    """
    if total_count == 0:
        return ""

    fraction = singleton_count / total_count
    pct = int(round(fraction * 100))

    # Determine health status
    if fraction <= 0.25:
        color = "#4CAF50"  # Green
        status = "Healthy"
    elif fraction <= 0.40:
        color = "#FF9800"  # Orange
        status = "Warning"
    else:
        color = "#F44336"  # Red
        status = "High"

    tooltip = f"Singleton fraction: {singleton_count}/{total_count} ({pct}%) - {status}"
    if single_frame_count is not None:
        sf_pct = int(round(single_frame_count / total_count * 100)) if total_count > 0 else 0
        tooltip += f" | Single-frame tracks: {single_frame_count} ({sf_pct}%)"

    return (
        f'<span class="sim-badge sim-badge-singleton-frac" '
        f'style="background-color: {color}; color: white; '
        f'padding: 2px 6px; border-radius: 3px; font-size: 0.8em; font-weight: bold; cursor: help;" '
        f'title="{tooltip}">'
        f"🎯 {pct}%</span>"
    )
