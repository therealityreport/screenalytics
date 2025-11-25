"""Centralized similarity badge rendering for SCREENALYTICS workspace UI.

This module provides consistent color-coded badges for all similarity metrics:
- IDENTITY (Blue): How similar clusters are for AUTO-GENERATED PEOPLE
- CAST (Purple): How similar clusters are for CAST MEMBERS (facebank)
- TRACK (Orange): How similar FRAMES within a TRACK are to each other
- FRAME (Light Orange): How similar a specific frame is to rest of frames in track
- CAST_TRACK (Teal): How similar a track is to other tracks assigned to same cast/identity
- CLUSTER (Green): How cohesive/similar all tracks in a cluster are
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class SimilarityType(str, Enum):
    """Enumeration of all similarity metric types."""

    IDENTITY = "identity"  # Blue - auto-generated people's clusters
    CAST = "cast"  # Purple - cast member clusters
    TRACK = "track"  # Orange - track internal consistency (frames within track)
    FRAME = "frame"  # Light Orange - individual frame vs track centroid
    CAST_TRACK = "cast_track"  # Teal - track vs other tracks in same person
    CLUSTER = "cluster"  # Green - cluster cohesion (tracks within cluster)


class ColorScheme(NamedTuple):
    """Color configuration for a similarity type."""

    strong: str  # High confidence color
    good: str  # Medium confidence color
    weak: str  # Low confidence color
    strong_threshold: float  # Value >= this gets strong color
    good_threshold: float  # Value >= this gets good color (else weak)


# Master color definitions matching plan
SIMILARITY_COLORS: dict[SimilarityType, ColorScheme] = {
    SimilarityType.IDENTITY: ColorScheme(
        strong="#2196F3",  # Blue
        good="#64B5F6",  # Light Blue
        weak="#BBDEFB",  # Very Light Blue
        strong_threshold=0.75,
        good_threshold=0.60,
    ),
    SimilarityType.CAST: ColorScheme(
        strong="#9C27B0",  # Purple
        good="#CE93D8",  # Light Purple
        weak="#F3E5F5",  # Very Light Purple
        strong_threshold=0.68,
        good_threshold=0.50,
    ),
    SimilarityType.TRACK: ColorScheme(
        strong="#FF9800",  # Orange
        good="#FFB74D",  # Light Orange
        weak="#FFE0B2",  # Very Light Orange
        strong_threshold=0.85,
        good_threshold=0.70,
    ),
    SimilarityType.FRAME: ColorScheme(
        strong="#FFA726",  # Light Orange (distinct from Track)
        good="#FFCC80",  # Lighter Orange
        weak="#FFE0B2",  # Very Light Orange
        strong_threshold=0.80,
        good_threshold=0.65,
    ),
    SimilarityType.CAST_TRACK: ColorScheme(
        strong="#00ACC1",  # Teal
        good="#4DD0E1",  # Light Teal
        weak="#B2EBF2",  # Very Light Teal
        strong_threshold=0.70,
        good_threshold=0.55,
    ),
    SimilarityType.CLUSTER: ColorScheme(
        strong="#8BC34A",  # Green
        good="#C5E1A5",  # Light Green
        weak="#E0E0E0",  # Gray
        strong_threshold=0.80,
        good_threshold=0.60,
    ),
}

# Badge label abbreviations for compact display
SIMILARITY_LABELS: dict[SimilarityType, str] = {
    SimilarityType.IDENTITY: "ID",
    SimilarityType.CAST: "CAST",
    SimilarityType.TRACK: "TRK",
    SimilarityType.FRAME: "FRM",
    SimilarityType.CAST_TRACK: "MATCH",
    SimilarityType.CLUSTER: "CLU",
}

# Colors that need dark text instead of white for readability
_LIGHT_COLORS = {"#E0E0E0", "#F3E5F5", "#BBDEFB", "#FFE0B2", "#C5E1A5", "#B2EBF2", "#FFCC80"}


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


def get_similarity_key_data() -> list[dict]:
    """Return data for rendering the similarity scores legend/key.

    Returns:
        List of dicts with keys: type, color, label, description
    """
    return [
        {
            "type": SimilarityType.IDENTITY,
            "color": SIMILARITY_COLORS[SimilarityType.IDENTITY].strong,
            "label": "Identity Similarity",
            "description": "How similar clusters are for AUTO-GENERATED PEOPLE",
            "thresholds": f">={int(SIMILARITY_COLORS[SimilarityType.IDENTITY].strong_threshold*100)}%: Strong, "
            f">={int(SIMILARITY_COLORS[SimilarityType.IDENTITY].good_threshold*100)}%: Good",
        },
        {
            "type": SimilarityType.CAST,
            "color": SIMILARITY_COLORS[SimilarityType.CAST].strong,
            "label": "Cast Similarity",
            "description": "How similar clusters are for CAST MEMBERS (facebank)",
            "thresholds": f">={int(SIMILARITY_COLORS[SimilarityType.CAST].strong_threshold*100)}%: Strong, "
            f">={int(SIMILARITY_COLORS[SimilarityType.CAST].good_threshold*100)}%: Good",
        },
        {
            "type": SimilarityType.TRACK,
            "color": SIMILARITY_COLORS[SimilarityType.TRACK].strong,
            "label": "Track Similarity",
            "description": "How similar FRAMES within a TRACK are to each other",
            "thresholds": f">={int(SIMILARITY_COLORS[SimilarityType.TRACK].strong_threshold*100)}%: Strong, "
            f">={int(SIMILARITY_COLORS[SimilarityType.TRACK].good_threshold*100)}%: Good",
        },
        {
            "type": SimilarityType.FRAME,
            "color": SIMILARITY_COLORS[SimilarityType.FRAME].strong,
            "label": "Frame Similarity",
            "description": "How similar a specific frame is to rest of frames in track",
            "thresholds": f">={int(SIMILARITY_COLORS[SimilarityType.FRAME].strong_threshold*100)}%: Strong, "
            f">={int(SIMILARITY_COLORS[SimilarityType.FRAME].good_threshold*100)}%: Good",
        },
        {
            "type": SimilarityType.CAST_TRACK,
            "color": SIMILARITY_COLORS[SimilarityType.CAST_TRACK].strong,
            "label": "Cast Track Score",
            "description": "How similar a track is to other tracks assigned to same cast/identity",
            "thresholds": f">={int(SIMILARITY_COLORS[SimilarityType.CAST_TRACK].strong_threshold*100)}%: Strong, "
            f">={int(SIMILARITY_COLORS[SimilarityType.CAST_TRACK].good_threshold*100)}%: Good",
        },
        {
            "type": SimilarityType.CLUSTER,
            "color": SIMILARITY_COLORS[SimilarityType.CLUSTER].strong,
            "label": "Cluster Similarity",
            "description": "How cohesive/similar all tracks in a cluster are",
            "thresholds": f">={int(SIMILARITY_COLORS[SimilarityType.CLUSTER].strong_threshold*100)}%: Strong, "
            f">={int(SIMILARITY_COLORS[SimilarityType.CLUSTER].good_threshold*100)}%: Good",
        },
    ]


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
    "Cast Track Score (Low to High)",  # Outliers first
    "Cast Track Score (High to Low)",
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
    elif sort_option == "Cast Track Score (Low to High)":
        # Sort by cast_track_score - how similar track is to other tracks of the same person
        tracks.sort(
            key=lambda t: (t.get("cast_track_score") if t.get("cast_track_score") is not None else 999.0)
        )
    elif sort_option == "Cast Track Score (High to Low)":
        tracks.sort(
            key=lambda t: (t.get("cast_track_score") if t.get("cast_track_score") is not None else -999.0),
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
