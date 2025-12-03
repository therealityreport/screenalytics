"""Unified MetricsStrip component for displaying metrics across all pages.

This component provides a consistent way to display similarity, quality, and
cohesion metrics as a compact horizontal strip with tooltips.

Usage:
    from metrics_strip import render_metrics_strip, MetricData, METRIC_HELP

    metrics = [
        MetricData("identity", 0.75, "Identity Similarity"),
        MetricData("cluster", 0.82, "Cluster Cohesion", min_val=0.65, max_val=0.91),
        MetricData("quality", 0.78, "Quality Score", breakdown={"det": 0.95, "sharp": 0.72, "area": 0.68}),
    ]
    render_metrics_strip(metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import streamlit as st

from similarity_badges import (
    SimilarityType,
    SIMILARITY_COLORS,
    SIMILARITY_LABELS,
    get_badge_color,
    render_similarity_badge,
    render_cluster_range_badge,
    render_quality_breakdown_badge,
    render_cast_rank_badge,
    render_track_with_dropout,
    render_ambiguity_badge,
    render_isolation_badge,
    render_confidence_trend_badge,
    render_temporal_badge,
)


# ============================================================================
# CENTRALIZED METRIC HELP CONFIGURATION
# ============================================================================
# This is the single source of truth for all metric descriptions, thresholds,
# and help text. Used by both tooltips and documentation pages.

METRIC_HELP: Dict[str, Dict[str, str]] = {
    "identity": {
        "title": "Identity Similarity",
        "emoji": "🆔",
        "description": "Measures how similar clusters are for an auto-generated person. Higher values indicate the clusters are likely the same person.",
        "thresholds": "≥75%: Strong match · 70–74%: Good match · <70%: Needs review",
        "interpretation": "High identity similarity means clusters assigned to this person are visually consistent. Low scores may indicate misassigned clusters.",
        "where_shown": "Person cards, Cast View, Faces Review",
    },
    "cast": {
        "title": "Cast Similarity",
        "emoji": "🎭",
        "description": "How similar this cluster is to the cast member's facebank (reference photos). Used for auto-assignment suggestions.",
        "thresholds": "≥68%: Auto-assign candidate · 50–67%: Manual review · <50%: Weak match",
        "interpretation": "The higher the cast similarity, the more confident we are this is the suggested cast member. Below 50% is unlikely to be correct.",
        "where_shown": "Cluster cards, Smart Suggestions, Cast assignment UI",
    },
    "track": {
        "title": "Track Similarity",
        "emoji": "🎬",
        "description": "How consistent the frames within a single track are. Measures frame-to-frame coherence within the same tracking sequence.",
        "thresholds": "≥85%: Strong consistency · 70–84%: Good · <70%: Weak (possible tracking errors)",
        "interpretation": "Low track similarity may indicate tracking errors (merged identities) or quality issues. Review tracks below 70%.",
        "where_shown": "Track cards, Cluster View, Frames View",
    },
    "cluster": {
        "title": "Cluster Cohesion",
        "emoji": "📦",
        "description": "How cohesive tracks within a cluster are. Measures whether all tracks in the cluster look like the same person.",
        "thresholds": "≥80%: Tight cluster · 60–79%: Moderate · <60%: Loose (review for splits)",
        "interpretation": "A tight cluster has tracks that all look similar. Loose clusters may contain different people and should be reviewed for splitting.",
        "where_shown": "Cluster cards, Person View, Faces Review",
    },
    "person_cohesion": {
        "title": "Person Cohesion",
        "emoji": "👤",
        "description": "How well a track fits with other tracks assigned to the same person. Detects outliers within a person's assignments.",
        "thresholds": "≥70%: Strong fit · 50–69%: Good fit · <50%: Poor fit (possible misassignment)",
        "interpretation": "Low person cohesion means this track may not belong with the other tracks. Consider reassigning or reviewing.",
        "where_shown": "Track View, Person View, Cast member detail",
    },
    "fit": {
        "title": "Person Fit",
        "emoji": "👤",
        "description": "Alias for Person Cohesion. How well a track fits with other tracks of the same person.",
        "thresholds": "≥70%: Strong fit · 50–69%: Good fit · <50%: Poor fit",
        "interpretation": "Low fit scores indicate potential misassignment.",
        "where_shown": "Track View",
    },
    "temporal": {
        "title": "Temporal Consistency",
        "emoji": "⏱️",
        "description": "How consistent a person's appearance is across different times in the episode. Detects costume changes, lighting variations, or misassignments.",
        "thresholds": "≥80%: Consistent appearance · 60–79%: Variable (possible costume change) · <60%: Significant changes (review)",
        "interpretation": "Low temporal consistency may indicate: costume/makeup changes (normal), lighting differences, or incorrectly merged identities.",
        "where_shown": "Cluster cards, Person View, Cast View",
    },
    "ambiguity": {
        "title": "Ambiguity Score",
        "emoji": "❓",
        "description": "The gap between the 1st and 2nd best match. A small gap means the assignment is risky—it could easily be either person.",
        "thresholds": "≥15%: Clear winner · 8–14%: OK but verify · <8%: Risky (could be either person)",
        "interpretation": "High ambiguity = clear assignment. LOW ambiguity (small gap) = risky, the 2nd best match is almost as good. Always review risky assignments.",
        "where_shown": "Cluster cards, Smart Suggestions, Assignment UI",
    },
    "isolation": {
        "title": "Cluster Isolation",
        "emoji": "🔒",
        "description": "Distance to the nearest cluster. Indicates how distinct this cluster is from others. Low isolation = potential merge candidate.",
        "thresholds": "≥40%: Well isolated · 25–39%: Moderate · <25%: Close (merge candidate)",
        "interpretation": "Clusters with low isolation look very similar to another cluster. Consider merging 'Close' clusters if they're the same person.",
        "where_shown": "Cluster cards, Person View",
    },
    "trend": {
        "title": "Confidence Trend",
        "emoji": "📈",
        "description": "Tracks whether assignment confidence is improving or degrading as more data is added. Early warning for problematic assignments.",
        "thresholds": "↑ Improving · → Stable · ↓ Degrading (early warning)",
        "interpretation": "A degrading trend means new evidence is lowering confidence. Investigate before the assignment becomes unreliable.",
        "where_shown": "Person View, Cast View",
    },
    "confidence_trend": {
        "title": "Confidence Trend",
        "emoji": "📈",
        "description": "Tracks whether assignment confidence is improving or degrading as more data is added.",
        "thresholds": "↑ Improving · → Stable · ↓ Degrading",
        "interpretation": "A degrading trend means new evidence is lowering confidence.",
        "where_shown": "Person View, Cast View",
    },
    "quality": {
        "title": "Quality Score",
        "emoji": "⭐",
        "description": "Composite score combining Detection confidence, Sharpness (crop standard deviation), and Face Area. Higher = better quality representative.",
        "thresholds": "≥85%: High quality (green) · 60–84%: Medium (amber) · <60%: Low quality (red)",
        "interpretation": "Quality affects how reliable face matching is. Low-quality faces may have poor detection, blurry crops, or small face areas.",
        "breakdown": {
            "det": "Detection confidence from the face detector (0-1)",
            "sharp": "Sharpness score from crop standard deviation (higher = sharper)",
            "area": "Face bounding box area in pixels (larger = more detail)",
        },
        "where_shown": "All views with face thumbnails",
    },
}

# Metric type to SimilarityType mapping
METRIC_TYPE_MAP = {
    "identity": SimilarityType.IDENTITY,
    "cast": SimilarityType.CAST,
    "track": SimilarityType.TRACK,
    "person_cohesion": SimilarityType.PERSON_COHESION,
    "fit": SimilarityType.PERSON_COHESION,  # Alias
    "cluster": SimilarityType.CLUSTER,
    "temporal": SimilarityType.TEMPORAL,
    "ambiguity": SimilarityType.AMBIGUITY,
    "isolation": SimilarityType.ISOLATION,
    "trend": SimilarityType.CONFIDENCE_TREND,
    "confidence_trend": SimilarityType.CONFIDENCE_TREND,
    "quality": None,  # Special handling
}


def get_metric_tooltip(metric_key: str) -> str:
    """Get formatted tooltip text for a metric.

    Args:
        metric_key: The metric type key (e.g., "temporal", "ambiguity")

    Returns:
        Formatted tooltip string combining description and thresholds
    """
    help_data = METRIC_HELP.get(metric_key.lower(), {})
    if not help_data:
        return ""

    title = help_data.get("title", metric_key.title())
    desc = help_data.get("description", "")
    thresholds = help_data.get("thresholds", "")

    parts = [title]
    if desc:
        parts.append(desc)
    if thresholds:
        parts.append(f"Thresholds: {thresholds}")

    return " | ".join(parts)


# Legacy format for backward compatibility
METRIC_DESCRIPTIONS = {
    key: get_metric_tooltip(key) for key in METRIC_HELP.keys()
}


@dataclass
class MetricData:
    """Data for a single metric in the strip."""

    metric_type: str  # One of the keys in METRIC_TYPE_MAP
    value: Optional[float] = None  # Primary value (0-1 scale)
    label: Optional[str] = None  # Override label
    # Optional additional data for enhanced rendering
    min_val: Optional[float] = None  # For cluster cohesion range
    max_val: Optional[float] = None  # For cluster cohesion range
    breakdown: Optional[Dict[str, float]] = None  # For quality: {det, sharp, area}
    rank: Optional[int] = None  # For cast: suggestion rank
    total: Optional[int] = None  # For cast: total suggestions
    excluded: Optional[int] = None  # For track: excluded frames
    trend_direction: Optional[str] = None  # For trend: "up", "stable", "down"
    second_match: Optional[str] = None  # For ambiguity: 2nd best match name
    first_match: Optional[str] = None  # For ambiguity: 1st best match name
    help_key: Optional[str] = None  # Override help lookup key (defaults to metric_type)


def _render_metric_badge(metric: MetricData) -> str:
    """Render a single metric as an HTML badge."""
    if metric.value is None and metric.metric_type != "trend":
        return ""

    metric_type = metric.metric_type.lower()

    # Special handling for quality
    if metric_type == "quality":
        breakdown = metric.breakdown or {}
        return render_quality_breakdown_badge(
            metric.value,
            detection=breakdown.get("det"),
            sharpness=breakdown.get("sharp"),
            area=breakdown.get("area"),
        )

    # Special handling for cluster with range
    if metric_type == "cluster" and (metric.min_val is not None or metric.max_val is not None):
        return render_cluster_range_badge(
            metric.value,
            min_sim=metric.min_val,
            max_sim=metric.max_val,
        )

    # Special handling for cast with rank
    if metric_type == "cast" and metric.rank is not None:
        return render_cast_rank_badge(
            metric.value,
            rank=metric.rank,
            total_suggestions=metric.total or 1,
            cast_name=metric.first_match,
        )

    # Special handling for track with dropout
    if metric_type == "track" and metric.excluded is not None:
        return render_track_with_dropout(
            metric.value,
            excluded_frames=metric.excluded,
        )

    # Special handling for ambiguity
    if metric_type == "ambiguity":
        return render_ambiguity_badge(
            metric.value,
            first_match_name=metric.first_match,
            second_match_name=metric.second_match,
        )

    # Special handling for isolation
    if metric_type == "isolation":
        return render_isolation_badge(metric.value)

    # Special handling for trend
    if metric_type in ("trend", "confidence_trend"):
        return render_confidence_trend_badge(metric.value)

    # Special handling for temporal
    if metric_type == "temporal":
        return render_temporal_badge(metric.value)

    # Default: use standard badge with label
    sim_type = METRIC_TYPE_MAP.get(metric_type)
    if sim_type:
        return render_similarity_badge(
            metric.value,
            sim_type,
            show_label=True,
            custom_label=metric.label,
        )

    return ""


def _build_help_tooltip_html(metric_key: str) -> str:
    """Build HTML for a help tooltip with a '?' icon.

    Args:
        metric_key: The metric type key

    Returns:
        HTML string for the help icon with tooltip
    """
    help_data = METRIC_HELP.get(metric_key.lower(), {})
    if not help_data:
        return ""

    # Build tooltip text (plain text for title attribute)
    title = help_data.get("title", metric_key.title())
    desc = help_data.get("description", "")
    thresholds = help_data.get("thresholds", "")
    interpretation = help_data.get("interpretation", "")

    tooltip_parts = [title]
    if desc:
        tooltip_parts.append(desc)
    if thresholds:
        tooltip_parts.append(f"Thresholds: {thresholds}")
    if interpretation:
        tooltip_parts.append(f"Tip: {interpretation}")

    tooltip_text = " | ".join(tooltip_parts)

    # Return help icon HTML
    return (
        f'<span class="metric-help-icon" title="{tooltip_text}" '
        f'style="cursor: help; font-size: 0.7em; opacity: 0.6; margin-left: 2px; '
        f'vertical-align: super;">❓</span>'
    )


def render_metrics_strip(
    metrics: List[MetricData],
    *,
    compact: bool = False,
    show_na: bool = False,
    container_class: str = "metrics-strip",
    show_help: bool = True,
) -> None:
    """Render a horizontal strip of metrics badges.

    Args:
        metrics: List of MetricData objects to display
        compact: Use smaller padding/font
        show_na: Show "N/A" for missing metrics instead of hiding them
        container_class: CSS class for the container
        show_help: Show help icon with tooltip on each metric
    """
    # Filter out None values unless show_na is True
    badges_html = []
    for metric in metrics:
        badge = _render_metric_badge(metric)
        if badge:
            # Use help_key if provided, otherwise fall back to metric_type
            help_key = metric.help_key or metric.metric_type.lower()

            # Wrap badge with tooltip (hover on entire badge shows help)
            tooltip = METRIC_DESCRIPTIONS.get(help_key, "")
            if tooltip:
                badge = f'<span title="{tooltip}" style="cursor: help;">{badge}</span>'

            # Add help icon if requested
            if show_help:
                help_icon = _build_help_tooltip_html(help_key)
                if help_icon:
                    badge = f'{badge}{help_icon}'

            badges_html.append(badge)
        elif show_na and metric.metric_type:
            label = SIMILARITY_LABELS.get(
                METRIC_TYPE_MAP.get(metric.metric_type.lower()),
                metric.metric_type.upper()[:4]
            )
            na_badge = (
                f'<span class="sim-badge sim-badge-na" '
                f'style="background-color: #9E9E9E; color: #FFFFFF; '
                f'padding: 2px 6px; border-radius: 3px; font-size: 0.8em; '
                f'font-weight: bold; opacity: 0.5;" '
                f'title="Not computed for this item">'
                f'{label}: N/A</span>'
            )
            badges_html.append(na_badge)

    if not badges_html:
        return

    padding = "4px 8px" if compact else "6px 12px"
    font_size = "0.85em" if compact else "0.9em"

    st.markdown(
        f"""
        <div class="{container_class}" style="
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            padding: {padding};
            background: rgba(0,0,0,0.1);
            border-radius: 6px;
            font-size: {font_size};
            align-items: center;
        ">
            {" ".join(badges_html)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics_strip_inline(
    metrics: List[MetricData],
    *,
    compact: bool = True,
    show_help: bool = True,
) -> str:
    """Return HTML for a metrics strip (for embedding in other HTML).

    Args:
        metrics: List of MetricData objects
        compact: Use smaller styling
        show_help: Show help icon with tooltip on each metric

    Returns:
        HTML string for the metrics strip
    """
    badges_html = []
    for metric in metrics:
        badge = _render_metric_badge(metric)
        if badge:
            help_key = metric.help_key or metric.metric_type.lower()
            tooltip = METRIC_DESCRIPTIONS.get(help_key, "")
            if tooltip:
                badge = f'<span title="{tooltip}" style="cursor: help;">{badge}</span>'
            if show_help:
                help_icon = _build_help_tooltip_html(help_key)
                if help_icon:
                    badge = f'{badge}{help_icon}'
            badges_html.append(badge)

    if not badges_html:
        return ""

    gap = "4px" if compact else "6px"
    return f'<span style="display: inline-flex; gap: {gap}; align-items: center;">{" ".join(badges_html)}</span>'


def render_metrics_documentation(
    metric_keys: Optional[List[str]] = None,
    show_thresholds_table: bool = True,
) -> None:
    """Render complete metrics documentation using Streamlit.

    This function renders documentation for the specified metrics using
    the centralized METRIC_HELP config. Useful for docs pages and help
    expanders.

    Args:
        metric_keys: List of metric keys to document. If None, shows all.
        show_thresholds_table: Whether to show a summary table at the end.
    """
    keys_to_show = metric_keys or [
        k for k in METRIC_HELP.keys()
        if k not in ("fit", "confidence_trend")  # Skip aliases
    ]

    for key in keys_to_show:
        help_data = METRIC_HELP.get(key, {})
        if not help_data:
            continue

        title = help_data.get("title", key.title())
        emoji = help_data.get("emoji", "📊")
        description = help_data.get("description", "")
        thresholds = help_data.get("thresholds", "")
        interpretation = help_data.get("interpretation", "")
        where_shown = help_data.get("where_shown", "")
        breakdown = help_data.get("breakdown", {})

        with st.container(border=True):
            st.markdown(f"### {emoji} {title}")
            st.markdown(f"**Description:** {description}")

            if thresholds:
                st.markdown(f"**Thresholds:** `{thresholds}`")

            if interpretation:
                st.info(f"💡 **Tip:** {interpretation}")

            if breakdown and isinstance(breakdown, dict):
                st.markdown("**Components:**")
                for comp_key, comp_desc in breakdown.items():
                    st.markdown(f"- **{comp_key.upper()}**: {comp_desc}")

            if where_shown:
                st.caption(f"📍 Shown in: {where_shown}")

    # Summary table
    if show_thresholds_table:
        st.markdown("---")
        st.markdown("### Quick Reference")
        threshold_data = []
        for key in keys_to_show:
            data = METRIC_HELP.get(key, {})
            if data:
                threshold_data.append({
                    "Metric": data.get("title", key.title()),
                    "Thresholds": data.get("thresholds", "N/A"),
                })
        if threshold_data:
            st.table(threshold_data)


def get_all_metric_keys() -> List[str]:
    """Get list of all metric keys (excluding aliases).

    Returns:
        List of metric type keys
    """
    return [k for k in METRIC_HELP.keys() if k not in ("fit", "confidence_trend")]


# ============================================================================
# PAGE-SPECIFIC METRIC BUILDERS
# ============================================================================


def build_cluster_metrics(
    cluster: Dict[str, Any],
    suggestion: Optional[Dict[str, Any]] = None,
) -> List[MetricData]:
    """Build metrics list for a cluster card.

    Args:
        cluster: Cluster data dict
        suggestion: Optional cast suggestion for this cluster

    Returns:
        List of MetricData for display
    """
    metrics = []

    # Identity/Cast similarity
    if suggestion:
        metrics.append(MetricData(
            metric_type="cast",
            value=suggestion.get("similarity") or suggestion.get("score"),
            rank=suggestion.get("rank", 1),
            total=suggestion.get("total_suggestions", 1),
            first_match=suggestion.get("name") or suggestion.get("cast_name"),
        ))
    elif cluster.get("identity_similarity") is not None:
        metrics.append(MetricData(
            metric_type="identity",
            value=cluster.get("identity_similarity"),
        ))

    # Cluster cohesion with range
    cohesion = cluster.get("cohesion")
    if cohesion is not None:
        metrics.append(MetricData(
            metric_type="cluster",
            value=cohesion,
            min_val=cluster.get("min_similarity"),
            max_val=cluster.get("max_similarity"),
        ))

    # Temporal consistency
    if cluster.get("temporal_consistency") is not None:
        metrics.append(MetricData(
            metric_type="temporal",
            value=cluster.get("temporal_consistency"),
        ))

    # Ambiguity
    if cluster.get("ambiguity") is not None:
        metrics.append(MetricData(
            metric_type="ambiguity",
            value=cluster.get("ambiguity"),
            first_match=cluster.get("first_match_name"),
            second_match=cluster.get("second_match_name"),
        ))

    # Isolation
    if cluster.get("isolation") is not None:
        metrics.append(MetricData(
            metric_type="isolation",
            value=cluster.get("isolation"),
        ))

    # Confidence trend
    if cluster.get("confidence_trend") is not None:
        metrics.append(MetricData(
            metric_type="trend",
            value=cluster.get("confidence_trend"),
        ))

    # Quality score (aggregate)
    quality = cluster.get("avg_quality") or cluster.get("quality")
    if quality is not None:
        metrics.append(MetricData(
            metric_type="quality",
            value=quality,
            breakdown=cluster.get("quality_breakdown"),
        ))

    return metrics


def build_track_metrics(
    track: Dict[str, Any],
    *,
    include_cluster: bool = False,
    cluster_data: Optional[Dict[str, Any]] = None,
) -> List[MetricData]:
    """Build metrics list for a track.

    Args:
        track: Track data dict
        include_cluster: Include cluster-level metrics as secondary
        cluster_data: Cluster data for cluster-level metrics

    Returns:
        List of MetricData for display
    """
    metrics = []

    # Track similarity
    track_sim = track.get("similarity") or track.get("track_similarity")
    if track_sim is not None:
        metrics.append(MetricData(
            metric_type="track",
            value=track_sim,
            excluded=track.get("excluded_frames"),
        ))

    # Quality score
    quality = track.get("quality") or track.get("quality_score")
    if isinstance(quality, dict):
        metrics.append(MetricData(
            metric_type="quality",
            value=quality.get("score"),
            breakdown={
                "det": quality.get("det"),
                "sharp": quality.get("std"),
                "area": quality.get("box_area"),
            },
        ))
    elif quality is not None:
        metrics.append(MetricData(
            metric_type="quality",
            value=quality,
        ))

    # Person cohesion
    person_cohesion = track.get("person_cohesion") or track.get("cast_track_score")
    if person_cohesion is not None:
        metrics.append(MetricData(
            metric_type="person_cohesion",
            value=person_cohesion,
        ))

    # Temporal consistency
    if track.get("temporal_consistency") is not None:
        metrics.append(MetricData(
            metric_type="temporal",
            value=track.get("temporal_consistency"),
        ))

    # Ambiguity
    if track.get("ambiguity") is not None:
        metrics.append(MetricData(
            metric_type="ambiguity",
            value=track.get("ambiguity"),
        ))

    # Cluster-level metrics if requested
    if include_cluster and cluster_data:
        cohesion = cluster_data.get("cohesion")
        if cohesion is not None:
            metrics.append(MetricData(
                metric_type="cluster",
                value=cohesion,
                label="CLU",
            ))

        isolation = cluster_data.get("isolation")
        if isolation is not None:
            metrics.append(MetricData(
                metric_type="isolation",
                value=isolation,
            ))

    return metrics


def build_frame_metrics(
    frame: Dict[str, Any],
    track_stats: Optional[Dict[str, Any]] = None,
) -> List[MetricData]:
    """Build metrics list for a frame.

    Args:
        frame: Frame data dict
        track_stats: Optional track statistics for outlier detection

    Returns:
        List of MetricData for display
    """
    metrics = []

    # Quality score with breakdown
    quality = frame.get("quality") or frame.get("quality_score")
    det = frame.get("det_score") or frame.get("det")
    sharp = frame.get("crop_std") or frame.get("sharpness")
    area = frame.get("box_area") or frame.get("area")

    if quality is not None or det is not None:
        metrics.append(MetricData(
            metric_type="quality",
            value=quality,
            breakdown={
                "det": det,
                "sharp": sharp,
                "area": area,
            },
        ))

    # Frame similarity to track (only show if it's an outlier)
    frame_sim = frame.get("similarity")
    if frame_sim is not None and frame_sim < 0.50:  # Outlier threshold
        metrics.append(MetricData(
            metric_type="track",  # Show as track similarity (deviation)
            value=frame_sim,
            label="FRM",
        ))

    return metrics


def build_cast_member_metrics(
    cast_member: Dict[str, Any],
    aggregated_stats: Optional[Dict[str, Any]] = None,
) -> List[MetricData]:
    """Build metrics list for a cast member summary.

    Args:
        cast_member: Cast member data
        aggregated_stats: Aggregated statistics across all their tracks/clusters

    Returns:
        List of MetricData for display
    """
    metrics = []
    stats = aggregated_stats or cast_member

    # Cast similarity (median/aggregate)
    cast_sim = stats.get("median_similarity") or stats.get("cast_similarity")
    if cast_sim is not None:
        metrics.append(MetricData(
            metric_type="cast",
            value=cast_sim,
        ))

    # Person cohesion (how well their tracks agree)
    cohesion = stats.get("person_cohesion") or stats.get("avg_cohesion")
    if cohesion is not None:
        metrics.append(MetricData(
            metric_type="person_cohesion",
            value=cohesion,
        ))

    # Temporal consistency
    if stats.get("temporal_consistency") is not None:
        metrics.append(MetricData(
            metric_type="temporal",
            value=stats.get("temporal_consistency"),
        ))

    # Ambiguity (worst case)
    if stats.get("min_ambiguity") is not None:
        metrics.append(MetricData(
            metric_type="ambiguity",
            value=stats.get("min_ambiguity"),
        ))

    # Cluster isolation
    if stats.get("isolation") is not None:
        metrics.append(MetricData(
            metric_type="isolation",
            value=stats.get("isolation"),
        ))

    # Confidence trend
    if stats.get("confidence_trend") is not None:
        metrics.append(MetricData(
            metric_type="trend",
            value=stats.get("confidence_trend"),
        ))

    return metrics


def build_person_metrics(
    person: Dict[str, Any],
    episode_clusters: Optional[List[Dict[str, Any]]] = None,
) -> List[MetricData]:
    """Build metrics list for an auto-generated person.

    Args:
        person: Person data dict
        episode_clusters: List of clusters for this person in current episode

    Returns:
        List of MetricData for display
    """
    metrics = []

    # Identity similarity
    identity_sim = person.get("identity_similarity") or person.get("avg_similarity")
    if identity_sim is not None:
        metrics.append(MetricData(
            metric_type="identity",
            value=identity_sim,
        ))

    # Cluster cohesion (aggregate)
    cohesion = person.get("avg_cohesion") or person.get("cohesion")
    if cohesion is not None:
        metrics.append(MetricData(
            metric_type="cluster",
            value=cohesion,
            min_val=person.get("min_cohesion"),
            max_val=person.get("max_cohesion"),
        ))

    # Temporal consistency
    if person.get("temporal_consistency") is not None:
        metrics.append(MetricData(
            metric_type="temporal",
            value=person.get("temporal_consistency"),
        ))

    # Ambiguity
    if person.get("ambiguity") is not None:
        metrics.append(MetricData(
            metric_type="ambiguity",
            value=person.get("ambiguity"),
        ))

    # Isolation
    if person.get("isolation") is not None:
        metrics.append(MetricData(
            metric_type="isolation",
            value=person.get("isolation"),
        ))

    # Confidence trend
    if person.get("confidence_trend") is not None:
        metrics.append(MetricData(
            metric_type="trend",
            value=person.get("confidence_trend"),
        ))

    # Quality (aggregate)
    quality = person.get("avg_quality") or person.get("quality")
    if quality is not None:
        metrics.append(MetricData(
            metric_type="quality",
            value=quality,
        ))

    return metrics
