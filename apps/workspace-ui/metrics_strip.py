"""Unified MetricsStrip component for displaying metrics across all pages.

This component provides a consistent way to display similarity, quality, and
cohesion metrics as a compact horizontal strip with tooltips.

Usage:
    from metrics_strip import render_metrics_strip, MetricData

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

# Full descriptions for tooltips
METRIC_DESCRIPTIONS = {
    "identity": "How similar clusters are for this auto-generated person. ≥75%: Strong, ≥70%: Good, <70%: Review",
    "cast": "How similar this cluster is to the cast member's facebank. ≥68%: Auto-assign, ≥50%: Review, <50%: Weak",
    "track": "How consistent frames within this track are. ≥85%: Strong, ≥70%: Good, <70%: Weak",
    "person_cohesion": "How well this track fits with other tracks of the same person. ≥70%: Strong, ≥50%: Good, <50%: Poor",
    "fit": "How well this track fits with other tracks of the same person. ≥70%: Strong, ≥50%: Good, <50%: Poor",
    "cluster": "How cohesive tracks in this cluster are. ≥80%: Tight, ≥60%: Moderate, <60%: Loose",
    "temporal": "How consistent this person's appearance is across time. ≥80%: Consistent, ≥60%: Variable, <60%: Changes",
    "ambiguity": "Gap between 1st and 2nd best match. ≥15%: Clear, ≥8%: OK, <8%: Risky",
    "isolation": "Distance to nearest cluster. ≥40%: Isolated, ≥25%: Moderate, <25%: Merge candidate",
    "trend": "Is confidence improving or degrading? ↑: Improving, →: Stable, ↓: Degrading",
    "confidence_trend": "Is confidence improving or degrading? ↑: Improving, →: Stable, ↓: Degrading",
    "quality": "Detection + Sharpness + Area score. ≥85%: High (green), ≥60%: Medium (amber), <60%: Low (red)",
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


def render_metrics_strip(
    metrics: List[MetricData],
    *,
    compact: bool = False,
    show_na: bool = False,
    container_class: str = "metrics-strip",
) -> None:
    """Render a horizontal strip of metrics badges.

    Args:
        metrics: List of MetricData objects to display
        compact: Use smaller padding/font
        show_na: Show "N/A" for missing metrics instead of hiding them
        container_class: CSS class for the container
    """
    # Filter out None values unless show_na is True
    badges_html = []
    for metric in metrics:
        badge = _render_metric_badge(metric)
        if badge:
            # Wrap with tooltip
            tooltip = METRIC_DESCRIPTIONS.get(metric.metric_type.lower(), "")
            if tooltip:
                badge = f'<span title="{tooltip}" style="cursor: help;">{badge}</span>'
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
) -> str:
    """Return HTML for a metrics strip (for embedding in other HTML).

    Args:
        metrics: List of MetricData objects
        compact: Use smaller styling

    Returns:
        HTML string for the metrics strip
    """
    badges_html = []
    for metric in metrics:
        badge = _render_metric_badge(metric)
        if badge:
            tooltip = METRIC_DESCRIPTIONS.get(metric.metric_type.lower(), "")
            if tooltip:
                badge = f'<span title="{tooltip}" style="cursor: help;">{badge}</span>'
            badges_html.append(badge)

    if not badges_html:
        return ""

    gap = "4px" if compact else "6px"
    return f'<span style="display: inline-flex; gap: {gap}; align-items: center;">{" ".join(badges_html)}</span>'


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
