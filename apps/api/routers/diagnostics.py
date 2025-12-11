"""Pipeline diagnostics API endpoints.

Provides AI-powered analysis of pipeline failures, explaining why
faces weren't detected, tracked, embedded, or clustered.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from apps.api.services.diagnostics import (
    collect_track_diagnostics,
    collect_unclustered_tracks,
    save_diagnostic_report,
    load_diagnostic_report,
    list_diagnostic_reports,
    get_pipeline_thresholds,
    DiagnosticReport,
)
from apps.api.services.openai_diagnostics import (
    analyze_pipeline_issues,
    is_openai_available,
)

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


# =============================================================================
# Pydantic Models
# =============================================================================


class DiagnoseTrackRequest(BaseModel):
    """Request for diagnosing a specific track."""
    use_ai: bool = Field(
        True,
        description="Use OpenAI for AI-powered analysis (falls back to rules if unavailable)",
    )
    force_refresh: bool = Field(
        False,
        description="Force re-analysis even if cached report exists",
    )


class DiagnoseUnclusteredRequest(BaseModel):
    """Request for batch diagnosing unclustered tracks."""
    use_ai: bool = Field(
        True,
        description="Use OpenAI for AI-powered analysis",
    )
    max_tracks: int = Field(
        50,
        ge=1,
        le=500,
        description="Maximum number of tracks to analyze",
    )


class TrackDiagnosticResponse(BaseModel):
    """Response for single track diagnostic."""
    track_id: int
    ep_id: str
    stage_reached: str
    stage_failed: Optional[str]
    raw_data: Dict[str, Any]
    config_used: Dict[str, Any]
    ai_analysis: Optional[Dict[str, Any]]
    cached: bool = False


class BatchDiagnosticResponse(BaseModel):
    """Response for batch diagnostics."""
    ep_id: str
    total_unclustered: int
    analyzed_count: int
    diagnostics: List[Dict[str, Any]]
    ai_available: bool


class DiagnosticSummary(BaseModel):
    """Summary of diagnostic reports for an episode."""
    track_id: int
    stage_reached: str
    stage_failed: Optional[str]
    has_ai_analysis: bool
    generated_at: str


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/episodes/{ep_id}/diagnose_track/{track_id}")
async def diagnose_track(
    ep_id: str,
    track_id: int,
    request: DiagnoseTrackRequest = DiagnoseTrackRequest(),
) -> TrackDiagnosticResponse:
    """Generate AI-powered diagnostic for a specific track.

    Collects data from all pipeline stages and uses GPT-4o to explain
    why the track wasn't processed successfully.

    Args:
        ep_id: Episode ID
        track_id: Track ID to diagnose
        request: Options for diagnosis

    Returns:
        Diagnostic report with AI analysis
    """
    # Check for cached report
    if not request.force_refresh:
        existing = load_diagnostic_report(ep_id, track_id)
        if existing and existing.ai_analysis:
            LOGGER.info(f"Returning cached diagnostic for track {track_id}")
            return TrackDiagnosticResponse(
                track_id=track_id,
                ep_id=ep_id,
                stage_reached=existing.diagnostics.stage_reached,
                stage_failed=existing.diagnostics.stage_failed,
                raw_data=existing.to_dict().get("raw_data", {}),
                config_used=existing.diagnostics.config_thresholds,
                ai_analysis=existing.ai_analysis,
                cached=True,
            )

    # Collect diagnostic data
    diagnostics = collect_track_diagnostics(ep_id, track_id)

    # Prepare data for AI analysis
    diag_dict = {
        "track_id": diagnostics.track_id,
        "ep_id": diagnostics.ep_id,
        "stage_reached": diagnostics.stage_reached,
        "stage_failed": diagnostics.stage_failed,
        "raw_data": {
            "frame_count": diagnostics.frame_count,
            "faces_count": diagnostics.faces_count,
            "faces_skipped": diagnostics.faces_skipped,
            "faces_with_embeddings": diagnostics.faces_with_embeddings,
            "skip_reasons": diagnostics.skip_reasons,
            "max_blur_score": diagnostics.max_blur_score,
            "avg_blur_score": diagnostics.avg_blur_score,
            "embedding_generated": diagnostics.embedding_generated,
            "cluster_id": diagnostics.cluster_id,
        },
    }

    # Get AI analysis
    ai_analysis = None
    if request.use_ai:
        ai_analysis = analyze_pipeline_issues(
            diag_dict,
            diagnostics.config_thresholds,
        )

    # Create and save report
    report = DiagnosticReport(
        track_id=track_id,
        ep_id=ep_id,
        generated_at=diagnostics.generated_at,
        diagnostics=diagnostics,
        ai_analysis=ai_analysis,
    )
    save_diagnostic_report(report)

    return TrackDiagnosticResponse(
        track_id=track_id,
        ep_id=ep_id,
        stage_reached=diagnostics.stage_reached,
        stage_failed=diagnostics.stage_failed,
        raw_data=diag_dict["raw_data"],
        config_used=diagnostics.config_thresholds,
        ai_analysis=ai_analysis,
        cached=False,
    )


@router.post("/episodes/{ep_id}/diagnose_unclustered")
async def diagnose_unclustered(
    ep_id: str,
    request: DiagnoseUnclusteredRequest = DiagnoseUnclusteredRequest(),
) -> BatchDiagnosticResponse:
    """Analyze all unclustered tracks with AI explanation.

    Finds tracks that are not in any cluster and generates
    diagnostic explanations for why they weren't clustered.

    Args:
        ep_id: Episode ID
        request: Batch diagnosis options

    Returns:
        Batch diagnostic report
    """
    unclustered_ids = collect_unclustered_tracks(ep_id)
    total = len(unclustered_ids)

    # Limit to max_tracks
    track_ids_to_analyze = unclustered_ids[:request.max_tracks]

    diagnostics_list = []
    for track_id in track_ids_to_analyze:
        try:
            # Collect diagnostic data
            diagnostics = collect_track_diagnostics(ep_id, track_id)

            diag_dict = {
                "track_id": diagnostics.track_id,
                "stage_reached": diagnostics.stage_reached,
                "stage_failed": diagnostics.stage_failed,
                "raw_data": {
                    "faces_count": diagnostics.faces_count,
                    "faces_skipped": diagnostics.faces_skipped,
                    "skip_reasons": diagnostics.skip_reasons,
                    "max_blur_score": diagnostics.max_blur_score,
                    "embedding_generated": diagnostics.embedding_generated,
                },
            }

            # Get AI analysis if requested
            ai_analysis = None
            if request.use_ai:
                ai_analysis = analyze_pipeline_issues(
                    diag_dict,
                    diagnostics.config_thresholds,
                )
                diag_dict["ai_analysis"] = ai_analysis

                # Save individual report
                report = DiagnosticReport(
                    track_id=track_id,
                    ep_id=ep_id,
                    generated_at=diagnostics.generated_at,
                    diagnostics=diagnostics,
                    ai_analysis=ai_analysis,
                )
                save_diagnostic_report(report)

            diagnostics_list.append(diag_dict)

        except Exception as e:
            LOGGER.warning(f"Failed to diagnose track {track_id}: {e}")
            diagnostics_list.append({
                "track_id": track_id,
                "error": str(e),
            })

    return BatchDiagnosticResponse(
        ep_id=ep_id,
        total_unclustered=total,
        analyzed_count=len(diagnostics_list),
        diagnostics=diagnostics_list,
        ai_available=is_openai_available(),
    )


@router.get("/episodes/{ep_id}/diagnostics")
def get_diagnostics(
    ep_id: str,
    track_id: Optional[int] = Query(None, description="Filter by track ID"),
) -> Dict[str, Any]:
    """Get saved diagnostic reports for episode.

    Args:
        ep_id: Episode ID
        track_id: Optional track ID to filter

    Returns:
        List of diagnostic reports
    """
    if track_id is not None:
        report = load_diagnostic_report(ep_id, track_id)
        if not report:
            raise HTTPException(
                status_code=404,
                detail=f"No diagnostic report found for track {track_id}",
            )
        return report.to_dict()

    reports = list_diagnostic_reports(ep_id)
    return {
        "ep_id": ep_id,
        "count": len(reports),
        "reports": reports,
    }


@router.get("/episodes/{ep_id}/thresholds")
def get_thresholds(ep_id: str) -> Dict[str, Any]:
    """Get current pipeline thresholds for reference.

    Args:
        ep_id: Episode ID (for future per-episode overrides)

    Returns:
        Pipeline thresholds from config files
    """
    return {
        "ep_id": ep_id,
        "thresholds": get_pipeline_thresholds(),
    }


@router.get("/status")
def diagnostics_status() -> Dict[str, Any]:
    """Check diagnostics service status.

    Returns:
        Status including OpenAI availability
    """
    return {
        "service": "diagnostics",
        "openai_available": is_openai_available(),
        "fallback_available": True,
    }
