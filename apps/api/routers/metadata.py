"""Metadata API endpoints for TRR canonical metadata.

This router provides read-only access to canonical metadata from the TRR
Postgres metadata database (core.* tables). All write operations and schema
migrations are managed by TRR BACKEND.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from apps.api.services.trr_metadata_db import (
    get_cast_for_show,
    get_episodes_for_show,
    get_seasons_for_show,
    get_show_by_slug,
)

router = APIRouter(prefix="/metadata", tags=["metadata"])
LOGGER = logging.getLogger(__name__)


@router.get("/shows/{show_slug}")
def read_show(show_slug: str):
    """Return show metadata from core.shows.

    Args:
        show_slug: Show identifier (e.g., 'RHOBH', 'RHOSLC')

    Returns:
        JSON object with show fields:
        - show_id: Primary key
        - show_slug: Show identifier
        - title: Display name
        - franchise: Franchise name (nullable)
        - network: Network name (nullable)
        - imdb_series_id: IMDb series ID (nullable)
        - tmdb_series_id: TMDb series ID (nullable)
        - is_active: Whether show is currently active

    Raises:
        HTTPException: 404 if show not found
        HTTPException: 500 if database connection or query fails
    """
    try:
        show = get_show_by_slug(show_slug)
    except Exception as exc:
        LOGGER.error("Failed to fetch show for show_slug=%r: %s", show_slug, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch show metadata from database",
        ) from exc

    if not show:
        raise HTTPException(
            status_code=404,
            detail=f"Show not found for show_slug={show_slug!r}",
        )

    return show


@router.get("/shows/{show_slug}/seasons")
def read_show_seasons(show_slug: str):
    """Return seasons for a show from core.seasons.

    Args:
        show_slug: Show identifier (e.g., 'RHOBH', 'RHOSLC')

    Returns:
        JSON object with:
        - show: Show metadata object
        - seasons: Array of season objects with fields:
          - season_id: Primary key
          - show_id: Foreign key to show
          - season_number: Season number
          - label: Display label (nullable)
          - is_current: Whether this is the current season

    Raises:
        HTTPException: 404 if show not found
        HTTPException: 500 if database connection or query fails
    """
    try:
        show = get_show_by_slug(show_slug)
        if not show:
            raise HTTPException(
                status_code=404,
                detail=f"Show not found for show_slug={show_slug!r}",
            )

        seasons = get_seasons_for_show(show["show_id"])
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.error("Failed to fetch seasons for show_slug=%r: %s", show_slug, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch seasons metadata from database",
        ) from exc

    return {"show": show, "seasons": seasons}


@router.get("/shows/{show_slug}/episodes")
def read_show_episodes(show_slug: str):
    """Return episodes for a show from core.episodes.

    Args:
        show_slug: Show identifier (e.g., 'RHOBH', 'RHOSLC')

    Returns:
        JSON object with:
        - show: Show metadata object
        - episodes: Array of episode objects with fields:
          - episode_id: Primary key
          - season_id: Foreign key to season
          - season_number: Season number (from join)
          - episode_number: Episode number within season
          - episode_code: Episode code (nullable, e.g. "S01E01")
          - title: Episode title (nullable)
          - air_date: Air date (nullable)
          - runtime_seconds: Runtime in seconds (nullable)

    Raises:
        HTTPException: 404 if show not found
        HTTPException: 500 if database connection or query fails
    """
    try:
        show = get_show_by_slug(show_slug)
        if not show:
            raise HTTPException(
                status_code=404,
                detail=f"Show not found for show_slug={show_slug!r}",
            )

        episodes = get_episodes_for_show(show["show_id"])
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.error("Failed to fetch episodes for show_slug=%r: %s", show_slug, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch episodes metadata from database",
        ) from exc

    return {"show": show, "episodes": episodes}


@router.get("/shows/{show_slug}/cast")
def read_show_cast(show_slug: str):
    """Return the canonical cast list for a given show slug from core.cast.

    Args:
        show_slug: Show identifier (e.g., 'RHOBH', 'RHOSLC')

    Returns:
        JSON array of cast member records with fields:
        - cast_id: Primary key (text or UUID)
        - show_slug: Show identifier
        - person_name: Cast member's name
        - imdb_person_id: IMDb person ID (nullable)
        - tmdb_person_id: TMDb person ID (nullable)
        - bravo_cast_slug: Stable UI slug (nullable)
        - sort_order: Display order hint (nullable)

    Raises:
        HTTPException: 404 if no cast found for the show
        HTTPException: 500 if database connection or query fails
    """
    try:
        cast_rows = get_cast_for_show(show_slug)
    except Exception as exc:
        LOGGER.error("Failed to fetch cast for show_slug=%r: %s", show_slug, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch cast metadata from database",
        ) from exc

    if not cast_rows:
        # Return 404 when no cast is found for the requested show
        raise HTTPException(
            status_code=404,
            detail=f"Show or cast not found for show_slug={show_slug!r}",
        )

    return cast_rows


__all__ = ["router"]
