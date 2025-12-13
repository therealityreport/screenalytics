"""TRR metadata database client for reading canonical metadata from core.* tables.

This module provides read-only access to the TRR Postgres metadata database.
All schema creation and migrations for core.* tables are owned by TRR BACKEND;
SCREENALYTICS is a read-only consumer.

Environment variables:
    TRR_DB_URL: Postgres connection URL (e.g., postgresql://user:pass@host:5432/trr_metadata)
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Mapping, Optional

# Optional dependency: psycopg2 is only needed if TRR_DB_URL is configured.
# This allows the module to be imported in CI/test contexts without psycopg2.
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None  # type: ignore[assignment]
    RealDictCursor = None  # type: ignore[assignment,misc]
    _PSYCOPG2_AVAILABLE = False

TRR_DB_URL_ENV = "TRR_DB_URL"
LOGGER = logging.getLogger(__name__)


def get_conn():
    """Create a new Postgres connection to the TRR metadata database.

    Returns:
        psycopg2 connection with RealDictCursor (returns rows as dicts)

    Raises:
        RuntimeError: If TRR_DB_URL is not set or psycopg2 is not installed
        psycopg2.Error: If connection fails
    """
    if not _PSYCOPG2_AVAILABLE:
        raise RuntimeError("psycopg2 is not installed; TRR metadata DB is unavailable")
    url = os.getenv(TRR_DB_URL_ENV)
    if not url:
        raise RuntimeError(f"{TRR_DB_URL_ENV} is not set")
    LOGGER.debug("Connecting to TRR metadata DB")
    return psycopg2.connect(url, cursor_factory=RealDictCursor)


def get_show_by_slug(show_slug: str) -> Optional[Mapping[str, Any]]:
    """Read show metadata by slug from core.shows.

    Expected columns in core.shows (owned by TRR BACKEND):
      - show_id (PK)
      - show_slug (text, e.g. 'RHOBH', 'RHOSLC')
      - title (text, display name)
      - franchise (nullable text)
      - network (nullable text)
      - imdb_series_id (nullable text)
      - tmdb_series_id (nullable text)
      - is_active (boolean)

    Args:
        show_slug: Show identifier (e.g., 'RHOBH', 'RHOSLC')

    Returns:
        Dict-like mapping of show record, or None if not found.

    Raises:
        RuntimeError: If TRR_DB_URL is not configured
        psycopg2.Error: If database query fails
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    show_id,
                    show_slug,
                    title,
                    franchise,
                    network,
                    imdb_series_id,
                    tmdb_series_id,
                    is_active
                FROM core.shows
                WHERE show_slug = %s
                LIMIT 1;
                """,
                (show_slug,),
            )
            row = cur.fetchone()
            LOGGER.debug("Retrieved show for show_slug=%r: %s", show_slug, "found" if row else "not found")
    finally:
        conn.close()

    return row


def get_seasons_for_show(show_id: str) -> List[Mapping[str, Any]]:
    """Read seasons for a show from core.seasons.

    Expected columns in core.seasons (owned by TRR BACKEND):
      - season_id (PK)
      - show_id (FK to core.shows)
      - season_number (int)
      - label (nullable text, e.g. "Season 1")
      - is_current (boolean)

    Args:
        show_id: Show primary key from core.shows

    Returns:
        List of season records as dict-like mappings, ordered by season_number.
        Returns empty list if no seasons found.

    Raises:
        RuntimeError: If TRR_DB_URL is not configured
        psycopg2.Error: If database query fails
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    season_id,
                    show_id,
                    season_number,
                    label,
                    is_current
                FROM core.seasons
                WHERE show_id = %s
                ORDER BY season_number;
                """,
                (show_id,),
            )
            rows = cur.fetchall()
            LOGGER.info("Retrieved %d seasons for show_id=%r", len(rows), show_id)
    finally:
        conn.close()

    return rows


def get_episodes_for_show(show_id: str) -> List[Mapping[str, Any]]:
    """Read episodes for a show from core.episodes joined with core.seasons.

    Expected columns (owned by TRR BACKEND):
      - episode_id (PK from core.episodes)
      - season_id (FK from core.episodes)
      - season_number (from core.seasons)
      - episode_number (int from core.episodes)
      - episode_code (nullable text, e.g. "S01E01")
      - title (nullable text, episode title)
      - air_date (nullable date)
      - runtime_seconds (nullable int)

    Args:
        show_id: Show primary key from core.shows

    Returns:
        List of episode records as dict-like mappings, ordered by season/episode number.
        Returns empty list if no episodes found.

    Raises:
        RuntimeError: If TRR_DB_URL is not configured
        psycopg2.Error: If database query fails
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    e.episode_id,
                    e.season_id,
                    s.season_number,
                    e.episode_number,
                    e.episode_code,
                    e.title,
                    e.air_date,
                    e.runtime_seconds
                FROM core.episodes e
                JOIN core.seasons s ON s.season_id = e.season_id
                WHERE s.show_id = %s
                ORDER BY s.season_number, e.episode_number;
                """,
                (show_id,),
            )
            rows = cur.fetchall()
            LOGGER.info("Retrieved %d episodes for show_id=%r", len(rows), show_id)
    finally:
        conn.close()

    return rows


def get_cast_for_show(show_slug: str) -> List[Mapping[str, Any]]:
    """Read canonical cast list for a given show from core.cast.

    Expected columns in core.cast (backfilled by TRR BACKEND):
      - cast_id (PK, text or UUID)
      - show_id (FK to core.shows)
      - show_slug (text, e.g. 'RHOBH')
      - person_name (text)
      - imdb_person_id (nullable text)
      - tmdb_person_id (nullable text)
      - bravo_cast_slug (nullable text, used as a stable slug in UI)
      - sort_order (nullable int, for display order)

    Args:
        show_slug: Show identifier (e.g., 'RHOBH', 'RHOSLC')

    Returns:
        List of cast member records as dict-like mappings.
        Returns empty list if no cast found for the show.

    Raises:
        RuntimeError: If TRR_DB_URL is not configured
        psycopg2.Error: If database query fails
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    cast_id,
                    show_id,
                    show_slug,
                    person_name,
                    imdb_person_id,
                    tmdb_person_id,
                    bravo_cast_slug,
                    sort_order
                FROM core.cast
                WHERE show_slug = %s
                ORDER BY sort_order NULLS LAST, person_name ASC;
                """,
                (show_slug,),
            )
            rows = cur.fetchall()
            LOGGER.info("Retrieved %d cast members for show_slug=%r", len(rows), show_slug)
    finally:
        conn.close()

    return rows


__all__ = [
    "get_conn",
    "get_show_by_slug",
    "get_seasons_for_show",
    "get_episodes_for_show",
    "get_cast_for_show",
    "TRR_DB_URL_ENV",
]
