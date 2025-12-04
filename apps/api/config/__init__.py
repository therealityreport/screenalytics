"""API Configuration Package.

Centralized configuration for Redis, Celery, Smart Suggestions, grouping, and related services.
"""

import os

# =============================================================================
# Core Infrastructure Configuration (from original config.py)
# =============================================================================

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Celery Configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Storage Configuration
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "s3")
S3_BUCKET = os.getenv("S3_BUCKET", os.getenv("BUCKET", ""))

# API Configuration
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Audio Pipeline API Keys
RESEMBLE_API_KEY = os.getenv("RESEMBLE_API_KEY", "")
PYANNOTE_AUTH_TOKEN = os.getenv("PYANNOTE_AUTH_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# =============================================================================
# Smart Suggestions Configuration
# =============================================================================

from .suggestions import (
    SUGGESTION_THRESHOLDS,
    GROUPING_THRESHOLDS,
    TIMEOUTS,
    API_BASE_URL,
    get_confidence_label,
    get_confidence_color,
    get_threshold_description,
    get_all_thresholds,
    # Quality Gate Profiles
    QUALITY_PROFILES,
    DEFAULT_QUALITY_PROFILE,
    get_quality_profile,
    get_sharpness_threshold,
)

__all__ = [
    # Core infrastructure
    "REDIS_URL",
    "CELERY_BROKER_URL",
    "CELERY_RESULT_BACKEND",
    "STORAGE_BACKEND",
    "S3_BUCKET",
    "API_HOST",
    "API_PORT",
    "RESEMBLE_API_KEY",
    "PYANNOTE_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    # Smart Suggestions
    "SUGGESTION_THRESHOLDS",
    "GROUPING_THRESHOLDS",
    "TIMEOUTS",
    "API_BASE_URL",
    "get_confidence_label",
    "get_confidence_color",
    "get_threshold_description",
    "get_all_thresholds",
    # Quality Gate
    "QUALITY_PROFILES",
    "DEFAULT_QUALITY_PROFILE",
    "get_quality_profile",
    "get_sharpness_threshold",
]
