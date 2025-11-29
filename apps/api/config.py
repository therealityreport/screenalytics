"""Central configuration for the SCREENALYTICS API.

Consolidates environment-based config for Redis, Celery, and other services.
"""

import os

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
