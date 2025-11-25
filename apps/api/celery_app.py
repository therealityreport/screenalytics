"""Celery application configuration for SCREANALYTICS background jobs.

This module sets up Celery with Redis as broker and result backend.
Used for long-running operations like cluster assignments and auto-grouping.

Worker command:
    celery -A apps.api.celery_app:celery_app worker -l info
"""

from celery import Celery

from apps.api.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

celery_app = Celery(
    "screenalytics",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_time_limit=1800,      # 30 min hard cap
    task_soft_time_limit=1200, # 20 min soft limit
    task_track_started=True,   # Track STARTED state
    result_expires=3600,       # Results expire after 1 hour
)

# Auto-discover tasks from the tasks module
celery_app.autodiscover_tasks(["apps.api.tasks"])
