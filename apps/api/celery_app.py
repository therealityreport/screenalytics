"""Celery application configuration for Screenalytics background jobs.

This module sets up Celery with Redis as broker and result backend.
Used for long-running operations like cluster assignments and auto-grouping.

THERMAL SAFETY: CPU thread limits are set at worker startup to prevent
laptop overheating. Override with SCREENALYTICS_MAX_CPU_THREADS env var.

Worker command:
    celery -A apps.api.celery_app:celery_app worker -l info
"""

from __future__ import annotations

# CRITICAL: Apply CPU thread limits BEFORE importing numpy/torch/etc.
from apps.common.cpu_limits import apply_global_cpu_limits

apply_global_cpu_limits()

import os

from kombu import Queue
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
    task_time_limit=7200,        # 2 hours hard cap (long videos can take 30+ min)
    task_soft_time_limit=6000,   # 100 min soft limit
    task_track_started=True,     # Track STARTED state
    result_expires=3600,         # Results expire after 1 hour
    worker_prefetch_multiplier=1,  # Don't hog tasks; process one at a time
    task_acks_late=True,           # Ack after completion (safer for long jobs)
    broker_connection_retry_on_startup=True,
)

# Auto-discover tasks from all task modules
celery_app.autodiscover_tasks(["apps.api.tasks", "apps.api.jobs_audio"])

# Audio pipeline queue routing
AUDIO_QUEUES = {
    "audio.ingest": "SCREENALYTICS_AUDIO_INGEST",
    "audio.separate": "SCREENALYTICS_AUDIO_SEPARATE",
    "audio.enhance": "SCREENALYTICS_AUDIO_ENHANCE",
    "audio.diarize": "SCREENALYTICS_AUDIO_DIARIZE",
    "audio.voices": "SCREENALYTICS_AUDIO_VOICES",
    "audio.transcribe": "SCREENALYTICS_AUDIO_TRANSCRIBE",
    "audio.align": "SCREENALYTICS_AUDIO_ALIGN",
    "audio.qc": "SCREENALYTICS_AUDIO_QC",
    "audio.export": "SCREENALYTICS_AUDIO_EXPORT",
    "audio.pipeline": "SCREENALYTICS_AUDIO_PIPELINE",
}

celery_app.conf.task_routes = {
    task_name: {"queue": queue_name}
    for task_name, queue_name in AUDIO_QUEUES.items()
}

# Explicit queue declarations so workers know what to consume
_all_queue_names = set(AUDIO_QUEUES.values()) | {"celery"}
celery_app.conf.task_queues = [Queue(q) for q in sorted(_all_queue_names)]
celery_app.conf.task_default_queue = "celery"

# Allow overriding worker queues/concurrency via env for dev perf tuning
celery_app.conf.worker_concurrency = int(os.getenv("CELERY_WORKER_CONCURRENCY", "2"))
