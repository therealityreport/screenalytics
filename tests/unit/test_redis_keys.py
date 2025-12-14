from __future__ import annotations

from apps.api.services import redis_keys


def test_job_lock_keys_include_canonical_first() -> None:
    keys = redis_keys.job_lock_keys("ep-1", "audio_pipeline")
    assert keys[0] == "screenalytics:job_lock:ep-1:audio_pipeline"
    assert "screanalytics:job_lock:ep-1:audio_pipeline" in keys[1:]


def test_job_lock_patterns_include_canonical_first() -> None:
    patterns = redis_keys.job_lock_patterns("audio_pipeline")
    assert patterns[0] == "screenalytics:job_lock:*:audio_pipeline"
    assert "screanalytics:job_lock:*:audio_pipeline" in patterns[1:]


def test_job_history_keys_include_canonical_first() -> None:
    keys = redis_keys.job_history_user_keys("user-1")
    assert keys[0] == "screenalytics:job_history:user-1"
    assert "screanalytics:job_history:user-1" in keys[1:]

