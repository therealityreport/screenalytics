from __future__ import annotations

CANONICAL_PREFIX = "screenalytics"
LEGACY_PREFIXES = ("screanalytics",)


def job_lock_key(episode_id: str, operation: str, *, prefix: str = CANONICAL_PREFIX) -> str:
    return f"{prefix}:job_lock:{episode_id}:{operation}"


def job_lock_keys(episode_id: str, operation: str) -> list[str]:
    return [job_lock_key(episode_id, operation, prefix=CANONICAL_PREFIX)] + [
        job_lock_key(episode_id, operation, prefix=legacy) for legacy in LEGACY_PREFIXES
    ]


def job_lock_pattern(operation: str, *, prefix: str = CANONICAL_PREFIX) -> str:
    return f"{prefix}:job_lock:*:{operation}"


def job_lock_patterns(operation: str) -> list[str]:
    return [job_lock_pattern(operation, prefix=CANONICAL_PREFIX)] + [
        job_lock_pattern(operation, prefix=legacy) for legacy in LEGACY_PREFIXES
    ]


def job_history_user_key(user_id: str | None, *, prefix: str = CANONICAL_PREFIX) -> str:
    return f"{prefix}:job_history:{user_id or 'anonymous'}"


def job_history_user_keys(user_id: str | None) -> list[str]:
    return [job_history_user_key(user_id, prefix=CANONICAL_PREFIX)] + [
        job_history_user_key(user_id, prefix=legacy) for legacy in LEGACY_PREFIXES
    ]

