from __future__ import annotations

import logging

from apps.common import cpu_limits


def test_get_max_threads_prefers_canonical_threads(monkeypatch) -> None:
    cpu_limits._warned_deprecated_env_vars.clear()
    monkeypatch.setenv("SCREENALYTICS_MAX_CPU_THREADS", "5")
    monkeypatch.setenv("SCREANALYTICS_MAX_CPU_THREADS", "2")

    assert cpu_limits.get_max_threads_from_env(default_cores=3) == 5


def test_get_max_threads_prefers_threads_over_percent(monkeypatch) -> None:
    cpu_limits._warned_deprecated_env_vars.clear()
    monkeypatch.setenv("SCREENALYTICS_MAX_CPU_THREADS", "1")
    monkeypatch.setenv("SCREENALYTICS_MAX_CPU_PERCENT", "900")

    assert cpu_limits.get_max_threads_from_env(default_cores=3) == 1


def test_get_max_threads_accepts_deprecated_threads_warns_once(monkeypatch, caplog) -> None:
    cpu_limits._warned_deprecated_env_vars.clear()
    monkeypatch.delenv("SCREENALYTICS_MAX_CPU_THREADS", raising=False)
    monkeypatch.setenv("SCREANALYTICS_MAX_CPU_THREADS", "4")

    with caplog.at_level(logging.WARNING):
        assert cpu_limits.get_max_threads_from_env(default_cores=3) == 4
        assert cpu_limits.get_max_threads_from_env(default_cores=3) == 4

    expected = "Deprecated env var SCREANALYTICS_MAX_CPU_THREADS is set; use SCREENALYTICS_MAX_CPU_THREADS instead"
    assert caplog.text.count(expected) == 1


def test_get_max_threads_accepts_deprecated_percent_warns_once(monkeypatch, caplog) -> None:
    cpu_limits._warned_deprecated_env_vars.clear()
    monkeypatch.delenv("SCREENALYTICS_MAX_CPU_THREADS", raising=False)
    monkeypatch.delenv("SCREENALYTICS_MAX_CPU_PERCENT", raising=False)
    monkeypatch.setenv("SCREANALYTICS_MAX_CPU_PERCENT", "250")

    with caplog.at_level(logging.WARNING):
        assert cpu_limits.get_max_threads_from_env(default_cores=3) == 2
        assert cpu_limits.get_max_threads_from_env(default_cores=3) == 2

    expected = "Deprecated env var SCREANALYTICS_MAX_CPU_PERCENT is set; use SCREENALYTICS_MAX_CPU_PERCENT instead"
    assert caplog.text.count(expected) == 1

