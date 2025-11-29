from __future__ import annotations

import logging

import numpy as np

from tools import episode_run


def test_singleton_merge_reduces_fraction(monkeypatch):
    # Configure singleton merge to trigger
    monkeypatch.setattr(
        episode_run,
        "_CLUSTERING_CONFIG_EARLY",
        {
            "singleton_merge": {
                "enabled": True,
                "trigger_singleton_frac": 0.4,
                "secondary_cluster_thresh": 0.8,
                "max_pairs_per_track": 5,
                "min_tracks_per_merged_cluster": 2,
                "max_singleton_merge_iters": 1,
            }
        },
    )

    # Four singletons; pairs (1,2) and (3,4) are similar
    embeds = {
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.99, 0.01], dtype=np.float32),
        3: np.array([0.0, 1.0], dtype=np.float32),
        4: np.array([0.01, 0.99], dtype=np.float32),
    }
    groups = [[1], [2], [3], [4]]

    merged, summary = episode_run._apply_singleton_merge(
        groups,
        embeds,
        primary_cluster_thresh=0.9,
        min_cluster_size=1,
    )

    assert summary["applied"] is True
    assert summary["singleton_fraction_after"] < summary["singleton_fraction_before"]
    assert summary["neighbor_top_k"] == 5
    assert summary["similarity_thresh"] == 0.8
    assert summary["merge_count"] == 2
    # We expect two merged clusters
    merged_sets = {tuple(sorted(g)) for g in merged}
    assert (1, 2) in merged_sets
    assert (3, 4) in merged_sets


def test_singleton_merge_skips_when_disabled(monkeypatch):
    monkeypatch.setattr(
        episode_run,
        "_CLUSTERING_CONFIG_EARLY",
        {"singleton_merge": {"enabled": False}},
    )
    embeds = {1: np.array([1.0, 0.0], dtype=np.float32), 2: np.array([0.0, 1.0], dtype=np.float32)}
    groups = [[1], [2]]
    merged, summary = episode_run._apply_singleton_merge(
        groups,
        embeds,
        primary_cluster_thresh=0.7,
        min_cluster_size=1,
    )
    assert merged == [[1], [2]]
    assert summary["applied"] is False


def test_guardrail_warns_when_merge_disabled(caplog):
    caplog.set_level(logging.WARNING)
    singleton_stats = {
        "enabled": False,
        "before": {"singleton_fraction": 0.7},
        "after": {"singleton_fraction": 0.7},
    }
    episode_run._emit_singleton_guardrail(
        singleton_stats,
        cluster_thresholds={"max_singleton_fraction": 0.5},
        merge_summary={},
        cluster_thresh=0.62,
    )
    assert any("[GUARDRAIL] High singleton fraction" in rec.message for rec in caplog.records)


def test_guardrail_logs_improvement_after_merge(caplog):
    caplog.set_level(logging.INFO)
    singleton_stats = {
        "enabled": True,
        "before": {"singleton_fraction": 0.64},
        "after": {"singleton_fraction": 0.31},
    }
    episode_run._emit_singleton_guardrail(
        singleton_stats,
        cluster_thresholds={"max_singleton_fraction": 0.5},
        merge_summary={"similarity_thresh": 0.6, "neighbor_top_k": 10},
        cluster_thresh=0.62,
    )
    assert any("improved singleton_fraction" in rec.message for rec in caplog.records)
    assert not any("[GUARDRAIL]" in rec.message for rec in caplog.records)


def test_guardrail_warns_when_after_still_high(caplog):
    caplog.set_level(logging.WARNING)
    singleton_stats = {
        "enabled": True,
        "before": {"singleton_fraction": 0.72},
        "after": {"singleton_fraction": 0.68},
    }
    episode_run._emit_singleton_guardrail(
        singleton_stats,
        cluster_thresholds={"max_singleton_fraction": 0.5},
        merge_summary={"similarity_thresh": 0.6, "neighbor_top_k": 5},
        cluster_thresh=0.62,
    )
    assert any("after singleton merge" in rec.message for rec in caplog.records)
