from __future__ import annotations

import sys
import types

import pytest

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional ML dependency
    pytest.skip("numpy is required for tracker gate tests", allow_module_level=True)

if "cv2" not in sys.modules:  # pragma: no cover - testing stub
    _stub = types.SimpleNamespace()
    _stub.IMWRITE_JPEG_QUALITY = 1
    _stub.COLOR_RGB2BGR = 0
    _stub.COLOR_GRAY2BGR = 0

    def _identity(value, *_, **__):
        return value

    _stub.cvtColor = staticmethod(_identity)

    def _imwrite(*_, **__):
        return True

    _stub.imwrite = staticmethod(_imwrite)
    sys.modules["cv2"] = _stub

from tools.episode_run import AppearanceGate, GateConfig


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        return vec
    return (vec / norm).astype(np.float32)


def test_gate_splits_on_hard_similarity() -> None:
    config = GateConfig(
        appear_t_hard=0.8,
        appear_t_soft=0.9,
        appear_streak=3,
        gate_iou=0.1,
        proto_momentum=0.9,
        emb_every=1,
    )
    gate = AppearanceGate(config)
    bbox = np.array([0, 0, 10, 10], dtype=np.float32)
    emb_a = _unit(np.ones(512, dtype=np.float32))
    split, reason, _, _ = gate.process(1, bbox, emb_a, frame_idx=1)
    assert not split
    assert reason is None

    emb_b = _unit(np.ones(512, dtype=np.float32) * -1)
    split, reason, sim, _ = gate.process(1, bbox, emb_b, frame_idx=2)
    assert split
    assert reason == "hard"
    assert sim is not None and sim < config.appear_t_hard


def test_gate_splits_after_soft_streak() -> None:
    config = GateConfig(
        appear_t_hard=0.3,
        appear_t_soft=0.8,
        appear_streak=2,
        gate_iou=0.1,
        proto_momentum=0.5,
        emb_every=1,
    )
    gate = AppearanceGate(config)
    bbox = np.array([0, 0, 5, 5], dtype=np.float32)
    base = _unit(np.ones(512, dtype=np.float32))
    gate.process(7, bbox, base, frame_idx=1)
    varied = np.concatenate([np.ones(200, dtype=np.float32), np.zeros(312, dtype=np.float32)])
    off = _unit(varied)
    sim = float(np.dot(base, off))
    assert config.appear_t_hard < sim < config.appear_t_soft
    split, reason, _, _ = gate.process(7, bbox, off, frame_idx=2)
    assert not split
    split, reason, _, _ = gate.process(7, bbox, off, frame_idx=3)
    assert split
    assert reason == "streak"


def test_gate_splits_on_iou_when_no_embedding() -> None:
    config = GateConfig(
        appear_t_hard=0.1,
        appear_t_soft=0.2,
        appear_streak=3,
        gate_iou=0.5,
        proto_momentum=0.9,
        emb_every=1,
    )
    gate = AppearanceGate(config)
    box_a = np.array([0, 0, 10, 10], dtype=np.float32)
    box_b = np.array([100, 100, 120, 120], dtype=np.float32)
    split, _, _, _ = gate.process(3, box_a, None, frame_idx=1)
    assert not split
    split, reason, _, iou = gate.process(3, box_b, None, frame_idx=2)
    assert split
    assert reason == "iou"
    assert iou < config.gate_iou


def test_gate_summary_reports_average_similarity() -> None:
    config = GateConfig(
        appear_t_hard=0.4,
        appear_t_soft=0.6,
        appear_streak=3,
        gate_iou=0.05,
        proto_momentum=0.8,
        emb_every=1,
    )
    gate = AppearanceGate(config)
    bbox = np.array([0, 0, 8, 8], dtype=np.float32)
    emb = _unit(np.ones(512, dtype=np.float32))
    gate.process(5, bbox, emb, frame_idx=1)
    gate.process(5, bbox, emb, frame_idx=2)
    summary = gate.summary()
    assert summary["splits"]["total"] >= 0
    assert summary["avg_sim_kept"] is None or 0.0 <= summary["avg_sim_kept"] <= 1.0
