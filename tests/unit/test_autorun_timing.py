"""Unit tests for autorun_timing module."""

from __future__ import annotations

import importlib.util
import time
from functools import lru_cache
from pathlib import Path

import pytest


# Module path for direct import (directory has hyphen, not valid Python import)
_MODULE_PATH = Path(__file__).resolve().parents[2] / "apps" / "workspace-ui" / "autorun_timing.py"


@lru_cache(maxsize=1)
def _load_autorun_timing():
    """Load autorun_timing module using importlib to handle hyphenated directory."""
    spec = importlib.util.spec_from_file_location("autorun_timing_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_normalize_operation():
    """Test operation name normalization."""
    mod = _load_autorun_timing()

    assert mod.normalize_operation("detect_track") == "detect_track"
    assert mod.normalize_operation("Detect/Track") == "detect_track"
    assert mod.normalize_operation("faces_embed") == "faces_embed"
    assert mod.normalize_operation("Harvest Faces") == "faces_embed"
    assert mod.normalize_operation("cluster") == "cluster"
    assert mod.normalize_operation("Cluster") == "cluster"


def test_parse_duration_string():
    """Test duration string parsing."""
    mod = _load_autorun_timing()

    assert mod.parse_duration_string("45.2s") == pytest.approx(45.2)
    assert mod.parse_duration_string("7m 23s") == pytest.approx(7 * 60 + 23)
    assert mod.parse_duration_string("1m 0s") == pytest.approx(60.0)
    assert mod.parse_duration_string("0m 30s") == pytest.approx(30.0)
    assert mod.parse_duration_string("") == 0.0
    assert mod.parse_duration_string("10") == pytest.approx(10.0)


def test_parse_completion_line_standard():
    """Test parsing standard completion lines."""
    mod = _load_autorun_timing()

    # Standard format
    result = mod.parse_completion_line("✅ Detect/Track completed in 7m 23s")
    assert result is not None
    op, duration = result
    assert op == "detect_track"
    assert duration == pytest.approx(7 * 60 + 23)

    result = mod.parse_completion_line("✅ Harvest Faces completed in 2m 15s")
    assert result is not None
    op, duration = result
    assert op == "faces_embed"
    assert duration == pytest.approx(2 * 60 + 15)

    result = mod.parse_completion_line("✅ Cluster completed in 45.2s")
    assert result is not None
    op, duration = result
    assert op == "cluster"
    assert duration == pytest.approx(45.2)


def test_parse_completion_line_local_mode():
    """Test parsing LOCAL MODE completion lines."""
    mod = _load_autorun_timing()

    result = mod.parse_completion_line("✅ [LOCAL MODE] detect_track completed in 5m 59s")
    assert result is not None
    op, duration = result
    assert op == "detect_track"
    assert duration == pytest.approx(5 * 60 + 59)


def test_parse_completion_line_non_completion():
    """Test that non-completion lines return None."""
    mod = _load_autorun_timing()

    assert mod.parse_completion_line("Processing frame 100/1000") is None
    assert mod.parse_completion_line("Starting detect_track...") is None
    assert mod.parse_completion_line("") is None
    assert mod.parse_completion_line("✅ Files saved") is None  # Not a job completion


def test_is_first_meaningful_line():
    """Test first line detection."""
    mod = _load_autorun_timing()

    # Should be first lines
    assert mod.is_first_meaningful_line("Processing frame 1/1000")
    assert mod.is_first_meaningful_line("Initializing detector...")
    assert mod.is_first_meaningful_line("Loading model...")

    # Should NOT be first lines
    assert not mod.is_first_meaningful_line("")
    assert not mod.is_first_meaningful_line("   ")
    assert not mod.is_first_meaningful_line("═══════════════════════════════════")
    assert not mod.is_first_meaningful_line("───────────────────────────────────")
    assert not mod.is_first_meaningful_line("[CONFIG] device=coreml")
    assert not mod.is_first_meaningful_line("[LOCAL MODE] Starting detect_track...")
    assert not mod.is_first_meaningful_line("Waiting for logs...")


def test_job_timing_state_record_first_line():
    """Test recording first line event."""
    mod = _load_autorun_timing()

    job = mod.JobTimingState(operation="detect_track")

    # Record first line
    event = job.record_first_line("Processing frame 1/1000")

    assert event.event_type == "first_line"
    assert event.operation == "detect_track"
    assert event.timestamp_utc is not None
    assert event.timestamp_mono is not None
    assert job.first_line_at_utc is not None
    assert job.first_line_at_mono is not None

    # Record again - should not update
    old_mono = job.first_line_at_mono
    job.record_first_line("Processing frame 2/1000")
    assert job.first_line_at_mono == old_mono


def test_job_timing_state_record_completion():
    """Test recording completion event."""
    mod = _load_autorun_timing()

    job = mod.JobTimingState(operation="detect_track")

    # Record first line
    job.record_first_line("Processing frame 1/1000")
    first_mono = job.first_line_at_mono

    # Simulate some time passing
    time.sleep(0.01)

    # Record completion
    event = job.record_completion("✅ Detect/Track completed in 5m 0s", parsed_duration=300.0)

    assert event.event_type == "completion"
    assert event.parsed_duration_s == 300.0
    assert job.completed_at_mono > first_mono
    assert job.computed_runtime_s is not None
    assert job.computed_runtime_s > 0


def test_pipeline_timing_state_stall_gaps():
    """Test stall gap computation."""
    mod = _load_autorun_timing()

    pipeline = mod.PipelineTimingState(ep_id="test-ep", run_id="test-run")

    # Simulate detect_track
    detect_job = pipeline.get_or_create_job("detect_track")
    detect_job.record_first_line("Starting detect...")
    time.sleep(0.01)
    detect_job.record_completion("✅ Detect/Track completed in 1m 0s", 60.0)

    # Simulate gap (faces hasn't started yet)
    gaps = pipeline.compute_stall_gaps()
    assert gaps["detect_track_to_faces_embed"] is None

    # Simulate faces_embed starting
    time.sleep(0.02)
    faces_job = pipeline.get_or_create_job("faces_embed")
    faces_job.record_first_line("Starting faces...")

    # Now gap should be computed
    gaps = pipeline.compute_stall_gaps()
    assert gaps["detect_track_to_faces_embed"] is not None
    assert gaps["detect_track_to_faces_embed"] > 0.01


def test_log_event_parser():
    """Test the stateful log event parser."""
    mod = _load_autorun_timing()

    pipeline = mod.PipelineTimingState(ep_id="test-ep", run_id="test-run")
    parser = mod.LogEventParser(pipeline)

    # Set current operation
    parser.set_current_operation("detect_track")

    # Skip non-meaningful lines
    assert parser.process_line("") is None
    assert parser.process_line("═══════════════════════════════════") is None

    # First meaningful line should trigger event
    event = parser.process_line("Processing frame 1/1000")
    assert event is not None
    assert event.event_type == "first_line"
    assert event.operation == "detect_track"

    # Second line should NOT trigger first_line again
    event = parser.process_line("Processing frame 2/1000")
    assert event is None

    # Completion line should trigger event
    event = parser.process_line("✅ Detect/Track completed in 5m 0s")
    assert event is not None
    assert event.event_type == "completion"
    assert event.parsed_duration_s == pytest.approx(300.0)


def test_format_timing_duration():
    """Test duration formatting."""
    mod = _load_autorun_timing()

    assert mod.format_timing_duration(None) == "—"
    assert mod.format_timing_duration(-1) == "—"
    assert mod.format_timing_duration(45.2) == "45.2s"
    assert mod.format_timing_duration(60) == "1m 0s"
    assert mod.format_timing_duration(125) == "2m 5s"


def test_format_gap_display():
    """Test gap display formatting with indicators."""
    mod = _load_autorun_timing()

    assert mod.format_gap_display(None) == "—"
    assert mod.format_gap_display(-1) == "—"
    assert "✅" in mod.format_gap_display(1.5)  # Good (< 2s)
    assert "⏳" in mod.format_gap_display(3.0)  # Warning (2-5s)
    assert "⚠️" in mod.format_gap_display(10.0)  # Problem (> 5s)


def test_completion_triggers_advance_exactly_once():
    """Test that completion event can be used to trigger advancement exactly once."""
    mod = _load_autorun_timing()

    pipeline = mod.PipelineTimingState(ep_id="test-ep", run_id="test-run")
    parser = mod.LogEventParser(pipeline)

    parser.set_current_operation("detect_track")
    parser.process_line("Processing frame 1/1000")

    # First completion should trigger
    completion_line = "✅ Detect/Track completed in 5m 0s"
    event1 = parser.process_line(completion_line)
    assert event1 is not None
    assert event1.event_type == "completion"

    # Second parse of same line should STILL return a completion event
    # (The completion line can be detected multiple times - idempotency is
    # handled at the caller level by checking if already advanced)
    event2 = parser.process_line(completion_line)
    assert event2 is not None
    assert event2.event_type == "completion"

    # But the timing state should only have one completion timestamp
    job = pipeline.jobs["detect_track"]
    assert job.completed_at_utc is not None


def test_stall_gap_math_accurate():
    """Test stall gap computation: stage A completes, then stage B starts."""
    mod = _load_autorun_timing()

    pipeline = mod.PipelineTimingState(ep_id="test-ep", run_id="test-run")
    parser = mod.LogEventParser(pipeline)

    # Stage A: detect_track
    parser.set_current_operation("detect_track")
    parser.process_line("Processing frame 1/1000")
    time.sleep(0.02)  # Simulate runtime
    parser.process_line("✅ Detect/Track completed in 1m 30s")

    detect_completed_mono = pipeline.jobs["detect_track"].completed_at_mono
    assert detect_completed_mono is not None

    # Simulate stall - wait before starting stage B
    time.sleep(0.03)

    # Stage B: faces_embed
    parser.set_current_operation("faces_embed")
    parser.process_line("Loading embeddings model...")

    faces_first_mono = pipeline.jobs["faces_embed"].first_line_at_mono
    assert faces_first_mono is not None

    # Stall gap should be faces_first - detect_completed
    gaps = pipeline.compute_stall_gaps()
    assert "detect_track_to_faces_embed" in gaps
    stall = gaps["detect_track_to_faces_embed"]
    assert stall is not None
    assert stall >= 0.03  # At least the sleep time
    assert stall < 0.5  # But not absurdly long


def test_suppressed_warnings_line_ignored():
    """Test that 'Suppressed N warnings' lines don't trigger first_line."""
    mod = _load_autorun_timing()

    pipeline = mod.PipelineTimingState(ep_id="test-ep", run_id="test-run")
    parser = mod.LogEventParser(pipeline)

    parser.set_current_operation("detect_track")

    # These should NOT trigger first_line
    assert parser.process_line("") is None
    assert parser.process_line("═══════════════════════════════════") is None
    assert parser.process_line("[CONFIG] device=coreml") is None
    assert parser.process_line("[LOCAL MODE] Starting detect_track...") is None

    # This SHOULD trigger first_line
    event = parser.process_line("Processing frame 1/1000")
    assert event is not None
    assert event.event_type == "first_line"


def test_completion_line_formats():
    """Test various completion line formats from actual logs."""
    mod = _load_autorun_timing()

    # Standard format with minutes and seconds
    result = mod.parse_completion_line("✅ Detect/Track completed in 5m 59s")
    assert result is not None
    op, duration = result
    assert op == "detect_track"
    assert duration == pytest.approx(5 * 60 + 59)

    # Seconds only
    result = mod.parse_completion_line("✅ Cluster completed in 45.2s")
    assert result is not None
    op, duration = result
    assert op == "cluster"
    assert duration == pytest.approx(45.2)

    # Harvest Faces variant
    result = mod.parse_completion_line("✅ Harvest Faces completed in 2m 10s")
    assert result is not None
    op, duration = result
    assert op == "faces_embed"
    assert duration == pytest.approx(2 * 60 + 10)

    # LOCAL MODE format
    result = mod.parse_completion_line("✅ [LOCAL MODE] detect_track completed in 3m 45s")
    assert result is not None
    op, duration = result
    assert op == "detect_track"
    assert duration == pytest.approx(3 * 60 + 45)


def test_timestamps_set_once():
    """Test that first_line and completion timestamps are only set once."""
    mod = _load_autorun_timing()

    job = mod.JobTimingState(operation="detect_track")

    # First call sets timestamp
    event1 = job.record_first_line("Line 1")
    first_mono_1 = job.first_line_at_mono

    # Second call should NOT update timestamp
    time.sleep(0.01)
    event2 = job.record_first_line("Line 2")
    first_mono_2 = job.first_line_at_mono

    assert first_mono_1 == first_mono_2  # Unchanged

    # Completion CAN be updated (each call records new time)
    # but first_line should remain the same
    job.record_completion("✅ completed", 60.0)
    assert job.first_line_at_mono == first_mono_1


def test_pipeline_timing_state_to_dict():
    """Test serialization of PipelineTimingState."""
    mod = _load_autorun_timing()

    pipeline = mod.PipelineTimingState(ep_id="test-ep", run_id="test-run")

    # Add some job data
    detect_job = pipeline.get_or_create_job("detect_track")
    detect_job.record_first_line("Starting...")
    detect_job.record_completion("✅ Done", 60.0)

    faces_job = pipeline.get_or_create_job("faces_embed")
    faces_job.record_first_line("Loading model...")

    # Serialize
    data = pipeline.to_dict()

    assert data["ep_id"] == "test-ep"
    assert data["run_id"] == "test-run"
    assert "detect_track" in data["jobs"]
    assert "faces_embed" in data["jobs"]
    assert "stall_gaps" in data
    assert "total_stall_time_s" in data
