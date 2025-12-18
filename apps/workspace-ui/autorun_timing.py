"""Auto-Run Pipeline Timing Helpers.

Provides log event parsing and timing capture for the Auto-Run pipeline.
Used to detect job completion events from log output and compute timing metrics.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

LOGGER = logging.getLogger(__name__)


# Completion line patterns
# Format: "✅ Detect/Track completed in 7m 23s"
_COMPLETION_PATTERNS = [
    # Standard completion with duration (handles "7m 23s" or "45.2s")
    re.compile(
        r"✅\s*(?P<operation>Detect/Track|Harvest\s*Faces|Cluster)\s+completed\s+in\s+(?P<duration>\d+m\s*\d+s|\d+(?:\.\d+)?s)"
    ),
    # Alternative formats (LOCAL MODE)
    re.compile(
        r"✅\s*\[LOCAL MODE\]\s*(?P<operation>\S+)\s+completed\s+in\s+(?P<duration>\d+m\s*\d+s|\d+(?:\.\d+)?s)"
    ),
    # Fallback: any line with ✅ and "completed"
    re.compile(r"✅.*(?P<operation>detect_track|faces_embed|cluster).*completed", re.IGNORECASE),
]

# Operation name normalization
_OP_NORMALIZE = {
    "detect/track": "detect_track",
    "detecttrack": "detect_track",
    "detect_track": "detect_track",
    "harvest faces": "faces_embed",
    "harvestfaces": "faces_embed",
    "faces_embed": "faces_embed",
    "faces harvest": "faces_embed",
    "cluster": "cluster",
}


def normalize_operation(op: str) -> str:
    """Normalize operation name to canonical form."""
    key = op.lower().replace(" ", "").replace("_", "")
    # First try direct lookup
    if op.lower().replace(" ", "") in _OP_NORMALIZE:
        return _OP_NORMALIZE[op.lower().replace(" ", "")]
    # Try without underscores
    for k, v in _OP_NORMALIZE.items():
        if k.replace("_", "").replace(" ", "") == key:
            return v
    return op.lower().replace(" ", "_")


def parse_duration_string(duration_str: str) -> float:
    """Parse duration string like '7m 23s' or '45.2s' to seconds."""
    if not duration_str:
        return 0.0

    total = 0.0

    # Handle "Xm Ys" format
    mins_match = re.search(r"(\d+(?:\.\d+)?)\s*m", duration_str, re.IGNORECASE)
    secs_match = re.search(r"(\d+(?:\.\d+)?)\s*s", duration_str, re.IGNORECASE)

    if mins_match:
        total += float(mins_match.group(1)) * 60
    if secs_match:
        total += float(secs_match.group(1))

    # If no match, try parsing as plain float
    if total == 0.0 and duration_str:
        try:
            total = float(duration_str.rstrip("s"))
        except ValueError:
            pass

    return total


@dataclass
class JobTimingEvent:
    """Represents a timing event for a job."""

    operation: str
    event_type: str  # "first_line", "completion", "error"
    timestamp_utc: str  # ISO format
    timestamp_mono: float  # monotonic time for accurate deltas
    log_line: Optional[str] = None
    parsed_duration_s: Optional[float] = None  # Duration parsed from completion line

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "event_type": self.event_type,
            "timestamp_utc": self.timestamp_utc,
            "timestamp_mono": self.timestamp_mono,
            "log_line": self.log_line,
            "parsed_duration_s": self.parsed_duration_s,
        }


@dataclass
class JobTimingState:
    """Tracks timing for a single job."""

    operation: str
    first_line_at_utc: Optional[str] = None
    first_line_at_mono: Optional[float] = None
    completed_at_utc: Optional[str] = None
    completed_at_mono: Optional[float] = None
    parsed_runtime_s: Optional[float] = None  # From log completion message
    computed_runtime_s: Optional[float] = None  # first_line to completion

    def record_first_line(self, line: str) -> JobTimingEvent:
        """Record first line event (only if not already recorded)."""
        if self.first_line_at_mono is not None:
            # Already recorded
            return JobTimingEvent(
                operation=self.operation,
                event_type="first_line",
                timestamp_utc=self.first_line_at_utc or "",
                timestamp_mono=self.first_line_at_mono,
                log_line=line,
            )

        now_utc = datetime.now(timezone.utc).isoformat()
        now_mono = time.monotonic()

        self.first_line_at_utc = now_utc
        self.first_line_at_mono = now_mono

        LOGGER.info(
            "[TIMING] %s first_line recorded at %s (mono=%.3f)",
            self.operation, now_utc, now_mono
        )

        return JobTimingEvent(
            operation=self.operation,
            event_type="first_line",
            timestamp_utc=now_utc,
            timestamp_mono=now_mono,
            log_line=line,
        )

    def record_completion(self, line: str, parsed_duration: Optional[float] = None) -> JobTimingEvent:
        """Record completion event."""
        now_utc = datetime.now(timezone.utc).isoformat()
        now_mono = time.monotonic()

        self.completed_at_utc = now_utc
        self.completed_at_mono = now_mono
        self.parsed_runtime_s = parsed_duration

        # Compute runtime from first_line to completion
        if self.first_line_at_mono is not None:
            self.computed_runtime_s = now_mono - self.first_line_at_mono

        LOGGER.info(
            "[TIMING] %s completion recorded at %s (mono=%.3f, parsed_duration=%.1fs, computed_runtime=%.1fs)",
            self.operation, now_utc, now_mono,
            parsed_duration or 0.0,
            self.computed_runtime_s or 0.0
        )

        return JobTimingEvent(
            operation=self.operation,
            event_type="completion",
            timestamp_utc=now_utc,
            timestamp_mono=now_mono,
            log_line=line,
            parsed_duration_s=parsed_duration,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "first_line_at_utc": self.first_line_at_utc,
            "first_line_at_mono": self.first_line_at_mono,
            "completed_at_utc": self.completed_at_utc,
            "completed_at_mono": self.completed_at_mono,
            "parsed_runtime_s": self.parsed_runtime_s,
            "computed_runtime_s": self.computed_runtime_s,
        }


@dataclass
class PipelineTimingState:
    """Tracks timing for an entire auto-run pipeline session."""

    ep_id: str
    run_id: Optional[str] = None
    started_at_utc: Optional[str] = None
    started_at_mono: Optional[float] = None
    jobs: Dict[str, JobTimingState] = field(default_factory=dict)

    def get_or_create_job(self, operation: str) -> JobTimingState:
        """Get or create timing state for a job."""
        op_key = normalize_operation(operation)
        if op_key not in self.jobs:
            self.jobs[op_key] = JobTimingState(operation=op_key)
        return self.jobs[op_key]

    def compute_stall_gaps(self) -> Dict[str, Optional[float]]:
        """Compute gap between each job's completion and the next job's first line.

        Returns dict mapping "{from_op}_to_{to_op}" -> gap_seconds.
        """
        gaps: Dict[str, Optional[float]] = {}
        job_order = ["detect_track", "faces_embed", "cluster"]

        for i in range(len(job_order) - 1):
            from_op = job_order[i]
            to_op = job_order[i + 1]
            gap_key = f"{from_op}_to_{to_op}"

            from_job = self.jobs.get(from_op)
            to_job = self.jobs.get(to_op)

            if (
                from_job
                and to_job
                and from_job.completed_at_mono is not None
                and to_job.first_line_at_mono is not None
            ):
                gaps[gap_key] = to_job.first_line_at_mono - from_job.completed_at_mono
            else:
                gaps[gap_key] = None

        return gaps

    def total_stall_time(self) -> float:
        """Compute total stall time across all inter-job gaps."""
        gaps = self.compute_stall_gaps()
        return sum(g for g in gaps.values() if g is not None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ep_id": self.ep_id,
            "run_id": self.run_id,
            "started_at_utc": self.started_at_utc,
            "started_at_mono": self.started_at_mono,
            "jobs": {k: v.to_dict() for k, v in self.jobs.items()},
            "stall_gaps": self.compute_stall_gaps(),
            "total_stall_time_s": self.total_stall_time(),
        }


def parse_completion_line(line: str) -> Optional[tuple[str, Optional[float]]]:
    """Parse a log line to extract completion event info.

    Returns:
        Tuple of (operation, parsed_duration_seconds) if line is a completion event,
        None otherwise.
    """
    for pattern in _COMPLETION_PATTERNS:
        match = pattern.search(line)
        if match:
            groups = match.groupdict()
            op = groups.get("operation", "")
            duration_str = groups.get("duration", "")

            normalized_op = normalize_operation(op)
            parsed_duration = parse_duration_string(duration_str) if duration_str else None

            LOGGER.debug(
                "[TIMING] Parsed completion line: op=%s (normalized=%s), duration=%s (%.1fs)",
                op, normalized_op, duration_str, parsed_duration or 0.0
            )
            return normalized_op, parsed_duration

    return None


def is_completion_line(line: str) -> bool:
    """Check if a log line is a completion event."""
    return parse_completion_line(line) is not None


def is_first_meaningful_line(line: str) -> bool:
    """Check if a line should be considered the 'first line' of job output.

    Excludes blank lines, config headers, and other non-progress output.
    """
    if not line or not line.strip():
        return False

    # Skip common non-progress prefixes
    skip_prefixes = [
        "═══",  # Config block separator
        "───",  # Section separator
        "[CONFIG]",
        "[LOCAL MODE] Starting",
        "Waiting for",
    ]
    stripped = line.strip()
    for prefix in skip_prefixes:
        if stripped.startswith(prefix):
            return False

    return True


class LogEventParser:
    """Stateful parser for detecting job events from log output stream.

    Tracks first-line and completion events for each job operation.
    """

    def __init__(self, pipeline_state: PipelineTimingState):
        self.pipeline_state = pipeline_state
        self._current_operation: Optional[str] = None
        self._first_line_recorded: Set[str] = set()

    def set_current_operation(self, operation: str) -> None:
        """Set the current operation being logged."""
        self._current_operation = normalize_operation(operation)

    def process_line(self, line: str) -> Optional[JobTimingEvent]:
        """Process a log line and return any timing event detected.

        Returns:
            JobTimingEvent if a first-line or completion event was detected,
            None otherwise.
        """
        if not line:
            return None

        # Check for completion event (any operation)
        completion = parse_completion_line(line)
        if completion:
            op, duration = completion
            job_state = self.pipeline_state.get_or_create_job(op)
            return job_state.record_completion(line, duration)

        # Check for first line (only for current operation)
        if (
            self._current_operation
            and self._current_operation not in self._first_line_recorded
            and is_first_meaningful_line(line)
        ):
            job_state = self.pipeline_state.get_or_create_job(self._current_operation)
            self._first_line_recorded.add(self._current_operation)
            return job_state.record_first_line(line)

        return None


def get_pipeline_timing_state(
    session_state: Dict[str, Any], ep_id: str, run_id: Optional[str] = None
) -> PipelineTimingState:
    """Get or create pipeline timing state from session state.

    Uses session state key pattern: {ep_id}::autorun_timing
    """
    key = f"{ep_id}::autorun_timing"

    if key not in session_state:
        state = PipelineTimingState(
            ep_id=ep_id,
            run_id=run_id,
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            started_at_mono=time.monotonic(),
        )
        session_state[key] = state
        LOGGER.info("[TIMING] Created new pipeline timing state for %s/%s", ep_id, run_id)
    else:
        state = session_state[key]
        # Update run_id if provided and different
        if run_id and state.run_id != run_id:
            state.run_id = run_id

    return state


def clear_pipeline_timing_state(session_state: Dict[str, Any], ep_id: str) -> None:
    """Clear pipeline timing state from session state."""
    key = f"{ep_id}::autorun_timing"
    session_state.pop(key, None)


def format_timing_duration(seconds: Optional[float]) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds is None:
        return "—"

    if seconds < 0:
        return "—"

    if seconds < 60:
        return f"{seconds:.1f}s"

    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"


def format_gap_display(gap_seconds: Optional[float]) -> str:
    """Format a stall gap for display, with warning if too long."""
    if gap_seconds is None:
        return "—"

    if gap_seconds < 0:
        return "—"

    formatted = format_timing_duration(gap_seconds)

    # Add warning indicator for gaps > 5 seconds
    if gap_seconds > 5:
        return f"⚠️ {formatted}"
    elif gap_seconds > 2:
        return f"⏳ {formatted}"
    else:
        return f"✅ {formatted}"
