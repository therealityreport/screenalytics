"""Log formatter for Local mode pipeline operations.

This module provides log line classification and human-friendly formatting
for the detect_track, faces_embed, and cluster pipelines.

Key features:
- Converts JSON progress blobs into readable summary lines
- Suppresses repeated ONNX warnings (shows count after first occurrence)
- Filters out debug noise (per-frame DEBUG lines, internal [job=...] lines)
- Formats PhaseTracker summaries cleanly
- Outputs canonical config blocks at run start
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class LogFormatterState:
    """Tracks state across multiple log lines for suppression and summarization."""

    # ONNX warning tracking
    onnx_warning_count: int = 0
    onnx_warning_shown: bool = False

    # Debug frame tracking
    debug_frame_count: int = 0
    last_debug_frame: int = -1

    # Config tracking (to avoid duplicate config lines)
    config_emitted: bool = False

    # Phase tracking
    current_phase: str = ""
    last_progress_update: Dict[str, Any] = field(default_factory=dict)

    # Job info tracking (for [job=...] lines)
    job_debug_count: int = 0


class LogFormatter:
    """Formats raw log lines into human-readable output.

    Usage:
        formatter = LogFormatter(episode_id="rhoslc-s06e08", operation="detect_track")
        for line in raw_lines:
            formatted = formatter.format_line(line)
            if formatted:
                yield formatted
        # Get final suppression summary
        final = formatter.finalize()
        if final:
            yield final
    """

    # Patterns for classification
    ONNX_WARNING_PATTERN = re.compile(
        r'(ONNXRuntime|onnxruntime|ORT|CoreML.*fallback|'
        r'CPUExecutionProvider|coreml.*not.*available|'
        r'EP Error|Execution provider|InferenceSession)',
        re.IGNORECASE
    )

    DEBUG_FRAME_PATTERN = re.compile(
        r'^\[DEBUG\]\s*[Ff]rame\s+(\d+)|'
        r'^Frame\s+(\d+)\s*:|'
        r'^\[frame=(\d+)\]',
        re.IGNORECASE
    )

    JOB_DEBUG_PATTERN = re.compile(
        r'^\[job=|^\[worker=|^\[task=|^\[celery',
        re.IGNORECASE
    )

    PHASE_TRACKER_SUMMARY_PATTERN = re.compile(
        r'^\[SUMMARY\]|^\[PhaseTracker\]|'
        r'phase.*complete|completed.*phase|'
        r'scanned.*frames.*stride|'
        r'total.*frames.*cuts',
        re.IGNORECASE
    )

    JSON_PROGRESS_PATTERN = re.compile(r'^\s*\{.*"phase".*\}\s*$')

    # Config value patterns
    CONFIG_LINE_PATTERN = re.compile(
        r'^(CPU threads|Device|Stride|Profile|Frames|FPS|detector|tracker|threshold)',
        re.IGNORECASE
    )

    # Noise patterns to completely suppress
    NOISE_PATTERNS = [
        re.compile(r'^Loading model', re.IGNORECASE),
        re.compile(r'^Model loaded', re.IGNORECASE),
        re.compile(r'^\s*$'),  # Empty lines
        re.compile(r'^INFO:'),  # Raw logging prefix
        re.compile(r'^DEBUG:'),
        re.compile(r'^WARNING:onnx', re.IGNORECASE),
        re.compile(r'^[\d\-]+\s+[\d:]+'),  # Timestamp prefixes
    ]

    def __init__(
        self,
        episode_id: str,
        operation: str,
        config: Dict[str, Any] | None = None,
    ):
        """Initialize formatter.

        Args:
            episode_id: Episode ID for context
            operation: Operation name (detect_track, faces_embed, cluster)
            config: Optional config dict to emit at start
        """
        self.episode_id = episode_id
        self.operation = operation
        self.config = config or {}
        self.state = LogFormatterState()
        self._formatted_lines: List[str] = []
        self._raw_lines: List[str] = []

    def format_line(self, raw_line: str) -> str | None:
        """Format a single log line.

        Args:
            raw_line: Raw log line from subprocess

        Returns:
            Formatted line, or None if line should be suppressed
        """
        self._raw_lines.append(raw_line)
        line = raw_line.strip()

        if not line:
            return None

        # Check for JSON progress blob
        if self._looks_like_json(line):
            formatted = self._format_json_progress(line)
            if formatted:
                self._formatted_lines.append(formatted)
            return formatted

        # Check for ONNX warnings
        if self.ONNX_WARNING_PATTERN.search(line):
            return self._handle_onnx_warning(line)

        # Check for debug frame noise
        match = self.DEBUG_FRAME_PATTERN.search(line)
        if match:
            return self._handle_debug_frame(line, match)

        # Check for job debug lines
        if self.JOB_DEBUG_PATTERN.match(line):
            return self._handle_job_debug(line)

        # Check for PhaseTracker summaries - always show these
        if self.PHASE_TRACKER_SUMMARY_PATTERN.search(line):
            formatted = self._format_phase_summary(line)
            self._formatted_lines.append(formatted)
            return formatted

        # Check for pure noise
        for pattern in self.NOISE_PATTERNS:
            if pattern.match(line):
                return None

        # Pass through other lines
        self._formatted_lines.append(line)
        return line

    def _looks_like_json(self, line: str) -> bool:
        """Check if line looks like a JSON object."""
        stripped = line.strip()
        return stripped.startswith('{') and stripped.endswith('}')

    def _format_json_progress(self, line: str) -> str | None:
        """Convert JSON progress blob to human-readable format."""
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return line  # Not valid JSON, pass through

        if not isinstance(data, dict):
            return line

        phase = data.get("phase", "")

        # Handle different phase types
        if phase == "done":
            return self._format_done_summary(data)
        elif phase == "error":
            return self._format_error_summary(data)
        elif phase in ("detect", "track", "scene_detect", "faces_embed", "cluster"):
            return self._format_progress_update(data)
        elif phase == "log":
            # Nested log message
            msg = data.get("message", "")
            stream = data.get("stream", "stdout")
            if stream == "stderr" and msg:
                return f"[stderr] {msg}"
            return msg if msg else None
        else:
            # Unknown phase - try to extract useful info
            return self._format_generic_progress(data)

    def _format_progress_update(self, data: Dict[str, Any]) -> str | None:
        """Format a progress update (detect, track, faces_embed, etc.)."""
        phase = data.get("phase", "unknown")

        # Extract common fields
        current = data.get("current", data.get("frame", data.get("frames_processed")))
        total = data.get("total", data.get("total_frames"))
        stride = data.get("stride", data.get("frame_stride"))
        fps = data.get("fps", data.get("fps_infer", data.get("inference_fps")))
        video_time = data.get("video_time", data.get("video_seconds"))
        video_total = data.get("video_total", data.get("video_duration"))

        # Build progress line
        parts = [f"[PROGRESS] phase={phase}"]

        if current is not None and total is not None:
            parts.append(f"frames={current}/{total}")
        elif current is not None:
            parts.append(f"frames={current}")

        if stride is not None:
            parts.append(f"stride={stride}")

        if video_time is not None and video_total is not None:
            parts.append(f"video={video_time:.1f}/{video_total:.1f}s")
        elif video_time is not None:
            parts.append(f"video={video_time:.1f}s")

        if fps is not None:
            parts.append(f"fps={fps:.1f}")

        # Add phase-specific info
        if phase == "cluster":
            clusters = data.get("clusters", data.get("identities_count"))
            if clusters is not None:
                parts.append(f"clusters={clusters}")

        if phase == "faces_embed":
            faces = data.get("faces", data.get("faces_count", data.get("faces_embedded")))
            tracks = data.get("tracks", data.get("tracks_processed"))
            if faces is not None:
                parts.append(f"faces={faces}")
            if tracks is not None:
                parts.append(f"tracks={tracks}")

        result = " ".join(parts)

        # Throttle identical progress updates
        last = self.state.last_progress_update
        if (last.get("phase") == phase and
            last.get("current") == current and
            last.get("total") == total):
            return None

        self.state.last_progress_update = {
            "phase": phase,
            "current": current,
            "total": total,
        }
        self.state.current_phase = phase

        return result

    def _format_done_summary(self, data: Dict[str, Any]) -> str:
        """Format completion summary."""
        summary = data.get("summary", {})
        runtime = summary.get("runtime_sec", data.get("elapsed_seconds"))

        parts = ["[DONE]"]

        if runtime is not None:
            if runtime >= 60:
                mins = int(runtime // 60)
                secs = int(runtime % 60)
                parts.append(f"runtime={mins}m{secs}s")
            else:
                parts.append(f"runtime={runtime:.1f}s")

        # Add summary stats
        for key in ("tracks", "faces", "clusters", "identities", "detections"):
            val = summary.get(key) or data.get(key)
            if val is not None:
                parts.append(f"{key}={val}")

        return " ".join(parts)

    def _format_error_summary(self, data: Dict[str, Any]) -> str:
        """Format error summary."""
        error = data.get("error", data.get("message", "Unknown error"))
        return f"[ERROR] {error}"

    def _format_generic_progress(self, data: Dict[str, Any]) -> str | None:
        """Format unknown progress data."""
        # Skip empty or trivial updates
        if len(data) <= 1:
            return None

        phase = data.get("phase", "")
        if phase == "init" or phase == "start":
            return None  # Skip initialization noise

        # Try to build something useful
        parts = []
        if phase:
            parts.append(f"[{phase.upper()}]")

        for key, val in data.items():
            if key in ("phase", "run_id", "ep_id", "timestamp"):
                continue
            if val is not None:
                parts.append(f"{key}={val}")

        if len(parts) <= 1:
            return None

        return " ".join(parts)

    def _handle_onnx_warning(self, line: str) -> str | None:
        """Handle ONNX runtime warnings - suppress repeats."""
        self.state.onnx_warning_count += 1

        if not self.state.onnx_warning_shown:
            self.state.onnx_warning_shown = True
            formatted = "[WARN] ONNXRuntime fallback: using CPUExecutionProvider for some ops"
            self._formatted_lines.append(formatted)
            return formatted

        # Suppress repeated warnings - will show count at end
        return None

    def _handle_debug_frame(self, line: str, match: re.Match) -> str | None:
        """Handle per-frame debug lines - suppress most."""
        self.state.debug_frame_count += 1

        # Extract frame number
        frame_num = None
        for group in match.groups():
            if group is not None:
                try:
                    frame_num = int(group)
                    break
                except ValueError:
                    pass

        if frame_num is not None:
            self.state.last_debug_frame = frame_num

        # Suppress all per-frame debug output
        return None

    def _handle_job_debug(self, line: str) -> str | None:
        """Handle internal job/worker debug lines."""
        self.state.job_debug_count += 1
        # Suppress all internal debug
        return None

    def _format_phase_summary(self, line: str) -> str:
        """Format PhaseTracker summary lines cleanly."""
        # Already in good format, just clean up
        line = line.strip()

        # Standardize prefix
        if not line.startswith("[SUMMARY]"):
            line = line.replace("[PhaseTracker]", "[SUMMARY]")

        return line

    def finalize(self) -> str | None:
        """Generate final suppression summary.

        Call this after processing all lines to get summary of suppressed items.
        """
        parts = []

        if self.state.onnx_warning_count > 1:
            count = self.state.onnx_warning_count - 1  # First one was shown
            parts.append(f"[WARN] Suppressed {count} repeated ONNXRuntime warnings")

        if self.state.debug_frame_count > 0:
            parts.append(f"[INFO] Processed {self.state.debug_frame_count} frame debug events (hidden)")

        if self.state.job_debug_count > 0:
            parts.append(f"[INFO] Suppressed {self.state.job_debug_count} internal debug messages")

        if not parts:
            return None

        result = "\n".join(parts)
        self._formatted_lines.extend(parts)
        return result

    def get_formatted_lines(self) -> List[str]:
        """Get all formatted lines processed so far."""
        return list(self._formatted_lines)

    def get_raw_lines(self) -> List[str]:
        """Get all raw lines received."""
        return list(self._raw_lines)


def format_config_block(
    operation: str,
    episode_id: str,
    device: str,
    profile: str,
    cpu_threads: int,
    *,
    stride: int | None = None,
    total_frames: int | None = None,
    fps: float | None = None,
    cpulimit_percent: int | None = None,
) -> str:
    """Generate a canonical config block for the start of a run.

    This replaces all the scattered config lines with a single, clean summary.

    Example output:
        [LOCAL MODE] Detect/Track started for rhoslc-s06e08
          Device: coreml
          Profile: balanced
          Stride: 6
          Total frames: 2,868 @ 23.98 fps
          CPU threads: 2 (capped for thermal safety, cpulimit=200%)
    """
    op_label = {
        "detect_track": "Detect/Track",
        "faces_embed": "Faces Embed",
        "cluster": "Cluster",
    }.get(operation, operation)

    lines = [f"[LOCAL MODE] {op_label} started for {episode_id}"]
    lines.append(f"  Device: {device}")
    lines.append(f"  Profile: {profile}")

    if stride is not None:
        lines.append(f"  Stride: {stride}")

    if total_frames is not None:
        frame_info = f"  Total frames: {total_frames:,}"
        if fps is not None:
            frame_info += f" @ {fps:.2f} fps"
        lines.append(frame_info)

    thread_info = f"  CPU threads: {cpu_threads}"
    if cpulimit_percent is not None and cpulimit_percent > 0:
        thread_info += f" (capped for thermal safety, cpulimit={cpulimit_percent}%)"
    else:
        thread_info += " (capped for thermal safety)"
    lines.append(thread_info)

    return "\n".join(lines)


def format_completion_summary(
    operation: str,
    status: str,
    elapsed_seconds: float,
    stats: Dict[str, Any] | None = None,
) -> str:
    """Generate a clean completion summary line.

    Example:
        [LOCAL MODE] detect_track completed in 7m 23s (tracks=142, faces=2,341)
    """
    # Format elapsed time
    if elapsed_seconds >= 60:
        mins = int(elapsed_seconds // 60)
        secs = int(elapsed_seconds % 60)
        elapsed_str = f"{mins}m {secs}s"
    else:
        elapsed_str = f"{elapsed_seconds:.1f}s"

    status_word = {
        "completed": "completed successfully",
        "error": "FAILED",
        "timeout": "timed out",
        "cancelled": "was cancelled",
    }.get(status, status)

    line = f"[LOCAL MODE] {operation} {status_word} in {elapsed_str}"

    # Add stats if available
    if stats:
        stat_parts = []
        for key in ("tracks", "faces", "clusters", "identities", "detections"):
            val = stats.get(key)
            if val is not None:
                stat_parts.append(f"{key}={val:,}" if isinstance(val, int) else f"{key}={val}")
        if stat_parts:
            line += f" ({', '.join(stat_parts)})"

    return line
