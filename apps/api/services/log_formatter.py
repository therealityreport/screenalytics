"""Log formatter for Local mode pipeline operations.

This module provides log line classification and human-friendly formatting
for the detect_track, faces_embed, and cluster pipelines.

Key features:
- Converts JSON progress blobs into readable summary lines (SCENE DETECT, DETECTION, TRACKING)
- Throttles log line output to every ~8 seconds for a cleaner UI
- Suppresses repeated ONNX warnings (shows single user-friendly message)
- Filters out debug noise (per-frame DEBUG lines, internal [job=...] lines, warnings.warn)
- Formats PhaseTracker summaries cleanly
- Outputs canonical config blocks at run start with cute formatting
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# Throttle interval for log line updates (seconds)
LOG_THROTTLE_INTERVAL = 8.0


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

    # Throttling - track last log time per phase
    last_log_time: float = 0.0
    last_log_phase: str = ""

    # Job info tracking (for [job=...] lines)
    job_debug_count: int = 0

    # warnings.warn noise tracking
    warnings_warn_count: int = 0


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
        re.compile(r'^warnings\.warn\('),  # warnings.warn( continuation lines
        re.compile(r'^  "'),  # Continuation of warnings.warn
        re.compile(r'^Applied providers:'),  # ONNX provider logs
        re.compile(r'^\[LOCAL MODE\] detect_track config:', re.IGNORECASE),  # Duplicate config
        re.compile(r'^device=\w+,\s*profile='),  # Duplicate config line
        re.compile(r'^frame_stride=\d+'),  # Duplicate config line
        re.compile(r'^detection_fps_limit='),  # Duplicate config line
        re.compile(r'^VerifyOutputSizes'),  # ONNX internal warnings
        re.compile(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}'),  # Timestamped debug lines
        # Cluster-specific noise
        re.compile(r'^No embeddings found for track'),  # Track without embeddings
        re.compile(r'^Track \d+: using low-quality fallback'),  # Low-quality fallback warnings
        re.compile(r'^\[LOCAL MODE\] (faces_embed|cluster) (starting|config)', re.IGNORECASE),  # Duplicate start messages
        re.compile(r'^device=\w+,\s*cluster_thresh='),  # Duplicate cluster config
        re.compile(r'^faces_total=\d+'),  # Duplicate faces count
        # Faces embed per-frame debug noise (very frequent)
        re.compile(r'^\[EMBED\]\s+Frame\s+\d+:', re.IGNORECASE),  # [EMBED] Frame 123: embedding X faces...
        # Initialization messages (shown in config block instead)
        re.compile(r'^\[INIT\]\s+', re.IGNORECASE),  # [INIT] Loading ArcFace embedder...
        # Standalone config lines
        re.compile(r'^device=\w+$'),  # device=coreml (standalone)
        re.compile(r'^min_face_size='),  # min_face_size=20.0
        re.compile(r'^cpu_threads=\d+$'),  # cpu_threads=2 (standalone)
        re.compile(r'^save_crops='),  # save_crops=True, save_frames=False
        re.compile(r'^total_frames=\d+$'),  # total_frames=61494 (standalone)
        # Demucs/MDX library noise - printed by BagOfModels.__repr__ during loading
        re.compile(r'^Call apply_model on this', re.IGNORECASE),  # Demucs BagOfModels repr
        re.compile(r'^BagOfModels\(', re.IGNORECASE),  # Demucs model repr
        re.compile(r'^<demucs\.', re.IGNORECASE),  # Demucs internal repr
    ]

    # Pattern for warnings.warn( which appears as a standalone line
    WARNINGS_WARN_PATTERN = re.compile(r'^warnings\.warn\(', re.IGNORECASE)

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
        """Format a progress update (detect, track, faces_embed, etc.) with throttling.

        Returns a friendly progress line like:
            DETECTION: 432 / 2,868 frames (15.1%) • 00:01:23 elapsed • ~6.5 fps
        """
        phase = data.get("phase", "unknown")

        # Extract common fields
        current = data.get("current", data.get("frame", data.get("frames_done", data.get("frames_processed"))))
        total = data.get("total", data.get("total_frames", data.get("frames_total")))
        fps = data.get("fps", data.get("fps_infer", data.get("inference_fps")))
        secs_done = data.get("secs_done", data.get("elapsed_seconds", 0))
        cuts = data.get("cuts", data.get("scene_cuts"))

        # Phase-specific data
        clusters = data.get("clusters", data.get("identities_count"))
        faces = data.get("faces", data.get("faces_count", data.get("faces_embedded")))
        tracks = data.get("tracks", data.get("tracks_processed"))

        # Map phase names to friendly labels
        phase_labels = {
            "scene_detect": "SCENE DETECT",
            "detect": "DETECTION",
            "track": "TRACKING",
            "faces_embed": "FACES",
            "cluster": "CLUSTER",
        }
        label = phase_labels.get(phase, phase.upper())

        # Check throttling - only emit log every ~8 seconds per phase
        now = time.time()
        is_phase_change = (phase != self.state.last_log_phase)
        time_since_last = now - self.state.last_log_time

        # Allow log if: phase changed, or 8+ seconds elapsed, or this is completion (100%)
        is_complete = (current is not None and total is not None and
                       total > 0 and current >= total)

        if not is_phase_change and time_since_last < LOG_THROTTLE_INTERVAL and not is_complete:
            # Throttled - update internal state but don't emit log line
            self.state.last_progress_update = {
                "phase": phase, "current": current, "total": total,
            }
            self.state.current_phase = phase
            return None

        # Update throttle state
        self.state.last_log_time = now
        self.state.last_log_phase = phase
        self.state.last_progress_update = {
            "phase": phase, "current": current, "total": total,
        }
        self.state.current_phase = phase

        # Build friendly progress line
        parts = [f"{label}:"]

        if current is not None and total is not None and total > 0:
            pct = (current / total) * 100
            parts.append(f"{current:,} / {total:,} frames ({pct:.1f}%)")
        elif current is not None:
            parts.append(f"{current:,} frames")

        # Add scene cuts for scene_detect phase
        if phase == "scene_detect" and cuts is not None:
            parts.append(f"• {cuts} cuts")

        # Add elapsed time
        if secs_done and secs_done > 0:
            mins = int(secs_done // 60)
            secs = int(secs_done % 60)
            if mins > 0:
                parts.append(f"• {mins:02d}:{secs:02d} elapsed")
            else:
                parts.append(f"• {secs:.1f}s")

        # Add FPS if available
        if fps is not None and fps > 0:
            parts.append(f"• ~{fps:.1f} fps")

        # Add phase-specific counts
        if phase == "cluster" and clusters is not None:
            parts.append(f"• {clusters} identities")

        if phase == "faces_embed":
            if faces is not None:
                parts.append(f"• {faces:,} faces")
            if tracks is not None:
                parts.append(f"• {tracks:,} tracks")

        return " ".join(parts)

    def _format_done_summary(self, data: Dict[str, Any]) -> str:
        """Format completion summary from JSON progress data."""
        summary = data.get("summary", {})
        runtime = summary.get("runtime_sec", data.get("elapsed_seconds"))

        # Map operation to friendly label
        op_label = {
            "detect_track": "Detect/Track",
            "faces_embed": "Harvest Faces",
            "cluster": "Cluster",
        }.get(self.operation, self.operation)

        # Format runtime
        if runtime is not None:
            if runtime >= 60:
                mins = int(runtime // 60)
                secs = int(runtime % 60)
                runtime_str = f"{mins}m {secs}s"
            else:
                runtime_str = f"{runtime:.1f}s"
        else:
            runtime_str = ""

        # Build stats line
        stat_parts = []
        for key in ("tracks", "faces", "clusters", "identities", "detections"):
            val = summary.get(key) or data.get(key)
            if val is not None:
                stat_parts.append(f"{key}: {val:,}" if isinstance(val, int) else f"{key}: {val}")

        line = f"✅ {op_label} completed"
        if runtime_str:
            line += f" in {runtime_str}"
        if stat_parts:
            line += f" ({', '.join(stat_parts)})"

        return line

    def _format_error_summary(self, data: Dict[str, Any]) -> str:
        """Format error summary."""
        error = data.get("error", data.get("message", "Unknown error"))
        return f"❌ ERROR: {error}"

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
        """Handle ONNX runtime warnings - show single friendly message."""
        self.state.onnx_warning_count += 1

        if not self.state.onnx_warning_shown:
            self.state.onnx_warning_shown = True
            # Single user-friendly message instead of raw ONNX output
            formatted = "[WARN] Some ops are falling back to CPU; this may be slower on your Mac."
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
        """Generate final summary if needed.

        Only generates output for significant suppressions (like repeated ONNX warnings).
        Debug frame counts and job debug messages are silently suppressed for cleaner logs.
        """
        # Only show ONNX warning count if there were many suppressed
        if self.state.onnx_warning_count > 5:
            count = self.state.onnx_warning_count - 1  # First one was shown
            msg = f"(Suppressed {count} repeated ONNX warnings)"
            self._formatted_lines.append(msg)
            return msg

        # Silently suppress debug counts - no need to clutter the log
        return None

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
    thermal_limit_info: str | None = None,
) -> str:
    """Generate a canonical config block for the start of a run.

    This replaces all the scattered config lines with a single, clean summary.

    Args:
        thermal_limit_info: String describing thermal limiter, e.g. "taskpolicy -c utility"
                           or "cpulimit 200%". None if no thermal limiting applied.

    Example output:
        [LOCAL MODE] Detect/Track started for rhoslc-s06e08
          Device: COREML  •  Profile: Balanced  •  Stride: 6
          Total: 2,868 frames @ 23.98 fps
          CPU threads: 2 (capped for thermal safety, taskpolicy -c utility)
    """
    op_label = {
        "detect_track": "Detect/Track",
        "faces_embed": "Harvest Faces",
        "cluster": "Cluster",
    }.get(operation, operation)

    lines = [f"[LOCAL MODE] {op_label} started for {episode_id}"]

    # Build compact config line with dots
    config_parts = [f"Device: {device.upper()}"]
    config_parts.append(f"Profile: {profile.title()}")
    if stride is not None:
        config_parts.append(f"Stride: {stride}")
    lines.append("  " + "  •  ".join(config_parts))

    # Total frames line
    if total_frames is not None:
        frame_info = f"  Total: {total_frames:,} frames"
        if fps is not None:
            frame_info += f" @ {fps:.2f} fps"
        lines.append(frame_info)

    # CPU threads line
    thread_info = f"  CPU threads: {cpu_threads}"
    if thermal_limit_info:
        thread_info += f" (capped for thermal safety, {thermal_limit_info})"
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
        ✅ Detect/Track completed in 7m 23s (tracks: 142, faces: 2,341)
    """
    # Map operation to friendly label
    op_label = {
        "detect_track": "Detect/Track",
        "faces_embed": "Harvest Faces",
        "cluster": "Cluster",
    }.get(operation, operation)

    # Format elapsed time
    if elapsed_seconds >= 60:
        mins = int(elapsed_seconds // 60)
        secs = int(elapsed_seconds % 60)
        elapsed_str = f"{mins}m {secs}s"
    else:
        elapsed_str = f"{elapsed_seconds:.1f}s"

    # Status icon and word
    status_map = {
        "completed": ("✅", "completed"),
        "error": ("❌", "FAILED"),
        "timeout": ("⏱️", "timed out"),
        "cancelled": ("⚠️", "cancelled"),
    }
    icon, status_word = status_map.get(status, ("ℹ️", status))

    line = f"{icon} {op_label} {status_word} in {elapsed_str}"

    # Add stats if available
    if stats:
        stat_parts = []
        for key in ("tracks", "faces", "clusters", "identities", "detections"):
            val = stats.get(key)
            if val is not None:
                stat_parts.append(f"{key}: {val:,}" if isinstance(val, int) else f"{key}: {val}")
        if stat_parts:
            line += f" ({', '.join(stat_parts)})"

    return line
