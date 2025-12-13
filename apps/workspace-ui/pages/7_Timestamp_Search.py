"""Timestamp Search - Visual debugging tool to preview frames with face detections."""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
import yaml
from PIL import Image
from urllib.parse import quote

# Optional: Drawing canvas
try:
    from streamlit_drawable_canvas import st_canvas
    HAS_DRAWABLE_CANVAS = True
except ImportError:
    HAS_DRAWABLE_CANVAS = False

# Optional: PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
PROJECT_ROOT = WORKSPACE_DIR.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402


def load_pipeline_configs() -> Dict[str, Any]:
    """Load pipeline config values from YAML files."""
    config_dir = PROJECT_ROOT / "config" / "pipeline"
    configs = {
        "detection": {},
        "tracking": {},
        "embedding": {},
        "clustering": {},
    }

    # Detection config
    det_path = config_dir / "detection.yaml"
    if det_path.exists():
        with det_path.open() as f:
            det = yaml.safe_load(f) or {}
            configs["detection"] = {
                "confidence_th": det.get("confidence_th", 0.50),
                "min_size": det.get("min_size", 16),
            }

    # Tracking config
    trk_path = config_dir / "tracking.yaml"
    if trk_path.exists():
        with trk_path.open() as f:
            trk = yaml.safe_load(f) or {}
            configs["tracking"] = {
                "track_thresh": trk.get("track_thresh", 0.55),
                "match_thresh": trk.get("match_thresh", 0.65),
                "track_buffer": trk.get("track_buffer", 90),
                "new_track_thresh": trk.get("new_track_thresh", 0.60),
            }

    # Embedding/sampling config
    emb_path = config_dir / "faces_embed_sampling.yaml"
    if emb_path.exists():
        with emb_path.open() as f:
            emb = yaml.safe_load(f) or {}
            qg = emb.get("quality_gating", {})
            configs["embedding"] = {
                "min_quality_score": qg.get("min_quality_score", 1.5),
                "min_confidence": qg.get("min_confidence", 0.45),
                "min_blur_score": qg.get("min_blur_score", 18.0),
                "max_yaw_angle": qg.get("max_yaw_angle", 60.0),
                "max_pitch_angle": qg.get("max_pitch_angle", 45.0),
            }

    # Clustering config
    cls_path = config_dir / "clustering.yaml"
    if cls_path.exists():
        with cls_path.open() as f:
            cls = yaml.safe_load(f) or {}
            configs["clustering"] = {
                "cluster_thresh": cls.get("cluster_thresh", 0.52),
                "min_cluster_size": cls.get("min_cluster_size", 1),
                "min_identity_sim": cls.get("min_identity_sim", 0.45),
            }

    return configs

# ── Page Init ─────────────────────────────────────────────────────────────────
cfg = helpers.init_page("Timestamp Search")

# ── Episode Selection ─────────────────────────────────────────────────────────
ep_id = helpers.get_ep_id()
if not ep_id:
    st.info("Select an episode from the sidebar to search timestamps.")
    st.stop()

# Canonicalize
ep_id = ep_id.lower()

# ── Page Title ────────────────────────────────────────────────────────────────
st.title("Timestamp Search")
st.caption("Enter a timestamp to see the frame with detected faces and their track/cluster assignments.")

# ── Session State Keys ────────────────────────────────────────────────────────
_ts_preview_key = f"{ep_id}::timestamp_preview_input"
_ts_preview_result_key = f"{ep_id}::timestamp_preview_result"
_frames_with_faces_key = f"{ep_id}::frames_with_faces"
_last_previewed_ts_key = f"{ep_id}::last_previewed_ts"


# ── Helper Functions ──────────────────────────────────────────────────────────
def _auto_format_timestamp(raw: str) -> str:
    """Auto-format raw digits into MM:SS.ms format.

    Examples:
        "012771" -> "01:27.71"
        "0127" -> "01:27"
        "130" -> "01:30"
        "01:27.71" -> "01:27.71" (already formatted)
        "00:3270" -> "00:32.70" (reformat seconds part)
    """
    # If already has colon, check if seconds part needs reformatting
    if ":" in raw:
        parts = raw.split(":", 1)
        mm_part = parts[0]
        ss_part = parts[1] if len(parts) > 1 else ""

        # If seconds part already has a dot, it's fully formatted
        if "." in ss_part:
            return raw

        # If seconds part is >2 digits, split into SS.ms
        ss_digits = "".join(c for c in ss_part if c.isdigit())
        if len(ss_digits) > 2:
            ss = min(int(ss_digits[:2]), 59)
            ms = ss_digits[2:]
            return f"{mm_part}:{ss:02d}.{ms}"

        return raw

    # Strip non-digits
    digits = "".join(c for c in raw if c.isdigit())
    if not digits:
        return "00:00"

    # Pad to at least 4 digits for MM:SS
    # Format: MMSS or MMSSFF (where FF is fractional/ms)
    if len(digits) <= 4:
        # MMSS format
        digits = digits.zfill(4)
        mm = min(int(digits[:2]), 99)  # Clamp to valid values
        ss = min(int(digits[2:4]), 59)  # Clamp seconds to 0-59
        return f"{mm:02d}:{ss:02d}"
    else:
        # MMSSFF format (6+ digits -> MM:SS.FF)
        digits = digits.zfill(6)
        mm = min(int(digits[:2]), 99)  # Clamp to valid values
        ss = min(int(digits[2:4]), 59)  # Clamp seconds to 0-59
        ms = digits[4:]
        return f"{mm:02d}:{ss:02d}.{ms}"


def _parse_timestamp_input(ts_str: str) -> float | None:
    """Parse MM:SS or MM:SS.ms format to seconds."""
    ts_str = ts_str.strip()
    if not ts_str:
        return None

    # Try MM:SS.ms format
    match = re.match(r"^(\d+):(\d{1,2})(?:\.(\d+))?$", ts_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        ms_str = match.group(3)
        ms = float(f"0.{ms_str}") if ms_str else 0.0
        return minutes * 60 + seconds + ms

    # Try just seconds
    try:
        return float(ts_str)
    except ValueError:
        return None


def _load_frames_with_faces() -> list:
    """Load the list of frame indices that have face detections."""
    cached = st.session_state.get(_frames_with_faces_key)
    if cached is not None:
        return cached

    try:
        resp = helpers.api_get(f"/episodes/{ep_id}/frames_with_faces", timeout=30)
        frames = resp.get("frames", [])
        st.session_state[_frames_with_faces_key] = frames
        return frames
    except requests.RequestException:
        return []


def _load_unassigned_intervals() -> list[tuple[float, float]]:
    """Load unassigned track intervals from screentime.json."""
    screentime_path = helpers.DATA_ROOT / "analytics" / ep_id / "screentime.json"
    if not screentime_path.exists():
        return []

    try:
        data = json.loads(screentime_path.read_text(encoding="utf-8"))
        timeline = data.get("timeline", [])
        intervals: list[tuple[float, float]] = []
        for entry in timeline:
            if entry.get("cast_id") is None:
                intervals.extend(entry.get("intervals", []))
        # Sort by start time
        intervals.sort(key=lambda x: x[0])
        return intervals
    except (json.JSONDecodeError, OSError):
        return []


def _find_prev_frame_with_faces(current_frame: int, frames_list: list) -> int | None:
    """Find the previous frame index that has face detections using binary search."""
    if not frames_list:
        return None

    # Binary search for the largest frame < current_frame
    import bisect
    idx = bisect.bisect_left(frames_list, current_frame)
    if idx > 0:
        return frames_list[idx - 1]
    return None


def _find_next_frame_with_faces(current_frame: int, frames_list: list) -> int | None:
    """Find the next frame index that has face detections using binary search."""
    if not frames_list:
        return None

    # Binary search for the smallest frame > current_frame
    import bisect
    idx = bisect.bisect_right(frames_list, current_frame)
    if idx < len(frames_list):
        return frames_list[idx]
    return None


def build_full_face_table(faces: list) -> list:
    """Build comprehensive face data table from captured faces."""
    rows = []
    for f in faces:
        bbox = f.get("bbox", [])
        scores = f.get("scores", {})

        # Size calculation
        if bbox and len(bbox) >= 4:
            width = int(bbox[2] - bbox[0])
            height = int(bbox[3] - bbox[1])
            size_str = f"{width}x{height}"
        else:
            size_str = "-"

        rows.append({
            "Name": f.get("name") or "-",
            "Track": f.get("track_id") or "-",
            "Identity": (f.get("identity_id") or "-")[:12] + "..." if f.get("identity_id") else "-",
            "Person": (f.get("person_id") or "-")[:12] + "..." if f.get("person_id") else "-",
            "Cast": (f.get("cast_id") or "-")[:12] + "..." if f.get("cast_id") else "-",
            "Conf": f"{f.get('conf', 0):.0%}" if f.get("conf") else "-",
            "Size": size_str,
            "Quality": f"{scores.get('quality', 0):.2f}" if scores.get("quality") else "-",
            "Blur": f"{scores.get('blur', 0):.1f}" if scores.get("blur") else "-",
            "Yaw": f"{scores.get('pose_yaw', 0):.0f}°" if scores.get("pose_yaw") is not None else "-",
            "Pitch": f"{scores.get('pose_pitch', 0):.0f}°" if scores.get("pose_pitch") is not None else "-",
            "Det": "✓" if f.get("detected", True) else "✗",
            "Trk": "✓" if f.get("tracked") else "✗",
            "Hrv": "✓" if f.get("harvested") else "✗",
            "Cls": "✓" if f.get("clustered") else "✗",
            "Issue": f.get("unidentified_reason") or "-",
        })
    return rows


def build_csv_export(ep_id: str, captured_timestamps: list) -> str:
    """Build CSV export from captured timestamps."""
    output = StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Timestamp", "Frame", "Captured At", "Notes",
        "Detected", "Tracked", "Harvested", "Clustered",
        "Face Name", "Track ID", "Identity ID", "Confidence",
        "Quality", "Blur", "Yaw", "Pitch", "BBox", "Issue"
    ])

    for entry in captured_timestamps:
        ts = entry["timestamp_s"]
        ts_str = f"{int(ts//60):02d}:{ts%60:05.2f}"
        frame = entry.get("frame_idx", "")
        captured_at = entry.get("captured_at", "")
        notes = entry.get("notes", "")
        summary = entry.get("pipeline_summary", {})

        # If no faces, write summary row
        faces = entry.get("faces", [])
        if not faces:
            writer.writerow([
                ts_str, frame, captured_at, notes,
                summary.get("detected", 0),
                summary.get("tracked", 0),
                summary.get("harvested", 0),
                summary.get("clustered", 0),
                "", "", "", "", "", "", "", "", "", ""
            ])
        else:
            for i, f in enumerate(faces):
                scores = f.get("scores", {})
                bbox = f.get("bbox", [])
                writer.writerow([
                    ts_str if i == 0 else "",
                    frame if i == 0 else "",
                    captured_at if i == 0 else "",
                    notes if i == 0 else "",
                    summary.get("detected", 0) if i == 0 else "",
                    summary.get("tracked", 0) if i == 0 else "",
                    summary.get("harvested", 0) if i == 0 else "",
                    summary.get("clustered", 0) if i == 0 else "",
                    f.get("name", ""),
                    f.get("track_id", ""),
                    f.get("identity_id", ""),
                    f"{f.get('conf', 0):.2f}" if f.get("conf") else "",
                    f"{scores.get('quality', 0):.2f}" if scores.get("quality") else "",
                    f"{scores.get('blur', 0):.1f}" if scores.get("blur") else "",
                    f"{scores.get('pose_yaw', 0):.0f}" if scores.get("pose_yaw") is not None else "",
                    f"{scores.get('pose_pitch', 0):.0f}" if scores.get("pose_pitch") is not None else "",
                    str(bbox) if bbox else "",
                    f.get("unidentified_reason", ""),
                ])

    return output.getvalue()


def apply_canvas_annotations(img: Image.Image, annotation_data: dict) -> Image.Image:
    """Apply canvas annotations to PIL image."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)

    objects = annotation_data.get("objects", [])
    for obj in objects:
        obj_type = obj.get("type")

        if obj_type == "path":
            # Freehand drawing
            path = obj.get("path", [])
            points = []
            for cmd in path:
                if len(cmd) >= 3 and cmd[0] in ["M", "L", "Q"]:
                    points.append((cmd[1], cmd[2]))
            if len(points) >= 2:
                draw.line(points, fill=obj.get("stroke", "red"), width=int(obj.get("strokeWidth", 3)))

        elif obj_type == "rect":
            # Rectangle
            left = obj.get("left", 0)
            top = obj.get("top", 0)
            width = obj.get("width", 0)
            height = obj.get("height", 0)
            draw.rectangle([left, top, left + width, top + height],
                          outline=obj.get("stroke", "red"),
                          width=int(obj.get("strokeWidth", 3)))

        elif obj_type == "circle":
            # Circle/ellipse
            left = obj.get("left", 0)
            top = obj.get("top", 0)
            radius = obj.get("radius", 0)
            draw.ellipse([left - radius, top - radius, left + radius, top + radius],
                        outline=obj.get("stroke", "red"),
                        width=int(obj.get("strokeWidth", 3)))

    return img


def generate_pdf_report(ep_id: str, captured_timestamps: list) -> bytes:
    """Generate PDF report with annotated images and face data."""
    if not HAS_REPORTLAB:
        return b""

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"Timestamp Analysis Report: {ep_id}", styles['Title']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    for entry in captured_timestamps:
        ts = entry["timestamp_s"]
        ts_str = f"{int(ts//60):02d}:{ts%60:05.2f}"

        # Section header
        story.append(Paragraph(f"Timestamp: {ts_str} (Frame {entry.get('frame_idx', '?')})", styles['Heading2']))

        # Notes
        if entry.get("notes"):
            story.append(Paragraph(f"<b>Notes:</b> {entry['notes']}", styles['Normal']))

        # Pipeline summary
        summary = entry.get("pipeline_summary", {})
        story.append(Paragraph(
            f"<b>Pipeline:</b> Detected: {summary.get('detected', 0)} | "
            f"Tracked: {summary.get('tracked', 0)} | "
            f"Harvested: {summary.get('harvested', 0)} | "
            f"Clustered: {summary.get('clustered', 0)}",
            styles['Normal']
        ))

        # Screenshot image (if available)
        screenshot_url = entry.get("screenshot_url")
        if screenshot_url:
            try:
                # Download or load image
                if screenshot_url.startswith("http"):
                    response = requests.get(screenshot_url, timeout=30)
                    img = Image.open(BytesIO(response.content))
                else:
                    img = Image.open(screenshot_url)

                # Apply annotations if present
                if entry.get("annotation_data"):
                    img = apply_canvas_annotations(img, entry["annotation_data"])

                # Save to buffer and add to PDF
                img_buffer = BytesIO()
                img.save(img_buffer, format="JPEG")
                img_buffer.seek(0)

                # Scale to fit page width
                aspect = img.height / img.width
                img_width = 6 * inch
                img_height = img_width * aspect

                rl_img = RLImage(img_buffer, width=img_width, height=img_height)
                story.append(rl_img)
            except Exception as e:
                story.append(Paragraph(f"[Image not available: {e}]", styles['Normal']))

        # Face data table
        faces = entry.get("faces", [])
        if faces:
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("<b>Faces:</b>", styles['Normal']))

            table_data = [["Name", "Track", "Conf", "Quality", "Status", "Issue"]]
            for f in faces:
                scores = f.get("scores", {})
                status = "OK" if f.get("clustered") else ("Not Clustered" if f.get("harvested") else "Not Harvested")
                table_data.append([
                    f.get("name") or "-",
                    str(f.get("track_id", "-")),
                    f"{f.get('conf', 0):.0%}" if f.get("conf") else "-",
                    f"{scores.get('quality', 0):.2f}" if scores.get("quality") else "-",
                    status,
                    f.get("unidentified_reason") or "-",
                ])

            table = Table(table_data, colWidths=[1.5*inch, 0.6*inch, 0.6*inch, 0.6*inch, 1*inch, 1.2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            story.append(table)

        story.append(Spacer(1, 0.3*inch))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def render_annotation_canvas(entry: dict, idx: int, ep_id: str):
    """Render drawable canvas for annotation."""
    if not HAS_DRAWABLE_CANVAS:
        st.caption("*Drawing annotations: Install `streamlit-drawable-canvas` to enable*")
        return

    st.markdown("**Draw Annotations:**")

    # Load background image
    screenshot_url = entry.get("screenshot_url")
    if not screenshot_url:
        st.warning("No screenshot available")
        return

    # Fetch image for canvas background
    try:
        if screenshot_url.startswith("http"):
            response = requests.get(screenshot_url, timeout=30)
            bg_image = Image.open(BytesIO(response.content))
        else:
            bg_image = Image.open(screenshot_url)
    except Exception as e:
        st.warning(f"Could not load image: {e}")
        return

    # Resize for canvas (max 700px width)
    scale = min(1.0, 700 / bg_image.width)
    canvas_width = int(bg_image.width * scale)
    canvas_height = int(bg_image.height * scale)

    try:
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=bg_image,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="freedraw",
            key=f"{ep_id}::canvas_{idx}",
        )

        # Save annotation data
        if canvas_result and canvas_result.json_data:
            entry["annotation_data"] = canvas_result.json_data
    except (AttributeError, Exception) as e:
        st.caption(f"*Drawing canvas unavailable (Streamlit version incompatibility)*")


# ── Check for pending timestamp from unassigned list ─────────────────────────
_pending_ts_key = f"{ep_id}::pending_unassigned_ts"
_pending_ts = st.session_state.pop(_pending_ts_key, None)
if _pending_ts is not None:
    # Set the input value and flag to trigger preview
    st.session_state[_ts_preview_key] = _pending_ts
    _auto_preview = True
else:
    _auto_preview = False

# ── Input UI ──────────────────────────────────────────────────────────────────
ts_col1, ts_col2 = st.columns([3, 1])
with ts_col1:
    raw_ts_input = st.text_input(
        "Timestamp (MM:SS or MM:SS.ms)",
        value="00:00",
        key=_ts_preview_key,
        placeholder="e.g., 0130 or 013050 → auto-formats",
        help="Type digits like 0130 for 01:30, or 013050 for 01:30.50. Colons added automatically.",
    )
    # Auto-format the input
    ts_input = _auto_format_timestamp(raw_ts_input)
    # Show formatted version if different from raw input
    if ts_input != raw_ts_input and raw_ts_input.strip():
        st.caption(f"→ {ts_input}")

with ts_col2:
    st.write("")  # Spacing
    preview_clicked = st.button("🔍 Preview", key=f"{ep_id}::ts_preview_btn", use_container_width=True)

# Auto-preview when timestamp changes (user types new value and presses Enter)
_last_previewed = st.session_state.get(_last_previewed_ts_key)
_parsed_ts = _parse_timestamp_input(ts_input)
if _parsed_ts is not None and ts_input != "00:00" and ts_input != _last_previewed:
    _auto_preview = True
    st.session_state[_last_previewed_ts_key] = ts_input

# ── Unassigned Track Timestamps ──────────────────────────────────────────────
unassigned_intervals = _load_unassigned_intervals()
if unassigned_intervals:
    with st.expander(f"Unassigned Track Timestamps ({len(unassigned_intervals)} intervals)", expanded=False):
        st.caption("Click a timestamp to load it for preview.")

        # Display in columns for compact layout
        cols_per_row = 5
        for i in range(0, len(unassigned_intervals), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(unassigned_intervals):
                    start_s, end_s = unassigned_intervals[idx]
                    # Format as MM:SS.ss
                    mins = int(start_s // 60)
                    secs = start_s % 60
                    label = f"{mins}:{secs:05.2f}"
                    with col:
                        if st.button(
                            label,
                            key=f"{ep_id}::unassigned_ts_{idx}",
                            use_container_width=True,
                        ):
                            # Set pending timestamp and rerun to update input + trigger preview
                            st.session_state[_pending_ts_key] = label
                            st.rerun()


# ── API Call ──────────────────────────────────────────────────────────────────
if preview_clicked or _auto_preview:
    timestamp_s = _parse_timestamp_input(ts_input)
    if timestamp_s is None:
        st.error("Invalid timestamp format. Use MM:SS or MM:SS.ms (e.g., 01:30 or 01:30.50)")
        st.stop()
    else:
        # Track this as the last previewed timestamp
        st.session_state[_last_previewed_ts_key] = ts_input
        with st.spinner(f"Loading frame at {ts_input}..."):
            try:
                preview_resp = helpers.api_get(
                    f"/episodes/{ep_id}/timestamp/{timestamp_s}/preview",
                    timeout=30,
                )
                st.session_state[_ts_preview_result_key] = preview_resp
            except requests.RequestException as exc:
                st.error(helpers.describe_error("timestamp preview", exc))
                st.session_state[_ts_preview_result_key] = None


# ── Helper for frame navigation ──────────────────────────────────────────────
def _load_frame_by_index(frame_index: int) -> None:
    """Load a specific frame by index."""
    with st.spinner(f"Loading frame {frame_index}..."):
        try:
            preview_resp = helpers.api_get(
                f"/episodes/{ep_id}/frame/{frame_index}/preview",
                timeout=30,
            )
            st.session_state[_ts_preview_result_key] = preview_resp
        except requests.RequestException as exc:
            st.error(helpers.describe_error("frame preview", exc))
            st.session_state[_ts_preview_result_key] = None  # Clear stale data on error


# ── Results Display ───────────────────────────────────────────────────────────
preview_result = st.session_state.get(_ts_preview_result_key)
if preview_result:
    # Show gap warning if we had to find a nearby frame
    gap_seconds = preview_result.get("gap_seconds", 0)
    actual_ts = preview_result.get("timestamp_s", 0)
    frame_idx = preview_result.get("frame_idx", 0)
    fps = preview_result.get("fps", 24)

    if gap_seconds > 0.5:
        st.warning(
            f"No faces detected at exact timestamp. Showing nearest frame with faces "
            f"(gap: {gap_seconds:.2f}s)"
        )

    # Frame info with navigation arrows
    actual_mm = int(actual_ts // 60)
    actual_ss = actual_ts % 60

    # Estimate max frame from duration and fps
    duration_s = preview_result.get("duration_s", 0)
    estimated_max_frame = int(duration_s * fps) if duration_s > 0 and fps > 0 else None

    # Load unassigned intervals for navigation (prev/next unassigned timestamp)
    nav_intervals = _load_unassigned_intervals()

    # Find prev/next unassigned interval relative to current timestamp
    prev_interval_ts = None
    next_interval_ts = None
    for start_s, end_s in nav_intervals:
        if start_s < actual_ts - 0.5:  # Allow small tolerance
            prev_interval_ts = start_s
        if start_s > actual_ts + 0.5 and next_interval_ts is None:
            next_interval_ts = start_s
            break

    nav_col1, nav_col2, nav_col3 = st.columns([1, 4, 1])
    with nav_col1:
        prev_disabled = prev_interval_ts is None
        if st.button("◀ Prev", key="prev_frame", use_container_width=True, disabled=prev_disabled):
            # Navigate to previous unassigned interval using pending pattern
            mins = int(prev_interval_ts // 60)
            secs = prev_interval_ts % 60
            st.session_state[_pending_ts_key] = f"{mins:02d}:{secs:05.2f}"
            st.session_state[_last_previewed_ts_key] = None  # Force re-preview
            st.rerun()
    with nav_col2:
        interval_count_str = f" ({len(nav_intervals)} unassigned)" if nav_intervals else ""
        st.markdown(
            f"<div style='text-align:center;padding:8px 0;'>"
            f"<strong>Frame {frame_idx}</strong> @ {actual_mm}:{actual_ss:05.2f} ({fps:.2f} fps)"
            f"{interval_count_str}</div>",
            unsafe_allow_html=True,
        )
    with nav_col3:
        next_disabled = next_interval_ts is None
        if st.button("Next ▶", key="next_frame", use_container_width=True, disabled=next_disabled):
            # Navigate to next unassigned interval using pending pattern
            mins = int(next_interval_ts // 60)
            secs = next_interval_ts % 60
            st.session_state[_pending_ts_key] = f"{mins:02d}:{secs:05.2f}"
            st.session_state[_last_previewed_ts_key] = None  # Force re-preview
            st.rerun()

    # Display the preview image
    preview_url = preview_result.get("url")
    if preview_url:
        # Handle local paths vs URLs
        if preview_url.startswith("/") or preview_url.startswith("data/"):
            # Local path - read and display
            local_path = Path(preview_url)
            if local_path.exists():
                st.image(str(local_path), use_container_width=True)
            else:
                st.error(f"Preview image not found: {preview_url}")
        else:
            # S3 presigned URL
            st.image(preview_url, use_container_width=True)

    # Display pipeline summary first
    pipeline_summary = preview_result.get("pipeline_summary", {})
    if pipeline_summary:
        sum_detected = pipeline_summary.get("detected", 0)
        sum_tracked = pipeline_summary.get("tracked", 0)
        sum_harvested = pipeline_summary.get("harvested", 0)
        sum_clustered = pipeline_summary.get("clustered", 0)

        # Show pipeline funnel as metrics
        pipe_cols = st.columns(4)
        pipe_cols[0].metric("Detected", sum_detected, help="Faces found by RetinaFace detector")
        pipe_cols[1].metric("Tracked", sum_tracked, help="Faces linked to ByteTrack tracks")
        pipe_cols[2].metric("Harvested", sum_harvested, help="Faces that passed quality gate and were embedded")
        pipe_cols[3].metric("Clustered", sum_clustered, help="Faces assigned to identity clusters")

        # Pipeline drop-off info available via metrics above - no automatic warnings

    # Display face info table
    faces = preview_result.get("faces", [])
    if faces:
        st.caption(f"**Detected faces: {len(faces)}**")

        # Build summary table with separate columns for each pipeline stage
        face_rows = []
        for f in faces:
            name = f.get("name")
            identity_id = f.get("identity_id")
            track_id = f.get("track_id")
            scores = f.get("scores", {})
            bbox = f.get("bbox", [])
            conf = f.get("conf", scores.get("det_score"))

            # Pipeline stages - infer from data presence (API may not return explicit flags)
            is_detected = f.get("detected", True)  # If we have the face, it was detected
            is_tracked = bool(track_id)  # Has track_id = was tracked
            is_harvested = f.get("harvested", bool(scores.get("quality") or scores.get("embedding_norm")))
            is_clustered = bool(identity_id)  # Has identity_id = was clustered

            # Additional metadata
            cluster_id = identity_id[:8] + "..." if identity_id else "-"

            # Format confidence as percentage
            conf_str = f"{conf * 100:.0f}%" if conf is not None else "-"

            # Calculate face size from bbox [x1, y1, x2, y2]
            if bbox and len(bbox) >= 4:
                face_width = int(bbox[2] - bbox[0])
                face_height = int(bbox[3] - bbox[1])
                total_pixels = face_width * face_height
                size_str = f"{face_width}x{face_height} ({total_pixels:,}px)"
                bbox_str = f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
            else:
                size_str = "-"
                bbox_str = "-"

            # Similarity scores
            identity_sim = scores.get("identity_similarity", scores.get("cluster_similarity", None))
            cast_sim = scores.get("cast_similarity", scores.get("facebank_similarity", None))
            track_sim = scores.get("track_similarity", scores.get("track_cohesion", None))
            ambiguity = scores.get("ambiguity", scores.get("ambiguity_score", None))

            if name:
                label = name
            elif identity_id:
                label = f"[{identity_id[:12]}...]"
            else:
                label = f"Track {track_id}"

            # Format quality scores
            quality_str = "-"
            if scores:
                parts = []
                if "quality" in scores:
                    parts.append(f"Q:{scores['quality']:.2f}")
                if "blur" in scores:
                    parts.append(f"B:{scores['blur']:.2f}")
                if "det_score" in scores:
                    parts.append(f"D:{scores['det_score']:.2f}")
                quality_str = " ".join(parts) if parts else "-"

            # Format similarity scores (show what's available)
            sim_parts = []
            if identity_sim is not None:
                sim_parts.append(f"Id:{identity_sim:.0%}")
            if cast_sim is not None:
                sim_parts.append(f"Cst:{cast_sim:.0%}")
            if track_sim is not None:
                sim_parts.append(f"Trk:{track_sim:.0%}")
            if ambiguity is not None:
                sim_parts.append(f"Amb:{ambiguity:.0%}")
            sim_str = " ".join(sim_parts) if sim_parts else "-"

            face_rows.append({
                "Label": label,
                "Track": track_id,
                "Conf": conf_str,
                "Size": size_str,
                "Cluster": cluster_id,
                "Det": "✓" if is_detected else "✗",
                "Trk": "✓" if is_tracked else "✗",
                "Hrv": "✓" if is_harvested else "✗",
                "Cls": "✓" if is_clustered else "✗",
                "Quality": quality_str,
                "Similarity": sim_str,
                "BBox": bbox_str,
            })

        st.dataframe(face_rows, use_container_width=True, hide_index=True)

        # INSIGHTS + DEBUG button
        st.markdown("---")
        if st.button("🔍 INSIGHTS + DEBUG", type="primary", key=f"{ep_id}::insights_debug"):
            st.session_state[f"{ep_id}::show_insights"] = True

        # Show insights panel
        if st.session_state.get(f"{ep_id}::show_insights"):
            st.markdown("### Pipeline Diagnostics")

            for f in faces:
                track_id = f.get("track_id")
                identity_id = f.get("identity_id")
                name = f.get("name")
                scores = f.get("scores", {})

                # Pipeline stages - infer from data presence (API may not return explicit flags)
                is_detected = f.get("detected", True)  # If we have the face, it was detected
                is_tracked = bool(track_id)  # Has track_id = was tracked
                is_harvested = f.get("harvested", bool(scores.get("quality") or scores.get("embedding_norm")))
                is_clustered = bool(identity_id)  # Has identity_id = was clustered
                person_id = f.get("person_id")
                cast_id = f.get("cast_id")

                # Build label
                if name:
                    face_label = f"**{name}** (Track {track_id})"
                elif identity_id:
                    face_label = f"**[{identity_id[:12]}...]** (Track {track_id})"
                else:
                    face_label = f"**Track {track_id}**"

                with st.expander(face_label, expanded=False):
                    # Build copy text as we go
                    copy_lines = []
                    copy_lines.append(f"Pipeline Diagnostics: {face_label.replace('**', '')}")
                    copy_lines.append(f"Episode: {ep_id}")
                    copy_lines.append("")

                    # Pipeline stage breakdown with WHY explanations
                    st.markdown("**Pipeline Stage Analysis:**")
                    copy_lines.append("=== Pipeline Stage Analysis ===")

                    # Detection stage
                    det_col1, det_col2 = st.columns([1, 4])
                    with det_col1:
                        st.markdown("**Det:**" + (" ✓" if is_detected else " ✗"))
                    with det_col2:
                        if is_detected:
                            det_score = scores.get("det_score", 0)
                            st.markdown(f"Face detected by RetinaFace (conf: {det_score:.2f})")
                            copy_lines.append(f"Det: ✓ - Face detected by RetinaFace (conf: {det_score:.2f})")
                        else:
                            st.error("Not detected - face not found by detector")
                            copy_lines.append("Det: ✗ - Not detected - face not found by detector")

                    # Tracking stage
                    trk_col1, trk_col2 = st.columns([1, 4])
                    with trk_col1:
                        st.markdown("**Trk:**" + (" ✓" if is_tracked else " ✗"))
                    with trk_col2:
                        if is_tracked:
                            st.markdown(f"Assigned to track {track_id} by ByteTrack")
                            copy_lines.append(f"Trk: ✓ - Assigned to track {track_id} by ByteTrack")
                        else:
                            st.error("Not tracked - detection not linked to any track (may be too brief or low confidence)")
                            copy_lines.append("Trk: ✗ - Not tracked - detection not linked to any track")

                    # Harvest stage
                    hrv_col1, hrv_col2 = st.columns([1, 4])
                    with hrv_col1:
                        st.markdown("**Hrv:**" + (" ✓" if is_harvested else " ✗"))
                    with hrv_col2:
                        if is_harvested:
                            quality = scores.get("quality", 0)
                            blur = scores.get("blur", 0)
                            st.markdown(f"Passed quality gating (Q:{quality:.2f}, Blur:{blur:.2f})")
                            copy_lines.append(f"Hrv: ✓ - Passed quality gating (Q:{quality:.2f}, Blur:{blur:.2f})")
                        else:
                            quality = scores.get("quality", 0)
                            blur = scores.get("blur", 0)
                            pose_yaw = scores.get("pose_yaw", 0)
                            pose_pitch = scores.get("pose_pitch", 0)
                            st.error("**NOT HARVESTED** - Failed quality gating")
                            st.caption(f"Quality: {quality:.3f} (threshold ~0.5) | Blur: {blur:.3f} | Pose: yaw={pose_yaw:.0f}°, pitch={pose_pitch:.0f}°")
                            copy_lines.append("Hrv: ✗ - NOT HARVESTED - Failed quality gating")
                            copy_lines.append(f"    Quality: {quality:.3f} | Blur: {blur:.3f} | Yaw: {pose_yaw:.0f}° | Pitch: {pose_pitch:.0f}°")
                            if quality < 0.5:
                                st.warning("↳ Quality score too low (likely occluded, partial, or low resolution)")
                                copy_lines.append("    → Quality score too low")
                            if abs(pose_yaw) > 45:
                                st.warning("↳ Face turned too far sideways (yaw > 45°)")
                                copy_lines.append("    → Face turned too far sideways (yaw > 45°)")
                            if abs(pose_pitch) > 30:
                                st.warning("↳ Face tilted too far up/down (pitch > 30°)")
                                copy_lines.append("    → Face tilted too far up/down (pitch > 30°)")

                    # Clustering stage
                    cls_col1, cls_col2 = st.columns([1, 4])
                    with cls_col1:
                        st.markdown("**Cls:**" + (" ✓" if is_clustered else " ✗"))
                    with cls_col2:
                        if is_clustered:
                            st.markdown(f"Assigned to identity `{identity_id[:12]}...`")
                            copy_lines.append(f"Cls: ✓ - Assigned to identity {identity_id}")
                        elif is_harvested:
                            st.error("**NOT CLUSTERED** - Embedding didn't match any cluster")
                            st.caption("Track may be an outlier, singleton, or new face not seen before")
                            copy_lines.append("Cls: ✗ - NOT CLUSTERED - Embedding didn't match any cluster")
                        else:
                            st.caption("Cannot cluster - face was not harvested")
                            copy_lines.append("Cls: - - Cannot cluster (face was not harvested)")

                    # Assignment chain (for clustered faces)
                    if is_clustered:
                        st.markdown("---")
                        st.markdown("**Assignment Chain:**")
                        copy_lines.append("")
                        copy_lines.append("=== Assignment Chain ===")

                        chain_steps = [f"Track {track_id}", f"→ Identity `{identity_id[:12]}...`"]
                        chain_copy = [f"Track {track_id}", f"→ Identity {identity_id}"]
                        issues = []

                        if person_id:
                            chain_steps.append(f"→ Person `{person_id}`")
                            chain_copy.append(f"→ Person {person_id}")
                        else:
                            chain_steps.append("→ ❌ **No person**")
                            chain_copy.append("→ ❌ No person")
                            issues.append("Identity not linked to any person record")

                        if cast_id:
                            chain_steps.append(f"→ Cast `{cast_id}`")
                            chain_copy.append(f"→ Cast {cast_id}")
                        elif person_id:
                            chain_steps.append("→ ❌ **No cast_id**")
                            chain_copy.append("→ ❌ No cast_id")
                            issues.append("Person record has no cast_id assigned")

                        if name:
                            chain_steps.append(f"→ **{name}**")
                            chain_copy.append(f"→ {name}")
                        elif cast_id:
                            chain_steps.append("→ ❌ **No name**")
                            chain_copy.append("→ ❌ No name")
                            issues.append("Cast member has no name set")

                        st.code(" ".join(chain_steps))
                        copy_lines.append(" ".join(chain_copy))

                        if issues:
                            st.markdown("**Why Not Identified:**")
                            copy_lines.append("")
                            copy_lines.append("Why Not Identified:")
                            for issue in issues:
                                st.warning(f"• {issue}")
                                copy_lines.append(f"  • {issue}")

                    # Quality scores detail
                    if scores:
                        st.markdown("---")
                        st.markdown("**Quality Scores:**")
                        copy_lines.append("")
                        copy_lines.append("=== Quality Scores ===")
                        sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
                        with sc1:
                            st.metric("Quality", f"{scores.get('quality', 0):.3f}")
                        with sc2:
                            st.metric("Blur", f"{scores.get('blur', 0):.3f}")
                        with sc3:
                            st.metric("Det", f"{scores.get('det_score', 0):.3f}")
                        with sc4:
                            st.metric("Yaw", f"{scores.get('pose_yaw', 0):.0f}°")
                        with sc5:
                            st.metric("Pitch", f"{scores.get('pose_pitch', 0):.0f}°")
                        with sc6:
                            st.metric("Emb Norm", f"{scores.get('embedding_norm', 0):.2f}")
                        copy_lines.append(f"Quality: {scores.get('quality', 0):.3f}")
                        copy_lines.append(f"Blur: {scores.get('blur', 0):.3f}")
                        copy_lines.append(f"Det Score: {scores.get('det_score', 0):.3f}")
                        copy_lines.append(f"Yaw: {scores.get('pose_yaw', 0):.0f}°")
                        copy_lines.append(f"Pitch: {scores.get('pose_pitch', 0):.0f}°")
                        copy_lines.append(f"Emb Norm: {scores.get('embedding_norm', 0):.2f}")

                    # =====================================================
                    # ACTIONABLE ANALYSIS - What to do to fix this
                    # =====================================================
                    st.markdown("---")
                    st.markdown("### 🔧 ACTION REQUIRED")

                    # Load actual config values
                    pipeline_cfg = load_pipeline_configs()
                    det_cfg = pipeline_cfg.get("detection", {})
                    trk_cfg = pipeline_cfg.get("tracking", {})
                    emb_cfg = pipeline_cfg.get("embedding", {})
                    cls_cfg = pipeline_cfg.get("clustering", {})

                    # Build analysis based on pipeline state
                    action_details = []

                    # Check each failure point and provide specific fix instructions
                    if not is_detected:
                        det_score = scores.get("det_score", 0)
                        curr_thresh = det_cfg.get("confidence_th", 0.50)
                        action_details.append({
                            "issue": "Face not detected",
                            "cause": f"RetinaFace confidence ({det_score:.2f}) below threshold ({curr_thresh})",
                            "fix": "Lower the detection confidence threshold",
                            "face_values": {"det_score": det_score},
                            "params": [
                                {"file": "config/pipeline/detection.yaml", "param": "confidence_th", "current": f"{curr_thresh}", "recommended": f"{max(0.3, det_score - 0.05):.2f}", "reason": f"Face det_score is {det_score:.2f}, lower threshold to include it"},
                            ],
                            "command": f"python tools/episode_run.py {ep_id} --stages detect --det-thresh {max(0.3, det_score - 0.05):.2f}",
                        })

                    elif not is_tracked:
                        det_score = scores.get("det_score", 0)
                        curr_track_thresh = trk_cfg.get("track_thresh", 0.55)
                        curr_match_thresh = trk_cfg.get("match_thresh", 0.65)
                        curr_buffer = trk_cfg.get("track_buffer", 90)
                        curr_new_thresh = trk_cfg.get("new_track_thresh", 0.60)
                        action_details.append({
                            "issue": "Detection not linked to track",
                            "cause": f"ByteTrack couldn't associate this detection (det_score={det_score:.2f}) with a track",
                            "fix": "Adjust ByteTrack parameters to be more permissive",
                            "face_values": {"det_score": det_score},
                            "params": [
                                {"file": "config/pipeline/tracking.yaml", "param": "track_thresh", "current": f"{curr_track_thresh}", "recommended": f"{max(0.3, curr_track_thresh - 0.15):.2f}", "reason": "Lower threshold allows weaker detections to form tracks"},
                                {"file": "config/pipeline/tracking.yaml", "param": "new_track_thresh", "current": f"{curr_new_thresh}", "recommended": f"{max(0.4, curr_new_thresh - 0.15):.2f}", "reason": "Lower threshold to start new tracks more easily"},
                                {"file": "config/pipeline/tracking.yaml", "param": "track_buffer", "current": f"{curr_buffer}", "recommended": f"{curr_buffer + 30}", "reason": "Longer buffer keeps tracks alive through brief occlusions"},
                                {"file": "config/pipeline/tracking.yaml", "param": "match_thresh", "current": f"{curr_match_thresh}", "recommended": f"{max(0.5, curr_match_thresh - 0.1):.2f}", "reason": "Lower IoU threshold allows more detection-to-track matches"},
                            ],
                            "command": f"python tools/episode_run.py {ep_id} --stages track --track-thresh {max(0.3, curr_track_thresh - 0.15):.2f}",
                        })

                    elif not is_harvested:
                        quality = scores.get("quality", 0)
                        blur = scores.get("blur", 0)
                        pose_yaw = scores.get("pose_yaw", 0)
                        pose_pitch = scores.get("pose_pitch", 0)

                        # Get actual config thresholds
                        curr_quality = emb_cfg.get("min_quality_score", 1.5)
                        curr_blur = emb_cfg.get("min_blur_score", 18.0)
                        curr_yaw = emb_cfg.get("max_yaw_angle", 60.0)
                        curr_pitch = emb_cfg.get("max_pitch_angle", 45.0)

                        params = []
                        # Note: quality score in API may be different scale than config
                        if quality < curr_quality:
                            new_val = max(0.5, quality - 0.3)
                            params.append({"file": "config/pipeline/faces_embed_sampling.yaml", "param": "min_quality_score", "current": f"{curr_quality}", "recommended": f"{new_val:.1f}", "reason": f"Face quality is {quality:.2f}, lower threshold to include it"})
                        if blur < curr_blur:
                            new_val = max(10.0, blur - 5.0)
                            params.append({"file": "config/pipeline/faces_embed_sampling.yaml", "param": "min_blur_score", "current": f"{curr_blur}", "recommended": f"{new_val:.0f}", "reason": f"Face blur score is {blur:.1f}, lower threshold to include blurrier faces"})
                        if abs(pose_yaw) > curr_yaw:
                            new_val = int(abs(pose_yaw) + 10)
                            params.append({"file": "config/pipeline/faces_embed_sampling.yaml", "param": "max_yaw_angle", "current": f"{curr_yaw}", "recommended": f"{new_val}", "reason": f"Face yaw is {pose_yaw:.0f}°, raise limit to include turned faces"})
                        if abs(pose_pitch) > curr_pitch:
                            new_val = int(abs(pose_pitch) + 10)
                            params.append({"file": "config/pipeline/faces_embed_sampling.yaml", "param": "max_pitch_angle", "current": f"{curr_pitch}", "recommended": f"{new_val}", "reason": f"Face pitch is {pose_pitch:.0f}°, raise limit to include tilted faces"})

                        if not params:
                            params.append({"file": "config/pipeline/faces_embed_sampling.yaml", "param": "min_quality_score", "current": f"{curr_quality}", "recommended": f"{max(0.5, curr_quality - 0.5):.1f}", "reason": "General quality threshold reduction"})

                        action_details.append({
                            "issue": "Face failed quality gating (not harvested)",
                            "cause": f"Quality={quality:.2f}, Blur={blur:.1f}, Yaw={pose_yaw:.0f}°, Pitch={pose_pitch:.0f}°",
                            "fix": "Adjust quality thresholds in config to include this face",
                            "face_values": {"quality": quality, "blur": blur, "yaw": pose_yaw, "pitch": pose_pitch},
                            "config_values": {"min_quality_score": curr_quality, "min_blur_score": curr_blur, "max_yaw_angle": curr_yaw, "max_pitch_angle": curr_pitch},
                            "params": params,
                            "command": f"python tools/episode_run.py {ep_id} --stages embed",
                        })

                    elif not is_clustered:
                        curr_thresh = cls_cfg.get("cluster_thresh", 0.52)
                        curr_min_size = cls_cfg.get("min_cluster_size", 1)
                        curr_min_sim = cls_cfg.get("min_identity_sim", 0.45)
                        action_details.append({
                            "issue": "Face not assigned to any cluster",
                            "cause": "Embedding didn't match any existing cluster (outlier, singleton, or new person)",
                            "fix": "Adjust clustering threshold or manually assign",
                            "config_values": {"cluster_thresh": curr_thresh, "min_cluster_size": curr_min_size, "min_identity_sim": curr_min_sim},
                            "params": [
                                {"file": "config/pipeline/clustering.yaml", "param": "cluster_thresh", "current": f"{curr_thresh}", "recommended": f"{max(0.40, curr_thresh - 0.08):.2f}", "reason": "Lower threshold allows looser cluster membership"},
                                {"file": "config/pipeline/clustering.yaml", "param": "min_cluster_size", "current": f"{curr_min_size}", "recommended": "1", "reason": "Allow single-track clusters (singletons)"},
                                {"file": "config/pipeline/clustering.yaml", "param": "min_identity_sim", "current": f"{curr_min_sim}", "recommended": f"{max(0.35, curr_min_sim - 0.1):.2f}", "reason": "Lower threshold to keep more tracks in clusters"},
                            ],
                            "command": f"python tools/episode_run.py {ep_id} --stages cluster --cluster-threshold {max(0.40, curr_thresh - 0.08):.2f}",
                            "where": f"Or manually: **Faces Review** → Find Track {track_id} → Assign to existing identity",
                        })

                    elif not person_id:
                        action_details.append({
                            "issue": "Identity not linked to person",
                            "cause": f"Identity `{identity_id}` exists but has no person_id assigned",
                            "fix": "Link this identity/cluster to a person record",
                            "where": f"**Faces Review** → Find cluster `{identity_id[:12]}...` → Click dropdown → Select or create person",
                            "api": f"PATCH /episodes/{ep_id}/identity/{identity_id} with {{\"person_id\": \"p_XXXX\"}}",
                        })

                    elif not cast_id:
                        action_details.append({
                            "issue": "Person has no cast_id",
                            "cause": f"Person `{person_id}` exists but isn't linked to any cast member",
                            "fix": "Assign this person to a cast member",
                            "where": f"**Cast** page → Find or create cast member → Link to person `{person_id}`",
                            "api": f"PATCH /people/{person_id} with {{\"cast_id\": \"cast_XXXX\"}}",
                        })

                    elif not name:
                        action_details.append({
                            "issue": "Cast member has no name",
                            "cause": f"Cast `{cast_id}` exists but the name field is empty/null",
                            "fix": "Set the display name for this cast member",
                            "where": f"**Cast** page → Find cast `{cast_id[:12]}...` → Edit → Set Name",
                            "api": f"PATCH /cast/{cast_id} with {{\"name\": \"Person Name\"}}",
                        })

                    else:
                        # Fully identified - no action needed
                        st.success("✅ **No action required** - This face is fully identified!")
                        copy_lines.append("")
                        copy_lines.append("=== STATUS ===")
                        copy_lines.append("✅ No action required - This face is fully identified!")

                    # Display action items with parameter table
                    if action_details:
                        copy_lines.append("")
                        copy_lines.append("=== ACTION REQUIRED ===")

                    for detail in action_details:
                        st.error(f"**Issue:** {detail['issue']}")
                        st.markdown(f"**Root Cause:** {detail['cause']}")
                        copy_lines.append(f"Issue: {detail['issue']}")
                        copy_lines.append(f"Root Cause: {detail['cause']}")

                        # Show face values vs config thresholds if available
                        if "face_values" in detail or "config_values" in detail:
                            st.markdown("**This Face's Values vs Config Thresholds:**")
                            copy_lines.append("")
                            copy_lines.append("Face Values vs Config Thresholds:")
                            compare_cols = st.columns(2)
                            with compare_cols[0]:
                                if "face_values" in detail:
                                    st.markdown("**Face Values:**")
                                    copy_lines.append("  Face Values:")
                                    for k, v in detail["face_values"].items():
                                        if isinstance(v, float):
                                            st.write(f"• {k}: `{v:.2f}`")
                                            copy_lines.append(f"    {k}: {v:.2f}")
                                        else:
                                            st.write(f"• {k}: `{v}`")
                                            copy_lines.append(f"    {k}: {v}")
                            with compare_cols[1]:
                                if "config_values" in detail:
                                    st.markdown("**Config Thresholds:**")
                                    copy_lines.append("  Config Thresholds:")
                                    for k, v in detail["config_values"].items():
                                        st.write(f"• {k}: `{v}`")
                                        copy_lines.append(f"    {k}: {v}")

                        st.markdown(f"**How to Fix:** {detail['fix']}")
                        copy_lines.append(f"How to Fix: {detail['fix']}")

                        # Show parameter change table if available
                        if "params" in detail and detail["params"]:
                            st.markdown("**Parameters to Change:**")
                            copy_lines.append("")
                            copy_lines.append("Parameters to Change:")
                            param_rows = []
                            for p in detail["params"]:
                                param_rows.append({
                                    "File": p["file"],
                                    "Parameter": p["param"],
                                    "Current": p["current"],
                                    "→ Recommended": p["recommended"],
                                    "Why": p["reason"],
                                })
                                copy_lines.append(f"  {p['file']}")
                                copy_lines.append(f"    {p['param']}: {p['current']} → {p['recommended']}")
                                copy_lines.append(f"    Reason: {p['reason']}")
                            st.dataframe(param_rows, use_container_width=True, hide_index=True)

                        if "command" in detail:
                            st.markdown("**Run Command:**")
                            st.code(detail["command"], language="bash")
                            copy_lines.append("")
                            copy_lines.append("Run Command:")
                            copy_lines.append(f"  {detail['command']}")

                        if "where" in detail:
                            st.info(f"📍 {detail['where']}")
                            copy_lines.append(f"Where: {detail['where'].replace('**', '')}")

                        if "api" in detail:
                            st.markdown("**API Call:**")
                            st.code(detail["api"], language="bash")
                            copy_lines.append(f"API Call: {detail['api']}")

                    # AI DIAGNOSE button - uses OpenAI for detailed analysis
                    st.markdown("---")
                    ai_btn_col1, ai_btn_col2 = st.columns([1, 3])
                    with ai_btn_col1:
                        if st.button("🔬 AI Diagnose", key=f"{ep_id}::ai_diag_{track_id}", type="secondary"):
                            st.session_state[f"{ep_id}::ai_diag_loading_{track_id}"] = True

                    # Show AI analysis result or loading state
                    if st.session_state.get(f"{ep_id}::ai_diag_loading_{track_id}"):
                        with st.spinner("Analyzing with AI..."):
                            try:
                                api_url = st.session_state.get("api_base") or "http://localhost:8000"
                                resp = requests.post(
                                    f"{api_url}/diagnostics/episodes/{ep_id}/diagnose_track/{track_id}",
                                    json={"use_ai": True, "force_refresh": False},
                                    timeout=60,
                                )
                                if resp.ok:
                                    ai_result = resp.json()
                                    st.session_state[f"{ep_id}::ai_diag_result_{track_id}"] = ai_result
                                    st.session_state[f"{ep_id}::ai_diag_loading_{track_id}"] = False
                                    st.rerun()
                                else:
                                    st.error(f"AI diagnosis failed: {resp.status_code}")
                                    st.session_state[f"{ep_id}::ai_diag_loading_{track_id}"] = False
                            except Exception as e:
                                st.error(f"AI diagnosis error: {e}")
                                st.session_state[f"{ep_id}::ai_diag_loading_{track_id}"] = False

                    # Display AI analysis result
                    ai_result = st.session_state.get(f"{ep_id}::ai_diag_result_{track_id}")
                    if ai_result and ai_result.get("ai_analysis"):
                        ai = ai_result["ai_analysis"]
                        # Avoid nested expanders (Streamlit limitation); this section already lives inside
                        # the per-face expander.
                        st.markdown("#### 🤖 AI Analysis")
                        st.markdown(f"**Explanation:** {ai.get('explanation', 'N/A')}")
                        st.markdown(f"**Root Cause:** `{ai.get('root_cause', 'N/A')}`")
                        st.markdown(f"**Blocked By:** `{ai.get('blocked_by', 'N/A')}`")

                        if ai.get("suggested_fixes"):
                            st.markdown("**Suggested Fixes:**")
                            for fix in ai["suggested_fixes"]:
                                st.markdown(f"- {fix}")

                        if ai.get("config_changes"):
                            st.markdown("**Config Changes:**")
                            for change in ai["config_changes"]:
                                st.code(
                                    f"{change.get('file', 'unknown')}\n"
                                    f"  {change.get('key', '?')}: {change.get('current', '?')} → {change.get('suggested', '?')}\n"
                                    f"  # {change.get('reason', '')}",
                                    language="yaml",
                                )

                        if ai.get("_fallback"):
                            st.caption("ℹ️ Rule-based analysis (OpenAI unavailable)")
                        else:
                            st.caption("ℹ️ Powered by GPT-4o")

                    # COPY DIAGNOSTICS button
                    st.markdown("---")
                    copy_text = "\n".join(copy_lines)
                    if st.button("📋 Copy Diagnostics", key=f"{ep_id}::copy_diag_{track_id}"):
                        st.session_state[f"{ep_id}::show_copy_{track_id}"] = True

                    if st.session_state.get(f"{ep_id}::show_copy_{track_id}"):
                        st.text_area(
                            "Copy this text:",
                            value=copy_text,
                            height=300,
                            key=f"{ep_id}::copy_text_{track_id}",
                        )
                        st.caption("Select all (Cmd+A) and copy (Cmd+C)")

            # Close insights button
            if st.button("Close Insights", key=f"{ep_id}::close_insights"):
                st.session_state.pop(f"{ep_id}::show_insights", None)
                st.rerun()
    else:
        st.info("No faces detected in this frame.")

    # ── Export Video Clip Section ────────────────────────────────────────────────
    st.divider()
    st.subheader("Export Video Clip")
    st.caption("Generate a video clip with face overlays around the current timestamp.")

    clip_col1, clip_col2, clip_col3 = st.columns([1, 1, 1])
    with clip_col1:
        clip_duration = st.selectbox(
            "Clip duration",
            options=[3, 5, 10, 15, 30],
            index=1,
            format_func=lambda x: f"{x} seconds",
            key=f"{ep_id}::clip_duration",
        )
    with clip_col2:
        clip_include_unidentified = st.checkbox(
            "Include unidentified faces",
            value=True,
            key=f"{ep_id}::clip_include_unidentified",
        )
    with clip_col3:
        generate_clip = st.button(
            "Generate Video Clip",
            type="secondary",
            use_container_width=True,
            key=f"{ep_id}::generate_clip",
        )

    if generate_clip:
        # Calculate clip range centered on current timestamp
        center_ts = actual_ts
        half_duration = clip_duration / 2
        clip_start = max(0, center_ts - half_duration)
        clip_end = clip_start + clip_duration

        # Clamp to video duration
        if clip_end > duration_s:
            clip_end = duration_s
            clip_start = max(0, clip_end - clip_duration)

        with st.spinner(f"Generating {clip_duration}s clip ({clip_start:.1f}s - {clip_end:.1f}s)..."):
            try:
                clip_resp = helpers.api_post(
                    f"/episodes/{ep_id}/video_clip",
                    {
                        "start_s": clip_start,
                        "end_s": clip_end,
                        "include_unidentified": clip_include_unidentified,
                    },
                )
                st.session_state[f"{ep_id}::clip_result"] = clip_resp
            except requests.RequestException as exc:
                st.error(helpers.describe_error("Video clip generation", exc))
                st.session_state.pop(f"{ep_id}::clip_result", None)

    # Display clip result if available
    clip_result = st.session_state.get(f"{ep_id}::clip_result")
    if clip_result:
        clip_url = clip_result.get("url")
        clip_start_s = clip_result.get("start_s", 0)
        clip_end_s = clip_result.get("end_s", 0)
        clip_duration_s = clip_result.get("duration_s", 0)
        clip_frames = clip_result.get("frame_count", 0)
        clip_faces = clip_result.get("faces_detected", 0)

        st.success(f"Video clip generated: {clip_duration_s:.1f}s ({clip_frames} frames, {clip_faces} face detections)")

        if clip_url:
            # Display video or download link
            if clip_url.startswith("http"):
                st.video(clip_url)
            else:
                # Local file - provide download link
                clip_path = Path(clip_url)
                try:
                    if clip_path.exists():
                        st.video(str(clip_path))
                        with open(clip_path, "rb") as f:
                            clip_data = f.read()
                        st.download_button(
                            "Download Clip",
                            data=clip_data,
                            file_name=clip_path.name,
                            mime="video/mp4",
                            key=f"{ep_id}::download_clip",
                        )
                    else:
                        st.warning(f"Clip file not found: {clip_url}")
                except (OSError, IOError) as e:
                    st.error(f"Error reading clip file: {e}")

# ── Full Video Export Section ────────────────────────────────────────────────
st.divider()
st.subheader("Export Full Video with Overlays")
st.caption("Generate a video of the entire episode with face bounding boxes and name labels.")

# Check if there's already a successful export - show Interactive Viewer button if so
_existing_export_url = None
try:
    export_jobs_resp = helpers.api_get(f"/jobs?ep_id={ep_id}&job_type=video_export")
    for job in export_jobs_resp.get("jobs", []):
        if job.get("state") == "succeeded":
            # Fetch the full job progress to get the URL (list_jobs doesn't include progress)
            job_id = job.get("job_id")
            if job_id:
                try:
                    progress_resp = helpers.api_get(f"/jobs/{job_id}/progress")
                    job_progress = progress_resp.get("progress", {})
                    _existing_export_url = job_progress.get("url")
                    if _existing_export_url:
                        break
                except requests.RequestException:
                    pass
except requests.RequestException:
    pass

if _existing_export_url:
    st.success("Video with overlays available!")
    iv_col1, iv_col2 = st.columns([1, 1])
    with iv_col1:
        if st.button("Open Interactive Viewer", key=f"{ep_id}::quick_interactive_viewer", type="primary", use_container_width=True):
            st.session_state[f"{ep_id}::interactive_video_url"] = _existing_export_url
            st.switch_page("pages/9_Interactive_Viewer.py")
    with iv_col2:
        st.markdown(
            f'<a href="{_existing_export_url}" target="_blank" style="'
            f'display: inline-block; padding: 8px 16px; width: 100%; text-align: center; '
            f'background-color: #262730; color: white; '
            f'text-decoration: none; border-radius: 5px; border: 1px solid #444;">'
            f'Open Video in New Tab</a>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.caption("**Re-export video:**")

export_col1, export_col2 = st.columns([3, 1])
with export_col1:
    export_include_unidentified = st.checkbox(
        "Include unidentified faces (gray boxes)",
        value=True,
        key=f"{ep_id}::export_include_unidentified",
    )
with export_col2:
    start_export = st.button(
        "Start Export",
        type="primary",
        use_container_width=True,
        key=f"{ep_id}::start_export",
    )

if start_export:
    with st.spinner("Starting video export job..."):
        try:
            resp = helpers.api_post(
                "/jobs/video_export",
                {
                    "ep_id": ep_id,
                    "include_unidentified": export_include_unidentified,
                },
            )
            if resp and resp.get("job_id"):
                st.session_state[f"{ep_id}::current_export_job"] = resp.get("job_id")
                st.success(f"Export job started: {resp['job_id'][:12]}...")
                st.rerun()
            else:
                st.error("Failed to start export job")
        except requests.RequestException as exc:
            st.error(helpers.describe_error("Video export", exc))

# Show export job progress
current_export_job = st.session_state.get(f"{ep_id}::current_export_job")
if current_export_job:
    try:
        job_resp = helpers.api_get(f"/jobs/{current_export_job}/progress")
        job_state = job_resp.get("state")
        progress = job_resp.get("progress", {})

        if job_state == "running":
            phase = progress.get("phase", "init")
            percent = progress.get("percent", 0)
            message = progress.get("message", "Processing...")

            st.info(f"Export job {current_export_job[:12]}... is running")
            st.progress(percent / 100)
            st.caption(f"**{phase.title()}**: {message}")

            # Auto-refresh every 3 seconds
            time.sleep(3)
            st.rerun()

        elif job_state == "succeeded":
            s3_url = progress.get("url")
            s3_key = progress.get("s3_key")
            st.success("Export complete!")

            if s3_url:
                st.markdown("### Video Ready")
                link_col, viewer_col, copy_col = st.columns([2, 2, 1])
                with link_col:
                    st.markdown(
                        f'<a href="{s3_url}" target="_blank" style="'
                        f'display: inline-block; padding: 10px 20px; '
                        f'background-color: #0066cc; color: white; '
                        f'text-decoration: none; border-radius: 5px; '
                        f'font-weight: bold;">'
                        f'Open in New Tab</a>',
                        unsafe_allow_html=True,
                    )
                with viewer_col:
                    if st.button("Interactive Viewer", key=f"{ep_id}::open_interactive_viewer", type="primary", use_container_width=True):
                        # Store video URL in session state for Interactive Viewer
                        st.session_state[f"{ep_id}::interactive_video_url"] = s3_url
                        st.switch_page("pages/9_Interactive_Viewer.py")
                with copy_col:
                    if st.button("Copy Link", key=f"{ep_id}::copy_s3_link"):
                        st.code(s3_url, language=None)
                st.caption(f"S3: `{s3_key}`")

            if st.button("Clear Export Status", key=f"{ep_id}::clear_export"):
                st.session_state.pop(f"{ep_id}::current_export_job", None)
                st.rerun()

        elif job_state == "failed":
            error_msg = progress.get("message", "Unknown error")
            st.error(f"Export failed: {error_msg}")
            if st.button("Clear Export Status", key=f"{ep_id}::clear_export_failed"):
                st.session_state.pop(f"{ep_id}::current_export_job", None)
                st.rerun()

    except requests.RequestException:
        if st.button("Clear Export Status", key=f"{ep_id}::clear_export_error"):
            st.session_state.pop(f"{ep_id}::current_export_job", None)
            st.rerun()

# Recent export jobs
try:
    export_jobs_resp = helpers.api_get(f"/jobs?ep_id={ep_id}&job_type=video_export")
    export_jobs = export_jobs_resp.get("jobs", [])
    if export_jobs:
        with st.expander("Recent Export Jobs", expanded=False):
            for job in export_jobs[:3]:
                job_id = job.get("job_id", "")
                job_id_short = job_id[:8]
                state = job.get("state", "unknown")
                ended_at = job.get("ended_at", "")
                if state == "running":
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"• `{job_id_short}...` - running")
                    with col2:
                        if st.button("Cancel", key=f"{ep_id}::cancel_job_{job_id_short}", type="secondary"):
                            try:
                                helpers.api_post(f"/jobs/{job_id}/cancel", {})
                                st.success("Job cancelled")
                                st.rerun()
                            except requests.RequestException:
                                st.error("Failed to cancel")
                elif ended_at:
                    st.write(f"• `{job_id_short}...` - {state} - {ended_at[:19]}")
                else:
                    st.write(f"• `{job_id_short}...` - {state}")
except requests.RequestException:
    pass

# ── Output Files Section ─────────────────────────────────────────────────────
st.divider()
st.subheader("Output Files")

analytics_dir = helpers.DATA_ROOT / "analytics" / ep_id
json_path = analytics_dir / "screentime.json"
csv_path = analytics_dir / "screentime.csv"

if json_path.exists():
    st.write(f"✅ JSON → {helpers.link_local(json_path)}")
else:
    st.write(f"⚠️ JSON → {json_path} (not yet generated)")

if csv_path.exists():
    st.write(f"✅ CSV → {helpers.link_local(csv_path)}")
else:
    st.write(f"⚠️ CSV → {csv_path} (not yet generated)")
