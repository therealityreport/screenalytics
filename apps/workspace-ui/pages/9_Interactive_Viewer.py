"""Interactive Viewer - Full-screen video player with timestamp capture and annotations."""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from urllib.parse import unquote

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

# ── Page Init ─────────────────────────────────────────────────────────────────
cfg = helpers.init_page("Interactive Viewer")

# ── Episode Selection ─────────────────────────────────────────────────────────
ep_id = helpers.get_ep_id()
if not ep_id:
    st.info("Select an episode from the sidebar.")
    st.stop()

ep_id = ep_id.lower()

# ── Page Title ────────────────────────────────────────────────────────────────
st.title("Interactive Viewer")
st.caption("Full-screen video with timestamp capture, annotations, and export.")

# ── Session State Keys ────────────────────────────────────────────────────────
# Define keys early so helper functions can reference them
_captured_ts_key = f"{ep_id}::captured_timestamps_v2"
_video_url_key = f"{ep_id}::interactive_video_url"
_capture_processed_key = f"{ep_id}::capture_processed"
_capture_log_key = f"{ep_id}::capture_log"

if _captured_ts_key not in st.session_state:
    st.session_state[_captured_ts_key] = []
if _capture_log_key not in st.session_state:
    st.session_state[_capture_log_key] = []


# ── Helper Functions ──────────────────────────────────────────────────────────
def _auto_format_timestamp(raw: str) -> str:
    """Auto-format raw digits into MM:SS.ms format."""
    if ":" in raw:
        return raw
    digits = "".join(c for c in raw if c.isdigit())
    if not digits:
        return "00:00"
    if len(digits) <= 4:
        digits = digits.zfill(4)
        mm = min(int(digits[:2]), 99)
        ss = min(int(digits[2:4]), 59)
        return f"{mm:02d}:{ss:02d}"
    else:
        digits = digits.zfill(6)
        mm = min(int(digits[:2]), 99)
        ss = min(int(digits[2:4]), 59)
        ms = digits[4:]
        return f"{mm:02d}:{ss:02d}.{ms}"


def _parse_timestamp_input(ts_str: str) -> float | None:
    """Parse MM:SS or MM:SS.ms format to seconds."""
    ts_str = ts_str.strip()
    if not ts_str:
        return None
    match = re.match(r"^(\d+):(\d{1,2})(?:\.(\d+))?$", ts_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        ms_str = match.group(3)
        ms = float(f"0.{ms_str}") if ms_str else 0.0
        return minutes * 60 + seconds + ms
    try:
        return float(ts_str)
    except ValueError:
        return None


def build_full_face_table(faces: list) -> list:
    """Build comprehensive face data table from captured faces."""
    rows = []
    for f in faces:
        bbox = f.get("bbox", [])
        scores = f.get("scores", {})
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
            "Conf": f"{f.get('conf', 0):.0%}" if f.get("conf") else "-",
            "Size": size_str,
            "Quality": f"{scores.get('quality', 0):.2f}" if scores.get("quality") else "-",
            "Blur": f"{scores.get('blur', 0):.1f}" if scores.get("blur") else "-",
            "Yaw": f"{scores.get('pose_yaw', 0):.0f}" if scores.get("pose_yaw") is not None else "-",
            "Det": "Y" if f.get("detected", True) else "N",
            "Trk": "Y" if f.get("tracked") else "N",
            "Hrv": "Y" if f.get("harvested") else "N",
            "Cls": "Y" if f.get("clustered") else "N",
            "Issue": f.get("unidentified_reason") or "-",
        })
    return rows


def build_csv_export(ep_id: str, captured_timestamps: list) -> str:
    """Build CSV export from captured timestamps."""
    output = StringIO()
    writer = csv.writer(output)
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
            path = obj.get("path", [])
            points = []
            for cmd in path:
                if len(cmd) >= 3 and cmd[0] in ["M", "L", "Q"]:
                    points.append((cmd[1], cmd[2]))
            if len(points) >= 2:
                draw.line(points, fill=obj.get("stroke", "red"), width=int(obj.get("strokeWidth", 3)))
        elif obj_type == "rect":
            left = obj.get("left", 0)
            top = obj.get("top", 0)
            width = obj.get("width", 0)
            height = obj.get("height", 0)
            draw.rectangle([left, top, left + width, top + height],
                          outline=obj.get("stroke", "red"),
                          width=int(obj.get("strokeWidth", 3)))
        elif obj_type == "circle":
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

    story.append(Paragraph(f"Timestamp Analysis Report: {ep_id}", styles['Title']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    for entry in captured_timestamps:
        ts = entry["timestamp_s"]
        ts_str = f"{int(ts//60):02d}:{ts%60:05.2f}"

        story.append(Paragraph(f"Timestamp: {ts_str} (Frame {entry.get('frame_idx', '?')})", styles['Heading2']))

        if entry.get("notes"):
            story.append(Paragraph(f"<b>Notes:</b> {entry['notes']}", styles['Normal']))

        summary = entry.get("pipeline_summary", {})
        story.append(Paragraph(
            f"<b>Pipeline:</b> Detected: {summary.get('detected', 0)} | "
            f"Tracked: {summary.get('tracked', 0)} | "
            f"Harvested: {summary.get('harvested', 0)} | "
            f"Clustered: {summary.get('clustered', 0)}",
            styles['Normal']
        ))

        screenshot_url = entry.get("screenshot_url")
        if screenshot_url:
            try:
                if screenshot_url.startswith("http"):
                    response = requests.get(screenshot_url, timeout=30)
                    img = Image.open(BytesIO(response.content))
                else:
                    img = Image.open(screenshot_url)

                if entry.get("annotation_data"):
                    img = apply_canvas_annotations(img, entry["annotation_data"])

                img_buffer = BytesIO()
                img.save(img_buffer, format="JPEG")
                img_buffer.seek(0)

                aspect = img.height / img.width
                img_width = 6 * inch
                img_height = img_width * aspect

                rl_img = RLImage(img_buffer, width=img_width, height=img_height)
                story.append(rl_img)
            except Exception as e:
                story.append(Paragraph(f"[Image not available: {e}]", styles['Normal']))

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


def _add_captured_timestamp(ep_id: str, ts_seconds: float, fps_hint: float | None = None) -> bool:
    """Fetch preview data and append a captured timestamp entry if not already present."""
    captured = st.session_state[_captured_ts_key]
    existing_ts = [e["timestamp_s"] for e in captured]

    # Only add if not already in the list (within 0.1s tolerance)
    is_duplicate = any(abs(ts_seconds - t) < 0.1 for t in existing_ts)
    if is_duplicate:
        _log_capture_event(f"Skipped duplicate capture at {ts_seconds:.3f}s")
        return False

    preview_resp = helpers.api_get(
        f"/episodes/{ep_id}/timestamp/{ts_seconds}/preview",
        timeout=90,  # Large episodes need more time for manifest loading
    )
    captured_entry = {
        "timestamp_s": ts_seconds,
        "frame_idx": preview_resp.get("frame_idx"),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "notes": "",
        "annotation_data": None,
        "screenshot_url": preview_resp.get("url"),
        "faces": preview_resp.get("faces", []),
        "pipeline_summary": preview_resp.get("pipeline_summary", {}),
        "fps": preview_resp.get("fps", fps_hint or 24),
    }
    captured.append(captured_entry)
    captured.sort(key=lambda x: x["timestamp_s"])
    st.session_state[_captured_ts_key] = captured
    st.session_state[_capture_processed_key] = f"{ts_seconds:.3f}"
    ts_str = f"{int(ts_seconds//60):02d}:{ts_seconds%60:05.2f}"
    st.toast(f"Added timestamp {ts_str} with {len(captured_entry['faces'])} faces")
    _log_capture_event(f"Captured at {ts_str} (frame {captured_entry.get('frame_idx', '?')})")
    return True


def render_annotation_canvas(entry: dict, idx: int, ep_id: str):
    """Render drawable canvas for annotation."""
    if not HAS_DRAWABLE_CANVAS:
        st.caption("*Drawing annotations: Install `streamlit-drawable-canvas` to enable*")
        return

    st.markdown("**Draw Annotations:**")
    screenshot_url = entry.get("screenshot_url")
    if not screenshot_url:
        st.warning("No screenshot available")
        return

    try:
        if screenshot_url.startswith("http"):
            response = requests.get(screenshot_url, timeout=30)
            bg_image = Image.open(BytesIO(response.content))
        else:
            bg_image = Image.open(screenshot_url)
    except Exception as e:
        st.warning(f"Could not load image: {e}")
        return

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
        if canvas_result and canvas_result.json_data:
            entry["annotation_data"] = canvas_result.json_data
    except (AttributeError, Exception):
        st.caption("*Drawing canvas unavailable (Streamlit version incompatibility)*")


def _log_capture_event(message: str) -> None:
    """Append a human-readable capture log entry."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log = st.session_state[_capture_log_key]
    log.append(f"[{now}] {message}")
    st.session_state[_capture_log_key] = log[-50:]

# ── Handle Auto-Capture from Video Player ─────────────────────────────────────
# Check for capture_ts query parameter (set by video player's Capture button)
_capture_ts_param = st.query_params.get("capture_ts")
if _capture_ts_param:
    # Clear query params FIRST to prevent retriggering
    st.query_params.clear()

    # Check if we already processed this exact capture (guard against rapid reruns)
    processed_ts = st.session_state.get(_capture_processed_key)
    if processed_ts != _capture_ts_param:
        try:
            ts_seconds = float(_capture_ts_param)
            try:
                _log_capture_event(f"Received capture_ts={ts_seconds:.3f}s from query params")
                added = _add_captured_timestamp(ep_id, ts_seconds, fps_hint=24)
                if not added:
                    _log_capture_event(f"Skipped duplicate capture at {ts_seconds:.3f}s")
            except requests.RequestException as exc:
                error_msg = helpers.describe_error(f"/episodes/{ep_id}/timestamp/{ts_seconds}/preview", exc)
                st.toast(error_msg, icon="⚠️")
                _log_capture_event(f"Error: {error_msg}")
        except (ValueError, TypeError):
            pass

    # Explicit rerun to ensure state is committed and UI refreshes
    st.rerun()


# ── Get Video URL ─────────────────────────────────────────────────────────────
# Check query params first (passed from Timestamp Search)
_overlay_video_url = None
_overlay_video_fps = 24.0
_video_duration = 0

# Check session state (set by Timestamp Search page)
if st.session_state.get(_video_url_key):
    _overlay_video_url = st.session_state[_video_url_key]

# If not in session state, check for recent export jobs
if not _overlay_video_url:
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
                        _overlay_video_url = job_progress.get("url")
                        if not _overlay_video_url:
                            _overlay_video_url = job_progress.get("output_path")
                        if _overlay_video_url:
                            break
                    except requests.RequestException:
                        pass
    except requests.RequestException:
        pass

# Fallback: look for a locally rendered overlay in analytics directory
if not _overlay_video_url:
    local_overlay = helpers.DATA_ROOT / "analytics" / ep_id / f"{ep_id}_overlay.mp4"
    if local_overlay.exists():
        _overlay_video_url = str(local_overlay)

# Get video metadata
try:
    video_meta = helpers.api_get(f"/episodes/{ep_id}/video_meta", timeout=10)
    _overlay_video_fps = video_meta.get("fps_detected", 24) or 24
    _video_duration = video_meta.get("duration_sec", 0) or 0
except requests.RequestException:
    pass

_safe_overlay_video_url = helpers.ensure_media_url(_overlay_video_url) if _overlay_video_url else None

# ── Main Content ──────────────────────────────────────────────────────────────
if not _safe_overlay_video_url:
    st.warning("No exported video with overlays found for this episode.")
    st.info("Go to **Timestamp Search** and export a video with overlays first, then return here.")

    # Link back to Timestamp Search
    if st.button("Go to Timestamp Search", type="primary"):
        st.switch_page("pages/7_Timestamp_Search.py")
    st.stop()

# ── Video Player Section ──────────────────────────────────────────────────────
st.markdown("## Video Player")

# Show video URL status for debugging
with st.expander("Video Info", expanded=False):
    st.code(_overlay_video_url or "No URL", language=None)
    st.caption(f"FPS: {_overlay_video_fps} | Duration: {_video_duration}s")

# Custom HTML5 video player with built-in timestamp capture
# Use the actual FPS for frame-based seeking
_frame_duration = 1.0 / _overlay_video_fps if _overlay_video_fps > 0 else 1.0 / 24.0

video_player_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    html, body {{
        margin: 0;
        padding: 0;
        overflow: visible;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #0e1117;
    }}
    .player-container {{
        width: 100%;
        display: flex;
        flex-direction: column;
    }}
    .video-wrapper {{
        width: 100%;
        background: #000;
        border-radius: 8px 8px 0 0;
        overflow: hidden;
        line-height: 0;
        flex-shrink: 0;
    }}
    video {{
        width: 100%;
        display: block;
        aspect-ratio: 16/9;
        max-height: 540px;
        object-fit: contain;
        background: #000;
    }}
    .controls {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        padding: 10px 12px;
        background: #1e1e2e;
        border-radius: 0 0 8px 8px;
    }}
    .play-pause {{
        background: #4CAF50;
        color: white;
        border: none;
        padding: 6px 14px;
        border-radius: 4px;
        cursor: pointer;
        font-weight: bold;
        font-size: 13px;
        min-width: 60px;
    }}
    .play-pause:hover {{ background: #45a049; }}
    .play-pause.playing {{ background: #ff9800; }}
    .play-pause.playing:hover {{ background: #f57c00; }}
    .time-display {{
        font-family: 'Courier New', monospace;
        font-size: 14px;
        color: #fff;
        background: #333;
        padding: 5px 10px;
        border-radius: 4px;
        min-width: 100px;
        text-align: center;
    }}
    .capture-btn {{
        background: linear-gradient(135deg, #ff4b4b, #ff6b6b);
        color: white;
        border: none;
        padding: 6px 14px;
        border-radius: 4px;
        cursor: pointer;
        font-weight: bold;
        font-size: 13px;
    }}
    .capture-btn:hover {{ background: linear-gradient(135deg, #ff3333, #ff5555); }}
    .captured-display {{
        font-family: 'Courier New', monospace;
        font-size: 13px;
        color: #4CAF50;
        background: #1a3d1a;
        padding: 5px 10px;
        border-radius: 4px;
        min-width: 100px;
        text-align: center;
    }}
    .seek-btns {{
        display: flex;
        gap: 3px;
    }}
    .seek-btn {{
        background: #333;
        color: white;
        border: 1px solid #555;
        padding: 4px 8px;
        border-radius: 3px;
        cursor: pointer;
        font-size: 11px;
    }}
    .seek-btn:hover {{ background: #444; }}
    .progress-container {{
        flex: 1;
        min-width: 80px;
        height: 6px;
        background: #333;
        border-radius: 3px;
        cursor: pointer;
    }}
    .progress-bar {{
        height: 100%;
        background: #ff4b4b;
        border-radius: 3px;
        width: 0%;
    }}
    .duration {{
        font-family: monospace;
        font-size: 12px;
        color: #888;
    }}
    .divider {{
        width: 1px;
        height: 24px;
        background: #444;
        margin: 0 4px;
    }}
    .volume-container {{
        display: flex;
        align-items: center;
        gap: 4px;
    }}
    .volume-icon {{
        font-size: 14px;
        cursor: pointer;
        user-select: none;
    }}
    .volume-slider {{
        width: 60px;
        height: 4px;
        -webkit-appearance: none;
        background: #444;
        border-radius: 2px;
        cursor: pointer;
    }}
    .volume-slider::-webkit-slider-thumb {{
        -webkit-appearance: none;
        width: 12px;
        height: 12px;
        background: #fff;
        border-radius: 50%;
        cursor: pointer;
    }}
    .status-msg {{
        font-size: 11px;
        color: #4CAF50;
    }}
</style>
</head>
<body>
<div class="player-container">
    <div class="video-wrapper">
        <video id="mainVideo" preload="metadata">
            <source src="{_safe_overlay_video_url}" type="video/mp4">
            Your browser does not support video playback.
        </video>
    </div>
    <div class="controls">
        <button class="play-pause" id="playPauseBtn" onclick="togglePlay()">Play</button>
        <div class="seek-btns">
            <button class="seek-btn" onclick="seekFrames(-5)">-5f</button>
            <button class="seek-btn" onclick="seekFrames(-1)">-1f</button>
            <button class="seek-btn" onclick="seekFrames(1)">+1f</button>
            <button class="seek-btn" onclick="seekFrames(5)">+5f</button>
        </div>
        <div class="time-display" id="currentTime">00:00.000</div>
        <div class="progress-container" id="progressContainer" onclick="seekToPosition(event)">
            <div class="progress-bar" id="progressBar"></div>
        </div>
        <span class="duration" id="duration">/ --:--</span>
        <div class="volume-container">
            <span class="volume-icon" id="volumeIcon" onclick="toggleMute()">🔊</span>
            <input type="range" class="volume-slider" id="volumeSlider" min="0" max="1" step="0.1" value="1" oninput="setVolume(this.value)">
        </div>
        <div class="divider"></div>
        <button class="capture-btn" onclick="captureAndAdd()">Capture Timestamp</button>
        <div class="captured-display" id="capturedTime">--:--</div>
        <span class="status-msg" id="statusMsg"></span>
    </div>
</div>
<script>
    const video = document.getElementById('mainVideo');
    const playPauseBtn = document.getElementById('playPauseBtn');
    const currentTimeEl = document.getElementById('currentTime');
    const capturedTimeEl = document.getElementById('capturedTime');
    const progressBar = document.getElementById('progressBar');
    const durationEl = document.getElementById('duration');
    const statusMsg = document.getElementById('statusMsg');
    const volumeIcon = document.getElementById('volumeIcon');
    const volumeSlider = document.getElementById('volumeSlider');
    const FRAME_DURATION = {_frame_duration};
    let lastVolume = 1;

    function formatTime(seconds) {{
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return mins.toString().padStart(2, '0') + ':' + secs.toFixed(3).padStart(6, '0');
    }}

    function formatDuration(seconds) {{
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return mins.toString().padStart(2, '0') + ':' + secs.toString().padStart(2, '0');
    }}

    function togglePlay() {{
        if (video.paused) {{
            video.play();
        }} else {{
            video.pause();
        }}
    }}

    function seekFrames(numFrames) {{
        video.currentTime = Math.max(0, Math.min(video.duration, video.currentTime + (numFrames * FRAME_DURATION)));
    }}

    function seekToPosition(event) {{
        const container = document.getElementById('progressContainer');
        const rect = container.getBoundingClientRect();
        const percent = (event.clientX - rect.left) / rect.width;
        video.currentTime = percent * video.duration;
    }}

    function setVolume(val) {{
        video.volume = val;
        lastVolume = val > 0 ? val : lastVolume;
        updateVolumeIcon();
    }}

    function toggleMute() {{
        if (video.volume > 0) {{
            lastVolume = video.volume;
            video.volume = 0;
            volumeSlider.value = 0;
        }} else {{
            video.volume = lastVolume || 1;
            volumeSlider.value = video.volume;
        }}
        updateVolumeIcon();
    }}

    function updateVolumeIcon() {{
        if (video.volume === 0) {{
            volumeIcon.textContent = '🔇';
        }} else if (video.volume < 0.5) {{
            volumeIcon.textContent = '🔉';
        }} else {{
            volumeIcon.textContent = '🔊';
        }}
    }}

    function captureAndAdd() {{
        const ts = video.currentTime;
        const formatted = formatTime(ts);
        capturedTimeEl.textContent = formatted;

        // Copy to clipboard
        navigator.clipboard.writeText(formatted).catch(() => {{}});

        // Store in localStorage for Streamlit to read via sync button
        const key = 'iv_pending_capture_{ep_id}';
        localStorage.setItem(key, ts.toFixed(3));
        localStorage.setItem(key + '_time', Date.now().toString());

        // Show success status
        statusMsg.textContent = 'Captured! Click Sync below.';
        statusMsg.style.color = '#4CAF50';

        // Try to navigate parent (may fail due to iframe security)
        try {{
            const url = new URL(window.parent.location.href);
            url.searchParams.set('capture_ts', ts.toFixed(3));
            window.parent.location.href = url.toString();
        }} catch (e) {{
            // Navigation blocked - user will need to use Sync button
            console.log('Parent navigation blocked, use Sync button');
        }}
    }}

    // Tell Streamlit our iframe height so the component renders correctly
    function reportHeight() {{
        const height = document.documentElement.scrollHeight;
        window.parent.postMessage({{
            isStreamlitMessage: true,
            type: 'streamlit:componentReady',
            height: height,
        }}, '*');
    }}
    window.addEventListener('load', reportHeight);
    window.addEventListener('resize', reportHeight);

    video.addEventListener('timeupdate', () => {{
        currentTimeEl.textContent = formatTime(video.currentTime);
        if (video.duration) {{
            progressBar.style.width = (video.currentTime / video.duration * 100) + '%';
        }}
    }});

    video.addEventListener('loadedmetadata', () => {{
        durationEl.textContent = '/ ' + formatDuration(video.duration);
    }});

    video.addEventListener('play', () => {{
        playPauseBtn.textContent = 'Pause';
        playPauseBtn.classList.add('playing');
    }});

    video.addEventListener('pause', () => {{
        playPauseBtn.textContent = 'Play';
        playPauseBtn.classList.remove('playing');
    }});

    video.addEventListener('ended', () => {{
        playPauseBtn.textContent = 'Play';
        playPauseBtn.classList.remove('playing');
    }});

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {{
        if (e.code === 'Space') {{
            e.preventDefault();
            togglePlay();
        }} else if (e.code === 'ArrowLeft') {{
            seekFrames(-1);
        }} else if (e.code === 'ArrowRight') {{
            seekFrames(1);
        }} else if (e.code === 'KeyC') {{
            captureAndAdd();
        }} else if (e.code === 'KeyM') {{
            toggleMute();
        }}
    }});
</script>
</body>
</html>
"""

# Provide enough height for 16:9 video (~540px) + controls (~50px) + padding
# Note: components.html() doesn't support bi-directional communication.
# Capture is handled via query parameters set by the JavaScript captureAndAdd() function.
components.html(
    video_player_html,
    height=620,
    scrolling=False,
)

# ── Keyboard Shortcuts ────────────────────────────────────────────────────────
st.caption("**Shortcuts:** Space = Play/Pause | ← → = ±1 frame | C = Capture Timestamp | M = Mute")

# ── Sync Captures from Video Player ──────────────────────────────────────────
# This provides a fallback when iframe navigation is blocked by browser security
_sync_key = f"{ep_id}::sync_capture_trigger"
_pending_capture_key = f"iv_pending_capture_{ep_id}"

sync_col1, sync_col2 = st.columns([1, 4])
with sync_col1:
    if st.button("Sync Capture", key=f"{ep_id}::iv_sync_btn", type="primary", help="Click after using Capture button in video player"):
        st.session_state[_sync_key] = True
        st.rerun()

# JavaScript to read localStorage and pass to Streamlit via query params
if st.session_state.get(_sync_key):
    st.session_state[_sync_key] = False
    # Inject JavaScript to read localStorage and redirect with the value
    sync_js = f"""
    <script>
        const key = '{_pending_capture_key}';
        const val = localStorage.getItem(key);
        const timeKey = key + '_time';
        const captureTime = localStorage.getItem(timeKey);

        if (val && captureTime) {{
            // Only process if capture is recent (within 60 seconds)
            const age = Date.now() - parseInt(captureTime);
            if (age < 60000) {{
                // Clear the pending capture
                localStorage.removeItem(key);
                localStorage.removeItem(timeKey);

                // Redirect with the captured timestamp
                const url = new URL(window.location.href);
                url.searchParams.set('capture_ts', val);
                window.location.href = url.toString();
            }} else {{
                // Stale capture, clear it
                localStorage.removeItem(key);
                localStorage.removeItem(timeKey);
            }}
        }}
    </script>
    """
    components.html(sync_js, height=0)

with sync_col2:
    st.caption("If the Capture button doesn't auto-add, click **Sync Capture** to add pending timestamps.")

# ── Capture Run Log ───────────────────────────────────────────────────────────
with st.expander("Capture Run Log", expanded=False):
    log_entries = list(reversed(st.session_state.get(_capture_log_key, [])))
    if log_entries:
        for entry in log_entries:
            st.markdown(f"- {entry}")
    else:
        st.caption("No capture events yet.")

# ── Manual Timestamp Input ────────────────────────────────────────────────────
with st.expander("Manual Timestamp Entry", expanded=False):
    st.caption("Use this if you prefer to manually enter a timestamp instead of using the Capture button.")

    input_col, btn_col = st.columns([3, 1])
    with input_col:
        capture_input = st.text_input(
            "Timestamp (MM:SS.mmm)",
            value="",
            key=f"{ep_id}::iv_capture_input",
            placeholder="e.g., 01:30.500",
            label_visibility="collapsed",
        )
    with btn_col:
        add_clicked = st.button("Add", key=f"{ep_id}::iv_add_capture", use_container_width=True, type="primary")

    if add_clicked and capture_input:
        ts_seconds = _parse_timestamp_input(_auto_format_timestamp(capture_input))
        if ts_seconds is not None and ts_seconds >= 0:
            captured = st.session_state[_captured_ts_key]
            existing_ts = [e["timestamp_s"] for e in captured]
            # Use tolerance-based duplicate check (consistent with auto-capture)
            is_duplicate = any(abs(ts_seconds - t) < 0.1 for t in existing_ts)
            if not is_duplicate:
                with st.spinner("Capturing frame data..."):
                    try:
                        preview_resp = helpers.api_get(
                            f"/episodes/{ep_id}/timestamp/{ts_seconds}/preview",
                            timeout=30,
                        )
                        captured_entry = {
                            "timestamp_s": ts_seconds,
                            "frame_idx": preview_resp.get("frame_idx"),
                            "captured_at": datetime.now(timezone.utc).isoformat(),
                            "notes": "",
                            "annotation_data": None,
                            "screenshot_url": preview_resp.get("url"),
                            "faces": preview_resp.get("faces", []),
                            "pipeline_summary": preview_resp.get("pipeline_summary", {}),
                            "fps": preview_resp.get("fps", _overlay_video_fps),
                        }
                        captured.append(captured_entry)
                        captured.sort(key=lambda x: x["timestamp_s"])
                        st.session_state[_captured_ts_key] = captured
                        st.success(f"Added {capture_input} with {len(captured_entry['faces'])} faces")
                        st.rerun()
                    except requests.RequestException as exc:
                        st.error(helpers.describe_error(f"/episodes/{ep_id}/timestamp/{ts_seconds}/preview", exc))
            else:
                st.warning("Timestamp already in list (within 0.1s tolerance)")
        else:
            st.error("Invalid timestamp format")

# ── Captured Timestamps Section ───────────────────────────────────────────────
captured_timestamps = st.session_state.get(_captured_ts_key, [])

st.markdown("---")
st.markdown(f"## Captured Timestamps ({len(captured_timestamps)})")

if captured_timestamps:
    # Clear all button
    if st.button("Clear All", key=f"{ep_id}::iv_clear_all"):
        st.session_state[_captured_ts_key] = []
        st.session_state.pop(f"{ep_id}::pdf_report", None)
        st.rerun()

    # Display each captured timestamp
    for i, entry in enumerate(captured_timestamps):
        ts = entry["timestamp_s"]
        ts_mm = int(ts // 60)
        ts_ss = ts % 60
        ts_formatted = f"{ts_mm:02d}:{ts_ss:06.3f}"
        face_count = len(entry.get("faces", []))

        with st.expander(f"{ts_formatted} - Frame {entry.get('frame_idx', '?')} ({face_count} faces)", expanded=False):
            # Image and notes side by side
            col_img, col_notes = st.columns([1, 1])

            with col_img:
                screenshot_url = entry.get("screenshot_url")
                if screenshot_url:
                    st.image(screenshot_url, use_container_width=True)
                else:
                    st.info("No screenshot available")

            with col_notes:
                # Notes field
                notes = st.text_area(
                    "Notes / Questions",
                    value=entry.get("notes", ""),
                    key=f"{ep_id}::iv_notes_{i}",
                    height=150,
                    placeholder="Why did you save this timestamp? What issues do you see?",
                )
                if notes != entry.get("notes", ""):
                    entry["notes"] = notes

                # Pipeline summary
                summary = entry.get("pipeline_summary", {})
                if summary:
                    st.markdown(
                        f"**Pipeline:** Det: {summary.get('detected', 0)} | "
                        f"Trk: {summary.get('tracked', 0)} | "
                        f"Hrv: {summary.get('harvested', 0)} | "
                        f"Cls: {summary.get('clustered', 0)}"
                    )

            # Face data table
            faces = entry.get("faces", [])
            if faces:
                st.markdown("**Faces in Frame:**")
                face_table = build_full_face_table(faces)
                st.dataframe(face_table, use_container_width=True, hide_index=True)
            else:
                st.info("No faces detected at this timestamp")

            # Drawing annotations
            if HAS_DRAWABLE_CANVAS:
                with st.expander("Draw Annotations", expanded=False):
                    render_annotation_canvas(entry, i, ep_id)
            else:
                st.caption("*Drawing: Install `streamlit-drawable-canvas` to enable*")

            # Action buttons
            action_col1, action_col2, action_col3 = st.columns(3)
            with action_col1:
                if st.button("Copy Timestamp", key=f"{ep_id}::iv_copy_{i}"):
                    st.code(ts_formatted, language=None)
            with action_col2:
                if st.button("View in Timestamp Search", key=f"{ep_id}::iv_view_{i}"):
                    # Set the preview result in session state for Timestamp Search
                    st.session_state[f"{ep_id}::timestamp_preview_result"] = {
                        "url": entry.get("screenshot_url"),
                        "faces": entry.get("faces", []),
                        "pipeline_summary": entry.get("pipeline_summary", {}),
                        "frame_idx": entry.get("frame_idx"),
                        "timestamp_s": entry["timestamp_s"],
                        "fps": entry.get("fps", _overlay_video_fps),
                    }
                    st.switch_page("pages/7_Timestamp_Search.py")
            with action_col3:
                if st.button("Remove", key=f"{ep_id}::iv_remove_{i}", type="secondary"):
                    captured_timestamps.pop(i)
                    st.rerun()

    # Export section
    st.markdown("---")
    st.markdown("## Export")

    export_col1, export_col2, export_col3 = st.columns(3)

    with export_col1:
        json_data = json.dumps(captured_timestamps, indent=2, default=str)
        st.download_button(
            "Download JSON",
            data=json_data,
            file_name=f"{ep_id}_timestamp_report.json",
            mime="application/json",
            key=f"{ep_id}::iv_export_json",
            use_container_width=True,
        )

    with export_col2:
        csv_data = build_csv_export(ep_id, captured_timestamps)
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name=f"{ep_id}_timestamp_report.csv",
            mime="text/csv",
            key=f"{ep_id}::iv_export_csv",
            use_container_width=True,
        )

    with export_col3:
        if HAS_REPORTLAB:
            if st.button("Generate PDF", key=f"{ep_id}::iv_export_pdf", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    pdf_bytes = generate_pdf_report(ep_id, captured_timestamps)
                    st.session_state[f"{ep_id}::pdf_report"] = pdf_bytes

            if st.session_state.get(f"{ep_id}::pdf_report"):
                st.download_button(
                    "Download PDF",
                    data=st.session_state[f"{ep_id}::pdf_report"],
                    file_name=f"{ep_id}_timestamp_report.pdf",
                    mime="application/pdf",
                    key=f"{ep_id}::iv_download_pdf",
                    use_container_width=True,
                )
        else:
            st.caption("*PDF: Install `reportlab` to enable*")

else:
    st.info("No timestamps captured yet. Pause the video and click 'Capture Timestamp' to begin.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"Episode: {ep_id} | FPS: {_overlay_video_fps:.2f} | Duration: {_video_duration:.1f}s")
