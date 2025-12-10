"""Timestamp Search - Visual debugging tool to preview frames with face detections."""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402

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


# ── Helper Functions ──────────────────────────────────────────────────────────
def _auto_format_timestamp(raw: str) -> str:
    """Auto-format raw digits into MM:SS.ms format.

    Examples:
        "012771" -> "01:27.71"
        "0127" -> "01:27"
        "130" -> "01:30"
        "01:27.71" -> "01:27.71" (already formatted)
    """
    # If already has colon, return as-is (already formatted)
    if ":" in raw:
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
        mm = digits[:2]
        ss = digits[2:4]
        return f"{mm}:{ss}"
    else:
        # MMSSFF format (6+ digits -> MM:SS.FF)
        digits = digits.zfill(6)
        mm = digits[:2]
        ss = digits[2:4]
        ms = digits[4:]
        return f"{mm}:{ss}.{ms}"


def _parse_timestamp_input(ts_str: str) -> float | None:
    """Parse MM:SS or MM:SS.ms format to seconds."""
    import re

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


# ── API Call ──────────────────────────────────────────────────────────────────
if preview_clicked:
    timestamp_s = _parse_timestamp_input(ts_input)
    if timestamp_s is None:
        st.error("Invalid timestamp format. Use MM:SS or MM:SS.ms (e.g., 01:30 or 01:30.50)")
    else:
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

    nav_col1, nav_col2, nav_col3 = st.columns([1, 4, 1])
    with nav_col1:
        if st.button("◀ Prev", key="prev_frame", use_container_width=True, disabled=frame_idx <= 0):
            _load_frame_by_index(frame_idx - 1)
            st.rerun()
    with nav_col2:
        st.markdown(
            f"<div style='text-align:center;padding:8px 0;'>"
            f"<strong>Frame {frame_idx}</strong> @ {actual_mm}:{actual_ss:05.2f} ({fps:.2f} fps)"
            f"</div>",
            unsafe_allow_html=True,
        )
    with nav_col3:
        if st.button("Next ▶", key="next_frame", use_container_width=True):
            _load_frame_by_index(frame_idx + 1)
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

        # Show drop-off warnings
        if sum_detected > 0:
            if sum_tracked < sum_detected:
                st.warning(f"⚠️ {sum_detected - sum_tracked} face(s) detected but NOT tracked (below track confidence threshold)")
            if sum_harvested < sum_tracked:
                st.info(f"ℹ️ {sum_tracked - sum_harvested} tracked face(s) NOT harvested (didn't pass quality gate or not sampled)")
            if sum_clustered < sum_harvested:
                st.info(f"ℹ️ {sum_harvested - sum_clustered} harvested face(s) NOT clustered yet")

    # Display face info table
    faces = preview_result.get("faces", [])
    if faces:
        st.markdown(f"**{len(faces)} face(s) in frame:**")

        # Status icon helper (defined once outside loop)
        def _status_icon(val: bool) -> str:
            return "✓" if val else "✗"

        face_rows = []
        face_data_for_links = []  # Store track/cluster info for link buttons
        for face in faces:
            track_id = face.get("track_id")
            identity_id = face.get("identity_id") or "—"
            name = face.get("name")
            bbox = face.get("bbox", [])
            conf = face.get("conf")

            # Pipeline status flags
            is_detected = face.get("detected", False)
            is_tracked = face.get("tracked", False)
            is_harvested = face.get("harvested", False)
            is_clustered = face.get("clustered", False)

            # Format bbox as readable string
            bbox_str = f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]" if bbox else "—"

            # Calculate face size from bbox [x1, y1, x2, y2]
            if bbox and len(bbox) >= 4:
                face_width = int(bbox[2] - bbox[0])
                face_height = int(bbox[3] - bbox[1])
                total_pixels = face_width * face_height
                size_str = f"{face_width}x{face_height} ({total_pixels:,}px)"
            else:
                size_str = "—"

            # Format confidence as percentage
            conf_str = f"{conf * 100:.0f}%" if conf is not None else "—"

            face_rows.append({
                "Track": f"T{track_id}" if track_id else "—",
                "Conf": conf_str,
                "Size": size_str,
                "Det": _status_icon(is_detected),
                "Trk": _status_icon(is_tracked),
                "Harv": _status_icon(is_harvested),
                "Clust": _status_icon(is_clustered),
                "Identity": identity_id[:16] + "…" if identity_id and len(str(identity_id)) > 16 else identity_id,
                "Name": name or "—",
                "BBox": bbox_str,
            })

            # Store for link buttons
            face_data_for_links.append({
                "track_id": track_id,
                "identity_id": identity_id if identity_id != "—" else None,
            })

        # Build copyable text table
        header = "Track\tConf\tSize\tDet\tTrk\tHarv\tClust\tIdentity\tName\tBBox"
        rows_text = [header]
        for row in face_rows:
            rows_text.append(
                f"{row['Track']}\t{row['Conf']}\t{row['Size']}\t{row['Det']}\t{row['Trk']}\t"
                f"{row['Harv']}\t{row['Clust']}\t{row['Identity']}\t{row['Name']}\t{row['BBox']}"
            )
        copyable_table = "\n".join(rows_text)

        # Show copyable text in expander
        with st.expander("📋 Copy Table", expanded=False):
            st.code(copyable_table, language=None)

        # Display dataframe
        import pandas as pd
        df = pd.DataFrame(face_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Legend for status columns
        st.caption("Pipeline status: Conf=Detection confidence, Det=Detected, Trk=Tracked, Harv=Harvested (quality gated + embedded), Clust=Clustered")

        # ── View Track/Cluster Links ─────────────────────────────────────────
        # Collect unique tracks and clusters
        unique_tracks = []
        unique_clusters = []
        seen_tracks = set()
        seen_clusters = set()

        for face_info in face_data_for_links:
            track_id = face_info["track_id"]
            identity_id = face_info["identity_id"]

            if track_id and track_id not in seen_tracks:
                seen_tracks.add(track_id)
                unique_tracks.append(track_id)

            if identity_id and identity_id not in seen_clusters:
                seen_clusters.add(identity_id)
                unique_clusters.append(identity_id)

        # Display in expander with organized sections
        with st.expander("🔗 View Tracks & Clusters", expanded=False):
            # Clusters section
            if unique_clusters:
                st.markdown("**Clusters:**")
                cluster_cols = st.columns(min(len(unique_clusters), 4))
                for idx, cluster_id in enumerate(unique_clusters):
                    col_idx = idx % len(cluster_cols)
                    with cluster_cols[col_idx]:
                        cluster_url = f"/Faces_Review?ep_id={ep_id}&view=cluster&cluster={cluster_id}"
                        st.markdown(
                            f"[👤 {cluster_id}]({cluster_url})",
                            unsafe_allow_html=True,
                        )

            # Tracks section
            if unique_tracks:
                if unique_clusters:
                    st.markdown("---")
                st.markdown("**Tracks:**")
                track_cols = st.columns(min(len(unique_tracks), 6))
                for idx, track_id in enumerate(unique_tracks):
                    col_idx = idx % len(track_cols)
                    with track_cols[col_idx]:
                        track_url = f"/Faces_Review?ep_id={ep_id}&view=track&track={track_id}"
                        st.markdown(
                            f"[🎬 T{track_id}]({track_url})",
                            unsafe_allow_html=True,
                        )
    else:
        st.info("No faces detected in this frame.")
