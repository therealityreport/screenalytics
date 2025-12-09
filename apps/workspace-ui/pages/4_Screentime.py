from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import NoReturn

import pandas as pd
import requests
import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402

cfg = helpers.init_page("Screentime")
st.title("Screen Time Analysis")


def _stop_forever() -> NoReturn:
    """Wrapper so type-checkers understand we don't return after st.stop()."""
    st.stop()
    raise RuntimeError("Streamlit stop returned unexpectedly")


def _require_episode() -> str:
    ep_id = helpers.get_ep_id()
    if ep_id:
        return ep_id
    try:
        payload = helpers.api_get("/episodes")
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{cfg['api_base']}/episodes", exc))
        _stop_forever()
    episodes = payload.get("episodes", [])
    if not episodes:
        st.info("No episodes yet.")
        _stop_forever()
    option_lookup = {ep["ep_id"]: ep for ep in episodes}
    selection = st.selectbox(
        "Episode",
        list(option_lookup.keys()),
        format_func=lambda eid: f"{eid} ({option_lookup[eid]['show_slug']})",
    )
    if st.button("Load episode", use_container_width=True):
        helpers.set_ep_id(selection)
        return selection
    _stop_forever()


ep_id = _require_episode()
helpers.set_ep_id(ep_id)

analytics_dir = helpers.DATA_ROOT / "analytics" / ep_id
json_path = analytics_dir / "screentime.json"
csv_path = analytics_dir / "screentime.csv"

# Job history section
st.subheader("Recent Jobs")
try:
    jobs_resp = helpers.api_get(f"/jobs?ep_id={ep_id}&job_type=screen_time_analyze")
    jobs_list = jobs_resp.get("jobs", [])
    if jobs_list:
        job_table = []
        for job in jobs_list[:5]:  # Show last 5 jobs
            job_table.append(
                {
                    "Job ID": job["job_id"][:8] + "...",
                    "Status": job["state"],
                    "Started": job.get("started_at", "N/A")[:19].replace("T", " "),
                    "Ended": (job.get("ended_at", "N/A")[:19].replace("T", " ") if job.get("ended_at") else "Running"),
                }
            )
        st.dataframe(job_table, use_container_width=True, hide_index=True)
    else:
        st.info("No screen time jobs have been run yet for this episode.")
except requests.RequestException:
    st.warning("Could not fetch job history.")

st.divider()

# Configuration options - preset defaults from config/pipeline/screen_time_v2.yaml (bravo_default)
PRESET_DEFAULTS = {
    "bravo_default": {
        "quality_min": 0.48,
        "gap_tolerance_s": 1.2,
        "use_video_decode": True,
        "screen_time_mode": "tracks",
        "edge_padding_s": 0.2,
        "track_coverage_min": 0.35,
    },
    "stricter": {
        "quality_min": 0.65,
        "gap_tolerance_s": 0.6,
        "use_video_decode": True,
        "screen_time_mode": "faces",
        "edge_padding_s": 0.05,
        "track_coverage_min": 0.5,
    },
}
DEFAULT_PRESET = "bravo_default"

with st.expander("Analysis Configuration", expanded=False):
    # Preset selector
    preset = st.selectbox(
        "Preset",
        options=list(PRESET_DEFAULTS.keys()),
        index=0,
        help="Select a configuration preset. 'bravo_default' is optimized for reality TV with quick cuts.",
    )
    preset_values = PRESET_DEFAULTS.get(preset, PRESET_DEFAULTS[DEFAULT_PRESET])

    st.caption(f"Using preset: **{preset}**")

    col1, col2 = st.columns(2)
    with col1:
        quality_min = st.slider(
            "Minimum face quality threshold",
            min_value=0.0,
            max_value=1.0,
            value=preset_values["quality_min"],
            step=0.05,
            help="Only face samples with quality >= this value will be counted",
        )
        gap_tolerance_s = st.number_input(
            "Gap tolerance (seconds)",
            min_value=0.0,
            max_value=5.0,
            value=preset_values["gap_tolerance_s"],
            step=0.1,
            help="Maximum gap between face samples to be considered continuous",
        )
        use_video_decode = st.checkbox(
            "Use video decode for timestamps",
            value=preset_values["use_video_decode"],
            help="Use video decoding for accurate timestamps (slower but more accurate)",
        )

    with col2:
        screen_time_mode = st.selectbox(
            "Screen time mode",
            options=["tracks", "faces"],
            index=0 if preset_values["screen_time_mode"] == "tracks" else 1,
            help="'tracks' uses full track spans (recommended), 'faces' uses per-face intervals",
        )
        edge_padding_s = st.number_input(
            "Edge padding (seconds)",
            min_value=0.0,
            max_value=1.0,
            value=preset_values["edge_padding_s"],
            step=0.05,
            help="Pad each interval to better match human 'in/out' perception",
        )
        track_coverage_min = st.slider(
            "Track coverage minimum",
            min_value=0.0,
            max_value=1.0,
            value=preset_values["track_coverage_min"],
            step=0.05,
            help="Require at least this detection coverage for tracks (only in 'tracks' mode)",
        )

# Job launch section
col1, col2 = st.columns([3, 1])
with col1:
    if st.button("Analyze Screen Time", use_container_width=True, type="primary"):
        try:
            payload = {
                "ep_id": ep_id,
                "quality_min": quality_min,
                "gap_tolerance_s": gap_tolerance_s,
                "use_video_decode": use_video_decode,
                "screen_time_mode": screen_time_mode,
                "edge_padding_s": edge_padding_s,
                "track_coverage_min": track_coverage_min,
                "preset": preset,
            }
            resp = helpers.api_post("/jobs/screen_time/analyze", payload)
            job_id = resp.get("job_id")
            st.session_state["current_screentime_job"] = job_id
            st.success(f"Screen time analysis job started: {job_id[:12]}...")
            st.rerun()
        except requests.RequestException as exc:
            st.error(helpers.describe_error(f"{cfg['api_base']}/jobs/screen_time/analyze", exc))
with col2:
    if st.button("Refresh", use_container_width=True):
        st.rerun()

# Progress monitoring section
# Screen time job phases: init -> loading -> analyzing -> writing -> done
SCREEN_TIME_PHASES = ["init", "loading", "analyzing", "writing", "done"]
PHASE_LABELS = {
    "init": "Initializing...",
    "loading": "Loading episode data...",
    "analyzing": "Analyzing screen time...",
    "writing": "Writing outputs...",
    "done": "Complete",
    "error": "Error",
}

current_job_id = st.session_state.get("current_screentime_job")
if current_job_id:
    try:
        job_progress_resp = helpers.api_get(f"/jobs/{current_job_id}/progress")
        job_state = job_progress_resp.get("state")
        progress_data = job_progress_resp.get("progress", {})

        if job_state == "running":
            st.info(f"Job {current_job_id[:12]}... is currently running")

            # Show progress based on phase (screen time jobs use phase-based progress)
            if progress_data:
                phase = progress_data.get("phase", "init")
                message = progress_data.get("message", "")
                cast_count = progress_data.get("cast_count")

                # Calculate progress based on phase
                if phase in SCREEN_TIME_PHASES:
                    phase_idx = SCREEN_TIME_PHASES.index(phase)
                    progress_pct = (phase_idx + 1) / len(SCREEN_TIME_PHASES)
                else:
                    progress_pct = 0.1

                st.progress(progress_pct)

                col1, col2 = st.columns(2)
                col1.metric("Phase", PHASE_LABELS.get(phase, phase.title()))
                if cast_count is not None:
                    col2.metric("Cast Members Found", cast_count)
                else:
                    col2.metric("Status", message[:50] if message else "Processing...")

            # Auto-refresh every 2 seconds
            time.sleep(2)
            st.rerun()

        elif job_state in ("succeeded", "failed"):
            if job_state == "succeeded":
                st.success(f"Job {current_job_id[:12]}... completed successfully!")
            else:
                error_msg = progress_data.get("message", "Unknown error") if progress_data else "Unknown error"
                st.error(f"Job {current_job_id[:12]}... failed: {error_msg}")

            # Clear the current job from session state
            if st.button("Clear Job Status"):
                st.session_state.pop("current_screentime_job", None)
                st.rerun()

    except requests.RequestException:
        # Job not found or API error - clear from session
        if st.button("Clear Job Status"):
            st.session_state.pop("current_screentime_job", None)
            st.rerun()

st.divider()

if json_path.exists():
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    st.subheader("Screen Time Results")

    # Display metadata
    generated_at = data.get("generated_at", "unknown")
    st.caption(f"Generated: {generated_at[:19].replace('T', ' ')}")

    metrics = data.get("metrics", [])
    if metrics:
        # Summary stats at the top
        total_time = sum(m.get("visual_s", 0.0) for m in metrics)
        total_tracks = sum(m.get("tracks_count", 0) for m in metrics)
        total_faces = sum(m.get("faces_count", 0) for m in metrics)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Screen Time", f"{total_time:.1f}s ({total_time / 60:.1f} min)")
        col2.metric("Cast Members", len(metrics))
        col3.metric("Total Faces", total_faces)

        st.divider()

        # Helper function to format seconds as MM:SS.milliseconds
        def format_time(seconds):
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            secs = int(remaining_seconds)
            milliseconds = int((remaining_seconds - secs) * 1000)
            return f"{minutes:02d}:{secs:02d}.{milliseconds:03d}"

        # Prepare data for display
        display_rows = []
        for m in metrics:
            visual_s = m.get("visual_s", 0.0)
            display_rows.append(
                {
                    "Name": m.get("name", "Unknown"),
                    "Time": format_time(visual_s),
                    "Percentage": f"{(visual_s / max(total_time, 1)) * 100:.1f}%",
                    "Tracks": m.get("tracks_count", 0),
                    "Faces": m.get("faces_count", 0),
                    "Confidence": f"{m.get('confidence', 0.0):.3f}",
                    "Person ID": m.get("person_id", "unknown"),
                    "Cast ID": m.get("cast_id", "unknown")[:12] + "...",
                    "_visual_s": visual_s,  # Hidden column for sorting
                }
            )

        # Convert to DataFrame for better display
        df = pd.DataFrame(display_rows)

        # Interactive table with sorting
        st.subheader("Per-Cast Breakdown")
        sort_by = st.selectbox(
            "Sort by",
            ["Time", "Percentage", "Tracks", "Faces", "Confidence", "Name"],
            index=0,
        )
        ascending = st.checkbox("Ascending order", value=False)

        # Sort the dataframe
        if sort_by == "Time":
            df_sorted = df.sort_values("_visual_s", ascending=ascending)
        else:
            df_sorted = df.sort_values(sort_by, ascending=ascending)

        # Display without the hidden _visual_s column
        display_df = df_sorted.drop(columns=["_visual_s"])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.divider()

        # Charts section
        st.subheader("Visualizations")

        # Tab layout for different chart types
        tab1, tab2, tab3 = st.tabs(["Bar Chart", "Pie Chart", "Time Distribution"])

        with tab1:
            # Bar chart of screen time - use name for display
            chart_data = pd.DataFrame(
                {
                    "Cast Member": [m.get("name", m.get("person_id", "unknown")) for m in metrics],
                    "Screen Time (s)": [m.get("visual_s", 0.0) for m in metrics],
                }
            ).sort_values("Screen Time (s)", ascending=False)

            st.bar_chart(chart_data.set_index("Cast Member"), height=400)

        with tab2:
            # Pie chart showing percentage distribution
            import plotly.express as px

            pie_data = pd.DataFrame(
                {
                    "Cast Member": [m.get("name", m.get("person_id", "unknown")) for m in metrics],
                    "Screen Time": [m.get("visual_s", 0.0) for m in metrics],
                }
            )

            fig = px.pie(
                pie_data,
                values="Screen Time",
                names="Cast Member",
                title="Screen Time Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Scatter plot: Tracks vs Faces
            scatter_data = pd.DataFrame(
                {
                    "Cast Member": [m.get("name", m.get("person_id", "unknown")) for m in metrics],
                    "Tracks": [m.get("tracks_count", 0) for m in metrics],
                    "Faces": [m.get("faces_count", 0) for m in metrics],
                    "Screen Time (s)": [m.get("visual_s", 0.0) for m in metrics],
                }
            )

            fig2 = px.scatter(
                scatter_data,
                x="Tracks",
                y="Faces",
                size="Screen Time (s)",
                hover_data=["Cast Member"],
                title="Tracks vs Faces (bubble size = screen time)",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # Detailed metrics expansion
        with st.expander("View Raw Metrics JSON"):
            st.json(metrics)

    else:
        # Show diagnostic-based empty state messages
        diagnostics = data.get("diagnostics", {})

        st.warning("No cast members found with screen time.")

        # Provide specific guidance based on diagnostics
        if diagnostics:
            with st.expander("Diagnostic Information", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Faces Loaded", diagnostics.get("faces_loaded", "N/A"))
                    st.metric("Tracks Loaded", diagnostics.get("tracks_loaded", "N/A"))
                    st.metric("Identities Loaded", diagnostics.get("identities_loaded", "N/A"))
                with col2:
                    st.metric(
                        "People with Cast ID",
                        diagnostics.get("people_with_cast_id", "N/A"),
                    )
                    st.metric(
                        "Tracks Mapped to Identity",
                        diagnostics.get("tracks_mapped_to_identity", "N/A"),
                    )
                    st.metric(
                        "Tracks with Cast ID",
                        diagnostics.get("tracks_with_cast_id", "N/A"),
                    )

                # Show specific issues
                if diagnostics.get("tracks_missing_identity", 0) > 0:
                    st.info(f"⚠️ {diagnostics['tracks_missing_identity']} tracks have no identity assignment")
                if diagnostics.get("tracks_missing_person", 0) > 0:
                    st.info(f"⚠️ {diagnostics['tracks_missing_person']} identities have no person mapping")
                if diagnostics.get("tracks_missing_cast", 0) > 0:
                    st.info(f"⚠️ {diagnostics['tracks_missing_cast']} people have no cast_id assignment")

        # Specific guidance based on what's missing
        people_with_cast = diagnostics.get("people_with_cast_id", 0)
        tracks_with_cast = diagnostics.get("tracks_with_cast_id", 0)

        if people_with_cast == 0:
            st.error("**No cast members are linked to this show.**")
            st.info("➡️ Please assign cast members in the **Cast page** before running screen time analysis.")
        elif tracks_with_cast == 0:
            st.error("**No identities in this episode are linked to cast members.**")
            st.info(
                "➡️ Confirm that identities have been assigned to cast-linked people in the "
                "**Faces & Tracks Review** page for this episode."
            )
        else:
            st.info("Make sure cast members are assigned in the Cast page before running screen time analysis.")
else:
    st.info("No screentime analytics yet. Click 'Analyze Screen Time' to generate results.")

# Timestamp Preview Section
st.divider()
st.subheader("Timestamp Preview")
st.caption("Enter a timestamp to see the video frame with detected faces and cast member names.")


def parse_timestamp(ts_str: str) -> float | None:
    """Parse timestamp string to seconds. Supports MM:SS, MM:SS.ms, or raw seconds."""
    ts_str = ts_str.strip()
    if not ts_str:
        return None
    try:
        # Try raw seconds first
        return float(ts_str)
    except ValueError:
        pass
    # Try MM:SS or MM:SS.ms format
    if ":" in ts_str:
        parts = ts_str.split(":")
        if len(parts) == 2:
            try:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            except ValueError:
                pass
        elif len(parts) == 3:
            try:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
            except ValueError:
                pass
    return None


col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    timestamp_input = st.text_input(
        "Timestamp",
        value=st.session_state.get(f"{ep_id}::preview_timestamp", "0:30"),
        placeholder="e.g., 1:30, 90, or 1:30.5",
        help="Enter timestamp as MM:SS, MM:SS.ms, HH:MM:SS, or seconds",
        key=f"{ep_id}::timestamp_input",
    )
with col2:
    include_unidentified = st.checkbox(
        "Show unidentified",
        value=True,
        help="Include faces without cast member assignment (shown in gray)",
        key=f"{ep_id}::include_unidentified",
    )
with col3:
    generate_preview = st.button(
        "Generate Preview",
        type="primary",
        use_container_width=True,
        key=f"{ep_id}::generate_preview",
    )

if generate_preview:
    timestamp_s = parse_timestamp(timestamp_input)
    if timestamp_s is None:
        st.error(f"Invalid timestamp format: '{timestamp_input}'. Use MM:SS, MM:SS.ms, or seconds.")
    else:
        st.session_state[f"{ep_id}::preview_timestamp"] = timestamp_input
        with st.spinner(f"Generating preview at {timestamp_input}..."):
            try:
                resp = helpers.api_get(
                    f"/episodes/{ep_id}/timestamp/{timestamp_s}/preview",
                    params={"include_unidentified": str(include_unidentified).lower()},
                )
                st.session_state[f"{ep_id}::preview_result"] = resp
            except requests.RequestException as exc:
                st.error(helpers.describe_error(f"Preview generation", exc))
                st.session_state.pop(f"{ep_id}::preview_result", None)

# Display preview result if available
preview_result = st.session_state.get(f"{ep_id}::preview_result")
if preview_result:
    preview_url = preview_result.get("url")
    frame_idx = preview_result.get("frame_idx")
    actual_timestamp = preview_result.get("timestamp_s", 0)
    requested_timestamp = preview_result.get("requested_timestamp_s", actual_timestamp)
    gap_frames = preview_result.get("gap_frames", 0)
    gap_seconds = preview_result.get("gap_seconds", 0)
    fps = preview_result.get("fps", 30)
    duration_s = preview_result.get("duration_s", 0)
    faces = preview_result.get("faces", [])

    # Format timestamp as MM:SS.ms
    mins = int(actual_timestamp // 60)
    secs = actual_timestamp % 60
    ts_display = f"{mins}:{secs:05.2f}"

    # Show gap warning if significant (> 0.5s)
    if gap_seconds > 0.5:
        st.warning(
            f"No faces detected at requested time ({requested_timestamp:.1f}s). "
            f"Showing nearest frame with faces: {gap_seconds:.1f}s away (frame gap: {gap_frames})"
        )

    st.caption(f"Frame {frame_idx} at {ts_display} ({fps:.1f} fps, {duration_s:.1f}s total)")

    # Display the preview image
    if preview_url:
        # Handle both local paths and URLs
        if preview_url.startswith("http"):
            st.image(preview_url, use_column_width=True)
        else:
            # Local file path
            from pathlib import Path as StPath

            local_preview = StPath(preview_url)
            if local_preview.exists():
                st.image(str(local_preview), use_column_width=True)
            else:
                st.warning(f"Preview file not found: {preview_url}")

    # Show detected faces table
    if faces:
        st.caption(f"**Detected faces: {len(faces)}**")
        face_rows = []
        for f in faces:
            name = f.get("name")
            identity_id = f.get("identity_id")
            track_id = f.get("track_id")
            if name:
                label = name
                status = "Identified"
            elif identity_id:
                label = f"[{identity_id[:12]}...]"
                status = "Clustered (no cast)"
            else:
                label = f"Track {track_id}"
                status = "Unassigned"
            face_rows.append(
                {
                    "Label": label,
                    "Status": status,
                    "Track ID": track_id,
                    "Person ID": f.get("person_id") or "-",
                }
            )
        st.dataframe(face_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No faces detected in this frame.")

    # Video clip export section
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
        center_ts = actual_timestamp
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
                from pathlib import Path as ClipPath

                clip_path = ClipPath(clip_url)
                if clip_path.exists():
                    st.video(str(clip_path))
                    with open(clip_path, "rb") as f:
                        st.download_button(
                            "Download Clip",
                            data=f.read(),
                            file_name=clip_path.name,
                            mime="video/mp4",
                            key=f"{ep_id}::download_clip",
                        )
                else:
                    st.warning(f"Clip file not found: {clip_url}")

st.divider()

st.subheader("Output Files")
if json_path.exists():
    st.write(f"✅ JSON → {helpers.link_local(json_path)}")
else:
    st.write(f"⚠️ JSON → {json_path} (not yet generated)")

if csv_path.exists():
    st.write(f"✅ CSV → {helpers.link_local(csv_path)}")
else:
    st.write(f"⚠️ CSV → {csv_path} (not yet generated)")
