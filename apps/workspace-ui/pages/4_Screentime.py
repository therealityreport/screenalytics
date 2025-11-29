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

# Configuration options
with st.expander("Analysis Configuration", expanded=False):
    quality_min = st.slider(
        "Minimum face quality threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Only face samples with quality >= this value will be counted",
    )
    gap_tolerance_s = st.number_input(
        "Gap tolerance (seconds)",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
        help="Maximum gap between face samples to be considered continuous",
    )
    use_video_decode = st.checkbox(
        "Use video decode for timestamps",
        value=True,
        help="Use video decoding for accurate timestamps (slower but more accurate)",
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
current_job_id = st.session_state.get("current_screentime_job")
if current_job_id:
    try:
        job_progress_resp = helpers.api_get(f"/jobs/{current_job_id}/progress")
        job_state = job_progress_resp.get("state")
        progress_data = job_progress_resp.get("progress", {})

        if job_state == "running":
            st.info(f"Job {current_job_id[:12]}... is currently running")

            # Show progress bar if we have progress data
            if progress_data:
                frames_done = progress_data.get("frames_done", 0)
                frames_total = progress_data.get("frames_total", 1)
                elapsed = progress_data.get("elapsed_sec", 0)

                progress_pct = frames_done / max(frames_total, 1)
                st.progress(progress_pct)

                col1, col2, col3 = st.columns(3)
                col1.metric("Frames Processed", f"{frames_done}/{frames_total}")
                col2.metric("Progress", f"{progress_pct * 100:.1f}%")
                col3.metric("Elapsed Time", f"{elapsed:.1f}s")

            # Auto-refresh every 2 seconds
            time.sleep(2)
            st.rerun()

        elif job_state in ("succeeded", "failed"):
            if job_state == "succeeded":
                st.success(f"Job {current_job_id[:12]}... completed successfully!")
            else:
                st.error(f"Job {current_job_id[:12]}... failed")

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
            # Bar chart of screen time
            chart_data = pd.DataFrame(
                {
                    "Person": [m.get("person_id", "unknown") for m in metrics],
                    "Screen Time (s)": [m.get("visual_s", 0.0) for m in metrics],
                }
            ).sort_values("Screen Time (s)", ascending=False)

            st.bar_chart(chart_data.set_index("Person"), height=400)

        with tab2:
            # Pie chart showing percentage distribution
            import plotly.express as px

            pie_data = pd.DataFrame(
                {
                    "Person": [m.get("person_id", "unknown") for m in metrics],
                    "Screen Time": [m.get("visual_s", 0.0) for m in metrics],
                }
            )

            fig = px.pie(
                pie_data,
                values="Screen Time",
                names="Person",
                title="Screen Time Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Scatter plot: Tracks vs Faces
            scatter_data = pd.DataFrame(
                {
                    "Person": [m.get("person_id", "unknown") for m in metrics],
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
                hover_data=["Person"],
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

st.subheader("Output Files")
if json_path.exists():
    st.write(f"✅ JSON → {helpers.link_local(json_path)}")
else:
    st.write(f"⚠️ JSON → {json_path} (not yet generated)")

if csv_path.exists():
    st.write(f"✅ CSV → {helpers.link_local(csv_path)}")
else:
    st.write(f"⚠️ CSV → {csv_path} (not yet generated)")
