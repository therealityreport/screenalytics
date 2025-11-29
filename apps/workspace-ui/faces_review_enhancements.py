"""
Faces Review UI Enhancement Components

This module contains UI components for the enhanced Faces Review functionality:
- Enhancement #4: Virtual Scrolling CSS Styles
- Enhancement #5: Keyboard Shortcuts
- Enhancement #7: Real-time Collaboration Indicators
- Enhancement #8: Intelligent Outlier Detection with Visual Highlighting
- Enhancement #9: My Jobs Panel
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st


# =============================================================================
# Enhancement #4: Virtual Scrolling CSS Styles
# =============================================================================

VIRTUAL_SCROLL_CSS = """
<style>
/* Virtual scrolling container for large lists */
.virtual-scroll-container {
    height: 600px;
    overflow-y: auto;
    scroll-behavior: smooth;
}

.virtual-scroll-item {
    min-height: 80px;
    padding: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

/* Loading skeleton */
.skeleton-loader {
    background: linear-gradient(90deg, #1e1e1e 25%, #2a2a2a 50%, #1e1e1e 75%);
    background-size: 200% 100%;
    animation: skeleton-pulse 1.5s ease-in-out infinite;
    border-radius: 4px;
}

@keyframes skeleton-pulse {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* Jump to search styling */
.jump-to-search {
    position: sticky;
    top: 0;
    background: var(--background-color);
    z-index: 100;
    padding: 8px 0;
}
</style>
"""


def render_virtual_scroll_styles():
    """Inject virtual scrolling CSS styles."""
    st.markdown(VIRTUAL_SCROLL_CSS, unsafe_allow_html=True)


def render_skeleton_grid(rows: int = 3, cols: int = 4, height: int = 100):
    """Render a skeleton loading grid."""
    for _ in range(rows):
        columns = st.columns(cols)
        for col in columns:
            with col:
                st.markdown(
                    f'<div class="skeleton-loader" style="height: {height}px;"></div>',
                    unsafe_allow_html=True,
                )


# =============================================================================
# Enhancement #5: Keyboard Shortcuts
# =============================================================================

KEYBOARD_SHORTCUTS_JS = """
<script>
(function() {
    if (window._facesReviewShortcutsInitialized) return;
    window._facesReviewShortcutsInitialized = true;

    document.addEventListener('keydown', function(e) {
        // Skip if typing in input fields
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if (e.target.contentEditable === 'true') return;

        const shortcuts = {
            'Escape': 'back_button',        // Go back
            'j': 'next_item',               // Next item
            'k': 'prev_item',               // Previous item
            'a': 'assign_button',           // Assign action
            'u': 'undo_button',             // Undo
        };

        const buttonId = shortcuts[e.key];
        if (buttonId) {
            // Find button by data-testid or class
            const btn = document.querySelector(`[data-testid="${buttonId}"], .${buttonId}`);
            if (btn) {
                e.preventDefault();
                btn.click();
            }
        }
    });
})();
</script>
"""


def render_keyboard_shortcuts_handler():
    """Render keyboard shortcuts JavaScript handler."""
    st.markdown(KEYBOARD_SHORTCUTS_JS, unsafe_allow_html=True)


def render_shortcuts_help():
    """Render keyboard shortcuts help panel."""
    with st.expander("Keyboard Shortcuts", expanded=False):
        st.markdown("""
        | Key | Action |
        |-----|--------|
        | `Esc` | Go back to previous view |
        | `J` | Navigate to next item |
        | `K` | Navigate to previous item |
        | `A` | Assign to last used cast member |
        | `U` | Undo last operation |
        """)


# =============================================================================
# Enhancement #7: Real-time Collaboration Indicators
# =============================================================================

COLLABORATION_CSS = """
<style>
.presence-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: rgba(76, 175, 80, 0.2);
    border-radius: 16px;
    font-size: 12px;
    color: #4CAF50;
}

.presence-dot {
    width: 8px;
    height: 8px;
    background: #4CAF50;
    border-radius: 50%;
    animation: presence-pulse 2s infinite;
}

@keyframes presence-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.recently-modified-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 6px;
    background: rgba(255, 193, 7, 0.2);
    border-radius: 8px;
    font-size: 10px;
    color: #FFC107;
}
</style>
"""


def render_collaboration_styles():
    """Inject collaboration indicator CSS styles."""
    st.markdown(COLLABORATION_CSS, unsafe_allow_html=True)


def render_presence_indicator(viewers: List[Dict[str, Any]]):
    """Render indicator showing other users viewing this episode.

    Args:
        viewers: List of viewer info dicts with 'name' key
    """
    if not viewers:
        return

    viewer_count = len(viewers)
    viewer_names = [v.get("name", "Anonymous") for v in viewers[:3]]

    if viewer_count == 1:
        names_str = viewer_names[0]
    elif viewer_count <= 3:
        names_str = ", ".join(viewer_names)
    else:
        names_str = f"{', '.join(viewer_names)} +{viewer_count - 3} others"

    st.markdown(
        f"""
        <div class="presence-indicator">
            <span class="presence-dot"></span>
            <span>Also viewing: {names_str}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recently_modified_badge(modified_by: str, modified_at: str):
    """Render badge showing recent modification info."""
    st.markdown(
        f"""
        <span class="recently-modified-badge">
            Modified by {modified_by} at {modified_at}
        </span>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Enhancement #8: Intelligent Outlier Detection with Visual Highlighting
# =============================================================================

OUTLIER_THRESHOLD = 0.65  # Tracks below this cast_track_score are considered outliers

OUTLIER_CSS = """
<style>
/* Outlier track highlighting */
.outlier-track {
    border: 2px solid #ff5252 !important;
    box-shadow: 0 0 10px rgba(255, 82, 82, 0.3);
    position: relative;
}

.outlier-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    background: rgba(255, 82, 82, 0.2);
    border: 1px solid #ff5252;
    border-radius: 12px;
    color: #ff5252;
    font-size: 11px;
    font-weight: 500;
}

.outlier-warning {
    background: rgba(255, 82, 82, 0.1);
    border-left: 3px solid #ff5252;
    padding: 8px 12px;
    margin: 8px 0;
    border-radius: 0 4px 4px 0;
}

/* Outlier filter toggle */
.outlier-filter-active {
    background: rgba(255, 82, 82, 0.2);
    border-color: #ff5252;
}
</style>
"""


def render_outlier_styles():
    """Inject outlier highlighting CSS styles."""
    st.markdown(OUTLIER_CSS, unsafe_allow_html=True)


def render_outlier_badge(score: float, threshold: float = OUTLIER_THRESHOLD) -> str:
    """Return HTML for an outlier warning badge if score is below threshold."""
    if score is None or score >= threshold:
        return ""
    return f"""
    <span class="outlier-badge">
        Low fit ({score:.0%})
    </span>
    """


def is_outlier_track(track_meta: Dict[str, Any], threshold: float = OUTLIER_THRESHOLD) -> bool:
    """Check if a track is an outlier based on cast_track_score."""
    score = track_meta.get("cast_track_score")
    if score is None:
        return False
    return score < threshold


def filter_outliers(tracks: List[Dict[str, Any]], threshold: float = OUTLIER_THRESHOLD) -> List[Dict[str, Any]]:
    """Filter tracks to only include outliers."""
    return [t for t in tracks if is_outlier_track(t, threshold)]


def render_outlier_summary(tracks: List[Dict[str, Any]], threshold: float = OUTLIER_THRESHOLD):
    """Render summary of outlier tracks with investigate button."""
    outliers = filter_outliers(tracks, threshold)

    if not outliers:
        return

    st.markdown(
        f"""
        <div class="outlier-warning">
            <strong>Potential Outliers Found:</strong> {len(outliers)} track(s)
            have low similarity to this person's profile (below {threshold:.0%})
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_outlier_distribution(tracks: List[Dict[str, Any]]) -> Dict[str, int]:
    """Get distribution of tracks by outlier severity.

    Returns:
        Dict with counts for 'severe' (<0.5), 'moderate' (0.5-0.65), 'normal' (>=0.65)
    """
    severe = 0
    moderate = 0
    normal = 0

    for track in tracks:
        score = track.get("cast_track_score")
        if score is None:
            continue
        if score < 0.5:
            severe += 1
        elif score < OUTLIER_THRESHOLD:
            moderate += 1
        else:
            normal += 1

    return {
        "severe": severe,
        "moderate": moderate,
        "normal": normal,
    }


# =============================================================================
# Enhancement #9: My Jobs Panel (Job History in Sidebar)
# =============================================================================

def render_my_jobs_panel(api_get_func):
    """Render active/recent jobs panel in sidebar.

    Args:
        api_get_func: Function to make API GET requests
    """
    with st.sidebar.expander("My Jobs", expanded=False):
        try:
            resp = api_get_func("/celery_jobs/active")
            active_jobs = resp.get("jobs", []) if resp else []

            if not active_jobs:
                st.caption("No active jobs")
                return

            for job in active_jobs[:5]:
                job_id = job.get("job_id", "unknown")[:8]
                op = job.get("operation", "unknown")
                ep = job.get("episode_id", "")[:20]
                status = job.get("status", "unknown")

                status_icon = {
                    "queued": "[Q]",
                    "in_progress": "[...]",
                    "success": "[OK]",
                    "failed": "[X]",
                }.get(status, "[?]")

                st.markdown(f"{status_icon} **{op}** on `{ep}` ({job_id})")

        except Exception as e:
            st.caption(f"Could not load jobs: {e}")


def render_job_history_panel(api_get_func, limit: int = 10):
    """Render job history panel.

    Args:
        api_get_func: Function to make API GET requests
        limit: Maximum number of jobs to show
    """
    with st.expander("Job History", expanded=False):
        try:
            resp = api_get_func(f"/celery_jobs/history?limit={limit}")
            jobs = resp.get("jobs", []) if resp else []

            if not jobs:
                st.caption("No job history")
                return

            for job in jobs:
                job_id = job.get("job_id", "unknown")[:8]
                op = job.get("operation", "unknown")
                ep = job.get("episode_id", "")[:15]
                status = job.get("status", "unknown")
                duration = job.get("duration_ms")

                duration_str = f" ({duration/1000:.1f}s)" if duration else ""
                status_icon = "[OK]" if status == "success" else "[X]"

                st.text(f"{status_icon} {op} on {ep}{duration_str}")

        except Exception:
            st.caption("Could not load history")


# =============================================================================
# Combined Enhancement Initializer
# =============================================================================

def init_all_enhancements():
    """Initialize all UI enhancements by injecting required styles and scripts."""
    render_virtual_scroll_styles()
    render_keyboard_shortcuts_handler()
    render_collaboration_styles()
    render_outlier_styles()


__all__ = [
    # Enhancement #4
    "render_virtual_scroll_styles",
    "render_skeleton_grid",
    # Enhancement #5
    "render_keyboard_shortcuts_handler",
    "render_shortcuts_help",
    # Enhancement #7
    "render_collaboration_styles",
    "render_presence_indicator",
    "render_recently_modified_badge",
    # Enhancement #8
    "OUTLIER_THRESHOLD",
    "render_outlier_styles",
    "render_outlier_badge",
    "is_outlier_track",
    "filter_outliers",
    "render_outlier_summary",
    "get_outlier_distribution",
    # Enhancement #9
    "render_my_jobs_panel",
    "render_job_history_panel",
    # Combined
    "init_all_enhancements",
]
