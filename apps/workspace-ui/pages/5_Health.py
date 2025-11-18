from __future__ import annotations

import shutil
import sys
from pathlib import Path

import requests
import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402

cfg = helpers.init_page("Health")
st.title("System Health")

st.write(f"API base: `{cfg['api_base']}`")
st.write(f"Backend: `{cfg['backend']}` | Bucket: `{cfg['bucket']}`")
st.write(f"Data root: {helpers.link_local(helpers.DATA_ROOT)}")

usage = shutil.disk_usage(helpers.DATA_ROOT)
st.write(
    f"Disk usage → total={usage.total // (1024**3)} GiB, used={usage.used // (1024**3)} GiB, "
    f"free={usage.free // (1024**3)} GiB"
)

ep_id = helpers.get_ep_id()
if ep_id:
    st.subheader("Current episode")
    st.write(f"ep_id: `{ep_id}`")
    try:
        detail = helpers.api_get(f"/episodes/{ep_id}")
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{cfg['api_base']}/episodes/{ep_id}", exc))
    else:
        # Robustly handle S3 status (key field may be missing)
        s3 = detail.get("s3")
        if not isinstance(s3, dict):
            st.caption("S3 status: not available for this check")
        else:
            bucket = s3.get("bucket")
            key = s3.get("key")
            exists = s3.get("exists")

            # Build path string that tolerates missing pieces
            if bucket and key:
                path = f"{bucket}/{key}"
            elif bucket:
                path = bucket
            elif key:
                path = key
            else:
                path = "(no S3 path)"

            # Safe representation of exists
            if exists is None:
                exists_str = "unknown"
            else:
                exists_str = "True" if bool(exists) else "False"

            st.write(f"S3 object `{path}` exists → {exists_str}")

        # Local path status
        local = detail.get("local")
        if isinstance(local, dict) and local.get("path"):
            local_exists = local.get("exists", "unknown")
            if isinstance(local_exists, bool):
                exists_str = "True" if local_exists else "False"
            else:
                exists_str = str(local_exists)
            st.write(
                f"Local path {helpers.link_local(local['path'])} exists → {exists_str}"
            )
        else:
            st.caption("Local path status: not available")

        # Display detect/track crop diagnostics if available
        st.subheader("Pipeline diagnostics")

        # Detect/Track phase
        detect_track = detail.get("detect_track") or {}
        dt_meta = detect_track.get("meta") or {}

        # Try different possible locations for crop diagnostics
        crop_attempts = dt_meta.get("crop_attempts")
        crop_errors = dt_meta.get("crop_errors")

        # Also check in detect_track_stats if meta doesn't have them
        if crop_attempts is None:
            dt_stats = dt_meta.get("detect_track_stats") or {}
            crop_attempts = dt_stats.get("crop_attempts")
            crop_errors = dt_stats.get("crop_errors")

        if crop_attempts is not None and crop_errors is not None and crop_attempts > 0:
            error_rate = crop_errors / crop_attempts if crop_attempts > 0 else 0.0
            st.caption(
                f"**Detect/Track crops:** {crop_errors} / {crop_attempts} failed "
                f"({error_rate:.1%} of attempts skipped due to invalid bboxes)."
            )
        elif crop_attempts is not None and crop_attempts > 0:
            st.caption(f"**Detect/Track crops:** {crop_attempts} attempts (no error stats available).")
        elif detect_track:
            st.caption("**Detect/Track:** No crop diagnostics available for this run.")

        # Show basic detect/track status if available
        dt_status = detect_track.get("status")
        if dt_status:
            st.caption(f"Detect/Track status: `{dt_status}`")

        # Faces phase
        faces = detail.get("faces") or {}
        faces_status = faces.get("status")
        if faces_status:
            st.caption(f"Faces status: `{faces_status}`")

        # Cluster phase
        cluster = detail.get("cluster") or {}
        cluster_status = cluster.get("status")
        if cluster_status:
            st.caption(f"Cluster status: `{cluster_status}`")
else:
    st.info("Set an ep_id from another page (Upload/Episodes) to probe specific objects.")
