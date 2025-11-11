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
        st.write(
            f"S3 object `{detail['s3']['bucket']}/{detail['s3']['key']}` exists → {detail['s3']['exists']}"
        )
        st.write(
            f"Local path {helpers.link_local(detail['local']['path'])} exists → {detail['local']['exists']}"
        )
else:
    st.info("Set an ep_id from another page (Upload/Episodes) to probe specific objects.")
