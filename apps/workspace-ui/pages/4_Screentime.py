from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
import streamlit as st

PAGE_PATH = Path(__file__).resolve()
WORKSPACE_DIR = PAGE_PATH.parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

import ui_helpers as helpers  # noqa: E402

cfg = helpers.init_page("Screentime")
st.title("Screentime")


def _require_episode() -> str:
    ep_id = helpers.get_ep_id()
    if ep_id:
        return ep_id
    try:
        payload = helpers.api_get("/episodes")
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{cfg['api_base']}/episodes", exc))
        st.stop()
    episodes = payload.get("episodes", [])
    if not episodes:
        st.info("No episodes yet.")
        st.stop()
    option_lookup = {ep["ep_id"]: ep for ep in episodes}
    selection = st.selectbox(
        "Episode",
        list(option_lookup.keys()),
        format_func=lambda eid: f"{eid} ({option_lookup[eid]['show_slug']})",
    )
    if st.button("Load episode", use_container_width=True):
        helpers.set_ep_id(selection)
    st.stop()


ep_id = _require_episode()
helpers.set_ep_id(ep_id)

analytics_dir = helpers.DATA_ROOT / "analytics" / ep_id
json_path = analytics_dir / "screentime.json"
csv_path = analytics_dir / "screentime.csv"

if st.button("Compute screentime", use_container_width=True):
    try:
        resp = helpers.api_post("/jobs/screentime", {"ep_id": ep_id})
    except requests.RequestException as exc:
        st.error(helpers.describe_error(f"{cfg['api_base']}/jobs/screentime", exc))
    else:
        st.success(resp.get("status", "screentime job queued"))

if json_path.exists():
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    rows = data.get("people") or data
    if isinstance(rows, list) and rows:
        st.subheader("Per-person screentime")
        helpers.ds(rows)
        chart_data = {row.get("person") or row.get("cast") or f"row-{idx}": row.get("seconds", 0) for idx, row in enumerate(rows)}
        st.bar_chart(chart_data)
    else:
        st.info("screentime.json present but empty.")
else:
    st.info("No screentime analytics yet.")

st.subheader("Artifacts")
st.write(f"JSON → {helpers.link_local(json_path)}")
st.write(f"CSV → {helpers.link_local(csv_path)}")
