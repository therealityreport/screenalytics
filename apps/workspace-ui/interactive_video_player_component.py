from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT_PATH = Path(__file__).parent / "components" / "interactive_video_player"

interactive_video_player = components.declare_component(
    "interactive_video_player",
    path=str(_COMPONENT_PATH),
)
