"""Similarity Scores Color Key Component for Episode Detail Page.

Insert this code block in apps/workspace-ui/pages/2_Episode_Detail.py
after line 144 (after flash_message handling).
"""

import streamlit as st


def render_similarity_scores_key():
    """Render collapsible similarity scores guide with color-coded legend."""
    with st.expander("📊 Similarity Scores Guide", expanded=False):
        st.markdown("""
        <style>
        .sim-key {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin: 8px 0;
        }
        .sim-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .sim-color {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .sim-label {
            font-size: 13px;
            font-weight: 500;
        }
        .sim-desc {
            font-size: 11px;
            opacity: 0.7;
            margin-top: 2px;
        }
        .sim-threshold {
            font-size: 10px;
            opacity: 0.6;
            font-family: monospace;
        }
        </style>

        <div class="sim-key">
            <div class="sim-item">
                <div class="sim-color" style="background: #9C27B0;"></div>
                <div>
                    <div class="sim-label">🟣 Cast Similarity</div>
                    <div class="sim-desc">Cluster → Cast member</div>
                    <div class="sim-threshold">≥0.68 to auto-assign</div>
                </div>
            </div>

            <div class="sim-item">
                <div class="sim-color" style="background: #2196F3;"></div>
                <div>
                    <div class="sim-label">🔵 Identity Similarity</div>
                    <div class="sim-desc">Frame → Track centroid</div>
                    <div class="sim-threshold">≥0.60 for rep selection</div>
                </div>
            </div>

            <div class="sim-item">
                <div class="sim-color" style="background: #4CAF50;"></div>
                <div>
                    <div class="sim-label">🟢 Cluster Similarity</div>
                    <div class="sim-desc">Face → Face grouping</div>
                    <div class="sim-threshold">≥0.35 for same cluster</div>
                </div>
            </div>

            <div class="sim-item">
                <div class="sim-color" style="background: #FF9800;"></div>
                <div>
                    <div class="sim-label">🟠 Track Similarity</div>
                    <div class="sim-desc">Frame → Track prototype</div>
                    <div class="sim-threshold">≥0.75 hard, ≥0.82 soft</div>
                </div>
            </div>

            <div class="sim-item">
                <div class="sim-color" style="background: linear-gradient(90deg, #4CAF50 0%, #FFC107 50%, #F44336 100%);"></div>
                <div>
                    <div class="sim-label">🟢🟡🔴 Assignment Confidence</div>
                    <div class="sim-desc">Auto-assign eligibility</div>
                    <div class="sim-threshold">Margin ≥0.10 required</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        **All scores use cosine similarity (0.0-1.0 scale):**

        | Range | Meaning |
        |-------|---------|
        | 0.9+ | Very high similarity (same person, similar conditions) |
        | 0.7-0.9 | High similarity (same person, varying conditions) |
        | 0.5-0.7 | Moderate similarity (possibly same person) |
        | < 0.5 | Low similarity (likely different people) |

        📚 **Full guide:** See `docs/similarity-scores-guide.md` for complete documentation.

        **Where these scores appear:**
        - 🟣 **Cast Similarity**: Cluster assignment section, identity linking
        - 🔵 **Identity Similarity**: Faces Review page ("ID: 87%" badges)
        - 🟢 **Cluster Similarity**: Clustering phase (DBSCAN grouping)
        - 🟠 **Track Similarity**: Detect/track logs (appearance gate splits)
        - 🟢🟡🔴 **Assignment Confidence**: Auto-assign decisions
        """)


# INTEGRATION INSTRUCTIONS:
#
# In apps/workspace-ui/pages/2_Episode_Detail.py, add this after line 144:
#
# st.title("Episode Detail")
# flash_message = st.session_state.pop("episode_detail_flash", None)
# if flash_message:
#     st.success(flash_message)
#
# # NEW: Add similarity scores key
# from similarity_key_component import render_similarity_scores_key
# render_similarity_scores_key()
#
# if "detector" in st.session_state:
#     del st.session_state["detector"]
# ...
#
# OR copy the function directly into the file and call it inline:
#
# st.title("Episode Detail")
# flash_message = st.session_state.pop("episode_detail_flash", None)
# if flash_message:
#     st.success(flash_message)
#
# # Similarity Scores Color Key
# with st.expander("📊 Similarity Scores Guide", expanded=False):
#     st.markdown("""...""")  # Copy content from render_similarity_scores_key()
