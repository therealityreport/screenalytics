"""Similarity Scores Color Key Component for Faces Review Page.

Insert this code block in apps/workspace-ui/pages/3_Faces_Review.py
after line 21 (after st.caption with backend info).
"""

import streamlit as st


def render_similarity_scores_key():
    """Render collapsible similarity scores guide with color-coded legend for Faces Review."""
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
                <div class="sim-color" style="background: #2196F3;"></div>
                <div>
                    <div class="sim-label">🔵 Identity Similarity</div>
                    <div class="sim-desc">Frame → Track centroid</div>
                    <div class="sim-threshold">ID: XX% badge on frames</div>
                </div>
            </div>

            <div class="sim-item">
                <div class="sim-color" style="background: #4CAF50;"></div>
                <div>
                    <div class="sim-label">🟢 Quality Score</div>
                    <div class="sim-desc">Detection + Sharpness + Area</div>
                    <div class="sim-threshold">Q: XX% badge on frames</div>
                </div>
            </div>

            <div class="sim-item">
                <div class="sim-color" style="background: #9C27B0;"></div>
                <div>
                    <div class="sim-label">🟣 Cast Similarity</div>
                    <div class="sim-desc">Cluster → Cast member</div>
                    <div class="sim-threshold">≥0.68 to auto-assign</div>
                </div>
            </div>

            <div class="sim-item">
                <div class="sim-color" style="background: #FF9800;"></div>
                <div>
                    <div class="sim-label">🟠 Track Similarity</div>
                    <div class="sim-desc">Appearance gate (logs)</div>
                    <div class="sim-threshold">≥0.75 hard, ≥0.82 soft</div>
                </div>
            </div>

            <div class="sim-item">
                <div class="sim-color" style="background: #4CAF50;"></div>
                <div>
                    <div class="sim-label">🟢 Cluster Similarity</div>
                    <div class="sim-desc">Face grouping (DBSCAN)</div>
                    <div class="sim-threshold">≥0.35 for same cluster</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        **Quality Indicators:**

        | Badge | Meaning |
        |-------|---------|
        | Q: 85%+ | High quality (sharp, complete face, good detection) |
        | Q: 60-84% | Medium quality (acceptable for most uses) |
        | Q: < 60% | Low quality (partial face, blurry, or low confidence) |
        | ID: 75%+ | Strong identity match to track |
        | ID: 60-74% | Good identity match |
        | ID: < 60% | Weak identity match (may be wrong person) |

        **Badges on frames:**
        - ★ BEST QUALITY (green): Complete face, high quality, good ID match
        - ⚠ BEST AVAILABLE (orange): Partial/low-quality, best available option
        - Partial (orange pill): Edge-clipped or incomplete face

        📚 **Full guide:** See `docs/similarity-scores-guide.md` for complete documentation.
        """)


# INTEGRATION INSTRUCTIONS:
#
# In apps/workspace-ui/pages/3_Faces_Review.py, add this after line 21:
#
# st.title("Faces & Tracks Review")
# st.caption(f"Backend: {cfg['backend']} · Bucket: {cfg.get('bucket') or 'n/a'}")
#
# # NEW: Add similarity scores key
# from similarity_key_component import render_similarity_scores_key
# render_similarity_scores_key()
#
# # Inject thumbnail CSS
# helpers.inject_thumb_css()
# ...
#
# OR copy the function content directly (already done in 3_Faces_Review.py)
