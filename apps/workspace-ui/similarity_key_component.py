"""Similarity Scores Color Key Component for Faces Review Page.

Uses centralized color definitions from similarity_badges.py for consistency.
"""

import streamlit as st

from similarity_badges import (
    SimilarityType,
    SIMILARITY_COLORS,
    get_similarity_key_data,
)


def render_similarity_scores_key():
    """Render collapsible similarity scores guide with color-coded legend for Faces Review."""
    with st.expander("📊 Similarity Scores Guide", expanded=False):
        # Get color data from centralized module
        key_data = get_similarity_key_data()

        # Build HTML for each score type
        items_html = ""
        for item in key_data:
            emoji_map = {
                SimilarityType.IDENTITY: "🔵",
                SimilarityType.CAST: "🟣",
                SimilarityType.TRACK: "🟠",
                SimilarityType.FRAME: "🟠",
                SimilarityType.CAST_TRACK: "🔷",
                SimilarityType.CLUSTER: "🟢",
            }
            emoji = emoji_map.get(item["type"], "⚪")
            items_html += f"""
            <div class="sim-item">
                <div class="sim-color" style="background: {item['color']};"></div>
                <div>
                    <div class="sim-label">{emoji} {item['label']}</div>
                    <div class="sim-desc">{item['description']}</div>
                    <div class="sim-threshold">{item['thresholds']}</div>
                </div>
            </div>
            """

        # Add Quality Score (separate from similarity)
        items_html += """
            <div class="sim-item">
                <div class="sim-color" style="background: #4CAF50;"></div>
                <div>
                    <div class="sim-label">🟢 Quality Score</div>
                    <div class="sim-desc">Detection + Sharpness + Area</div>
                    <div class="sim-threshold">Q: XX% badge on frames</div>
                </div>
            </div>
        """

        st.markdown(
            f"""
        <style>
        .sim-key {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin: 8px 0;
        }}
        .sim-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .sim-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .sim-label {{
            font-size: 13px;
            font-weight: 500;
        }}
        .sim-desc {{
            font-size: 11px;
            opacity: 0.7;
            margin-top: 2px;
        }}
        .sim-threshold {{
            font-size: 10px;
            opacity: 0.6;
            font-family: monospace;
        }}
        </style>

        <div class="sim-key">
            {items_html}
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
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
        """
        )
