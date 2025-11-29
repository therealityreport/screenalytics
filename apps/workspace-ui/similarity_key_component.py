"""Similarity Scores Color Key Component for Faces Review Page.

Uses centralized color definitions from similarity_badges.py for consistency.
Updated Nov 2024 with new metrics: Temporal, Ambiguity, Isolation, Confidence Trend
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

        st.markdown("### Core Metrics")

        # Build HTML for core metrics (first 5)
        core_html = ""
        for item in key_data[:5]:
            emoji = item.get("emoji", "⚪")
            core_html += f"""
            <div class="sim-item">
                <div class="sim-color" style="background: {item['color']};"></div>
                <div>
                    <div class="sim-label">{emoji} {item['label']}</div>
                    <div class="sim-desc">{item['description']}</div>
                    <div class="sim-threshold">{item['thresholds']}</div>
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
            align-items: flex-start;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            min-width: 280px;
            flex: 1;
        }}
        .sim-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            flex-shrink: 0;
            margin-top: 2px;
        }}
        .sim-label {{
            font-size: 13px;
            font-weight: 600;
        }}
        .sim-desc {{
            font-size: 11px;
            opacity: 0.8;
            margin-top: 2px;
        }}
        .sim-threshold {{
            font-size: 10px;
            opacity: 0.6;
            font-family: monospace;
            margin-top: 4px;
        }}
        .new-badge {{
            background: #E91E63;
            color: white;
            padding: 1px 4px;
            border-radius: 3px;
            font-size: 9px;
            font-weight: bold;
            margin-left: 4px;
        }}
        </style>

        <div class="sim-key">
            {core_html}
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("### New Metrics <span class='new-badge'>NOV 2024</span>", unsafe_allow_html=True)

        # Build HTML for new metrics (items 5-8)
        new_html = ""
        for item in key_data[5:9]:
            emoji = item.get("emoji", "⚪")
            new_html += f"""
            <div class="sim-item">
                <div class="sim-color" style="background: {item['color']};"></div>
                <div>
                    <div class="sim-label">{emoji} {item['label']}</div>
                    <div class="sim-desc">{item['description']}</div>
                    <div class="sim-threshold">{item['thresholds']}</div>
                </div>
            </div>
            """

        st.markdown(
            f"""
        <div class="sim-key">
            {new_html}
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Quality Score section
        st.markdown("### Quality Score")

        st.markdown(
            """
        <div class="sim-key">
            <div class="sim-item">
                <div class="sim-color" style="background: #4CAF50;"></div>
                <div>
                    <div class="sim-label">🟢 Quality Score (Q: XX%)</div>
                    <div class="sim-desc">Composite of Detection + Sharpness + Face Area. Hover for breakdown.</div>
                    <div class="sim-threshold">≥85%: High (green), ≥60%: Medium (amber), <60%: Low (red)</div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Enhanced indicators table
        st.markdown(
            """
### Enhanced Badge Features

| Badge | Enhancement |
|-------|-------------|
| **CAST: 68% (#1 of 5)** | Now shows rank among all suggestions |
| **CLU: 72% (58-89%)** | Shows min-max range for cluster cohesion |
| **TRK: 85% (3 excl)** | Shows frames excluded from centroid |
| **Q: 82%** | Hover to see Det/Sharp/Area breakdown |
| **OUTLIER: 45% ⚠️** | Only shown on frames below threshold, with severity |
| **AMB: Risky (3%)** | Ambiguity score - gap to 2nd best match |
| **ISO: Close** | Cluster isolation - merge candidate indicator |
| **↑ Improving** | Confidence trend over time |

### Quality Indicators

| Badge | Meaning |
|-------|---------|
| Q: 85%+ | High quality (sharp, complete face, good detection) |
| Q: 60-84% | Medium quality (acceptable for most uses) |
| Q: < 60% | Low quality (partial face, blurry, or low confidence) |
| ID: 75%+ | Strong identity match |
| ID: 70-74% | Good identity match |
| ID: < 70% | Needs review (threshold raised from 60%) |

### Frame Badges

- **★ BEST QUALITY** (green): Complete face, high quality, good ID match
- **⚠ BEST AVAILABLE** (orange): Partial/low-quality, best available option
- **Partial** (orange pill): Edge-clipped or incomplete face
- **OUTLIER: XX%** (orange/red): Frame differs significantly from track (only shown on outliers)

### Understanding New Metrics

**Temporal Consistency (TIME)**: Measures how consistent a person looks across different times in the episode. Low scores may indicate lighting changes, costume changes, or potential misassignments.

**Ambiguity Score (AMB)**: The gap between the best and second-best match. A small gap means the assignment is risky - could easily be either person. Look for "Risky" badges.

**Cluster Isolation (ISO)**: How far this cluster is from other clusters. "Close" clusters are merge candidates - they may actually be the same person.

**Confidence Trend (↑/→/↓)**: Tracks whether assignment confidence is improving or degrading as more data is added. A degrading trend is an early warning sign.

📚 **Full guide:** See `docs/similarity-scores-guide.md` for complete documentation.
        """
        )
