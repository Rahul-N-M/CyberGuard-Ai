"""CyberGuard-AI Streamlit entry point."""

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.data import DashboardDataError
from src.dashboard.views import (
    assets,
    overview,
    performance,
    prioritization,
)


st.set_page_config(
    page_title="CyberGuard-AI · Risk Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --cg-bg: #0b1220;
        --cg-panel: #111c2d;
        --cg-border: #31445e;
        --cg-cyan: #38bdf8;
        --text-color: #e6edf7;
        --secondary-text-color: #a9b8cc;
    }
    .stApp { background: var(--cg-bg); }
    [data-testid="stSidebar"] { background: #0e1929; border-right: 1px solid var(--cg-border); }
    [data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stSidebar"],
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stWidgetLabel"],
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    label,
    .stSelectbox,
    .stMultiSelect,
    .stTextInput,
    .stDateInput,
    .stSlider,
    [data-testid="stMetricLabel"] {
        color: #e6edf7 !important;
    }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    [data-testid="stWidgetLabel"],
    .metric-label,
    .metric-detail {
        color: #b5c4d8 !important;
    }
    h1, h2, h3 { letter-spacing: -0.025em; }
    h1 { margin-bottom: .2rem; }
    .eyebrow { color: var(--cg-cyan); font-size: .72rem; font-weight: 700; letter-spacing: .14em; margin-bottom: .45rem; }
    .metric-card { background: var(--cg-panel); border: 1px solid var(--cg-border); border-radius: 10px; padding: 1rem 1.05rem; min-height: 112px; }
    .metric-label { color: #9aabc0; font-size: .78rem; font-weight: 600; }
    .metric-value { color: #f1f5f9; font-size: 1.65rem; font-weight: 700; margin: .35rem 0 .2rem; }
    .metric-detail { font-size: .72rem; }
    div[data-testid="stMetric"] { background: var(--cg-panel); border: 1px solid var(--cg-border); border-radius: 10px; padding: .8rem 1rem; }
    div[data-testid="stDataFrame"] { border: 1px solid var(--cg-border); border-radius: 8px; }
    hr { border-color: var(--cg-border); }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🛡️ CyberGuard")
    st.caption("BUSINESS-AWARE RISK CONSOLE")
    page = st.selectbox(
        "Workspace",
        [
            "Executive remediation console",
            "Academic benchmarks & ML performance",
            "Vulnerability prioritization matrix",
            "Enterprise asset inventory",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Local verified artifacts · Zero live API latency")
    if st.button("Refresh cached data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

renderers = {
    "Executive remediation console": overview.render,
    "Academic benchmarks & ML performance": performance.render,
    "Vulnerability prioritization matrix": prioritization.render,
    "Enterprise asset inventory": assets.render,
}



try:
    renderers[page]()
except DashboardDataError as exc:
    st.error(f"Dashboard data unavailable: {exc}")
    st.caption("Check the repository's data/processed and data/raw files. No model training or live feed refresh was attempted.")
except (ImportError, RuntimeError, ValueError) as exc:
    st.error(f"Could not render this dashboard view: {exc}")
    st.caption("The underlying error is shown above; check dependencies and source data before retrying.")
