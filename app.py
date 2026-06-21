"""
CrimeLens AI - Main Application Entry Point
Karnataka State Police Intelligence Platform
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_loader import load_crime_data
from utils.prediction_utils import ensure_models
from utils.theme import apply_custom_css, render_sidebar_branding

st.set_page_config(
    page_title="CrimeLens AI | Karnataka State Police",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()
render_sidebar_branding()


@st.cache_resource(show_spinner="Initializing ML models...")
def initialize_models():
    """Ensure dataset and ML models exist and load on first launch."""
    df = load_crime_data()
    ensure_models(df)
    return True


initialize_models()

st.markdown(
    """
    <div class="main-header">
        <h1>🛡️ CrimeLens AI</h1>
        <p>AI-Powered Crime Intelligence & Predictive Policing Platform — Karnataka State Police</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        ### Welcome, Officer

        **CrimeLens AI** transforms disconnected crime records into actionable intelligence
        using advanced analytics, geospatial mapping, network analysis, and machine learning.

        Navigate using the **sidebar** to access intelligence modules:
        """
    )

    st.markdown(
        """
        | Module | Capability |
        |--------|------------|
        | 📊 **Dashboard** | Real-time KPIs, trends, and crime analytics |
        | 🗺️ **Crime Heatmap** | Interactive geospatial crime visualization |
        | 🔗 **Network Analysis** | Criminal association & organized crime detection |
        | 📈 **Predictive Analytics** | ML-powered risk forecasting & anomaly detection |
        """
    )

with col2:
    try:
        df = load_crime_data()
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Districts Covered", df["district"].nunique())
        st.metric("Crime Categories", df["crime_type"].nunique())
        st.metric("Date Range", f"{df['date'].min().strftime('%b %Y')} – {df['date'].max().strftime('%b %Y')}")
    except Exception as e:
        st.error(f"Data load error: {e}")

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#90a4ae; font-size:0.85rem;">
        CrimeLens AI v1.0 | Karnataka State Police | Intelligence-Led Policing Platform<br>
        <span style="color:#4fc3f7;">CONFIDENTIAL — For Official Use Only</span>
    </div>
    """,
    unsafe_allow_html=True,
)
