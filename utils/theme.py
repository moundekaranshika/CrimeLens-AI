"""Shared UI theme and styling utilities for CrimeLens AI."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
LOGO_PATH = BASE_DIR / "assets" / "logo.png"

# Police Intelligence Dashboard palette
COLORS = {
    "navy": "#0d1b2a",
    "dark_navy": "#1a237e",
    "white": "#ffffff",
    "light_blue": "#4fc3f7",
    "red_alert": "#e53935",
    "card_bg": "#1b2838",
    "text_muted": "#90a4ae",
}

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_COLORS = ["#4fc3f7", "#e53935", "#ffd740", "#66bb6a", "#ab47bc", "#26c6da", "#ff7043"]


def apply_custom_css() -> None:
    """Inject custom CSS for police intelligence dashboard theme."""
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            html, body, [class*="css"] {{
                font-family: 'Inter', sans-serif;
            }}

            .stApp {{
                background: linear-gradient(135deg, {COLORS['navy']} 0%, #0a1628 50%, {COLORS['dark_navy']} 100%);
            }}

            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, {COLORS['dark_navy']} 0%, {COLORS['navy']} 100%);
                border-right: 1px solid {COLORS['light_blue']}33;
            }}

            [data-testid="stSidebar"] .stMarkdown h1,
            [data-testid="stSidebar"] .stMarkdown h2,
            [data-testid="stSidebar"] .stMarkdown h3 {{
                color: {COLORS['white']};
            }}

            .main-header {{
                background: linear-gradient(90deg, {COLORS['dark_navy']}, {COLORS['navy']});
                padding: 1.5rem 2rem;
                border-radius: 12px;
                border: 1px solid {COLORS['light_blue']}44;
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }}

            .main-header h1 {{
                color: {COLORS['white']};
                margin: 0;
                font-size: 2rem;
                font-weight: 700;
            }}

            .main-header p {{
                color: {COLORS['light_blue']};
                margin: 0.3rem 0 0 0;
                font-size: 0.95rem;
            }}

            .kpi-card {{
                background: {COLORS['card_bg']};
                border: 1px solid {COLORS['light_blue']}33;
                border-radius: 10px;
                padding: 1.2rem;
                text-align: center;
                box-shadow: 0 2px 12px rgba(0,0,0,0.2);
                transition: transform 0.2s;
            }}

            .kpi-card:hover {{
                transform: translateY(-2px);
                border-color: {COLORS['light_blue']}66;
            }}

            .kpi-value {{
                font-size: 2rem;
                font-weight: 700;
                color: {COLORS['light_blue']};
                margin: 0;
            }}

            .kpi-value.alert {{
                color: {COLORS['red_alert']};
            }}

            .kpi-label {{
                font-size: 0.85rem;
                color: {COLORS['text_muted']};
                margin-top: 0.3rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}

            .insight-box {{
                background: {COLORS['card_bg']};
                border-left: 4px solid {COLORS['light_blue']};
                padding: 1rem 1.2rem;
                border-radius: 0 8px 8px 0;
                margin: 0.5rem 0;
                color: {COLORS['white']};
            }}

            .alert-box {{
                background: rgba(229, 57, 53, 0.15);
                border-left: 4px solid {COLORS['red_alert']};
                padding: 1rem 1.2rem;
                border-radius: 0 8px 8px 0;
                margin: 0.5rem 0;
                color: {COLORS['white']};
            }}

            div[data-testid="stMetric"] {{
                background: {COLORS['card_bg']};
                border: 1px solid {COLORS['light_blue']}33;
                border-radius: 10px;
                padding: 1rem;
            }}

            div[data-testid="stMetric"] label {{
                color: {COLORS['text_muted']} !important;
            }}

            div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
                color: {COLORS['light_blue']} !important;
            }}

            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
            }}

            .stTabs [data-baseweb="tab"] {{
                background: {COLORS['card_bg']};
                border-radius: 8px;
                color: {COLORS['text_muted']};
                border: 1px solid transparent;
            }}

            .stTabs [aria-selected="true"] {{
                background: {COLORS['dark_navy']} !important;
                color: {COLORS['light_blue']} !important;
                border: 1px solid {COLORS['light_blue']}44 !important;
            }}

            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str = "") -> None:
    """Render consistent page header."""
    st.markdown(
        f"""
        <div class="main-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(value: str | int | float, label: str, alert: bool = False) -> str:
    """Return HTML for a KPI card."""
    value_class = "kpi-value alert" if alert else "kpi-value"
    return f"""
    <div class="kpi-card">
        <p class="{value_class}">{value}</p>
        <p class="kpi-label">{label}</p>
    </div>
    """


def render_sidebar_filters(df) -> tuple:
    """Render common sidebar filters and return filter values."""
    st.sidebar.markdown("### 🔍 Filters")

    districts = sorted(df["district"].unique().tolist())
    selected_districts = st.sidebar.multiselect(
        "District",
        options=districts,
        default=districts,
        key="filter_districts",
    )

    crime_types = sorted(df["crime_type"].unique().tolist())
    selected_crimes = st.sidebar.multiselect(
        "Crime Type",
        options=crime_types,
        default=crime_types,
        key="filter_crime_types",
    )

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="filter_date_range",
    )

    return selected_districts, selected_crimes, date_range


def render_sidebar_branding() -> None:
    """Render sidebar logo and branding."""
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width=120)
    st.sidebar.markdown(
        """
        ### 🛡️ CrimeLens AI
        **Karnataka State Police**
        Intelligence Platform
        ---
        """
    )
