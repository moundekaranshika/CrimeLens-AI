"""Page 2: Crime Heatmap."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_loader import filter_data, load_crime_data
from utils.map_utils import build_crime_map, get_map_statistics
from utils.theme import apply_custom_css, render_page_header, render_sidebar_branding, render_sidebar_filters

st.set_page_config(page_title="Crime Heatmap | CrimeLens AI", page_icon="🗺️", layout="wide")
apply_custom_css()
render_sidebar_branding()

df = load_crime_data()
districts, crime_types, date_range = render_sidebar_filters(df)
filtered = filter_data(df, districts, crime_types, date_range)

render_page_header(
    "🗺️ Crime Heatmap",
    "Interactive geospatial visualization of crime incidents across Karnataka",
)

# Map layer controls
st.sidebar.markdown("### 🗺️ Map Layers")
show_markers = st.sidebar.checkbox("Crime Markers", value=True)
show_heatmap = st.sidebar.checkbox("Heatmap Layer", value=True)
show_clusters = st.sidebar.checkbox("Hotspot Clusters", value=True)

# Statistics row
stats = get_map_statistics(filtered)
s1, s2, s3, s4 = st.columns(4)
s1.metric("Incident Count", f"{stats['incident_count']:,}")
s2.metric("Top Crime Type", stats["top_crime_type"])
s3.metric("Top District", stats["top_district"])
s4.metric("Unique Locations", stats["unique_locations"])

st.markdown("<br>", unsafe_allow_html=True)

if len(filtered):
    crime_map = build_crime_map(
        filtered,
        show_markers=show_markers,
        show_heatmap=show_heatmap,
        show_clusters=show_clusters,
    )
    map_data = st_folium(crime_map, width=None, height=600, returned_objects=["last_active_drawing"])

    # Incident details table
    with st.expander("📋 Incident Details", expanded=False):
        display_cols = ["crime_id", "date", "crime_type", "district", "location", "status", "risk_zone"]
        st.dataframe(
            filtered[display_cols].sort_values("date", ascending=False).head(100),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.warning("No incidents match the selected filters. Adjust filters to view the map.")

# District crime density summary
st.subheader("District Crime Density")
if len(filtered):
    density = (
        filtered.groupby("district")
        .agg(
            incidents=("crime_id", "count"),
            avg_risk=("risk_score", "mean"),
            arrests=("arrested", "sum"),
        )
        .reset_index()
        .sort_values("incidents", ascending=False)
    )
    density["avg_risk"] = density["avg_risk"].round(3)
    st.dataframe(density, use_container_width=True, hide_index=True)
