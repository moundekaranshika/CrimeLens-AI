"""Geospatial mapping utilities for CrimeLens AI."""

from __future__ import annotations

from typing import Optional

import folium
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster

# Law enforcement color palette
CRIME_COLORS: dict[str, str] = {
    "Theft": "#3498db",
    "Assault": "#e74c3c",
    "Cybercrime": "#9b59b6",
    "Fraud": "#f39c12",
    "Robbery": "#c0392b",
    "Murder": "#8b0000",
    "Drug Offense": "#27ae60",
    "Domestic Violence": "#e67e22",
    "Burglary": "#1abc9c",
    "Kidnapping": "#2c3e50",
}

KARNATAKA_CENTER = [14.5, 76.5]
DEFAULT_ZOOM = 7


def create_base_map(center: Optional[list[float]] = None, zoom: int = DEFAULT_ZOOM) -> folium.Map:
    """Create a dark law-enforcement styled base Folium map."""
    loc = center or KARNATAKA_CENTER
    m = folium.Map(
        location=loc,
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)
    folium.TileLayer("CartoDB positron", name="Light Map").add_to(m)
    return m


def add_crime_markers(m: folium.Map, df: pd.DataFrame, max_markers: int = 500) -> folium.Map:
    """Add individual crime incident markers to the map."""
    if df.empty:
        return m

    sample = df.head(max_markers)
    marker_group = folium.FeatureGroup(name="Crime Markers", show=True)

    for _, row in sample.iterrows():
        color = CRIME_COLORS.get(row.get("crime_type", ""), "#3498db")
        popup_html = f"""
        <div style="font-family:Arial; min-width:200px;">
            <b style="color:#1a237e;">{row.get('crime_id', 'N/A')}</b><br>
            <b>Type:</b> {row.get('crime_type', 'N/A')}<br>
            <b>District:</b> {row.get('district', 'N/A')}<br>
            <b>Date:</b> {row.get('date', 'N/A')}<br>
            <b>Location:</b> {row.get('location', 'N/A')}<br>
            <b>Status:</b> {row.get('status', 'N/A')}<br>
            <b>Risk:</b> {row.get('risk_zone', 'N/A')}
        </div>
        """
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row.get('crime_type')} - {row.get('district')}",
        ).add_to(marker_group)

    marker_group.add_to(m)
    return m


def add_heatmap_layer(m: folium.Map, df: pd.DataFrame) -> folium.Map:
    """Add crime density heatmap layer."""
    if df.empty:
        return m

    heat_data = df[["latitude", "longitude"]].dropna().values.tolist()
    if heat_data:
        HeatMap(
            heat_data,
            name="Crime Heatmap",
            min_opacity=0.3,
            max_zoom=13,
            radius=18,
            blur=15,
            gradient={0.2: "blue", 0.4: "lime", 0.6: "orange", 1: "red"},
        ).add_to(m)
    return m


def add_hotspot_clusters(m: folium.Map, df: pd.DataFrame, max_points: int = 800) -> folium.Map:
    """Add marker cluster layer for hotspot visualization."""
    if df.empty:
        return m

    cluster = MarkerCluster(name="Hotspot Clusters").add_to(m)
    sample = df.head(max_points)

    for _, row in sample.iterrows():
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=f"{row.get('crime_type')} | {row.get('district')} | {row.get('date')}",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(cluster)

    return m


def build_crime_map(
    df: pd.DataFrame,
    show_markers: bool = True,
    show_heatmap: bool = True,
    show_clusters: bool = True,
) -> folium.Map:
    """Build complete interactive crime map with all layers."""
    if df.empty:
        m = create_base_map()
        folium.Marker(
            KARNATAKA_CENTER,
            popup="No data for selected filters",
            icon=folium.Icon(color="gray"),
        ).add_to(m)
        return m

    center_lat = df["latitude"].median()
    center_lon = df["longitude"].median()
    m = create_base_map(center=[center_lat, center_lon], zoom=8)

    if show_heatmap:
        add_heatmap_layer(m, df)
    if show_clusters:
        add_hotspot_clusters(m, df)
    if show_markers:
        add_crime_markers(m, df)

    folium.LayerControl(collapsed=False).add_to(m)

    # Add district legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 9999;
                background: rgba(26,35,126,0.9); color: white; padding: 12px;
                border-radius: 8px; font-size: 12px; border: 1px solid #4fc3f7;">
        <b>CrimeLens AI Map</b><br>
        <span style="color:#ff5252;">●</span> High Density Areas<br>
        <span style="color:#4fc3f7;">●</span> Incident Markers<br>
        <span style="color:#ffd740;">●</span> Hotspot Clusters
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def get_map_statistics(df: pd.DataFrame) -> dict[str, int | str]:
    """Compute map sidebar statistics."""
    if df.empty:
        return {
            "incident_count": 0,
            "top_crime_type": "N/A",
            "top_district": "N/A",
            "unique_locations": 0,
        }

    return {
        "incident_count": len(df),
        "top_crime_type": df["crime_type"].mode().iloc[0] if len(df) else "N/A",
        "top_district": df["district"].mode().iloc[0] if len(df) else "N/A",
        "unique_locations": int(df["location"].nunique()),
    }
