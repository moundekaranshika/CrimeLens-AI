"""Page 1: Crime Intelligence Dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_loader import filter_data, get_kpi_metrics, load_crime_data
from utils.theme import (
    PLOTLY_COLORS,
    PLOTLY_TEMPLATE,
    apply_custom_css,
    render_kpi_card,
    render_page_header,
    render_sidebar_branding,
    render_sidebar_filters,
)

st.set_page_config(page_title="Dashboard | CrimeLens AI", page_icon="📊", layout="wide")
apply_custom_css()
render_sidebar_branding()

df = load_crime_data()
districts, crime_types, date_range = render_sidebar_filters(df)
filtered = filter_data(df, districts, crime_types, date_range)
kpis = get_kpi_metrics(filtered)

render_page_header(
    "📊 Crime Intelligence Dashboard",
    "Real-time analytics and key performance indicators for Karnataka crime data",
)

# KPI Row
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(render_kpi_card(f"{kpis['total_crimes']:,}", "Total Crimes"), unsafe_allow_html=True)
with k2:
    st.markdown(render_kpi_card(kpis["active_cases"], "Active Cases", alert=True), unsafe_allow_html=True)
with k3:
    st.markdown(render_kpi_card(kpis["arrests_made"], "Arrests Made"), unsafe_allow_html=True)
with k4:
    st.markdown(render_kpi_card(kpis["repeat_offenders"], "Repeat Offenders", alert=True), unsafe_allow_html=True)
with k5:
    st.markdown(
        render_kpi_card(kpis["high_risk_district"][:12], f"High Risk ({kpis['high_risk_count']})"),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Charts Row 1
c1, c2 = st.columns(2)

with c1:
    st.subheader("Crime by District")
    if len(filtered):
        district_counts = filtered["district"].value_counts().reset_index()
        district_counts.columns = ["district", "count"]
        fig = px.bar(
            district_counts.head(15),
            x="count",
            y="district",
            orientation="h",
            color="count",
            color_continuous_scale=["#1a237e", "#4fc3f7", "#e53935"],
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(
            showlegend=False,
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for selected filters.")

with c2:
    st.subheader("Crime by Type")
    if len(filtered):
        type_counts = filtered["crime_type"].value_counts().reset_index()
        type_counts.columns = ["crime_type", "count"]
        fig = px.pie(
            type_counts,
            values="count",
            names="crime_type",
            color_discrete_sequence=PLOTLY_COLORS,
            template=PLOTLY_TEMPLATE,
            hole=0.4,
        )
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for selected filters.")

# Charts Row 2
c3, c4 = st.columns(2)

with c3:
    st.subheader("Monthly Crime Trend")
    if len(filtered):
        monthly = filtered.groupby("month_year").size().reset_index(name="count")
        monthly = monthly.sort_values("month_year")
        fig = px.area(
            monthly,
            x="month_year",
            y="count",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=["#4fc3f7"],
        )
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Month",
            yaxis_title="Incidents",
        )
        fig.update_traces(fillcolor="rgba(79,195,247,0.3)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for selected filters.")

with c4:
    st.subheader("Arrest Rate by District")
    if len(filtered):
        arrest_data = (
            filtered.groupby("district")
            .agg(total=("crime_id", "count"), arrested=("arrested", "sum"))
            .reset_index()
        )
        arrest_data["arrest_rate"] = (arrest_data["arrested"] / arrest_data["total"] * 100).round(1)
        arrest_data = arrest_data.sort_values("arrest_rate", ascending=False).head(12)
        fig = px.bar(
            arrest_data,
            x="district",
            y="arrest_rate",
            color="arrest_rate",
            color_continuous_scale=["#1a237e", "#4fc3f7", "#66bb6a"],
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(
            showlegend=False,
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-45,
            yaxis_title="Arrest Rate (%)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for selected filters.")

# Charts Row 3
c5, c6 = st.columns(2)

with c5:
    st.subheader("Repeat Offender Distribution")
    if len(filtered):
        repeat_counts = (
            filtered[filtered["repeat_offender"] == 1]
            .groupby("suspect_id")
            .size()
            .reset_index(name="crimes")
        )
        if len(repeat_counts):
            bins = [1, 2, 3, 5, 10, repeat_counts["crimes"].max() + 1]
            labels = ["2", "3", "4-5", "6-10", "10+"]
            repeat_counts["bucket"] = pd.cut(
                repeat_counts["crimes"], bins=bins, labels=labels, right=False
            )
            bucket_counts = repeat_counts["bucket"].value_counts().reset_index()
            bucket_counts.columns = ["offenses", "offenders"]
            fig = px.bar(
                bucket_counts,
                x="offenses",
                y="offenders",
                color_discrete_sequence=["#e53935"],
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Number of Offenses",
                yaxis_title="Offenders",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No repeat offenders in filtered data.")
    else:
        st.info("No data for selected filters.")

with c6:
    st.subheader("Top Crime Hotspots")
    if len(filtered):
        hotspots = (
            filtered.groupby(["district", "location"])
            .size()
            .reset_index(name="incidents")
            .sort_values("incidents", ascending=False)
            .head(10)
        )
        hotspots["label"] = hotspots["district"] + " — " + hotspots["location"]
        fig = go.Figure(go.Bar(
            x=hotspots["incidents"],
            y=hotspots["label"],
            orientation="h",
            marker=dict(
                color=hotspots["incidents"],
                colorscale=[[0, "#1a237e"], [0.5, "#4fc3f7"], [1, "#e53935"]],
            ),
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=350,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed"),
            xaxis_title="Incidents",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for selected filters.")

# Intelligence Report Export
st.markdown("---")
st.subheader("📄 Intelligence Report")
if st.button("📥 Generate PDF Report", type="primary"):
    from utils.network_utils import (
        build_crime_network,
        compute_centrality,
        detect_repeat_offenders,
        find_organized_crime_clusters,
        generate_network_insights,
    )
    from utils.prediction_utils import detect_anomalies, forecast_district_risk
    from utils.report_generator import generate_intelligence_report

    with st.spinner("Generating report..."):
        G = build_crime_network(filtered)
        repeat = detect_repeat_offenders(filtered)
        centrality = compute_centrality(G)
        clusters = find_organized_crime_clusters(G)
        insights = generate_network_insights(G, filtered, centrality, repeat, clusters)
        forecast = forecast_district_risk(filtered)
        anomalies = detect_anomalies(filtered)
        recommendations = [
            "Increase patrol density in top 3 high-risk districts during peak hours.",
            "Deploy cybercrime task force to districts with rising fraud incidents.",
            "Prioritize repeat offender surveillance using network centrality scores.",
            "Coordinate inter-district operations for organized crime clusters.",
            "Review anomaly alerts weekly for emerging crime trend response.",
        ]
        pdf_bytes = generate_intelligence_report(
            filtered, kpis, insights, forecast, anomalies, recommendations
        )
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"CrimeLens_Report_{filtered['date'].max().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
        )
