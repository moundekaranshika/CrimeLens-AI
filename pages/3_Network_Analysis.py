"""Page 3: Criminal Network Analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_loader import filter_data, load_crime_data
from utils.network_utils import (
    build_crime_network,
    compute_centrality,
    detect_communities,
    detect_repeat_offenders,
    find_organized_crime_clusters,
    generate_network_insights,
    render_pyvis_network,
)
from utils.theme import PLOTLY_TEMPLATE, apply_custom_css, render_page_header, render_sidebar_branding, render_sidebar_filters

st.set_page_config(page_title="Network Analysis | CrimeLens AI", page_icon="🔗", layout="wide")
apply_custom_css()
render_sidebar_branding()

df = load_crime_data()
districts, crime_types, date_range = render_sidebar_filters(df)
filtered = filter_data(df, districts, crime_types, date_range)

render_page_header(
    "🔗 Criminal Network Analysis",
    "Relationship intelligence — suspects, victims, locations, and police stations",
)

st.sidebar.markdown("### ⚙️ Network Settings")
max_edges = st.sidebar.slider("Max Records to Analyze", 100, 2000, 800, 100)

if len(filtered) == 0:
    st.warning("No data for selected filters.")
    st.stop()

# Build network
G = build_crime_network(filtered, max_edges=max_edges)
repeat_offenders = detect_repeat_offenders(filtered)
centrality_df = compute_centrality(G)
clusters = find_organized_crime_clusters(G)
communities = detect_communities(G)
insights = generate_network_insights(G, filtered, centrality_df, repeat_offenders, clusters)

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Network Nodes", G.number_of_nodes())
m2.metric("Associations", G.number_of_edges())
m3.metric("Repeat Offenders", len(repeat_offenders))
m4.metric("Crime Clusters", len(clusters))

# Insights
st.subheader("🔍 Automated Intelligence Insights")
for insight in insights:
    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Network Graph", "Centrality Scores", "Repeat Offenders", "Organized Crime"])

with tab1:
    st.markdown("**Interactive Criminal Association Network** — Red nodes indicate repeat offenders.")
    if G.number_of_nodes() > 0:
        html_path = render_pyvis_network(G)
        with open(html_path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=620, scrolling=True)
    else:
        st.info("Insufficient data to build network.")

    # Legend
    st.markdown(
        """
        | Node Color | Entity Type |
        |------------|-------------|
        | 🔴 Red | Repeat Offender (Suspect) |
        | 🟠 Orange-Red | Suspect |
        | 🔵 Blue | Victim |
        | 🟢 Green | Location |
        | 🟣 Purple | Police Station |
        """
    )

with tab2:
    st.subheader("Node Centrality Metrics")
    if len(centrality_df):
        suspects = centrality_df[centrality_df["node_type"] == "suspect"].head(20)
        st.dataframe(suspects, use_container_width=True, hide_index=True)

        fig = px.scatter(
            suspects,
            x="degree",
            y="betweenness",
            size="degree",
            color="repeat_offender",
            hover_data=["label"],
            color_discrete_map={True: "#e53935", False: "#4fc3f7"},
            template=PLOTLY_TEMPLATE,
            title="Suspect Centrality Distribution",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Most Connected Criminals")
        top5 = suspects.head(5)
        for _, row in top5.iterrows():
            badge = "🔴 REPEAT" if row["repeat_offender"] else ""
            st.markdown(
                f"**{row['label']}** {badge} — Degree: `{row['degree']}` | "
                f"Betweenness: `{row['betweenness']}`"
            )
    else:
        st.info("No centrality data available.")

with tab3:
    st.subheader("Repeat Offender Detection")
    if len(repeat_offenders):
        st.dataframe(repeat_offenders.head(25), use_container_width=True, hide_index=True)

        fig = px.bar(
            repeat_offenders.head(15),
            x="suspect_id",
            y="crime_count",
            color="crime_count",
            color_continuous_scale=["#4fc3f7", "#e53935"],
            template=PLOTLY_TEMPLATE,
            title="Top Repeat Offenders by Incident Count",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No repeat offenders detected in filtered data.")

with tab4:
    st.subheader("Organized Crime Cluster Discovery")
    if clusters:
        for i, cluster in enumerate(clusters[:5], 1):
            with st.expander(f"Cluster {i} — {cluster['size']} members (density: {cluster['density']})"):
                st.write(f"**Members:** {', '.join(cluster['members'][:10])}")
                st.write(f"**Average Degree:** {cluster['avg_degree']}")
    else:
        st.info("No organized crime clusters detected with current parameters.")

    # Community detection summary
    if communities:
        community_df = pd.DataFrame([
            {"node": k, "community": v} for k, v in communities.items()
        ])
        comm_counts = community_df["community"].value_counts().reset_index()
        comm_counts.columns = ["community", "members"]
        st.subheader("Community Detection Summary")
        st.dataframe(comm_counts.head(10), use_container_width=True, hide_index=True)
