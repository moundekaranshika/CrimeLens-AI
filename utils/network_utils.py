"""Criminal network analysis utilities using NetworkX and PyVis."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from pyvis.network import Network


NODE_COLORS = {
    "suspect": "#e74c3c",
    "victim": "#3498db",
    "location": "#2ecc71",
    "police_station": "#9b59b6",
}


def build_crime_network(df: pd.DataFrame, max_edges: int = 2000) -> nx.Graph:
    """
    Build a criminal association network graph.

    Nodes: suspects, victims, locations, police stations
    Edges: co-occurrence in crimes, repeat interactions
    """
    G = nx.Graph()

    if df.empty:
        return G

    sample = df.head(max_edges)

    for _, row in sample.iterrows():
        suspect = f"S:{row['suspect_id']}"
        victim = f"V:{row['victim_id']}"
        location = f"L:{row.get('location', 'Unknown')}"
        station = f"P:{row.get('police_station', 'Unknown')}"

        G.add_node(suspect, node_type="suspect", label=str(row["suspect_id"]))
        G.add_node(victim, node_type="victim", label=str(row["victim_id"]))
        G.add_node(location, node_type="location", label=str(row.get("location", "Unknown")))
        G.add_node(station, node_type="police_station", label=str(row.get("police_station", "Unknown")))

        G.add_edge(suspect, victim, relation="crime_link", weight=1)
        G.add_edge(suspect, location, relation="location_link", weight=1)
        G.add_edge(victim, location, relation="incident_at", weight=1)
        G.add_edge(station, location, relation="jurisdiction", weight=1)

        if row.get("repeat_offender", 0) == 1:
            G.nodes[suspect]["repeat_offender"] = True

    # Aggregate edge weights for duplicate connections
    edge_weights: dict[tuple, int] = {}
    for u, v, data in G.edges(data=True):
        key = tuple(sorted([u, v]))
        edge_weights[key] = edge_weights.get(key, 0) + data.get("weight", 1)

    G_clean = nx.Graph()
    for node, attrs in G.nodes(data=True):
        G_clean.add_node(node, **attrs)
    for (u, v), w in edge_weights.items():
        G_clean.add_edge(u, v, weight=w)

    return G_clean


def detect_repeat_offenders(df: pd.DataFrame, min_crimes: int = 2) -> pd.DataFrame:
    """Identify suspects with multiple crime records."""
    if df.empty or "suspect_id" not in df.columns:
        return pd.DataFrame(columns=["suspect_id", "crime_count", "districts", "crime_types"])

    counts = (
        df.groupby("suspect_id")
        .agg(
            crime_count=("crime_id", "count"),
            districts=("district", lambda x: ", ".join(sorted(x.unique()[:3]))),
            crime_types=("crime_type", lambda x: ", ".join(sorted(x.unique()[:3]))),
        )
        .reset_index()
    )
    return counts[counts["crime_count"] >= min_crimes].sort_values("crime_count", ascending=False)


def detect_communities(G: nx.Graph) -> dict[str, int]:
    """Run community detection using Louvain-style greedy modularity."""
    if G.number_of_nodes() == 0:
        return {}

    try:
        from networkx.algorithms.community import greedy_modularity_communities

        communities = greedy_modularity_communities(G)
        partition: dict[str, int] = {}
        for idx, community in enumerate(communities):
            for node in community:
                partition[node] = idx
        return partition
    except Exception:
        return {node: 0 for node in G.nodes()}


def compute_centrality(G: nx.Graph) -> pd.DataFrame:
    """Compute degree and betweenness centrality for network nodes."""
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node", "node_type", "degree", "betweenness", "label"])

    degree_cent = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G)

    rows = []
    for node in G.nodes():
        node_type = G.nodes[node].get("node_type", "unknown")
        rows.append({
            "node": node,
            "node_type": node_type,
            "degree": round(degree_cent.get(node, 0), 4),
            "betweenness": round(between_cent.get(node, 0), 4),
            "label": G.nodes[node].get("label", node),
            "repeat_offender": G.nodes[node].get("repeat_offender", False),
        })

    return pd.DataFrame(rows).sort_values("degree", ascending=False)


def find_organized_crime_clusters(G: nx.Graph, min_size: int = 4) -> list[dict[str, Any]]:
    """Identify densely connected subgraphs suggesting organized crime."""
    if G.number_of_nodes() < min_size:
        return []

    suspect_subgraph = G.subgraph(
        [n for n, d in G.nodes(data=True) if d.get("node_type") == "suspect"]
    )
    clusters = []
    for component in nx.connected_components(suspect_subgraph):
        if len(component) >= min_size:
            subgraph = G.subgraph(component)
            clusters.append({
                "size": len(component),
                "members": [G.nodes[n].get("label", n) for n in component],
                "density": round(nx.density(subgraph), 4) if len(component) > 1 else 0,
                "avg_degree": round(sum(dict(subgraph.degree()).values()) / len(component), 2),
            })

    return sorted(clusters, key=lambda x: x["size"], reverse=True)


def generate_network_insights(
    G: nx.Graph,
    df: pd.DataFrame,
    centrality_df: pd.DataFrame,
    repeat_offenders: pd.DataFrame,
    clusters: list[dict],
) -> list[str]:
    """Generate automatic criminal association insights."""
    insights: list[str] = []

    if G.number_of_nodes() == 0:
        return ["No network data available for the selected filters."]

    insights.append(
        f"Network comprises {G.number_of_nodes()} entities and {G.number_of_edges()} associations."
    )

    suspect_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "suspect"]
    insights.append(f"Identified {len(suspect_nodes)} unique suspects in the network.")

    if len(repeat_offenders):
        top = repeat_offenders.iloc[0]
        insights.append(
            f"Top repeat offender: {top['suspect_id']} with {int(top['crime_count'])} recorded incidents "
            f"across districts: {top['districts']}."
        )

    if len(centrality_df):
        top_criminal = centrality_df[centrality_df["node_type"] == "suspect"].head(1)
        if len(top_criminal):
            row = top_criminal.iloc[0]
            insights.append(
                f"Most connected suspect: {row['label']} (degree centrality: {row['degree']})."
            )

    if clusters:
        insights.append(
            f"Detected {len(clusters)} organized crime cluster(s). "
            f"Largest cluster has {clusters[0]['size']} linked suspects."
        )

    cyber = df[df["crime_type"] == "Cybercrime"] if "crime_type" in df.columns else pd.DataFrame()
    if len(cyber) > 0:
        insights.append(
            f"Cybercrime network activity: {len(cyber)} incidents across "
            f"{cyber['district'].nunique()} districts."
        )

    high_risk = df[df["risk_zone"] == "High Risk"] if "risk_zone" in df.columns else pd.DataFrame()
    if len(high_risk) > 0:
        insights.append(
            f"{len(high_risk)} high-risk incidents linked to {high_risk['suspect_id'].nunique()} suspects."
        )

    return insights


def render_pyvis_network(G: nx.Graph, height: str = "600px") -> str:
    """Render interactive PyVis network and return HTML path."""
    net = Network(
        height=height,
        width="100%",
        bgcolor="#0d1b2a",
        font_color="#ffffff",
        directed=False,
    )
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "springLength": 120,
                "springConstant": 0.04
            }
        },
        "nodes": {
            "font": {"size": 14, "color": "#ffffff"},
            "borderWidth": 2
        },
        "edges": {
            "color": {"color": "#4fc3f7", "opacity": 0.6},
            "smooth": {"type": "continuous"}
        }
    }
    """)

    communities = detect_communities(G)

    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("node_type", "unknown")
        color = NODE_COLORS.get(node_type, "#95a5a6")
        size = 15 + G.degree(node) * 3
        if attrs.get("repeat_offender"):
            color = "#ff1744"
            size += 10

        title = (
            f"Type: {node_type}\\n"
            f"Label: {attrs.get('label', node)}\\n"
            f"Degree: {G.degree(node)}\\n"
            f"Community: {communities.get(node, 0)}"
        )
        net.add_node(
            node,
            label=attrs.get("label", node)[:20],
            color=color,
            size=min(size, 50),
            title=title,
            group=communities.get(node, 0),
        )

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1)
        net.add_edge(u, v, value=weight, title=f"Connections: {weight}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
    net.save_graph(tmp.name)
    return tmp.name
