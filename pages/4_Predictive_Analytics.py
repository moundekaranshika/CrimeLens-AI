"""Page 4: Predictive Analytics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_loader import filter_data, load_crime_data
from utils.prediction_utils import (
    RISK_COLORS,
    detect_anomalies,
    forecast_district_risk,
    get_model_metrics,
    predict_hotspot_risk,
)
from utils.theme import PLOTLY_COLORS, PLOTLY_TEMPLATE, apply_custom_css, render_page_header, render_sidebar_branding, render_sidebar_filters

st.set_page_config(page_title="Predictive Analytics | CrimeLens AI", page_icon="📈", layout="wide")
apply_custom_css()
render_sidebar_branding()

df = load_crime_data()
districts, crime_types, date_range = render_sidebar_filters(df)
filtered = filter_data(df, districts, crime_types, date_range)

render_page_header(
    "📈 Predictive Analytics",
    "Machine learning models for hotspot prediction, anomaly detection, and risk forecasting",
)

metrics = get_model_metrics(filtered)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Hotspot Model", "✅ Ready" if metrics["hotspot_trained"] else "❌ Not Trained")
m2.metric("Model Accuracy", f"{metrics['hotspot_accuracy']}")
m3.metric("Anomaly Model", "✅ Ready" if metrics["anomaly_trained"] else "❌ Not Trained")
m4.metric("Training Samples", metrics["anomaly_samples"])

tab1, tab2, tab3 = st.tabs([
    "🎯 Hotspot Prediction (XGBoost)",
    "⚠️ Anomaly Detection (Isolation Forest)",
    "📊 Risk Forecasting",
])

with tab1:
    st.subheader("Model 1: Crime Hotspot Prediction")
    st.markdown(
        """
        **Algorithm:** XGBoost Classifier  
        **Target:** High Risk Zone classification  
        **Features:** District, Crime Type, Population Density, Unemployment Rate, Repeat Offender  
        **Output:** Low Risk | Medium Risk | High Risk
        """
    )

    if len(filtered):
        predictions = predict_hotspot_risk(filtered)
        risk_dist = predictions["predicted_risk"].value_counts().reset_index()
        risk_dist.columns = ["risk_level", "count"]

        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.pie(
                risk_dist,
                values="count",
                names="risk_level",
                color="risk_level",
                color_discrete_map=RISK_COLORS,
                template=PLOTLY_TEMPLATE,
                hole=0.35,
                title="Predicted Risk Distribution",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            district_risk = (
                predictions.groupby(["district", "predicted_risk"])
                .size()
                .reset_index(name="count")
            )
            fig = px.bar(
                district_risk,
                x="district",
                y="count",
                color="predicted_risk",
                color_discrete_map=RISK_COLORS,
                template=PLOTLY_TEMPLATE,
                title="Risk by District",
                barmode="stack",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400,
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("View Prediction Details"):
            display_cols = [
                "crime_id", "district", "crime_type", "risk_zone",
                "predicted_risk", "risk_confidence",
            ]
            st.dataframe(
                predictions[display_cols].head(50),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No data for prediction.")

with tab2:
    st.subheader("Model 2: Anomaly Detection")
    st.markdown(
        """
        **Algorithm:** Isolation Forest  
        **Detects:** Unusual crime spikes, suspicious activity, emerging crime trends
        """
    )

    anomalies = detect_anomalies(filtered)

    if len(anomalies):
        anomaly_count = int(anomalies["is_anomaly"].sum())
        st.metric("Anomalies Detected", anomaly_count, delta=f"{anomaly_count} alerts", delta_color="inverse")

        c1, c2 = st.columns(2)

        with c1:
            fig = px.scatter(
                anomalies,
                x="crime_count",
                y="avg_risk",
                color="is_anomaly",
                size="repeat_count",
                hover_data=["district", "month_year", "severity"],
                color_discrete_map={True: "#e53935", False: "#4fc3f7"},
                template=PLOTLY_TEMPLATE,
                title="Anomaly Scatter Plot",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            anomaly_rows = anomalies[anomalies["is_anomaly"]].head(15)
            if len(anomaly_rows):
                fig = go.Figure(go.Bar(
                    x=anomaly_rows["crime_count"],
                    y=anomaly_rows["district"] + " (" + anomaly_rows["month_year"] + ")",
                    orientation="h",
                    marker_color="#e53935",
                ))
                fig.update_layout(
                    template=PLOTLY_TEMPLATE,
                    title="Anomalous Crime Spikes",
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No anomalies detected in current period.")

        # Alert boxes for critical anomalies
        critical = anomalies[(anomalies["is_anomaly"]) & (anomalies["severity"] == "Critical")]
        for _, row in critical.head(5).iterrows():
            st.markdown(
                f'<div class="alert-box">🚨 <b>{row["district"]}</b> — {row["month_year"]}: '
                f'{int(row["crime_count"])} crimes detected (Score: {row["anomaly_score"]})</div>',
                unsafe_allow_html=True,
            )

        with st.expander("Full Anomaly Report"):
            st.dataframe(anomalies, use_container_width=True, hide_index=True)
    else:
        st.info("Insufficient data for anomaly detection.")

with tab3:
    st.subheader("Model 3: District Risk Forecasting")
    st.markdown("Predicts **next-month crime risk scores** and generates **district risk rankings**.")

    forecast = forecast_district_risk(filtered)

    if len(forecast):
        c1, c2 = st.columns([2, 1])

        with c1:
            fig = px.bar(
                forecast.head(15),
                x="district",
                y="forecast_risk_score",
                color="risk_level",
                color_discrete_map=RISK_COLORS,
                template=PLOTLY_TEMPLATE,
                title="District Risk Forecast — Next Month",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Top 10 High-Risk Districts")
            top10 = forecast.head(10)[["risk_rank", "district", "forecast_risk_score", "trend"]]
            st.dataframe(top10, use_container_width=True, hide_index=True)

        # Risk heatmap matrix
        st.subheader("Risk Factor Breakdown")
        heatmap_data = forecast.head(12)[
            ["district", "avg_risk", "repeat_rate", "unemployment", "recent_crimes"]
        ].set_index("district")
        fig = px.imshow(
            heatmap_data.T,
            color_continuous_scale=[[0, "#1a237e"], [0.5, "#4fc3f7"], [1, "#e53935"]],
            template=PLOTLY_TEMPLATE,
            aspect="auto",
            title="Risk Factor Heatmap (Top 12 Districts)",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Full Risk Forecast Table"):
            st.dataframe(forecast, use_container_width=True, hide_index=True)
    else:
        st.info("No data for risk forecasting.")
