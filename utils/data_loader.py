"""Data loading and preprocessing utilities for CrimeLens AI."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# Karnataka district metadata: name -> (lat, lon, population_density, unemployment_rate)
DISTRICT_META: dict[str, dict[str, float]] = {
    "Bengaluru Urban": {"lat": 12.9716, "lon": 77.5946, "pop_density": 4380, "unemployment": 4.2},
    "Bengaluru Rural": {"lat": 13.2257, "lon": 77.5750, "pop_density": 520, "unemployment": 5.1},
    "Mysuru": {"lat": 12.2958, "lon": 76.6394, "pop_density": 890, "unemployment": 4.8},
    "Mangaluru": {"lat": 12.9141, "lon": 74.8560, "pop_density": 1120, "unemployment": 5.3},
    "Hubballi": {"lat": 15.3647, "lon": 75.1240, "pop_density": 780, "unemployment": 5.6},
    "Belagavi": {"lat": 15.8497, "lon": 74.4977, "pop_density": 650, "unemployment": 5.9},
    "Kalaburagi": {"lat": 17.3297, "lon": 76.8343, "pop_density": 420, "unemployment": 6.2},
    "Ballari": {"lat": 15.1394, "lon": 76.9214, "pop_density": 380, "unemployment": 6.5},
    "Shivamogga": {"lat": 13.9299, "lon": 75.5681, "pop_density": 340, "unemployment": 5.4},
    "Hassan": {"lat": 13.0072, "lon": 76.0962, "pop_density": 290, "unemployment": 5.0},
    "Tumakuru": {"lat": 13.3379, "lon": 77.1173, "pop_density": 310, "unemployment": 5.2},
    "Davangere": {"lat": 14.4644, "lon": 75.9218, "pop_density": 360, "unemployment": 5.7},
    "Raichur": {"lat": 16.2076, "lon": 77.3463, "pop_density": 280, "unemployment": 6.8},
    "Vijayapura": {"lat": 16.8302, "lon": 75.7100, "pop_density": 320, "unemployment": 6.1},
    "Dharwad": {"lat": 15.4589, "lon": 75.0078, "pop_density": 410, "unemployment": 5.5},
    "Udupi": {"lat": 13.3409, "lon": 74.7421, "pop_density": 450, "unemployment": 4.6},
    "Chitradurga": {"lat": 14.2251, "lon": 76.3980, "pop_density": 260, "unemployment": 5.8},
    "Kodagu": {"lat": 12.4244, "lon": 75.7382, "pop_density": 180, "unemployment": 4.3},
    "Kolar": {"lat": 13.1360, "lon": 78.1290, "pop_density": 470, "unemployment": 5.4},
    "Mandya": {"lat": 12.5218, "lon": 76.8951, "pop_density": 390, "unemployment": 5.1},
    "Chikkamagaluru": {"lat": 13.3161, "lon": 75.7720, "pop_density": 220, "unemployment": 4.9},
    "Haveri": {"lat": 14.7951, "lon": 75.3990, "pop_density": 270, "unemployment": 5.6},
    "Gadag": {"lat": 15.4316, "lon": 75.6269, "pop_density": 250, "unemployment": 5.9},
    "Bagalkot": {"lat": 16.1691, "lon": 75.6615, "pop_density": 300, "unemployment": 6.0},
    "Chikkaballapur": {"lat": 13.4355, "lon": 77.7315, "pop_density": 350, "unemployment": 5.3},
    "Yadgir": {"lat": 16.7700, "lon": 77.1400, "pop_density": 240, "unemployment": 7.0},
    "Koppal": {"lat": 15.3500, "lon": 76.1500, "pop_density": 230, "unemployment": 6.7},
    "Ramanagara": {"lat": 12.7239, "lon": 77.2812, "pop_density": 400, "unemployment": 5.0},
    "Chamarajanagar": {"lat": 11.9261, "lon": 76.9437, "pop_density": 200, "unemployment": 5.5},
    "Uttara Kannada": {"lat": 14.7936, "lon": 74.6869, "pop_density": 190, "unemployment": 5.2},
    "Vijayanagara": {"lat": 15.3350, "lon": 76.4600, "pop_density": 310, "unemployment": 6.3},
}

CRIME_TYPES = [
    "Theft",
    "Assault",
    "Cybercrime",
    "Fraud",
    "Robbery",
    "Murder",
    "Drug Offense",
    "Domestic Violence",
    "Burglary",
    "Kidnapping",
]

STATUSES = ["Active", "Closed", "Arrested"]

POLICE_STATIONS = [
    "Central PS", "North PS", "South PS", "East PS", "West PS",
    "Cyber Crime Cell", "Women PS", "Traffic PS", "Rural PS", "Special Task Force",
]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Karnataka_CrimeLens_Synthetic_Dataset.csv"


def get_data_path() -> Path:
    """Return absolute path to the crime dataset."""
    return DATA_PATH


def generate_synthetic_dataset(n_records: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic Karnataka crime dataset for demo purposes."""
    rng = np.random.default_rng(seed)
    districts = list(DISTRICT_META.keys())
    records: list[dict] = []

    suspect_pool = [f"SUS-{i:05d}" for i in range(1, 801)]
    victim_pool = [f"VIC-{i:05d}" for i in range(1, 1201)]
    repeat_offenders = set(rng.choice(suspect_pool, size=120, replace=False))

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 6, 1)

    for i in range(1, n_records + 1):
        district = rng.choice(districts)
        meta = DISTRICT_META[district]
        crime_type = rng.choice(CRIME_TYPES, p=[0.22, 0.15, 0.12, 0.10, 0.08, 0.03, 0.10, 0.09, 0.08, 0.03])
        suspect_id = rng.choice(suspect_pool)
        is_repeat = suspect_id in repeat_offenders

        days_range = (end_date - start_date).days
        crime_date = start_date + pd.Timedelta(days=int(rng.integers(0, days_range)))

        lat = meta["lat"] + rng.normal(0, 0.08)
        lon = meta["lon"] + rng.normal(0, 0.08)
        pop_density = meta["pop_density"] * rng.uniform(0.85, 1.15)
        unemployment = meta["unemployment"] * rng.uniform(0.9, 1.1)

        risk_score = (
            0.3 * (pop_density / 4500)
            + 0.25 * (unemployment / 8)
            + 0.25 * (1 if is_repeat else 0)
            + 0.2 * rng.random()
        )
        if risk_score >= 0.65:
            risk_zone = "High Risk"
        elif risk_score >= 0.4:
            risk_zone = "Medium Risk"
        else:
            risk_zone = "Low Risk"

        status = rng.choice(STATUSES, p=[0.25, 0.45, 0.30])
        arrested = 1 if status == "Arrested" else 0

        records.append({
            "crime_id": f"CR-{i:06d}",
            "date": crime_date.strftime("%Y-%m-%d"),
            "district": district,
            "crime_type": crime_type,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "location": f"{district} Zone {rng.integers(1, 6)}",
            "suspect_id": suspect_id,
            "victim_id": rng.choice(victim_pool),
            "police_station": f"{district} {rng.choice(POLICE_STATIONS)}",
            "status": status,
            "arrested": arrested,
            "repeat_offender": int(is_repeat),
            "population_density": round(pop_density, 2),
            "unemployment_rate": round(unemployment, 2),
            "risk_zone": risk_zone,
            "risk_score": round(risk_score, 4),
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich crime data with derived columns."""
    data = df.copy()

    # Handle missing values
    numeric_cols = [
        "latitude", "longitude", "population_density",
        "unemployment_rate", "risk_score", "arrested", "repeat_offender",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            if col in ("latitude", "longitude"):
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(0)

    categorical_cols = ["district", "crime_type", "status", "risk_zone", "police_station", "location"]
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].fillna("Unknown")

    for col in ["suspect_id", "victim_id", "crime_id"]:
        if col in data.columns:
            data[col] = data[col].fillna("Unknown")

    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["date"] = data["date"].fillna(pd.Timestamp("2024-01-01"))
        data["year"] = data["date"].dt.year
        data["month"] = data["date"].dt.month
        data["month_year"] = data["date"].dt.to_period("M").astype(str)
        data["day_of_week"] = data["date"].dt.day_name()

    if "risk_score" not in data.columns and "risk_zone" in data.columns:
        zone_map = {"Low Risk": 0.25, "Medium Risk": 0.55, "High Risk": 0.85}
        data["risk_score"] = data["risk_zone"].map(zone_map).fillna(0.5)

    return data


@st.cache_data(show_spinner="Loading crime intelligence data...")
def load_crime_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load and preprocess crime data with Streamlit caching."""
    data_path = Path(path) if path else DATA_PATH

    if not data_path.exists():
        df = generate_synthetic_dataset()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)

    df = pd.read_csv(data_path, parse_dates=["date"], dayfirst=False)
    return preprocess_data(df)


def filter_data(
    df: pd.DataFrame,
    districts: Optional[list[str]] = None,
    crime_types: Optional[list[str]] = None,
    date_range: Optional[tuple] = None,
) -> pd.DataFrame:
    """Apply sidebar filters to the crime dataset."""
    filtered = df.copy()

    if districts:
        filtered = filtered[filtered["district"].isin(districts)]

    if crime_types:
        filtered = filtered[filtered["crime_type"].isin(crime_types)]

    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]

    return filtered


def get_kpi_metrics(df: pd.DataFrame) -> dict[str, int | float | str]:
    """Compute dashboard KPI metrics from filtered data."""
    total_crimes = len(df)
    active_cases = int((df["status"] == "Active").sum()) if "status" in df.columns else 0
    arrests = int(df["arrested"].sum()) if "arrested" in df.columns else 0
    repeat_offenders = int(df[df["repeat_offender"] == 1]["suspect_id"].nunique()) if len(df) else 0

    if len(df) and "district" in df.columns:
        district_counts = df["district"].value_counts()
        high_risk_district = district_counts.index[0]
        high_risk_count = int(district_counts.iloc[0])
    else:
        high_risk_district = "N/A"
        high_risk_count = 0

    arrest_rate = round((arrests / total_crimes * 100), 1) if total_crimes else 0.0

    return {
        "total_crimes": total_crimes,
        "active_cases": active_cases,
        "arrests_made": arrests,
        "repeat_offenders": repeat_offenders,
        "high_risk_district": high_risk_district,
        "high_risk_count": high_risk_count,
        "arrest_rate": arrest_rate,
    }

