"""Machine learning prediction utilities for CrimeLens AI."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBClassifier = GradientBoostingClassifier  # type: ignore
    XGBOOST_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
HOTSPOT_MODEL_PATH = MODELS_DIR / "hotspot_model.pkl"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_model.pkl"

RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
RISK_COLORS = {"Low Risk": "#2ecc71", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"}


def _encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode categorical features for ML models."""
    encoders: dict = {}
    data = df.copy()

    for col in ["district", "crime_type"]:
        if col in data.columns:
            le = LabelEncoder()
            data[f"{col}_encoded"] = le.fit_transform(data[col].astype(str))
            encoders[col] = le

    feature_cols = [
        "district_encoded", "crime_type_encoded",
        "population_density", "unemployment_rate", "repeat_offender",
    ]
    for col in feature_cols:
        if col not in data.columns:
            data[col] = 0

    return data, encoders


def train_hotspot_model(df: pd.DataFrame) -> tuple:
    """Train XGBoost (or GradientBoosting fallback) classifier for crime hotspot risk prediction."""
    data, encoders = _encode_features(df)

    zone_map = {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2}
    data["risk_label"] = data["risk_zone"].map(zone_map).fillna(1).astype(int)

    feature_cols = [
        "district_encoded", "crime_type_encoded",
        "population_density", "unemployment_rate", "repeat_offender",
    ]
    X = data[feature_cols].values
    y = data["risk_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if XGBOOST_AVAILABLE:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
    model.fit(X_train, y_train)

    metadata = {
        "feature_cols": feature_cols,
        "encoders": encoders,
        "accuracy": round(float(model.score(X_test, y_test)), 4),
        "zone_map": zone_map,
        "model_type": "XGBoost" if XGBOOST_AVAILABLE else "GradientBoosting (fallback)",
    }
    return model, metadata, feature_cols


def train_anomaly_model(df: pd.DataFrame) -> tuple[IsolationForest, dict]:
    """Train Isolation Forest for anomaly detection on crime patterns."""
    monthly = (
        df.groupby(["district", "month_year"])
        .agg(
            crime_count=("crime_id", "count"),
            avg_risk=("risk_score", "mean"),
            repeat_count=("repeat_offender", "sum"),
        )
        .reset_index()
    )

    if monthly.empty:
        monthly = pd.DataFrame({
            "crime_count": [10, 15, 8],
            "avg_risk": [0.4, 0.6, 0.3],
            "repeat_count": [1, 2, 0],
        })

    features = monthly[["crime_count", "avg_risk", "repeat_count"]].fillna(0).values

    model = IsolationForest(
        n_estimators=100,
        contamination=0.08,
        random_state=42,
    )
    model.fit(features)

    metadata = {
        "feature_cols": ["crime_count", "avg_risk", "repeat_count"],
        "training_samples": len(monthly),
    }
    return model, metadata


def save_models(df: pd.DataFrame) -> None:
    """Train and persist all ML models."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    hotspot_model, hotspot_meta, _ = train_hotspot_model(df)
    joblib.dump({"model": hotspot_model, "metadata": hotspot_meta}, HOTSPOT_MODEL_PATH)

    anomaly_model, anomaly_meta = train_anomaly_model(df)
    joblib.dump({"model": anomaly_model, "metadata": anomaly_meta}, ANOMALY_MODEL_PATH)


def _load_model_bundle(path: Path) -> tuple[Optional[object], dict]:
    """Load a persisted model bundle, returning empty values if unpickling fails."""
    if not path.exists():
        return None, {}
    try:
        bundle = joblib.load(path)
        model = bundle.get("model")
        metadata = bundle.get("metadata", {})
        if model is None:
            return None, {}
        return model, metadata
    except (
        ModuleNotFoundError,
        ImportError,
        AttributeError,
        EOFError,
        pickle.UnpicklingError,
        ValueError,
    ):
        return None, {}


def load_hotspot_model() -> tuple[Optional[object], dict]:
    """Load trained hotspot prediction model."""
    return _load_model_bundle(HOTSPOT_MODEL_PATH)


def load_anomaly_model() -> tuple[Optional[IsolationForest], dict]:
    """Load trained anomaly detection model."""
    model, metadata = _load_model_bundle(ANOMALY_MODEL_PATH)
    return model, metadata


def models_are_ready() -> bool:
    """Return True when both ML models load successfully."""
    hotspot_model, _ = load_hotspot_model()
    anomaly_model, _ = load_anomaly_model()
    return hotspot_model is not None and anomaly_model is not None


def ensure_models(df: pd.DataFrame) -> None:
    """Train models when missing or when saved pickles are incompatible with the runtime."""
    if models_are_ready():
        return
    save_models(df)


def predict_hotspot_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Predict risk zones for crime records using XGBoost model."""
    model, metadata = load_hotspot_model()
    if model is None:
        result = df.copy()
        result["predicted_risk"] = df.get("risk_zone", "Unknown")
        result["risk_confidence"] = 0.0
        return result

    data, _ = _encode_features(df)
    encoders = metadata.get("encoders", {})

    for col, le in encoders.items():
        encoded_col = f"{col}_encoded"
        try:
            data[encoded_col] = le.transform(data[col].astype(str))
        except ValueError:
            data[encoded_col] = 0

    feature_cols = metadata.get("feature_cols", [])
    X = data[feature_cols].fillna(0).values
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    result = df.copy()
    result["predicted_risk"] = [RISK_LABELS.get(int(p), "Medium Risk") for p in predictions]
    result["risk_confidence"] = probabilities.max(axis=1).round(3)
    return result


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Detect anomalous crime spikes using Isolation Forest."""
    model, metadata = load_anomaly_model()
    if model is None:
        return pd.DataFrame()

    monthly = (
        df.groupby(["district", "month_year"])
        .agg(
            crime_count=("crime_id", "count"),
            avg_risk=("risk_score", "mean"),
            repeat_count=("repeat_offender", "sum"),
        )
        .reset_index()
    )

    if monthly.empty:
        return monthly

    features = monthly[["crime_count", "avg_risk", "repeat_count"]].fillna(0).values
    predictions = model.predict(features)
    scores = model.decision_function(features)

    monthly = monthly.copy()
    monthly["is_anomaly"] = predictions == -1
    monthly["anomaly_score"] = scores.round(4)
    monthly["severity"] = monthly["anomaly_score"].apply(
        lambda s: "Critical" if s < -0.15 else ("High" if s < -0.05 else "Normal")
    )
    return monthly.sort_values("anomaly_score")


def forecast_district_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Forecast next-month crime risk scores by district."""
    if df.empty:
        return pd.DataFrame(columns=["district", "forecast_risk_score", "risk_rank", "trend"])

    district_stats = (
        df.groupby("district")
        .agg(
            total_crimes=("crime_id", "count"),
            avg_risk=("risk_score", "mean"),
            repeat_rate=("repeat_offender", "mean"),
            unemployment=("unemployment_rate", "mean"),
            pop_density=("population_density", "mean"),
            recent_crimes=("date", lambda x: len(x[x >= (x.max() - pd.Timedelta(days=90))])),
        )
        .reset_index()
    )

    # Weighted risk forecast formula
    district_stats["forecast_risk_score"] = (
        0.30 * (district_stats["total_crimes"] / district_stats["total_crimes"].max())
        + 0.25 * district_stats["avg_risk"]
        + 0.20 * district_stats["repeat_rate"]
        + 0.15 * (district_stats["unemployment"] / 8)
        + 0.10 * (district_stats["recent_crimes"] / district_stats["recent_crimes"].max())
    ).round(4)

    district_stats["risk_rank"] = district_stats["forecast_risk_score"].rank(ascending=False).astype(int)
    district_stats["trend"] = district_stats["forecast_risk_score"].apply(
        lambda s: "Rising" if s >= 0.6 else ("Stable" if s >= 0.35 else "Declining")
    )
    district_stats["risk_level"] = district_stats["forecast_risk_score"].apply(
        lambda s: "High Risk" if s >= 0.6 else ("Medium Risk" if s >= 0.35 else "Low Risk")
    )

    return district_stats.sort_values("forecast_risk_score", ascending=False)


def get_model_metrics(df: pd.DataFrame) -> dict:
    """Return model performance and status metrics."""
    hotspot_model, hotspot_meta = load_hotspot_model()
    anomaly_model, anomaly_meta = load_anomaly_model()

    return {
        "hotspot_accuracy": hotspot_meta.get("accuracy", "N/A"),
        "hotspot_trained": hotspot_model is not None,
        "anomaly_trained": anomaly_model is not None,
        "anomaly_samples": anomaly_meta.get("training_samples", 0),
        "total_records": len(df),
    }
