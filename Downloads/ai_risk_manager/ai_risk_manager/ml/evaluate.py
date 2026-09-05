import json
import joblib
import pandas as pd
import numpy as np

from ml.features import load_raw, get_feature_matrix, engineer_features
from ml.risk_scoring import probability_to_risk_score, severity_band
from ml.spike_detection import detect_spikes

MODEL_PATH = "models/fraud_model.joblib"
METRICS_PATH = "models/metrics.json"
FEATURE_META_PATH = "models/feature_columns.json"


def load_model_and_metrics():
    model = joblib.load(MODEL_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    with open(FEATURE_META_PATH) as f:
        feature_cols = json.load(f)
    return model, metrics, feature_cols


def score_full_dataset():
    """
    Scores every transaction with the trained model and attaches risk
    score/severity + spike detection results. Used to power the dashboard.
    """
    model, metrics, feature_cols = load_model_and_metrics()
    raw = load_raw()
    fdf = engineer_features(raw)
    X = fdf[feature_cols].fillna(0)

    proba = model.predict_proba(X)[:, 1]
    threshold = metrics["chosen_threshold"]

    fdf["fraud_probability"] = proba
    fdf["risk_score"] = probability_to_risk_score(proba)
    fdf["is_flagged"] = (proba >= threshold).astype(int)
    fdf["severity"] = fdf["risk_score"].apply(severity_band)

    spikes = detect_spikes(fdf[["merchant_id", "day", "is_flagged"]])

    return fdf, spikes, metrics, model, feature_cols


if __name__ == "__main__":
    fdf, spikes, metrics, model, feature_cols = score_full_dataset()
    print("Scored dataset shape:", fdf.shape)
    print("Flagged transactions:", int(fdf["is_flagged"].sum()))
    print("Detected spikes:", int(spikes["is_spike"].sum()))
    print("\nHeld-out test metrics (from training):")
    print(json.dumps(metrics["test_metrics"], indent=2))
