import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import pytest

from ml.features import engineer_features, get_feature_matrix, FEATURE_COLUMNS


def make_sample_df(n=50):
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2025-01-01")
    rows = []
    for i in range(n):
        rows.append({
            "transaction_id": f"T{i}",
            "timestamp": base + pd.Timedelta(minutes=i * 10),
            "merchant_id": "M0001" if i % 3 else "M0002",
            "amount": rng.uniform(50, 500),
            "payment_method": rng.choice(["UPI", "Credit Card"]),
            "location": rng.choice(["Mumbai", "Delhi"]),
            "device_id": f"D{i % 5}",
            "customer_id": f"C{i % 7}",
            "transaction_status": rng.choice(["SUCCESS", "FAILED"], p=[0.9, 0.1]),
            "fraud_label": 1 if i % 20 == 0 else 0,
        })
    return pd.DataFrame(rows)


def test_engineer_features_returns_expected_columns():
    df = make_sample_df()
    fdf = engineer_features(df)
    for col in ["amount_zscore_vs_merchant", "merchant_txn_count_60min",
                "device_txn_count_60min", "is_geo_anomaly", "merchant_spike_ratio"]:
        assert col in fdf.columns


def test_engineer_features_no_future_leakage():
    """The first transaction for a merchant must not see any 'running' history."""
    df = make_sample_df()
    fdf = engineer_features(df)
    first_per_merchant = fdf.sort_values("timestamp").groupby("merchant_id").first()
    for merchant_id, row in first_per_merchant.iterrows():
        assert row["merchant_running_count"] == 0


def test_get_feature_matrix_shapes_match():
    df = make_sample_df()
    X, y, fdf = get_feature_matrix(df)
    assert len(X) == len(df)
    assert len(y) == len(df)
    assert X.isnull().sum().sum() == 0  # no NaNs should remain after fillna


def test_feature_matrix_columns_are_numeric():
    df = make_sample_df()
    X, y, fdf = get_feature_matrix(df)
    for col in X.columns:
        assert pd.api.types.is_numeric_dtype(X[col]) or X[col].dtype == bool
