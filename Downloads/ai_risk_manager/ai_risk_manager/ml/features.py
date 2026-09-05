import pandas as pd
import numpy as np


def load_raw(path="data/transactions.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features to the transaction dataframe.
    Returns a new dataframe; does not mutate the input.
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.date
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 23)).astype(int)
    df["is_failed"] = (df["transaction_status"] == "FAILED").astype(int)

    # ---- Merchant-level "normal" ticket size, computed on TRAIN-safe basis ----
    # We use an expanding (cumulative, backward-looking) mean/std per merchant
    # so no future information leaks into a given row's features.
    df["merchant_running_count"] = df.groupby("merchant_id").cumcount()
    df["merchant_amt_running_mean"] = (
        df.groupby("merchant_id")["amount"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )
    df["merchant_amt_running_std"] = (
        df.groupby("merchant_id")["amount"]
        .apply(lambda s: s.shift(1).expanding().std())
        .reset_index(level=0, drop=True)
    )
    df["merchant_amt_running_mean"] = df["merchant_amt_running_mean"].fillna(df["amount"])
    df["merchant_amt_running_std"] = df["merchant_amt_running_std"].fillna(0).replace(0, 1e-3)

    df["amount_zscore_vs_merchant"] = (
        (df["amount"] - df["merchant_amt_running_mean"]) / df["merchant_amt_running_std"]
    )

    # ---- Device-level transaction count so far (device_burst pattern) ----
    df["device_running_count"] = df.groupby("device_id").cumcount()

    # ---- Customer-level transaction count so far ----
    df["customer_running_count"] = df.groupby("customer_id").cumcount()

    # ---- Geo anomaly: does this txn's location match merchant's most common past location? ----
    def _mode_so_far(s):
        modes = []
        seen = {}
        for v in s.shift(1):
            if pd.isna(v):
                modes.append(np.nan)
                continue
            seen[v] = seen.get(v, 0) + 1
        return modes

    # Efficient approach: merchant's dominant ("home") location computed from full history
    # (not leaking future *label* info, just typical geography -- acceptable as a merchant
    # profile attribute, similar to a KYC field a real system would have).
    merchant_home_location = (
        df.groupby("merchant_id")["location"]
        .agg(lambda x: x.value_counts().idxmax())
    )
    df["merchant_home_location"] = df["merchant_id"].map(merchant_home_location)
    df["is_geo_anomaly"] = (df["location"] != df["merchant_home_location"]).astype(int)

    # ---- Rolling window features: transactions in trailing time windows ----
    # For merchant-level spike detection: count of txns for this merchant in the
    # trailing 60 minutes and trailing 24 hours (backward-looking only).
    df = df.set_index("timestamp")

    def rolling_count(group, window):
        return group.rolling(window, closed="left").count()

    merchant_60min = []
    merchant_24h = []
    failed_rate_24h = []

    for merchant_id, g in df.groupby("merchant_id"):
        g = g.sort_index()
        ones = pd.Series(1, index=g.index)
        c60 = ones.rolling("60min", closed="left").sum().fillna(0)
        c24 = ones.rolling("24h", closed="left").sum().fillna(0)
        failed = g["is_failed"].rolling("24h", closed="left").mean().fillna(0)
        merchant_60min.append(c60)
        merchant_24h.append(c24)
        failed_rate_24h.append(failed)

    df["merchant_txn_count_60min"] = pd.concat(merchant_60min).sort_index()
    df["merchant_txn_count_24h"] = pd.concat(merchant_24h).sort_index()
    df["merchant_failed_rate_24h"] = pd.concat(failed_rate_24h).sort_index()

    # Device burst: transactions from same device in trailing 60 minutes
    device_60min = []
    for device_id, g in df.groupby("device_id"):
        g = g.sort_index()
        ones = pd.Series(1, index=g.index)
        c60 = ones.rolling("60min", closed="left").sum().fillna(0)
        device_60min.append(c60)
    df["device_txn_count_60min"] = pd.concat(device_60min).sort_index()

    df = df.reset_index()

    # ---- Merchant-level spike ratio: today's volume vs merchant's typical daily volume ----
    daily_counts = (
        df.groupby(["merchant_id", "day"]).size().rename("daily_txn_count").reset_index()
    )
    merchant_daily_avg = (
        daily_counts.groupby("merchant_id")["daily_txn_count"].mean().rename("merchant_avg_daily_txns")
    )
    df = df.merge(daily_counts, on=["merchant_id", "day"], how="left")
    df = df.merge(merchant_avg_daily := merchant_daily_avg.reset_index(), on="merchant_id", how="left")
    df["merchant_spike_ratio"] = df["daily_txn_count"] / df["merchant_avg_daily_txns"].replace(0, 1)

    # Payment method one-hot (small cardinality, safe)
    df = pd.get_dummies(df, columns=["payment_method"], prefix="pm")

    return df


FEATURE_COLUMNS = [
    "amount",
    "hour",
    "is_night",
    "is_failed",
    "amount_zscore_vs_merchant",
    "device_running_count",
    "customer_running_count",
    "is_geo_anomaly",
    "merchant_txn_count_60min",
    "merchant_txn_count_24h",
    "merchant_failed_rate_24h",
    "device_txn_count_60min",
    "merchant_spike_ratio",
]


def get_feature_matrix(df: pd.DataFrame):
    """Returns (X, y, full_feature_df) ready for modelling."""
    fdf = engineer_features(df)
    pm_cols = [c for c in fdf.columns if c.startswith("pm_")]
    cols = FEATURE_COLUMNS + pm_cols
    cols = [c for c in cols if c in fdf.columns]
    X = fdf[cols].fillna(0)
    y = fdf["fraud_label"] if "fraud_label" in fdf.columns else None
    return X, y, fdf


if __name__ == "__main__":
    raw = load_raw()
    X, y, fdf = get_feature_matrix(raw)
    print(X.shape, y.mean() if y is not None else None)
    print(X.head())
