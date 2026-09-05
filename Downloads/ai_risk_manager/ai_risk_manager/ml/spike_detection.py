import numpy as np
import pandas as pd


def detect_spikes(flagged_df: pd.DataFrame, z_threshold: float = 3.0, min_count: int = 5):
    """
    flagged_df must contain columns: merchant_id, day, is_flagged (0/1)

    Returns a dataframe of per-merchant-per-day suspicious counts with a
    `is_spike` boolean and `spike_severity` label.
    """
    daily = (
        flagged_df.groupby(["merchant_id", "day"])
        .agg(
            suspicious_count=("is_flagged", "sum"),
            total_txns=("is_flagged", "count"),
        )
        .reset_index()
    )
    daily = daily.sort_values(["merchant_id", "day"])

    daily["rolling_mean"] = (
        daily.groupby("merchant_id")["suspicious_count"]
        .transform(lambda s: s.shift(1).rolling(14, min_periods=3).mean())
    )
    daily["rolling_std"] = (
        daily.groupby("merchant_id")["suspicious_count"]
        .transform(lambda s: s.shift(1).rolling(14, min_periods=3).std())
    )
    daily["rolling_mean"] = daily["rolling_mean"].fillna(daily["suspicious_count"].median())
    daily["rolling_std"] = daily["rolling_std"].fillna(1.0).replace(0, 1.0)

    daily["spike_z_score"] = (
        (daily["suspicious_count"] - daily["rolling_mean"]) / daily["rolling_std"]
    )
    daily["is_spike"] = (
        (daily["spike_z_score"] >= z_threshold) & (daily["suspicious_count"] >= min_count)
    )

    def severity(row):
        if not row["is_spike"]:
            return "None"
        if row["spike_z_score"] >= 6:
            return "Critical"
        elif row["spike_z_score"] >= 4.5:
            return "High"
        elif row["spike_z_score"] >= 3.5:
            return "Medium"
        else:
            return "Low"

    daily["spike_severity"] = daily.apply(severity, axis=1)
    return daily


def explain_spike(row, avg_amount=None):
    reasons = []
    reasons.append(
        f"{int(row['suspicious_count'])} suspicious transactions flagged for "
        f"{row['merchant_id']} on {row['day']}, vs a typical trailing average of "
        f"{row['rolling_mean']:.1f} (+/- {row['rolling_std']:.1f})."
    )
    reasons.append(f"Spike z-score: {row['spike_z_score']:.2f} standard deviations above baseline.")
    if avg_amount is not None:
        reasons.append(f"Average amount involved: Rs. {avg_amount:,.2f}")
    return reasons
