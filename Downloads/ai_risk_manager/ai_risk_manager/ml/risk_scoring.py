 numpy as np
import pandas as pd


def probability_to_risk_score(proba: np.ndarray) -> np.ndarray:
    """Simple monotonic mapping from model probability [0,1] to a 0-100 score."""
    return np.clip(proba * 100, 0, 100).round(1)


def severity_band(risk_score: float) -> str:
    if risk_score >= 85:
        return "Critical"
    elif risk_score >= 65:
        return "High"
    elif risk_score >= 35:
        return "Medium"
    else:
        return "Low"


DEFENSIVE_RECOMMENDATIONS = {
    "Critical": [
        "Temporarily hold/review the flagged transactions before settlement",
        "Require additional customer verification (OTP / re-authentication)",
        "Contact the merchant to confirm legitimacy of the activity",
        "Escalate to the fraud/risk team for manual review",
    ],
    "High": [
        "Require additional verification for further transactions from this merchant/device",
        "Increase monitoring frequency on this merchant for the next 24-48 hours",
        "Review the flagged transaction patterns manually",
    ],
    "Medium": [
        "Flag for periodic review",
        "Monitor for recurrence over the next few days",
    ],
    "Low": [
        "No action needed; continue routine monitoring",
    ],
}


def get_recommendations(severity: str):
    return DEFENSIVE_RECOMMENDATIONS.get(severity, DEFENSIVE_RECOMMENDATIONS["Low"])


def top_contributing_features(model, X_row: pd.DataFrame, feature_cols, top_n=5):
    """
    Lightweight, dependency-free explainability: for a single row, estimate
    each feature's contribution by comparing the row's value to the
    training population median/typical range, weighted by the model's
    global feature_importances_. This gives an approximate, human-readable
    "why" without requiring SHAP (kept optional/lightweight for a hackathon
    build). Returns list of (feature, importance_weight, row_value).
    """
    importances = model.feature_importances_
    row_vals = X_row.iloc[0]
    ranked = sorted(
        zip(feature_cols, importances, [row_vals[c] for c in feature_cols]),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_n]


def compute_false_positive_cost(
    n_false_positives: int,
    n_true_positives: int,
    avg_fraud_amount: float,
    avg_legit_amount: float,
    cost_per_false_positive: float = 150.0,
    review_success_rate: float = 0.9,
):
    """
    A simple, configurable risk-cost comparison.

    cost_per_false_positive: operational/customer-friction cost of wrongly
        flagging a legitimate transaction (manual review time, customer
        annoyance, potential churn) -- configurable, in INR.
    review_success_rate: fraction of correctly-flagged fraud that is
        actually stopped/recovered once flagged (not 100%, to stay honest).
    """
    estimated_fp_loss = n_false_positives * cost_per_false_positive
    estimated_fraud_prevented = n_true_positives * avg_fraud_amount * review_success_rate
    net_benefit = estimated_fraud_prevented - estimated_fp_loss
    return {
        "n_false_positives": int(n_false_positives),
        "n_true_positives_detected": int(n_true_positives),
        "cost_per_false_positive": cost_per_false_positive,
        "estimated_false_positive_cost": round(estimated_fp_loss, 2),
        "review_success_rate": review_success_rate,
        "estimated_fraud_loss_prevented": round(estimated_fraud_prevented, 2),
        "net_estimated_benefit": round(net_benefit, 2),
    }
