import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from ml.risk_scoring import (
    probability_to_risk_score, severity_band, get_recommendations,
    compute_false_positive_cost,
)
from ml.spike_detection import detect_spikes


def test_probability_to_risk_score_range():
    proba = np.array([0.0, 0.5, 1.0])
    scores = probability_to_risk_score(proba)
    assert list(scores) == [0.0, 50.0, 100.0]


def test_severity_band_thresholds():
    assert severity_band(90) == "Critical"
    assert severity_band(70) == "High"
    assert severity_band(50) == "Medium"
    assert severity_band(10) == "Low"


def test_get_recommendations_never_offensive_keywords():
    banned_terms = ["bypass", "exploit", "spoof", "attack merchant", "evade detection"]
    for severity in ["Critical", "High", "Medium", "Low"]:
        recs = " ".join(get_recommendations(severity)).lower()
        for term in banned_terms:
            assert term not in recs


def test_compute_false_positive_cost_math():
    result = compute_false_positive_cost(
        n_false_positives=10, n_true_positives=5,
        avg_fraud_amount=1000, avg_legit_amount=200,
        cost_per_false_positive=100, review_success_rate=0.8,
    )
    assert result["estimated_false_positive_cost"] == 1000.0
    assert result["estimated_fraud_loss_prevented"] == 4000.0
    assert result["net_estimated_benefit"] == 3000.0


def test_detect_spikes_flags_injected_spike():
    """Build a synthetic merchant history with an obvious spike day and confirm detection."""
    rows = []
    merchant = "M_TEST"
    for day in range(20):
        count = 2  # normal baseline suspicious count per day
        if day == 15:
            count = 40  # obvious spike
        for _ in range(count):
            rows.append({"merchant_id": merchant, "day": f"2025-01-{day+1:02d}", "is_flagged": 1})
        # also add some non-flagged txns so total counts differ
        for _ in range(5):
            rows.append({"merchant_id": merchant, "day": f"2025-01-{day+1:02d}", "is_flagged": 0})

    df = pd.DataFrame(rows)
    result = detect_spikes(df, z_threshold=3.0, min_count=5)
    spike_day = result[result["day"] == "2025-01-16"]
    assert len(spike_day) == 1
    assert bool(spike_day.iloc[0]["is_spike"]) is True
    assert spike_day.iloc[0]["spike_severity"] in ["Low", "Medium", "High", "Critical"]


def test_detect_spikes_no_false_alarm_on_stable_merchant():
    rows = []
    merchant = "M_STABLE"
    for day in range(20):
        for _ in range(3):  # consistently 3 suspicious txns/day, no spike
            rows.append({"merchant_id": merchant, "day": f"2025-01-{day+1:02d}", "is_flagged": 1})
    df = pd.DataFrame(rows)
    result = detect_spikes(df, z_threshold=3.0, min_count=5)
    assert result["is_spike"].sum() == 0
