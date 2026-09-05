"""
This app uses Streamlit (no separate REST server), so these tests cover the
equivalent "API layer" -- the data-access functions the UI calls
(db.database and ml.evaluate.score_full_dataset) -- end to end.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os as _os
import pytest

from db.database import init_db, upsert_alert_status, get_alert_status, get_all_statuses, DB_PATH


@pytest.fixture(autouse=True)
def clean_db():
    if _os.path.exists(DB_PATH):
        _os.remove(DB_PATH)
    init_db()
    yield
    if _os.path.exists(DB_PATH):
        _os.remove(DB_PATH)


def test_init_db_creates_table():
    assert _os.path.exists(DB_PATH)


def test_upsert_and_get_alert_status():
    upsert_alert_status("A1", "M0001", "2025-01-05", "Under Review", "checking device logs")
    result = get_alert_status("A1")
    assert result["status"] == "Under Review"
    assert result["analyst_note"] == "checking device logs"


def test_get_alert_status_defaults_when_missing():
    result = get_alert_status("DOES_NOT_EXIST")
    assert result["status"] == "Open"
    assert result["analyst_note"] == ""


def test_upsert_alert_status_updates_existing():
    upsert_alert_status("A2", "M0002", "2025-01-06", "Open", "")
    upsert_alert_status("A2", "M0002", "2025-01-06", "Confirmed Fraud", "verified with merchant")
    result = get_alert_status("A2")
    assert result["status"] == "Confirmed Fraud"
    assert result["analyst_note"] == "verified with merchant"


def test_get_all_statuses_returns_multiple():
    upsert_alert_status("A3", "M0003", "2025-01-07", "Resolved", "")
    upsert_alert_status("A4", "M0004", "2025-01-08", "False Positive", "")
    all_statuses = get_all_statuses()
    assert "A3" in all_statuses
    assert "A4" in all_statuses
    assert all_statuses["A3"]["status"] == "Resolved"


@pytest.mark.skipif(
    not os.path.exists("models/fraud_model.joblib"),
    reason="Model must be trained first (run: python -m ml.train_model)"
)
def test_score_full_dataset_end_to_end():
    from ml.evaluate import score_full_dataset
    fdf, spikes, metrics, model, feature_cols = score_full_dataset()
    assert len(fdf) > 0
    assert "risk_score" in fdf.columns
    assert "is_flagged" in fdf.columns
    assert fdf["risk_score"].between(0, 100).all()
    assert set(fdf["severity"].unique()).issubset({"Low", "Medium", "High", "Critical"})
    assert "is_spike" in spikes.columns
