import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import pytest

from ml.train_model import compute_metrics, time_based_split, choose_threshold
from ml.features import load_raw, get_feature_matrix


MODEL_PATH = "models/fraud_model.joblib"
METRICS_PATH = "models/metrics.json"


def test_compute_metrics_basic_case():
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 0, 1]
    m = compute_metrics(y_true, y_pred)
    assert m["confusion_matrix"]["tp"] == 2
    assert m["confusion_matrix"]["fp"] == 1
    assert m["confusion_matrix"]["fn"] == 1
    assert m["confusion_matrix"]["tn"] == 2
    assert 0 <= m["precision"] <= 1
    assert 0 <= m["recall"] <= 1
    assert 0 <= m["f1_score"] <= 1


def test_compute_metrics_perfect_predictions():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    m = compute_metrics(y_true, y_pred)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["false_positive_rate"] == 0.0


def test_time_based_split_no_overlap():
    raw = load_raw()
    _, _, fdf = get_feature_matrix(raw.head(5000))
    train_mask, val_mask, test_mask, day_sets = time_based_split(fdf)
    train_days, val_days, test_days = day_sets
    assert train_days.isdisjoint(val_days)
    assert val_days.isdisjoint(test_days)
    assert train_days.isdisjoint(test_days)
    # every row assigned to exactly one split
    assert (train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int)).eq(1).all()


def test_choose_threshold_returns_valid_probability_range():
    y_val = np.array([0] * 90 + [1] * 10)
    proba = np.concatenate([np.random.uniform(0, 0.3, 90), np.random.uniform(0.6, 1.0, 10)])
    threshold = choose_threshold(y_val, proba, target_recall=0.5)
    assert 0.0 <= threshold <= 1.0


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not yet trained")
def test_saved_model_metrics_are_realistic():
    """Sanity check the persisted, held-out test metrics are plausible (not fabricated placeholders)."""
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    tm = metrics["test_metrics"]
    assert 0 < tm["precision"] <= 1
    assert 0 < tm["recall"] <= 1
    assert metrics["test_size"] > 0
    # test set must not overlap train/val days
    assert set(metrics["test_days"]).isdisjoint(set(metrics["train_days"]))
    assert set(metrics["test_days"]).isdisjoint(set(metrics["val_days"]))
