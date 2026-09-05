import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_auc_score, precision_recall_curve
)

from ml.features import load_raw, get_feature_matrix, FEATURE_COLUMNS

MODEL_PATH = "models/fraud_model.joblib"
METRICS_PATH = "models/metrics.json"
FEATURE_META_PATH = "models/feature_columns.json"


def time_based_split(fdf: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    days_sorted = sorted(fdf["day"].unique())
    n = len(days_sorted)
    train_days = set(days_sorted[: int(n * train_frac)])
    val_days = set(days_sorted[int(n * train_frac): int(n * (train_frac + val_frac))])
    test_days = set(days_sorted[int(n * (train_frac + val_frac)):])

    train_mask = fdf["day"].isin(train_days)
    val_mask = fdf["day"].isin(val_days)
    test_mask = fdf["day"].isin(test_days)
    return train_mask, val_mask, test_mask, (train_days, val_days, test_days)


def compute_metrics(y_true, y_pred, y_proba=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if y_proba is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def choose_threshold(y_val, val_proba, target_recall=0.75):
    """
    Pick a decision threshold on the VALIDATION set only (never on test),
    aiming for at least `target_recall` while maximizing precision at that
    recall level. This is a simple, defensible way to tune away from the
    default 0.5 cutoff, which performs poorly under heavy class imbalance.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
    # precision_recall_curve returns thresholds of len(n-1); align arrays
    best_threshold = 0.5
    best_precision = -1
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if r >= target_recall and p > best_precision:
            best_precision = p
            best_threshold = t
    return float(best_threshold)


def train():
    raw = load_raw()
    X, y, fdf = get_feature_matrix(raw)
    pm_cols = [c for c in fdf.columns if c.startswith("pm_")]
    feature_cols = [c for c in FEATURE_COLUMNS if c in fdf.columns] + pm_cols

    train_mask, val_mask, test_mask, day_sets = time_based_split(fdf)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Train: {len(X_train)} rows ({y_train.mean()*100:.2f}% fraud)")
    print(f"Val:   {len(X_val)} rows ({y_val.mean()*100:.2f}% fraud)")
    print(f"Test:  {len(X_test)} rows ({y_test.mean()*100:.2f}% fraud)")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # --- Threshold tuning on VALIDATION set only ---
    val_proba = model.predict_proba(X_val)[:, 1]
    threshold = choose_threshold(y_val, val_proba, target_recall=0.75)
    val_pred = (val_proba >= threshold).astype(int)
    val_metrics = compute_metrics(y_val, val_pred, val_proba)
    print("\n--- Validation metrics (threshold tuned here) ---")
    print(json.dumps(val_metrics, indent=2))

    # --- Final, ONE-TIME evaluation on held-out TEST set ---
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    test_metrics = compute_metrics(y_test, test_pred, test_proba)
    print("\n--- HELD-OUT TEST metrics (final, reported in README) ---")
    print(json.dumps(test_metrics, indent=2))

    # Also report default 0.5 threshold on test for transparency/comparison
    test_pred_default = (test_proba >= 0.5).astype(int)
    test_metrics_default = compute_metrics(y_test, test_pred_default, test_proba)

    feature_importances = dict(
        sorted(
            zip(feature_cols, model.feature_importances_.tolist()),
            key=lambda x: x[1], reverse=True
        )
    )

    results = {
        "chosen_threshold": threshold,
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "train_fraud_rate": float(y_train.mean()),
        "val_fraud_rate": float(y_val.mean()),
        "test_fraud_rate": float(y_test.mean()),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_metrics_default_threshold_0.5": test_metrics_default,
        "feature_importances": feature_importances,
        "train_days": [str(d) for d in sorted(day_sets[0])],
        "val_days": [str(d) for d in sorted(day_sets[1])],
        "test_days": [str(d) for d in sorted(day_sets[2])],
    }

    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(FEATURE_META_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
    return model, results


if __name__ == "__main__":
    train()
