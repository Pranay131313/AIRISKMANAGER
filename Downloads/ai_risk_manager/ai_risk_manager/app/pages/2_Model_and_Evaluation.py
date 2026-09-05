"""
Model & Evaluation page -- explains the ML methodology, evaluation
protocol, and reports the honest held-out test results.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import pandas as pd
import plotly.express as px

from app.data_service import get_scored_data

st.set_page_config(page_title="Model & Evaluation", page_icon="📐", layout="wide")
st.title("📐 Model & Evaluation")

fdf, spikes, metrics, model, feature_cols = get_scored_data()

st.header("1. Problem & Approach")
st.markdown("""
This system detects **statistically unusual transactions and sudden merchant-level
spikes** in fraud-like activity, so a human fraud/risk team can investigate. It is a
**defense-only decision-support tool** — it never blocks, executes, or automates any
action on payment systems.
""")

st.header("2. Algorithm: Random Forest Classifier")
st.markdown("""
We use a **Random Forest** (`scikit-learn`, `class_weight="balanced"`), trained on
engineered features per transaction. Reasons for this choice:

- **Non-linear interactions**: fraud is rarely a single extreme feature — it's
  combinations (e.g. high amount **and** unfamiliar device **and** odd hour).
  Tree ensembles capture this naturally without manual interaction terms.
- **No feature scaling required**, unlike linear/SVM models, which simplifies the
  pipeline given a mix of counts, ratios, and z-scores.
- **Class-conditional probabilities** let us tune a decision threshold for a
  precision/recall trade-off rather than being stuck with a fixed cutoff.
- **`feature_importances_`** give a straightforward, if approximate, explainability
  layer for the alert-investigation view.
- Handles the **~2% class imbalance** via `class_weight="balanced"` rather than
  oversampling, which avoids duplicating rows across the train/val/test boundary.

Isolation Forest was also considered (and is a good complementary *unsupervised*
option when fraud labels are unavailable in production), but since this synthetic
dataset includes labels, a supervised model gives materially better precision/recall
for a comparable amount of engineering effort.
""")

st.header("3. Feature Engineering")
st.markdown("""
All features are computed using **only backward-looking / already-available
information** relative to each transaction's timestamp (expanding means, trailing
rolling windows) — never future transactions — to avoid leakage:
""")
feat_desc = pd.DataFrame({
    "Feature": [
        "amount_zscore_vs_merchant", "merchant_txn_count_60min", "merchant_txn_count_24h",
        "device_txn_count_60min", "merchant_failed_rate_24h", "merchant_spike_ratio",
        "is_geo_anomaly", "is_night", "is_failed", "device_running_count", "customer_running_count",
    ],
    "Description": [
        "How unusual this transaction's amount is vs. the merchant's running average",
        "Transactions from this merchant in the trailing 60 minutes (burst detection)",
        "Transactions from this merchant in the trailing 24 hours",
        "Transactions from this device in the trailing 60 minutes (device burst / card testing)",
        "Merchant's failed-transaction rate over the trailing 24 hours (probing pattern)",
        "Today's transaction volume for this merchant vs. its typical daily volume (spike ratio)",
        "Whether the transaction location differs from the merchant's usual/home location",
        "Whether the transaction occurred late night / early morning",
        "Whether the transaction failed",
        "How many prior transactions this device has made (new/rare devices are riskier)",
        "How many prior transactions this customer has made",
    ]
})
st.dataframe(feat_desc, hide_index=True, use_container_width=True)

st.header("4. Train / Validation / Test Methodology")
st.markdown(f"""
We use a **chronological (time-based) split**, not a random split, because fraud
spikes are time-clustered — a random split would leak information about a spike
across train and test. The dataset's **{metrics['train_days'].__len__() + metrics['val_days'].__len__() + metrics['test_days'].__len__()} days**
are split as:

| Split | Days | Rows | Fraud rate |
|---|---|---|---|
| Train | {len(metrics['train_days'])} days | {metrics['train_size']:,} | {metrics['train_fraud_rate']*100:.2f}% |
| Validation | {len(metrics['val_days'])} days | {metrics['val_size']:,} | {metrics['val_fraud_rate']*100:.2f}% |
| **Test (held out)** | {len(metrics['test_days'])} days | {metrics['test_size']:,} | {metrics['test_fraud_rate']*100:.2f}% |

- The model is **fit only on Train**.
- The **decision threshold** (converting probability → flagged/not-flagged) is tuned
  **only on Validation**, targeting ≥75% recall while maximizing precision at that
  recall level.
- The **Test set is touched exactly once**, after the threshold is locked in, purely
  to report final numbers below. It is never used for any tuning decision.
""")

st.header("5. Held-Out Test Results (Final, Honest)")
tm = metrics["test_metrics"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precision", f"{tm['precision']*100:.1f}%")
c2.metric("Recall", f"{tm['recall']*100:.1f}%")
c3.metric("F1-score", f"{tm['f1_score']*100:.1f}%")
c4.metric("Accuracy", f"{tm['accuracy']*100:.2f}%")

cm = tm["confusion_matrix"]
cm_df = pd.DataFrame(
    [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
    index=["Actual: Legitimate", "Actual: Fraud"],
    columns=["Predicted: Legitimate", "Predicted: Fraud"],
)
st.subheader("Confusion Matrix (Test Set)")
fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", aspect="auto")
fig.update_layout(height=350)
st.plotly_chart(fig, use_container_width=True)

st.metric("False Positive Rate", f"{tm['false_positive_rate']*100:.3f}%")
st.metric("ROC-AUC", f"{tm.get('roc_auc', 0)*100:.2f}%")

with st.expander("Compare: default 0.5 threshold vs. tuned threshold (both on test set)"):
    default_tm = metrics["test_metrics_default_threshold_0.5"]
    comp = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-score", "False Positive Rate"],
        f"Tuned threshold ({metrics['chosen_threshold']:.3f})": [
            f"{tm['precision']*100:.1f}%", f"{tm['recall']*100:.1f}%",
            f"{tm['f1_score']*100:.1f}%", f"{tm['false_positive_rate']*100:.3f}%"
        ],
        "Default threshold (0.5)": [
            f"{default_tm['precision']*100:.1f}%", f"{default_tm['recall']*100:.1f}%",
            f"{default_tm['f1_score']*100:.1f}%", f"{default_tm['false_positive_rate']*100:.3f}%"
        ],
    })
    st.dataframe(comp, hide_index=True, use_container_width=True)
    st.caption("Threshold was chosen on the VALIDATION set only, then applied unchanged to test.")

st.header("6. Global Feature Importance")
fi = pd.Series(metrics["feature_importances"]).sort_values(ascending=True).tail(12)
fig2 = px.bar(x=fi.values, y=fi.index, orientation="h",
              labels={"x": "Importance", "y": ""}, color=fi.values, color_continuous_scale="Reds")
fig2.update_layout(height=420, showlegend=False, coloraxis_showscale=False)
st.plotly_chart(fig2, use_container_width=True)

st.header("7. Limitations")
st.markdown("""
- **Synthetic data**: patterns are realistic but simplified vs. real-world fraud,
  which evolves adversarially and would require continuous retraining.
- **Recall < 100%**: the model will miss some fraud (see false negatives in the
  confusion matrix above) — it is a *lead generator*, not a final verdict.
- **Precision is high but not perfect**: some legitimate transactions will still be
  flagged, incurring a false-positive/friction cost (see dashboard).
- **Spike detection threshold (z-score)** is a simple statistical rule, not a second
  learned model — chosen for auditability over marginal accuracy gains.
- **No real payment-network integration**: this is a demo/decision-support layer,
  not a production fraud-blocking system.
""")

st.header("8. Future Scope")
st.markdown("""
- Add gradient-boosted trees (XGBoost/LightGBM) and compare against Random Forest
  under the same time-based split.
- Incorporate graph features (shared devices/customers across merchants) for
  collusion-style fraud rings.
- Add SHAP-based per-transaction explanations in place of the approximate
  global-importance view.
- Introduce periodic retraining with drift monitoring, since fraud patterns evolve.
- A/B test alternate thresholds against real analyst feedback (confirmed fraud vs.
  false positive labels) to refine the cost model.
""")
