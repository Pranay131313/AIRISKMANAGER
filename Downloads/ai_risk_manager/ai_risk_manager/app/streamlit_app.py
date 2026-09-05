"""
streamlit_app.py
-----------------
AI Risk Manager -- main dashboard (fraud-spike detection for merchants).

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.data_service import get_scored_data
from ml.risk_scoring import compute_false_positive_cost

st.set_page_config(
    page_title="AI Risk Manager | Fraud Spike Detection",
    page_icon="🛡️",
    layout="wide",
)

# ---------- Minimal fintech-style styling ----------
st.markdown("""
<style>
    .risk-card {
        background-color: #111827;
        border-radius: 10px;
        padding: 18px 20px;
        border: 1px solid #1f2937;
    }
    .metric-label { color: #9ca3af; font-size: 0.85rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .disclaimer {
        background-color: #1f2937; border-left: 4px solid #f59e0b;
        padding: 10px 14px; border-radius: 6px; font-size: 0.85rem; color: #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ AI Risk Manager")
st.caption("Defense-only fraud-spike detection & investigation for merchants (India) — synthetic demo data")

fdf, spikes, metrics, model, feature_cols = get_scored_data()

st.markdown(
    '<div class="disclaimer">⚠️ This model flags <b>statistically unusual</b> transactions for human review. '
    'It is <b>not perfectly accurate</b> — see the "Model & Evaluation" page for real precision/recall on a held-out test set, '
    'and treat every alert as a lead for investigation, not a verdict.</div>',
    unsafe_allow_html=True,
)
st.write("")

# ---------------- Top-line KPIs ----------------
total_txns = len(fdf)
suspicious_txns = int(fdf["is_flagged"].sum())
fraud_detection_rate = metrics["test_metrics"]["recall"]
overall_risk_score = fdf["risk_score"].mean()
n_spikes = int(spikes["is_spike"].sum())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Transactions", f"{total_txns:,}")
col2.metric("Suspicious Transactions", f"{suspicious_txns:,}", help="Transactions flagged above the model's tuned decision threshold")
col3.metric("Fraud Detection Rate (Recall)", f"{fraud_detection_rate*100:.1f}%", help="Measured on the held-out TEST set")
col4.metric("Avg. Risk Score", f"{overall_risk_score:.1f}/100")
col5.metric("Active Fraud Spikes", n_spikes, delta=None)

st.divider()

# ---------------- Fraud spikes over time ----------------
left, right = st.columns([2, 1])

with left:
    st.subheader("📈 Suspicious Transactions Over Time")
    daily = fdf.groupby("day").agg(
        total=("transaction_id", "count"),
        suspicious=("is_flagged", "sum"),
    ).reset_index()
    daily["day"] = pd.to_datetime(daily["day"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["total"], name="Total transactions",
                              line=dict(color="#374151", width=1), fill="tozeroy", fillcolor="rgba(55,65,81,0.15)"))
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["suspicious"], name="Suspicious (flagged)",
                              line=dict(color="#ef4444", width=2)))

    spike_days = spikes[spikes["is_spike"]].groupby("day")["suspicious_count"].sum().reset_index()
    if len(spike_days):
        spike_days["day"] = pd.to_datetime(spike_days["day"])
        fig.add_trace(go.Scatter(
            x=spike_days["day"], y=spike_days["suspicious_count"], mode="markers",
            name="Spike day", marker=dict(color="#f59e0b", size=12, symbol="triangle-up",
                                           line=dict(width=1, color="white"))
        ))
    fig.update_layout(height=380, legend=dict(orientation="h", y=1.15), margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("🚨 Severity Breakdown")
    sev_counts = fdf[fdf["is_flagged"] == 1]["severity"].value_counts().reindex(
        ["Critical", "High", "Medium", "Low"]).fillna(0)
    colors = {"Critical": "#dc2626", "High": "#f97316", "Medium": "#eab308", "Low": "#22c55e"}
    fig2 = go.Figure(go.Bar(
        x=sev_counts.values, y=sev_counts.index, orientation="h",
        marker_color=[colors[s] for s in sev_counts.index],
        text=sev_counts.values.astype(int), textposition="outside",
    ))
    fig2.update_layout(height=380, margin=dict(t=10, b=10), xaxis_title="Flagged transactions")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------- High-risk merchants / devices / customers ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("🏪 High-Risk Merchants")
    merch = fdf.groupby("merchant_id").agg(
        suspicious=("is_flagged", "sum"), total=("transaction_id", "count"),
        avg_risk=("risk_score", "mean"),
    ).reset_index()
    merch["flag_rate_%"] = (merch["suspicious"] / merch["total"] * 100).round(2)
    merch = merch.sort_values("suspicious", ascending=False).head(8)
    st.dataframe(
        merch[["merchant_id", "suspicious", "total", "flag_rate_%", "avg_risk"]]
        .rename(columns={"merchant_id": "Merchant", "suspicious": "Flagged",
                          "total": "Total Txns", "avg_risk": "Avg Risk"}),
        hide_index=True, use_container_width=True,
    )

with c2:
    st.subheader("📱 High-Risk Devices")
    dev = fdf[fdf["is_flagged"] == 1].groupby("device_id").agg(
        suspicious=("is_flagged", "sum"), avg_risk=("risk_score", "mean")
    ).reset_index().sort_values("suspicious", ascending=False).head(8)
    st.dataframe(
        dev.rename(columns={"device_id": "Device", "suspicious": "Flagged Txns", "avg_risk": "Avg Risk"}),
        hide_index=True, use_container_width=True,
    )

with c3:
    st.subheader("👤 High-Risk Customers")
    cust = fdf[fdf["is_flagged"] == 1].groupby("customer_id").agg(
        suspicious=("is_flagged", "sum"), avg_risk=("risk_score", "mean")
    ).reset_index().sort_values("suspicious", ascending=False).head(8)
    st.dataframe(
        cust.rename(columns={"customer_id": "Customer", "suspicious": "Flagged Txns", "avg_risk": "Avg Risk"}),
        hide_index=True, use_container_width=True,
    )

st.divider()

# ---------------- Model quality + false-positive cost ----------------
c4, c5 = st.columns([1, 1])

with c4:
    st.subheader("🎯 Model Performance (held-out test set)")
    tm = metrics["test_metrics"]
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Precision", f"{tm['precision']*100:.1f}%")
    mcol2.metric("Recall", f"{tm['recall']*100:.1f}%")
    mcol3.metric("F1-score", f"{tm['f1_score']*100:.1f}%")
    mcol4.metric("False Positive Rate", f"{tm['false_positive_rate']*100:.3f}%")
    st.caption(
        f"Confusion matrix (test set): TP={tm['confusion_matrix']['tp']}, "
        f"FP={tm['confusion_matrix']['fp']}, FN={tm['confusion_matrix']['fn']}, "
        f"TN={tm['confusion_matrix']['tn']}. See **Model & Evaluation** page for full methodology."
    )

with c5:
    st.subheader("💰 False-Positive Cost vs. Fraud Prevented")
    cost_per_fp = st.slider("Cost per false positive (₹)", 50, 1000, 150, step=50)
    review_rate = st.slider("Review success rate on flagged fraud", 0.5, 1.0, 0.9, step=0.05)

    fraud_amounts = fdf[(fdf["is_flagged"] == 1) & (fdf["fraud_label"] == 1)]["amount"]
    avg_fraud_amount = fraud_amounts.mean() if len(fraud_amounts) else 0

    cost_info = compute_false_positive_cost(
        n_false_positives=tm["confusion_matrix"]["fp"],
        n_true_positives=tm["confusion_matrix"]["tp"],
        avg_fraud_amount=avg_fraud_amount,
        avg_legit_amount=fdf["amount"].mean(),
        cost_per_false_positive=cost_per_fp,
        review_success_rate=review_rate,
    )
    st.metric("Est. False-Positive Cost", f"₹{cost_info['estimated_false_positive_cost']:,.0f}")
    st.metric("Est. Fraud Loss Prevented", f"₹{cost_info['estimated_fraud_loss_prevented']:,.0f}")
    st.metric("Net Estimated Benefit", f"₹{cost_info['net_estimated_benefit']:,.0f}")

st.info("👉 Go to **Alerts & Investigation** in the sidebar to drill into individual fraud spikes.")
