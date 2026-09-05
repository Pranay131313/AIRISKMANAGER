"""
Alerts & Investigation page.
Lists detected fraud spikes as alerts; selecting one shows a full
investigation view with explainability and defensive recommendations.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import pandas as pd
import plotly.express as px

from app.data_service import get_scored_data
from ml.spike_detection import explain_spike
from ml.risk_scoring import get_recommendations, top_contributing_features
from db.database import upsert_alert_status, get_alert_status

st.set_page_config(page_title="Alerts & Investigation", layout="wide")
st.title(" Alerts & Investigation")
st.caption("Each alert represents a detected fraud spike for a merchant on a given day.")

fdf, spikes, metrics, model, feature_cols = get_scored_data()

alert_rows = spikes[spikes["is_spike"]].sort_values("spike_z_score", ascending=False).reset_index(drop=True)

if len(alert_rows) == 0:
    st.success("No active fraud spikes detected in the current dataset.")
    st.stop()

alert_rows["alert_id"] = alert_rows["merchant_id"] + "_" + alert_rows["day"].astype(str)

st.subheader(f"Recent Alerts ({len(alert_rows)})")

severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
alert_rows["_sev_order"] = alert_rows["spike_severity"].map(severity_order)
alert_rows = alert_rows.sort_values(["_sev_order", "spike_z_score"], ascending=[True, False])

display_df = alert_rows[["alert_id", "merchant_id", "day", "spike_severity",
                          "suspicious_count", "total_txns", "spike_z_score"]].copy()
display_df.columns = ["Alert ID", "Merchant", "Day", "Severity", "Suspicious Txns", "Total Txns", "Z-score"]
display_df["Z-score"] = display_df["Z-score"].round(2)

sev_filter = st.multiselect("Filter by severity", ["Critical", "High", "Medium", "Low"],
                             default=["Critical", "High", "Medium", "Low"])
display_df = display_df[display_df["Severity"].isin(sev_filter)]

st.dataframe(display_df, hide_index=True, use_container_width=True, height=250)

st.divider()

# ------------- Alert selector -------------
alert_options = alert_rows["alert_id"].tolist()
if not alert_options:
    st.warning("No alerts match the selected severity filter.")
    st.stop()

selected_alert = st.selectbox(" Select an alert to investigate", alert_options)

alert = alert_rows[alert_rows["alert_id"] == selected_alert].iloc[0]
merchant_id = alert["merchant_id"]
day = alert["day"]

st.header(f"Investigation: {merchant_id} — {day}")

sev_color = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
st.markdown(f"### {sev_color.get(alert['spike_severity'],'')} Severity: **{alert['spike_severity']}**")

# ------------- Alert reason -------------
merchant_txns_day = fdf[(fdf["merchant_id"] == merchant_id) & (fdf["day"].astype(str) == str(day))]
flagged_txns_day = merchant_txns_day[merchant_txns_day["is_flagged"] == 1]
amount_involved = flagged_txns_day["amount"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Risk Score (avg)", f"{flagged_txns_day['risk_score'].mean():.1f}/100" if len(flagged_txns_day) else "N/A")
col2.metric("Suspicious Transactions", int(alert["suspicious_count"]))
col3.metric("Amount Involved", f"₹{amount_involved:,.0f}")
col4.metric("Time Period", str(day))

st.subheader(" Alert Reason")
for reason in explain_spike(alert, avg_amount=flagged_txns_day["amount"].mean() if len(flagged_txns_day) else None):
    st.write(f"- {reason}")

# ------------- Transaction pattern chart -------------
st.subheader(" Transaction Pattern on Alert Day")
if len(merchant_txns_day):
    hourly = merchant_txns_day.groupby(["hour", "is_flagged"]).size().reset_index(name="count")
    hourly["is_flagged"] = hourly["is_flagged"].map({0: "Normal", 1: "Suspicious"})
    fig = px.bar(hourly, x="hour", y="count", color="is_flagged",
                 color_discrete_map={"Normal": "#374151", "Suspicious": "#ef4444"},
                 labels={"hour": "Hour of day", "count": "Transactions"})
    fig.update_layout(height=320, legend_title="")
    st.plotly_chart(fig, use_container_width=True)

# ------------- Contributing features (explainability) -------------
st.subheader(" Contributing Features (Explainability)")
if len(flagged_txns_day):
    example_row = flagged_txns_day.sort_values("risk_score", ascending=False).iloc[[0]]
    X_row = example_row[feature_cols].fillna(0)
    contribs = top_contributing_features(model, X_row, feature_cols, top_n=6)
    contrib_df = pd.DataFrame(contribs, columns=["Feature", "Model Importance", "Value in this transaction"])
    contrib_df["Model Importance"] = (contrib_df["Model Importance"] * 100).round(1).astype(str) + "%"
    st.caption("Based on the single highest-risk transaction in this alert. Importance reflects the model's "
               "*global* reliance on each feature (approximate explainability for the hackathon build).")
    st.dataframe(contrib_df, hide_index=True, use_container_width=True)
else:
    st.write("No individually-flagged transactions found for this alert window.")

# ------------- Recommended defensive action -------------
st.subheader(" Recommended Defensive Action")
st.warning("These are investigative/defensive suggestions only. No automated blocking or offensive action is taken by this system.")
for rec in get_recommendations(alert["spike_severity"]):
    st.write(f" {rec}")

# ------------- Investigation status -------------
st.divider()
st.subheader(" Investigation Status")
current = get_alert_status(selected_alert)
status = st.selectbox("Status", ["Open", "Under Review", "Confirmed Fraud", "False Positive", "Resolved"],
                       index=["Open", "Under Review", "Confirmed Fraud", "False Positive", "Resolved"].index(current["status"]) if current["status"] in
                       ["Open", "Under Review", "Confirmed Fraud", "False Positive", "Resolved"] else 0)
note = st.text_area("Analyst note", value=current["analyst_note"])
if st.button(" Save investigation status"):
    upsert_alert_status(selected_alert, merchant_id, day, status, note)
    st.success("Saved.")

# ------------- Raw flagged transactions -------------
with st.expander("View raw flagged transactions for this alert"):
    cols = ["transaction_id", "timestamp", "amount", "payment_method", "location",
            "device_id", "customer_id", "transaction_status", "risk_score", "severity"]
    cols = [c for c in cols if c in flagged_txns_day.columns]
    st.dataframe(flagged_txns_day[cols].sort_values("risk_score", ascending=False),
                 hide_index=True, use_container_width=True)
