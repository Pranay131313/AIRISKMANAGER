"""
generate_dataset.py
--------------------
Generates a realistic SYNTHETIC transaction dataset for Indian merchants,
with embedded normal behaviour and several distinct FRAUD PATTERNS
(including sudden merchant-level fraud spikes). No real data is used.

This is for building/evaluating a DEFENSIVE fraud detector only.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Surat", "Nagpur",
    "Indore", "Kochi", "Chandigarh"
]

PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Wallet"]

N_MERCHANTS = 60
N_CUSTOMERS = 4000
N_DEVICES = 4500

MERCHANT_IDS = [f"M{str(i).zfill(4)}" for i in range(1, N_MERCHANTS + 1)]
CUSTOMER_IDS = [f"C{str(i).zfill(5)}" for i in range(1, N_CUSTOMERS + 1)]
DEVICE_IDS = [f"D{str(i).zfill(5)}" for i in range(1, N_DEVICES + 1)]

# Each merchant gets a "home city" and a typical average ticket size,
# so normal behaviour is merchant-specific (more realistic).
MERCHANT_PROFILE = {
    m: {
        "home_city": random.choice(CITIES),
        "avg_ticket": np.random.uniform(300, 4000),
        "daily_volume": np.random.randint(10, 60),
    }
    for m in MERCHANT_IDS
}

START_DATE = datetime(2025, 1, 1)
N_DAYS = 60


def _new_txn_id():
    return "T" + uuid.uuid4().hex[:12].upper()


def generate_normal_transaction(merchant_id, day_offset):
    profile = MERCHANT_PROFILE[merchant_id]
    ts = START_DATE + timedelta(
        days=int(day_offset),
        hours=int(np.random.choice(range(6, 23))),  # most legit txns happen 6am-11pm
        minutes=int(np.random.randint(0, 60)),
        seconds=int(np.random.randint(0, 60)),
    )
    amount = max(20, np.random.normal(profile["avg_ticket"], profile["avg_ticket"] * 0.35))
    # 90% of normal transactions happen near the merchant's home city
    location = profile["home_city"] if np.random.rand() < 0.9 else random.choice(CITIES)
    status = np.random.choice(["SUCCESS", "FAILED"], p=[0.95, 0.05])

    return {
        "transaction_id": _new_txn_id(),
        "timestamp": ts,
        "merchant_id": merchant_id,
        "amount": round(amount, 2),
        "payment_method": np.random.choice(PAYMENT_METHODS, p=[0.45, 0.2, 0.2, 0.1, 0.05]),
        "location": location,
        "device_id": random.choice(DEVICE_IDS),
        "customer_id": random.choice(CUSTOMER_IDS),
        "transaction_status": status,
        "fraud_label": 0,
    }


def generate_fraud_transaction(merchant_id, day_offset, pattern):
    """
    Generates a fraudulent transaction following one of several realistic
    fraud patterns. Patterns are labelled purely for dataset realism /
    explainability downstream -- no offensive logic is produced.
    """
    profile = MERCHANT_PROFILE[merchant_id]
    ts = START_DATE + timedelta(
        days=int(day_offset),
        hours=int(np.random.randint(0, 24)),
        minutes=int(np.random.randint(0, 60)),
        seconds=int(np.random.randint(0, 60)),
    )

    if pattern == "high_value_outlier":
        amount = profile["avg_ticket"] * np.random.uniform(6, 15)
        location = profile["home_city"]
        device = random.choice(DEVICE_IDS)
        customer = random.choice(CUSTOMER_IDS)
        status = np.random.choice(["SUCCESS", "FAILED"], p=[0.7, 0.3])

    elif pattern == "device_burst":
        # one device used for many rapid transactions across many customers
        amount = max(20, np.random.normal(profile["avg_ticket"], profile["avg_ticket"] * 0.5))
        location = profile["home_city"]
        device = "D_FRAUD_DEVICE"
        customer = random.choice(CUSTOMER_IDS)
        status = np.random.choice(["SUCCESS", "FAILED"], p=[0.6, 0.4])

    elif pattern == "geo_anomaly":
        amount = max(20, np.random.normal(profile["avg_ticket"], profile["avg_ticket"] * 0.4))
        far_cities = [c for c in CITIES if c != profile["home_city"]]
        location = random.choice(far_cities)
        device = random.choice(DEVICE_IDS)
        customer = random.choice(CUSTOMER_IDS)
        status = np.random.choice(["SUCCESS", "FAILED"], p=[0.65, 0.35])

    elif pattern == "failed_probing":
        # card/credential testing: many rapid low-value failed attempts
        amount = round(np.random.uniform(10, 100), 2)
        location = profile["home_city"]
        device = "D_PROBE_DEVICE"
        customer = random.choice(CUSTOMER_IDS)
        status = np.random.choice(["SUCCESS", "FAILED"], p=[0.15, 0.85])

    else:  # "merchant_spike" -- sudden burst of many fraud-like txns for one merchant
        amount = max(20, np.random.normal(profile["avg_ticket"] * 1.5, profile["avg_ticket"] * 0.6))
        location = profile["home_city"]
        device = random.choice(DEVICE_IDS)
        customer = random.choice(CUSTOMER_IDS)
        status = np.random.choice(["SUCCESS", "FAILED"], p=[0.55, 0.45])

    return {
        "transaction_id": _new_txn_id(),
        "timestamp": ts,
        "merchant_id": merchant_id,
        "amount": round(amount, 2),
        "payment_method": np.random.choice(PAYMENT_METHODS),
        "location": location,
        "device_id": device,
        "customer_id": customer,
        "transaction_status": status,
        "fraud_label": 1,
    }


def generate_dataset(save_path="data/transactions.csv"):
    rows = []

    # --- Normal transaction volume across 90 days ---
    for day in range(N_DAYS):
        for merchant_id in MERCHANT_IDS:
            n_txns = np.random.poisson(MERCHANT_PROFILE[merchant_id]["daily_volume"])
            for _ in range(n_txns):
                rows.append(generate_normal_transaction(merchant_id, day))

    # --- Scattered "background" fraud (few percent, always present) ---
    n_background_fraud = int(len(rows) * 0.015)
    for _ in range(n_background_fraud):
        merchant_id = random.choice(MERCHANT_IDS)
        day = np.random.randint(0, N_DAYS)
        pattern = random.choice(["high_value_outlier", "geo_anomaly"])
        rows.append(generate_fraud_transaction(merchant_id, day, pattern))

    # --- Deliberate fraud SPIKES: a handful of merchant/day combos get hit hard ---
    n_spikes = 14
    spike_days_merchants = set()
    for _ in range(n_spikes):
        merchant_id = random.choice(MERCHANT_IDS)
        day = np.random.randint(5, N_DAYS - 5)
        pattern = random.choice(["device_burst", "failed_probing", "merchant_spike"])
        n_fraud_in_spike = np.random.randint(25, 90)
        spike_days_merchants.add((merchant_id, day))
        for _ in range(n_fraud_in_spike):
            rows.append(generate_fraud_transaction(merchant_id, day, pattern))

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(save_path, index=False)

    print(f"Dataset generated: {len(df)} transactions")
    print(f"Fraud transactions: {df['fraud_label'].sum()} "
          f"({100 * df['fraud_label'].mean():.2f}%)")
    print(f"Spike merchant/day combos: {sorted(spike_days_merchants)}")
    return df


if __name__ == "__main__":
    generate_dataset()
