
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("Loading Olist datasets...")

items = pd.read_csv("data/raw/olist_order_items_dataset.csv")
payments = pd.read_csv("data/raw/olist_order_payments_dataset.csv")

# Aggregate order item information
order_items = items.groupby("order_id").agg(
    item_count=("product_id", "count"),
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum")
).reset_index()

# Aggregate payment information
order_payments = payments.groupby("order_id").agg(
    payment_value=("payment_value", "sum"),
    installments=("payment_installments", "max")
).reset_index()

# Combine order-level information
data = order_items.merge(
    order_payments,
    on="order_id",
    how="inner"
)

features = [
    "item_count",
    "total_price",
    "total_freight",
    "payment_value",
    "installments"
]

data[features] = data[features].replace(
    [np.inf, -np.inf], np.nan
)

data = data.dropna(subset=features).copy()

# Train Isolation Forest
print("Training Isolation Forest model...")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("isolation_forest", IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(data[features])

# Predict anomalies
predictions = model.predict(data[features])

data["Anomaly"] = np.where(predictions == -1, "Unusual", "Normal")
data["Anomaly_Score"] = model.decision_function(data[features])

# Save reports and model
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

data.to_csv(
    "reports/anomaly_detection_results.csv",
    index=False
)

joblib.dump(
    model,
    "models/anomaly_detection_model.joblib"
)

# Summary
print("\n===== ANOMALY DETECTION RESULTS =====")
print("\nOrder Classification:")
print(data["Anomaly"].value_counts())

print("\nSample Unusual Orders:")
print(
    data[data["Anomaly"] == "Unusual"][
        ["order_id"] + features + ["Anomaly_Score"]
    ].head(10).to_string(index=False)
)

# Visualization
plt.figure(figsize=(9, 6))

normal = data[data["Anomaly"] == "Normal"]
unusual = data[data["Anomaly"] == "Unusual"]

plt.scatter(
    normal["total_price"],
    normal["payment_value"],
    alpha=0.3,
    label="Normal"
)

plt.scatter(
    unusual["total_price"],
    unusual["payment_value"],
    alpha=0.7,
    label="Unusual"
)

plt.xlabel("Total Product Price (BRL)")
plt.ylabel("Payment Value (BRL)")
plt.title("E-Commerce Order Anomaly Detection")
plt.legend()
plt.tight_layout()

plt.savefig(
    "reports/anomaly_detection.png",
    dpi=300
)

plt.close()

print("\nFiles saved successfully!")
