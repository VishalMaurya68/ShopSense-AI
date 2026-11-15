

import pandas as pd
import os

print("Loading Olist datasets...")

orders = pd.read_csv("data/raw/olist_orders_dataset.csv")
customers = pd.read_csv("data/raw/olist_customers_dataset.csv")
items = pd.read_csv("data/raw/olist_order_items_dataset.csv")

# Convert dates
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

orders = orders.dropna(
    subset=["order_purchase_timestamp"]
)

# Attach unique customer IDs
data = orders.merge(
    customers[["customer_id", "customer_unique_id", "customer_state"]],
    on="customer_id",
    how="left"
)

# Calculate order value
order_values = items.groupby("order_id").agg(
    order_value=("price", "sum"),
    freight_value=("freight_value", "sum")
).reset_index()

data = data.merge(order_values, on="order_id", how="left")

data["order_value"] = (
    data["order_value"].fillna(0) +
    data["freight_value"].fillna(0)
)

# Define observation windows
max_date = data["order_purchase_timestamp"].max()
# Keep the complete 180-day future window inside the dataset
snapshot_date = max_date - pd.Timedelta(days=180)

feature_start = snapshot_date - pd.Timedelta(days=180)

target_end = max_date

print("\nSnapshot date:", snapshot_date)
print("Feature window:", feature_start, "to", snapshot_date)
print("Target window:", snapshot_date, "to", target_end)

# Past 180 days: customer behaviour
past_data = data[
    (data["order_purchase_timestamp"] >= feature_start) &
    (data["order_purchase_timestamp"] < snapshot_date)
].copy()

# Next 90 days: future purchases
future_data = data[
    (data["order_purchase_timestamp"] >= snapshot_date) &
    (data["order_purchase_timestamp"] <= target_end)
].copy()

# Customer features from past data
customer_features = past_data.groupby("customer_unique_id").agg(
    last_purchase=("order_purchase_timestamp", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_value", "sum"),
    customer_state=("customer_state", "first")
).reset_index()

customer_features["recency"] = (
    snapshot_date - customer_features["last_purchase"]
).dt.days

customer_features["average_order_value"] = (
    customer_features["monetary"] /
    customer_features["frequency"]
)

# Future purchase information
future_customers = set(
    future_data["customer_unique_id"].dropna()
)

# 1 = No purchase in next 90 days
customer_features["churn"] = (
    ~customer_features["customer_unique_id"].isin(future_customers)
).astype(int)

# Save dataset
os.makedirs("data/processed", exist_ok=True)

customer_features.to_csv(
    "data/processed/churn_model_data.csv",
    index=False
)

print("\nFUTURE CHURN DATA PREPARATION COMPLETED!")
print("Dataset shape:", customer_features.shape)

print("\nChurn distribution:")
print(customer_features["churn"].value_counts())

print("\nChurn percentage:")
print(round(customer_features["churn"].mean() * 100, 2), "%")

print("\nSaved: data/processed/churn_model_data.csv")
