
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

print("Loading Olist datasets...")

orders = pd.read_csv(RAW / "olist_orders_dataset.csv")
items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
customers = pd.read_csv(RAW / "olist_customers_dataset.csv")
sellers = pd.read_csv(RAW / "olist_sellers_dataset.csv")
products = pd.read_csv(RAW / "olist_products_dataset.csv")
payments = pd.read_csv(RAW / "olist_order_payments_dataset.csv")

# Convert date columns
date_columns = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    orders[column] = pd.to_datetime(orders[column], errors="coerce")

# Keep only delivered orders with actual and estimated delivery dates
data = orders[
    (orders["order_status"] == "delivered")
    & orders["order_delivered_customer_date"].notna()
    & orders["order_estimated_delivery_date"].notna()
].copy()

# Target: 1 means delivered late, 0 means delivered on time
delay_days = (
    data["order_delivered_customer_date"]
    - data["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

data["is_late"] = (delay_days > 0).astype(int)

# Customer state
data = data.merge(
    customers[["customer_id", "customer_state"]],
    on="customer_id",
    how="left"
)

# Product and seller details for each order item
item_details = items.merge(
    products[["product_id", "product_category_name"]],
    on="product_id",
    how="left"
)

item_details = item_details.merge(
    sellers[["seller_id", "seller_state"]],
    on="seller_id",
    how="left"
)

# Aggregate item information at order level
item_summary = item_details.groupby("order_id").agg(
    item_count=("order_item_id", "count"),
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum"),
    product_category=("product_category_name",
                      lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown"),
    seller_state=("seller_state",
                  lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")
).reset_index()

data = data.merge(item_summary, on="order_id", how="inner")

# Aggregate payment information at order level
payment_summary = payments.groupby("order_id").agg(
    payment_value=("payment_value", "sum"),
    payment_installments=("payment_installments", "max")
).reset_index()

data = data.merge(payment_summary, on="order_id", how="left")

# Features known at or before order processing
purchase_time = data["order_purchase_timestamp"]
data["estimated_delivery_days"] = (
    data["order_estimated_delivery_date"]
    - data["order_purchase_timestamp"]
).dt.total_seconds() / 86400

data["purchase_month"] = purchase_time.dt.month
data["purchase_year"] = purchase_time.dt.year
data["purchase_day"] = purchase_time.dt.day
data["purchase_weekday"] = purchase_time.dt.dayofweek
data["purchase_hour"] = purchase_time.dt.hour

# Select model columns
feature_columns = [
    "purchase_month",
    "purchase_year",
    "estimated_delivery_days",
    "purchase_day",
    "purchase_weekday",
    "purchase_hour",
    "item_count",
    "total_price",
    "total_freight",
    "payment_value",
    "payment_installments",
    "customer_state",
    "seller_state",
    "product_category"
]

final_data = data[
    ["order_id", "order_purchase_timestamp", "is_late"] + feature_columns
].copy()

# Fill missing feature values
numeric_columns = [
    "item_count", "total_price", "total_freight",
    "payment_value", "payment_installments"
]

categorical_columns = [
    "customer_state", "seller_state", "product_category"
]

for column in numeric_columns:
    final_data[column] = final_data[column].fillna(
        final_data[column].median()
    )

for column in categorical_columns:
    final_data[column] = final_data[column].fillna("unknown")

output_path = PROCESSED / "delivery_model_data.csv"
final_data.to_csv(output_path, index=False)

print("\nDELIVERY DATA PREPARATION COMPLETED!")
print("Dataset shape:", final_data.shape)
print("\nLate delivery distribution:")
print(final_data["is_late"].value_counts().rename(
    index={0: "On Time", 1: "Late"}
))
print("\nLate delivery percentage:")
print(round(final_data["is_late"].mean() * 100, 2), "%")
print("\nSaved to:", output_path)
