
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("SHOPSENSE AI - CUSTOMER RFM ANALYSIS")
print("-" * 45)

# Load datasets
orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv")
customers = pd.read_csv(RAW_DIR / "olist_customers_dataset.csv")
items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")

# Convert purchase timestamp
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

# Keep completed orders only
orders = orders[
    (orders["order_status"] == "delivered") &
    (orders["order_purchase_timestamp"].notna())
].copy()

# Join orders with customers
data = orders.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="inner"
)

# Join order items
data = data.merge(
    items[["order_id", "price", "freight_value"]],
    on="order_id",
    how="inner"
)

# Calculate order value
data["order_value"] = data["price"] + data["freight_value"]

# Reference date
reference_date = data["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

# Calculate RFM
rfm = data.groupby("customer_unique_id").agg(
    Recency=("order_purchase_timestamp",
             lambda x: (reference_date - x.max()).days),

    Frequency=("order_id", "nunique"),

    Monetary=("order_value", "sum")
).reset_index()

# Calculate RFM scores
rfm["R_score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm["F_score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["M_score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

# Combined score
rfm["RFM_Score"] = (
    rfm["R_score"].astype(str) +
    rfm["F_score"].astype(str) +
    rfm["M_score"].astype(str)
)

# Customer segments
def segment_customer(row):
    r = row["R_score"]
    f = row["F_score"]
    m = row["M_score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3:
        return "Loyal Customers"
    elif r >= 4 and f <= 2:
        return "New / Promising"
    elif r <= 2 and f >= 3:
        return "At Risk"
    else:
        return "Hibernating"

rfm["Segment"] = rfm.apply(segment_customer, axis=1)

# Save results
rfm.to_csv(REPORT_DIR / "rfm_customers.csv", index=False)

summary = rfm.groupby("Segment").agg(
    Customers=("customer_unique_id", "count"),
    Average_Monetary=("Monetary", "mean")
).reset_index()

summary.to_csv(REPORT_DIR / "rfm_summary.csv", index=False)

# Display results
print("\nTotal customers analyzed:", len(rfm))
print("\nCustomer Segments:")
print(summary.to_string(index=False))

print("\nRFM analysis completed!")
print("Files saved in reports folder.")
