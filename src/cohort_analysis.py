
import pandas as pd
import matplotlib.pyplot as plt
import os

print("Loading customer purchase data...")

orders = pd.read_csv(
    "data/raw/olist_orders_dataset.csv"
)

customers = pd.read_csv(
    "data/raw/olist_customers_dataset.csv"
)

# Convert dates
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

# Keep delivered orders
orders = orders[
    (orders["order_status"] == "delivered") &
    (orders["order_purchase_timestamp"].notna())
].copy()

# Merge unique customer IDs
data = orders.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="left"
)

# Remove missing customer IDs
data = data.dropna(subset=["customer_unique_id"])

# Calculate order month and cohort month
data["order_month"] = (
    data["order_purchase_timestamp"]
    .dt.to_period("M")
)

data["cohort_month"] = (
    data.groupby("customer_unique_id")["order_month"]
    .transform("min")
)

# Calculate cohort index
data["cohort_index"] = (
    (data["order_month"].dt.year - data["cohort_month"].dt.year) * 12
    + (data["order_month"].dt.month - data["cohort_month"].dt.month)
    + 1
)

# Cohort customer counts
cohort_data = data.groupby(
    ["cohort_month", "cohort_index"]
)["customer_unique_id"].nunique().reset_index()

cohort_table = cohort_data.pivot(
    index="cohort_month",
    columns="cohort_index",
    values="customer_unique_id"
)

# Retention percentage
cohort_sizes = cohort_table.iloc[:, 0]

retention = cohort_table.divide(
    cohort_sizes,
    axis=0
) * 100

os.makedirs("reports", exist_ok=True)

retention.to_csv("reports/cohort_retention.csv")

# Plot heatmap
plt.figure(figsize=(14, 9))

plt.imshow(
    retention,
    aspect="auto",
    cmap="YlGnBu",
    vmin=0,
    vmax=100
)

plt.colorbar(label="Customer Retention (%)")

plt.xticks(
    range(len(retention.columns)),
    retention.columns
)

plt.yticks(
    range(len(retention.index)),
    retention.index.astype(str)
)

plt.xlabel("Months Since First Purchase")
plt.ylabel("Customer Cohort")
plt.title("Olist Customer Cohort Retention Analysis")

plt.tight_layout()
plt.savefig("reports/cohort_retention.png")
plt.show()

print("\nCOHORT ANALYSIS COMPLETED!")
print("\nRetention Table:")
print(retention.round(2))

print("\nSaved: reports/cohort_retention.csv")
print("Saved: reports/cohort_retention.png")
