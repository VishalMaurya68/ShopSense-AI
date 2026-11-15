
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "processed" / "shopsense.db"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

connection = sqlite3.connect(DB_PATH)

# Load data
orders = pd.read_sql_query("SELECT * FROM orders", connection)
reviews = pd.read_sql_query("SELECT * FROM reviews", connection)

connection.close()

# 1. Order Status Analysis
status = orders["order_status"].value_counts()

plt.figure(figsize=(9, 5))
status.plot(kind="bar")
plt.title("Order Status Distribution")
plt.xlabel("Order Status")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig(REPORT_DIR / "order_status.png")
plt.close()

# 2. Review Score Analysis
scores = reviews["review_score"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
scores.plot(kind="bar")
plt.title("Customer Review Score Distribution")
plt.xlabel("Review Score")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(REPORT_DIR / "review_scores.png")
plt.close()

# 3. Monthly Order Trend
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

monthly = (
    orders.dropna(subset=["order_purchase_timestamp"])
    .set_index("order_purchase_timestamp")
    .resample("ME")
    .size()
)

plt.figure(figsize=(11, 5))
monthly.plot()
plt.title("Monthly Order Trend")
plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig(REPORT_DIR / "monthly_orders.png")
plt.close()

print("EDA ANALYSIS COMPLETED!")
print("Order status chart saved.")
print("Review score chart saved.")
print("Monthly order chart saved.")
print("Charts location:", REPORT_DIR)
