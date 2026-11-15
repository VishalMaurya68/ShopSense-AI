
import pandas as pd
from pathlib import Path

# Project folder location
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

# Load datasets
orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")

print("SHOP SENSE AI")
print("Dataset loaded successfully!")

print("\nOrders:", orders.shape)
print("Customers:", customers.shape)
print("Order Items:", items.shape)

print("\nFirst 5 Orders:")
print(orders.head())
