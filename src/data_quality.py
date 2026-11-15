
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

files = [
    "olist_orders_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv"
]

print("=" * 55)
print("       SHOPSENSE AI - DATA QUALITY REPORT")
print("=" * 55)

for filename in files:
    path = DATA_DIR / filename

    if not path.exists():
        print(f"\nMISSING FILE: {filename}")
        continue

    df = pd.read_csv(path)

    print(f"\nDataset: {filename}")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Missing Values: {df.isnull().sum().sum():,}")
    print(f"Duplicate Rows: {df.duplicated().sum():,}")

print("\n" + "=" * 55)
print("DATA QUALITY CHECK COMPLETED!")
input("\nPress Enter to close...")
