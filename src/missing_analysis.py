
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
    "olist_sellers_dataset.csv"
]

for filename in files:
    df = pd.read_csv(DATA_DIR / filename)

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    print("\nDataset:", filename)
    print("Total rows:", len(df))

    if missing.empty:
        print("No missing values!")
    else:
        print("Missing values by column:")
        print(missing)

print("\nMissing Value Analysis Completed!")
