
import pandas as pd
from pathlib import Path

data_path = Path("data/raw")

files = [
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_order_reviews_dataset.csv"
]

for file in files:
    df = pd.read_csv(data_path / file)

    print("\n==============================")
    print("Dataset:", file)
    print("Total Rows:", len(df))

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values!")
    else:
        print("Missing values by column:")
        print(missing)

print("\nAnalysis Completed!")
