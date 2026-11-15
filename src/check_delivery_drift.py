

import pandas as pd

df = pd.read_csv(
    "data/processed/delivery_model_data.csv",
    parse_dates=["order_purchase_timestamp"]
)

df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

print("\n===== DELIVERY DATA DRIFT CHECK =====")

for name, part in [
    ("TRAINING", df.iloc[:train_end]),
    ("VALIDATION", df.iloc[train_end:val_end]),
    ("TESTING", df.iloc[val_end:])
]:
    print("\n", name)
    print("Total Orders:", len(part))
    print("Late Orders:", part["is_late"].sum())
    print("Late Delivery Rate:",
          round(part["is_late"].mean() * 100, 2), "%")
