
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("SHOP SENSE AI - DATA CLEANING")
print("-" * 40)

# 1. Load datasets
orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv")
products = pd.read_csv(RAW_DIR / "olist_products_dataset.csv")
reviews = pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv")

# 2. Clean Orders
date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in date_columns:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

# Preserve missing delivery dates; don't fill them.
orders = orders.drop_duplicates()

# 3. Clean Products
products = products.drop_duplicates()

products["product_category_name"] = (
    products["product_category_name"].fillna("unknown")
)

numeric_columns = [
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]

for col in numeric_columns:
    products[col] = products[col].fillna(products[col].median())

# 4. Clean Reviews
reviews = reviews.drop_duplicates()

reviews["review_comment_title"] = (
    reviews["review_comment_title"].fillna("")
)

reviews["review_comment_message"] = (
    reviews["review_comment_message"].fillna("")
)

# 5. Save cleaned datasets
orders.to_csv(PROCESSED_DIR / "orders_clean.csv", index=False)
products.to_csv(PROCESSED_DIR / "products_clean.csv", index=False)
reviews.to_csv(PROCESSED_DIR / "reviews_clean.csv", index=False)

# 6. Display results
print("\nCleaning completed successfully!")

print("\nOrders:", orders.shape)
print("Products:", products.shape)
print("Reviews:", reviews.shape)

print("\nCleaned files saved in data/processed/")
