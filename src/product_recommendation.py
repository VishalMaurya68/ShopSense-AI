
import pandas as pd
import os
import joblib
from collections import Counter, defaultdict
from itertools import combinations

print("Loading Olist datasets...")

# Load datasets
items = pd.read_csv("data/raw/olist_order_items_dataset.csv")
products = pd.read_csv("data/raw/olist_products_dataset.csv")
translation = pd.read_csv(
    "data/raw/product_category_name_translation.csv"
)

# Clean required columns
items = items.dropna(subset=["order_id", "product_id"])
items["product_id"] = items["product_id"].astype(str)

# Count product purchases
product_counts = items["product_id"].value_counts()

# Map product categories to English
products["product_id"] = products["product_id"].astype(str)

products = products.merge(
    translation,
    on="product_category_name",
    how="left"
)

category_column = "product_category_name_english"

if category_column in products.columns:
    category_map = dict(
        zip(products["product_id"], products[category_column])
    )
else:
    category_map = dict(
        zip(products["product_id"], products["product_category_name"])
    )

# Group products purchased in the same order
order_products = (
    items.groupby("order_id")["product_id"]
    .apply(lambda x: sorted(set(x)))
)

# Count products frequently bought together
pair_counts = Counter()

for product_list in order_products:
    for product_a, product_b in combinations(product_list, 2):
        pair_counts[(product_a, product_b)] += 1

# Create recommendation dictionary
recommendations = defaultdict(list)

for (product_a, product_b), count in pair_counts.items():
    recommendations[product_a].append((product_b, count))
    recommendations[product_b].append((product_a, count))

# Prepare recommendation report
recommendation_rows = []

for product_id, suggested_products in recommendations.items():

    suggested_products.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for recommended_id, count in suggested_products[:5]:
        recommendation_rows.append({
            "Product_ID": product_id,
            "Product_Category": category_map.get(product_id, "Unknown"),
            "Recommended_Product_ID": recommended_id,
            "Recommended_Category": category_map.get(
                recommended_id, "Unknown"
            ),
            "Co_Purchase_Count": count
        })

recommendation_df = pd.DataFrame(recommendation_rows)

# Create output folders
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Save reports
recommendation_df.to_csv(
    "reports/product_recommendations.csv",
    index=False
)

top_products = product_counts.head(10).reset_index()
top_products.columns = ["Product_ID", "Purchase_Count"]

top_products.to_csv(
    "reports/top_products.csv",
    index=False
)

# Save recommendation model
joblib.dump(
    dict(recommendations),
    "models/product_recommendation_model.joblib"
)

print("\n===== PRODUCT RECOMMENDATION SYSTEM =====")
print("\nTop 10 Most Purchased Products:")
print(top_products.to_string(index=False))

print("\nSample Product Recommendations:")
print(recommendation_df.head(10).to_string(index=False))

print("\nTotal recommendation records:", len(recommendation_df))
print("\nFiles saved successfully!")
