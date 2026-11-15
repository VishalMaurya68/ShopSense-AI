
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

files = [
    "orders_clean.csv",
    "products_clean.csv",
    "reviews_clean.csv"
]

for file in files:
    df = pd.read_csv(DATA_DIR / file, keep_default_na=False)

    print("\nDataset:", file)
    print("Shape:", df.shape)
    print("Duplicate rows:", df.duplicated().sum())
    print("Missing values:", df.isnull().sum().sum())

    if file == "reviews_clean.csv":
        print("Empty review titles:", (df["review_comment_title"] == "").sum())
        print("Empty review messages:", (df["review_comment_message"] == "").sum())

print("\nDATA VALIDATION COMPLETED!")
