
import pandas as pd
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

DB_PATH = DATA_DIR / "shopsense.db"

# Connect to SQLite database
connection = sqlite3.connect(DB_PATH)

datasets = {
    "orders": "orders_clean.csv",
    "products": "products_clean.csv",
    "reviews": "reviews_clean.csv"
}

for table_name, filename in datasets.items():
    df = pd.read_csv(DATA_DIR / filename)

    df.to_sql(
        table_name,
        connection,
        if_exists="replace",
        index=False
    )

    print(f"{table_name} table created: {len(df)} rows")

connection.close()

print("\nSHOP SENSE AI SQL DATABASE CREATED!")
print("Database:", DB_PATH)
