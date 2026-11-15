
import sqlite3
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "processed" / "shopsense.db"

connection = sqlite3.connect(DB_PATH)

queries = {
    "Order Status": """
        SELECT order_status, COUNT(*) AS total_orders
        FROM orders
        GROUP BY order_status
        ORDER BY total_orders DESC
    """,

    "Review Score": """
        SELECT review_score, COUNT(*) AS total_reviews
        FROM reviews
        GROUP BY review_score
        ORDER BY review_score
    """,

    "Top Product Categories": """
        SELECT product_category_name, COUNT(*) AS total_products
        FROM products
        GROUP BY product_category_name
        ORDER BY total_products DESC
        LIMIT 10
    """
}

for title, query in queries.items():
    print("\n==============================")
    print(title)
    print("==============================")

    result = pd.read_sql_query(query, connection)
    print(result.to_string(index=False))

connection.close()

print("\nSQL BUSINESS ANALYSIS COMPLETED!")
