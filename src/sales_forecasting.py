
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Loading Olist datasets...")

orders = pd.read_csv("data/raw/olist_orders_dataset.csv")
items = pd.read_csv("data/raw/olist_order_items_dataset.csv")

# Convert purchase date
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

# Use delivered orders only
orders = orders[
    (orders["order_status"] == "delivered") &
    (orders["order_purchase_timestamp"].notna())
].copy()

# Calculate product sales (excluding freight)
order_sales = items.groupby("order_id")["price"].sum().reset_index()
order_sales.rename(columns={"price": "sales"}, inplace=True)

data = orders.merge(order_sales, on="order_id", how="inner")

# Monthly sales
data["month"] = (
    data["order_purchase_timestamp"].dt.to_period("M")
)

monthly_sales = data.groupby("month")["sales"].sum()

# Fill missing months with zero
all_months = pd.period_range(
    monthly_sales.index.min(),
    monthly_sales.index.max(),
    freq="M"
)

monthly_sales = monthly_sales.reindex(
    all_months, fill_value=0
)

df = pd.DataFrame({
    "month": monthly_sales.index,
    "sales": monthly_sales.values
})

# Feature engineering
df["year"] = df["month"].dt.year
df["month_number"] = df["month"].dt.month
df["time_index"] = np.arange(len(df))

for lag in [1, 2, 3]:
    df[f"lag_{lag}"] = df["sales"].shift(lag)

df["rolling_3"] = df["sales"].shift(1).rolling(3).mean()

df = df.dropna().reset_index(drop=True)

features = [
    "year", "month_number", "time_index",
    "lag_1", "lag_2", "lag_3", "rolling_3"
]

X = df[features]
y = df["sales"]

# Last 3 months for testing
X_train = X.iloc[:-3]
X_test = X.iloc[-3:]
y_train = y.iloc[:-3]
y_test = y.iloc[-3:]

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# Train model
print("\nTraining Random Forest Sales Model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n===== SALES FORECAST PERFORMANCE =====")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

comparison = pd.DataFrame({
    "Month": df["month"].iloc[-3:].astype(str).values,
    "Actual Sales": y_test.values,
    "Predicted Sales": predictions
})

print("\nActual vs Predicted Sales:")
print(comparison.to_string(index=False))

# Forecast next 3 months recursively
model.fit(X, y)

history = list(df["sales"].values)
future_results = []

last_month = df["month"].iloc[-1]

for step in range(1, 4):
    next_month = last_month + step

    next_index = len(history)

    row = pd.DataFrame([{
        "year": next_month.year,
        "month_number": next_month.month,
        "time_index": next_index,
        "lag_1": history[-1],
        "lag_2": history[-2],
        "lag_3": history[-3],
        "rolling_3": np.mean(history[-3:])
    }])

    forecast = model.predict(row[features])[0]
    forecast = max(0, forecast)

    history.append(forecast)

    future_results.append({
        "Month": str(next_month),
        "Forecast Sales (BRL)": round(forecast, 2)
    })

forecast_df = pd.DataFrame(future_results)

# Save outputs
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

comparison.to_csv(
    "reports/sales_model_evaluation.csv",
    index=False
)

forecast_df.to_csv(
    "reports/sales_forecast.csv",
    index=False
)

import joblib
joblib.dump(model, "models/sales_forecasting_model.joblib")

# Plot actual and predicted sales
plt.figure(figsize=(12, 6))

plt.plot(
    comparison["Month"],
    comparison["Actual Sales"],
    marker="o",
    label="Actual Sales"
)

plt.plot(
    comparison["Month"],
    comparison["Predicted Sales"],
    marker="o",
    label="Predicted Sales"
)

plt.xlabel("Month")
plt.ylabel("Sales (BRL)")
plt.title("Actual vs Predicted Monthly Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("reports/sales_forecast_evaluation.png")
plt.show()

print("\n===== NEXT 3 MONTHS SALES FORECAST =====")
print(forecast_df.to_string(index=False))

print("\nSALES FORECASTING COMPLETED!")
print("Model saved in models folder.")
print("Reports saved in reports folder.")
