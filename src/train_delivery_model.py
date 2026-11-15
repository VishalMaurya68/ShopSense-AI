

import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve
)

DATA_PATH = Path("data/processed/delivery_model_data.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["order_purchase_timestamp"]
)

df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

numeric_features = [
    "purchase_month", "purchase_day", "purchase_weekday",
    "purchase_hour", "item_count", "total_price","purchase_year",
    "total_freight", "payment_value", "payment_installments"
]

categorical_features = [
    "customer_state", "seller_state", "product_category"
]

X = df[numeric_features + categorical_features]
y = df["is_late"]

# Chronological 70% train, 15% validation, 15% test
n = len(df)
train_end = int(n * 0.70)
validation_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:validation_end]
y_val = y.iloc[train_end:validation_end]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

print("\nTraining model...")
pipeline.fit(X_train, y_train)

# Tune threshold on validation data only
val_probability = pipeline.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val, val_probability
)

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

best_index = f1_values.argmax()
best_threshold = float(thresholds[best_index])

print("\nBest validation threshold:", round(best_threshold, 4))
print("Validation F1:", round(f1_values[best_index], 4))

# Final evaluation on untouched test data
test_probability = pipeline.predict_proba(X_test)[:, 1]
y_pred = (test_probability >= best_threshold).astype(int)

print("\n===== FINAL TEST PERFORMANCE =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1 Score:", round(f1_score(y_test, y_pred, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, test_probability), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save pipeline and selected threshold
joblib.dump(
    {"pipeline": pipeline, "threshold": best_threshold},
    MODEL_DIR / "delivery_delay_model.joblib"
)

results = pd.DataFrame({
    "actual_is_late": y_test,
    "predicted_is_late": y_pred,
    "late_probability": test_probability
})

results.to_csv(REPORT_DIR / "delivery_predictions.csv", index=False)

print("\nIMPROVED MODEL TRAINING COMPLETED!")
print("Model and threshold saved in models folder.")
print("Test predictions saved in reports folder.")
