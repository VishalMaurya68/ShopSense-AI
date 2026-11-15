
import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve
)
from xgboost import XGBClassifier

DATA_PATH = Path("data/processed/delivery_model_data.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

print("Loading delivery dataset...")

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["order_purchase_timestamp"]
)

df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

numeric_features = [
    "purchase_year",
    "purchase_month",
    "purchase_day",
    "purchase_weekday",
    "estimated_delivery_days",
    "purchase_hour",
    "item_count",
    "total_price",
    "total_freight",
    "payment_value",
    "payment_installments"
]

categorical_features = [
    "customer_state",
    "seller_state",
    "product_category"
]

X = df[numeric_features + categorical_features]
y = df["is_late"]

# Chronological split: 70% train, 15% validation, 15% test
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

# Handle missing values and encode categories
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

# Calculate class imbalance from training data only
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
scale_weight = negative_count / positive_count

print("Positive class weight:", round(scale_weight, 2))

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=5,
    scale_pos_weight=scale_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb_model)
])

print("\nTraining XGBoost model...")
pipeline.fit(X_train, y_train)

# Select threshold using validation data only
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

# Evaluate once on the untouched test set
test_probability = pipeline.predict_proba(X_test)[:, 1]
y_pred = (test_probability >= best_threshold).astype(int)

print("\n===== XGBOOST TEST PERFORMANCE =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1 Score:", round(f1_score(y_test, y_pred, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, test_probability), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and threshold
joblib.dump(
    {"pipeline": pipeline, "threshold": best_threshold},
    MODEL_DIR / "delivery_xgboost_model.joblib"
)

results = pd.DataFrame({
    "actual_is_late": y_test,
    "predicted_is_late": y_pred,
    "late_probability": test_probability
})

results.to_csv(
    REPORT_DIR / "delivery_xgboost_predictions.csv",
    index=False
)

print("\nXGBOOST TRAINING COMPLETED!")
print("Model saved in models folder.")
print("Predictions saved in reports folder.")
