
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("Loading churn dataset...")

df = pd.read_csv("data/processed/churn_model_data.csv")

# Input features
# Recency is excluded because churn label is created from recency.
X = df[["frequency", "monetary"]]

# Target
y = df["churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Train Random Forest
print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n===== CHURN MODEL PERFORMANCE =====")

print("Accuracy:", round(
    accuracy_score(y_test, predictions), 4
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("ROC-AUC:", round(
    roc_auc_score(y_test, probabilities), 4
))

# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/churn_random_forest_model.joblib"
)

print("\nCHURN MODEL TRAINING COMPLETED!")
print("Model saved in models folder.")
