
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load trained XGBoost model
saved_model = joblib.load("models/delivery_xgboost_model.joblib")
model = saved_model["pipeline"]

# Get feature importance
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()
importance = classifier.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance", ascending=False
).head(15)

print("\nTOP 15 IMPORTANT FEATURES")
print(importance_df.to_string(index=False))

# Save results
importance_df.to_csv(
    "reports/delivery_feature_importance.csv",
    index=False
)

# Plot chart
plt.figure(figsize=(10, 7))
plt.barh(
    importance_df["Feature"][::-1],
    importance_df["Importance"][::-1]
)
plt.xlabel("Importance")
plt.title("Top 15 Delivery Prediction Features")
plt.tight_layout()
plt.savefig("reports/delivery_feature_importance.png")
plt.show()

print("\nFeature importance analysis completed!")
