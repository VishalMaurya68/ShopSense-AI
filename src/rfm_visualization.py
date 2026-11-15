
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"

# Load RFM summary
df = pd.read_csv(REPORT_DIR / "rfm_summary.csv")

# Customer segment bar chart
plt.figure(figsize=(10, 6))
plt.bar(df["Segment"], df["Customers"])

plt.title("Customer Segmentation - RFM Analysis")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(REPORT_DIR / "rfm_segments.png")
plt.close()

# Average monetary value chart
plt.figure(figsize=(10, 6))
plt.bar(df["Segment"], df["Average_Monetary"])

plt.title("Average Customer Monetary Value")
plt.xlabel("Customer Segment")
plt.ylabel("Average Monetary Value (BRL)")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(REPORT_DIR / "rfm_monetary.png")
plt.close()

print("RFM VISUALIZATION COMPLETED!")
print("Charts saved in reports folder.")
