# 🛒 ShopSense AI – E-Commerce Intelligence Platform

An end-to-end Data Science and Machine Learning project built using the Olist Brazilian E-Commerce dataset.

## 🚀 Project Overview

ShopSense AI is an interactive analytics platform designed to explore e-commerce data and generate useful business insights using data analysis, machine learning, and visualization.

## ✨ Key Features

- **Sales Forecasting:** Experimental forecasting of monthly sales.
- **Customer Segmentation:** RFM-based customer grouping.
- **Delivery Prediction:** Machine learning model to identify potentially late deliveries.
- **Retention Analysis:** Customer cohort and retention analysis.
- **Product Recommendations:** Product co-purchase recommendations.
- **Anomaly Detection:** Identification of unusual order patterns.
- **Interactive Dashboard:** Seven-page Streamlit dashboard with Plotly visualizations.

## 🧰 Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn and XGBoost
- SQLite and SQL
- Streamlit
- Plotly
- Matplotlib and Seaborn
- Git and GitHub

## 📊 Dataset

Olist Brazilian E-Commerce Public Dataset.

Source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

The dataset contains historical Brazilian e-commerce records, primarily from 2016–2018.

## 🌐 Live Demo

https://shopsense-ai-ppaju9yxtvjai6cqthmzb8.streamlit.app/

## 📁 Project Structure

```text
ShopSense-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── data_cleaning.py
│   ├── validate_data.py
│   ├── load_to_sqlite.py
│   ├── sql_analysis.py
│   ├── eda_analysis.py
│   ├── rfm_analysis.py
│   ├── prepare_delivery_data.py
│   ├── cohort_analysis.py
│   ├── sales_forecasting.py
│   ├── product_recommendation.py
│   └── anomaly_detection.py
│
└── reports/
    └── Generated analysis reports and charts
```

## ⚠️ Project Limitations

- The dataset is historical and does not represent current market activity.
- Delivery prediction performance is modest and needs further improvement.
- Sales forecasts are experimental and based on limited historical observations.
- Product recommendations are based on observed co-purchases.
- Anomaly detection identifies unusual patterns, not confirmed fraud.

## 🎯 Project Objective

To develop practical skills in data cleaning, exploratory data analysis, SQL, machine learning, customer analytics, model evaluation, and dashboard development.

## 🚀 Live Delivery Delay Prediction

ShopSense AI includes an interactive XGBoost-based delivery delay prediction feature in the Streamlit dashboard.

Users can enter 14 order-related inputs:
- Purchase year, month, day, weekday and hour
- Estimated delivery days
- Item count
- Total product price
- Total freight
- Payment value
- Payment installments
- Customer state
- Seller state
- Product category

The saved preprocessing pipeline and XGBoost model generate a late-delivery probability. A classification threshold of approximately 0.5739 is used to classify the prediction.

**Limitation:** The model has modest test performance. Its predictions are experimental and are not guaranteed delivery outcomes.

## 💻 Run Locally

1. Clone the repository:
   `git clone https://github.com/VishalMaurya68/ShopSense-AI.git`
2. Open the project folder:
   `cd ShopSense-AI`
3. Create a virtual environment:
   `python -m venv venv`
4. Activate it on Windows:
   `venv\Scripts\activate`
5. Install dependencies:
   `pip install -r requirements.txt`
6. Run the dashboard:
   `streamlit run app.py`

**Note:** Live delivery prediction also requires `models/delivery_xgboost_model.joblib`.

## 👨‍💻 Author

**Vishal Maurya**

B.Tech – Computer Science (Data Science & AI)

GitHub: https://github.com/VishalMaurya68