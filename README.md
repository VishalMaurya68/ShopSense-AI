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

## 👨‍💻 Author

**Vishal Maurya**

B.Tech – Computer Science (Data Science & AI)

GitHub: https://github.com/VishalMaurya68