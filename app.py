
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
REPORTS = BASE_DIR / "reports"

# Page configuration
st.set_page_config(
    page_title="ShopSense AI",
    page_icon="🛒",
    layout="wide"
)

# Dark dashboard styling

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #081827, #102b3f, #0b1f30);
    color: #e2f1f5;
}

[data-testid="stSidebar"] {
    background-color: #0b2030;
    border-right: 1px solid #164e63;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #123247, #164e63);
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #0e7490;
}

[data-testid="stMetricLabel"] {
    color: #67e8f9;
}

[data-testid="stMetricValue"] {
    color: #ffffff;
}

h1, h2, h3 {
    color: #67e8f9;
}

.stButton > button {
    background-color: #0d9488;
    color: white;
    border: 1px solid #2dd4bf;
    border-radius: 10px;
}

.stButton > button:hover {
    background-color: #14b8a6;
    color: white;
    border: 1px solid #67e8f9;
}
</style>
""", unsafe_allow_html=True)



def load_csv(filename):
    path = REPORTS / filename
    if path.exists():
        return pd.read_csv(path)
    return None


def show_image(filename, caption):
    path = REPORTS / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{filename} is not available yet.")


# Sidebar navigation
st.sidebar.title("🛒 ShopSense AI")
st.sidebar.caption("E-Commerce Intelligence Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Sales Forecasting",
        "Customer Segmentation",
        "Delivery Prediction",
        "Retention Analysis",
        "Product Recommendations",
        "Anomaly Detection"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Built with Python, Machine Learning & Streamlit")


# OVERVIEW PAGE
if page == "Overview":

    st.title("🛒 ShopSense AI")
    st.subheader("E-Commerce Intelligence Dashboard")
    st.write(
        "Analyze sales, understand customers, predict delivery delays "
        "and discover unusual order patterns."
    )

    sales = load_csv("sales_forecast.csv")
    recommendations = load_csv("product_recommendations.csv")
    anomalies = load_csv("anomaly_detection_results.csv")
    rfm = load_csv("rfm_customers.csv")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Dataset", "Olist")
    col2.metric(
        "Recommendations",
        f"{len(recommendations):,}" if recommendations is not None else "N/A"
    )
    col3.metric(
        "Unusual Orders",
        f"{(anomalies['Anomaly'] == 'Unusual').sum():,}"
        if anomalies is not None and "Anomaly" in anomalies.columns
        else "N/A"
    )
    col4.metric(
        "Customers Analyzed",
        f"{len(rfm):,}" if rfm is not None else "N/A"
    )

    st.divider()
    st.subheader("Project Modules")

    c1, c2, c3 = st.columns(3)
    c1.info("📊 Sales Forecasting")
    c2.info("👥 Customer Segmentation")
    c3.info("🚚 Delivery Prediction")

    c4, c5, c6 = st.columns(3)
    c4.info("🔄 Retention Analysis")
    c5.info("🛍️ Product Recommendations")
    c6.info("🚨 Anomaly Detection")

    st.divider()
    st.subheader("Sales Forecast Preview")

    if sales is not None:
        st.dataframe(sales, use_container_width=True)
    else:
        st.warning("Sales forecast report not found.")

    st.subheader("Project Visualizations")
    show_image("monthly_orders.png", "Monthly Orders")
    show_image("rfm_segments.png", "Customer Segments")


# SALES FORECASTING PAGE

# SALES FORECASTING PAGE
elif page == "Sales Forecasting":

    st.title("📊 Sales Intelligence & Forecasting")
    st.write("Analyze historical sales and explore experimental forecasts.")

    evaluation = load_csv("sales_model_evaluation.csv")
    forecast = load_csv("sales_forecast.csv")

    if evaluation is not None and not evaluation.empty:

        st.subheader("📈 Model Performance")

        actual_col = "Actual Sales"
        predicted_col = "Predicted Sales"

        if actual_col in evaluation.columns and predicted_col in evaluation.columns:

            mae = (
                evaluation[actual_col] - evaluation[predicted_col]
            ).abs().mean()

            total_actual = evaluation[actual_col].sum()
            total_predicted = evaluation[predicted_col].sum()

            c1, c2, c3 = st.columns(3)

            c1.metric("Test Months", len(evaluation))
            c2.metric("Mean Absolute Error", f"₹{mae:,.0f}")
            c3.metric("Actual Test Sales", f"₹{total_actual:,.0f}")

            st.subheader("Actual vs Predicted Sales")

            chart_data = evaluation.melt(
                id_vars="Month",
                value_vars=[actual_col, predicted_col],
                var_name="Sales Type",
                value_name="Sales"
            )

            fig = px.line(
                chart_data,
                x="Month",
                y="Sales",
                color="Sales Type",
                markers=True,
                title="Monthly Actual vs Predicted Sales"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                evaluation,
                use_container_width=True,
                hide_index=True
            )

    if forecast is not None and not forecast.empty:

        st.divider()
        st.subheader("🔮 Experimental Sales Forecast")

        if "Forecast Sales (BRL)" in forecast.columns:

            total_forecast = forecast["Forecast Sales (BRL)"].sum()
            average_forecast = forecast["Forecast Sales (BRL)"].mean()

            c1, c2 = st.columns(2)
            c1.metric("Forecast Total", f"₹{total_forecast:,.0f}")
            c2.metric("Average Forecast", f"₹{average_forecast:,.0f}")

            fig = px.bar(
                forecast,
                x="Month",
                y="Forecast Sales (BRL)",
                title="Forecasted Monthly Sales"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            forecast,
            use_container_width=True,
            hide_index=True
        )

    st.info(
        "Note: This forecast uses historical Olist data ending in 2018. "
        "It is an academic forecasting experiment, not a live 2026 sales forecast."
    )


    st.title("📊 Sales Forecasting")

    evaluation = load_csv("sales_model_evaluation.csv")
    forecast = load_csv("sales_forecast.csv")

    if evaluation is not None:
        st.subheader("Actual vs Predicted Sales")
        st.dataframe(evaluation, use_container_width=True)

        if {"Month", "Actual Sales", "Predicted Sales"}.issubset(evaluation.columns):
            chart_data = evaluation.melt(
                id_vars="Month",
                value_vars=["Actual Sales", "Predicted Sales"],
                var_name="Sales Type",
                value_name="Sales"
            )
            fig = px.line(
                chart_data,
                x="Month",
                y="Sales",
                color="Sales Type",
                markers=True,
                title="Actual vs Predicted Sales"
            )
            st.plotly_chart(fig, use_container_width=True)

    if forecast is not None:
        st.subheader("Future Forecast")
        st.dataframe(forecast, use_container_width=True)


# CUSTOMER SEGMENTATION PAGE

# CUSTOMER SEGMENTATION PAGE
elif page == "Customer Segmentation":

    st.title("👥 Customer Segmentation")
    st.write("Explore customers using Recency, Frequency and Monetary analysis.")

    rfm = load_csv("rfm_customers.csv")
    summary = load_csv("rfm_summary.csv")

    if summary is not None:
        st.subheader("📊 RFM Segment Summary")
        st.dataframe(summary, use_container_width=True)

    if rfm is not None and not rfm.empty:

        st.subheader("🔎 Customer Search & Segment Filter")

        segment_column = next(
            (c for c in rfm.columns if "segment" in c.lower()),
            None
        )

        filtered = rfm.copy()

        if segment_column:
            segments = sorted(
                rfm[segment_column].dropna().astype(str).unique()
            )

            selected_segment = st.selectbox(
                "Select Customer Segment",
                ["All Segments"] + segments
            )

            if selected_segment != "All Segments":
                filtered = filtered[
                    filtered[segment_column].astype(str) == selected_segment
                ]

        customer_column = next(
            (c for c in rfm.columns if "customer" in c.lower()),
            None
        )

        if customer_column:
            customer_search = st.text_input(
                "Search Customer ID",
                placeholder="Enter customer ID..."
            )

            if customer_search.strip():
                filtered = filtered[
                    filtered[customer_column].astype(str).str.contains(
                        customer_search.strip(),
                        case=False,
                        na=False
                    )
                ]

        st.metric("Matching Customers", f"{len(filtered):,}")

        st.dataframe(
            filtered.head(200),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("RFM customer data not found.")

    st.subheader("Customer Visualizations")

    show_image("rfm_segments.png", "RFM Customer Segments")
    show_image("rfm_monetary.png", "Customer Monetary Value")


    st.title("👥 Customer Segmentation")

    rfm = load_csv("rfm_customers.csv")
    summary = load_csv("rfm_summary.csv")

    if summary is not None:
        st.subheader("RFM Segment Summary")
        st.dataframe(summary, use_container_width=True)

    if rfm is not None:
        st.subheader("Customer RFM Data")
        st.dataframe(rfm.head(100), use_container_width=True)

    show_image("rfm_segments.png", "RFM Customer Segments")
    show_image("rfm_monetary.png", "Customer Monetary Value")


# DELIVERY PREDICTION PAGE

# DELIVERY PREDICTION PAGE
elif page == "Delivery Prediction":

    st.title("🚚 Delivery Delay Prediction")
    st.write("Explore delivery model results and important prediction features.")

    # LIVE DELIVERY PREDICTION
    import joblib

    st.divider()
    st.subheader("🚀 Try Live Delivery Prediction")
    st.caption("Enter order details to estimate the probability of delivery delay.")

    with st.form("delivery_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            purchase_year = st.number_input("Purchase Year", 2016, 2018, 2018)
            purchase_month = st.number_input("Purchase Month", 1, 12, 6)
            purchase_day = st.number_input("Purchase Day", 1, 31, 15)
            purchase_weekday = st.number_input("Purchase Weekday (0=Monday)", 0, 6, 2)
            purchase_hour = st.number_input("Purchase Hour", 0, 23, 12)
            estimated_delivery_days = st.number_input("Estimated Delivery Days", 1, 60, 15)
            item_count = st.number_input("Item Count", 1, 100, 1)

        with col2:
            total_price = st.number_input("Total Product Price (BRL)", min_value=0.0, value=150.0)
            total_freight = st.number_input("Total Freight (BRL)", min_value=0.0, value=20.0)
            payment_value = st.number_input("Payment Value (BRL)", min_value=0.0, value=170.0)
            payment_installments = st.number_input("Payment Installments", 1, 24, 1)
            customer_state = st.text_input("Customer State", value="SP")
            seller_state = st.text_input("Seller State", value="SP")
            product_category = st.text_input("Product Category", value="beleza_saude")

        submitted = st.form_submit_button("Predict Delivery Status")

    if submitted:
        try:
            model_path = BASE_DIR / "models" / "delivery_xgboost_model.joblib"
            model_bundle = joblib.load(model_path)

            input_data = pd.DataFrame([{
                "purchase_year": purchase_year,
                "purchase_month": purchase_month,
                "purchase_day": purchase_day,
                "purchase_weekday": purchase_weekday,
                "estimated_delivery_days": estimated_delivery_days,
                "purchase_hour": purchase_hour,
                "item_count": item_count,
                "total_price": total_price,
                "total_freight": total_freight,
                "payment_value": payment_value,
                "payment_installments": payment_installments,
                "customer_state": customer_state.strip().upper(),
                "seller_state": seller_state.strip().upper(),
                "product_category": product_category.strip()
            }])

            pipeline = model_bundle["pipeline"]
            threshold = model_bundle["threshold"]

            probability = pipeline.predict_proba(input_data)[0][1]
            prediction = int(probability >= threshold)

            st.metric("Estimated Late Delivery Probability", f"{probability:.2%}")

            if prediction == 1:
                st.warning("Model Prediction: Possible Delivery Delay")
            else:
                st.success("Model Prediction: Likely On-Time Delivery")

            st.caption(f"Classification threshold: {threshold:.4f}")

        except FileNotFoundError:
            st.error("Model file not found. Please check the models folder.")
        except Exception as error:
            st.error(f"Prediction error: {error}")


    predictions = load_csv("delivery_xgboost_predictions.csv")
    importance = load_csv("delivery_feature_importance.csv")

    if predictions is not None and not predictions.empty:

        st.subheader("📊 Prediction Overview")

        col1, col2 = st.columns(2)
        col1.metric("Total Predictions", f"{len(predictions):,}")
        col2.metric("Available Features", f"{len(predictions.columns):,}")

        st.subheader("🔎 Search Predictions")

        search = st.text_input(
            "Search in prediction results",
            placeholder="Enter an order ID or value..."
        )

        filtered = predictions.copy()

        if search.strip():
            mask = filtered.astype(str).apply(
                lambda column: column.str.contains(
                    search.strip(),
                    case=False,
                    na=False
                )
            ).any(axis=1)

            filtered = filtered[mask]

        st.caption(f"Showing {len(filtered):,} matching records")

        st.dataframe(
            filtered.head(200),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("Delivery prediction file not found.")

    st.subheader("⭐ Model Feature Importance")

    if importance is not None:
        st.dataframe(
            importance.head(15),
            use_container_width=True
        )

    show_image(
        "delivery_feature_importance.png",
        "Delivery Model Feature Importance"
    )


    st.title("🚚 Delivery Delay Prediction")

    show_image(
        "delivery_feature_importance.png",
        "Delivery Model Feature Importance"
    )

    predictions = load_csv("delivery_xgboost_predictions.csv")

    if predictions is not None:
        st.subheader("Model Predictions")
        st.dataframe(predictions.head(100), use_container_width=True)
    else:
        st.info("Delivery prediction report is not available.")


# RETENTION PAGE

# RETENTION ANALYSIS PAGE
elif page == "Retention Analysis":

    st.title("🔄 Customer Retention Analysis")
    st.write("Understand how customer activity changes across purchase cohorts.")

    cohort = load_csv("cohort_retention.csv")

    if cohort is not None and not cohort.empty:

        st.subheader("📊 Cohort Retention Data")

        st.metric("Cohort Records", f"{len(cohort):,}")

        st.dataframe(
            cohort,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("🔎 Filter Cohorts")

        cohort_column = next(
            (c for c in cohort.columns if "cohort" in c.lower()),
            None
        )

        if cohort_column:
            cohort_values = sorted(
                cohort[cohort_column].dropna().astype(str).unique()
            )

            selected_cohort = st.selectbox(
                "Select Cohort",
                ["All Cohorts"] + cohort_values
            )

            filtered = cohort.copy()

            if selected_cohort != "All Cohorts":
                filtered = filtered[
                    filtered[cohort_column].astype(str) == selected_cohort
                ]

            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True
            )

        st.subheader("📈 Retention Visualization")

        show_image(
            "cohort_retention.png",
            "Customer Cohort Retention"
        )

    else:
        st.warning("Cohort retention report not found.")


    st.title("🔄 Customer Retention & Cohort Analysis")

    cohort = load_csv("cohort_retention.csv")

    if cohort is not None:
        st.subheader("Cohort Retention Data")
        st.dataframe(cohort, use_container_width=True)

    show_image("cohort_retention.png", "Customer Cohort Retention")


# PRODUCT RECOMMENDATIONS PAGE

# PRODUCT RECOMMENDATIONS PAGE
elif page == "Product Recommendations":

    st.title("🛍️ Product Recommendation System")
    st.write("Search products and explore items frequently purchased together.")

    recommendations = load_csv("product_recommendations.csv")
    top_products = load_csv("top_products.csv")

    if top_products is not None:
        st.subheader("🏆 Top Purchased Products")
        st.dataframe(top_products, use_container_width=True)

    if recommendations is not None and not recommendations.empty:

        st.subheader("🔎 Find Product Recommendations")

        # Category filter
        categories = ["All Categories"] + sorted(
            recommendations["Product_Category"]
            .fillna("Unknown")
            .astype(str)
            .unique()
            .tolist()
        )

        selected_category = st.selectbox(
            "Select Product Category",
            categories
        )

        # Product search
        search_text = st.text_input(
            "Search Product ID",
            placeholder="Enter product ID..."
        )

        filtered = recommendations.copy()

        if selected_category != "All Categories":
            filtered = filtered[
                filtered["Product_Category"].fillna("Unknown")
                == selected_category
            ]

        if search_text.strip():
            filtered = filtered[
                filtered["Product_ID"].astype(str).str.contains(
                    search_text.strip(),
                    case=False,
                    na=False
                )
            ]

        st.metric("Matching Recommendations", f"{len(filtered):,}")

        st.subheader("Recommended Products")

        if not filtered.empty:
            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No matching recommendations found.")

        # Category chart
        category_counts = (
            recommendations["Product_Category"]
            .fillna("Unknown")
            .value_counts()
            .head(10)
            .reset_index()
        )

        category_counts.columns = [
            "Category",
            "Recommendation Count"
        ]

        fig = px.bar(
            category_counts,
            x="Category",
            y="Recommendation Count",
            title="Top Recommended Categories"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Product recommendation data is not available.")


    st.title("🛍️ Product Recommendation System")

    recommendations = load_csv("product_recommendations.csv")
    top_products = load_csv("top_products.csv")

    if top_products is not None:
        st.subheader("Top Purchased Products")
        st.dataframe(top_products, use_container_width=True)

    if recommendations is not None:
        st.subheader("Recommended Products")
        st.dataframe(recommendations.head(100), use_container_width=True)

        if "Product_Category" in recommendations.columns:
            category_counts = (
                recommendations["Product_Category"]
                .fillna("Unknown")
                .value_counts()
                .head(10)
                .reset_index()
            )
            category_counts.columns = ["Category", "Recommendations"]

            fig = px.bar(
                category_counts,
                x="Category",
                y="Recommendations",
                title="Top Recommended Categories"
            )
            st.plotly_chart(fig, use_container_width=True)


# ANOMALY DETECTION PAGE

# ANOMALY DETECTION PAGE
elif page == "Anomaly Detection":

    st.title("🚨 Anomaly Detection")
    st.write("Identify unusual patterns in e-commerce orders using Isolation Forest.")

    anomalies = load_csv("anomaly_detection_results.csv")

    if anomalies is not None and not anomalies.empty:

        total_orders = len(anomalies)
        unusual_count = (anomalies["Anomaly"] == "Unusual").sum()
        normal_count = (anomalies["Anomaly"] == "Normal").sum()
        anomaly_percentage = (
            unusual_count / total_orders * 100
            if total_orders else 0
        )

        # KPI CARDS
        st.subheader("📊 Anomaly Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Orders", f"{total_orders:,}")
        c2.metric("Normal Orders", f"{normal_count:,}")
        c3.metric("Unusual Orders", f"{unusual_count:,}")
        c4.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")

        st.divider()

        # PIE CHART
        st.subheader("Normal vs Unusual Orders")

        counts = anomalies["Anomaly"].value_counts().reset_index()
        counts.columns = ["Order Type", "Count"]

        fig = px.pie(
            counts,
            names="Order Type",
            values="Count",
            hole=0.45,
            title="Order Anomaly Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        # FILTER
        st.subheader("🔎 Explore Order Records")

        selected_type = st.selectbox(
            "Filter by Order Type",
            ["All", "Normal", "Unusual"]
        )

        filtered = anomalies.copy()

        if selected_type != "All":
            filtered = filtered[
                filtered["Anomaly"] == selected_type
            ]

        if "order_id" in filtered.columns:
            search = st.text_input(
                "Search Order ID",
                placeholder="Enter order ID..."
            )

            if search.strip():
                filtered = filtered[
                    filtered["order_id"].astype(str).str.contains(
                        search.strip(),
                        case=False,
                        na=False
                    )
                ]

        st.caption(f"Matching orders: {len(filtered):,}")

        st.dataframe(
            filtered.head(200),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        show_image(
            "anomaly_detection.png",
            "Order Anomaly Visualization"
        )

        st.info(
            "Unusual orders are statistical anomalies, not confirmed fraud."
        )

    else:
        st.warning("Anomaly detection report not found.")


    st.title("🚨 Anomaly Detection")

    anomalies = load_csv("anomaly_detection_results.csv")

    if anomalies is not None and "Anomaly" in anomalies.columns:

        counts = anomalies["Anomaly"].value_counts()

        col1, col2 = st.columns(2)
        col1.metric("Total Orders", f"{len(anomalies):,}")
        col2.metric(
            "Unusual Orders",
            f"{(anomalies['Anomaly'] == 'Unusual').sum():,}"
        )

        fig = px.pie(
            names=counts.index,
            values=counts.values,
            title="Normal vs Unusual Orders"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Unusual Order Details")
        unusual = anomalies[anomalies["Anomaly"] == "Unusual"]
        st.dataframe(unusual.head(100), use_container_width=True)

    show_image("anomaly_detection.png", "Anomaly Detection Visualization")


st.sidebar.divider()
st.sidebar.caption("ShopSense AI | Academic Portfolio Project")
