
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

# --------------------------------------------------
# PROJECT SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
REPORTS = BASE_DIR / "reports"
MODELS = BASE_DIR / "models"

st.set_page_config(
    page_title="ShopSense AI",
    page_icon="🛒",
    layout="wide"
)

# --------------------------------------------------
# DARK THEME
# --------------------------------------------------

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
    color: white;
}
h1, h2, h3 {
    color: #67e8f9;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

@st.cache_data
def load_csv(filename):
    path = REPORTS / filename
    if path.exists():
        return pd.read_csv(path)
    return None


def find_column(df, keyword):
    """Find a column using a keyword, ignoring case."""
    if df is None:
        return None

    for column in df.columns:
        if keyword.lower() in str(column).lower():
            return column

    return None


def show_table(df, rows=200):
    if df is not None and not df.empty:
        st.dataframe(
            df.head(rows),
            use_container_width=True,
            hide_index=True
        )


def money(value):
    return f"R$ {value:,.0f}"


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

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


# ==================================================
# 1. OVERVIEW
# ==================================================

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

    if anomalies is not None and "Anomaly" in anomalies.columns:
        unusual = (anomalies["Anomaly"] == "Unusual").sum()
        col3.metric("Unusual Orders", f"{unusual:,}")
    else:
        col3.metric("Unusual Orders", "N/A")

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
        show_table(sales)
    else:
        st.warning("Sales forecast report not found.")

    if sales is not None and "Forecast Sales (BRL)" in sales.columns:
        fig = px.bar(
            sales,
            x="Month",
            y="Forecast Sales (BRL)",
            title="Monthly Sales Forecast"
        )
        st.plotly_chart(fig, use_container_width=True)

    if anomalies is not None and "Anomaly" in anomalies.columns:
        counts = anomalies["Anomaly"].value_counts().reset_index()
        counts.columns = ["Order Type", "Count"]

        fig = px.pie(
            counts,
            names="Order Type",
            values="Count",
            hole=0.4,
            title="Order Anomaly Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)


# ==================================================
# 2. SALES FORECASTING
# ==================================================

elif page == "Sales Forecasting":

    st.title("📊 Sales Intelligence & Forecasting")
    st.write("Explore historical sales performance and experimental forecasts.")

    evaluation = load_csv("sales_model_evaluation.csv")
    forecast = load_csv("sales_forecast.csv")

    if evaluation is not None and not evaluation.empty:

        st.subheader("📈 Actual vs Predicted Sales")

        if {"Month", "Actual Sales", "Predicted Sales"}.issubset(evaluation.columns):

            months = ["All Months"] + evaluation["Month"].astype(str).tolist()

            selected_month = st.selectbox(
                "Select Test Month",
                months,
                key="sales_month_filter"
            )

            filtered = evaluation.copy()

            if selected_month != "All Months":
                filtered = filtered[
                    filtered["Month"].astype(str) == selected_month
                ]

            mae = (
                filtered["Actual Sales"] - filtered["Predicted Sales"]
            ).abs().mean()

            c1, c2, c3 = st.columns(3)
            c1.metric("Selected Test Months", len(filtered))
            c2.metric("Mean Absolute Error", money(mae))
            c3.metric("Actual Sales", money(filtered["Actual Sales"].sum()))

            chart_data = filtered.melt(
                id_vars="Month",
                value_vars=["Actual Sales", "Predicted Sales"],
                var_name="Sales Type",
                value_name="Sales"
            )

            fig = px.bar(
                chart_data,
                x="Month",
                y="Sales",
                color="Sales Type",
                barmode="group",
                title="Actual vs Predicted Sales"
            )

            st.plotly_chart(fig, use_container_width=True)
            show_table(filtered)

        else:
            st.warning("Required sales evaluation columns are missing.")

    else:
        st.warning("Sales evaluation report not found.")

    st.divider()
    st.subheader("🔮 Experimental Sales Forecast")

    if forecast is not None and not forecast.empty:

        if {"Month", "Forecast Sales (BRL)"}.issubset(forecast.columns):

            forecast_months = ["All Months"] + forecast["Month"].astype(str).tolist()

            selected_forecast_month = st.selectbox(
                "Select Forecast Month",
                forecast_months,
                key="forecast_month_filter"
            )

            forecast_filtered = forecast.copy()

            if selected_forecast_month != "All Months":
                forecast_filtered = forecast_filtered[
                    forecast_filtered["Month"].astype(str)
                    == selected_forecast_month
                ]

            c1, c2 = st.columns(2)
            c1.metric(
                "Forecast Total",
                money(forecast_filtered["Forecast Sales (BRL)"].sum())
            )
            c2.metric(
                "Average Forecast",
                money(forecast_filtered["Forecast Sales (BRL)"].mean())
            )

            fig = px.bar(
                forecast_filtered,
                x="Month",
                y="Forecast Sales (BRL)",
                title="Forecasted Monthly Sales"
            )

            st.plotly_chart(fig, use_container_width=True)
            show_table(forecast_filtered)

        else:
            show_table(forecast)

    st.info(
        "This forecast uses historical Olist data ending in 2018. "
        "It is an experimental historical forecast, not a live 2026 forecast."
    )


# ==================================================
# 3. CUSTOMER SEGMENTATION
# ==================================================

elif page == "Customer Segmentation":

    st.title("👥 Customer Segmentation")
    st.write("Explore customers using Recency, Frequency and Monetary analysis.")

    rfm = load_csv("rfm_customers.csv")
    summary = load_csv("rfm_summary.csv")

    if summary is not None:
        st.subheader("📊 RFM Segment Summary")
        show_table(summary)

    if rfm is not None and not rfm.empty:

        segment_column = find_column(rfm, "segment")
        customer_column = find_column(rfm, "customer")
        monetary_column = find_column(rfm, "monetary")
        frequency_column = find_column(rfm, "frequency")

        filtered = rfm.copy()

        if segment_column:

            segments = sorted(
                rfm[segment_column].dropna().astype(str).unique()
            )

            selected_segment = st.selectbox(
                "Select Customer Segment",
                ["All Segments"] + segments,
                key="rfm_segment_filter"
            )

            if selected_segment != "All Segments":
                filtered = filtered[
                    filtered[segment_column].astype(str) == selected_segment
                ]

        if customer_column:

            search = st.text_input(
                "Search Customer ID",
                placeholder="Enter customer ID..."
            )

            if search.strip():
                filtered = filtered[
                    filtered[customer_column].astype(str).str.contains(
                        search.strip(),
                        case=False,
                        na=False
                    )
                ]

        st.metric("Matching Customers", f"{len(filtered):,}")

        st.subheader("📊 Filtered Customer Data")
        show_table(filtered)

        st.subheader("📈 Customer Segment Chart")

        if segment_column:

            segment_counts = (
                filtered[segment_column]
                .astype(str)
                .value_counts()
                .reset_index()
            )
            segment_counts.columns = ["Segment", "Customers"]

            fig = px.bar(
                segment_counts,
                x="Segment",
                y="Customers",
                title="Customers by Selected Segment"
            )

            st.plotly_chart(fig, use_container_width=True)

        if monetary_column and frequency_column:

            fig = px.scatter(
                filtered,
                x=frequency_column,
                y=monetary_column,
                color=segment_column if segment_column else None,
                title="Customer Frequency vs Monetary Value"
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("RFM customer data not found.")


# ==================================================
# 4. DELIVERY PREDICTION
# ==================================================

elif page == "Delivery Prediction":

    st.title("🚚 Delivery Delay Prediction")
    st.write("Explore model results and try a live delivery prediction.")

    st.subheader("🚀 Live Delivery Prediction")

    st.caption(
        "Enter order details to estimate the probability of delivery delay."
    )

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
            model_path = MODELS / "delivery_xgboost_model.joblib"
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

            st.metric(
                "Estimated Late Delivery Probability",
                f"{probability:.2%}"
            )

            if prediction == 1:
                st.warning("Model Prediction: Possible Delivery Delay")
            else:
                st.success("Model Prediction: Likely On-Time Delivery")

            st.caption(f"Classification threshold: {threshold:.4f}")

        except FileNotFoundError:
            st.error("Model file not found in the models folder.")
        except Exception as error:
            st.error(f"Prediction error: {error}")

    st.divider()

    predictions = load_csv("delivery_xgboost_predictions.csv")
    importance = load_csv("delivery_feature_importance.csv")

    if predictions is not None and not predictions.empty:

        st.subheader("📊 Prediction Results")

        st.metric("Total Predictions", f"{len(predictions):,}")

        search = st.text_input(
            "Search Prediction Records",
            placeholder="Enter an order ID or value..."
        )

        filtered_predictions = predictions.copy()

        if search.strip():

            mask = filtered_predictions.astype(str).apply(
                lambda column: column.str.contains(
                    search.strip(),
                    case=False,
                    na=False
                )
            ).any(axis=1)

            filtered_predictions = filtered_predictions[mask]

        st.caption(f"Matching records: {len(filtered_predictions):,}")
        show_table(filtered_predictions)

    if importance is not None and not importance.empty:

        st.subheader("⭐ Model Feature Importance")

        feature_column = find_column(importance, "feature")
        importance_column = find_column(importance, "importance")

        if feature_column and importance_column:

            top_features = importance.sort_values(
                importance_column,
                ascending=False
            ).head(15)

            fig = px.bar(
                top_features,
                x=importance_column,
                y=feature_column,
                orientation="h",
                title="Top 15 Important Features"
            )

            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        show_table(importance, 15)

    st.info(
        "The model provides experimental estimates. Its test performance is modest "
        "and predictions should not be treated as guaranteed delivery outcomes."
    )


# ==================================================
# 5. RETENTION ANALYSIS
# ==================================================

elif page == "Retention Analysis":

    st.title("🔄 Customer Retention Analysis")
    st.write("Explore how customer activity changes across purchase cohorts.")

    cohort = load_csv("cohort_retention.csv")

    if cohort is not None and not cohort.empty:

        st.metric("Cohort Records", f"{len(cohort):,}")

        cohort_column = find_column(cohort, "cohort")
        index_column = find_column(cohort, "cohort_index")
        retention_column = find_column(cohort, "retention")

        filtered_cohort = cohort.copy()

        if cohort_column:

            cohort_values = sorted(
                cohort[cohort_column].dropna().astype(str).unique()
            )

            selected_cohort = st.selectbox(
                "Select Customer Cohort",
                ["All Cohorts"] + cohort_values,
                key="retention_cohort_filter"
            )

            if selected_cohort != "All Cohorts":
                filtered_cohort = filtered_cohort[
                    filtered_cohort[cohort_column].astype(str)
                    == selected_cohort
                ]

        st.subheader("📊 Filtered Cohort Data")
        show_table(filtered_cohort)

        st.subheader("📈 Interactive Retention Chart")

        if index_column and retention_column:

            fig = px.line(
                filtered_cohort,
                x=index_column,
                y=retention_column,
                color=cohort_column if cohort_column else None,
                markers=True,
                title="Customer Retention by Cohort"
            )

            st.plotly_chart(fig, use_container_width=True)

        elif cohort_column:

            numeric_columns = filtered_cohort.select_dtypes(
                include="number"
            ).columns.tolist()

            if numeric_columns:

                selected_metric = st.selectbox(
                    "Select Cohort Metric",
                    numeric_columns,
                    key="retention_metric_filter"
                )

                chart_data = (
                    filtered_cohort.groupby(cohort_column)[selected_metric]
                    .sum()
                    .reset_index()
                )

                fig = px.bar(
                    chart_data,
                    x=cohort_column,
                    y=selected_metric,
                    title=f"{selected_metric} by Cohort"
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("No numeric cohort metric is available for charting.")

        else:
            st.warning(
                "Could not identify cohort columns. Check cohort_retention.csv."
            )

    else:
        st.warning("Cohort retention report not found.")


# ==================================================
# 6. PRODUCT RECOMMENDATIONS
# ==================================================

elif page == "Product Recommendations":

    st.title("🛍️ Product Recommendation System")
    st.write("Explore products frequently purchased together.")

    recommendations = load_csv("product_recommendations.csv")
    top_products = load_csv("top_products.csv")

    if top_products is not None:
        st.subheader("🏆 Top Purchased Products")
        show_table(top_products)

    if recommendations is not None and not recommendations.empty:

        category_column = find_column(recommendations, "Product_Category")
        product_column = find_column(recommendations, "Product_ID")

        filtered = recommendations.copy()

        if category_column:

            categories = ["All Categories"] + sorted(
                recommendations[category_column]
                .fillna("Unknown")
                .astype(str)
                .unique()
                .tolist()
            )

            selected_category = st.selectbox(
                "Select Product Category",
                categories,
                key="recommendation_category_filter"
            )

            if selected_category != "All Categories":
                filtered = filtered[
                    filtered[category_column].fillna("Unknown").astype(str)
                    == selected_category
                ]

        if product_column:

            search = st.text_input(
                "Search Product ID",
                placeholder="Enter product ID..."
            )

            if search.strip():
                filtered = filtered[
                    filtered[product_column].astype(str).str.contains(
                        search.strip(),
                        case=False,
                        na=False
                    )
                ]

        st.metric("Matching Recommendations", f"{len(filtered):,}")

        st.subheader("Recommended Products")
        show_table(filtered)

        if category_column:

            st.subheader("📊 Recommendations by Category")

            category_counts = (
                filtered[category_column]
                .fillna("Unknown")
                .value_counts()
                .head(10)
                .reset_index()
            )

            category_counts.columns = [
                "Category",
                "Recommendation Count"
            ]

            if not category_counts.empty:

                fig = px.bar(
                    category_counts,
                    x="Category",
                    y="Recommendation Count",
                    title="Recommendations for Selected Category"
                )

                st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Product recommendation data is not available.")


# ==================================================
# 7. ANOMALY DETECTION
# ==================================================

elif page == "Anomaly Detection":

    st.title("🚨 Anomaly Detection")
    st.write("Explore unusual e-commerce order patterns.")

    anomalies = load_csv("anomaly_detection_results.csv")

    if anomalies is not None and not anomalies.empty:

        if "Anomaly" in anomalies.columns:

            selected_type = st.selectbox(
                "Filter by Order Type",
                ["All", "Normal", "Unusual"],
                key="anomaly_type_filter"
            )

            filtered = anomalies.copy()

            if selected_type != "All":
                filtered = filtered[
                    filtered["Anomaly"] == selected_type
                ]

            search = st.text_input(
                "Search Order ID",
                placeholder="Enter order ID..."
            )

            if search.strip() and "order_id" in filtered.columns:
                filtered = filtered[
                    filtered["order_id"].astype(str).str.contains(
                        search.strip(),
                        case=False,
                        na=False
                    )
                ]

            total_orders = len(filtered)
            unusual_count = (filtered["Anomaly"] == "Unusual").sum()
            normal_count = (filtered["Anomaly"] == "Normal").sum()

            anomaly_rate = (
                unusual_count / total_orders * 100
                if total_orders else 0
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Filtered Orders", f"{total_orders:,}")
            c2.metric("Normal Orders", f"{normal_count:,}")
            c3.metric("Unusual Orders", f"{unusual_count:,}")
            c4.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

            st.subheader("📊 Filtered Anomaly Distribution")

            counts = (
                filtered["Anomaly"]
                .value_counts()
                .reset_index()
            )

            counts.columns = ["Order Type", "Count"]

            if not counts.empty:

                fig = px.pie(
                    counts,
                    names="Order Type",
                    values="Count",
                    hole=0.45,
                    title="Normal vs Unusual Orders"
                )

                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Order Details")
            st.caption(f"Matching orders: {len(filtered):,}")
            show_table(filtered)

            st.info(
                "Unusual orders are statistical anomalies, not confirmed fraud."
            )

        else:
            st.warning("Anomaly column is missing from the dataset.")

    else:
        st.warning("Anomaly detection report not found.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.caption("ShopSense AI | E-Commerce Intelligence Platform")
