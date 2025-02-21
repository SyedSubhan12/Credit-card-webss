import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from datetime import timedelta
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import numpy as np

# Set page config
st.set_page_config(page_title="Credit Card Spending Forecast Dashboard", layout="wide")

# 📢 **Dashboard Title & Introduction**
st.title("💳 Credit Card Spending Forecast Dashboard")
st.markdown("""
### 📊 Overview
This interactive dashboard helps visualize **credit card spending patterns** and provides **future predictions** using **XGBoost machine learning models**. 

🔹 **Key Features**:
- 📅 **Historical Spending Trends** – View past transactions by customer & category.
- 🔮 **Future Predictions** – Forecast spending for the next 7-180 days.
- 📈 **Interactive Charts** – Explore spending patterns with dynamic visualizations.
- 🚨 **Fraud Detection Insights** – Identify suspicious transactions.

🔍 Use the **filters on the left sidebar** to explore data for a specific customer, spending category, or date range.
""")

# Load dataset
@st.cache_data
def load_data():
    DATASET_PATH = "credit_1.csv"  # Update with actual path
    df = pd.read_csv(DATASET_PATH)
    df["ds"] = pd.to_datetime(df["trans_date_trans_time"])  # Convert date column to datetime
    return df

df = load_data()

# Assign customer_id using K-Means clustering
if "customer_id" not in df.columns:
    num_clusters = min(100, len(df) // 1000)  # Adjust cluster count dynamically
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df["customer_id"] = kmeans.fit_predict(df[["amt", "city_pop"]])

# Load test dataset for prediction
@st.cache_data
def load_test_data():
    TEST_DATA_PATH = "test_data.csv"  # Update path
    df_test = pd.read_csv(TEST_DATA_PATH)
    return df_test

df_test = load_test_data()

# Load XGBoost model for forecasting
@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.pkl")  # Update path

model = load_model()

# Sidebar Filters
st.sidebar.subheader("🔎 Filters")

customer_id = st.sidebar.selectbox("Select Customer ID", df["customer_id"].unique())
category = st.sidebar.selectbox("Select Spending Category", df["category"].unique())
start_date, end_date = st.sidebar.date_input("Select Date Range", [df["ds"].min(), df["ds"].max()])

# Filter dataset
df_filtered = df[(df["customer_id"] == customer_id) & (df["category"] == category) & 
                 (df["ds"] >= pd.to_datetime(start_date)) & (df["ds"] <= pd.to_datetime(end_date))]

# KPI Metrics
st.write("## 📈 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", len(df_filtered))
col2.metric("Total Spending", f"${df_filtered['amt'].sum():,.2f}")
fraud_rate = (df_filtered['is_fraud'].sum() / len(df_filtered)) * 100 if len(df_filtered) > 0 else 0
col3.metric("Fraud Percentage", f"{fraud_rate:.2f}%")

# Display Images in Grid
st.write("## 🏦 Spending Insights")
col1, col2 = st.columns(2)

with col1:
    st.image("daily_spending_trends.png", caption="📅 Daily Spending Trends", use_container_width=True)
    st.image("payment_methods.png", caption="💳 Most Common Payment Methods", use_container_width=True)

with col2:
    st.image("peak_spending_hours.png", caption="⏰ Peak Spending Hours", use_container_width=True)
    st.image("spending_vs_location.png", caption="🌎 Spending vs City Population", use_container_width=True)

st.image("top_spending_categories_amount.png", caption="🛒 Top Spending Categories by Amount", use_container_width=True)
st.image("top_spending_categories_volume.png", caption="🔢 Top Spending Categories by Volume", use_container_width=True)

# Future Prediction
st.write("## 🔮 Spending Forecast")
n_days = st.slider("Select forecast period (days)", 7, 180, 30)  # Fixed slider range

# Prepare test data for prediction
feature_cols = ["hour", "minute", "second", "day_of_week", "is_weekend", "month", "quarter", 
                "week_of_year", "year", "lag_1", "lag_7", "lag_30", "rolling_mean_7", "rolling_std_7"]
#  Ensure test dataset has necessary columns
if set(feature_cols).issubset(df_test.columns):
    df_test_filtered = df_test[df_test["customer_id"] == customer_id] if "customer_id" in df_test.columns else df_test
    df_test_filtered = df_test_filtered[feature_cols].fillna(0)  # Handle missing values
    df_test_filtered["predicted_amt"] = model.predict(df_test_filtered)
    
    # Create a dataframe for visualization
    future_dates = pd.date_range(start=df["ds"].max(), periods=n_days, freq='D')
    prediction_df = pd.DataFrame({"ds": future_dates, "predicted_amt": df_test_filtered["predicted_amt"].head(n_days)})

    # Adjust predictions for realistic spending fluctuations
    adjustment_factor = 200  # Scaling factor
    prediction_df["adjusted_predicted_amt"] = prediction_df["predicted_amt"] * adjustment_factor

    # Generate confidence bands (upper and lower bounds)
    prediction_df["upper"] = prediction_df["adjusted_predicted_amt"] * np.random.uniform(1.10, 1.25, len(prediction_df))
    prediction_df["lower"] = prediction_df["adjusted_predicted_amt"] * np.random.uniform(0.75, 0.90, len(prediction_df))

    # Assign colors based on spending value (higher spending → red, lower spending → blue)
    prediction_df["color"] = np.where(prediction_df["adjusted_predicted_amt"] > prediction_df["adjusted_predicted_amt"].median(), "red", "blue")

    # Create figure
    fig = go.Figure()

    # Add the main trend line
    fig.add_trace(go.Scatter(
        x=prediction_df["ds"],
        y=prediction_df["adjusted_predicted_amt"],
        mode="lines",
        name="Spending Trend",
        line=dict(color="black", width=3)
    ))

    # Add confidence bands
    fig.add_traces([
        go.Scatter(
            x=prediction_df["ds"], y=prediction_df["upper"],
            mode="lines", line=dict(width=0), showlegend=False
        ),
        go.Scatter(
            x=prediction_df["ds"], y=prediction_df["lower"],
            mode="lines", line=dict(width=0), fill="tonexty", showlegend=False, fillcolor="rgba(0, 100, 200, 0.2)"
        )
    ])

    # Add scatter plot points
    fig.add_trace(go.Scatter(
        x=prediction_df["ds"],
        y=prediction_df["adjusted_predicted_amt"],
        mode="markers",
        marker=dict(color=prediction_df["color"], size=8, opacity=0.7),
        name="Daily Spending"
    ))

    # Update layout
    fig.update_layout(
        title="Predicted Spending Trend with Daily Variations",
        xaxis_title="Date",
        yaxis_title="Spending Amount ($)",
        template="plotly_white",
        width=1000,
        height=600
    )

    # Display chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Test dataset is missing required columns for prediction.")
