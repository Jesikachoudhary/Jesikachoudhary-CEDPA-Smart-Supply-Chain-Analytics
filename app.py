import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.linear_model import LinearRegression
from app.ml_models.risk_prediction import train_risk_model


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="CEDPA Smart Supply Chain Analytics",
    layout="wide",
    page_icon="📊"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #050816;
    }

    h1, h2, h3, h4 {
        color: white;
    }

    p {
        color: #cbd5e1;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    .metric-card {
        background: linear-gradient(135deg, #111827, #1e293b);
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #334155;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    }

    .metric-title {
        color: #94a3b8;
        font-size: 18px;
    }

    .metric-value {
        color: white;
        font-size: 38px;
        font-weight: bold;
    }

    .hero-title {
        font-size: 52px;
        font-weight: 800;
        color: white;
    }

    .hero-subtitle {
        font-size: 20px;
        color: #94a3b8;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# HERO SECTION
# =====================================================

st.markdown(
    """
    <div class="hero-title">
    CEDPA Smart Supply Chain Analytics Platform
    </div>

    <div class="hero-subtitle">
    AI Powered Predictive Analytics • Demand Forecasting • Inventory Intelligence
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📌 CEDPA Navigation")

section = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "Dashboard Overview",
        "AI Risk Prediction",
        "Demand Analysis",
        "Inventory Insights",
        "Demand Forecasting",
        "Geo Analytics"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Supply Chain CSV",
    type=["csv"]
)

# =====================================================
# LOAD DATA
# =====================================================

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

else:

    data = pd.read_csv(
        "app/data/sample_data/complete_dataset_large.csv"
    )

# =====================================================
# KPI SECTION
# =====================================================

st.subheader("📈 Business KPI Dashboard")

k1, k2, k3 = st.columns(3)

with k1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
            Total Records
            </div>

            <div class="metric-value">
            {len(data)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
            Total Columns
            </div>

            <div class="metric-value">
            {len(data.columns)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">
            System Status
            </div>

            <div class="metric-value">
            Active
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# DASHBOARD OVERVIEW
# =====================================================

if section == "Dashboard Overview":

    st.subheader("📦 Supply Chain Dataset")

    st.dataframe(
        data.head(10),
        use_container_width=True
    )

    st.subheader("📋 Dataset Columns")

    st.write(data.columns.tolist())

# =====================================================
# AI RISK PREDICTION
# =====================================================

elif section == "AI Risk Prediction":

    st.subheader(
        "🤖 AI Supply Chain Risk Prediction"
    )

    model, accuracy = train_risk_model(data)

    st.success(
        f"Model Accuracy: {accuracy:.2f}"
    )

    risk_counts = data[
        "risk_level"
    ].value_counts()

    risk_df = pd.DataFrame({
        "Risk": [
            "Low Risk",
            "High Risk"
        ],
        "Count": risk_counts.values
    })

    fig_risk = px.pie(
        risk_df,
        names="Risk",
        values="Count",
        hole=0.5,
        title="Supply Chain Risk Distribution"
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )

    st.subheader(
        "🔮 Predict Supply Chain Risk"
    )

    col1, col2 = st.columns(2)

    with col1:

        jan = st.number_input(
            "January Demand",
            value=2000
        )

        feb = st.number_input(
            "February Demand",
            value=1800
        )

        mar = st.number_input(
            "March Demand",
            value=2200
        )

        apr = st.number_input(
            "April Demand",
            value=2100
        )

        may = st.number_input(
            "May Demand",
            value=1900
        )

        jun = st.number_input(
            "June Demand",
            value=2300
        )

    with col2:

        jul = st.number_input(
            "July Demand",
            value=2400
        )

        aug = st.number_input(
            "August Demand",
            value=2500
        )

        sep = st.number_input(
            "September Demand",
            value=2100
        )

        octo = st.number_input(
            "October Demand",
            value=2600
        )

        nov = st.number_input(
            "November Demand",
            value=2700
        )

        dec = st.number_input(
            "December Demand",
            value=3000
        )

    lead_time = st.number_input(
        "Lead Time",
        value=5
    )

    quantity_on_hand = st.number_input(
        "Quantity On Hand",
        value=500
    )

    backlog = st.number_input(
        "Backlog",
        value=1
    )

    if st.button("Predict Risk"):

        input_data = [[
            jan, feb, mar, apr,
            may, jun, jul, aug,
            sep, octo, nov, dec,
            lead_time,
            quantity_on_hand,
            backlog
        ]]

        prediction = model.predict(
            np.array(input_data)
        )

        if prediction[0] == 1:

            st.error(
                "⚠️ High Supply Chain Risk Detected"
            )

        else:

            st.success(
                "✅ Low Supply Chain Risk"
            )

# =====================================================
# DEMAND ANALYSIS
# =====================================================

elif section == "Demand Analysis":

    st.subheader(
        "📉 Monthly Demand Analysis"
    )

    monthly_columns = [
        "jan", "feb", "mar", "apr",
        "may", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec"
    ]

    monthly_demand = data[
        monthly_columns
    ].sum()

    demand_df = pd.DataFrame({
        "Month": monthly_columns,
        "Demand": monthly_demand.values
    })

    fig = px.line(
        demand_df,
        x="Month",
        y="Demand",
        markers=True,
        title="Monthly Demand Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# INVENTORY INSIGHTS
# =====================================================

elif section == "Inventory Insights":

    st.subheader(
        "📦 Inventory Distribution"
    )

    fig2 = px.histogram(
        data,
        x="quantity_on_hand",
        nbins=20,
        title="Inventory Quantity Distribution"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader(
        "🚚 Lead Time Analysis"
    )

    fig3 = px.box(
        data,
        y="lead-time",
        title="Lead Time Distribution"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# =====================================================
# DEMAND FORECASTING
# =====================================================

elif section == "Demand Forecasting":

    st.subheader(
        "📈 Demand Forecasting"
    )

    monthly_columns = [
        "jan", "feb", "mar", "apr",
        "may", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec"
    ]

    monthly_avg = data[
        monthly_columns
    ].mean()

    forecast_df = pd.DataFrame({
        "Month": monthly_columns,
        "Average Demand": monthly_avg.values
    })

    forecast_df["Forecast"] = (
        forecast_df["Average Demand"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    fig_forecast = px.line(
        forecast_df,
        x="Month",
        y=["Average Demand", "Forecast"],
        markers=True,
        title="Demand Forecasting Analysis"
    )

    st.plotly_chart(
        fig_forecast,
        use_container_width=True
    )

# =====================================================
# GEO ANALYTICS
# =====================================================

elif section == "Geo Analytics":

    st.subheader(
        "🌍 Global Supply Chain Geo Analytics"
    )

    geo_data = pd.DataFrame({

        "City": [
            "New York",
            "Los Angeles",
            "London",
            "Berlin",
            "Dubai",
            "Mumbai",
            "New Delhi",
            "Singapore",
            "Shanghai",
            "Sydney",
            "São Paulo",
            "Johannesburg"
        ],

        "Country": [
            "USA",
            "USA",
            "UK",
            "Germany",
            "UAE",
            "India",
            "India",
            "Singapore",
            "China",
            "Australia",
            "Brazil",
            "South Africa"
        ],

        "Latitude": [
            40.7128,
            34.0522,
            51.5074,
            52.5200,
            25.2048,
            19.0760,
            28.6139,
            1.3521,
            31.2304,
            -33.8688,
            -23.5505,
            -26.2041
        ],

        "Longitude": [
            -74.0060,
            -118.2437,
            -0.1278,
            13.4050,
            55.2708,
            72.8777,
            77.2090,
            103.8198,
            121.4737,
            151.2093,
            -46.6333,
            28.0473
        ],

        "Demand": [
            12500,
            8200,
            11400,
            6800,
            9600,
            16200,
            18700,
            13200,
            20100,
            9100,
            7800,
            2700
        ]
    })

    st.subheader(
        "🗺️ Interactive Supply Chain Demand Map"
    )

    fig_map = px.scatter_mapbox(

        geo_data,

        lat="Latitude",

        lon="Longitude",

        hover_name="City",

        hover_data={
            "Country": True,
            "Demand": True
        },

        size="Demand",

        color="Demand",

        color_continuous_scale="Turbo",

        zoom=1.2,

        height=750
    )

    fig_map.update_layout(

        mapbox_style="open-street-map",

        margin=dict(
            l=0,
            r=0,
            t=50,
            b=0
        )
    )

    st.plotly_chart(
        fig_map,
        use_container_width=True
    )

    st.subheader(
        "📋 Location Demand Details"
    )

    st.dataframe(
        geo_data,
        use_container_width=True
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <h4 style='color:gray;'>
    Developed by Jesika Choudhary •
    AI Powered Supply Chain Analytics
    </h4>
    </center>
    """,
    unsafe_allow_html=True
)