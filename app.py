import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

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
        "Geo Analytics",
        "AI Insights"
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

            <h1 style='
                color:white;
                margin-top:15px;
                font-size:42px;
            '>
                {len(data)}
            </h1>
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

            <h1 style='
                color:white;
                margin-top:15px;
                font-size:42px;
            '>
                {len(data.columns)}
            </h1>
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

            <h1 style='
                color:#22c55e;
                margin-top:15px;
                font-size:42px;
            '>
                Active
            </h1>
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

    if "risk_level" in data.columns:

        risk_counts = data["risk_level"].value_counts()

        risk_df = pd.DataFrame({
            "Risk": risk_counts.index,
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
# GEO ANALYTICS V3
# =====================================================

elif section == "Geo Analytics":

    st.subheader(
        "🌍 Real-Time Global Geo Analytics"
    )

    geo_data = pd.DataFrame({

        "City": [
            "New York",
            "Los Angeles",
            "Chicago",
            "London",
            "Paris",
            "Berlin",
            "Dubai",
            "Mumbai",
            "New Delhi",
            "Bangalore",
            "Singapore",
            "Tokyo",
            "Shanghai",
            "Sydney",
            "São Paulo"
        ],

        "Country": [
            "USA",
            "USA",
            "USA",
            "UK",
            "France",
            "Germany",
            "UAE",
            "India",
            "India",
            "India",
            "Singapore",
            "Japan",
            "China",
            "Australia",
            "Brazil"
        ],

        "Latitude": [
            40.7128,
            34.0522,
            41.8781,
            51.5074,
            48.8566,
            52.5200,
            25.2048,
            19.0760,
            28.6139,
            12.9716,
            1.3521,
            35.6762,
            31.2304,
            -33.8688,
            -23.5505
        ],

        "Longitude": [
            -74.0060,
            -118.2437,
            -87.6298,
            -0.1278,
            2.3522,
            13.4050,
            55.2708,
            72.8777,
            77.2090,
            77.5946,
            103.8198,
            139.6503,
            121.4737,
            151.2093,
            -46.6333
        ],

        "Demand": [
            15000,
            12000,
            9800,
            13200,
            10400,
            9100,
            14200,
            21000,
            23500,
            19400,
            14800,
            18200,
            22500,
            9600,
            12000
        ]
    })

    st.subheader("📊 Geo Analytics KPIs")

    g1, g2, g3 = st.columns(3)

    with g1:
        st.metric(
            "Locations",
            len(geo_data)
        )

    with g2:
        st.metric(
            "Total Demand",
            f"{geo_data['Demand'].sum():,}"
        )

    with g3:
        st.metric(
            "Average Demand",
            f"{geo_data['Demand'].mean():.0f}"
        )

    st.subheader(
        "🗺️ Interactive Logistics Map"
    )

    m = folium.Map(
        location=[20, 0],
        zoom_start=2,
        tiles="OpenStreetMap"
    )

    heat_data = [

        [
            row["Latitude"],
            row["Longitude"],
            row["Demand"]
        ]

        for _, row in geo_data.iterrows()
    ]

    HeatMap(
        heat_data,
        radius=25
    ).add_to(m)

    for _, row in geo_data.iterrows():

        popup_text = f"""
        <b>City:</b> {row['City']}<br>
        <b>Country:</b> {row['Country']}<br>
        <b>Demand:</b> {row['Demand']}
        """

        folium.CircleMarker(

            location=[
                row["Latitude"],
                row["Longitude"]
            ],

            radius=row["Demand"] / 3000,

            popup=popup_text,

            color="red",

            fill=True,

            fill_color="red",

            fill_opacity=0.7

        ).add_to(m)

    st_folium(
        m,
        width=1400,
        height=750
    )

    st.subheader(
        "📋 Demand Analytics Table"
    )

    sorted_geo = geo_data.sort_values(
        by="Demand",
        ascending=False
    )

    top_regions = sorted_geo.reset_index(drop=True)

    highest_city = top_regions.iloc[0]
    lowest_city = top_regions.iloc[-1]

    st.dataframe(
        top_regions,
        use_container_width=True
    )

    st.subheader(
        "📈 Demand Comparison"
    )

    fig_geo = px.bar(

        top_regions,

        x="City",

        y="Demand",

        color="Demand",

        text="Demand",

        color_continuous_scale="Turbo",

        title="Global Demand Distribution"
    )

    st.plotly_chart(
        fig_geo,
        use_container_width=True
    )

    st.subheader(
        "🤖 AI Generated Geo Insights"
    )

    st.success(
        f"""
        🚀 Highest global demand detected in
        {highest_city['City']},
        {highest_city['Country']}
        with total demand of
        {highest_city['Demand']:,} units.
        """
    )

    st.warning(
        f"""
        ⚠️ Lowest demand observed in
        {lowest_city['City']},
        {lowest_city['Country']}
        with only
        {lowest_city['Demand']:,} units.
        """
    )

    st.subheader(
        "📥 Export Geo Analytics Report"
    )

    geo_csv = top_regions.to_csv(
        index=False
    )

    st.download_button(
        label="Download Geo Analytics CSV",
        data=geo_csv,
        file_name="geo_analytics_report.csv",
        mime="text/csv"
    )

# =====================================================
# AI INSIGHTS
# =====================================================

elif section == "AI Insights":

    st.subheader(
        "🤖 AI Powered Business Insights"
    )

    monthly_columns = [
        "jan", "feb", "mar", "apr",
        "may", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec"
    ]

    total_demand = data[
        monthly_columns
    ].sum().sum()

    avg_inventory = data[
        "quantity_on_hand"
    ].mean()

    avg_lead_time = data[
        "lead-time"
    ].mean()

    backlog_products = len(
        data[data["backlog"] > 0]
    )

    high_risk_products = len(
        data[
            (data["quantity_on_hand"] < 1000)
            &
            (data["backlog"] > 0)
        ]
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            "Total Demand",
            f"{int(total_demand):,}"
        )

    with k2:
        st.metric(
            "Average Inventory",
            f"{avg_inventory:.0f}"
        )

    with k3:
        st.metric(
            "Average Lead Time",
            f"{avg_lead_time:.2f}"
        )

    with k4:
        st.metric(
            "Backlog Products",
            backlog_products
        )

    st.subheader(
        "🧠 Smart Recommendations"
    )

    if avg_inventory < 2000:

        st.warning(
            """
            ⚠️ Inventory levels are below recommended thresholds.
            """
        )

    else:

        st.success(
            """
            ✅ Inventory levels appear stable.
            """
        )

    if avg_lead_time > 10:

        st.error(
            """
            🚚 Lead time is significantly high.
            """
        )

    else:

        st.info(
            """
            🚀 Lead time performance is operating normally.
            """
        )

    st.subheader(
        "📈 AI Demand Trend Analysis"
    )

    monthly_demand = data[
        monthly_columns
    ].sum()

    top_month = monthly_demand.idxmax()

    lowest_month = monthly_demand.idxmin()

    st.success(
        f"📊 Highest demand observed during {top_month.upper()}"
    )

    st.warning(
        f"📉 Lowest demand observed during {lowest_month.upper()}"
    )

    st.subheader(
        "📋 Executive AI Summary"
    )

    st.info(
        f"""
        Total demand processed:
        {int(total_demand):,} units.

        Average inventory:
        {avg_inventory:.0f} units.

        Current logistics lead time:
        {avg_lead_time:.2f} days.
        """
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