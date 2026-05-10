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
# GEO ANALYTICS V2
# =====================================================

elif section == "Geo Analytics":

    st.subheader(
        "🌍 Global Supply Chain Geo Analytics V2"
    )

    geo_data = pd.DataFrame({

        "City": [
            "New York",
            "Los Angeles",
            "Chicago",
            "Houston",
            "Toronto",
            "London",
            "Paris",
            "Berlin",
            "Dubai",
            "Mumbai",
            "New Delhi",
            "Singapore",
            "Tokyo",
            "Shanghai",
            "Sydney",
            "São Paulo",
            "Johannesburg",
            "Bangkok",
            "Seoul",
            "Mexico City"
        ],

        "Country": [
            "USA",
            "USA",
            "USA",
            "USA",
            "Canada",
            "UK",
            "France",
            "Germany",
            "UAE",
            "India",
            "India",
            "Singapore",
            "Japan",
            "China",
            "Australia",
            "Brazil",
            "South Africa",
            "Thailand",
            "South Korea",
            "Mexico"
        ],

        "Latitude": [
            40.7128,
            34.0522,
            41.8781,
            29.7604,
            43.6532,
            51.5074,
            48.8566,
            52.5200,
            25.2048,
            19.0760,
            28.6139,
            1.3521,
            35.6762,
            31.2304,
            -33.8688,
            -23.5505,
            -26.2041,
            13.7563,
            37.5665,
            19.4326
        ],

        "Longitude": [
            -74.0060,
            -118.2437,
            -87.6298,
            -95.3698,
            -79.3832,
            -0.1278,
            2.3522,
            13.4050,
            55.2708,
            72.8777,
            77.2090,
            103.8198,
            139.6503,
            121.4737,
            151.2093,
            -46.6333,
            28.0473,
            100.5018,
            126.9780,
            -99.1332
        ],

        "Demand": [
            15000,
            12000,
            9800,
            8500,
            7200,
            13200,
            10400,
            9100,
            14200,
            21000,
            23500,
            14800,
            18200,
            22500,
            9600,
            12000,
            4300,
            11100,
            15400,
            12400
        ]
    })

    st.subheader("📊 Global Geo KPIs")

    g1, g2, g3, g4 = st.columns(4)

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
            "High Demand Regions",
            len(
                geo_data[
                    geo_data["Demand"] > 15000
                ]
            )
        )

    with g4:
        st.metric(
            "Average Demand",
            f"{geo_data['Demand'].mean():.0f}"
        )

    st.subheader(
        "🌎 Geo Analytics Filters"
    )

    country_filter = st.multiselect(
        "Select Countries",
        options=geo_data["Country"].unique(),
        default=geo_data["Country"].unique()
    )

    demand_filter = st.slider(
        "Minimum Demand",
        min_value=0,
        max_value=int(
            geo_data["Demand"].max()
        ),
        value=5000
    )

    filtered_geo = geo_data[
        (geo_data["Country"].isin(country_filter))
        &
        (geo_data["Demand"] >= demand_filter)
    ]

    st.subheader(
        "🔥 Global Demand Density Heatmap"
    )

    density_map = px.density_mapbox(
        filtered_geo,
        lat="Latitude",
        lon="Longitude",
        z="Demand",
        radius=35,
        zoom=1.3,
        center=dict(
            lat=20,
            lon=0
        ),
        height=750,
        mapbox_style="carto-positron",
        color_continuous_scale="Turbo"
    )

    density_map.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=40,
            b=0
        )
    )

    st.plotly_chart(
        density_map,
        use_container_width=True
    )

    st.subheader(
        "📍 Interactive Logistics Demand Map"
    )

    city_map = px.scatter_mapbox(
        filtered_geo,
        lat="Latitude",
        lon="Longitude",
        hover_name="City",
        hover_data={
            "Country": True,
            "Demand": True
        },
        size="Demand",
        color="Demand",
        size_max=35,
        zoom=1.3,
        height=800,
        width=None,
        color_continuous_scale="Turbo"
    )

    city_map.update_layout(
        mapbox_style="open-street-map",
        margin=dict(
            l=0,
            r=0,
            t=40,
            b=0
        )
    )

    st.plotly_chart(
        city_map,
        use_container_width=True
    )

    st.subheader(
        "🏆 Highest Demand Regions"
    )

    top_regions = filtered_geo.sort_values(
        by="Demand",
        ascending=False
    )

    top_regions = top_regions.reset_index(
        drop=True
    )

    st.dataframe(
        top_regions,
        use_container_width=True
    )

    st.subheader(
        "📈 Global Demand Comparison"
    )

    demand_chart = px.bar(
        top_regions,
        x="City",
        y="Demand",
        color="Demand",
        text="Demand",
        color_continuous_scale="Turbo",
        title="Demand Distribution Across Global Cities"
    )

    demand_chart.update_layout(
        height=600
    )

    st.plotly_chart(
        demand_chart,
        use_container_width=True
    )

    st.subheader(
        "🌐 Country-Wise Demand Analysis"
    )

    country_analysis = filtered_geo.groupby(
        "Country"
    )["Demand"].sum().reset_index()

    country_chart = px.pie(
        country_analysis,
        names="Country",
        values="Demand",
        hole=0.45,
        title="Country Wise Demand Share"
    )

    st.plotly_chart(
        country_chart,
        use_container_width=True
    )

    st.subheader(
        "🤖 AI Generated Geo Insights"
    )

    highest_city = top_regions.iloc[0]
    lowest_city = top_regions.iloc[-1]

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

    # --------------------------------
    # KPI CARDS
    # --------------------------------

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

    # --------------------------------
    # AI RECOMMENDATIONS
    # --------------------------------

    st.subheader(
        "🧠 Smart Recommendations"
    )

    if avg_inventory < 2000:

        st.warning(
            """
            ⚠️ Inventory levels are below
            recommended thresholds.
            Consider increasing warehouse stock.
            """
        )

    else:

        st.success(
            """
            ✅ Inventory levels appear stable
            across supply chain operations.
            """
        )

    if avg_lead_time > 10:

        st.error(
            """
            🚚 Lead time is significantly high.
            Supplier optimization is recommended.
            """
        )

    else:

        st.info(
            """
            🚀 Lead time performance is operating
            within safe logistics range.
            """
        )

    if high_risk_products > 0:

        st.error(
            f"""
            ⚠️ {high_risk_products}
            high-risk products detected.
            Immediate replenishment recommended.
            """
        )

    else:

        st.success(
            """
            ✅ No critical supply chain risks
            detected currently.
            """
        )

    # --------------------------------
    # DEMAND TREND ANALYSIS
    # --------------------------------

    st.subheader(
        "📈 AI Demand Trend Analysis"
    )

    monthly_demand = data[
        monthly_columns
    ].sum()

    top_month = monthly_demand.idxmax()

    lowest_month = monthly_demand.idxmin()

    st.success(
        f"""
        📊 Highest demand observed during
        {top_month.upper()}.
        """
    )

    st.warning(
        f"""
        📉 Lowest demand observed during
        {lowest_month.upper()}.
        """
    )

    # --------------------------------
    # AI SUMMARY
    # --------------------------------

    st.subheader(
        "📋 Executive AI Summary"
    )

    st.info(
        f"""
        The AI engine analyzed supply chain
        operations across inventory,
        demand forecasting,
        logistics efficiency,
        and backlog risk.

        Total demand processed:
        {int(total_demand):,} units.

        Average inventory:
        {avg_inventory:.0f} units.

        Current logistics lead time:
        {avg_lead_time:.2f} days.

        The platform recommends
        continuous monitoring of
        high-demand periods and
        proactive inventory optimization.
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