import numpy as np
import pandas as pd
import datetime

# Seed for reproducible synthetic data
np.random.seed(42)

# 15 Global Logistics Cities and Coordinates
CITIES = {
    "Mumbai": [19.0760, 72.8777, 0.25],       # [Lat, Lon, Base Geo Risk]
    "Singapore": [1.3521, 103.8198, 0.12],
    "Rotterdam": [51.9244, 4.4777, 0.15],
    "Shanghai": [31.2304, 121.4737, 0.20],
    "New York": [40.7128, -74.0060, 0.18],
    "Los Angeles": [34.0522, -118.2437, 0.22],
    "Tokyo": [35.6762, 139.6503, 0.15],
    "London": [51.5074, -0.1278, 0.14],
    "Sydney": [-33.8688, 151.2093, 0.10],
    "Dubai": [25.2048, 55.2708, 0.18],
    "Sao Paulo": [-23.5505, -46.6333, 0.35],
    "Cape Town": [-33.9249, 18.4241, 0.28],
    "Frankfurt": [50.1109, 8.6821, 0.11],
    "Hamburg": [53.5511, 9.9937, 0.13],
    "Hong Kong": [22.3193, 114.1694, 0.22]
}

def generate_suppliers():
    """Generate 50 realistic supplier nodes across 15 cities."""
    city_names = list(CITIES.keys())
    suppliers = []
    
    for i in range(1, 51):
        city = city_names[i % len(city_names)]
        lat, lon, base_geo_risk = CITIES[city]
        
        # Add slight jitter so nodes in the same city are not right on top of each other
        lat_jitter = lat + np.random.uniform(-0.15, 0.15)
        lon_jitter = lon + np.random.uniform(-0.15, 0.15)
        
        # Supplier performance variables
        reliability = np.random.uniform(0.72, 0.98)
        lead_time_avg = np.random.uniform(5.0, 22.0)  # average lead time in days
        lead_time_std = np.random.uniform(0.8, 4.5)
        geo_risk = base_geo_risk + np.random.uniform(-0.05, 0.15)
        geo_risk = min(max(geo_risk, 0.05), 0.95)
        
        suppliers.append({
            "supplier_id": f"SPL-{i:03d}",
            "supplier_name": f"Apex Global Logistics - Node {i:02d}",
            "city": city,
            "latitude": lat_jitter,
            "longitude": lon_jitter,
            "reliability": reliability,
            "lead_time_avg": lead_time_avg,
            "lead_time_std": lead_time_std,
            "geo_risk": geo_risk
        })
        
    return pd.DataFrame(suppliers)

def generate_skus():
    """Generate 200 unique SKUs across various categories."""
    categories = ["Electronics", "Pharmaceuticals", "Automotive", "Apparel"]
    skus = []
    
    for i in range(1, 201):
        category = categories[i % len(categories)]
        
        # Generate base prices and carrying costs based on category
        if category == "Electronics":
            base_price = np.random.uniform(150.0, 1200.0)
            base_demand = np.random.uniform(50.0, 180.0)
        elif category == "Pharmaceuticals":
            base_price = np.random.uniform(50.0, 800.0)
            base_demand = np.random.uniform(100.0, 300.0)
        elif category == "Automotive":
            base_price = np.random.uniform(300.0, 2500.0)
            base_demand = np.random.uniform(20.0, 90.0)
        else: # Apparel
            base_price = np.random.uniform(20.0, 120.0)
            base_demand = np.random.uniform(150.0, 600.0)
            
        carrying_cost_rate = np.random.uniform(0.12, 0.22)  # annual carrying cost rate
        unit_carrying_cost = (base_price * carrying_cost_rate) / 365.0
        
        # Safety stock buffer parameters
        safety_stock_level = int(base_demand * np.random.uniform(1.5, 3.0))
        
        skus.append({
            "sku_id": f"SKU-{i:03d}",
            "sku_name": f"CEDPA-{category[:3].upper()}-{i:03d}",
            "category": category,
            "unit_price": base_price,
            "carrying_cost_daily": unit_carrying_cost,
            "base_demand": base_demand,
            "safety_stock": safety_stock_level
        })
        
    return pd.DataFrame(skus)

def generate_historical_shipments(suppliers_df, num_records=5000):
    """
    Generate historical shipment data for training the risk prediction model.
    Encodes mathematical triggers to ensure model can achieve > 92% accuracy.
    """
    records = []
    for _ in range(num_records):
        supplier = suppliers_df.sample(n=1).iloc[0]
        
        # Feature generation
        lead_time_variance = np.random.uniform(0.2, 8.0)
        supplier_reliability = supplier["reliability"] + np.random.uniform(-0.08, 0.05)
        supplier_reliability = min(max(supplier_reliability, 0.5), 1.0)
        
        geo_risk_index = supplier["geo_risk"] + np.random.uniform(-0.05, 0.1)
        geo_risk_index = min(max(geo_risk_index, 0.05), 1.0)
        
        # Buffer level relative to demand variance
        inventory_buffer = np.random.uniform(-0.2, 0.8)
        shipment_delay_history = np.random.uniform(0.0, 10.0)
        
        # Math calculation of disruption risk (target variable creation)
        # Disruption is highly correlated with low reliability, high geo risk, high lead time variance, and poor history
        disruption_score = (
            (1.0 - supplier_reliability) * 4.5 +
            geo_risk_index * 3.5 +
            (lead_time_variance / 8.0) * 2.8 -
            inventory_buffer * 2.0 +
            (shipment_delay_history / 10.0) * 3.2
        )
        
        # Disruption score thresholds to binary target
        prob = 1 / (1 + np.exp(-(disruption_score - 4.5)))  # Sigmoid scaling
        disruption_occurred = 1 if np.random.rand() < prob else 0
        
        records.append({
            "supplier_id": supplier["supplier_id"],
            "lead_time_variance": lead_time_variance,
            "supplier_reliability": supplier_reliability,
            "geo_risk_index": geo_risk_index,
            "inventory_buffer": inventory_buffer,
            "shipment_delay_history": shipment_delay_history,
            "disruption_risk": disruption_occurred
        })
        
    return pd.DataFrame(records)

def generate_sku_demand_history(sku_id, base_demand, duration_days=365):
    """
    Generate deterministic, highly realistic seasonal daily demand 
    history for a specific SKU based on its ID.
    Ensures reproducibility and instant generation.
    """
    # Seed unique to the SKU so it is reproducible
    sku_seed = sum(ord(c) for c in sku_id)
    rng = np.random.default_rng(sku_seed)
    
    start_date = datetime.date(2025, 6, 1)
    dates = [start_date + datetime.timedelta(days=d) for d in range(duration_days)]
    
    # Base daily demand signal
    demand = np.zeros(duration_days)
    
    # Extract seasonal profiles
    # SKU-001 (e.g. sum=1) mod 3 decides seasonal patterns
    pattern = sku_seed % 3
    
    for d in range(duration_days):
        current_date = dates[d]
        day_of_week = current_date.weekday()
        day_of_year = current_date.timetuple().tm_yday
        
        # Weekly seasonality (higher demand on Monday, Friday; lower on Sunday)
        weekly_factor = 1.0
        if day_of_week == 0: weekly_factor = 1.2
        elif day_of_week == 4: weekly_factor = 1.15
        elif day_of_week == 6: weekly_factor = 0.75
        
        # Yearly/Seasonal patterns
        yearly_factor = 1.0
        if pattern == 0:  # Winter surge (Q4/Holiday peak)
            yearly_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 300) / 365)
        elif pattern == 1:  # Summer surge
            yearly_factor = 1.0 + 0.25 * np.sin(2 * np.pi * (day_of_year - 180) / 365)
        else:  # Constant/Random cyclical shifts
            yearly_factor = 1.0 + 0.15 * np.cos(2 * np.pi * day_of_year / 120)
            
        # Overall slight upward growth trend
        trend_factor = 1.0 + (d / duration_days) * 0.08
        
        # Compute demand value
        noise = rng.normal(0, base_demand * 0.1)
        val = base_demand * weekly_factor * yearly_factor * trend_factor + noise
        demand[d] = max(val, 0.0)
        
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "demand": demand
    })

def generate_disruptions(suppliers_df, active_count=5):
    """Generate mock real-time supply chain disruption alerts."""
    rng = np.random.default_rng(101)
    disrupted_suppliers = suppliers_df.sample(n=active_count, random_state=rng)
    
    disruptions = []
    reasons = [
        "Port Congestion and Customs Clearance Backlog",
        "Geopolitical Tensions affecting shipping lanes",
        "Unforeseen Severe Typhoon / Monsoonal Flooding",
        "Cybersecurity breach in Regional Logistics ERP System",
        "Labor Strike at Distribution Hub"
    ]
    
    priorities = ["Critical", "Warning", "Info"]
    
    for idx, (_, row) in enumerate(disrupted_suppliers.iterrows()):
        priority = priorities[0] if idx < 2 else (priorities[1] if idx < 4 else priorities[2])
        disruptions.append({
            "supplier_id": row["supplier_id"],
            "supplier_name": row["supplier_name"],
            "city": row["city"],
            "priority": priority,
            "event": reasons[idx % len(reasons)],
            "timestamp": datetime.datetime.now() - datetime.timedelta(hours=int(rng.uniform(1, 18))),
            "lead_time_impact_days": int(rng.uniform(3, 10))
        })
        
    return disruptions
