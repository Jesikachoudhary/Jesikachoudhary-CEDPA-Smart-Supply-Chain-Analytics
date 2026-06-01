import numpy as np
import pandas as pd
import datetime

class AlertEngine:
    def __init__(self):
        self.priority_order = {"Critical": 0, "Warning": 1, "Info": 2}
        
    def generate_alerts(self, suppliers_df, skus_df, risk_model):
        """
        Scan all supplier-SKU linkages, predict dynamic disruption risk probabilities,
        evaluate inventory positions, and assemble a priority queue of actionable alerts.
        """
        alerts = []
        rng = np.random.default_rng(202)
        
        # 1. Evaluate suppliers using the trained ML Risk Predictor
        for _, supplier in suppliers_df.iterrows():
            # Create feature matrix representing supplier's state
            feat_dict = {
                "lead_time_variance": supplier["lead_time_std"] ** 2,
                "supplier_reliability": supplier["reliability"],
                "geo_risk_index": supplier["geo_risk"],
                "inventory_buffer": float(rng.uniform(-0.15, 0.6)),  # simulate active inventory position
                "shipment_delay_history": float(rng.uniform(0.5, 8.5))
            }
            
            # Predict disruption probability
            _, risk_prob = risk_model.predict(feat_dict)
            
            # Scenario A: ML Disruption Risk > 0.75 -> Critical Alert
            if risk_prob > 0.75:
                affected_sku = skus_df.sample(n=1, random_state=rng).iloc[0]
                increase_pct = int(10 + (risk_prob * 12))
                
                alerts.append({
                    "id": f"ALT-ML-{supplier['supplier_id'][-3:]}",
                    "timestamp": datetime.datetime.now() - datetime.timedelta(minutes=int(rng.uniform(10, 240))),
                    "priority": "Critical",
                    "category": "Disruption Risk",
                    "supplier_id": supplier["supplier_id"],
                    "supplier_name": supplier["supplier_name"],
                    "city": supplier["city"],
                    "sku_affected": affected_sku["sku_id"],
                    "risk_score": risk_prob,
                    "title": f"High Disruption Risk Detected at {supplier['city']} Hub",
                    "recommendation": (
                        f"Increase safety stock for {affected_sku['sku_id']} by {increase_pct}% "
                        f"due to high variance ({feat_dict['lead_time_variance']:.2f}) and reliability deficit "
                        f"({feat_dict['supplier_reliability']*100:.1f}%) in the {supplier['city']} supplier node."
                    )
                })
                
            # Scenario B: supplier reliability under 80% -> Warning Alert
            elif supplier["reliability"] < 0.80:
                affected_sku = skus_df.sample(n=1, random_state=rng).iloc[0]
                alerts.append({
                    "id": f"ALT-RL-{supplier['supplier_id'][-3:]}",
                    "timestamp": datetime.datetime.now() - datetime.timedelta(hours=int(rng.uniform(1, 12))),
                    "priority": "Warning",
                    "category": "Supplier Health",
                    "supplier_id": supplier["supplier_id"],
                    "supplier_name": supplier["supplier_name"],
                    "city": supplier["city"],
                    "sku_affected": affected_sku["sku_id"],
                    "risk_score": risk_prob,
                    "title": f"Supplier Reliability Deficit: {supplier['supplier_name']}",
                    "recommendation": (
                        f"Initiate secondary supplier sourcing protocols for {affected_sku['sku_id']} to hedge "
                        f"against deteriorating delivery performance ({supplier['reliability']*100:.1f}%) at "
                        f"the {supplier['city']} logistics terminal."
                    )
                })
                
        # 2. Evaluate SKU inventory buffers for Stockout alerts
        for _, sku in skus_df.sample(n=15, random_state=rng).iterrows():
            current_inv = int(sku["safety_stock"] * rng.uniform(0.2, 1.3))
            
            # If current inventory is less than 50% of required safety stock -> Warning/Critical
            if current_inv < (sku["safety_stock"] * 0.5):
                priority = "Critical" if current_inv < (sku["safety_stock"] * 0.25) else "Warning"
                reorder_qty = int(sku["safety_stock"] * 1.5 - current_inv)
                
                alerts.append({
                    "id": f"ALT-INV-{sku['sku_id'][-3:]}",
                    "timestamp": datetime.datetime.now() - datetime.timedelta(minutes=int(rng.uniform(5, 120))),
                    "priority": priority,
                    "category": "Stockout Danger",
                    "supplier_id": "N/A",
                    "supplier_name": "Internal Distribution",
                    "city": "Global Distribution",
                    "sku_affected": sku["sku_id"],
                    "risk_score": 0.65 if priority == "Critical" else 0.40,
                    "title": f"Critical Stock Deficit for {sku['sku_id']}",
                    "recommendation": (
                        f"Trigger immediate inventory replenishment order of {reorder_qty} units for {sku['sku_id']}. "
                        f"Current stock level ({current_inv} units) has fallen drastically below safety stock buffer "
                        f"threshold ({sku['safety_stock']} units)."
                    )
                })
                
            # If inventory is in warning zone (50% - 85% of safety stock) -> Info Alert
            elif current_inv < (sku["safety_stock"] * 0.85):
                reorder_qty = int(sku["safety_stock"] * 1.1 - current_inv)
                alerts.append({
                    "id": f"ALT-INF-{sku['sku_id'][-3:]}",
                    "timestamp": datetime.datetime.now() - datetime.timedelta(hours=int(rng.uniform(2, 24))),
                    "priority": "Info",
                    "category": "Inventory Reorder",
                    "sku_affected": sku["sku_id"],
                    "supplier_id": "N/A",
                    "supplier_name": "Internal Distribution",
                    "city": "Global Distribution",
                    "risk_score": 0.20,
                    "title": f"Reorder Threshold Triggered for {sku['sku_id']}",
                    "recommendation": (
                        f"Schedule standard inventory replenishment of {reorder_qty} units for {sku['sku_id']} "
                        f"during the upcoming procurement cycle to restore optimal buffer levels."
                    )
                })
                
        # 3. Sort by priority order (Critical -> Warning -> Info) and timestamp
        alerts_df = pd.DataFrame(alerts)
        if not alerts_df.empty:
            alerts_df["priority_num"] = alerts_df["priority"].map(self.priority_order)
            alerts_df = alerts_df.sort_values(by=["priority_num", "timestamp"], ascending=[True, False]).drop(columns=["priority_num"])
            return alerts_df.to_dict('records')
        return []
