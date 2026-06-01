"""
DataCo Supply Chain Dataset → CEDPA 6-feature converter.

Column mapping
--------------
lead_time_variance    ← (Days for shipping real − scheduled)²  clipped [0.2, 9.0]
supplier_reliability  ← scheduled / real  clipped [0.55, 1.0]
geo_risk_index        ← Order Country  (Spanish + English lookup, 60+ countries)
inventory_buffer      ← Order Item Profit Ratio  normalised to [−0.5, 0.8]
shipment_delay_history← delay days (real − scheduled, ≥0) scaled 0–10
disruption_risk       ← Late_delivery_risk  (already 0/1 — no transformation needed)
"""

import pandas as pd
import numpy as np

DATACO_PATH = "data/DataCoSupplyChainDataset.csv"

# Country name → geographic risk score
# Includes Spanish / Portuguese names as they appear in DataCo
DATACO_COUNTRY_RISK: dict[str, float] = {
    # ── North America ──────────────────────────────
    "estados unidos": 0.18, "usa": 0.18, "united states": 0.18, "us": 0.18,
    "canadá": 0.13,  "canada": 0.13,
    "méxico": 0.30,  "mexico": 0.30,
    # ── Central America / Caribbean ───────────────
    "el salvador": 0.35,
    "guatemala": 0.36,
    "honduras": 0.38,
    "nicaragua": 0.38,
    "costa rica": 0.30,
    "panamá": 0.32, "panama": 0.32,
    "cuba": 0.40,
    "república dominicana": 0.32, "republica dominicana": 0.32,
    "puerto rico": 0.25,
    "haití": 0.50, "haiti": 0.50,
    # ── South America ─────────────────────────────
    "brasil": 0.32, "brazil": 0.32,
    "argentina": 0.35,
    "colombia": 0.35,
    "venezuela": 0.42,
    "perú": 0.37,   "peru": 0.37,
    "chile": 0.25,
    "ecuador": 0.36,
    "bolivia": 0.40,
    "paraguay": 0.38,
    "uruguay": 0.30,
    # ── Europe ────────────────────────────────────
    "alemania": 0.11, "germany": 0.11,
    "francia": 0.14,  "france": 0.14,
    "reino unido": 0.13, "uk": 0.13, "united kingdom": 0.13,
    "españa": 0.16,  "espana": 0.16, "spain": 0.16,
    "italia": 0.18,  "italy": 0.18,
    "portugal": 0.15,
    "países bajos": 0.12, "paises bajos": 0.12, "netherlands": 0.12,
    "bélgica": 0.12, "belgica": 0.12, "belgium": 0.12,
    "suiza": 0.10,   "switzerland": 0.10,
    "suecia": 0.11,  "sweden": 0.11,
    "noruega": 0.12, "norway": 0.12,
    "dinamarca": 0.11, "denmark": 0.11,
    "polonia": 0.17, "poland": 0.17,
    "rusia": 0.38,   "russia": 0.38,
    "ucrania": 0.45, "ukraine": 0.45,
    "turquía": 0.28, "turquia": 0.28, "turkey": 0.28,
    "grecia": 0.22,  "greece": 0.22,
    # ── Asia ──────────────────────────────────────
    "china": 0.22,
    "japón": 0.15,   "japon": 0.15, "japan": 0.15,
    "corea del sur": 0.16, "south korea": 0.16, "korea": 0.16,
    "india": 0.25,
    "indonesia": 0.28,
    "vietnam": 0.28, "viet nam": 0.28,
    "tailandia": 0.22, "thailand": 0.22,
    "malasia": 0.20, "malaysia": 0.20,
    "filipinas": 0.30, "philippines": 0.30,
    "singapur": 0.11, "singapore": 0.11,
    "bangladesh": 0.38,
    "pakistán": 0.40, "pakistan": 0.40,
    "myanmar": 0.40, "birmania": 0.40,
    "camboya": 0.35, "cambodia": 0.35,
    # ── Middle East ───────────────────────────────
    "arabia saudita": 0.22, "saudi arabia": 0.22,
    "emiratos árabes": 0.18, "uae": 0.18, "united arab emirates": 0.18,
    "iraq": 0.55, "irán": 0.50, "iran": 0.50,
    "israel": 0.25,
    # ── Africa ────────────────────────────────────
    "nigeria": 0.48,
    "kenia": 0.38,   "kenya": 0.38,
    "etiopía": 0.42, "ethiopia": 0.42,
    "ghana": 0.36,
    "sudáfrica": 0.30, "sudafrica": 0.30, "south africa": 0.30,
    "egipto": 0.35,  "egypt": 0.35,
    "marruecos": 0.32, "morocco": 0.32,
    "tanzania": 0.40,
    "angola": 0.45,
    "mozambique": 0.42,
    # ── Oceania ───────────────────────────────────
    "australia": 0.10,
    "nueva zelanda": 0.10, "new zealand": 0.10,
}


def convert(filepath: str = DATACO_PATH, sample_n: int | None = None) -> pd.DataFrame:
    """
    Load DataCoSupplyChainDataset.csv and return a CEDPA-ready dataframe
    with exactly the 6 required feature columns.

    Parameters
    ----------
    filepath  : path to the CSV (relative to the app root or absolute)
    sample_n  : if set, return only a random sample of this many rows
                (useful for quick testing without retraining on all 180k rows)
    """
    df = pd.read_csv(filepath, encoding="latin-1")

    if sample_n is not None:
        df = df.sample(n=min(sample_n, len(df)), random_state=42).reset_index(drop=True)

    real_days  = pd.to_numeric(df["Days for shipping (real)"],      errors="coerce").fillna(3).clip(0, 10)
    sched_days = pd.to_numeric(df["Days for shipment (scheduled)"], errors="coerce").fillna(3).clip(0, 10)

    # ── 1. lead_time_variance ─────────────────────────────────────────
    # Squared deviation of real from scheduled shipping days.
    # Values: 0 (on time) → 36 (6-day late), clipped to [0.2, 9.0].
    deviation = (real_days - sched_days).clip(-3, 6)
    lead_time_variance = (deviation ** 2).clip(0.2, 9.0)

    # ── 2. supplier_reliability ───────────────────────────────────────
    # Ratio: scheduled / real.  On-time → 1.0; 2× late → 0.5.
    # Floored at 0.55 to keep within realistic supplier range.
    supplier_reliability = (sched_days / real_days.replace(0, 1)).clip(0.55, 1.0)

    # ── 3. geo_risk_index ─────────────────────────────────────────────
    # Map Order Country (Spanish / English) → risk score.
    # Unknown countries default to 0.25 (global average).
    geo_risk_index = (
        df["Order Country"]
        .astype(str).str.strip().str.lower()
        .map(DATACO_COUNTRY_RISK)
        .fillna(0.25)
    )

    # ── 4. inventory_buffer ───────────────────────────────────────────
    # Order Item Profit Ratio: raw range −2.75 → +0.50.
    # Min-max scaled to CEDPA range [−0.5, 0.8].
    profit = pd.to_numeric(df["Order Item Profit Ratio"], errors="coerce").fillna(0.10)
    RAW_MIN, RAW_MAX = -2.75, 0.50
    TGT_MIN, TGT_MAX = -0.50, 0.80
    profit_scaled = (
        (profit - RAW_MIN) / (RAW_MAX - RAW_MIN) * (TGT_MAX - TGT_MIN) + TGT_MIN
    )
    inventory_buffer = profit_scaled.clip(TGT_MIN, TGT_MAX)

    # ── 5. shipment_delay_history ─────────────────────────────────────
    # Days late (real − scheduled, floored at 0) scaled to 0–10.
    delay_pos = (real_days - sched_days).clip(lower=0)
    max_delay  = delay_pos.max()
    shipment_delay_history = (
        (delay_pos / max_delay * 10).clip(0, 10) if max_delay > 0
        else pd.Series(0.0, index=df.index)
    )

    # ── 6. disruption_risk ────────────────────────────────────────────
    # Late_delivery_risk is already a binary 0/1 label. No transformation.
    disruption_risk = (
        pd.to_numeric(df["Late_delivery_risk"], errors="coerce")
        .fillna(0).astype(int)
    )

    result = pd.DataFrame({
        "lead_time_variance":     lead_time_variance.values,
        "supplier_reliability":   supplier_reliability.values,
        "geo_risk_index":         geo_risk_index.values,
        "inventory_buffer":       inventory_buffer.values,
        "shipment_delay_history": shipment_delay_history.values,
        "disruption_risk":        disruption_risk.values,
    })

    return result.reset_index(drop=True)
