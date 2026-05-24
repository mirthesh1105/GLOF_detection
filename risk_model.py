"""
============================================================
GLOF Risk Detection & Forecasting System
Author : Mirthesh M | Kings Engineering College | Batch 2027
============================================================
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

RISK_WEIGHTS = {
    "lake_area_growth_rate": 0.40,   # primary driver
    "slope_mean_deg":        0.20,   # terrain instability
    "ndsi_mean":             0.20,   # ice melt proxy
    "ndwi_max":              0.20,   # water intensity
    # dist_to_moraine_m and lst_anomaly_c excluded:
    # dist is 0 for all years (flat DEM result) = useless
    # lst is empty = no data collected
}

def _safe_minmax(series: pd.Series) -> pd.Series:
    """Scale to [0,1]. If all values identical, return 0.5 neutral."""
    series = pd.to_numeric(series, errors='coerce').fillna(0.0)
    lo, hi = series.min(), series.max()
    if (hi - lo) < 1e-8:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - lo) / (hi - lo)

def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Force all feature columns to numeric, fill blanks with 0
    for col in ["lake_area_growth_rate", "slope_mean_deg",
                "ndsi_mean", "ndwi_max"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    scaled = pd.DataFrame(index=df.index)
    scaled["lake_area_growth_rate"] = _safe_minmax(df["lake_area_growth_rate"])
    scaled["slope_mean_deg"]        = _safe_minmax(df["slope_mean_deg"])
    # Invert NDSI: less ice = more melt = higher risk
    scaled["ndsi_mean"]             = 1.0 - _safe_minmax(df["ndsi_mean"])
    scaled["ndwi_max"]              = _safe_minmax(df["ndwi_max"])

    score = sum(scaled[f] * w for f, w in RISK_WEIGHTS.items())
    df["risk_score"] = score.round(4)

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-0.001, 0.40, 0.70, 1.001],
        labels=["LOW", "MEDIUM", "HIGH"]
    )
    return df

def explain_risk(row: pd.Series) -> str:
    parts = []
    growth = pd.to_numeric(row.get("lake_area_growth_rate", 0), errors='coerce') or 0
    ndsi   = pd.to_numeric(row.get("ndsi_mean", 1),             errors='coerce') or 1
    ndwi   = pd.to_numeric(row.get("ndwi_max", 0),              errors='coerce') or 0
    score  = pd.to_numeric(row.get("risk_score", 0),            errors='coerce') or 0

    if growth > 0.5:
        parts.append(f"rapid lake expansion (+{growth:.2f} km2/yr)")
    elif growth < -0.5:
        parts.append(f"lake shrinking ({growth:.2f} km2/yr) - post-burst drainage")
    if ndsi < 0.35:
        parts.append("significant ice melt detected (low NDSI)")
    if ndwi > 0.35:
        parts.append(f"high water intensity (NDWI={ndwi:.2f})")
    if score > 0.70:
        parts.append("CRITICAL: multiple high-risk factors converging")
    if not parts:
        parts.append("moderate baseline conditions")
    return "; ".join(parts)