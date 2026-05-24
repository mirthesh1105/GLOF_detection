"""
============================================================
GLOF Risk Detection & Forecasting System
Author : Mirthesh M | Kings Engineering College | Batch 2027
Domain : Remote Sensing | Geospatial AI | Climate Risk Analysis
============================================================
"""

"""
Central config. Change study area, paths, and thresholds here.
South Lhonak Lake coordinates used as default (the lake that
actually burst in October 2023, killing 14 people in Sikkim).
"""

STUDY_AREA = {
    "name": "South Lhonak Lake",
    "lon_min": 88.180596,
    "lat_min": 27.904521,
    "lon_max": 88.210465,
    "lat_max": 27.921510,
}

# Years to build the historical time-series from
YEARS = list(range(2016, 2025))

# Landsat band indices (1-based, Landsat 8 TOA)
BANDS = {"blue": 2, "green": 3, "red": 4, "nir": 5, "swir1": 6, "swir2": 7}

# NDWI threshold for water pixel classification
NDWI_WATER_THRESHOLD = 0.15

# Moraine proximity danger zone (metres)
MORAINE_BUFFER_M = 500

# Risk thresholds (tune with domain knowledge)
RISK_HIGH   = 0.70
RISK_MEDIUM = 0.40

FEATURES = [
    "lake_area_km2",
    "lake_area_growth_rate",      # km² / year
    "ndsi_mean",                  # ice cover proxy
    "ndwi_max",                   # water intensity
    "slope_mean_deg",
    "dist_to_moraine_m",
    "lst_anomaly_c",              # land surface temp anomaly
]

