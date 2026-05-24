"""
============================================================
GLOF Risk Detection & Forecasting System
Author : Mirthesh M | Kings Engineering College | Batch 2027
============================================================
"""

"""
Builds a yearly time-series DataFrame of GLOF risk features.

Each row = one year. The ML models (risk scorer + LSTM) are
trained on this multi-year feature table.

Why year-level features instead of pixel-level?
Because GLOF hazard is a lake-level phenomenon (is this lake
going to burst?) not a pixel-level one. Pixel classification
tells you WHERE water is. Year-level features tell you if it's
growing dangerously.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.ndimage import sobel, binary_dilation, distance_transform_edt

from lake_extractor import extract_lake_area_km2


def compute_slope_stats(dem_path: str, lake_mask: np.ndarray) -> dict:
    """
    Compute slope stats around the lake boundary.
    Resizes lake_mask to match DEM dimensions if they differ.
    """
    import rasterio
    from scipy.ndimage import sobel, binary_dilation, distance_transform_edt
    from skimage.transform import resize

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        pixel_size_m = abs(src.transform.a)

    # ── Resize lake_mask to match DEM shape ──────────────────────
    # Landsat is 30m, SRTM is ~90m, so they have different pixel counts.
    # We resize the boolean mask to match the DEM before any spatial ops.
    if lake_mask.shape != dem.shape:
        lake_mask_resized = resize(
            lake_mask.astype(np.float32),
            dem.shape,
            order=0,               # nearest-neighbour — preserves binary values
            anti_aliasing=False
        ) > 0.5                    # back to boolean
    else:
        lake_mask_resized = lake_mask

    # ── Compute slope in degrees ──────────────────────────────────
    sx = sobel(dem, axis=1) / (8 * pixel_size_m)
    sy = sobel(dem, axis=0) / (8 * pixel_size_m)
    slope_deg = np.degrees(np.arctan(np.hypot(sx, sy)))

    # ── Surrounding terrain (3-pixel buffer around lake) ─────────
    surrounding = binary_dilation(lake_mask_resized, iterations=3) & ~lake_mask_resized

    if surrounding.sum() == 0:
        return {"slope_mean_deg": 0.0, "dist_to_moraine_m": 0.0}

    slope_mean = float(slope_deg[surrounding].mean())

    # ── Distance to moraine (slope > 25°) ────────────────────────
    moraine_mask = slope_deg > 25.0
    if moraine_mask.any() and lake_mask_resized.any():
        dist_px = distance_transform_edt(~moraine_mask)
        dist_m  = float(dist_px[lake_mask_resized].min()) * pixel_size_m
    else:
        dist_m = 9999.0

    return {
        "slope_mean_deg":    round(slope_mean, 2),
        "dist_to_moraine_m": round(dist_m, 1)
    }


def build_timeseries(data_dir: str, dem_path: str,
                     years: list[int]) -> pd.DataFrame:
    """
    Build the year-level feature DataFrame.

    Expected directory structure:
      data_dir/
        landsat_2016.tif
        landsat_2017.tif
        ...
        landsat_2024.tif

    Returns a DataFrame with one row per year and columns:
      year, lake_area_km2, lake_area_growth_rate,
      ndwi_max, ndsi_mean, slope_mean_deg,
      dist_to_moraine_m, lst_anomaly_c
    """
    records = []
    prev_area = None

    for year in sorted(years):
        tif = Path(data_dir) / f"landsat_{year}.tif"
        if not tif.exists():
            print(f"  [WARN] Missing: {tif}, skipping year {year}")
            continue

        print(f"  Processing {year}...", end=" ")
        extracted = extract_lake_area_km2(str(tif))
        terrain   = compute_slope_stats(dem_path, extracted["lake_mask"])

        # Growth rate: how much the lake grew vs previous year
        # First year gets NaN (no prior reference)
        growth = (extracted["lake_area_km2"] - prev_area) if prev_area else np.nan
        prev_area = extracted["lake_area_km2"]

        # LST anomaly: placeholder — replace with MODIS MOD11A1 data
        # or ERA5 2m temperature for proper thermal analysis.
        # A rising anomaly (>0) indicates accelerating glacier melt.
        lst_anomaly = np.nan   # fill from external source

        records.append({
            "year":                year,
            "lake_area_km2":       extracted["lake_area_km2"],
            "lake_area_growth_rate": round(growth, 4) if not np.isnan(growth) else np.nan,
            "ndwi_max":            extracted["ndwi_max"],
            "ndsi_mean":           extracted["ndsi_mean"],
            "slope_mean_deg":      terrain["slope_mean_deg"],
            "dist_to_moraine_m":   terrain["dist_to_moraine_m"],
            "lst_anomaly_c":       lst_anomaly,
        })
        print("✓")

    df = pd.DataFrame(records)
    df["lake_area_growth_rate"] = df["lake_area_growth_rate"].bfill()
    df["lst_anomaly_c"] = df["lst_anomaly_c"].fillna(0.0)   # conservative default
    return df
