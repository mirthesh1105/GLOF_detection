"""
============================================================
GLOF Risk Detection & Forecasting System
Author : Mirthesh M | Kings Engineering College | Batch 2027
Domain : Remote Sensing | Geospatial AI | Climate Risk Analysis
============================================================
"""

"""
Extracts glacier lake boundary and area from a Landsat GeoTIFF.

Why NDWI + morphological cleaning instead of raw threshold?
Because a bare threshold picks up cloud shadows and wet rock.
Opening removes small false positives; closing fills interior holes.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_opening, binary_closing, label


def load_bands(tif_path: str) -> dict:
    """
    Load bands from a stacked GeoTIFF created by stack_bands.py.
    
    Our stack order (set in stack_bands.py) is:
      Band 1 → Green  (original Landsat B3)
      Band 2 → Red    (original Landsat B4)
      Band 3 → NIR    (original Landsat B5)
      Band 4 → SWIR1  (original Landsat B6)
    """
    with rasterio.open(tif_path) as src:
        print(f"    [DEBUG] Bands in file: {src.count}")  # shows how many bands exist
        meta = {
            "transform": src.transform,
            "crs":       src.crs,
            "height":    src.height,
            "width":     src.width
        }
        bands = {
            "green": src.read(1).astype(np.float32),
            "red":   src.read(2).astype(np.float32),
            "nir":   src.read(3).astype(np.float32),
            "swir1": src.read(4).astype(np.float32),
        }
    return bands, meta

def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    NDWI (McFeeters 1996): (Green - NIR) / (Green + NIR)
    Positive values indicate open water.
    """
    eps = 1e-8
    return (green - nir) / (green + nir + eps)


def compute_ndsi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    NDSI: (Green - SWIR1) / (Green + SWIR1)
    Values > 0.4 indicate snow/ice. Used as glacier melt proxy.
    """
    eps = 1e-8
    return (green - swir1) / (green + swir1 + eps)


def extract_lake_mask(ndwi: np.ndarray,
                      threshold: float = 0.25) -> np.ndarray:
    """
    Stricter threshold (0.25 vs 0.15) to exclude snow/wetland.
    Then keeps only the single largest water body (the lake itself).
    """
    raw_mask = ndwi > threshold
    cleaned  = binary_opening(raw_mask, structure=np.ones((3, 3)))
    cleaned  = binary_closing(cleaned,  structure=np.ones((5, 5)))

    labeled, n = label(cleaned)
    if n == 0:
        return np.zeros_like(raw_mask, dtype=bool)

    # Keep ONLY the largest connected water body
    sizes      = [(labeled == i).sum() for i in range(1, n + 1)]
    lake_label = np.argmax(sizes) + 1
    return labeled == lake_label


def pixel_area_m2(transform) -> float:
    """
    Compute area of a single pixel in m² using the affine transform.
    For Landsat 8 at 30m resolution this returns 900 m².
    """
    return abs(transform.a * transform.e)   # pixel_width × pixel_height


def extract_lake_area_km2(tif_path: str,
                           threshold: float = 0.15) -> dict:
    """
    Full extraction pipeline for one year's GeoTIFF.

    Returns a dict with:
      - lake_area_km2   : float
      - ndwi_max        : float (peak water intensity)
      - ndsi_mean       : float (mean ice index over lake region)
      - lake_mask       : np.ndarray (bool, for visualisation)
    """
    bands, meta = load_bands(tif_path)
    ndwi        = compute_ndwi(bands["green"], bands["nir"])
    ndsi        = compute_ndsi(bands["green"], bands["swir1"])
    lake_mask   = extract_lake_mask(ndwi, threshold)

    px_area_m2  = pixel_area_m2(meta["transform"])
    lake_pixels = lake_mask.sum()
    area_km2    = (lake_pixels * px_area_m2) / 1e6   # m² → km²

    ndwi_max    = float(ndwi[lake_mask].max()) if lake_pixels > 0 else 0.0
    ndsi_mean   = float(ndsi[lake_mask].mean()) if lake_pixels > 0 else 0.0

    return {
        "lake_area_km2": round(area_km2, 4),
        "ndwi_max":      round(ndwi_max, 4),
        "ndsi_mean":     round(ndsi_mean, 4),
        "lake_mask":     lake_mask,
        "meta":          meta,
    }

