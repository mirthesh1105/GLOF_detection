"""
============================================================
GLOF Risk Detection & Forecasting System
Author : Mirthesh M | Kings Engineering College | Batch 2027
============================================================
"""

# stack_bands.py — run this once per year to prepare your data
import rasterio
import numpy as np
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

def stack_bands(band_paths: list, output_path: str):
    """Stack individual Landsat band TIFs into one multi-band file."""
    bands = []
    with rasterio.open(band_paths[0]) as ref:
        meta = ref.meta.copy()
        meta.update(count=len(band_paths))

    for path in band_paths:
        with rasterio.open(path) as src:
            bands.append(src.read(1))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, 'w', **meta) as dst:
        for i, band in enumerate(bands, start=1):
            dst.write(band, i)
    print(f"[OK] Saved {output_path}")

# ── Edit these paths to match your downloaded Landsat filenames ──
YEARS = {
    2016: [
        r"D:\Mittu\GLOF_Upgradation\raw\2016\LC08_L1TP_139041_20160129_20200907_02_T1_B3.TIF",  # Green
        r"D:\Mittu\GLOF_Upgradation\raw\2016\LC08_L1TP_139041_20160129_20200907_02_T1_B4.TIF",  # Red
        r"D:\Mittu\GLOF_Upgradation\raw\2016\LC08_L1TP_139041_20160129_20200907_02_T1_B5.TIF",  # NIR
        r"D:\Mittu\GLOF_Upgradation\raw\2016\LC08_L1TP_139041_20160129_20200907_02_T1_B6.TIF",  # SWIR1
    ],
    2017: [
        r"D:\Mittu\GLOF_Upgradation\raw\2017\LC08_L1TP_139041_20170115_20200905_02_T1_B3 (1).TIF",  # Green
        r"D:\Mittu\GLOF_Upgradation\raw\2017\LC08_L1TP_139041_20170115_20200905_02_T1_B4 (1).TIF",  # Red
        r"D:\Mittu\GLOF_Upgradation\raw\2017\LC08_L1TP_139041_20170115_20200905_02_T1_B5 (1).TIF",  # NIR
        r"D:\Mittu\GLOF_Upgradation\raw\2017\LC08_L1TP_139041_20170115_20200905_02_T1_B6 (1).TIF",  # SWIR1
    ],
    2018: [
        r"D:\Mittu\GLOF_Upgradation\raw\2018\LC08_L1TP_139041_20180118_20200902_02_T1_B3.TIF",  # Green
        r"D:\Mittu\GLOF_Upgradation\raw\2018\LC08_L1TP_139041_20180118_20200902_02_T1_B4.TIF",  # Red
        r"D:\Mittu\GLOF_Upgradation\raw\2018\LC08_L1TP_139041_20180118_20200902_02_T1_B5.TIF",  # NIR
        r"D:\Mittu\GLOF_Upgradation\raw\2018\LC08_L1TP_139041_20180118_20200902_02_T1_B6.TIF",  # SWIR1
    ],
    2019: [
        r"D:\Mittu\GLOF_Upgradation\raw\2019\LC08_L1TP_139041_20190121_20200830_02_T1_B3.TIF",  # Green
        r"D:\Mittu\GLOF_Upgradation\raw\2019\LC08_L1TP_139041_20190121_20200830_02_T1_B4.TIF",  # Red
        r"D:\Mittu\GLOF_Upgradation\raw\2019\LC08_L1TP_139041_20190121_20200830_02_T1_B5.TIF",  # NIR
        r"D:\Mittu\GLOF_Upgradation\raw\2019\LC08_L1TP_139041_20190121_20200830_02_T1_B6.TIF",  # SWIR1
    ],
    2020: [
        r"D:\Mittu\GLOF_Upgradation\raw\2020\LC08_L1TP_139041_20200124_20200823_02_T1_B3.TIF",  # Green
        r"D:\Mittu\GLOF_Upgradation\raw\2020\LC08_L1TP_139041_20200124_20200823_02_T1_B4.TIF",  # Red
        r"D:\Mittu\GLOF_Upgradation\raw\2020\LC08_L1TP_139041_20200124_20200823_02_T1_B5.TIF",  # NIR
        r"D:\Mittu\GLOF_Upgradation\raw\2020\LC08_L1TP_139041_20200124_20200823_02_T1_B6.TIF",  # SWIR1
    ],
    2021: [
        r"D:\Mittu\GLOF_Upgradation\raw\2021\LC08_L1TP_139041_20210126_20210305_02_T1_B3.TIF",  # Green
        r"D:\Mittu\GLOF_Upgradation\raw\2021\LC08_L1TP_139041_20210126_20210305_02_T1_B4.TIF",  # Red
        r"D:\Mittu\GLOF_Upgradation\raw\2021\LC08_L1TP_139041_20210126_20210305_02_T1_B5.TIF",  # NIR
        r"D:\Mittu\GLOF_Upgradation\raw\2021\LC08_L1TP_139041_20210126_20210305_02_T1_B6.TIF",  # SWIR1
    ],
    2022: [
        r"D:\Mittu\GLOF_Upgradation\raw\2022\LC08_L1TP_139041_20220129_20220204_02_T1_B3.TIF",  # Green
        r"D:\Mittu\GLOF_Upgradation\raw\2022\LC08_L1TP_139041_20220129_20220204_02_T1_B4.TIF",  # Red
        r"D:\Mittu\GLOF_Upgradation\raw\2022\LC08_L1TP_139041_20220129_20220204_02_T1_B5.TIF",  # NIR
        r"D:\Mittu\GLOF_Upgradation\raw\2022\LC08_L1TP_139041_20220129_20220204_02_T1_B6.TIF",  # SWIR1
    ],
    2023: [
        r"D:\Mittu\GLOF_Upgradation\raw\2023\LC09_L1TP_139041_20230124_20230312_02_T1_B3.TIF",  # Green
        r"D:\Mittu\GLOF_Upgradation\raw\2023\LC09_L1TP_139041_20230124_20230312_02_T1_B4.TIF",  # Red
        r"D:\Mittu\GLOF_Upgradation\raw\2023\LC09_L1TP_139041_20230124_20230312_02_T1_B5.TIF",  # NIR
        r"D:\Mittu\GLOF_Upgradation\raw\2023\LC09_L1TP_139041_20230124_20230312_02_T1_B6.TIF",  # SWIR1
    ],
    2024: [
        r"D:\Mittu\GLOF_Upgradation\raw\2024\LC08_L1TP_139041_20240119_20240129_02_T1_B3.TIF",  # Green
        r"D:\Mittu\GLOF_Upgradation\raw\2024\LC08_L1TP_139041_20240119_20240129_02_T1_B4.TIF",  # Red
        r"D:\Mittu\GLOF_Upgradation\raw\2024\LC08_L1TP_139041_20240119_20240129_02_T1_B5.TIF",  # NIR
        r"D:\Mittu\GLOF_Upgradation\raw\2024\LC08_L1TP_139041_20240119_20240129_02_T1_B6.TIF",  # SWIR1
    ]
}

for year, paths in YEARS.items():
    stack_bands(paths, f"data/landsat/landsat_{year}.tif")