import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from lake_extractor import extract_lake_area_km2
from time_series_features import build_timeseries
from risk_model import compute_risk_scores, explain_risk
from forecast import forecast_area

FEATURE_COLS = [
    "lake_area_km2", "lake_area_growth_rate",
    "ndwi_max", "ndsi_mean", "slope_mean_deg"
]

def run_pipeline(data_dir: str, dem_path: str,
                 years: list, forecast_years: int = 3):
    os.makedirs("output", exist_ok=True)

    print("\n[1/4] Extracting lake features from time-series imagery...")
    df = build_timeseries(data_dir, dem_path, years)
    df.to_csv("output/timeseries_features.csv", index=False)
    print(f"      Saved {len(df)} years of features.")

    print("\n[2/4] Computing risk scores...")
    df_risk = compute_risk_scores(df)
    df_risk["explanation"] = df_risk.apply(explain_risk, axis=1)
    df_risk.to_csv("output/risk_scores.csv", index=False)

    print("\n  Historical Risk Summary:")
    print(df_risk[["year", "lake_area_km2", "risk_score", "risk_level", "explanation"]]
          .to_string(index=False))

    print(f"\n[3/4] Forecasting lake area for next {forecast_years} years...")
    forecast_df = forecast_area(df, FEATURE_COLS, n_years=forecast_years,
                                critical_area_km2=1.67)
    forecast_df.to_csv("output/forecast.csv", index=False)

    print("\n[4/4] Generating visualisations...")
    _plot_timeseries(df_risk, forecast_df)
    print("\n[DONE] Pipeline complete. Outputs saved to output/")


def _plot_timeseries(df_risk: pd.DataFrame, forecast_df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    fig.suptitle("South Lhonak Lake — GLOF Risk Analysis (2016-2027)",
                 fontsize=15, fontweight="bold")

    # ── Panel 1: Lake area trend + forecast ──────────────────────
    ax1.plot(df_risk["year"], df_risk["lake_area_km2"],
             "o-", color="#2196F3", linewidth=2.5,
             markersize=7, label="Observed lake area")

    if len(forecast_df):
        ax1.plot(forecast_df["year"], forecast_df["predicted_area_km2"],
                 "s--", color="#FF5722", linewidth=2.5,
                 markersize=7, label="LSTM Forecast")
        ax1.fill_between(
            forecast_df["year"],
            forecast_df["predicted_area_km2"] * 0.88,
            forecast_df["predicted_area_km2"] * 1.12,
            alpha=0.25, color="#FF5722", label="+-12% uncertainty band"
        )
        for _, row in forecast_df.iterrows():
            ax1.annotate(f"{row['predicted_area_km2']:.2f} km2",
                         (row["year"], row["predicted_area_km2"]),
                         textcoords="offset points", xytext=(0, 10),
                         fontsize=8, color="#FF5722", ha="center")

    ax1.axhline(1.67, color="red", linestyle=":", linewidth=1.5,
                label="Critical threshold (1.67 km2)")
    ax1.axvline(2023, color="black", linestyle="--",
                linewidth=1, alpha=0.5, label="Actual burst (Oct 2023)")
    ax1.set_ylabel("Lake Area (km2)", fontsize=11)
    ax1.set_title("Lake Area Trend & 3-Year LSTM Forecast", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Risk score bar chart ────────────────────────────
    risk_scores = df_risk["risk_score"].fillna(0)
    risk_levels = df_risk["risk_level"].astype(str)

    color_map = {
        "LOW":    "#4CAF50",
        "MEDIUM": "#FF9800",
        "HIGH":   "#F44336",
        "nan":    "#AAAAAA"
    }
    bar_colors = [color_map.get(str(lvl), "#AAAAAA") for lvl in risk_levels]

    ax2.bar(df_risk["year"], risk_scores,
            color=bar_colors, edgecolor="white",
            linewidth=0.8, width=0.6)

    for x, y, lvl in zip(df_risk["year"], risk_scores, risk_levels):
        ax2.text(x, float(y) + 0.01, f"{float(y):.2f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.axhline(0.70, color="#F44336", linestyle="--",
                alpha=0.7, linewidth=1.2, label="High threshold (0.70)")
    ax2.axhline(0.40, color="#FF9800", linestyle="--",
                alpha=0.7, linewidth=1.2, label="Medium threshold (0.40)")
    ax2.axvline(2023, color="black", linestyle="--",
                linewidth=1, alpha=0.5)
    ax2.set_ylim(0, 1.10)
    ax2.set_ylabel("Composite Risk Score", fontsize=11)
    ax2.set_xlabel("Year", fontsize=11)
    ax2.set_title("Annual GLOF Risk Score (0 = Safe, 1 = Critical)", fontsize=12)

    patches = [
        mpatches.Patch(color="#4CAF50", label="LOW  (< 0.40)"),
        mpatches.Patch(color="#FF9800", label="MEDIUM (0.40-0.70)"),
        mpatches.Patch(color="#F44336", label="HIGH  (> 0.70)"),
        mpatches.Patch(color="#AAAAAA", label="No data"),
    ]
    ax2.legend(handles=patches, fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    all_years = list(df_risk["year"]) + list(forecast_df["year"])
    ax2.set_xticks(all_years)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/glof_risk_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved -> output/glof_risk_analysis.png")


if __name__ == "__main__":
    run_pipeline(
        data_dir       = r"D:\Mittu\GLOF_Upgradation\data\landsat",
        dem_path       = r"D:\Mittu\GLOF_Upgradation\data\STRM_DEM.tif",
        years          = list(range(2016, 2025)),
        forecast_years = 3,
    )