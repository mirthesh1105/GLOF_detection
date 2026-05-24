"""
Forecasts future lake area using LSTM (if ≥6 data points exist)
or linear extrapolation (fallback for short series).

IMPORTANT HONESTY NOTE:
Predicting an exact outburst date is not scientifically possible
from satellite imagery alone. What we CAN estimate:
  - Whether the lake is on a growth trajectory (trend)
  - The approximate year the lake area may exceed a critical threshold
  - A risk window (e.g., "HIGH risk expected between 2025-2027")

This is what real GLOF early warning systems do (e.g., ICIMOD, NDMA).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


class LakeLSTM(nn.Module):
    """
    LSTM that learns the temporal pattern of lake area growth.
    Input:  (batch, seq_len, n_features)
    Output: (batch, 1) — predicted lake area for next timestep

    Architecture kept small intentionally: we typically have only
    8–15 years of Landsat data, so a large model would overfit.
    """
    def __init__(self, n_features: int = 5, hidden: int = 32, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def prepare_sequences(df: pd.DataFrame,
                       feature_cols: list[str],
                       target_col: str = "lake_area_km2",
                       seq_len: int = 3):
    """
    Create overlapping sequences for LSTM training.

    seq_len=3 means: given 3 consecutive years of features,
    predict the lake area in year 4.
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(df[feature_cols])
    y_scaled = scaler_y.fit_transform(df[[target_col]])

    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i + seq_len])
        y_seq.append(y_scaled[i + seq_len])

    return (np.array(X_seq), np.array(y_seq),
            scaler_X, scaler_y)


def train_lstm(X: np.ndarray, y: np.ndarray,
               n_features: int, epochs: int = 300) -> LakeLSTM:
    """Train the LSTM on historical sequences."""
    model   = LakeLSTM(n_features=n_features)
    opt     = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model.train()
    for epoch in range(epochs):
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1:3d} | Loss: {loss.item():.6f}")

    return model


def forecast_area(df: pd.DataFrame,
                  feature_cols: list[str],
                  n_years: int = 3,
                  critical_area_km2: float = None) -> pd.DataFrame:
    """
    Forecast lake area for the next n_years.

    Uses LSTM if ≥6 historical data points exist.
    Falls back to linear extrapolation otherwise.

    critical_area_km2: If provided, estimates the year when the lake
    will exceed this size (a proxy for outburst risk threshold).
    Historical South Lhonak Lake grew from ~1.5 km² (2000) to
    ~167 ha (≈1.67 km²) by 2023 before it burst — use that as reference.
    """
    if len(df) < 6:
        # Linear fallback: fit a line through lake_area_km2 vs year
        print("  [INFO] Fewer than 6 years of data. Using linear trend.")
        coeffs   = np.polyfit(df["year"], df["lake_area_km2"], 1)
        future_y = [df["year"].max() + i + 1 for i in range(n_years)]
        forecast_areas = np.polyval(coeffs, future_y)
        return pd.DataFrame({"year": future_y,
                              "predicted_area_km2": forecast_areas.round(4),
                              "method": "linear"})

    X, y, scX, scY = prepare_sequences(df, feature_cols)
    model = train_lstm(X, y, n_features=len(feature_cols))

    # Autoregressive rollout: use last seq_len years → predict year+1
    # then feed prediction back in for year+2, etc.
    SEQ_LEN = 3
    results = []
    last_seq = scX.transform(df[feature_cols].values)[-SEQ_LEN:]  # (3, F)

    model.eval()
    with torch.no_grad():
        for i in range(n_years):
            x_in = torch.tensor(last_seq[None], dtype=torch.float32)
            pred_scaled = model(x_in).item()
            pred_area   = float(scY.inverse_transform([[pred_scaled]])[0][0])
            future_year = int(df["year"].max()) + i + 1
            results.append({"year": future_year,
                             "predicted_area_km2": round(pred_area, 4),
                             "method": "lstm"})

            # Update rolling window: drop oldest, append new prediction
            # For simplicity, hold non-area features constant at last known values
            new_row = last_seq[-1].copy()
            new_row[feature_cols.index("lake_area_km2")] = scX.transform(
                [[pred_area if c == "lake_area_km2" else 0 for c in feature_cols]]
            )[0][feature_cols.index("lake_area_km2")]
            last_seq = np.vstack([last_seq[1:], new_row])

    forecast_df = pd.DataFrame(results)

    # Estimate outburst risk window
    if critical_area_km2:
        breach_rows = forecast_df[
            forecast_df["predicted_area_km2"] >= critical_area_km2
        ]
        if len(breach_rows):
            breach_year = int(breach_rows["year"].min())
            print(f"\n  ⚠️  Lake projected to exceed {critical_area_km2} km² "
                  f"by {breach_year}. ELEVATED OUTBURST RISK WINDOW.")
        else:
            print(f"\n  ✅ Lake not projected to exceed {critical_area_km2} km² "
                  f"in {n_years}-year forecast window.")

    return forecast_df
