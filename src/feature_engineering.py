"""Feature engineering for time-series demand forecasting.
Guarantees strict leakage prevention using shifted lags and rolling windows.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger("feature_engineering")


def create_forecasting_features(
    df: pd.DataFrame,
    target_col: str = "demand",
    date_col: str = "date",
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Construct calendar, lag, and rolling features for time-series forecasting.
    
    All features are derived from historical values strictly prior to the target period,
    preventing any future data leakage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Contiguous time-series dataframe with date and target demand.
    target_col : str
        Name of demand column to forecast.
    date_col : str
        Name of date column.
    lags : list of int, optional
        Lags in periods. Defaults to [1, 2, 4, 8].
    rolling_windows : list of int, optional
        Windows for rolling statistics. Defaults to [4, 8].
    drop_na : bool
        Whether to drop rows with NaN values resulting from lag creation.
    """
    if lags is None:
        lags = [1, 2, 4, 8]
    if rolling_windows is None:
        rolling_windows = [4, 8]

    data = df.sort_values(date_col).copy()
    data[date_col] = pd.to_datetime(data[date_col])

    # 1. Calendar / Deterministic Features
    data["time_index"] = np.arange(len(data), dtype=int)
    data["week_of_year"] = data[date_col].dt.isocalendar().week.astype(int)
    data["month"] = data[date_col].dt.month.astype(int)
    data["quarter"] = data[date_col].dt.quarter.astype(int)

    # Cyclical seasonal encoding for annual weekly seasonality
    data["sin_week"] = np.sin(2 * np.pi * data["week_of_year"] / 52.18)
    data["cos_week"] = np.cos(2 * np.pi * data["week_of_year"] / 52.18)

    # 2. Lag Features (Target value shifted by k periods)
    # lag 1 = value at t-1
    for lag in lags:
        data[f"lag_{lag}"] = data[target_col].shift(lag)

    # 3. Rolling Features (Shifted by 1 to strictly exclude current period t)
    shifted_target = data[target_col].shift(1)
    for w in rolling_windows:
        data[f"rolling_mean_{w}"] = shifted_target.rolling(window=w, min_periods=max(2, w // 2)).mean()
        data[f"rolling_std_{w}"] = (
            shifted_target.rolling(window=w, min_periods=max(2, w // 2)).std().fillna(0.0)
        )

    if drop_na:
        data = data.dropna().reset_index(drop=True)

    return data


def build_recursive_feature_row(
    history_demand: List[float],
    next_date: pd.Timestamp,
    time_idx: int,
    lags: List[int],
    rolling_windows: List[int],
) -> pd.DataFrame:
    """Build a single row of features for multi-step recursive forecasting.
    
    Parameters
    ----------
    history_demand : list of float
        Observed or recursively predicted demand values up to step t-1.
    next_date : pd.Timestamp
        Target date for prediction step t.
    time_idx : int
        Deterministic time index for step t.
    lags : list of int
        Lag values matching the training features.
    rolling_windows : list of int
        Rolling window sizes matching the training features.
    """
    row: dict = {
        "time_index": time_idx,
        "week_of_year": int(next_date.isocalendar().week),
        "month": int(next_date.month),
        "quarter": int(next_date.quarter),
        "sin_week": float(np.sin(2 * np.pi * next_date.isocalendar().week / 52.18)),
        "cos_week": float(np.cos(2 * np.pi * next_date.isocalendar().week / 52.18)),
    }

    # Extract historical lags
    n_hist = len(history_demand)
    for lag in lags:
        if n_hist >= lag:
            row[f"lag_{lag}"] = float(history_demand[-lag])
        else:
            row[f"lag_{lag}"] = float(history_demand[0]) if n_hist > 0 else 0.0

    # Extract rolling statistics from history
    for w in rolling_windows:
        slice_vals = history_demand[-w:] if n_hist >= w else history_demand
        if len(slice_vals) > 0:
            row[f"rolling_mean_{w}"] = float(np.mean(slice_vals))
            row[f"rolling_std_{w}"] = float(np.std(slice_vals, ddof=1)) if len(slice_vals) > 1 else 0.0
        else:
            row[f"rolling_mean_{w}"] = 0.0
            row[f"rolling_std_{w}"] = 0.0

    return pd.DataFrame([row])
