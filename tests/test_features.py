"""Unit tests for feature engineering and leakage prevention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import build_recursive_feature_row, create_forecasting_features


def test_create_forecasting_features_leakage():
    # 10 weeks of deterministic linear demand: 10, 20, 30, ...
    dates = pd.date_range("2022-01-03", periods=10, freq="W-MON")
    df = pd.DataFrame({"date": dates, "demand": [10.0 * (i + 1) for i in range(10)]})

    featured = create_forecasting_features(df, lags=[1, 2], rolling_windows=[2], drop_na=False)

    # Check lag 1 at index 3 (demand was 40.0 at t=3, t-1 demand was 30.0)
    assert featured.loc[3, "lag_1"] == 30.0
    assert featured.loc[3, "lag_2"] == 20.0

    # Rolling window of size 2 prior to t=3 (values at t-1 and t-2 are 30 and 20 -> mean is 25)
    assert featured.loc[3, "rolling_mean_2"] == 25.0

    # Ensure index 0 has NaN for lag_1 because no prior history exists (leakage prevention)
    assert pd.isna(featured.loc[0, "lag_1"])


def test_build_recursive_feature_row():
    history = [10.0, 20.0, 30.0, 40.0]
    next_date = pd.Timestamp("2022-02-07")
    row = build_recursive_feature_row(
        history_demand=history,
        next_date=next_date,
        time_idx=4,
        lags=[1, 2],
        rolling_windows=[2],
    )

    assert row.loc[0, "time_index"] == 4
    assert row.loc[0, "lag_1"] == 40.0
    assert row.loc[0, "lag_2"] == 30.0
    assert row.loc[0, "rolling_mean_2"] == 35.0  # mean of [30, 40]
