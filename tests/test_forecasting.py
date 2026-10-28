"""Unit tests for forecasting models: Naive baseline and Random Forest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting import DemandForecaster


@pytest.fixture
def sample_series():
    # 30 weeks of synthetic demand
    np.random.seed(42)
    dates = pd.date_range("2022-01-03", periods=30, freq="W-MON")
    demand = 50.0 + 10.0 * np.sin(np.linspace(0, 3.14 * 2, 30)) + np.random.normal(0, 3, 30)
    return pd.DataFrame({"date": dates, "demand": np.clip(demand, 0, None)})


def test_naive_forecaster(sample_series):
    forecaster = DemandForecaster(model_type="naive")
    forecaster.fit(sample_series)

    forecast_df = forecaster.predict_future(sample_series, horizon=4)
    assert len(forecast_df) == 4
    assert "predicted_demand" in forecast_df.columns
    assert "lower_bound" in forecast_df.columns
    assert "upper_bound" in forecast_df.columns

    # Naive repeats the last observed value
    last_val = sample_series["demand"].iloc[-1]
    assert np.allclose(forecast_df["predicted_demand"], last_val, atol=0.01)

    # Future dates must strictly follow historical dates
    assert forecast_df["forecast_date"].iloc[0] > sample_series["date"].iloc[-1]


def test_random_forest_forecaster(sample_series):
    forecaster = DemandForecaster(model_type="random_forest", lags=[1, 2, 4], rolling_windows=[4])
    forecaster.fit(sample_series)

    forecast_df = forecaster.predict_future(sample_series, horizon=6)
    assert len(forecast_df) == 6
    # Demand non-negativity constraint
    assert (forecast_df["predicted_demand"] >= 0).all()
    assert (forecast_df["lower_bound"] >= 0).all()

    # Dates are strictly in the future
    assert forecast_df["forecast_date"].iloc[0] > sample_series["date"].iloc[-1]


def test_holdout_evaluation(sample_series):
    forecaster = DemandForecaster(model_type="random_forest", lags=[1, 2, 4], rolling_windows=[4])
    metrics, holdout_df = forecaster.evaluate_holdout(sample_series, test_size=0.2)

    assert metrics["MAE"] is not None
    assert metrics["RMSE"] is not None
    assert metrics["count"] > 0
    assert len(holdout_df) == metrics["count"]


def test_insufficient_history_rejection():
    short_df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=5, freq="W-MON"),
        "demand": [10, 20, 15, 25, 30],
    })
    forecaster = DemandForecaster(model_type="random_forest", lags=[1, 2, 4, 8])
    with pytest.raises(ValueError, match="Insufficient training data"):
        forecaster.fit(short_df)
