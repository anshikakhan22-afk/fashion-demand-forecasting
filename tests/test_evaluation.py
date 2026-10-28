"""Unit tests for evaluation metrics and chronological validation splitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import compute_metrics, split_chronological


def test_compute_metrics_standard():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 33.0])

    metrics = compute_metrics(actual, predicted)
    assert metrics["MAE"] == pytest.approx(2.33, rel=1e-2)
    assert metrics["RMSE"] == pytest.approx(2.38, rel=1e-2)
    # WAPE = (|2| + |2| + |3|) / (10 + 20 + 30) = 7 / 60 * 100 = 11.67%
    assert metrics["WAPE"] == pytest.approx(11.67, rel=1e-2)


def test_compute_metrics_zero_actual():
    actual = np.array([0.0, 0.0, 0.0])
    predicted = np.array([1.0, 2.0, 1.0])

    metrics = compute_metrics(actual, predicted)
    assert metrics["MAE"] == pytest.approx(1.33, rel=1e-2)
    # WAPE and MAPE handle denominator 0 safely
    assert metrics["WAPE"] is None
    assert metrics["MAPE"] is None


def test_compute_metrics_empty():
    metrics = compute_metrics([], [])
    assert metrics["MAE"] is None
    assert metrics["RMSE"] is None
    assert metrics["count"] == 0


def test_split_chronological():
    dates = pd.date_range("2022-01-01", periods=20, freq="W-MON")
    df = pd.DataFrame({"date": dates, "demand": range(20)})

    train, test = split_chronological(df, test_size=0.25, min_train_periods=10)
    assert len(train) == 15
    assert len(test) == 5
    assert train["date"].max() < test["date"].min()


def test_split_chronological_insufficient_history():
    df = pd.DataFrame({"date": [1, 2, 3], "demand": [10, 20, 30]})
    with pytest.raises(ValueError, match="Insufficient data history"):
        split_chronological(df, test_size=0.2, min_train_periods=10)
