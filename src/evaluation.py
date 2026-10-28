"""Evaluation metrics and chronological validation for demand forecasting.
Safe division handling, zero-demand resilience, and metrics interpretation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_metrics(
    actual: np.ndarray | pd.Series | List[float],
    predicted: np.ndarray | pd.Series | List[float],
) -> Dict[str, Optional[float]]:
    """Compute time-series demand forecasting metrics safely.
    
    Returns
    -------
    dict
        Dictionary containing MAE, RMSE, WAPE, and MAPE (if defined).
    """
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)

    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        return {
            "MAE": None,
            "RMSE": None,
            "WAPE": None,
            "MAPE": None,
            "count": 0,
        }

    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # WAPE = sum(|y - y_hat|) / sum(y)
    sum_actual = float(np.sum(np.abs(y_true)))
    if sum_actual > 1e-9:
        wape = float((np.sum(abs_errors) / sum_actual) * 100.0)
    else:
        wape = None

    # MAPE = (1/n) * sum(|y - y_hat| / |y|)
    # Undefined if actual is 0. Calculate on non-zero entries if reasonable.
    non_zero_mask = np.abs(y_true) > 1e-9
    if np.any(non_zero_mask):
        mape = float(np.mean(abs_errors[non_zero_mask] / np.abs(y_true[non_zero_mask])) * 100.0)
    else:
        mape = None

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "WAPE": round(wape, 2) if wape is not None else None,
        "MAPE": round(mape, 2) if mape is not None else None,
        "count": len(y_true),
    }


def split_chronological(
    df: pd.DataFrame,
    test_size: float = 0.2,
    min_train_periods: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split time-series dataset into train and validation sets chronologically.
    
    Guarantees no shuffling and respects minimum training history requirements.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset sorted by date.
    test_size : float
        Fraction of observations to reserve for holdout test.
    min_train_periods : int
        Minimum number of records required for training.
    """
    n = len(df)
    if n < min_train_periods:
        raise ValueError(
            f"Insufficient data history ({n} periods). At least {min_train_periods} "
            "periods are required for reliable time-series modeling."
        )

    split_idx = int(np.floor(n * (1.0 - test_size)))
    if split_idx < min_train_periods:
        split_idx = min_train_periods

    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    return train_df, test_df
