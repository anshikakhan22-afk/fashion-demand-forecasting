"""Forecasting models: Naive baseline, Seasonal/MA baseline, and Random Forest.
Includes multi-step recursive forecasting and empirical uncertainty intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
try:
    from sklearn.ensemble import RandomForestRegressor
except (ImportError, Exception):
    from sklearn.ensemble._forest import RandomForestRegressor

from src.evaluation import compute_metrics
from src.feature_engineering import build_recursive_feature_row, create_forecasting_features
from src.utils import get_models_dir, setup_logger

logger = setup_logger("forecasting")


@dataclass
class ForecastResult:
    """Structure holding forecasting outputs and diagnostic metadata."""

    model_name: str
    target_name: str
    horizon_weeks: int
    forecast_df: pd.DataFrame
    validation_metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    historical_holdout_df: Optional[pd.DataFrame] = None
    feature_importances: Optional[Dict[str, float]] = None
    uncertainty_method: str = "Residual empirical interval (90% confidence)"


class DemandForecaster:
    """Time-series demand forecaster supporting Baselines and Random Forest."""

    def __init__(
        self,
        model_type: str = "random_forest",
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type.lower()
        self.lags = lags if lags is not None else [1, 2, 4, 8]
        self.rolling_windows = rolling_windows if rolling_windows is not None else [4, 8]
        self.random_state = random_state

        self.model: Optional[Any] = None
        self.feature_cols: List[str] = []
        self.residual_std: float = 0.0
        self.is_fitted: bool = False
        self.metadata: Dict[str, Any] = {}

    def fit(self, train_series: pd.DataFrame) -> "DemandForecaster":
        """Fit model on training time series.
        
        Parameters
        ----------
        train_series : pd.DataFrame
            Dataframe with 'date' and 'demand' columns.
        """
        if len(train_series) < max(self.lags) + 2:
            raise ValueError(
                f"Insufficient training data: {len(train_series)} rows. "
                f"Minimum required is {max(self.lags) + 2} periods."
            )

        if self.model_type in ["naive", "seasonal_naive", "moving_average"]:
            self.is_fitted = True
            # Compute baseline in-sample residual error
            demands = train_series["demand"].values
            if len(demands) > 1:
                diffs = demands[1:] - demands[:-1]
                self.residual_std = float(np.std(diffs))
            return self

        # Random Forest model
        featured = create_forecasting_features(
            train_series,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            drop_na=True,
        )

        target = "demand"
        exclude_cols = ["date", target, "Category", "category", "StockCode", "Description"]
        self.feature_cols = [
            c for c in featured.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(featured[c])
        ]

        X_train = featured[self.feature_cols]
        y_train = featured[target]

        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        # In-sample residuals for uncertainty intervals
        train_preds = self.model.predict(X_train)
        residuals = y_train.values - train_preds
        self.residual_std = float(np.std(residuals))
        self.is_fitted = True

        logger.info(
            "Fitted Random Forest on %d samples (residual_std=%.2f)",
            len(X_train),
            self.residual_std,
        )
        return self

    def predict_future(
        self,
        history_df: pd.DataFrame,
        horizon: int = 12,
        freq: str = "W-MON",
    ) -> pd.DataFrame:
        """Generate multi-step future demand forecast.
        
        Parameters
        ----------
        history_df : pd.DataFrame
            Historical contiguous series up to forecast origin with 'date' and 'demand'.
        horizon : int
            Forecast horizon periods (e.g. 1 to 12 weeks).
        freq : str
            Date frequency.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        last_date = pd.to_datetime(history_df["date"].iloc[-1])
        # Generate future dates strictly after the last observed date
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=7 if "W" in freq else 1),
            periods=horizon,
            freq=freq,
        )

        history_demands = list(history_df["demand"].astype(float).values)
        predictions = []
        lower_bounds = []
        upper_bounds = []

        if self.model_type == "naive":
            last_val = history_demands[-1]
            for step in range(1, horizon + 1):
                pred = max(0.0, last_val)
                # Uncertainty widens with step horizon
                margin = 1.645 * self.residual_std * np.sqrt(step)
                predictions.append(pred)
                lower_bounds.append(max(0.0, pred - margin))
                upper_bounds.append(pred + margin)

        elif self.model_type in ["seasonal_naive", "moving_average"]:
            # 4-week moving average baseline
            window = min(4, len(history_demands))
            ma_val = float(np.mean(history_demands[-window:]))
            for step in range(1, horizon + 1):
                pred = max(0.0, ma_val)
                margin = 1.645 * self.residual_std * np.sqrt(step)
                predictions.append(pred)
                lower_bounds.append(max(0.0, pred - margin))
                upper_bounds.append(pred + margin)

        else:
            # Recursive Multi-Step Machine Learning Forecast
            temp_history = list(history_demands)
            curr_time_idx = len(history_df)

            for step_idx, f_date in enumerate(future_dates, start=1):
                # Build single-step features from historical & predicted values
                feat_row = build_recursive_feature_row(
                    history_demand=temp_history,
                    next_date=f_date,
                    time_idx=curr_time_idx,
                    lags=self.lags,
                    rolling_windows=self.rolling_windows,
                )

                # Ensure exact feature column alignment
                X_step = feat_row[self.feature_cols]
                step_pred = float(self.model.predict(X_step)[0])
                # Non-negativity constraint on retail demand
                step_pred = max(0.0, step_pred)

                # Recursive feed: append prediction to history for subsequent lags
                temp_history.append(step_pred)
                curr_time_idx += 1

                # Dynamic uncertainty band based on residual variance and multi-step expansion
                margin = 1.645 * self.residual_std * np.sqrt(1.0 + 0.15 * step_idx)
                predictions.append(step_pred)
                lower_bounds.append(max(0.0, step_pred - margin))
                upper_bounds.append(step_pred + margin)

        forecast_df = pd.DataFrame({
            "forecast_date": future_dates,
            "predicted_demand": np.round(predictions, 2),
            "planning_quantity": np.round(predictions).astype(int),
            "lower_bound": np.round(lower_bounds, 2),
            "upper_bound": np.round(upper_bounds, 2),
        })

        return forecast_df

    def evaluate_holdout(
        self,
        full_series: pd.DataFrame,
        test_size: float = 0.2,
    ) -> Tuple[Dict[str, Optional[float]], pd.DataFrame]:
        """Perform chronological holdout evaluation.
        
        Parameters
        ----------
        full_series : pd.DataFrame
            Complete time series.
        test_size : float
            Fraction for holdout evaluation.
        """
        n = len(full_series)
        split_idx = int(np.floor(n * (1.0 - test_size)))
        min_hist = max(self.lags) + 4
        if split_idx < min_hist:
            split_idx = min_hist

        train_part = full_series.iloc[:split_idx].copy().reset_index(drop=True)
        test_part = full_series.iloc[split_idx:].copy().reset_index(drop=True)

        if test_part.empty:
            return {"MAE": None, "RMSE": None, "WAPE": None, "MAPE": None, "count": 0}, pd.DataFrame()

        # Fit specifically on training portion
        eval_forecaster = DemandForecaster(
            model_type=self.model_type,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            random_state=self.random_state,
        )
        eval_forecaster.fit(train_part)

        # Predict multi-step matching holdout length
        test_horizon = len(test_part)
        test_preds_df = eval_forecaster.predict_future(
            train_part,
            horizon=test_horizon,
        )

        test_part["predicted_demand"] = test_preds_df["predicted_demand"].values
        test_part["lower_bound"] = test_preds_df["lower_bound"].values
        test_part["upper_bound"] = test_preds_df["upper_bound"].values

        metrics = compute_metrics(
            actual=test_part["demand"].values,
            predicted=test_part["predicted_demand"].values,
        )

        return metrics, test_part

    def save(self, filepath: Path) -> None:
        """Serialize model artifact to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model_type": self.model_type,
            "lags": self.lags,
            "rolling_windows": self.rolling_windows,
            "random_state": self.random_state,
            "model": self.model,
            "feature_cols": self.feature_cols,
            "residual_std": self.residual_std,
            "metadata": self.metadata,
        }
        joblib.dump(artifact, filepath)
        logger.info("Saved model artifact to %s", filepath)

    @classmethod
    def load(cls, filepath: Path) -> "DemandForecaster":
        """Load model artifact from disk."""
        artifact = joblib.load(filepath)
        instance = cls(
            model_type=artifact["model_type"],
            lags=artifact["lags"],
            rolling_windows=artifact["rolling_windows"],
            random_state=artifact["random_state"],
        )
        instance.model = artifact["model"]
        instance.feature_cols = artifact["feature_cols"]
        instance.residual_std = artifact["residual_std"]
        instance.metadata = artifact.get("metadata", {})
        instance.is_fitted = True
        return instance
