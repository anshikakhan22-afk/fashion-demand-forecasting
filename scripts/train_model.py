"""Script to train and evaluate baseline and Random Forest models on weekly demand.
Usage:
    python scripts/train_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.evaluation import compute_metrics
from src.forecasting import DemandForecaster
from src.utils import get_models_dir, get_processed_data_dir, get_reports_dir, setup_logger

logger = setup_logger("scripts.train_model")


def main() -> None:
    """Execute model training, chronological evaluation, and model artifact saving."""
    proc_dir = get_processed_data_dir()
    models_dir = get_models_dir()
    reports_dir = get_reports_dir()

    weekly_path = proc_dir / "weekly_demand.csv"
    if not weekly_path.exists():
        print(f"ERROR: Processed weekly demand file not found at {weekly_path}")
        print("Please run 'python scripts/prepare_data.py' first.")
        sys.exit(1)

    df_weekly = pd.read_csv(weekly_path)
    df_weekly["date"] = pd.to_datetime(df_weekly["date"])

    print("=" * 70)
    print("FASHION DEMAND FORECASTING: MODEL TRAINING & VALIDATION")
    print("=" * 70)

    # Filter primary series (Tops or All Categories)
    available_categories = list(df_weekly["Category"].unique())
    selected_category = "Tops" if "Tops" in available_categories else available_categories[0]
    print(f"\nTraining benchmark models on category: '{selected_category}'")

    series = df_weekly[df_weekly["Category"] == selected_category].sort_values("date").reset_index(drop=True)
    print(f"Total time periods: {len(series)} weeks ({series['date'].min().strftime('%Y-%m-%d')} to {series['date'].max().strftime('%Y-%m-%d')})")
    print(f"Demand range: {series['demand'].min():.0f} to {series['demand'].max():.0f} units (mean: {series['demand'].mean():.1f})")

    # Chronological validation split (holdout: last 20% of periods)
    test_size = 0.2
    n = len(series)
    split_idx = int(n * (1.0 - test_size))
    train_df = series.iloc[:split_idx].reset_index(drop=True)
    test_df = series.iloc[split_idx:].reset_index(drop=True)

    print(f"\nChronological Holdout Split:")
    print(f"  Training set:   {len(train_df)} weeks ({train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')})")
    print(f"  Validation set: {len(test_df)} weeks ({test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')})")

    candidate_models = [
        ("Naive Baseline", "naive"),
        ("4-Week Moving Average Baseline", "moving_average"),
        ("Random Forest Regressor (ML)", "random_forest"),
    ]

    results_table = []
    trained_forecasters = {}

    for display_name, m_type in candidate_models:
        print(f"\nTraining and evaluating: {display_name}...")
        forecaster = DemandForecaster(model_type=m_type)
        metrics, holdout_preds = forecaster.evaluate_holdout(series, test_size=test_size)

        # Fit on full series for final future predictions
        full_forecaster = DemandForecaster(model_type=m_type)
        full_forecaster.fit(series)
        trained_forecasters[m_type] = full_forecaster

        mae_str = f"{metrics['MAE']:.2f}" if metrics['MAE'] is not None else "N/A"
        rmse_str = f"{metrics['RMSE']:.2f}" if metrics['RMSE'] is not None else "N/A"
        wape_str = f"{metrics['WAPE']:.2f}%" if metrics['WAPE'] is not None else "N/A"

        print(f"  Validation Metrics -> MAE: {mae_str} units | RMSE: {rmse_str} units | WAPE: {wape_str}")

        results_table.append({
            "model_name": display_name,
            "model_type": m_type,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "WAPE": metrics["WAPE"],
            "MAPE": metrics["MAPE"],
            "test_periods": len(test_df),
        })

    # Save best ML forecaster (Random Forest)
    best_forecaster = trained_forecasters["random_forest"]
    best_forecaster.metadata = {
        "category": selected_category,
        "n_training_samples": len(series),
        "start_date": str(series["date"].min()),
        "end_date": str(series["date"].max()),
        "validation_metrics": next(r for r in results_table if r["model_type"] == "random_forest"),
        "target_convention": "Gross weekly unit sales",
    }
    artifact_path = models_dir / "demand_forecaster.joblib"
    best_forecaster.save(artifact_path)
    print(f"\nSaved production Random Forest model artifact to {artifact_path}")

    # Generate 12-week future forecast
    print("\nGenerating 12-week multi-step future demand forecast...")
    forecast_df = best_forecaster.predict_future(series, horizon=12, freq="W-MON")
    forecast_df["category"] = selected_category
    forecast_export_path = reports_dir / "sample_forecast.csv"
    forecast_df.to_csv(forecast_export_path, index=False)
    print(f"Exported sample 12-week forecast to {forecast_export_path}")

    # Save model comparison report
    comparison_report_path = reports_dir / "model_comparison.json"
    with open(comparison_report_path, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)
    print(f"Saved model evaluation comparison to {comparison_report_path}")

    print("\n" + "=" * 70)
    print("MODEL VALIDATION SUMMARY (LOWER IS BETTER)")
    print("=" * 70)
    for r in results_table:
        print(f"  {r['model_name']:<32} | MAE: {r['MAE']:>6} units | RMSE: {r['RMSE']:>6} units | WAPE: {r['WAPE']:>6}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
