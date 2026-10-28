"""Script to clean raw retail transactions and prepare aggregated weekly demand series.
Usage:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_raw_dataset, run_data_quality_report
from src.preprocessing import aggregate_demand_series, clean_retail_data
from src.utils import get_processed_data_dir, get_reports_dir, setup_logger

logger = setup_logger("scripts.prepare_data")


def main() -> None:
    """Execute data cleaning and aggregation pipeline."""
    proc_dir = get_processed_data_dir()
    reports_dir = get_reports_dir()

    print("=" * 70)
    print("FASHION DEMAND FORECASTING: DATA PREPARATION PIPELINE")
    print("=" * 70)

    # 1. Load dataset
    print("\n[1/4] Loading source transactions dataset...")
    df_raw, meta = load_raw_dataset()
    print(f"Dataset: {meta.name}")
    print(f"Loaded: {meta.total_rows:,} rows across {meta.total_columns} columns")
    print(f"Date Range: {meta.min_date} to {meta.max_date}")

    # 2. Preprocessing & Fashion Category Taxonomy
    print("\n[2/4] Running data cleaning & category classification...")
    df_clean, stats = clean_retail_data(df_raw, include_only_positive_demand=True)

    print(f"Raw rows: {stats.raw_rows:,}")
    print(f"Duplicates removed: {stats.duplicates_removed:,}")
    print(f"Missing descriptions removed: {stats.missing_descriptions_removed:,}")
    print(f"Missing dates removed: {stats.missing_dates_removed:,}")
    print(f"Cancellations / returns filtered: {stats.cancellations_filtered:,}")
    print(f"Retained clean records: {stats.net_records_retained:,}")
    print(f"Unique fashion categories: {stats.unique_categories}")
    print(f"Fashion category coverage: {stats.fashion_coverage_pct:.1f}%")

    # 3. Save cleaned transaction data
    clean_csv_path = proc_dir / "cleaned_sales.csv"
    print(f"\n[3/4] Saving cleaned records to {clean_csv_path}...")
    df_clean.to_csv(clean_csv_path, index=False)

    # 4. Aggregate demand into weekly time series
    print("\n[4/4] Aggregating weekly demand series by category and overall...")
    categories = sorted(list(df_clean["Category"].unique()))

    weekly_series_list = []
    # Overall total demand
    overall_weekly = aggregate_demand_series(df_clean, group_col=None, group_val=None, freq="W-MON")
    overall_weekly["Category"] = "All Categories"
    weekly_series_list.append(overall_weekly)

    # Per-category demand
    for cat in categories:
        cat_weekly = aggregate_demand_series(df_clean, group_col="Category", group_val=cat, freq="W-MON")
        cat_weekly["Category"] = cat
        weekly_series_list.append(cat_weekly)

    all_weekly = pd.concat(weekly_series_list, ignore_index=True)
    weekly_csv_path = proc_dir / "weekly_demand.csv"
    all_weekly.to_csv(weekly_csv_path, index=False)
    print(f"Saved aggregated weekly demand to {weekly_csv_path} ({len(all_weekly):,} total category-week observations)")

    # 5. Generate and save summary report
    summary = {
        "dataset_name": meta.name,
        "is_sample": meta.is_sample,
        "source_path": meta.source_path,
        "raw_rows": stats.raw_rows,
        "retained_rows": stats.net_records_retained,
        "duplicates_removed": stats.duplicates_removed,
        "cancellations_filtered": stats.cancellations_filtered,
        "fashion_coverage_pct": stats.fashion_coverage_pct,
        "target_convention": stats.target_convention,
        "categories": categories,
        "weekly_records": len(all_weekly),
    }

    report_path = reports_dir / "data_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved preparation diagnostic report to {report_path}")
    print("=" * 70)
    print("DATA PREPARATION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
