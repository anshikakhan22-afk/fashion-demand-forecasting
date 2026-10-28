"""Script to acquire raw retail dataset or initialize sample benchmark fallback.
Usage:
    python scripts/download_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import download_uci_online_retail, generate_sample_fashion_data, run_data_quality_report
from src.utils import get_raw_data_dir, get_sample_data_dir, setup_logger

logger = setup_logger("scripts.download_data")


def main() -> None:
    """Execute data acquisition pipeline."""
    raw_dir = get_raw_data_dir()
    sample_dir = get_sample_data_dir()

    print("=" * 70)
    print("FASHION DEMAND FORECASTING: DATASET ACQUISITION")
    print("=" * 70)

    # 1. Attempt official download
    print(f"\n[1/2] Checking raw data directory: {raw_dir}")
    downloaded_path = download_uci_online_retail(raw_dir)

    if downloaded_path and downloaded_path.exists():
        size_mb = downloaded_path.stat().st_size / (1024 * 1024)
        print(f"SUCCESS: Raw retail dataset acquired at {downloaded_path} ({size_mb:.2f} MB)")
    else:
        print("NOTICE: Automatic download from public UCI archive was not completed.")
        print("Manual download link: https://archive.ics.uci.edu/dataset/352/online+retail")
        print("Place 'Online Retail.xlsx' or 'online_retail.csv' in 'data/raw/'.")

    # 2. Always ensure a deterministic offline benchmark sample is generated
    sample_path = sample_dir / "fashion_retail_sample.csv"
    print(f"\n[2/2] Ensuring offline benchmark sample exists: {sample_path}")
    sample_df = generate_sample_fashion_data(sample_path)
    sample_size_mb = sample_path.stat().st_size / (1024 * 1024)
    print(f"SUCCESS: Deterministic fashion sample dataset ready ({len(sample_df):,} records, {sample_size_mb:.2f} MB)")
    print("\nDataset acquisition complete.")


if __name__ == "__main__":
    main()
