"""Data loader module for retail and fashion sales datasets.
Handles dataset acquisition, integrity checks, and fallback mechanisms.
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.utils import (
    get_data_dir,
    get_processed_data_dir,
    get_raw_data_dir,
    get_sample_data_dir,
    setup_logger,
)

logger = setup_logger("data_loader")

# UCI Online Retail Dataset official URL and high-speed mirror
UCI_ONLINE_RETAIL_ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
UCI_MIRROR_CSV_URL = (
    "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Online_Retail/Online_Retail.csv"
)


@dataclass
class DatasetMetadata:
    """Metadata describing the loaded dataset."""

    name: str
    source_path: str
    is_sample: bool
    total_rows: int
    total_columns: int
    columns: list[str] = field(default_factory=list)
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    description: str = ""


def generate_sample_fashion_data(output_path: Optional[Path] = None) -> pd.DataFrame:
    """Generate a realistic, deterministic sample fashion transaction dataset.
    
    Used strictly as an offline or network-failure fallback for pipeline testing.
    Clearly labelled as sample data.
    """
    if output_path is None:
        output_path = get_sample_data_dir() / "fashion_retail_sample.csv"

    import numpy as np

    # Deterministic generation
    rng = np.random.RandomState(42)

    categories = {
        "Tops": [
            ("T101", "VINTAGE COTTON T-SHIRT WHITE"),
            ("T102", "SILK BUTTON-DOWN BLOUSE CREAM"),
            ("T103", "GRAPHIC PRINT CREWNECK TEE"),
            ("T104", "LINEN CASUAL SUMMER SHIRT"),
        ],
        "Dresses": [
            ("D201", "FLORAL WRAP MIDI DRESS"),
            ("D202", "LITTLE BLACK EVENING DRESS"),
            ("D203", "SUMMER BOHO MAXI DRESS"),
            ("D204", "PLEATED SHIRT DRESS NAVY"),
        ],
        "Outerwear": [
            ("O301", "CLASSIC WOOL OVERCOAT CAMEL"),
            ("O302", "DENIM JACKET VINTAGE WASH"),
            ("O303", "QUILTED PUFFER JACKET BLACK"),
            ("O304", "WATERPROOF TRENCH COAT KHAKI"),
        ],
        "Trousers": [
            ("P401", "SLIM FIT CHINO TROUSERS NAVY"),
            ("P402", "HIGH-WAIST STRAIGHT LEG JEANS"),
            ("P403", "TAILORED WOOL BLEND TROUSERS"),
            ("P404", "RELAXED LINEN WIDE-LEG PANTS"),
        ],
        "Accessories": [
            ("A501", "CASHMERE KNIT WINTER SCARF"),
            ("A502", "GENUINE LEATHER TOTE BAG"),
            ("A503", "WOOL FEDORA HAT CHARCOAL"),
            ("A504", "LEATHER BELT CLASSIC BUCKLE"),
        ],
        "Footwear": [
            ("F601", "CLASSIC LEATHER ANKLE BOOTS"),
            ("F602", "CANVAS LOW-TOP SNEAKERS WHITE"),
            ("F603", "SUEDE LOAFERS DARK BROWN"),
            ("F604", "LEATHER STRAPPY SUMMER SANDALS"),
        ],
    }

    # Generate daily dates across 104 weeks (2 full years: 2021-01-01 to 2022-12-30)
    dates = pd.date_range(start="2021-01-01", end="2022-12-30", freq="D")
    records = []
    invoice_id = 100000

    all_items = []
    for cat, items in categories.items():
        for code, desc in items:
            all_items.append((cat, code, desc))

    for current_date in dates:
        # Base seasonal factor: Winter bump (Q4/holiday), summer wave
        doy = current_date.dayofyear
        seasonal_wave = 1.0 + 0.3 * np.sin(2 * np.pi * (doy - 40) / 365.25)
        is_weekend = current_date.weekday() >= 5
        weekend_multiplier = 1.35 if is_weekend else 1.0

        # Number of transactions on this day
        n_transactions = rng.poisson(lam=18 * seasonal_wave * weekend_multiplier)

        for _ in range(n_transactions):
            invoice_id += 1
            cat, code, desc = all_items[rng.choice(len(all_items))]

            # Category-specific seasonality
            cat_multiplier = 1.0
            month = current_date.month
            if cat == "Outerwear" and month in [10, 11, 12, 1, 2]:
                cat_multiplier = 1.8
            elif cat == "Dresses" and month in [5, 6, 7, 8]:
                cat_multiplier = 1.6
            elif cat == "Accessories" and month in [11, 12]:
                cat_multiplier = 2.0  # Holiday gifting

            qty = int(max(1, rng.negative_binomial(n=3, p=0.4) * cat_multiplier))
            unit_price = round(rng.uniform(15.0, 120.0), 2)
            cust_id = int(rng.choice(range(1001, 1500)))

            records.append({
                "InvoiceNo": str(invoice_id),
                "StockCode": code,
                "Description": desc,
                "Quantity": qty,
                "InvoiceDate": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                "UnitPrice": unit_price,
                "CustomerID": cust_id,
                "Country": "United Kingdom",
            })

            # Small realistic return rate (~2.5% of transactions)
            if rng.random() < 0.025:
                records.append({
                    "InvoiceNo": f"C{invoice_id}",
                    "StockCode": code,
                    "Description": desc,
                    "Quantity": -max(1, int(qty * 0.5)),
                    "InvoiceDate": (current_date + pd.Timedelta(days=rng.randint(1, 5))).strftime("%Y-%m-%d %H:%M:%S"),
                    "UnitPrice": unit_price,
                    "CustomerID": cust_id,
                    "Country": "United Kingdom",
                })

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved sample fashion retail dataset to %s (%d records)", output_path, len(df))
    return df


def download_uci_online_retail(target_dir: Optional[Path] = None) -> Optional[Path]:
    """Attempt to download the UCI Online Retail dataset.
    
    Tries the primary UCI zip archive first, then the direct CSV mirror.
    Returns path to downloaded file or None if download fails.
    """
    if target_dir is None:
        target_dir = get_raw_data_dir()

    target_dir.mkdir(parents=True, exist_ok=True)
    csv_dest = target_dir / "online_retail.csv"
    xlsx_dest = target_dir / "Online Retail.xlsx"
    zip_dest = target_dir / "online_retail.zip"

    if csv_dest.exists() and csv_dest.stat().st_size > 10000:
        logger.info("Raw CSV already exists: %s", csv_dest)
        return csv_dest
    if xlsx_dest.exists() and xlsx_dest.stat().st_size > 10000:
        logger.info("Raw Excel already exists: %s", xlsx_dest)
        return xlsx_dest

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    import requests

    # Attempt 1: Direct CSV Mirror via requests
    logger.info("Attempting download from CSV mirror: %s", UCI_MIRROR_CSV_URL)
    try:
        resp = requests.get(UCI_MIRROR_CSV_URL, headers=headers, timeout=30, stream=True)
        if resp.status_code == 200:
            with open(csv_dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            if csv_dest.stat().st_size > 10000:
                logger.info("Successfully downloaded CSV to %s (%d bytes)", csv_dest, csv_dest.stat().st_size)
                return csv_dest
    except Exception as exc:
        logger.warning("Direct CSV download failed: %s", exc)

    # Attempt 2: Official UCI Zip
    logger.info("Attempting download from official UCI archive: %s", UCI_ONLINE_RETAIL_ZIP_URL)
    try:
        req = urllib.request.Request(UCI_ONLINE_RETAIL_ZIP_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=45) as response:
            zip_bytes = response.read()
            if len(zip_bytes) > 10000:
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                    z.extractall(target_dir)
                logger.info("Extracted UCI archive into %s", target_dir)
                # Check for extracted xlsx or csv
                for ext in ["*.xlsx", "*.xls", "*.csv"]:
                    matches = list(target_dir.glob(ext))
                    if matches:
                        return matches[0]
    except Exception as exc:
        logger.warning("Official UCI zip download failed: %s", exc)

    return None


def load_raw_dataset(force_sample: bool = False) -> Tuple[pd.DataFrame, DatasetMetadata]:
    """Load retail transaction dataset.
    
    Checks raw data directory, then attempts download, and falls back to
    sample fashion retail dataset if offline.
    """
    raw_dir = get_raw_data_dir()
    sample_dir = get_sample_data_dir()
    sample_path = sample_dir / "fashion_retail_sample.csv"

    if force_sample:
        if not sample_path.exists():
            generate_sample_fashion_data(sample_path)
        df = pd.read_csv(sample_path)
        meta = DatasetMetadata(
            name="Sample Fashion Retail Dataset (Deterministic Fallback)",
            source_path=str(sample_path),
            is_sample=True,
            total_rows=len(df),
            total_columns=len(df.columns),
            columns=list(df.columns),
            min_date=str(df["InvoiceDate"].min()) if "InvoiceDate" in df.columns else None,
            max_date=str(df["InvoiceDate"].max()) if "InvoiceDate" in df.columns else None,
            description="Synthetic fashion retail dataset generated with seasonal trends for testing.",
        )
        return df, meta

    # Search for existing raw files
    candidates = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.xlsx")) + list(raw_dir.glob("*.xls"))
    valid_file = None
    for cand in candidates:
        if cand.stat().st_size > 10000:
            valid_file = cand
            break

    # If no file found, try downloading
    if valid_file is None:
        valid_file = download_uci_online_retail(raw_dir)

    # If still no file, use sample fallback
    if valid_file is None or not valid_file.exists():
        logger.warning("No valid raw dataset found and network download unavailable. Using sample fallback.")
        if not sample_path.exists():
            generate_sample_fashion_data(sample_path)
        df = pd.read_csv(sample_path)
        meta = DatasetMetadata(
            name="Sample Fashion Retail Dataset (Offline Fallback)",
            source_path=str(sample_path),
            is_sample=True,
            total_rows=len(df),
            total_columns=len(df.columns),
            columns=list(df.columns),
            min_date=str(df["InvoiceDate"].min()) if "InvoiceDate" in df.columns else None,
            max_date=str(df["InvoiceDate"].max()) if "InvoiceDate" in df.columns else None,
            description="Pre-packaged fashion transaction benchmark dataset.",
        )
        return df, meta

    # Load from existing file
    logger.info("Loading dataset from %s", valid_file)
    if valid_file.suffix.lower() == ".csv":
        # Encoding check: Latin-1 is standard for UCI Online Retail
        try:
            df = pd.read_csv(valid_file, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(valid_file, encoding="ISO-8859-1")
    else:
        df = pd.read_excel(valid_file)

    meta = DatasetMetadata(
        name=f"UCI Online Retail ({valid_file.name})",
        source_path=str(valid_file),
        is_sample=False,
        total_rows=len(df),
        total_columns=len(df.columns),
        columns=list(df.columns),
        min_date=str(df["InvoiceDate"].min()) if "InvoiceDate" in df.columns else None,
        max_date=str(df["InvoiceDate"].max()) if "InvoiceDate" in df.columns else None,
        description="Public transaction dataset from UK-based online retail with apparel & giftware categories.",
    )
    return df, meta


def run_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect dataset quality and return a diagnostic dictionary."""
    total_records = len(df)
    missing_counts = df.isnull().sum().to_dict()
    duplicate_rows = int(df.duplicated().sum())

    report: Dict[str, Any] = {
        "total_rows": total_records,
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": missing_counts,
        "duplicate_rows": duplicate_rows,
        "negative_quantities": 0,
        "cancellations": 0,
        "unique_products": 0,
        "min_date": None,
        "max_date": None,
    }

    qty_col = next((c for c in df.columns if c.lower() in ["quantity", "qty"]), None)
    if qty_col:
        report["negative_quantities"] = int((df[qty_col] < 0).sum())

    inv_col = next((c for c in df.columns if c.lower() in ["invoiceno", "invoice_no", "invoice"]), None)
    if inv_col:
        report["cancellations"] = int(df[inv_col].astype(str).str.startswith("C").sum())

    desc_col = next((c for c in df.columns if c.lower() in ["description", "stockcode", "product_id"]), None)
    if desc_col:
        report["unique_products"] = int(df[desc_col].nunique())

    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col:
        try:
            parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
            valid_dates = parsed_dates.dropna()
            if not valid_dates.empty:
                report["min_date"] = str(valid_dates.min())
                report["max_date"] = str(valid_dates.max())
                report["date_parse_failures"] = int(parsed_dates.isna().sum())
        except Exception:
            pass

    return report
