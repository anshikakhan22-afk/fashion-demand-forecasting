"""Data preprocessing and category mapping module for fashion retail demand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger("preprocessing")

# Comprehensive keyword-based fashion taxonomy
FASHION_CATEGORY_RULES: Dict[str, List[str]] = {
    "Tops": [
        "shirt", "blouse", "tee", "t-shirt", "top", "tunic", "polo",
        "sweatshirt", "hoodie", "sweater", "cardigan", "knitwear", "jumper"
    ],
    "Dresses": [
        "dress", "gown", "skirt", "sundress", "midi", "maxi", "frock"
    ],
    "Outerwear": [
        "coat", "jacket", "blazer", "parka", "trench", "overcoat", "poncho",
        "anorak", "gilet", "vest", "puffer"
    ],
    "Trousers": [
        "pant", "pants", "trouser", "trousers", "jean", "jeans", "legging",
        "leggings", "short", "shorts", "chino", "jogger"
    ],
    "Accessories": [
        "bag", "tote", "handbag", "clutch", "scarf", "scarves", "glove",
        "gloves", "hat", "cap", "beanie", "belt", "umbrella", "purse",
        "wallet", "necklace", "bracelet", "earring", "ring", "jewelry",
        "brooch", "sunglasses"
    ],
    "Footwear": [
        "boot", "boots", "shoe", "shoes", "sneaker", "sneakers", "sandal",
        "sandals", "slipper", "slippers", "heel", "heels", "loafer",
        "loafers", "sock", "socks"
    ],
}


@dataclass
class CleaningStats:
    """Statistics tracking data transformation steps."""

    raw_rows: int
    duplicates_removed: int
    missing_descriptions_removed: int
    missing_dates_removed: int
    cancellations_filtered: int
    net_records_retained: int
    unique_products: int
    unique_categories: int
    fashion_coverage_pct: float
    target_convention: str = "Gross sales (forward purchases with returns accounted separately)"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to expected schema."""
    col_mapping = {}
    for col in df.columns:
        norm = re.sub(r"[_\s]+", "", str(col)).lower()
        if norm in ["invoiceno", "invoice", "invoicenumber"]:
            col_mapping[col] = "InvoiceNo"
        elif norm in ["stockcode", "itemcode", "productcode", "code"]:
            col_mapping[col] = "StockCode"
        elif norm in ["description", "itemdescription", "productname", "title"]:
            col_mapping[col] = "Description"
        elif norm in ["quantity", "qty", "units"]:
            col_mapping[col] = "Quantity"
        elif norm in ["invoicedate", "date", "timestamp"]:
            col_mapping[col] = "InvoiceDate"
        elif norm in ["unitprice", "price", "rate"]:
            col_mapping[col] = "UnitPrice"
        elif norm in ["customerid", "customer", "user"]:
            col_mapping[col] = "CustomerID"
        elif norm in ["country", "region"]:
            col_mapping[col] = "Country"
        else:
            col_mapping[col] = col.strip()

    return df.rename(columns=col_mapping)


def assign_fashion_category(description: Any) -> str:
    """Classify a product description into a fashion category using keyword rules.
    
    Returns 'Other' if no fashion keywords match.
    """
    if not isinstance(description, str) or not description.strip():
        return "Other"

    desc_lower = description.lower()
    # Word boundary regex matching to avoid partial word mismatches (e.g. 'tape' != 'top')
    for category, keywords in FASHION_CATEGORY_RULES.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, desc_lower):
                return category

    return "Other"


def clean_retail_data(
    df: pd.DataFrame,
    include_only_positive_demand: bool = True,
    target_convention: str = "gross_sales",
) -> Tuple[pd.DataFrame, CleaningStats]:
    """Clean raw retail transactions according to strict data integrity rules.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input raw dataframe.
    include_only_positive_demand : bool
        If True, isolates forward sales transactions (Quantity > 0).
    target_convention : str
        Either 'gross_sales' (Quantity > 0) or 'net_sales' (accounting for returns).
    """
    raw_count = len(df)
    df_clean = normalize_columns(df.copy())

    # 1. Deduplication
    dups_count = int(df_clean.duplicated().sum())
    df_clean = df_clean.drop_duplicates()

    # 2. Date parsing
    if "InvoiceDate" in df_clean.columns:
        df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"], errors="coerce", format="mixed")
        null_dates = int(df_clean["InvoiceDate"].isna().sum())
        df_clean = df_clean.dropna(subset=["InvoiceDate"])
    else:
        null_dates = 0

    # 3. Missing descriptions
    if "Description" in df_clean.columns:
        null_desc = int(df_clean["Description"].isna().sum())
        df_clean = df_clean.dropna(subset=["Description"])
        df_clean["Description"] = df_clean["Description"].astype(str).str.strip()
    else:
        null_desc = 0
        df_clean["Description"] = "Unknown"

    # 4. Handle cancellations and negative quantities
    cancellations_count = 0
    if "InvoiceNo" in df_clean.columns:
        df_clean["InvoiceNo"] = df_clean["InvoiceNo"].astype(str).str.strip()
        is_cancel = df_clean["InvoiceNo"].str.startswith("C", na=False)
        cancellations_count = int(is_cancel.sum())
    else:
        is_cancel = pd.Series(False, index=df_clean.index)

    if "Quantity" in df_clean.columns:
        df_clean["Quantity"] = pd.to_numeric(df_clean["Quantity"], errors="coerce").fillna(0)
        negative_quantities = int((df_clean["Quantity"] < 0).sum())
        cancellations_count = max(cancellations_count, negative_quantities)

        if include_only_positive_demand or target_convention == "gross_sales":
            # Forward sales only
            df_clean = df_clean[(~is_cancel) & (df_clean["Quantity"] > 0)]
        else:
            # Net sales calculation: remove cancellation prefix records but retain net logic
            df_clean = df_clean[~is_cancel]

    # 5. Assign Fashion Categories
    df_clean["Category"] = df_clean["Description"].apply(assign_fashion_category)

    # Calculate fashion coverage
    fashion_rows = (df_clean["Category"] != "Other").sum()
    coverage_pct = round((fashion_rows / len(df_clean) * 100), 2) if len(df_clean) > 0 else 0.0

    stats = CleaningStats(
        raw_rows=raw_count,
        duplicates_removed=dups_count,
        missing_descriptions_removed=null_desc,
        missing_dates_removed=null_dates,
        cancellations_filtered=cancellations_count,
        net_records_retained=len(df_clean),
        unique_products=int(df_clean["StockCode"].nunique()) if "StockCode" in df_clean.columns else 0,
        unique_categories=int(df_clean["Category"].nunique()),
        fashion_coverage_pct=coverage_pct,
        target_convention="Gross weekly unit sales (forward purchases with returns separated)",
    )

    logger.info(
        "Cleaned dataset: raw=%d -> retained=%d (fashion coverage=%.1f%%)",
        raw_count,
        len(df_clean),
        coverage_pct,
    )
    return df_clean, stats


def aggregate_demand_series(
    df: pd.DataFrame,
    group_col: Optional[str] = "Category",
    group_val: Optional[str] = None,
    freq: str = "W-MON",
    fill_missing_periods: bool = True,
) -> pd.DataFrame:
    """Aggregate cleaned transactions into a regular, contiguous time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions dataframe with InvoiceDate and Quantity.
    group_col : str, optional
        Column to filter on (e.g., 'Category' or 'StockCode').
    group_val : str, optional
        Specific value to filter for (e.g. 'Tops', or None for overall total).
    freq : str
        Resampling frequency ('W-MON' for weekly, 'D' for daily).
    fill_missing_periods : bool
        If True, reindexes with full contiguous range and fills missing periods with 0.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "demand"])

    data = df.copy()
    if group_col and group_val:
        data = data[data[group_col] == group_val]

    if data.empty:
        return pd.DataFrame(columns=["date", "demand"])

    # Group by date frequency
    data["date"] = pd.to_datetime(data["InvoiceDate"])
    aggregated = (
        data.set_index("date")
        .resample(freq)["Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"Quantity": "demand"})
    )

    if fill_missing_periods and not aggregated.empty:
        # Guarantee full regular date range without missing steps
        full_range = pd.date_range(
            start=aggregated["date"].min(),
            end=aggregated["date"].max(),
            freq=freq,
            name="date",
        )
        aggregated = (
            aggregated.set_index("date")
            .reindex(full_range, fill_value=0.0)
            .reset_index()
            .rename(columns={"index": "date"})
        )

    # Ensure non-negative demand and clean numeric types
    aggregated["demand"] = aggregated["demand"].astype(float).clip(lower=0.0)
    return aggregated
