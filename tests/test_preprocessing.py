"""Unit tests for data preprocessing and category classification.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import (
    aggregate_demand_series,
    assign_fashion_category,
    clean_retail_data,
    normalize_columns,
)


def test_normalize_columns():
    df = pd.DataFrame({
        "invoice_no": [101],
        "stock_code": ["A1"],
        "description": ["T-Shirt"],
        "qty": [2],
        "invoice_date": ["2022-01-01"],
        "unit_price": [19.99],
    })
    normalized = normalize_columns(df)
    assert "InvoiceNo" in normalized.columns
    assert "StockCode" in normalized.columns
    assert "Description" in normalized.columns
    assert "Quantity" in normalized.columns
    assert "InvoiceDate" in normalized.columns
    assert "UnitPrice" in normalized.columns


def test_assign_fashion_category():
    assert assign_fashion_category("VINTAGE COTTON T-SHIRT WHITE") == "Tops"
    assert assign_fashion_category("FLORAL SUMMER DRESS") == "Dresses"
    assert assign_fashion_category("WOOL TRENCH COAT") == "Outerwear"
    assert assign_fashion_category("SLIM FIT CHINO TROUSERS") == "Trousers"
    assert assign_fashion_category("LEATHER TOTE BAG") == "Accessories"
    assert assign_fashion_category("CLASSIC ANKLE BOOTS") == "Footwear"
    assert assign_fashion_category("CERAMIC COFFEE MUG") == "Other"
    assert assign_fashion_category("") == "Other"
    assert assign_fashion_category(None) == "Other"


def test_clean_retail_data_duplicates_and_cancellations():
    raw_data = pd.DataFrame({
        "InvoiceNo": ["1001", "1001", "C1002", "1003", "1004"],
        "StockCode": ["T1", "T1", "T1", "T2", "T3"],
        "Description": ["Cotton Top", "Cotton Top", "Cotton Top", "Floral Dress", None],
        "Quantity": [2, 2, -1, 3, 5],
        "InvoiceDate": ["2022-01-01", "2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
    })
    cleaned, stats = clean_retail_data(raw_data, include_only_positive_demand=True)

    # 1 duplicate dropped, 1 cancellation dropped, 1 missing description dropped
    assert len(cleaned) == 2
    assert stats.duplicates_removed == 1
    assert stats.cancellations_filtered >= 1
    assert stats.missing_descriptions_removed == 1
    assert all(cleaned["Quantity"] > 0)


def test_aggregate_demand_series():
    df = pd.DataFrame({
        "InvoiceDate": pd.to_datetime(["2022-01-03", "2022-01-17"]),
        "Quantity": [10, 20],
        "Category": ["Tops", "Tops"],
    })
    agg = aggregate_demand_series(df, group_col="Category", group_val="Tops", freq="W-MON")

    assert len(agg) == 3  # Jan 3, Jan 10 (filled 0), Jan 17
    assert "date" in agg.columns
    assert "demand" in agg.columns
    # Check that middle week without sales is filled with 0
    assert agg.loc[agg["date"] == "2022-01-10", "demand"].values[0] == 0.0
    assert agg.loc[agg["date"] == "2022-01-03", "demand"].values[0] == 10.0
    assert agg.loc[agg["date"] == "2022-01-17", "demand"].values[0] == 20.0


def test_empty_dataframe_handling():
    empty_df = pd.DataFrame()
    cleaned, stats = clean_retail_data(empty_df)
    assert len(cleaned) == 0
    assert stats.raw_rows == 0

    agg = aggregate_demand_series(cleaned)
    assert agg.empty
    assert list(agg.columns) == ["date", "demand"]
