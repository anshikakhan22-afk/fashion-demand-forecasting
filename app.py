"""Streamlit Web Application: Fashion Demand Forecasting.
Interactive retail analytics dashboard for demand forecasting and inventory planning.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_raw_dataset
from src.evaluation import compute_metrics
from src.forecasting import DemandForecaster
from src.preprocessing import aggregate_demand_series, clean_retail_data
from src.utils import (
    format_metric,
    get_models_dir,
    get_processed_data_dir,
    get_reports_dir,
    setup_logger,
)
from src.visualization import (
    plot_category_breakdown,
    plot_forecast,
    plot_historical_demand,
)

logger = setup_logger("app")

# Page configuration
st.set_page_config(
    page_title="Fashion Demand Forecasting",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Load or generate processed weekly demand and metadata with caching."""
    proc_dir = get_processed_data_dir()
    weekly_csv = proc_dir / "weekly_demand.csv"
    summary_json = get_reports_dir() / "data_summary.json"

    if weekly_csv.exists():
        df_weekly = pd.read_csv(weekly_csv)
        df_weekly["date"] = pd.to_datetime(df_weekly["date"])
        summary = {}
        if summary_json.exists():
            with open(summary_json, "r", encoding="utf-8") as f:
                summary = json.load(f)
        return df_weekly, summary

    # If processed files don't exist yet, process on the fly
    raw_df, meta = load_raw_dataset()
    clean_df, stats = clean_retail_data(raw_df)

    categories = sorted(list(clean_df["Category"].unique()))
    weekly_list = []

    overall = aggregate_demand_series(clean_df, group_col=None, group_val=None, freq="W-MON")
    overall["Category"] = "All Categories"
    weekly_list.append(overall)

    for cat in categories:
        cat_w = aggregate_demand_series(clean_df, group_col="Category", group_val=cat, freq="W-MON")
        cat_w["Category"] = cat
        weekly_list.append(cat_w)

    df_weekly = pd.concat(weekly_list, ignore_index=True)
    summary = {
        "dataset_name": meta.name,
        "is_sample": meta.is_sample,
        "raw_rows": stats.raw_rows,
        "retained_rows": stats.net_records_retained,
        "fashion_coverage_pct": stats.fashion_coverage_pct,
        "categories": categories,
    }
    return df_weekly, summary


@st.cache_resource(show_spinner=False)
def get_trained_models(
    series_data: pd.DataFrame,
    category: str,
) -> Tuple[Dict[str, DemandForecaster], Dict[str, Dict], pd.DataFrame]:
    """Fit candidate models on historical series and compute chronological holdout evaluation."""
    test_size = 0.2
    model_types = [
        ("Random Forest Regressor (ML)", "random_forest"),
        ("4-Week Moving Average Baseline", "moving_average"),
        ("Naive Baseline (Last Value)", "naive"),
    ]

    forecasters = {}
    eval_metrics = {}
    best_holdout_df = None

    for name, m_type in model_types:
        try:
            fc = DemandForecaster(model_type=m_type)
            metrics, holdout = fc.evaluate_holdout(series_data, test_size=test_size)

            # Fit on complete historical series for future forecasting
            full_fc = DemandForecaster(model_type=m_type)
            full_fc.fit(series_data)

            forecasters[name] = full_fc
            eval_metrics[name] = metrics

            if m_type == "random_forest":
                best_holdout_df = holdout
        except Exception as exc:
            logger.warning("Could not evaluate %s: %s", name, exc)
            eval_metrics[name] = {"MAE": None, "RMSE": None, "WAPE": None, "MAPE": None}

    return forecasters, eval_metrics, best_holdout_df


def main():
    # Header
    st.title("👗 Fashion Demand Forecasting")
    st.caption("Machine learning insights for smarter retail inventory planning.")

    # Load data
    try:
        df_weekly, data_summary = load_and_prepare_data()
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        st.info("Run `python scripts/prepare_data.py` from your terminal to initialize datasets.")
        return

    # Sidebar
    st.sidebar.header("🕹️ Forecast Controls")

    # Available categories
    available_cats = list(df_weekly["Category"].unique())
    default_cat_idx = (
        available_cats.index("Tops") if "Tops" in available_cats else 0
    )
    selected_category = st.sidebar.selectbox(
        "Product Category",
        options=available_cats,
        index=default_cat_idx,
        help="Select a clothing category or choose 'All Categories' for aggregate demand.",
    )

    # Forecast Horizon
    horizon_weeks = st.sidebar.slider(
        "Forecast Horizon (Weeks)",
        min_value=1,
        max_value=12,
        value=8,
        step=1,
        help="Number of future weeks to predict.",
    )

    # Model Selection
    model_choice = st.sidebar.selectbox(
        "Forecasting Model",
        options=[
            "Random Forest Regressor (ML)",
            "4-Week Moving Average Baseline",
            "Naive Baseline (Last Value)",
        ],
        index=0,
        help="Choose between machine learning or statistical naive baselines.",
    )

    # Filter data for selected category
    cat_series = (
        df_weekly[df_weekly["Category"] == selected_category]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if cat_series.empty or len(cat_series) < 6:
        st.warning(f"Insufficient historical data available for '{selected_category}'.")
        return

    # Historical Date Filter
    min_date = cat_series["date"].min().date()
    max_date = cat_series["date"].max().date()

    date_range = st.sidebar.date_input(
        "Historical View Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter historical range displayed in charts.",
    )

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_series = cat_series[
            (cat_series["date"] >= start_d) & (cat_series["date"] <= end_d)
        ]
    else:
        filtered_series = cat_series

    # Sidebar Data Status Badge
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Dataset Information")
    is_sample = data_summary.get("is_sample", False)
    st.sidebar.markdown(
        f"**Source:** {'⚡ Synthetic Benchmark (Offline Fallback)' if is_sample else '📦 UCI Online Retail'}"
    )
    st.sidebar.markdown(f"**Target:** Weekly Gross Units Sold")
    st.sidebar.markdown(f"**Total Observed Weeks:** {len(cat_series)}")
    st.sidebar.markdown(f"**Date Range:** {min_date} to {max_date}")

    # Model Training and Holdout Evaluation
    with st.spinner("Calculating forecasts and validating models..."):
        forecasters, eval_metrics, holdout_df = get_trained_models(cat_series, selected_category)

    active_forecaster = forecasters.get(model_choice)
    if active_forecaster is None:
        st.error(f"Selected model '{model_choice}' could not be fitted.")
        return

    # Generate future forecast
    forecast_df = active_forecaster.predict_future(cat_series, horizon=horizon_weeks)
    forecast_df["Category"] = selected_category
    forecast_df["Model"] = model_choice

    # Key Performance Indicators (KPI Cards)
    total_hist_units = filtered_series["demand"].sum()
    avg_hist_weekly = filtered_series["demand"].mean()
    total_forecast_units = forecast_df["predicted_demand"].sum()
    chosen_metrics = eval_metrics.get(model_choice, {})
    val_mae = chosen_metrics.get("MAE")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Historical Units Sold",
            value=f"{total_hist_units:,.0f}",
            help="Total units sold in selected category over historical filter window.",
        )
    with col2:
        st.metric(
            label="Avg Weekly Demand",
            value=f"{avg_hist_weekly:,.1f}",
            help="Average units sold per week historically.",
        )
    with col3:
        st.metric(
            label=f"Forecast Demand ({horizon_weeks}W)",
            value=f"{total_forecast_units:,.0f}",
            delta=f"{(total_forecast_units / (avg_hist_weekly * horizon_weeks + 1e-6) - 1.0) * 100:+.1f}% vs baseline",
            help="Expected units to sell over future forecast horizon.",
        )
    with col4:
        st.metric(
            label="Holdout Validation MAE",
            value=format_metric(val_mae, unit="units"),
            help="Mean Absolute Error on unseen chronological holdout test period.",
        )

    st.markdown("---")

    # Main Tabs
    tab_forecast, tab_eval, tab_inventory, tab_data = st.tabs([
        "📈 Demand Forecast",
        "🎯 Model Evaluation",
        "📦 Inventory Planning",
        "📋 Data & Methodology",
    ])

    with tab_forecast:
        st.subheader("Future Demand Forecast vs Historical Actuals")
        st.caption(
            "Visualizing historical sales, chronological validation predictions, and multi-step future demand with 90% uncertainty intervals."
        )

        fig_forecast = plot_forecast(
            historical_df=filtered_series,
            forecast_df=forecast_df,
            holdout_df=holdout_df if model_choice == "Random Forest Regressor (ML)" else None,
            title="Weekly Demand Trajectory & Multi-Step Forecast",
            category_name=selected_category,
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Forecast Table and Download
        st.subheader("Forecast Breakdown by Week")
        display_forecast = forecast_df[[
            "forecast_date",
            "predicted_demand",
            "planning_quantity",
            "lower_bound",
            "upper_bound",
        ]].copy()
        display_forecast["forecast_date"] = display_forecast["forecast_date"].dt.strftime("%Y-%m-%d")
        display_forecast.columns = [
            "Week Starting",
            "Predicted Demand",
            "Planning Quantity (Units)",
            "Lower Bound (90%)",
            "Upper Bound (90%)",
        ]

        st.dataframe(display_forecast, use_container_width=True, hide_index=True)

        # CSV Download Button
        csv_export = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Forecast as CSV",
            data=csv_export,
            file_name=f"forecast_{selected_category.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Export predicted weekly units and uncertainty intervals to CSV.",
        )

    with tab_eval:
        st.subheader("Model Performance & Chronological Holdout Comparison")
        st.write(
            "All candidate models were strictly evaluated on the final 20% holdout window using chronological splitting to ensure zero forward data leakage."
        )

        comparison_rows = []
        for m_name, m_stats in eval_metrics.items():
            comparison_rows.append({
                "Model": m_name,
                "MAE (Units)": format_metric(m_stats.get("MAE")),
                "RMSE (Units)": format_metric(m_stats.get("RMSE")),
                "WAPE (%)": format_metric(m_stats.get("WAPE"), unit="%"),
                "MAPE (%)": format_metric(m_stats.get("MAPE"), unit="%"),
                "Holdout Observations": m_stats.get("count", 0),
            })

        st.table(pd.DataFrame(comparison_rows))

        st.info(
            "💡 **Metrics Interpretation:**\n\n"
            "- **MAE (Mean Absolute Error):** Average magnitude of forecasting errors in physical units sold. Lower is better.\n"
            "- **RMSE (Root Mean Squared Error):** Penalizes larger deviation errors more heavily. Useful when large under-forecasts are costly.\n"
            "- **WAPE (Weighted Absolute Percentage Error):** Overall percentage error weighted by sales volume. Robust against zero-actual weeks.\n"
            "- **MAPE (Mean Absolute Percentage Error):** Average relative percentage error, calculated on non-zero periods."
        )

        if holdout_df is not None and not holdout_df.empty:
            st.subheader("Holdout Predictions vs Actual Sales")
            holdout_display = holdout_df[["date", "demand", "predicted_demand"]].copy()
            holdout_display["date"] = holdout_display["date"].dt.strftime("%Y-%m-%d")
            holdout_display["error"] = (
                holdout_display["demand"] - holdout_display["predicted_demand"]
            ).round(2)
            holdout_display.columns = [
                "Week Date",
                "Actual Demand",
                "Validation Predicted",
                "Residual Error",
            ]
            st.dataframe(holdout_display, use_container_width=True, hide_index=True)

    with tab_inventory:
        st.subheader("Retail Replenishment & Inventory Insights")
        st.write(
            "Translate machine learning demand forecasts into actionable replenishment recommendations."
        )

        inv_col1, inv_col2, inv_col3 = st.columns(3)
        with inv_col1:
            supplier_lead_time = st.number_input(
                "Supplier Lead Time (Weeks)",
                min_value=1,
                max_value=8,
                value=2,
                step=1,
                help="Weeks required from order placement to stock delivery.",
            )
        with inv_col2:
            current_stock = st.number_input(
                "Current On-Hand Stock (Units)",
                min_value=0,
                max_value=10000,
                value=int(avg_hist_weekly * 1.5),
                step=10,
                help="Units currently in inventory.",
            )
        with inv_col3:
            safety_factor = st.slider(
                "Safety Stock Buffer (%)",
                min_value=0,
                max_value=50,
                value=15,
                step=5,
                help="Extra buffer percentage to mitigate demand spikes or delays.",
            )

        # Compute suggested replenishment
        lead_time_forecast = forecast_df.iloc[:supplier_lead_time]["predicted_demand"].sum()
        safety_stock_units = lead_time_forecast * (safety_factor / 100.0)
        required_inventory = lead_time_forecast + safety_stock_units
        recommended_order = max(0, int(np.ceil(required_inventory - current_stock)))

        st.markdown("#### Recommended Stock Order Summary")
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            st.metric(
                "Expected Lead Time Demand",
                f"{lead_time_forecast:,.0f} units",
                f"Next {supplier_lead_time} weeks",
            )
        with rcol2:
            st.metric("Safety Stock Buffer", f"{safety_stock_units:,.0f} units", f"+{safety_factor}%")
        with rcol3:
            st.metric(
                "Suggested Replenishment Order",
                f"{recommended_order:,.0f} units",
                delta="Order Required" if recommended_order > 0 else "Sufficient Stock",
            )

        st.caption(
            "⚠️ *Planning note: This heuristic calculates replenishment based on forecasted mean demand and safety buffer. "
            "It does not account for supplier minimum order quantities (MOQ) or perishable shelf life.*"
        )

    with tab_data:
        st.subheader("Data Provenance & Pipeline Architecture")

        st.markdown(
            f"""
            - **Dataset Source:** [{data_summary.get('dataset_name', 'Retail Sales Dataset')}](https://archive.ics.uci.edu/dataset/352/online+retail)
            - **Raw Records Processed:** {data_summary.get('raw_rows', 'N/A'):,}
            - **Clean Retained Records:** {data_summary.get('retained_rows', 'N/A'):,}
            - **Fashion Category Coverage:** {data_summary.get('fashion_coverage_pct', 'N/A')}%
            - **Aggregation Frequency:** Weekly (resampled by Monday week-start)
            - **Feature Engineering:**
                - Calendar: Week of Year, Month, Quarter, Cyclical Sine/Cosine transformations.
                - Shifted Lags: $t-1$, $t-2$, $t-4$, $t-8$ (leakage-free).
                - Shifted Rolling Windows: 4-week and 8-week moving average and volatility.
            - **Forecasting Strategy:** Multi-step recursive forecasting with non-negativity constraint.
            """
        )

        # Category sales breakdown chart
        st.subheader("Historical Demand Across Categories")
        cat_summary = (
            df_weekly[df_weekly["Category"] != "All Categories"]
            .groupby("Category")["demand"]
            .agg(total_units="sum", avg_weekly="mean")
            .reset_index()
            .rename(columns={"Category": "category"})
        )
        fig_cat = plot_category_breakdown(cat_summary)
        st.plotly_chart(fig_cat, use_container_width=True)


if __name__ == "__main__":
    main()
