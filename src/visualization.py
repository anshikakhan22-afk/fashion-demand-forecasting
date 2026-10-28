"""Visualization components using Plotly for interactive forecasting charts.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_historical_demand(
    df: pd.DataFrame,
    title: str = "Historical Demand Over Time",
    category_name: str = "All Categories",
) -> go.Figure:
    """Create an interactive time-series line chart for historical demand."""
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(
            text="No historical demand data available for this selection.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        return fig

    # Compute 4-week rolling average for smooth trend display
    df_sorted = df.sort_values("date").copy()
    rolling_ma = df_sorted["demand"].rolling(window=4, min_periods=1).mean()

    # Actual demand series
    fig.add_trace(
        go.Scatter(
            x=df_sorted["date"],
            y=df_sorted["demand"],
            mode="lines+markers",
            name="Actual Demand (Units)",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=5),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Units:</b> %{y:,.0f}<extra></extra>",
        )
    )

    # 4-week trend
    fig.add_trace(
        go.Scatter(
            x=df_sorted["date"],
            y=rolling_ma,
            mode="lines",
            name="4-Week Moving Average",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>4W Trend:</b> %{y:,.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"<b>{title}</b> — <i>{category_name}</i>",
        xaxis_title="Date",
        yaxis_title="Units Sold (Weekly)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
    )

    return fig


def plot_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    holdout_df: Optional[pd.DataFrame] = None,
    title: str = "Weekly Demand Forecast",
    category_name: str = "Selected Category",
) -> go.Figure:
    """Plot historical actuals, validation predictions, and future forecasts with uncertainty."""
    fig = go.Figure()

    if historical_df.empty and forecast_df.empty:
        fig.add_annotation(
            text="No forecast data available to plot.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        return fig

    # 1. Historical Actuals
    if not historical_df.empty:
        hist_sorted = historical_df.sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=hist_sorted["date"],
                y=hist_sorted["demand"],
                mode="lines+markers",
                name="Historical Actuals",
                line=dict(color="#2c3e50", width=2),
                marker=dict(size=4),
                hovertemplate="<b>Historical:</b> %{x|%Y-%m-%d}<br><b>Actual Demand:</b> %{y:,.0f}<extra></extra>",
            )
        )

    # 2. Holdout Validation Predictions (if available)
    if holdout_df is not None and not holdout_df.empty and "predicted_demand" in holdout_df.columns:
        fig.add_trace(
            go.Scatter(
                x=holdout_df["date"],
                y=holdout_df["predicted_demand"],
                mode="lines+markers",
                name="Validation Predictions (Holdout)",
                line=dict(color="#e67e22", width=2, dash="dot"),
                marker=dict(size=5, symbol="diamond"),
                hovertemplate="<b>Validation:</b> %{x|%Y-%m-%d}<br><b>Holdout Pred:</b> %{y:,.1f}<extra></extra>",
            )
        )

    # 3. Future Forecast with Uncertainty Intervals
    if not forecast_df.empty:
        f_dates = forecast_df["forecast_date"]
        f_pred = forecast_df["predicted_demand"]

        # Upper bound
        if "upper_bound" in forecast_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=f_dates,
                    y=forecast_df["upper_bound"],
                    mode="lines",
                    name="Upper Bound (90%)",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Lower bound with fill to upper
        if "lower_bound" in forecast_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=f_dates,
                    y=forecast_df["lower_bound"],
                    mode="lines",
                    name="90% Prediction Interval",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(46, 204, 113, 0.2)",
                    hoverinfo="skip",
                )
            )

        # Forecast Line
        fig.add_trace(
            go.Scatter(
                x=f_dates,
                y=f_pred,
                mode="lines+markers",
                name="Future Forecast",
                line=dict(color="#27ae60", width=3),
                marker=dict(size=7, symbol="circle"),
                hovertemplate="<b>Forecast:</b> %{x|%Y-%m-%d}<br><b>Predicted Demand:</b> %{y:,.1f}<extra></extra>",
            )
        )

        # Vertical line marking forecast origin
        if not historical_df.empty:
            origin_date = historical_df["date"].max()
            fig.add_vline(
                x=origin_date,
                line_width=2,
                line_dash="dash",
                line_color="#7f8c8d",
                annotation_text="Forecast Origin",
                annotation_position="top left",
            )

    fig.update_layout(
        title=f"<b>{title}</b> — <i>{category_name}</i>",
        xaxis_title="Date",
        yaxis_title="Units Sold (Weekly)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
    )

    return fig


def plot_category_breakdown(category_summary_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing total volume and average weekly demand across categories."""
    fig = go.Figure()

    if category_summary_df.empty:
        return fig

    sorted_df = category_summary_df.sort_values("total_units", ascending=True)

    fig.add_trace(
        go.Bar(
            y=sorted_df["category"],
            x=sorted_df["total_units"],
            orientation="h",
            marker=dict(color="#3498db"),
            name="Total Units Sold",
            hovertemplate="<b>Category:</b> %{y}<br><b>Total Units:</b> %{x:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Total Historical Sales Volume by Category</b>",
        xaxis_title="Units Sold",
        yaxis_title="Category",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig
