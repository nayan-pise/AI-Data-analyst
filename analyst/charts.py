"""Plotly chart construction from a small, validated spec (used by the LLM tool and the offline engine)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CHART_TYPES = ["bar", "line", "pie", "scatter", "histogram", "box", "heatmap", "area"]
_TEMPLATE = "plotly_white"


class ChartError(ValueError):
    """Chart spec cannot be rendered with the given data."""


def _need(df: pd.DataFrame, *cols: str | None) -> None:
    for c in cols:
        if c and c not in df.columns:
            raise ChartError(f"Column '{c}' not in result. Available columns: {', '.join(map(str, df.columns))}")


def build_chart(df: pd.DataFrame, chart_type: str, x: str | None = None, y: str | None = None,
                color: str | None = None, title: str | None = None) -> go.Figure:
    if df is None or df.empty:
        raise ChartError("The query returned no rows, so there is nothing to plot.")
    chart_type = (chart_type or "").lower()
    if chart_type not in CHART_TYPES:
        raise ChartError(f"Unsupported chart_type '{chart_type}'. Choose from: {', '.join(CHART_TYPES)}")
    _need(df, x, y, color)
    df = df.copy()
    for c in df.columns:  # DuckDB DATE objects -> datetimes so axes behave
        if df[c].dtype == object and len(df[c].dropna()) and hasattr(df[c].dropna().iloc[0], "isoformat"):
            df[c] = pd.to_datetime(df[c])

    try:
        if chart_type == "bar":
            if not (x and y):
                raise ChartError("Bar charts need both x and y.")
            fig = px.bar(df, x=x, y=y, color=color, title=title, template=_TEMPLATE, text_auto=".3s")
        elif chart_type == "line":
            if not (x and y):
                raise ChartError("Line charts need both x and y.")
            fig = px.line(df, x=x, y=y, color=color, title=title, template=_TEMPLATE, markers=True)
        elif chart_type == "area":
            if not (x and y):
                raise ChartError("Area charts need both x and y.")
            fig = px.area(df, x=x, y=y, color=color, title=title, template=_TEMPLATE)
        elif chart_type == "pie":
            if not (x and y):
                raise ChartError("Pie charts need x (labels) and y (values).")
            fig = px.pie(df, names=x, values=y, title=title, template=_TEMPLATE, hole=0.35)
        elif chart_type == "scatter":
            if not (x and y):
                raise ChartError("Scatter charts need both x and y.")
            fig = px.scatter(df, x=x, y=y, color=color, title=title, template=_TEMPLATE, opacity=0.75)
        elif chart_type == "histogram":
            if not (x or y):
                raise ChartError("Histograms need a column in x.")
            fig = px.histogram(df, x=x or y, color=color, title=title, template=_TEMPLATE, nbins=30)
        elif chart_type == "box":
            if not (y or x):
                raise ChartError("Box plots need a numeric column in y.")
            fig = px.box(df, x=x if y else None, y=y or x, color=color, title=title, template=_TEMPLATE)
        else:  # heatmap
            if x and y and color:
                pivot = df.pivot_table(index=y, columns=x, values=color, aggfunc="sum")
            else:
                pivot = df.select_dtypes("number").corr()
            fig = px.imshow(pivot, aspect="auto", title=title or "Heatmap", template=_TEMPLATE,
                            color_continuous_scale="Blues", text_auto=".2f" if x is None else False)
    except ChartError:
        raise
    except Exception as exc:  # plotly raises many exception types for bad specs
        raise ChartError(f"Could not build {chart_type} chart: {exc}") from exc

    fig.update_layout(margin=dict(l=20, r=20, t=60 if title else 20, b=20), legend_title_text="")
    return fig


def forecast_figure(history: pd.DataFrame, forecast: pd.DataFrame, title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=history["period"], y=history["actual"], name="Actual", mode="lines+markers")
    fig.add_scatter(x=forecast["period"], y=forecast["upper"], name="Upper bound", mode="lines",
                    line=dict(width=0), showlegend=False)
    fig.add_scatter(x=forecast["period"], y=forecast["lower"], name="Interval", mode="lines",
                    line=dict(width=0), fill="tonexty", fillcolor="rgba(99,110,250,0.2)")
    fig.add_scatter(x=forecast["period"], y=forecast["forecast"], name="Forecast", mode="lines+markers",
                    line=dict(dash="dash"))
    fig.update_layout(title=title, template=_TEMPLATE, yaxis_title=y_label, margin=dict(l=20, r=20, t=60, b=20))
    return fig
