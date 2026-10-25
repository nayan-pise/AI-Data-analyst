"""Lightweight, explainable forecasting: linear trend (+ additive monthly seasonality when 24+ months)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_FREQ = {"day": "D", "week": "W-MON", "month": "MS"}


@dataclass
class ForecastResult:
    history: pd.DataFrame
    forecast: pd.DataFrame
    method: str
    notes: str


def forecast_series(df: pd.DataFrame, date_column: str, value_column: str, periods: int = 6,
                    freq: str = "month", agg: str = "sum") -> ForecastResult:
    if freq not in _FREQ:
        raise ValueError(f"freq must be one of {sorted(_FREQ)}")
    if agg not in ("sum", "mean", "count"):
        raise ValueError("agg must be 'sum', 'mean' or 'count'")
    for c in (date_column, value_column):
        if c not in df.columns:
            raise KeyError(f"Unknown column '{c}'. Available: {', '.join(df.columns)}")
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        raise ValueError(f"'{date_column}' is not a date column.")
    periods = max(1, min(int(periods), 36))

    d = df[[date_column, value_column]].dropna()
    series = d.set_index(date_column)[value_column].resample(_FREQ[freq]).agg(agg).astype(float)
    notes = []
    dates = d[date_column].drop_duplicates().sort_values()
    sub_monthly = len(dates) > 2 and float(dates.diff().dt.days.median()) < 15   # daily/weekly rows, not one row per month
    if freq == "month" and len(series) > 1 and sub_monthly:
        last = d[date_column].max()
        if last.day < 0.9 * last.days_in_month:
            series = series.iloc[:-1]
            notes.append(f"Dropped the incomplete final month ({last:%b %Y}) so it does not bias the trend.")
    if len(series) < 6:
        raise ValueError(f"Need at least 6 {freq}s of history to forecast; found {len(series)}.")

    t = np.arange(len(series), dtype=float)
    y = series.values
    seasonal = freq == "month" and len(series) >= 24

    def design(tt, idx):
        """Intercept + linear trend (+ month-of-year dummies), fitted jointly by least squares."""
        cols = [np.ones_like(tt), tt]
        if seasonal:
            months = np.asarray(idx.month)
            cols += [(months == m).astype(float) for m in range(2, 13)]
        return np.column_stack(cols)

    def fit_predict(t_train, y_train, t_pred, idx_train, idx_pred):
        x_train = design(t_train, idx_train)
        beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
        resid = y_train - x_train @ beta
        dof = max(len(y_train) - x_train.shape[1], 1)
        return design(t_pred, idx_pred) @ beta, float(np.sqrt(np.sum(resid ** 2) / dof))

    # Backtest on the most recent points to report honest accuracy.
    hold = min(3, max(1, len(series) // 5))
    bt_pred, _ = fit_predict(t[:-hold], y[:-hold], t[-hold:], series.index[:-hold], series.index[-hold:])
    denom = np.where(y[-hold:] == 0, np.nan, np.abs(y[-hold:]))
    mape = float(np.nanmean(np.abs(y[-hold:] - bt_pred) / denom) * 100) if not np.all(np.isnan(denom)) else float("nan")

    future_idx = pd.date_range(series.index[-1], periods=periods + 1, freq=_FREQ[freq])[1:]
    t_future = np.arange(len(series), len(series) + periods, dtype=float)
    pred, sigma = fit_predict(t, y, t_future, series.index, future_idx)
    pred = np.maximum(pred, 0) if (y >= 0).all() else pred

    forecast = pd.DataFrame({"period": future_idx, "forecast": pred,
                             "lower": pred - 1.96 * sigma, "upper": pred + 1.96 * sigma})
    if (y >= 0).all():
        forecast["lower"] = forecast["lower"].clip(lower=0)
    history = series.rename("actual").reset_index().rename(columns={date_column: "period"})
    method = "Linear trend" + (" + monthly seasonality" if seasonal else "")
    notes.append(f"Method: {method.lower()} fitted on {len(series)} {freq}s; back-test MAPE on the last {hold} "
                 f"{freq}(s): {mape:.1f}%." if not np.isnan(mape) else f"Method: {method.lower()}.")
    notes.append("Intervals are +/-1.96 residual standard deviations; treat as indicative, not guaranteed.")
    return ForecastResult(history=history, forecast=forecast, method=method, notes=" ".join(notes))
