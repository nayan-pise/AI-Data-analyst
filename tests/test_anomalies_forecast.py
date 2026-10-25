import numpy as np
import pandas as pd
import pytest

from analyst.anomalies import detect_anomalies
from analyst.forecast import forecast_series


def test_finds_planted_anomalies_in_sample(store):
    res = detect_anomalies(store.frames["orders"])
    hits = res.frame[res.frame["column"] == "quantity"]
    assert (hits["value"] >= 40).sum() == 6
    rev = res.frame[res.frame["column"] == "revenue"]
    assert rev["value"].max() > 30000 and (rev["value"] < 0).sum() == 3
    assert res.by_column["quantity"] == 6


def test_explanations_are_human_readable(store):
    res = detect_anomalies(store.frames["orders"])
    reason = res.frame.iloc[0]["reason"]
    assert "robust z-scores" in reason and "typical range" in reason


def test_clean_data_has_no_flags():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(100, 5, 500)})
    res = detect_anomalies(df)
    assert res.total_flagged == 0 and "No anomalies" in res.summary


def test_single_spike_flagged_and_context_columns():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"row_id": range(200), "x": rng.normal(50, 3, 200), "city": "A"})
    df.loc[17, "x"] = 500
    res = detect_anomalies(df)
    assert res.total_flagged == 1
    row = res.frame.iloc[0]
    assert row["row_index"] == 17 and row["row_id"] == 17


def test_id_columns_ignored():
    df = pd.DataFrame({"customer_id": list(range(100)) + [10**9], "v": np.random.default_rng(2).normal(0, 1, 101)})
    assert "customer_id" not in detect_anomalies(df).scanned


def test_unknown_column_raises(store):
    with pytest.raises(KeyError):
        detect_anomalies(store.frames["orders"], columns=["nope"])


def test_period_anomaly_detected():
    idx = pd.date_range("2023-01-01", periods=730, freq="D")
    df = pd.DataFrame({"d": idx, "v": 10.0})
    df.loc[df["d"].dt.strftime("%Y-%m") == "2023-06", "v"] = 100.0
    res = detect_anomalies(df, columns=["v"], date_column="d", value_column="v")
    assert (res.frame.get("period") == "2023-06").any()


def _monthly(n=36, seasonal=True):
    idx = pd.date_range("2022-01-01", periods=n, freq="MS")
    t = np.arange(n)
    y = 1000 + 20 * t + (150 * np.sin(2 * np.pi * idx.month / 12) if seasonal else 0)
    return pd.DataFrame({"d": idx, "v": y})


def test_forecast_tracks_trend_and_seasonality():
    df = _monthly(36)
    train, test = df.iloc[:30], df.iloc[30:]
    res = forecast_series(train, "d", "v", periods=6, freq="month")
    mape = np.mean(np.abs(res.forecast["forecast"].values - test["v"].values) / test["v"].values)
    assert mape < 0.02
    assert "seasonality" in res.method and (res.forecast["lower"] <= res.forecast["upper"]).all()


def test_forecast_requires_history_and_date_column():
    with pytest.raises(ValueError, match="at least 6"):
        forecast_series(_monthly(4), "d", "v")
    with pytest.raises(ValueError, match="not a date"):
        forecast_series(pd.DataFrame({"d": ["a"] * 10, "v": range(10)}), "d", "v")
    with pytest.raises(KeyError):
        forecast_series(_monthly(12), "zz", "v")


def test_forecast_drops_partial_last_month():
    days = pd.date_range("2022-01-01", "2023-01-03", freq="D")
    df = pd.DataFrame({"d": days, "v": 1.0})
    res = forecast_series(df, "d", "v", periods=2)
    assert "incomplete" in res.notes and res.history["period"].max() == pd.Timestamp("2022-12-01")


def test_monthly_rows_keep_their_last_month():
    res = forecast_series(_monthly(12), "d", "v", periods=2)
    assert "incomplete" not in res.notes and res.history["period"].max() == pd.Timestamp("2022-12-01")
