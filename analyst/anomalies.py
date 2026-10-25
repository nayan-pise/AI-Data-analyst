"""Explainable anomaly detection: robust z-scores + IQR fences, and period-level spikes."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_store import is_id_like


@dataclass
class AnomalyResult:
    frame: pd.DataFrame
    summary: str
    scanned: list[str] = field(default_factory=list)
    total_flagged: int = 0
    by_column: dict = field(default_factory=dict)


def _fmt(v: float) -> str:
    return f"{v:,.2f}" if abs(v) < 1e6 else f"{v:,.0f}"


def _context_columns(df: pd.DataFrame, exclude: set[str], limit: int = 4) -> list[str]:
    """Identifier / label columns that help a human locate the flagged row."""
    ids = [c for c in df.columns if is_id_like(c) and c not in exclude]
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) and c not in exclude]
    cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c]) and c not in exclude and c not in ids]
    return (ids[:2] + dates[:1] + cats[:2])[:limit]


def detect_anomalies(df: pd.DataFrame, columns: list[str] | None = None, *,
                     date_column: str | None = None, value_column: str | None = None,
                     z_threshold: float = 3.5, iqr_k: float = 3.0, max_results: int = 25) -> AnomalyResult:
    """Flag unusual numeric values (row level) and unusual periods (if a date column is given).

    Row level: a value is flagged only if BOTH its modified z-score (median/MAD based) exceeds
    `z_threshold` AND it lies beyond `iqr_k` x IQR from the quartiles. Requiring both keeps false
    positives low. Columns spanning more than 10x between their 10th and 90th percentile are scored on a signed-log scale.
    """
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
               and not pd.api.types.is_bool_dtype(df[c]) and not is_id_like(c)]
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"Unknown column(s): {', '.join(missing)}. Available: {', '.join(df.columns)}")
        numeric = [c for c in columns if c in numeric] or list(columns)
    rows: list[dict] = []
    scanned: list[str] = []
    ctx_cols = _context_columns(df, set(numeric))

    for col in numeric:
        s = df[col].dropna().astype(float)
        if len(s) < 8 or s.nunique() < 5:
            continue
        scanned.append(col)
        # Decide the scale from percentiles (robust to the outliers we are hunting): a column whose
        # 10th-90th percentile range spans > 10x is multiplicative (prices, revenue) -> signed-log scale.
        p10, p90 = float(s.quantile(0.10)), float(s.quantile(0.90))
        use_log = bool(p10 > 0 and p90 / p10 > 10)
        score_s = np.sign(s) * np.log1p(s.abs()) if use_log else s
        z_thr, k = (min(z_threshold, 3.0), 1.5) if use_log else (z_threshold, iqr_k)
        med = float(score_s.median())
        mad = float((score_s - med).abs().median())
        if mad == 0:
            mad = float((score_s - med).abs().mean()) * 1.2533  # fallback: mean absolute deviation
        if mad == 0:
            continue
        modz = 0.6745 * (score_s - med) / mad
        q1, q3 = score_s.quantile(0.25), score_s.quantile(0.75)
        iqr = q3 - q1
        outside = (score_s < q1 - k * iqr) | (score_s > q3 + k * iqr)
        flagged = s.index[(modz.abs() > z_thr) & outside]
        # typical range expressed in original units for the explanation
        lo_t, hi_t = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        inv = (lambda v: float(np.sign(v) * np.expm1(abs(v)))) if use_log else float
        lo_t, hi_t, med_orig = inv(lo_t), inv(hi_t), inv(med)
        for idx in flagged:
            v = float(s.loc[idx])
            z = float(modz.loc[idx])
            direction = "above" if z > 0 else "below"
            reason = (f"{col} = {_fmt(v)} is {abs(z):.1f} robust z-scores {direction} the median "
                      f"({_fmt(med_orig)}); typical range is "
                      f"{_fmt(lo_t)} to {_fmt(hi_t)}" + (" (log scale)" if use_log else ""))
            record = {"row_index": int(idx), "column": col, "value": v, "score": round(abs(z), 2), "reason": reason}
            for c in ctx_cols:
                record[c] = df.loc[idx, c]
            rows.append(record)

    period_note = ""
    if date_column:
        value_col = value_column or (numeric[0] if numeric else None)
        if value_col and date_column in df.columns:
            period_rows, period_note = _period_anomalies(df, date_column, value_col, z_threshold - 0.5)
            rows.extend(period_rows)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("score", ascending=False).reset_index(drop=True)
    total = len(frame)
    by_column = frame.groupby("column").size().sort_values(ascending=False).to_dict() if total else {}
    frame = frame.head(max_results)
    if total:
        cols_hit = ", ".join(by_column)
        summary = (f"Flagged {total} anomalous value(s) across {len(scanned)} scanned numeric column(s) "
                   f"(hits in: {cols_hit}). {period_note}").strip()
    else:
        summary = (f"No anomalies found across {len(scanned)} scanned numeric column(s) "
                   f"using robust z-score > {z_threshold} and {iqr_k}x IQR fences. {period_note}").strip()
    return AnomalyResult(frame=frame, summary=summary, scanned=scanned, total_flagged=total, by_column=by_column)


def _period_anomalies(df: pd.DataFrame, date_col: str, value_col: str, z_thr: float) -> tuple[list[dict], str]:
    d = df[[date_col, value_col]].dropna()
    if d.empty or not pd.api.types.is_datetime64_any_dtype(d[date_col]):
        return [], f"(period check skipped: '{date_col}' is not a date column)"
    series = d.set_index(date_col)[value_col].resample("MS").sum()
    if len(series) < 6:
        return [], "(period check skipped: fewer than 6 months of data)"
    med = float(series.median())
    mad = float((series - med).abs().median()) or float((series - med).abs().mean()) * 1.2533
    if mad == 0:
        return [], ""
    out = []
    for ts, v in series.items():
        z = 0.6745 * (float(v) - med) / mad
        if abs(z) > z_thr:
            direction = "above" if z > 0 else "below"
            out.append({"row_index": -1, "column": f"{value_col} (monthly total)", "value": float(v),
                        "score": round(abs(z), 2), "period": ts.strftime("%Y-%m"),
                        "reason": f"Monthly {value_col} total {_fmt(float(v))} in {ts:%b %Y} is {abs(z):.1f} robust "
                                  f"z-scores {direction} the typical month ({_fmt(med)})"})
    return out, f"Monthly totals of {value_col} were also checked for unusual periods."
