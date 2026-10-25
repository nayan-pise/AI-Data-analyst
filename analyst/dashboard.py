"""Auto-generated dashboard: KPIs, trend, top categories and distribution for the main table."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .charts import build_chart
from .data_store import DataStore
from .offline import OfflineAnalyst, _label, _n


@dataclass
class Dashboard:
    table: str
    kpis: list[tuple[str, str]] = field(default_factory=list)
    figures: list[tuple[str, object]] = field(default_factory=list)


def build_dashboard(store: DataStore, table: str | None = None) -> Dashboard:
    t = store.require_table(table)
    info, df = store.tables[t], store.frames[t]
    helper = OfflineAnalyst(store)
    measure = helper._find_measure(info, set())
    date = helper._date_col(info)
    dash = Dashboard(table=t)
    dash.kpis.append(("Rows", f"{info.rows:,}"))
    if measure:
        s = df[measure]
        dash.kpis += [(f"Total {_label(measure)}", _n(s.sum())), (f"Average {_label(measure)}", _n(s.mean())),
                      (f"Median {_label(measure)}", _n(s.median()))]
    if date:
        c = info.col(date)
        dash.kpis.append(("Period", f"{c.min:%b %Y} - {c.max:%b %Y}"))

    if measure and date:
        monthly = df.set_index(date)[measure].resample("MS").sum().reset_index()
        dash.figures.append((f"Monthly {measure}", build_chart(monthly, "line", date, measure, None, f"Monthly {_label(measure)}")))

    if measure:
        dims = [(t, c.name) for c in info.by_kind("categorical") if c.unique <= 15][:2]
        for j in store.suggest_joins():
            if j["child"] == t and len(dims) < 4:
                cand = [c for c in store.tables[j["parent"]].columns
                        if c.kind in ("categorical", "text") and not c.is_id and c.unique <= 25]
                if cand:
                    dims.append((j["parent"], cand[0].name))
        for dim_table, col in dims[:4]:
            data = df
            if dim_table != t:
                join = next(j for j in store.suggest_joins() if j["parent"] == dim_table and j["child"] == t)
                data = df.merge(store.frames[dim_table][[join["column"], col]], on=join["column"], how="inner")
            agg = data.groupby(col)[measure].sum().sort_values(ascending=False).head(10).reset_index()
            dash.figures.append((f"{measure} by {col}", build_chart(agg, "bar", col, measure, None,
                                                                    f"{_label(measure).title()} by {_label(col)}")))
        dash.figures.append((f"{measure} distribution", build_chart(df[[measure]], "histogram", measure, None, None,
                                                                    f"Distribution of {_label(measure)}")))
    nums = [c.name for c in info.by_kind("integer", "numeric")]
    if len(nums) >= 3:
        dash.figures.append(("Correlation", build_chart(df[nums], "heatmap", None, None, None, "Correlation matrix")))
    return dash
