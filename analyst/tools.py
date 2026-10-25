"""Tool definitions (JSON schema for the LLM) and the executor that runs them safely."""
from __future__ import annotations

import ast
import concurrent.futures
import json
from typing import Any

import numpy as np
import pandas as pd

from .anomalies import detect_anomalies
from .charts import CHART_TYPES, ChartError, build_chart, forecast_figure
from .data_store import DataStore
from .forecast import forecast_series
from .logging_utils import timed
from .models import TurnArtifacts
from .sql_safety import SQLSafetyError

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "run_sql",
        "description": ("Run a read-only DuckDB SQL query over the loaded tables and return the rows. "
                        "Use for aggregations, rankings, filters, joins and time series. Always prefer this "
                        "over guessing numbers. Quote nothing unless needed; table/column names are snake_case."),
        "input_schema": {"type": "object", "properties": {
            "sql": {"type": "string", "description": "A single SELECT/WITH statement (DuckDB dialect)."},
            "purpose": {"type": "string", "description": "One short sentence: what this query answers."}},
            "required": ["sql", "purpose"]},
    },
    {
        "name": "run_pandas",
        "description": ("Run pandas code when SQL is awkward (e.g. rolling windows, pivots, custom stats). Every "
                        "loaded table is a DataFrame variable named after the table; `pd` and `np` are available. "
                        "Assign the final output to a variable named `result`. No imports or file access."),
        "input_schema": {"type": "object", "properties": {
            "code": {"type": "string"}, "purpose": {"type": "string"}}, "required": ["code", "purpose"]},
    },
    {
        "name": "create_chart",
        "description": ("Create an interactive chart. Provide a SQL query that returns the already-aggregated "
                        "data, then name the columns for the axes. Use bar for rankings/categories, line for "
                        "time trends, pie for shares (<=8 slices), scatter for relationships, histogram for "
                        "distributions, box for spread, heatmap for matrices."),
        "input_schema": {"type": "object", "properties": {
            "sql": {"type": "string"}, "chart_type": {"type": "string", "enum": CHART_TYPES},
            "x": {"type": "string"}, "y": {"type": "string"}, "color": {"type": "string"},
            "title": {"type": "string"}}, "required": ["sql", "chart_type", "title"]},
    },
    {
        "name": "detect_anomalies",
        "description": ("Detect unusual values with explainable statistics (robust z-score + IQR). Returns each "
                        "flagged row with the reason. Optionally also checks monthly totals for spikes."),
        "input_schema": {"type": "object", "properties": {
            "table": {"type": "string"},
            "columns": {"type": "array", "items": {"type": "string"}, "description": "Numeric columns; default all."},
            "date_column": {"type": "string", "description": "Optional; enables unusual-month detection."},
            "value_column": {"type": "string", "description": "Measure for the monthly check."}},
            "required": ["table"]},
    },
    {
        "name": "data_quality_report",
        "description": "List data-quality issues: missing values, duplicates, negatives, inconsistent labels, orphaned keys.",
        "input_schema": {"type": "object", "properties": {"table": {"type": "string", "description": "Omit for all tables."}}},
    },
    {
        "name": "forecast",
        "description": "Forecast a measure over time (linear trend + seasonality when enough history) with intervals and a back-test error.",
        "input_schema": {"type": "object", "properties": {
            "table": {"type": "string"}, "date_column": {"type": "string"}, "value_column": {"type": "string"},
            "periods": {"type": "integer", "description": "Periods ahead (default 6)."},
            "freq": {"type": "string", "enum": ["day", "week", "month"]},
            "agg": {"type": "string", "enum": ["sum", "mean", "count"]}},
            "required": ["table", "date_column", "value_column"]},
    },
    {
        "name": "describe_table",
        "description": "Statistical profile of one table: describe() for numerics and top values for categoricals.",
        "input_schema": {"type": "object", "properties": {"table": {"type": "string"}}, "required": ["table"]},
    },
]

# ----------------------------------------------------------------------- pandas sandbox
_BLOCKED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars",
                  "getattr", "setattr", "delattr", "os", "sys", "subprocess", "importlib", "builtins", "breakpoint"}
_BLOCKED_ATTRS = {"to_csv", "to_excel", "to_pickle", "to_parquet", "to_json", "to_sql", "to_feather", "to_hdf",
                  "to_clipboard", "to_stata", "to_orc", "to_xml", "to_latex", "eval", "system", "popen"}
_SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                  for n in ("len", "range", "min", "max", "sum", "sorted", "abs", "round", "list", "dict", "set",
                            "tuple", "enumerate", "zip", "float", "int", "str", "bool", "any", "all", "map",
                            "filter", "reversed", "isinstance", "pow", "divmod")}


class PandasCodeError(ValueError):
    pass


def validate_pandas_code(code: str) -> ast.Module:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise PandasCodeError(f"Syntax error: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise PandasCodeError("Imports are not allowed; `pd` and `np` are already available.")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in _BLOCKED_ATTRS
                                                 or node.attr.startswith("read_")):
            raise PandasCodeError(f"Attribute '{node.attr}' is not allowed.")
        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            raise PandasCodeError(f"Name '{node.id}' is not allowed.")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.ClassDef)):
            raise PandasCodeError("Only plain expressions and assignments are allowed.")
    return tree


def run_pandas_code(code: str, frames: dict[str, pd.DataFrame], timeout: int = 20) -> Any:
    tree = validate_pandas_code(code)
    namespace: dict[str, Any] = {"pd": pd, "np": np, "__builtins__": _SAFE_BUILTINS}
    namespace.update({name: df.copy() for name, df in frames.items()})
    compiled = compile(tree, "<analysis>", "exec")

    def _run():
        exec(compiled, namespace)  # noqa: S102 - sandboxed by AST validation + restricted builtins
        return namespace.get("result")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            raise PandasCodeError(f"Code exceeded the {timeout}s time limit.") from exc


def _to_frame(obj: Any) -> pd.DataFrame:
    if obj is None:
        raise PandasCodeError("Assign your final output to a variable named `result`.")
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index() if not isinstance(obj.index, pd.RangeIndex) else obj
    if isinstance(obj, pd.Series):
        return obj.reset_index()
    if isinstance(obj, dict):
        return pd.DataFrame([obj]) if all(np.isscalar(v) for v in obj.values()) else pd.DataFrame(obj)
    if isinstance(obj, (list, tuple)):
        return pd.DataFrame(obj)
    return pd.DataFrame({"result": [obj]})


class ToolExecutor:
    """Executes one tool call, records artifacts for the UI and returns compact text for the LLM."""

    def __init__(self, store: DataStore, artifacts: TurnArtifacts) -> None:
        self.store = store
        self.art = artifacts
        self.last_df: pd.DataFrame | None = None
        self.limit = store.settings.max_rows_to_llm

    # -- public
    def execute(self, name: str, args: dict) -> str:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return f"ERROR: unknown tool '{name}'."
        try:
            with timed(f"tool.{name}"):
                return handler(**(args or {}))
        except (SQLSafetyError, PandasCodeError, ChartError, KeyError, ValueError, TypeError) as exc:
            msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            self.art.add_step(f"{name} failed: {msg}")
            return f"ERROR: {msg}. Fix the problem and try again."

    # -- helpers
    def _df_for_llm(self, df: pd.DataFrame, truncated: bool = False) -> str:
        shown = df.head(self.limit)
        text = shown.to_csv(index=False, float_format="%.4g")
        note = f"\n[{len(df)} row(s) returned" + (", results truncated at the display limit" if truncated else "")
        if len(df) > self.limit:
            note += f"; first {self.limit} shown"
        return text + note + "]"

    # -- tools
    def _tool_run_sql(self, sql: str, purpose: str = "") -> str:
        res = self.store.query(sql)
        self.last_df = res.df
        self.art.add_code("sql", sql.strip(), purpose, len(res.df))
        self.art.add_table(purpose or "Query result", res.df)
        self.art.add_step(f"Ran SQL: {purpose or 'query'} ({len(res.df)} rows{', cached' if res.cached else ''}).")
        return self._df_for_llm(res.df, res.truncated)

    def _tool_run_pandas(self, code: str, purpose: str = "") -> str:
        result = _to_frame(run_pandas_code(code, self.store.frames))
        result = result.head(self.store.settings.max_rows_returned)
        self.last_df = result
        self.art.add_code("python", code.strip(), purpose, len(result))
        self.art.add_table(purpose or "Pandas result", result)
        self.art.add_step(f"Ran pandas: {purpose or 'analysis'} ({len(result)} rows).")
        return self._df_for_llm(result)

    def _tool_create_chart(self, sql: str, chart_type: str, title: str, x: str | None = None,
                           y: str | None = None, color: str | None = None) -> str:
        res = self.store.query(sql)
        fig = build_chart(res.df, chart_type, x, y, color, title)
        self.last_df = res.df
        self.art.add_code("sql", sql.strip(), f"Data for chart: {title}", len(res.df))
        self.art.add_figure(title, fig)
        self.art.add_step(f"Built a {chart_type} chart: {title}.")
        return (f"Chart '{title}' ({chart_type}) was created and is displayed to the user. "
                f"Data preview:\n{self._df_for_llm(res.df.head(15))}")

    def _tool_detect_anomalies(self, table: str, columns: list[str] | None = None,
                               date_column: str | None = None, value_column: str | None = None) -> str:
        t = self.store.require_table(table)
        result = detect_anomalies(self.store.frames[t], columns, date_column=date_column, value_column=value_column)
        self.art.add_step(f"Scanned {len(result.scanned)} numeric columns of '{t}' for anomalies "
                          f"({result.total_flagged} flagged).")
        if not result.frame.empty:
            self.art.add_table(f"Anomalies in {t}", result.frame.drop(columns=["reason"], errors="ignore"))
        self.last_df = result.frame
        body = result.frame.head(15)[["row_index", "column", "value", "reason"]].to_csv(index=False) \
            if not result.frame.empty else "(none)"
        return f"{result.summary}\n{body}"

    def _tool_data_quality_report(self, table: str | None = None) -> str:
        t = self.store.require_table(table) if table else None
        issues = self.store.quality_report(t)
        self.art.add_step(f"Ran data-quality checks ({len(issues)} issue(s) found).")
        if issues:
            self.art.add_table("Data quality issues", pd.DataFrame(issues))
        return "\n".join(f"[{i['severity']}] {i['table']}.{i['column'] or '*'}: {i['issue']} - {i['detail']}"
                         for i in issues) or "No data-quality issues found."

    def _tool_forecast(self, table: str, date_column: str, value_column: str, periods: int = 6,
                       freq: str = "month", agg: str = "sum") -> str:
        t = self.store.require_table(table)
        res = forecast_series(self.store.frames[t], date_column, value_column, periods, freq, agg)
        fig = forecast_figure(res.history, res.forecast, f"{value_column} forecast ({res.method})", value_column)
        self.art.add_figure(f"{value_column} forecast", fig)
        self.art.add_table("Forecast", res.forecast.round({"forecast": 2, "lower": 2, "upper": 2}))
        self.art.add_step(f"Forecast {periods} {freq}(s) of {value_column} with {res.method.lower()}.")
        return res.forecast.round({"forecast": 1, "lower": 1, "upper": 1}).to_csv(index=False) + "\n" + res.notes

    def _tool_describe_table(self, table: str) -> str:
        t = self.store.require_table(table)
        df = self.store.frames[t]
        parts = [df.describe(include="number").round(2).to_csv()] if not df.select_dtypes("number").empty else []
        for c in self.store.tables[t].by_kind("categorical", include_ids=True):
            parts.append(f"{c.name} top values: " + json.dumps(df[c.name].value_counts().head(8).to_dict(), default=str))
        self.art.add_step(f"Profiled table '{t}'.")
        return "\n".join(parts)
