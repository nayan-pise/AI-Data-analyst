"""CSV ingestion, validation, profiling and safe SQL access (DuckDB, in-memory)."""
from __future__ import annotations

import csv
import io
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .config import Settings, get_settings
from .logging_utils import log_event, timed
from .sql_safety import SQLSafetyError, validate_select


class DataValidationError(ValueError):
    """A user-supplied file cannot be used; the message is safe to show in the UI."""


_RESERVED = {
    "select", "from", "where", "group", "order", "by", "table", "limit", "union", "join",
    "left", "right", "inner", "outer", "on", "as", "and", "or", "not", "null", "case",
    "when", "then", "else", "end", "having", "distinct", "all", "in", "is", "like",
    "between", "exists", "primary", "key", "default", "check", "desc", "asc", "user",
}
_ID_PATTERN = re.compile(r"(^|_)(id|uuid|key)$|^id_|_id_")
_NUMERIC_STR = re.compile(r"^\s*[-+(]?\s*[$€£₹]?\s*\d[\d,]*\.?\d*\s*%?\)?\s*$|^\s*[-+(]?\s*[$€£₹]?\s*\.\d+\s*%?\)?\s*$")
_DATE_STR = re.compile(
    r"^\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|"
    r"[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{4})([ T]\d{1,2}:\d{2}(:\d{2})?)?\s*$"
)
_NEGATIVE_SUSPECT = re.compile(r"(qty|quantity|price|amount|revenue|cost|sales|total|units|count)")


def sanitize_identifier(name: object, taken: set[str] | None = None) -> str:
    """snake_case, SQL-friendly, unique identifier."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower() or "col"
    if s[0].isdigit():
        s = "c_" + s
    if s in _RESERVED:
        s += "_col"
    base, i = s, 2
    while taken is not None and s in taken:
        s = f"{base}_{i}"
        i += 1
    return s


def is_id_like(name: str) -> bool:
    return bool(_ID_PATTERN.search(name.lower()))


@dataclass
class ColumnInfo:
    name: str
    original_name: str
    kind: str                # integer | numeric | datetime | boolean | categorical | text
    nulls: int
    unique: int
    min: object = None
    max: object = None
    mean: float | None = None
    top_values: list = field(default_factory=list)
    is_id: bool = False


@dataclass
class TableInfo:
    name: str
    source: str
    rows: int
    columns: list[ColumnInfo]

    def col(self, name: str) -> ColumnInfo | None:
        return next((c for c in self.columns if c.name == name), None)

    def by_kind(self, *kinds: str, include_ids: bool = False) -> list[ColumnInfo]:
        return [c for c in self.columns if c.kind in kinds and (include_ids or not c.is_id)]


@dataclass
class LoadResult:
    table: str
    info: TableInfo
    warnings: list[str]


@dataclass
class QueryResult:
    df: pd.DataFrame
    truncated: bool
    cached: bool = False


def _fmt(v: object) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:,.2f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return str(v)


class DataStore:
    """Holds uploaded tables as DataFrames (for pandas/anomaly code) and DuckDB tables (for SQL)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.con = duckdb.connect(":memory:")
        self.con.execute("SET enable_external_access=false")  # no file/http access from SQL
        self.frames: dict[str, pd.DataFrame] = {}
        self.tables: dict[str, TableInfo] = {}
        self.version = 0
        self._cache: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
        self._lock = threading.RLock()
        self.cache_hits = 0
        self.cache_misses = 0

    # ------------------------------------------------------------------ loading
    def load_csv(self, data: bytes | str | Path | io.IOBase, filename: str | None = None) -> LoadResult:
        raw, filename = self._read_bytes(data, filename)
        if not filename.lower().endswith((".csv", ".tsv", ".txt")):
            raise DataValidationError(f"'{filename}' is not a CSV file (expected .csv, .tsv or .txt).")
        limit = self.settings.max_upload_mb * 1024 * 1024
        if len(raw) > limit:
            raise DataValidationError(f"'{filename}' is {len(raw)/1e6:.1f} MB; the limit is {self.settings.max_upload_mb} MB.")
        if not raw.strip():
            raise DataValidationError(f"'{filename}' is empty.")

        text = self._decode(raw)
        warnings: list[str] = []
        with timed("load_csv", file=filename, bytes=len(raw)):
            df = self._parse(text, filename, warnings)
            df = self._clean(df, warnings)
            table = self._unique_table_name(Path(filename).stem)
            self._register(table, df)
        info = self._profile(table, df, filename)
        self.tables[table] = info
        self.version += 1
        self._cache.clear()
        log_event("table_loaded", table=table, rows=len(df), columns=len(df.columns))
        return LoadResult(table, info, warnings)

    @staticmethod
    def _read_bytes(data, filename):
        if isinstance(data, (str, Path)):
            p = Path(data)
            if not p.exists():
                raise DataValidationError(f"File not found: {p}")
            return p.read_bytes(), filename or p.name
        if isinstance(data, (bytes, bytearray)):
            return bytes(data), filename or "upload.csv"
        if hasattr(data, "read"):
            name = filename or getattr(data, "name", "upload.csv")
            content = data.read()
            return (content.encode() if isinstance(content, str) else content), Path(name).name
        raise DataValidationError("Unsupported upload type.")

    @staticmethod
    def _decode(raw: bytes) -> str:
        for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
            try:
                return raw.decode(enc)
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise DataValidationError("Could not decode the file; please save it as UTF-8.")

    @staticmethod
    def _parse(text: str, filename: str, warnings: list[str]) -> pd.DataFrame:
        sample = text[:8192]
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = "\t" if filename.lower().endswith(".tsv") else ","
        try:
            df = pd.read_csv(io.StringIO(text), sep=delimiter, skipinitialspace=True,
                             on_bad_lines="skip", low_memory=False)
        except pd.errors.EmptyDataError as exc:
            raise DataValidationError(f"'{filename}' has no data rows or header.") from exc
        except (pd.errors.ParserError, ValueError) as exc:
            raise DataValidationError(f"'{filename}' could not be parsed as CSV: {exc}") from exc
        expected = max(text.count("\n") - 1, 0)
        if expected - len(df) > 2 and text.strip():
            warnings.append(f"About {expected - len(df)} malformed or blank lines were skipped.")
        return df

    def _clean(self, df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty or len(df.columns) == 0:
            raise DataValidationError("The file contains a header but no usable data rows.")
        taken: set[str] = set()
        renamed = {}
        for col in df.columns:
            new = sanitize_identifier(col, taken)
            taken.add(new)
            renamed[col] = new
        if any(str(k) != v for k, v in renamed.items()):
            warnings.append("Column names were normalised to snake_case for SQL compatibility.")
        df = df.rename(columns=renamed)
        for col in df.columns:
            s = df[col]
            if s.dtype == object or pd.api.types.is_string_dtype(s):
                s = s.astype("string").str.strip().replace("", pd.NA)
                s = self._coerce_numeric(col, s)
                if pd.api.types.is_string_dtype(s):
                    s = self._coerce_datetime(col, s, warnings)
                df[col] = s
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def _coerce_numeric(col: str, s: pd.Series) -> pd.Series:
        non_null = s.dropna()
        if non_null.empty or is_id_like(col) or re.search(r"(zip|postal|phone|code)", col):
            return s
        if non_null.map(lambda v: bool(_NUMERIC_STR.match(v))).mean() < 0.95:
            return s
        if non_null.str.match(r"^0\d").any():   # leading zeros -> identifier, keep text
            return s
        cleaned = (s.str.replace(r"[,$€£₹%\s]", "", regex=True)
                     .str.replace(r"^\((.*)\)$", r"-\1", regex=True))
        return pd.to_numeric(cleaned, errors="coerce")

    @staticmethod
    def _coerce_datetime(col: str, s: pd.Series, warnings: list[str]) -> pd.Series:
        non_null = s.dropna()
        if non_null.empty:
            return s
        probe = non_null.head(200)
        if probe.map(lambda v: bool(_DATE_STR.match(v))).mean() < 0.9:
            return s
        parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        bad = int(parsed.isna().sum() - s.isna().sum())
        if bad > 0:
            warnings.append(f"{bad} value(s) in '{col}' were not valid dates and were set to missing.")
        return parsed

    def _unique_table_name(self, stem: str) -> str:
        return sanitize_identifier(stem)

    def _register(self, table: str, df: pd.DataFrame) -> None:
        with self._lock:
            self.con.register("_incoming", df)
            try:
                self.con.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM _incoming')
            finally:
                self.con.unregister("_incoming")
            self.frames[table] = df

    def drop_table(self, table: str) -> None:
        with self._lock:
            self.con.execute(f'DROP TABLE IF EXISTS "{table}"')
            self.frames.pop(table, None)
            self.tables.pop(table, None)
            self.version += 1
            self._cache.clear()

    # ---------------------------------------------------------------- profiling
    def _profile(self, table: str, df: pd.DataFrame, source: str) -> TableInfo:
        cols = []
        for name in df.columns:
            s = df[name]
            if pd.api.types.is_bool_dtype(s):
                kind = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(s):
                kind = "datetime"
            elif pd.api.types.is_integer_dtype(s):
                kind = "integer"
            elif pd.api.types.is_numeric_dtype(s):
                kind = "numeric"
            else:
                nun = s.nunique(dropna=True)
                kind = "categorical" if nun <= 50 or nun / max(len(s), 1) <= 0.2 else "text"
            info = ColumnInfo(name=name, original_name=name, kind=kind, nulls=int(s.isna().sum()),
                              unique=int(s.nunique(dropna=True)), is_id=is_id_like(name))
            if kind in ("integer", "numeric"):
                info.min, info.max, info.mean = s.min(), s.max(), float(s.mean())
            elif kind == "datetime":
                info.min, info.max = s.min(), s.max()
            elif kind in ("categorical", "text", "boolean"):
                info.top_values = list(s.value_counts().head(6).index.astype(str))
            cols.append(info)
        return TableInfo(name=table, source=source, rows=len(df), columns=cols)

    def schema_text(self) -> str:
        """Compact schema description injected into the LLM system prompt."""
        if not self.tables:
            return "No tables loaded."
        out = []
        for t in self.tables.values():
            out.append(f"TABLE {t.name} ({t.rows:,} rows, from {t.source})")
            for c in t.columns:
                line = f"  - {c.name} [{c.kind}{', id' if c.is_id else ''}]"
                if c.kind in ("integer", "numeric"):
                    line += f" min={_fmt(c.min)} max={_fmt(c.max)} mean={_fmt(c.mean)}"
                elif c.kind == "datetime":
                    line += f" range {_fmt(c.min)} to {_fmt(c.max)}"
                elif c.top_values:
                    line += " e.g. " + ", ".join(c.top_values[:5])
                if c.nulls:
                    line += f" (nulls={c.nulls})"
                out.append(line)
        joins = self.suggest_joins()
        if joins:
            out.append("LIKELY JOINS:")
            out += [f"  - {j['child']}.{j['column']} = {j['parent']}.{j['column']} "
                    f"({j['match_pct']:.0f}% of child values match)" for j in joins]
        return "\n".join(out)

    def suggest_joins(self) -> list[dict]:
        """Find shared key columns where one side is unique (a likely primary key)."""
        found = []
        names = list(self.frames)
        for parent in names:
            for child in names:
                if parent == child:
                    continue
                for col in set(self.frames[parent].columns) & set(self.frames[child].columns):
                    pv, cv = self.frames[parent][col].dropna(), self.frames[child][col].dropna()
                    if pv.empty or cv.empty or pv.nunique() != len(pv):
                        continue
                    if not (is_id_like(col) or pv.nunique() == len(pv) and pv.dtype != float):
                        continue
                    pct = 100 * cv.isin(set(pv)).mean()
                    if pct >= 50:
                        found.append({"parent": parent, "child": child, "column": col, "match_pct": pct})
        return found

    # ------------------------------------------------------------ data quality
    def quality_report(self, table: str | None = None) -> list[dict]:
        """List data-quality issues as dicts with severity/table/column/issue/detail."""
        issues: list[dict] = []
        targets = [table] if table else list(self.frames)
        for t in targets:
            if t not in self.frames:
                raise KeyError(f"Unknown table '{t}'. Available: {', '.join(self.frames) or 'none'}")
            df, info = self.frames[t], self.tables[t]

            def add(sev, col, issue, detail):
                issues.append({"severity": sev, "table": t, "column": col, "issue": issue, "detail": detail})

            dups = int(df.duplicated().sum())
            if dups:
                add("warning", "", "Duplicate rows", f"{dups} fully duplicated rows ({dups/len(df):.1%}).")
            for c in info.columns:
                if c.nulls:
                    pct = c.nulls / len(df)
                    add("warning" if pct > 0.05 else "info", c.name, "Missing values",
                        f"{c.nulls} missing ({pct:.1%}).")
                if c.unique <= 1 and len(df) > 1:
                    add("info", c.name, "Constant column", "Only one distinct value; it carries no signal.")
                if c.kind in ("integer", "numeric") and not c.is_id:
                    if _NEGATIVE_SUSPECT.search(c.name) and (df[c.name] < 0).any():
                        add("warning", c.name, "Negative values",
                            f"{int((df[c.name] < 0).sum())} negative value(s) in a column that is normally non-negative.")
                if c.kind in ("categorical", "text"):
                    s = df[c.name].dropna().astype(str)
                    if s.nunique() > s.str.strip().str.lower().nunique():
                        add("warning", c.name, "Inconsistent labels",
                            "Values differ only by case/spacing (e.g. 'North' vs 'north').")
                if c.kind == "datetime":
                    future = int((df[c.name] > pd.Timestamp.now()).sum())
                    if future:
                        add("info", c.name, "Future dates", f"{future} date(s) are in the future.")
        if not table:
            for j in self.suggest_joins():
                if j["match_pct"] < 100:
                    child = self.frames[j["child"]][j["column"]].dropna()
                    orphans = int((~child.isin(set(self.frames[j["parent"]][j["column"]]))).sum())
                    issues.append({"severity": "warning", "table": j["child"], "column": j["column"],
                                   "issue": "Orphaned keys",
                                   "detail": f"{orphans} row(s) reference values missing from {j['parent']}.{j['column']}."})
        return issues

    # --------------------------------------------------------------------- SQL
    def query(self, sql: str, max_rows: int | None = None) -> QueryResult:
        """Run a validated read-only query. Results are cached per data version."""
        max_rows = max_rows or self.settings.max_rows_returned
        cleaned = validate_select(sql)
        key = (self.version, cleaned, max_rows)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.cache_hits += 1
                df = self._cache[key]
                return QueryResult(df.copy(), len(df) > max_rows, cached=True)
            self.cache_misses += 1
            timer = threading.Timer(self.settings.query_timeout_s, self.con.interrupt)
            timer.start()
            try:
                with timed("sql_query", sql=cleaned[:300]):
                    df = self.con.execute(f"SELECT * FROM ({cleaned}) LIMIT {max_rows + 1}").df()
            except duckdb.InterruptException as exc:
                raise SQLSafetyError(f"Query exceeded the {self.settings.query_timeout_s}s time limit.") from exc
            except duckdb.Error as exc:
                raise SQLSafetyError(f"SQL error: {exc}") from exc
            finally:
                timer.cancel()
            self._cache[key] = df
            while len(self._cache) > self.settings.cache_size:
                self._cache.popitem(last=False)
        truncated = len(df) > max_rows
        return QueryResult(df.head(max_rows).copy(), truncated)

    def table_names(self) -> list[str]:
        return list(self.tables)

    def require_table(self, name: str | None) -> str:
        """Resolve a table name (case-insensitive); defaults to the only/largest table."""
        if not self.frames:
            raise KeyError("No data loaded. Upload a CSV first.")
        if not name:
            return max(self.frames, key=lambda t: len(self.frames[t]))
        key = sanitize_identifier(name)
        if key in self.frames:
            return key
        raise KeyError(f"Unknown table '{name}'. Available: {', '.join(self.frames)}")
