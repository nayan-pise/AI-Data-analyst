"""Deterministic natural-language analyst used when no API key is set or the LLM is unavailable.

It understands the common analytical intents (ranking, trends, shares, totals, anomalies, quality,
forecast, summaries, SQL/pandas requests) by mapping the question onto the real schema. It is not a
replacement for the LLM agent but makes the app fully usable (and testable) offline, and keeps
follow-up context ("now by product", "show the SQL").
"""
from __future__ import annotations

import re
import time

import pandas as pd

from .agent import ConversationMemory
from .anomalies import detect_anomalies
from .charts import build_chart, forecast_figure
from .data_store import DataStore, TableInfo
from .forecast import forecast_series
from .models import AgentResponse, TurnArtifacts
from .sql_safety import SQLSafetyError

_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_SYNONYMS = {"sales": "revenue", "income": "revenue", "turnover": "revenue", "earnings": "revenue",
             "spend": "revenue", "qty": "quantity", "units": "quantity", "value": "revenue"}
_DEFAULT_MEASURES = ["revenue", "sales", "amount", "total", "profit", "price", "cost", "quantity"]
_STOP_DIM_TOKENS = {"id", "name", "date", "time"}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = set(words)
    for w in words:
        if w.endswith("ies") and len(w) > 4:
            out.add(w[:-3] + "y")
        elif w.endswith("s") and len(w) > 3:
            out.add(w[:-1])
        if w in _SYNONYMS:
            out.add(_SYNONYMS[w])
    return out


def _n(v: float) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}"


def _label(col: str) -> str:
    return re.sub(r" name$", "", col.replace("_", " "))


class OfflineAnalyst:
    """Rule-based analyst with per-session follow-up context."""

    def __init__(self, store: DataStore) -> None:
        self.store = store
        self.ctx: dict = {}

    # ----------------------------------------------------------------- entry
    def run(self, question: str, memory: ConversationMemory | None = None) -> AgentResponse:
        start = time.perf_counter()
        art = TurnArtifacts()
        warnings: list[str] = []
        if not self.store.tables:
            return AgentResponse("Please upload at least one CSV file (or load the sample data) first.", art, "offline")
        try:
            text = self._dispatch(question, art)
        except (SQLSafetyError, KeyError, ValueError) as exc:
            msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            text = (f"**I couldn't complete that analysis:** {msg}\n\nTry rephrasing, or ask for a summary "
                    f"of the data to see what is available.")
            warnings.append(str(msg))
        return AgentResponse(text, art, "offline", time.perf_counter() - start, warnings=warnings)

    # --------------------------------------------------------------- routing
    def _dispatch(self, question: str, art: TurnArtifacts) -> str:
        q = question.lower().strip()
        qt = _tokens(q)
        fact = self._fact_table(qt)
        info = self.store.tables[fact]
        wants_code = re.search(r"\b(sql|pandas|code|query)\b", q) and re.search(r"\b(generate|write|show|give|provide|see|need|what|how|for)\b", q)

        if re.search(r"\b(hi|hello|hey|help|what can you do)\b", q) and len(q.split()) <= 5:
            return self._help()
        if wants_code and not re.search(r"\b(anomal|outlier)\b", q):
            return self._show_code(q, art)
        if re.search(r"summar|insight|overview|describe|tell me about|key (finding|metric|takeaway)|analy[sz]e (the|my|this)|explore|profile", q):
            return self._insights(fact, info, art)
        if re.search(r"anomal|outlier|unusual|suspicious|abnormal|spike", q):
            return self._anomalies(fact, info, art)
        if re.search(r"quality|missing|duplicate|clean|null|integrity|orphan|invalid", q):
            return self._quality(art)
        if re.search(r"forecast|predict|projection|expected|next (\d+|\w+)? ?(month|week|day|quarter)", q):
            return self._forecast(q, fact, info, qt, art)
        if re.search(r"correlat|relationship between|scatter| vs |versus", q):
            return self._correlation(q, fact, info, qt, art)
        if re.search(r"distribution|histogram|spread|variation", q):
            return self._distribution(fact, info, qt, art)
        if re.search(r"trend|over time|monthly|weekly|daily|yearly|quarterly|per month|by month|by week|by year|seasonal|growth", q):
            return self._trend(q, fact, info, qt, art)
        if re.search(r"under ?perform|lowest|worst|bottom|least|weakest|poor|declin|struggl", q):
            return self._rank(q, fact, info, qt, art, ascending=True)
        if re.search(r"\b(share|breakdown|proportion|split|percentage|pie)\b|contribut", q):
            return self._rank(q, fact, info, qt, art, ascending=False, chart="pie")
        if re.search(r"\b(how many|number of|count)\b", q):
            return self._count(q, fact, info, qt, art)
        if re.search(r"\b(top|highest|best|most|biggest|largest|leading|which|who|rank|ranking)\b", q):
            return self._rank(q, fact, info, qt, art, ascending=False)
        if re.search(r"\b(total|sum|average|mean|avg|overall)\b", q):
            return self._aggregate(q, fact, info, qt, art)
        if re.search(r"\b(by|per|each|across|for every)\b", q) and self._find_dimension(qt, fact):
            return self._rank(q, fact, info, qt, art, ascending=False)
        if self.ctx and (self._find_dimension(qt, fact) or self._find_measure(info, qt, explicit=True)
                         or re.search(r"\b(20\d\d)\b", q)):
            return self._rerun_with_context(q, fact, info, qt, art)
        return self._help(unrecognised=True)

    # ------------------------------------------------------------- resolving
    def _fact_table(self, qt: set[str]) -> str:
        """The largest table is treated as the fact table (orders, transactions, events...)."""
        return max(self.store.tables, key=lambda t: self.store.tables[t].rows)

    def _find_measure(self, info: TableInfo, qt: set[str], explicit: bool = False) -> str | None:
        numeric = info.by_kind("integer", "numeric")
        for c in numeric:
            if set(c.name.split("_")) & qt:
                return c.name
        if explicit:
            return None
        for pref in _DEFAULT_MEASURES:
            for c in numeric:
                if pref in c.name:
                    return c.name
        return numeric[0].name if numeric else None

    def _find_dimension(self, qt: set[str], fact: str, exclude: set[str] | None = None) -> tuple[str, str] | None:
        """Pick the categorical column the question refers to (may live in a joined table)."""
        fact_rows = self.store.tables[fact].rows
        cands = []   # (score, is_attribute, table, column)
        for t, info in self.store.tables.items():
            for c in info.columns:
                if c.kind in ("integer", "numeric", "datetime", "boolean") and not c.is_id:
                    continue
                if exclude and c.name in exclude:
                    continue
                toks = set(c.name.split("_")) - _STOP_DIM_TOKENS
                if not toks or not (toks & qt):
                    continue
                if t == fact and c.unique >= 0.9 * fact_rows:      # row identifier, not a dimension
                    continue
                is_attr = not c.is_id and not c.name.endswith("_name")
                score = len(toks & qt) / len(toks) + (0.5 if c.name.endswith("_name") else 0) \
                    + (0.0 if c.is_id else 0.25) + (0.1 if t == fact else 0)
                cands.append((score, is_attr, t, c.name))
        if not cands:
            return None
        if any(a for _, a, _, _ in cands):   # "customer segment" -> segment, not customer_name
            cands = [c for c in cands if c[1]]
        _, _, t, name = max(cands)
        return t, name

    def _date_col(self, info: TableInfo) -> str | None:
        cols = info.by_kind("datetime")
        return cols[0].name if cols else None

    def _join_from(self, fact: str, dim: tuple[str, str] | None) -> tuple[str, str | None]:
        if dim is None or dim[0] == fact:
            return f"{fact} f", (f"f.{dim[1]}" if dim else None)
        join = next((j for j in self.store.suggest_joins() if j["parent"] == dim[0] and j["child"] == fact), None)
        if not join:
            raise ValueError(f"I can't relate '{dim[1]}' (table {dim[0]}) to '{fact}': no shared key column found.")
        return f"{fact} f JOIN {dim[0]} d ON f.{join['column']} = d.{join['column']}", f"d.{dim[1]}"

    def _filters(self, q: str, fact: str, info: TableInfo) -> tuple[list[str], list[str]]:
        where, notes = [], []
        date = self._date_col(info)
        years = sorted(set(re.findall(r"\b(20\d\d)\b", q)))
        if years and date:
            where.append("year(f.{d}) IN ({y})".format(d=date, y=", ".join(years)))
            notes.append("limited to " + " and ".join(years))
        if re.search(r"(exclud|without|ignore|except|not)\w* ?(the )?cancel", q) or "completed only" in q:
            for c in info.by_kind("categorical"):
                vals = self.store.frames[fact][c.name].dropna().astype(str)
                hit = next((v for v in vals.unique() if v.lower() == "cancelled"), None)
                if hit:
                    where.append(f"f.{c.name} <> '{hit}'")
                    notes.append(f"excluding {hit.lower()} rows")
                    break
        return where, notes

    @staticmethod
    def _top_n(q: str, default: int) -> int:
        m = re.search(r"(?:top|bottom|best|worst|lowest|highest)\s+(\d+|" + "|".join(_NUM_WORDS) + r")\b", q)
        if m:
            val = m.group(1)
            return int(val) if val.isdigit() else _NUM_WORDS[val]
        return default

    def _run(self, art: TurnArtifacts, sql: str, purpose: str) -> pd.DataFrame:
        res = self.store.query(sql)
        art.add_code("sql", sql, purpose, len(res.df))
        art.add_table(purpose, res.df)
        art.add_step(f"Ran SQL: {purpose} ({len(res.df)} rows).")
        self.ctx["last_sql"] = sql
        return res.df

    def _caveat(self, fact: str, measure: str) -> str:
        s = self.store.frames[fact][measure]
        neg = int((s < 0).sum())
        return f" Note: {neg} row(s) have negative `{measure}` (likely returns) and are included." if neg else ""

    @staticmethod
    def _compose(answer: str, insights: list[str], how: str) -> str:
        parts = [f"**Answer:** {answer}"]
        if insights:
            parts.append("**Insights:**\n" + "\n".join(f"- {i}" for i in insights))
        parts.append(f"**How I got this:** {how}")
        return "\n\n".join(parts)

    # -------------------------------------------------------------- intents
    def _rank(self, q, fact, info, qt, art, ascending: bool, chart: str = "bar") -> str:
        measure = self._find_measure(info, qt)
        dim = self._find_dimension(qt, fact)
        if not measure or dim is None:
            raise ValueError("I couldn't tell which measure or category to rank. Try e.g. 'top 5 products by revenue'.")
        agg = "AVG" if re.search(r"average|avg|mean", q) else "SUM"
        where, notes = self._filters(q, fact, info)
        frm, dcol = self._join_from(fact, dim)
        n = self._top_n(q, 10 if not ascending else 5)
        date = self._date_col(info)
        wh = f"WHERE {' AND '.join(where)}" if where else ""
        art.add_step(f"Interpreted as: rank `{dim[1]}` by {agg.lower()} of `{measure}` ({'lowest' if ascending else 'highest'} first).")

        if ascending and date and agg == "SUM":
            sql = (f"WITH base AS (SELECT {dcol} AS dim, f.{date} AS dt, f.{measure} AS m FROM {frm} {wh}),\n"
                   f"mx AS (SELECT max(dt) AS mx FROM base)\n"
                   f"SELECT dim AS {dim[1]}, ROUND(SUM(m), 2) AS total_{measure},\n"
                   f"  ROUND(SUM(CASE WHEN dt > mx - INTERVAL 1 YEAR THEN m ELSE 0 END), 2) AS last_12m,\n"
                   f"  ROUND(SUM(CASE WHEN dt <= mx - INTERVAL 1 YEAR AND dt > mx - INTERVAL 2 YEAR THEN m ELSE 0 END), 2) AS prior_12m,\n"
                   f"  ROUND((SUM(CASE WHEN dt > mx - INTERVAL 1 YEAR THEN m ELSE 0 END) /\n"
                   f"    NULLIF(SUM(CASE WHEN dt <= mx - INTERVAL 1 YEAR AND dt > mx - INTERVAL 2 YEAR THEN m ELSE 0 END), 0) - 1) * 100, 1) AS change_pct\n"
                   f"FROM base, mx GROUP BY dim ORDER BY total_{measure} ASC LIMIT {n}")
            df = self._run(art, sql, f"Lowest {n} {_label(dim[1])} by {_label(measure)}, with year-over-year change")
            self.ctx.update(intent="rank", spec=dict(fact=fact, dim=dim, measure=measure, agg=agg, asc=True, n=n, where=where))
            low = df.iloc[0]
            declining = df[df["change_pct"].fillna(0) < 0]
            insights = [f"{r[dim[1]]}: total {_n(r['total_' + measure])}, last-12-month change "
                        f"{'n/a' if pd.isna(r['change_pct']) else f'{r.change_pct:+.1f}%'}" for _, r in df.head(5).iterrows()]
            if not declining.empty:
                insights.append(f"{len(declining)} of these {len(df)} also declined year-over-year, so they are shrinking as well as small.")
            art.add_figure("Underperformers", build_chart(df, "bar", dim[1], f"total_{measure}", None,
                                                          f"Lowest {_label(dim[1])} by {_label(measure)}"))
            how = (f"I totalled `{measure}` per `{dim[1]}`, ranked ascending, and compared the latest 12 months in the data "
                   f"with the 12 before to see whether each is declining." + self._caveat(fact, measure))
            return self._compose(f"The weakest {_label(dim[1])} is **{low[dim[1]]}** with {_n(low['total_' + measure])} "
                                 f"in total {_label(measure)}.", insights, how)

        order = "ASC" if ascending else "DESC"
        sql = (f"SELECT {dcol} AS {dim[1]}, ROUND({agg}(f.{measure}), 2) AS {agg.lower()}_{measure}\n"
               f"FROM {frm} {wh}\nGROUP BY 1 ORDER BY 2 {order} LIMIT {n}")
        df = self._run(art, sql, f"{'Lowest' if ascending else 'Top'} {n} {_label(dim[1])} by {agg.lower()} of {_label(measure)}")
        if df.empty:
            return "No rows matched that question."
        self.ctx.update(intent="rank", spec=dict(fact=fact, dim=dim, measure=measure, agg=agg, asc=ascending, n=n, where=where))
        col = f"{agg.lower()}_{measure}"
        total_sql = f"SELECT {agg}(f.{measure}) AS t FROM {frm} {wh}"
        grand = float(self.store.query(total_sql).df["t"].iloc[0])
        top = df.iloc[0]
        share = 100 * top[col] / grand if grand and agg == "SUM" else None
        insights = [f"{i + 1}. {r[dim[1]]}: {_n(r[col])}" + (f" ({100 * r[col] / grand:.1f}% of total)" if agg == 'SUM' and grand else "")
                    for i, r in df.head(5).iterrows()]
        if len(df) > 1 and df[col].iloc[1]:
            insights.append(f"#1 leads #2 by {_n(df[col].iloc[0] - df[col].iloc[1])} "
                            f"({100 * (df[col].iloc[0] / df[col].iloc[1] - 1):.1f}% higher).")
        use_chart = "pie" if chart == "pie" else "bar"
        art.add_figure(f"{_label(dim[1])} by {_label(measure)}",
                       build_chart(df, use_chart, dim[1], col, None, f"{_label(dim[1]).title()} by {_label(measure)}"))
        scope = f" ({'; '.join(notes)})" if notes else ""
        how = (f"I {'averaged' if agg == 'AVG' else 'summed'} `{measure}` grouped by `{dim[1]}`{scope}, sorted "
               f"{'ascending' if ascending else 'descending'}." + self._caveat(fact, measure))
        label = "lowest" if ascending else "highest"
        ans = (f"**{top[dim[1]]}** has the {label} {'average' if agg == 'AVG' else 'total'} {_label(measure)}: {_n(top[col])}"
               + (f", {share:.1f}% of the overall {_n(grand)}." if share is not None else "."))
        return self._compose(ans, insights, how)

    def _trend(self, q, fact, info, qt, art) -> str:
        date = self._date_col(info)
        measure = self._find_measure(info, qt)
        if not date or not measure:
            raise ValueError("I need a date column and a numeric measure to compute a trend.")
        grain = "year" if re.search(r"year|annual", q) else "quarter" if "quarter" in q else \
            "week" if "week" in q else "day" if re.search(r"\bdaily|per day\b", q) else "month"
        agg = "AVG" if re.search(r"average|avg|mean", q) else "SUM"
        where, notes = self._filters(q, fact, info)
        wh = f"WHERE {' AND '.join(where)}" if where else ""
        dim = self._find_dimension(qt, fact)
        frm, dcol = self._join_from(fact, dim)
        extra = f", {dcol} AS {dim[1]}" if dim else ""
        group = "1, 3" if dim else "1"
        sql = (f"SELECT date_trunc('{grain}', f.{date})::DATE AS {grain}{extra}, ROUND({agg}(f.{measure}), 2) AS {agg.lower()}_{measure}\n"
               f"FROM {frm} {wh}\nGROUP BY {'1, 2' if dim else '1'} ORDER BY 1")
        df = self._run(art, sql, f"{grain.title()}ly {_label(measure)}" + (f" by {_label(dim[1])}" if dim else ""))
        col = f"{agg.lower()}_{measure}"
        self.ctx.update(intent="trend", spec=dict(fact=fact, dim=dim, measure=measure, agg=agg, grain=grain, where=where, date=date))
        art.add_figure(f"{grain}ly {measure}", build_chart(df, "line", grain, col, dim[1] if dim else None,
                                                           f"{grain.title()}ly {_label(measure)}"))
        series = df.groupby(grain)[col].sum() if dim else df.set_index(grain)[col]
        if len(series) < 2:
            return self._compose(f"Only one {grain} of data is available ({_n(series.iloc[0])}).", [], "Aggregated by date.")
        peak, trough = series.idxmax(), series.idxmin()
        complete = series.iloc[:-1] if len(series) > 3 else series
        last, prev = complete.iloc[-1], complete.iloc[-2]
        fmt = "%b %Y" if grain in ("month", "week", "day", "quarter") else "%Y"
        insights = [f"Peak: {pd.Timestamp(peak):{fmt}} ({_n(series[peak])}); lowest: {pd.Timestamp(trough):{fmt}} ({_n(series[trough])}).",
                    f"Latest full {grain} ({pd.Timestamp(complete.index[-1]):{fmt}}) was {_n(last)}, "
                    f"{100 * (last / prev - 1):+.1f}% vs the previous {grain}." if prev else ""]
        half = len(series) // 2
        if half >= 2:
            first_h, second_h = series.iloc[:half].mean(), series.iloc[half:].mean()
            insights.append(f"Average {grain}ly {_label(measure)} in the second half of the period is "
                            f"{100 * (second_h / first_h - 1):+.1f}% vs the first half.")
        if grain == "month" and len(series) >= 12:
            by_m = series.groupby(pd.to_datetime(series.index).month).mean()
            insights.append(f"Seasonality: month {int(by_m.idxmax())} is typically strongest, month {int(by_m.idxmin())} weakest.")
        how = (f"I {'averaged' if agg == 'AVG' else 'summed'} `{measure}` per {grain} using `{date}`"
               + (f" ({'; '.join(notes)})" if notes else "") + "." + self._caveat(fact, measure))
        return self._compose(f"{_label(measure).title()} {'moved' if len(series) > 1 else 'was'} from {_n(series.iloc[0])} to "
                             f"{_n(series.iloc[-1])} over {len(series)} {grain}s; total {_n(series.sum())}.",
                             [i for i in insights if i], how)

    def _count(self, q, fact, info, qt, art) -> str:
        for t, tinfo in self.store.tables.items():
            if t in qt or t.rstrip("s") in qt:
                where, notes = self._filters(q, t, tinfo) if t == fact else ([], [])
                wh = f"WHERE {' AND '.join(where)}" if where else ""
                sql = f"SELECT COUNT(*) AS {t}_count FROM {t} f {wh}"
                df = self._run(art, sql, f"Number of rows in {t}")
                scope = f" ({'; '.join(notes)})" if notes else ""
                return self._compose(f"There are **{int(df.iloc[0, 0]):,}** records in `{t}`{scope}.", [], f"Counted rows in `{t}`{scope}.")
        dim = self._find_dimension(qt, fact)
        where, notes = self._filters(q, fact, info)
        wh = f"WHERE {' AND '.join(where)}" if where else ""
        if dim and dim[0] == fact:
            sql = f"SELECT COUNT(DISTINCT f.{dim[1]}) AS distinct_{dim[1]} FROM {fact} f {wh}"
            df = self._run(art, sql, f"Distinct {_label(dim[1])}")
            return self._compose(f"There are **{int(df.iloc[0, 0]):,}** distinct {_label(dim[1])} values.", [], f"Counted distinct `{dim[1]}`.")
        sql = f"SELECT COUNT(*) AS row_count FROM {fact} f {wh}"
        df = self._run(art, sql, f"Row count of {fact}")
        return self._compose(f"There are **{int(df.iloc[0, 0]):,}** rows in `{fact}`" + (f" ({'; '.join(notes)})." if notes else "."),
                             [], f"Counted rows in `{fact}`.")

    def _aggregate(self, q, fact, info, qt, art) -> str:
        measure = self._find_measure(info, qt)
        if not measure:
            raise ValueError("I couldn't find a numeric column to aggregate.")
        dim = self._find_dimension(qt, fact)
        if dim:
            return self._rank(q, fact, info, qt, art, ascending=False)
        agg = "AVG" if re.search(r"average|avg|mean", q) else "SUM"
        where, notes = self._filters(q, fact, info)
        wh = f"WHERE {' AND '.join(where)}" if where else ""
        sql = f"SELECT ROUND({agg}(f.{measure}), 2) AS {agg.lower()}_{measure}, COUNT(*) AS rows FROM {fact} f {wh}"
        df = self._run(art, sql, f"{'Average' if agg == 'AVG' else 'Total'} {_label(measure)}")
        self.ctx.update(intent="aggregate", spec=dict(fact=fact, dim=None, measure=measure, agg=agg, where=where))
        v = df.iloc[0, 0]
        return self._compose(f"{'Average' if agg == 'AVG' else 'Total'} {_label(measure)} is **{_n(v)}** across {int(df['rows'].iloc[0]):,} rows"
                             + (f" ({'; '.join(notes)})." if notes else "."), [],
                             f"{'Averaged' if agg == 'AVG' else 'Summed'} `{measure}` in `{fact}`." + self._caveat(fact, measure))

    def _distribution(self, fact, info, qt, art) -> str:
        measure = self._find_measure(info, qt)
        if not measure:
            raise ValueError("No numeric column found for a distribution.")
        df = self._run(art, f"SELECT {measure} FROM {fact}", f"Values of {_label(measure)}")
        s = df[measure].dropna()
        art.add_figure("Distribution", build_chart(df, "histogram", measure, None, None, f"Distribution of {_label(measure)}"))
        insights = [f"Median {_n(s.median())}, mean {_n(s.mean())} (mean {'above' if s.mean() > s.median() else 'below'} median "
                    f"suggests {'right' if s.mean() > s.median() else 'left'} skew).",
                    f"Middle 50% of values lie between {_n(s.quantile(.25))} and {_n(s.quantile(.75))}; range {_n(s.min())} to {_n(s.max())}."]
        return self._compose(f"`{measure}` ranges from {_n(s.min())} to {_n(s.max())} with a median of {_n(s.median())}.",
                             insights, "Computed summary statistics and a histogram over all rows.")

    def _correlation(self, q, fact, info, qt, art) -> str:
        nums = [c.name for c in info.by_kind("integer", "numeric")]
        mentioned = [c for c in nums if set(c.split("_")) & qt]
        df = self._run(art, f"SELECT {', '.join(nums)} FROM {fact}", f"Numeric columns of {fact}")
        corr = df.corr(numeric_only=True)
        if len(mentioned) >= 2:
            a, b = mentioned[:2]
            r = corr.loc[a, b]
            art.add_figure("Relationship", build_chart(df, "scatter", a, b, None, f"{_label(a)} vs {_label(b)} (r = {r:.2f})"))
            strength = "strong" if abs(r) >= .7 else "moderate" if abs(r) >= .4 else "weak"
            return self._compose(f"`{a}` and `{b}` have a {strength} {'positive' if r > 0 else 'negative'} correlation (r = {r:.2f}).",
                                 ["Correlation does not imply causation; derived columns (e.g. revenue = quantity x price) correlate by construction."],
                                 "Pearson correlation on all rows with both values.")
        tri = corr.abs().where(pd.DataFrame(
            [[i < j for j in range(len(corr))] for i in range(len(corr))], index=corr.index, columns=corr.columns)).stack()
        a, b = tri.idxmax()
        art.add_figure("Correlation matrix", build_chart(df, "heatmap", None, None, None, "Correlation matrix"))
        return self._compose(f"The strongest relationship is between `{a}` and `{b}` (r = {corr.loc[a, b]:.2f}).", [],
                             "Pearson correlation matrix across numeric columns.")

    def _anomalies(self, fact, info, art) -> str:
        df = self.store.frames[fact]
        measure = self._find_measure(info, set())
        date = self._date_col(info)
        res = detect_anomalies(df, date_column=date, value_column=measure)
        art.add_step(f"Scanned {len(res.scanned)} numeric columns in '{fact}' with robust z-score + IQR rules.")
        code = (f"from analyst.anomalies import detect_anomalies\n"
                f"result = detect_anomalies({fact}, date_column={date!r}, value_column={measure!r})")
        art.add_code("python", code, "Anomaly detection (robust z-score AND IQR fences)", res.total_flagged)
        self.ctx.update(intent=None, last_sql=None, last_python=code)
        if res.frame.empty:
            return self._compose(res.summary, [], "A value is flagged only when both the robust z-score and IQR rules agree, to avoid false alarms.")
        art.add_table(f"Anomalies in {fact}", res.frame.drop(columns=["reason"]))
        top = res.frame.head(5)
        insights = [r["reason"] for _, r in top.iterrows()]
        insights.append("Flags by column: " + ", ".join(f"{k} ({v})" for k, v in res.by_column.items()) + ".")
        how = ("For each numeric (non-ID) column I computed a median/MAD-based modified z-score and IQR fences; a value is flagged only when "
               "BOTH say it is extreme, and multiplicative columns (P90/P10 > 10) are scored on a signed-log scale. Monthly totals are checked the same way.")
        return self._compose(res.summary, insights, how)

    def _quality(self, art) -> str:
        issues = self.store.quality_report()
        art.add_step(f"Ran data-quality checks across {len(self.store.tables)} table(s).")
        if not issues:
            return self._compose("No data-quality issues found.", [], "Checked duplicates, missing values, negatives, label consistency and key integrity.")
        art.add_table("Data quality issues", pd.DataFrame(issues))
        warn = [i for i in issues if i["severity"] == "warning"]
        insights = [f"{i['table']}.{i['column'] or '*'}: {i['issue'].lower()} - {i['detail']}" for i in (warn or issues)[:6]]
        return self._compose(f"Found {len(issues)} data-quality issue(s), {len(warn)} of them warnings.", insights,
                             "Checked duplicates, missing values, negative amounts, inconsistent labels and orphaned join keys.")

    def _forecast(self, q, fact, info, qt, art) -> str:
        date, measure = self._date_col(info), self._find_measure(info, qt)
        if not date or not measure:
            raise ValueError("Forecasting needs a date column and a numeric measure.")
        m = re.search(r"(\d+|" + "|".join(_NUM_WORDS) + r")\s*(day|week|month|quarter)", q)
        periods = 6
        freq = "month"
        if m:
            periods = int(m.group(1)) if m.group(1).isdigit() else _NUM_WORDS[m.group(1)]
            freq = m.group(2)
            if freq == "quarter":
                freq, periods = "month", periods * 3
        res = forecast_series(self.store.frames[fact], date, measure, periods, freq)
        art.add_figure("Forecast", forecast_figure(res.history, res.forecast, f"{_label(measure).title()} forecast", measure))
        art.add_table("Forecast", res.forecast.round({"forecast": 2, "lower": 2, "upper": 2}))
        code = f"from analyst.forecast import forecast_series\nresult = forecast_series({fact}, '{date}', '{measure}', periods={periods}, freq='{freq}')"
        art.add_code("python", code, res.method)
        self.ctx.update(intent=None, last_sql=None, last_python=code)
        nxt = res.forecast.iloc[0]
        tot = res.forecast["forecast"].sum()
        return self._compose(f"Projected {_label(measure)} for the next {periods} {freq}(s) totals about {_n(tot)}; "
                             f"the first period is {_n(nxt['forecast'])} (range {_n(nxt['lower'])} to {_n(nxt['upper'])}).",
                             [res.notes], f"{res.method} on {freq}ly aggregates of `{measure}`.")

    def _insights(self, fact, info, art) -> str:
        measure = self._find_measure(info, set())
        date = self._date_col(info)
        rows = info.rows
        bullets = []
        head = f"`{fact}` has {rows:,} rows and {len(info.columns)} columns"
        if date:
            c = info.col(date)
            head += f", covering {c.min:%b %Y} to {c.max:%b %Y}"
        if measure:
            tot = float(self.store.query(f"SELECT SUM({measure}) t FROM {fact}").df["t"].iloc[0])
            art.add_code("sql", f"SELECT SUM({measure}) FROM {fact}", f"Total {measure}")
            head += f"; total {_label(measure)} is {_n(tot)}"
            dims = [(fact, c.name) for c in info.by_kind("categorical") if c.unique <= 12][:2]
            for j in self.store.suggest_joins():
                if j["child"] == fact:
                    pinfo = self.store.tables[j["parent"]]
                    cand = [c for c in pinfo.columns if c.kind in ("categorical", "text") and not c.is_id and c.unique <= 40]
                    if cand:
                        dims.append((j["parent"], cand[0].name))
            for dim in dims[:4]:
                frm, dcol = self._join_from(fact, dim)
                sql = (f"SELECT {dcol} AS {dim[1]}, ROUND(SUM(f.{measure}), 2) AS total_{measure} FROM {frm} "
                       f"GROUP BY 1 ORDER BY 2 DESC")
                df = self._run(art, sql, f"{_label(measure).title()} by {_label(dim[1])}")
                if len(df) >= 2 and tot:
                    bullets.append(f"Top {_label(dim[1])}: **{df.iloc[0, 0]}** ({100 * df.iloc[0, 1] / tot:.1f}% of {_label(measure)}); "
                                   f"weakest: {df.iloc[-1, 0]} ({100 * df.iloc[-1, 1] / tot:.1f}%).")
            if date:
                tr = self._run(art, f"SELECT date_trunc('month', {date})::DATE AS month, ROUND(SUM({measure}), 2) AS total_{measure} "
                                    f"FROM {fact} GROUP BY 1 ORDER BY 1", f"Monthly {_label(measure)}")
                art.add_figure("Monthly trend", build_chart(tr, "line", "month", f"total_{measure}", None, f"Monthly {_label(measure)}"))
                if len(tr) > 3:
                    s = tr.set_index("month")[f"total_{measure}"]
                    bullets.append(f"Strongest month: {pd.Timestamp(s.idxmax()):%b %Y} ({_n(s.max())}); weakest: {pd.Timestamp(s.idxmin()):%b %Y} ({_n(s.min())}).")
        an = detect_anomalies(self.store.frames[fact], date_column=date, value_column=measure)
        if an.total_flagged:
            bullets.append(f"{an.total_flagged} anomalous value(s) were flagged; ask me to 'detect anomalies' for details.")
        issues = self.store.quality_report()
        if issues:
            bullets.append(f"{len(issues)} data-quality issue(s) found (duplicates, missing values, etc.); ask for a data quality check.")
        return self._compose(head + ".", bullets, "Profiled the largest table, broke the main measure down by its key categories and time, and ran anomaly and quality scans.")

    # ------------------------------------------------------------ code & misc
    def _show_code(self, q: str, art: TurnArtifacts) -> str:
        spec, sql = self.ctx.get("spec"), self.ctx.get("last_sql")
        if not sql and self.ctx.get("last_python"):
            code = self.ctx["last_python"]
            art.add_code("python", code, "Code for the last analysis")
            return self._compose("This analysis uses a Python routine rather than SQL; here is the call.", [],
                                 "Anomaly and forecast logic live in the `analyst` package (see README).") + f"\n\n```python\n{code}\n```"
        if not sql:
            return ("I haven't run an analysis in this session yet. Ask a question first (for example "
                    "'top 5 customers by revenue') and then I can show you the SQL or pandas code.")
        if "pandas" in q and spec:
            code = self._pandas_for(spec)
            art.add_code("python", code, "Pandas equivalent of the last analysis")
            return self._compose("Here is the pandas equivalent of the last analysis.", [], "Translated the structured analysis spec into pandas.") + f"\n\n```python\n{code}\n```"
        art.add_code("sql", sql, "SQL for the last analysis")
        return self._compose("Here is the SQL behind the last analysis (DuckDB dialect).", [],
                             "This is the exact query that produced the table and chart above.") + f"\n\n```sql\n{sql}\n```"

    def _pandas_for(self, spec: dict) -> str:
        fact, dim, measure, agg = spec["fact"], spec.get("dim"), spec["measure"], spec.get("agg", "SUM").lower()
        lines = [f"df = {fact}.copy()"]
        if dim and dim[0] != fact:
            j = next(j for j in self.store.suggest_joins() if j["parent"] == dim[0] and j["child"] == fact)
            lines.append(f"df = df.merge({dim[0]}[['{j['column']}', '{dim[1]}']], on='{j['column']}')")
        for w in spec.get("where", []):
            if w.startswith("year("):
                years = re.findall(r"20\d\d", w)
                lines.append(f"df = df[df['{spec.get('date') or self._date_col(self.store.tables[fact])}'].dt.year.isin({[int(y) for y in years]})]")
        if spec.get("intent") == "trend":
            lines.append(f"result = (df.groupby(df['{spec['date']}'].dt.to_period('{spec['grain'][0].upper()}').dt.to_timestamp())['{measure}']"
                         f".{agg}().round(2).reset_index())")
        elif dim:
            lines.append(f"result = (df.groupby('{dim[1]}')['{measure}'].{agg}().round(2)\n          .sort_values(ascending={bool(spec.get('asc'))})"
                         f".head({spec.get('n', 10)}).reset_index())")
        else:
            lines.append(f"result = df['{measure}'].{agg}()")
        return "\n".join(lines)

    def _rerun_with_context(self, q, fact, info, qt, art) -> str:
        intent = self.ctx.get("intent")
        if intent == "trend":
            return self._trend(q, fact, info, qt | _tokens("monthly"), art)
        if intent == "aggregate":
            return self._aggregate(q, fact, info, qt, art)
        return self._rank(q, fact, info, qt, art, ascending=bool(self.ctx.get("spec", {}).get("asc")))

    def _help(self, unrecognised: bool = False) -> str:
        tables = ", ".join(f"`{t}`" for t in self.store.tables)
        intro = "I'm not sure what you'd like to know. " if unrecognised else "Hi! "
        return (f"{intro}I can analyse your data ({tables}). Try asking:\n"
                "- *Which region generated the highest revenue?*\n- *Show monthly sales trends.*\n"
                "- *Which products are underperforming?*\n- *What are the top five customers?*\n"
                "- *Detect anomalies in the dataset.* / *Run a data quality check.*\n"
                "- *Forecast revenue for the next 6 months.*\n- *Generate SQL for this analysis.* (or pandas)\n"
                "- *Give me a summary of the data.*")
