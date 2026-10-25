from pathlib import Path

import pandas as pd
import pytest

from analyst.agent import ConversationMemory
from analyst.dashboard import build_dashboard
from analyst.data_store import DataStore
from analyst.models import Turn
from analyst.offline import OfflineAnalyst
from analyst.report import build_html, build_markdown


@pytest.fixture()
def bot(store):
    return OfflineAnalyst(store)


def truth(store):
    o = store.frames["orders"]
    return o, store.frames["customers"], store.frames["products"]


def test_highest_revenue_region_matches_pandas(bot, store):
    o, *_ = truth(store)
    expected = o.groupby("region")["revenue"].sum().idxmax()
    resp = bot.run("Which region generated the highest revenue?")
    assert f"**{expected}**" in resp.text and resp.artifacts.figures and resp.artifacts.code[0]["language"] == "sql"


def test_top_five_customers_joins_and_matches(bot, store):
    o, c, _ = truth(store)
    m = o.merge(c, on="customer_id").groupby("customer_name")["revenue"].sum().nlargest(5)
    resp = bot.run("What are the top five customers?")
    shown = resp.artifacts.tables[0]["df"]
    assert list(shown.iloc[:, 0]) == list(m.index) and "JOIN customers" in resp.artifacts.code[0]["code"]


def test_underperforming_products_flags_planted_ones(bot):
    resp = bot.run("Which products are underperforming?")
    names = set(resp.artifacts.tables[0]["df"]["product_name"].head(2))
    assert names == {"Kappa Cable", "Mu Speaker"}


def test_monthly_trend(bot, store):
    resp = bot.run("Show monthly sales trends.")
    df = resp.artifacts.tables[0]["df"]
    o, *_ = truth(store)
    assert len(df) == 24 and df.iloc[:, -1].sum() == pytest.approx(o["revenue"].sum(), rel=1e-6)
    assert "Nov 2025" in resp.text


def test_year_filter_and_exclusion(bot, store):
    o, *_ = truth(store)
    resp = bot.run("total revenue in 2025 excluding cancelled")
    expected = o[(o.order_date.dt.year == 2025) & (o.status != "Cancelled")]["revenue"].sum()
    assert resp.artifacts.tables[0]["df"].iloc[0, 0] == pytest.approx(expected, abs=0.01)


def test_count_question_with_year(bot, store):
    o, *_ = truth(store)
    resp = bot.run("how many orders were placed in 2025?")
    assert f"{(o.order_date.dt.year == 2025).sum():,}" in resp.text


def test_cross_table_dimension(bot):
    resp = bot.run("revenue by customer segment")
    assert set(resp.artifacts.tables[0]["df"]["segment"]) == {"Consumer", "SMB", "Enterprise"}


def test_anomalies_and_quality_intents(bot):
    a = bot.run("Detect anomalies in the dataset.")
    assert "12 anomalous" in a.text and "robust z-scores" in a.text
    q = bot.run("run a data quality check")
    assert "orphaned keys" in q.text.lower()


def test_forecast_intent(bot):
    r = bot.run("forecast revenue for the next 3 months")
    assert len(r.artifacts.tables[0]["df"]) == 3 and r.artifacts.figures


def test_summary_intent(bot):
    r = bot.run("Give me a summary of the data")
    assert "4,212 rows" in r.text and "Top region" in r.text


def test_follow_up_sql_and_pandas_reflect_last_analysis(bot):
    bot.run("Which region generated the highest revenue?")
    sql = bot.run("Generate SQL for this analysis.").text
    assert "GROUP BY" in sql and "region" in sql
    pandas_code = bot.run("show the pandas version").text
    assert "groupby('region')" in pandas_code


def test_follow_up_changes_dimension(bot):
    bot.run("Which products are underperforming?")
    r = bot.run("what about by category?")
    assert "category" in r.text.lower() and "Audio" in r.text


def test_sql_request_without_context(bot):
    assert "haven't run an analysis" in bot.run("generate sql").text


def test_no_data_and_unrecognised(store):
    assert "upload" in OfflineAnalyst(DataStore()).run("anything").text.lower()
    assert "Try asking" in OfflineAnalyst(store).run("xyzzy plugh").text


def test_reports(store, bot):
    turns = []
    mem = ConversationMemory()
    for q in ["Which region generated the highest revenue?", "Detect anomalies in the dataset."]:
        r = bot.run(q)
        turns.append(Turn(q, r.text, "", r))
    md, html = build_markdown(turns, store), build_html(turns, store)
    assert "Which region" in md and "```sql" in md
    assert "<html" in html and "plotly" in html.lower() and "<strong>East</strong>" in html


def test_html_report_escapes_user_text(store):
    turns = [Turn("<script>alert(1)</script>", "**x** <b>y</b>")]
    html = build_html(turns, store)
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html


def test_dashboard_builds(store):
    d = build_dashboard(store)
    assert d.table == "orders" and len(d.kpis) >= 4 and len(d.figures) >= 4


def test_app_smoke_flow():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    [b for b in at.sidebar.button if "sample" in b.label.lower()][0].click().run()
    at.chat_input[0].set_value("Which region generated the highest revenue?").run()
    at.chat_input[0].set_value("Generate SQL for this analysis.").run()
    assert not at.exception
    assert len(at.session_state["messages"]) == 4
    assert "SELECT" in at.session_state["messages"][-1]["content"]
