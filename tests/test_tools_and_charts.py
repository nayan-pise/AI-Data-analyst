import pandas as pd
import pytest

from analyst.charts import CHART_TYPES, ChartError, build_chart
from analyst.models import TurnArtifacts
from analyst.tools import PandasCodeError, ToolExecutor, run_pandas_code


@pytest.fixture()
def ex(store):
    return ToolExecutor(store, TurnArtifacts())


def test_run_sql_records_artifacts(ex):
    out = ex.execute("run_sql", {"sql": "SELECT region, COUNT(*) AS n FROM orders GROUP BY 1 ORDER BY 2 DESC", "purpose": "orders per region"})
    assert out.startswith("region,n") and "East" in out
    assert ex.art.code[0]["language"] == "sql" and ex.art.tables and ex.art.steps


def test_sql_error_is_returned_not_raised(ex):
    out = ex.execute("run_sql", {"sql": "SELECT nope FROM orders", "purpose": "x"})
    assert out.startswith("ERROR")
    out = ex.execute("run_sql", {"sql": "DROP TABLE orders", "purpose": "x"})
    assert out.startswith("ERROR") and "read-only" in out.lower()


def test_unknown_tool_and_bad_args(ex):
    assert ex.execute("nope", {}).startswith("ERROR")
    assert ex.execute("run_sql", {"wrong": 1}).startswith("ERROR")


def test_pandas_tool(ex):
    out = ex.execute("run_pandas", {"code": "result = orders.groupby('region')['revenue'].sum().round(0)", "purpose": "rev"})
    assert "East" in out and ex.art.code[0]["language"] == "python"


@pytest.mark.parametrize("code", [
    "import os\nresult = 1",
    "result = open('/etc/passwd').read()",
    "result = orders.to_csv('/tmp/x.csv')",
    "result = orders.__class__",
    "result = pd.read_csv('x.csv')",
    "result = eval('1+1')",
    "x = getattr(orders, 'shape')",
])
def test_pandas_sandbox_blocks(code, store):
    with pytest.raises(PandasCodeError):
        run_pandas_code(code, store.frames)


def test_pandas_does_not_mutate_store(store):
    before = store.frames["orders"]["revenue"].sum()
    run_pandas_code("orders['revenue'] = 0\nresult = orders.shape", store.frames)
    assert store.frames["orders"]["revenue"].sum() == before


def test_pandas_requires_result(ex):
    assert "result" in ex.execute("run_pandas", {"code": "x = 1", "purpose": "p"})


def test_create_chart_tool(ex):
    out = ex.execute("create_chart", {"sql": "SELECT region, SUM(revenue) AS rev FROM orders GROUP BY 1", "chart_type": "bar",
                                      "x": "region", "y": "rev", "title": "Revenue by region"})
    assert "created" in out and len(ex.art.figures) == 1


def test_create_chart_bad_column_gives_actionable_error(ex):
    out = ex.execute("create_chart", {"sql": "SELECT region FROM orders LIMIT 5", "chart_type": "bar", "x": "region", "y": "zzz", "title": "t"})
    assert out.startswith("ERROR") and "Available columns" in out


def test_anomaly_quality_forecast_describe_tools(ex):
    assert "anomalous" in ex.execute("detect_anomalies", {"table": "orders", "date_column": "order_date", "value_column": "revenue"})
    assert "Duplicate" in ex.execute("data_quality_report", {})
    assert "forecast" in ex.execute("forecast", {"table": "orders", "date_column": "order_date", "value_column": "revenue", "periods": 3})
    assert "revenue" in ex.execute("describe_table", {"table": "orders"})
    assert ex.execute("detect_anomalies", {"table": "missing"}).startswith("ERROR")


@pytest.mark.parametrize("ctype,kwargs", [
    ("bar", dict(x="a", y="b")), ("line", dict(x="a", y="b")), ("pie", dict(x="a", y="b")),
    ("scatter", dict(x="b", y="b")), ("histogram", dict(x="b")), ("box", dict(y="b")),
    ("area", dict(x="a", y="b")), ("heatmap", dict()),
])
def test_all_chart_types_build(ctype, kwargs):
    df = pd.DataFrame({"a": list("xyzw"), "b": [1.0, 2.0, 3.0, 5.0], "c": [2.0, 1.0, 4.0, 3.0]})
    assert build_chart(df, ctype, title="t", **kwargs) is not None
    assert set(CHART_TYPES) >= {ctype}


def test_chart_errors():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ChartError):
        build_chart(df.iloc[0:0], "bar", "a", "a")
    with pytest.raises(ChartError):
        build_chart(df, "radar", "a", "a")
    with pytest.raises(ChartError):
        build_chart(df, "bar", "a")
