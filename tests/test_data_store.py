import pandas as pd
import pytest

from analyst.config import Settings
from analyst.data_store import DataStore, DataValidationError, sanitize_identifier
from analyst.sql_safety import SQLSafetyError


def load(csv_text: str, name="t.csv", store=None):
    store = store or DataStore()
    return store, store.load_csv(csv_text.encode(), name)


def test_loads_and_profiles_sample(store):
    assert set(store.tables) == {"orders", "customers", "products"}
    info = store.tables["orders"]
    assert info.rows == 4212
    kinds = {c.name: c.kind for c in info.columns}
    assert kinds["order_date"] == "datetime"
    assert kinds["revenue"] == "numeric"
    assert kinds["region"] == "categorical"
    assert info.col("order_id").is_id


def test_schema_text_lists_tables_and_joins(store):
    text = store.schema_text()
    assert "TABLE orders" in text and "revenue" in text
    assert "orders.customer_id = customers.customer_id" in text


def test_suggest_joins(store):
    pairs = {(j["child"], j["parent"], j["column"]) for j in store.suggest_joins()}
    assert ("orders", "customers", "customer_id") in pairs
    assert ("orders", "products", "product_id") in pairs


def test_quality_report_finds_planted_issues(store):
    issues = {(i["table"], i["column"], i["issue"]) for i in store.quality_report()}
    assert ("orders", "", "Duplicate rows") in issues
    assert ("orders", "discount", "Missing values") in issues
    assert ("orders", "quantity", "Negative values") in issues
    assert ("orders", "customer_id", "Orphaned keys") in issues


def test_quality_report_unknown_table(store):
    with pytest.raises(KeyError):
        store.quality_report("nope")


def test_semicolon_delimiter_and_currency_and_dates():
    csv = "Order Date;Amount;Region\n2024-01-05;$1,200.50;north\n2024-02-07;(50.00);South\n2024-03-09;300;South\n"
    store, res = load(csv)
    df = store.frames["t"]
    assert list(df.columns) == ["order_date", "amount", "region"]
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])
    assert df["amount"].tolist() == [1200.5, -50.0, 300.0]
    assert any("snake_case" in w for w in res.warnings)


def test_leading_zero_codes_stay_text():
    _, res = load("code,val\n00123,1\n00456,2\n00789,3\n")
    assert res.info.col("code").kind != "numeric"


def test_duplicate_and_reserved_column_names():
    store, _ = load("Name,name,order,Group\na,b,1,x\nc,d,2,y\n")
    cols = list(store.frames["t"].columns)
    assert len(set(cols)) == 4 and "order_col" in cols and "group_col" in cols


@pytest.mark.parametrize("raw,fname,msg", [
    (b"", "a.csv", "empty"),
    (b"a,b\n", "a.csv", "no usable data|no data"),
    (b"x,y\n1,2\n", "a.pdf", "not a CSV"),
])
def test_validation_errors(raw, fname, msg):
    with pytest.raises(DataValidationError, match=msg):
        DataStore().load_csv(raw, fname)


def test_size_limit():
    s = DataStore(Settings(model="m", api_key=None, max_upload_mb=0))
    with pytest.raises(DataValidationError, match="limit"):
        s.load_csv(b"a,b\n1,2\n", "a.csv")


def test_latin1_encoding():
    store, _ = load("name,city\nJosé,Zürich\n".encode("latin-1").decode("latin-1"))
    assert len(store.frames["t"]) == 1


def test_query_basic_and_truncation(store):
    res = store.query("SELECT region, SUM(revenue) AS r FROM orders GROUP BY 1 ORDER BY 2 DESC")
    assert res.df.iloc[0]["region"] == "East" and len(res.df) == 5
    res = store.query("SELECT * FROM orders", max_rows=10)
    assert len(res.df) == 10 and res.truncated


def test_query_cache(store):
    sql = "SELECT COUNT(*) AS n FROM orders"
    first = store.query(sql)
    second = store.query(sql)
    assert not first.cached and second.cached
    assert store.cache_hits == 1


def test_cache_invalidated_on_new_data(store):
    store.query("SELECT COUNT(*) FROM orders")
    store.load_csv(b"a,b\n1,2\n3,4\n", "extra.csv")
    assert not store.query("SELECT COUNT(*) FROM orders").cached


def test_query_errors_are_friendly(store):
    with pytest.raises(SQLSafetyError, match="SQL error"):
        store.query("SELECT nonexistent FROM orders")
    with pytest.raises(SQLSafetyError):
        store.query("DROP TABLE orders")


def test_external_access_disabled(store):
    with pytest.raises(Exception):
        store.con.execute("SELECT * FROM read_csv('/etc/hostname')").fetchall()


def test_require_table_resolution(store):
    assert store.require_table(None) == "orders"
    assert store.require_table("Customers") == "customers"
    with pytest.raises(KeyError):
        store.require_table("zzz")


def test_drop_table(store):
    store.drop_table("products")
    assert "products" not in store.tables
    with pytest.raises(SQLSafetyError):
        store.query("SELECT * FROM products")


def test_sanitize_identifier():
    assert sanitize_identifier("Order Date!") == "order_date"
    assert sanitize_identifier("1st") == "c_1st"
    assert sanitize_identifier("x", {"x"}) == "x_2"
