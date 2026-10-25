import pytest

from analyst.sql_safety import SQLSafetyError, validate_select


@pytest.mark.parametrize("sql", [
    "SELECT * FROM orders",
    "select region, sum(revenue) from orders group by 1;",
    "WITH a AS (SELECT 1) SELECT * FROM a",
    "SELECT status FROM orders WHERE status = 'create or drop'",     # keywords inside literals are fine
    "SELECT REPLACE(region, 'N', 'X') FROM orders",                   # REPLACE() the function is fine
    "SELECT \"update\" FROM t",                                       # quoted identifier
    "-- comment\nSELECT 1",
    "(SELECT 1) UNION ALL (SELECT 2)",
    "FROM orders SELECT region",
])
def test_allowed(sql):
    assert validate_select(sql)


@pytest.mark.parametrize("sql", [
    "",
    "   ",
    "DROP TABLE orders",
    "DELETE FROM orders",
    "INSERT INTO orders VALUES (1)",
    "UPDATE orders SET revenue = 0",
    "CREATE TABLE x AS SELECT 1",
    "SELECT 1; DROP TABLE orders",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_parquet('x.parquet')",
    "COPY orders TO '/tmp/x.csv'",
    "ATTACH 'x.db'",
    "PRAGMA database_list",
    "SET enable_external_access=true",
    "SELECT * FROM glob('*')",
    "/* hide */ DROP TABLE orders",
    "SELECT 1 /* ok */; SELECT 2",
])
def test_blocked(sql):
    with pytest.raises(SQLSafetyError):
        validate_select(sql)
