"""Guardrails for LLM-generated SQL: read-only, single statement, no file/network access."""
from __future__ import annotations

import re


class SQLSafetyError(ValueError):
    """Raised when a query is not a safe, read-only analytical statement."""


_ALLOWED_START = ("select", "with", "from", "values", "describe", "summarize")

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|export|import|install|"
    r"load|pragma|call|set|reset|truncate|vacuum|checkpoint|grant|revoke|use)\b",
    re.IGNORECASE,
)
# Table functions / settings that touch the filesystem, network or engine internals.
_FORBIDDEN_FUNCTIONS = re.compile(
    r"\b(read_\w+|glob|parquet_scan|csv_scan|json_scan|sniff_csv|duckdb_\w+|pragma_\w+|"
    r"current_setting|getenv|query_table|httpfs)\b",
    re.IGNORECASE,
)

_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_QUOTED_IDENT = re.compile(r'"(?:[^"]|"")*"')
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _mask(sql: str) -> str:
    """Blank out comments, string literals and quoted identifiers before keyword checks."""
    masked = _BLOCK_COMMENT.sub(" ", sql)
    masked = _LINE_COMMENT.sub(" ", masked)
    masked = _STRING_LITERAL.sub("''", masked)
    masked = _QUOTED_IDENT.sub('"x"', masked)
    return masked


def validate_select(sql: str) -> str:
    """Return the cleaned query or raise SQLSafetyError."""
    if not isinstance(sql, str) or not sql.strip():
        raise SQLSafetyError("Empty SQL query.")
    cleaned = sql.strip().rstrip(";").strip()
    masked = _mask(cleaned)

    if ";" in masked:
        raise SQLSafetyError("Only a single SQL statement is allowed.")
    if not masked.lstrip("( \n\t").lower().startswith(_ALLOWED_START):
        raise SQLSafetyError("Only read-only SELECT / WITH queries are allowed.")
    match = _FORBIDDEN_KEYWORDS.search(masked)
    if match:
        raise SQLSafetyError(f"Keyword '{match.group(0).upper()}' is not allowed (read-only access).")
    match = _FORBIDDEN_FUNCTIONS.search(masked)
    if match:
        raise SQLSafetyError(f"Function '{match.group(0)}' is not allowed (no file or system access).")
    return cleaned
