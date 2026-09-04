"""
Tests for app.utils.security.validate_sql.

No DuckDB, Bedrock, or pandas involved — this is pure SQL-text-in,
ValidationResult-out testing.

A note on error-code strictness: most rejection tests assert one exact
error_code, per the project's convention. Two categories are intentionally
more permissive (asserting the code is one of a small, still-meaningful set)
because the precise sqlglot AST node produced for certain DuckDB-specific
constructs cannot be fully confirmed without executing sqlglot directly:

- DuckDB extension commands (ATTACH/DETACH/COPY/PRAGMA/INSTALL/LOAD)
  may surface as either DISALLOWED_STATEMENT_TYPE or PARSE_ERROR,
  depending on how sqlglot's DuckDB dialect models each one.
- Table-valued function calls in FROM position (read_csv, sqlite_scan,
  ...) may be caught by either the function allowlist or the table
  allowlist, depending on how sqlglot structures that AST node.
  Both outcomes in each case are safe rejections (is_valid=False) —
  the permissiveness is about which specific code, never about whether
  the query is correctly blocked.
"""

import pytest
import sqlglot
from sqlglot import exp

from app.config import config
from app.utils.security import validate_sql


ORDERS = ["orders"]


# ===========================================================================
# Should pass
# ===========================================================================

def test_validate_sql_allows_simple_select():
    result = validate_sql(
        "SELECT * FROM orders",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True
    assert result.cleaned_sql is not None
    assert result.errors == []


def test_validate_sql_allows_where_clause():
    result = validate_sql(
        "SELECT id, status FROM orders WHERE status = 'shipped'",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True


def test_validate_sql_allows_group_by():
    result = validate_sql(
        "SELECT status, COUNT(*) FROM orders GROUP BY status",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True


def test_validate_sql_allows_order_by():
    result = validate_sql(
        "SELECT id FROM orders ORDER BY id DESC",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True


def test_validate_sql_allows_aggregates():
    result = validate_sql(
        "SELECT status, COUNT(*) AS cnt, AVG(total) AS avg_total, "
        "SUM(total) AS sum_total, MIN(total) AS min_total, "
        "MAX(total) AS max_total "
        "FROM orders GROUP BY status",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True


def test_validate_sql_allows_self_join():
    result = validate_sql(
        "SELECT a.id, b.id FROM orders AS a "
        "JOIN orders AS b ON a.customer_id = b.customer_id",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True


def test_validate_sql_allows_cte():
    result = validate_sql(
        "WITH recent AS "
        "(SELECT * FROM orders WHERE id > 100) "
        "SELECT * FROM recent",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True


# ===========================================================================
# CASE expression regression tests
# ===========================================================================

def test_validate_sql_allows_case_expression():
    """
    SQL CASE expressions are safe read-only expressions.

    sqlglot represents CASE internally using Case + If nodes.
    The internal If must not be mistaken for a callable SQL function.
    """

    result = validate_sql(
        "SELECT CASE "
        "WHEN total > 100 THEN 'High' "
        "ELSE 'Low' "
        "END "
        "FROM orders",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True
    assert result.cleaned_sql is not None


def test_validate_sql_allows_case_with_multiple_when_branches():
    """
    Multiple WHEN branches should remain valid.
    """

    result = validate_sql(
        "SELECT CASE "
        "WHEN total > 100 THEN 'High' "
        "WHEN total > 50 THEN 'Medium' "
        "ELSE 'Low' "
        "END "
        "FROM orders",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True
    assert result.cleaned_sql is not None


def test_validate_sql_rejects_explicit_if_function():
    """
    CASE is allowed, but an explicit IF(...) function is still subject
    to the function allowlist and must remain rejected.
    """

    result = validate_sql(
        "SELECT IF(total > 100, 'High', 'Low') "
        "FROM orders",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is False
    assert result.errors[0].code == "DISALLOWED_FUNCTION"


# ===========================================================================
# LIMIT enforcement
# ===========================================================================

def test_validate_sql_preserves_existing_limit_within_bounds():
    result = validate_sql(
        "SELECT * FROM orders LIMIT 50",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True
    assert "50" in result.cleaned_sql
    assert str(config.max_query_result_rows) not in result.cleaned_sql


def test_validate_sql_preserves_zero_limit():
    """
    LIMIT 0 is a valid, plain integer LIMIT and should remain unchanged.
    """

    result = validate_sql(
        "SELECT * FROM orders LIMIT 0",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True
    assert result.cleaned_sql is not None
    assert "LIMIT 0" in result.cleaned_sql.upper()


def test_validate_sql_injects_limit_when_absent():
    result = validate_sql(
        "SELECT * FROM orders",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True
    assert "LIMIT" in result.cleaned_sql.upper()
    assert str(config.max_query_result_rows) in result.cleaned_sql


def test_validate_sql_clamps_oversized_limit():
    result = validate_sql(
        "SELECT * FROM orders LIMIT 999999999",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True
    assert "999999999" not in result.cleaned_sql
    assert str(config.max_query_result_rows) in result.cleaned_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders LIMIT -1",
        "SELECT * FROM orders LIMIT 10.5",
        "SELECT * FROM orders LIMIT 5 + 5",
    ],
)
def test_validate_sql_replaces_unsafe_limit_expression_with_max(sql):
    """
    LIMIT expressions that are not plain non-negative integer literals
    must not be interpreted partially.

    They are sanitized by replacing the LIMIT with the configured maximum.
    """

    result = validate_sql(
        sql,
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True
    assert result.cleaned_sql is not None
    assert str(config.max_query_result_rows) in result.cleaned_sql


def test_validate_sql_does_not_preserve_negative_limit():
    """
    Regression test for the old bug where sqlglot represented LIMIT -1
    as Neg(Literal(1)), while _extract_limit_value() incorrectly returned 1.
    """

    result = validate_sql(
        "SELECT * FROM orders LIMIT -1",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True
    assert result.cleaned_sql is not None
    assert "LIMIT -1" not in result.cleaned_sql.upper()
    assert (
        f"LIMIT {config.max_query_result_rows}"
        in result.cleaned_sql.upper()
    )


def test_validate_sql_does_not_partially_interpret_limit_expression():
    """
    Regression test for the old bug where LIMIT 5 + 5 was interpreted
    as 5 because only the left side of the AST was extracted.
    """

    result = validate_sql(
        "SELECT * FROM orders LIMIT 5 + 5",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True
    assert result.cleaned_sql is not None
    assert "LIMIT 5 + 5" not in result.cleaned_sql.upper()
    assert (
        f"LIMIT {config.max_query_result_rows}"
        in result.cleaned_sql.upper()
    )


def test_validate_sql_does_not_preserve_decimal_limit():
    """
    Decimal LIMIT values are not trusted as integer row limits.
    """

    result = validate_sql(
        "SELECT * FROM orders LIMIT 10.5",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True
    assert result.cleaned_sql is not None
    assert "LIMIT 10.5" not in result.cleaned_sql.upper()
    assert (
        f"LIMIT {config.max_query_result_rows}"
        in result.cleaned_sql.upper()
    )


# ===========================================================================
# Should fail — DML/DDL
# ===========================================================================

@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO orders (id) VALUES (1)",
        "UPDATE orders SET status = 'shipped'",
        "DELETE FROM orders",
        "DROP TABLE orders",
        "ALTER TABLE orders ADD COLUMN note TEXT",
        "CREATE TABLE new_table (id INT)",
    ],
)
def test_validate_sql_rejects_dml_ddl_statements(sql):
    result = validate_sql(
        sql,
        allowed_tables=ORDERS,
    )

    assert result.is_valid is False
    assert result.errors[0].code == "DISALLOWED_STATEMENT_TYPE"


# ===========================================================================
# Should fail — DuckDB extension commands
# ===========================================================================

@pytest.mark.parametrize(
    "sql",
    [
        "ATTACH 'other.db' AS other",
        "DETACH other",
        "COPY orders TO 'orders.csv'",
        "PRAGMA table_info(orders)",
        "INSTALL httpfs",
        "LOAD httpfs",
    ],
)
def test_validate_sql_rejects_duckdb_extension_commands(sql):
    result = validate_sql(
        sql,
        allowed_tables=ORDERS,
    )

    assert result.is_valid is False
    assert result.errors[0].code in {
        "DISALLOWED_STATEMENT_TYPE",
        "PARSE_ERROR",
    }


# ===========================================================================
# Should fail — statement stacking
# ===========================================================================

def test_validate_sql_rejects_multiple_statements():
    result = validate_sql(
        "SELECT * FROM orders; DROP TABLE orders",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is False
    assert result.errors[0].code == "MULTIPLE_STATEMENTS"


# ===========================================================================
# Comment content must be inert
# ===========================================================================

def test_validate_sql_comment_content_is_inert():
    """
    Dangerous-looking text inside a comment must never become a second,
    executable statement.

    Verified independently of validate_sql()'s own logic by parsing
    cleaned_sql directly with sqlglot.

    Comment text may legitimately remain in cleaned_sql; it is inert either
    way, since it is still just a comment.
    """

    result = validate_sql(
        "SELECT * FROM orders -- ; DROP TABLE orders",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is True
    assert result.cleaned_sql is not None

    reparsed = [
        statement
        for statement in sqlglot.parse(
            result.cleaned_sql,
            read="duckdb",
        )
        if statement is not None
    ]

    assert len(reparsed) == 1
    assert isinstance(
        reparsed[0],
        (exp.Select, exp.With),
    )


# ===========================================================================
# Should fail — file-access / cross-database functions
# ===========================================================================

@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM read_parquet('s3://bucket/file.parquet')",
    ],
)
def test_validate_sql_rejects_file_access_functions(sql):
    result = validate_sql(
        sql,
        allowed_tables=ORDERS,
    )

    assert result.is_valid is False
    assert result.errors[0].code in {
        "DISALLOWED_FUNCTION",
        "DISALLOWED_TABLE_REFERENCE",
    }


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM sqlite_scan('other.db', 'orders')",
        "SELECT * FROM postgres_scan('conn_str', 'public', 'orders')",
    ],
)
def test_validate_sql_rejects_cross_database_functions(sql):
    result = validate_sql(
        sql,
        allowed_tables=ORDERS,
    )

    assert result.is_valid is False
    assert result.errors[0].code in {
        "DISALLOWED_FUNCTION",
        "DISALLOWED_TABLE_REFERENCE",
    }


# ===========================================================================
# Should fail — other cases
# ===========================================================================

def test_validate_sql_rejects_unregistered_table():
    result = validate_sql(
        "SELECT * FROM secret_table",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is False
    assert result.errors[0].code == "DISALLOWED_TABLE_REFERENCE"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM main.orders",
        "SELECT * FROM catalog.main.orders",
    ],
)
def test_validate_sql_rejects_qualified_table_references(sql):
    """The registered in-memory table must not be schema-qualified."""

    result = validate_sql(
        sql,
        allowed_tables=ORDERS,
    )

    assert result.is_valid is False
    assert result.errors[0].code == "QUALIFIED_TABLE_REFERENCE"


def test_validate_sql_rejects_unparseable_sql():
    result = validate_sql(
        "SELECT * FROM orders WHERE (id = 1",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is False
    assert result.errors[0].code == "PARSE_ERROR"


def test_validate_sql_rejects_case_and_whitespace_obfuscated_drop():
    """
    SQL keywords are case-insensitive and whitespace-insignificant by
    the SQL standard.

    A real parser, unlike a naive blocklist, is not fooled by this.
    """

    result = validate_sql(
        "   dRoP     TABLE   orders  ",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is False
    assert result.errors[0].code in {
        "DISALLOWED_STATEMENT_TYPE",
        "PARSE_ERROR",
    }


def test_validate_sql_rejects_empty_string():
    result = validate_sql(
        "",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is False
    assert result.errors[0].code == "EMPTY_SQL"

def test_validate_sql_rejects_union_with_unregistered_table():
    result = validate_sql(
        "SELECT * FROM orders "
        "UNION ALL "
        "SELECT * FROM secret_table",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is False
    assert result.errors[0].code == "DISALLOWED_STATEMENT_TYPE"

def test_validate_sql_rejects_cte_shadowing_with_unregistered_table():
    result = validate_sql(
        "WITH orders AS "
        "(SELECT * FROM secret_table) "
        "SELECT * FROM orders",
        allowed_tables=ORDERS,
    )

    assert result.is_valid is False
    assert result.errors[0].code == "DISALLOWED_TABLE_REFERENCE"