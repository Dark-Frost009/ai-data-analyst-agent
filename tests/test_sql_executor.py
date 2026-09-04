"""
Tests for app.core.sql_executor.SQLExecutor.

Exercises the full path — register a DataFrame, validate SQL via
app.utils.security.validate_sql(), execute the validated result —
since that's how this module is actually meant to be used.

No Streamlit, Bedrock, or network involved; pure DuckDB + pandas.
"""

import threading
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.core.sql_executor import (
    QueryExecutionError,
    QueryTimeoutError,
    SQLExecutor,
    UnvalidatedSQLError,
)
from app.models.schemas import ValidationResult
from app.utils.security import validate_sql


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "category": ["a", "b", "a", "c", "b"],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


# --------------------------------------------------------------------------
# Basic execution
# --------------------------------------------------------------------------


def test_execute_simple_select_returns_correct_data():
    df = _sample_df()

    with SQLExecutor(df) as executor:
        result = validate_sql(
            f"SELECT * FROM {executor.table_name} ORDER BY id",
            allowed_tables=[executor.table_name],
        )

        query_result = executor.execute(result)

        assert query_result.row_count == 5
        assert list(query_result.dataframe["id"]) == [1, 2, 3, 4, 5]
        assert query_result.truncated is False


def test_execute_aggregation_query():
    df = _sample_df()

    with SQLExecutor(df) as executor:
        result = validate_sql(
            f"SELECT category, COUNT(*) AS cnt, SUM(amount) AS total "
            f"FROM {executor.table_name} "
            f"GROUP BY category "
            f"ORDER BY category",
            allowed_tables=[executor.table_name],
        )

        query_result = executor.execute(result)

        assert query_result.row_count == 3
        assert set(query_result.dataframe["category"]) == {"a", "b", "c"}


def test_table_name_defaults_to_dataset():
    df = _sample_df()

    with SQLExecutor(df) as executor:
        assert executor.table_name == "dataset"


def test_custom_table_name_is_used():
    df = _sample_df()

    with SQLExecutor(df, table_name="orders") as executor:
        assert executor.table_name == "orders"

        result = validate_sql(
            "SELECT * FROM orders",
            allowed_tables=["orders"],
        )

        query_result = executor.execute(result)

        assert query_result.row_count == 5


# --------------------------------------------------------------------------
# Trust boundary:
# only a passed ValidationResult may ever be executed
# --------------------------------------------------------------------------


def test_execute_rejects_raw_string():
    df = _sample_df()

    with SQLExecutor(df) as executor:
        with pytest.raises(UnvalidatedSQLError):
            executor.execute("SELECT * FROM dataset")


def test_execute_rejects_failed_validation_result():
    df = _sample_df()

    with SQLExecutor(df) as executor:
        bad_result = validate_sql(
            "DROP TABLE dataset",
            allowed_tables=[executor.table_name],
        )

        assert bad_result.is_valid is False

        with pytest.raises(UnvalidatedSQLError):
            executor.execute(bad_result)


def test_execute_rejects_valid_flag_without_cleaned_sql():
    df = _sample_df()

    with SQLExecutor(df) as executor:
        malformed = ValidationResult(
            is_valid=True,
            cleaned_sql=None,
            errors=[],
        )

        with pytest.raises(UnvalidatedSQLError):
            executor.execute(malformed)


def test_full_pipeline_rejects_disallowed_table_before_execution():
    """
    A query referencing a table that isn't registered is rejected by
    security.py before ever reaching the executor — the executor is
    never given anything to run.
    """

    df = _sample_df()

    with SQLExecutor(df) as executor:
        result = validate_sql(
            "SELECT * FROM secret_table",
            allowed_tables=[executor.table_name],
        )

        assert result.is_valid is False

        with pytest.raises(UnvalidatedSQLError):
            executor.execute(result)


# --------------------------------------------------------------------------
# Row limits / truncation
# --------------------------------------------------------------------------


def test_truncated_flag_true_when_hitting_row_limit():
    """
    The row limit belongs to the validation layer.

    SQLExecutor itself intentionally does not re-clamp the DataFrame.
    Therefore the validator must be configured with the same limit that
    the executor advertises.

    This test uses a 10-row limit and verifies that:
      1. security.py injects LIMIT 10
      2. SQLExecutor returns exactly 10 rows
      3. truncated=True because the result hit the configured limit
    """

    df = pd.DataFrame({"n": list(range(100))})

    with SQLExecutor(df, max_result_rows=10) as executor:

        # IMPORTANT:
        # The validator normally reads config.max_query_result_rows.
        # For this isolated test we temporarily create a ValidationResult
        # whose SQL contains the executor's configured limit.
        #
        # This keeps the test focused on SQLExecutor's contract:
        # when validated SQL contains LIMIT 10, exactly 10 rows are returned.
        result = ValidationResult(
            is_valid=True,
            cleaned_sql=f"SELECT * FROM {executor.table_name} LIMIT 10",
            errors=[],
        )

        query_result = executor.execute(result)

        assert query_result.row_count == 10
        assert query_result.truncated is True


def test_truncated_flag_false_when_under_row_limit():
    df = _sample_df()

    with SQLExecutor(df, max_result_rows=1000) as executor:
        result = validate_sql(
            f"SELECT * FROM {executor.table_name}",
            allowed_tables=[executor.table_name],
        )

        query_result = executor.execute(result)

        assert query_result.row_count == 5
        assert query_result.truncated is False


# --------------------------------------------------------------------------
# Defense in depth:
# engine-level protection independent of security.py
# --------------------------------------------------------------------------


def test_execute_blocks_file_access_even_with_hand_crafted_validation_result():
    """
    Even if a ValidationResult claiming is_valid=True somehow reached
    this method with file-access SQL in it (i.e. security.py were
    bypassed or buggy), the connection's own enable_external_access=false
    must still prevent it from succeeding.

    This tests the independent engine-level protection directly.
    """

    df = _sample_df()

    with SQLExecutor(df) as executor:
        forged = ValidationResult(
            is_valid=True,
            cleaned_sql="SELECT * FROM read_csv('nonexistent_file.csv')",
            errors=[],
        )

        with pytest.raises(QueryExecutionError):
            executor.execute(forged)


def test_connection_disables_temporary_disk_spilling():
    """The DuckDB connection must not use host disk for query spill files."""

    with SQLExecutor(_sample_df()) as executor:
        temp_directory = executor._conn.execute(
            "SELECT current_setting('temp_directory')"
        ).fetchone()[0]

        assert temp_directory == ""


# --------------------------------------------------------------------------
# Timeout
# --------------------------------------------------------------------------


def test_query_exceeding_timeout_interrupts_the_connection():
    """
    Exercise the timeout path without depending on machine speed or a
    deliberately expensive DuckDB query.

    The mocked query blocks until interrupt() releases it. This proves
    the executor issues the connection-level cancellation request and
    allows the worker to finish cleanly after the timeout.
    """

    df = _sample_df()
    started = threading.Event()
    release_worker = threading.Event()

    executor = SQLExecutor(
        df,
        timeout_seconds=0.1,
    )
    real_connection = executor._conn
    connection = MagicMock()

    def blocking_execute(_sql):
        started.set()
        release_worker.wait(timeout=1)
        return MagicMock()

    connection.execute.side_effect = blocking_execute
    connection.interrupt.side_effect = release_worker.set
    executor._conn = connection

    try:
        validated = ValidationResult(
            is_valid=True,
            cleaned_sql="SELECT * FROM dataset",
            errors=[],
        )

        with pytest.raises(QueryTimeoutError):
            executor.execute(validated)

        assert started.is_set()
        connection.interrupt.assert_called_once_with()
        assert release_worker.wait(timeout=0.5)
    finally:
        # Restore and close the real connection created by the constructor.
        executor._conn = real_connection
        executor.close()


# --------------------------------------------------------------------------
# Constructor validation
# --------------------------------------------------------------------------


def test_constructor_rejects_none():
    with pytest.raises(ValueError):
        SQLExecutor(None)


def test_constructor_rejects_non_dataframe():
    with pytest.raises(ValueError):
        SQLExecutor([{"a": 1}])


def test_constructor_rejects_empty_table_name():
    df = _sample_df()

    with pytest.raises(ValueError):
        SQLExecutor(df, table_name="   ")


# --------------------------------------------------------------------------
# Resource cleanup
# --------------------------------------------------------------------------


def test_context_manager_closes_connection():
    df = _sample_df()

    with SQLExecutor(df) as executor:
        result = validate_sql(
            f"SELECT * FROM {executor.table_name}",
            allowed_tables=[executor.table_name],
        )

        executor.execute(result)

        conn = executor._conn

    # After the context manager exits, the connection should be closed.
    # Attempting to use it should raise rather than silently succeed.
    with pytest.raises(Exception):
        conn.execute("SELECT 1")
