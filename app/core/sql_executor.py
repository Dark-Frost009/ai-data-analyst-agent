"""
SQL execution layer.

Registers a pandas DataFrame (from `data_loader.load_csv()`) into an
in-memory DuckDB connection, and executes SQL that has already passed
`app.utils.security.validate_sql()`. This is the only module allowed to
hold a live DuckDB connection or actually run a query.

## Design notes

- `execute()` accepts only a `ValidationResult`, never a raw SQL string.
  This mirrors the architecture agreed before `security.py` was built:
  callers cannot accidentally skip validation, because there is no code
  path here that takes a bare string at all.

- Runtime limits are enforced independently of `security.py`'s SQL-level
  checks — defense in depth, not a duplicate of the same check:

  - `enable_external_access=false` on the connection itself blocks
    DuckDB's file/network access at the engine level, even if a
    validation bug somehow let a file-access function through.

  - `memory_limit` / `threads` cap resource usage per connection.

  - `temp_directory=''` disables disk spilling. Queries that cannot run
    within the memory budget fail instead of consuming host disk space.

  - A soft timeout is enforced by running the query in a worker thread
    and calling `connection.interrupt()` if it overruns.

  - `row_count`/`truncated` on the returned `QueryResult` reflect the
    configured maximum result size.

- Each `SQLExecutor` owns one isolated in-memory DuckDB connection and
  one registered table — no state is shared across instances/sessions.
"""

import concurrent.futures
from typing import Optional

import duckdb
import pandas as pd

from app.config import config
from app.models.schemas import QueryResult, ValidationResult
from app.utils.logger import get_logger


logger = get_logger(__name__)

DEFAULT_TABLE_NAME = "dataset"


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class SQLExecutorError(Exception):
    """Base class for all errors raised by SQLExecutor."""


class UnvalidatedSQLError(SQLExecutorError):
    """execute() was called without a successful ValidationResult."""


class QueryTimeoutError(SQLExecutorError):
    """The query exceeded the configured timeout and was cancelled."""


class QueryExecutionError(SQLExecutorError):
    """DuckDB failed to execute the query for any other reason."""


class SQLExecutor:
    """
    Owns one in-memory DuckDB connection with one registered DataFrame.

    Usage:
        executor = SQLExecutor(df)
        result = validate_sql(
            sql,
            allowed_tables=[executor.table_name],
            max_rows=executor.max_result_rows,
        )
        query_result = executor.execute(result)
        executor.close()

    Also usable as a context manager:

        with SQLExecutor(df) as executor:
            ...
    """

    def __init__(
        self,
        df: pd.DataFrame,
        table_name: str = DEFAULT_TABLE_NAME,
        max_result_rows: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ):
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        if not table_name or not table_name.strip():
            raise ValueError("table_name must be a non-empty string")

        self._df = df
        self._table_name = table_name

        self._max_result_rows = (
            max_result_rows
            if max_result_rows is not None
            else config.max_query_result_rows
        )

        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else config.query_timeout_seconds
        )

        if self._max_result_rows <= 0:
            raise ValueError("max_result_rows must be greater than zero")

        if self._timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self._conn = self._create_connection()
        self._conn.register(self._table_name, self._df)

        logger.info(
            "SQLExecutor initialized | table=%s | rows=%d | "
            "memory_limit=%s | threads=%s | timeout=%ss",
            self._table_name,
            len(self._df),
            config.duckdb_memory_limit,
            config.duckdb_thread_limit,
            self._timeout_seconds,
        )

    @property
    def table_name(self) -> str:
        """The name this executor's DataFrame is registered under."""
        return self._table_name

    @property
    def max_result_rows(self) -> int:
        """
        Maximum number of rows permitted in a query result.

        This value should be passed to `validate_sql(..., max_rows=...)`
        so the SQL-level LIMIT and executor-level result metadata use
        the same configured boundary.
        """
        return self._max_result_rows

    def execute(
        self,
        validation_result: ValidationResult,
    ) -> QueryResult:
        """
        Execute an already-validated SQL statement.

        Parameters
        ----------
        validation_result:
            The output of `app.utils.security.validate_sql()`.
            Must have `is_valid=True` and `cleaned_sql` set.

        Raises
        ------
        UnvalidatedSQLError
            If validation_result is not a successful ValidationResult.

        QueryTimeoutError
            If the query exceeds the configured timeout.

        QueryExecutionError
            If DuckDB fails to execute the query for any other reason.
        """
        if not isinstance(validation_result, ValidationResult):
            raise UnvalidatedSQLError(
                "execute() requires a ValidationResult produced by "
                "app.utils.security.validate_sql() — never a raw SQL string."
            )

        if (
            not validation_result.is_valid
            or not validation_result.cleaned_sql
        ):
            raise UnvalidatedSQLError(
                "execute() was given a ValidationResult with "
                "is_valid=False (or no cleaned_sql). Only successfully "
                "validated SQL may be executed."
            )

        sql = validation_result.cleaned_sql

        df = self._run_with_timeout(sql)

        row_count = len(df)

        # A result that reaches the configured limit may indicate that
        # additional rows were available but were intentionally excluded
        # by the SQL LIMIT.
        truncated = row_count >= self._max_result_rows

        logger.info(
            "Query executed | table=%s | rows=%d | truncated=%s",
            self._table_name,
            row_count,
            truncated,
        )

        return QueryResult(
            dataframe=df,
            row_count=row_count,
            truncated=truncated,
        )

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._conn.close()

    def __enter__(self) -> "SQLExecutor":
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ) -> None:
        self.close()

    def _create_connection(self) -> "duckdb.DuckDBPyConnection":
        """
        Create and configure an isolated in-memory DuckDB connection.
        """
        conn = duckdb.connect(":memory:")

        conn.execute(
            f"SET memory_limit='{config.duckdb_memory_limit}'"
        )

        conn.execute(
            f"SET threads={config.duckdb_thread_limit}"
        )

        # Keep query intermediates in memory. Without this setting,
        # DuckDB may spill large sorts, joins, or aggregations to the
        # host's temporary directory, bypassing the intended memory
        # budget with disk consumption.
        conn.execute(
            "SET temp_directory=''"
        )

        # Key defense against DuckDB file/network-access functions.
        #
        # This holds even if a validation bug ever lets a dangerous
        # function through security.py.
        conn.execute(
            "SET enable_external_access=false"
        )

        return conn

    def _run_with_timeout(
        self,
        sql: str,
    ) -> pd.DataFrame:
        """
        Run SQL in a worker thread and enforce the configured timeout.

        A ThreadPoolExecutor context manager is deliberately avoided
        because its implicit shutdown(wait=True) would block the caller
        after a timeout until the worker thread finished.

        Instead, shutdown(wait=False) lets this method return promptly
        after interrupting the DuckDB connection.
        """

        def _run() -> pd.DataFrame:
            cursor = self._conn.execute(sql)
            return _fetch_dataframe(cursor)

        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )

        future = pool.submit(_run)

        try:
            result = future.result(
                timeout=self._timeout_seconds
            )

        except concurrent.futures.TimeoutError:
            # DuckDB's Python API exposes interrupt() as the practical
            # cooperative cancellation mechanism.
            self._conn.interrupt()

            pool.shutdown(wait=False)

            raise QueryTimeoutError(
                f"Query exceeded the {self._timeout_seconds}s "
                "timeout and was cancelled."
            ) from None

        except Exception as exc:
            pool.shutdown(wait=False)

            raise QueryExecutionError(
                f"DuckDB failed to execute the query: {exc}"
            ) from exc

        else:
            pool.shutdown(wait=False)
            return result


def _fetch_dataframe(cursor) -> pd.DataFrame:
    """
    Fetch a DuckDB cursor's result as a pandas DataFrame.

    Supports multiple DuckDB Python API variants.
    """
    if hasattr(cursor, "fetchdf"):
        return cursor.fetchdf()

    if hasattr(cursor, "fetch_df"):
        return cursor.fetch_df()

    if hasattr(cursor, "df"):
        return cursor.df()

    raise QueryExecutionError(
        "The installed duckdb version's cursor has no known "
        "DataFrame-fetch method (tried fetchdf, fetch_df, df)."
    )
