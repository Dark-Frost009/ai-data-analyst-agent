"""
Typed data contracts shared across pipeline stages.

`ColumnProfile`/`DatasetProfile` are the shape
`app.core.data_profiler.profile_dataframe()` returns.
`ValidationError`/`ValidationResult` are the shape
`app.utils.security.validate_sql()` returns.
`QueryResult` is the shape `app.core.sql_executor.SQLExecutor.execute()`
returns.

These live here, not inside the modules that produce them, because later
milestones consume these same shapes too — the query planner will read
`DatasetProfile` for prompt construction, and `chart_generator.py` /
`explainer.py` will consume `QueryResult` directly.

Using Pydantic (rather than plain dataclasses) also gets clean, built-in
JSON serialization for free, which matters once these start getting
embedded into LLM prompts or returned from an API layer. `QueryResult` is
the one exception to that JSON-friendliness — it holds a real pandas
DataFrame (the actual query result data for later stages to consume),
which Pydantic cannot serialize or deep-validate, so it's an internal
pipeline artifact only, never sent to the LLM or serialized directly.
"""

from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

ColumnType = Literal["integer", "float", "boolean", "datetime", "string", "mixed", "empty"]


class ColumnProfile(BaseModel):
    """Profile of a single DataFrame column."""

    name: str
    pandas_dtype: str
    inferred_type: ColumnType
    null_count: int
    null_percentage: float
    unique_count: int
    min: Optional[Any] = None
    max: Optional[Any] = None
    mean: Optional[float] = None


class DatasetProfile(BaseModel):
    """Profile of an entire DataFrame."""

    row_count: int
    column_count: int
    columns: List[ColumnProfile]
    sample_rows: List[Dict[str, Any]]

    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]


class ValidationError(BaseModel):
    """A single reason a SQL statement was rejected by the security layer."""

    code: str
    message: str
    detail: Optional[str] = None


class ValidationResult(BaseModel):
    """
    Result of validating a single SQL statement before execution.

    `sql_executor.py` (a later milestone) is expected to accept only a
    `ValidationResult` with `is_valid=True` — never a raw SQL string —
    so this is the sole gate between LLM-generated SQL and DuckDB.
    """

    is_valid: bool
    cleaned_sql: Optional[str] = None
    errors: List[ValidationError] = Field(default_factory=list)


class QueryResult(BaseModel):
    """
    Result of executing an already-validated SQL query against DuckDB.

    Unlike the other models in this file, `dataframe` holds a real
    pandas DataFrame rather than JSON-safe primitives — this model is an
    internal pipeline artifact for chart_generator.py/explainer.py to
    consume, not something serialized to JSON or sent to the LLM as-is.
    `arbitrary_types_allowed` is required for Pydantic to accept a
    DataFrame as a field type at all.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataframe: pd.DataFrame
    row_count: int
    truncated: bool