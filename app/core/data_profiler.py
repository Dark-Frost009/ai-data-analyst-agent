"""
Dataset profiling.

Takes the pandas DataFrame produced by `app.core.data_loader.load_csv()`
and computes a structural summary: row/column counts, per-column inferred
types, null/unique statistics, numeric min/max/mean where applicable, and
a small JSON-safe sample of rows.

This module is intentionally narrow in scope: it only *observes* the
DataFrame. It never mutates it, never touches DuckDB, Streamlit, or AWS,
and does not attempt analysis (outliers, correlations, duplicates) —
that belongs to later, LLM/SQL-driven milestones.

Design notes
------------
- Never mutates the input DataFrame. Every pandas operation used here
  (`.dropna()`, `.isna()`, `.min()`, `pd.to_datetime(...)`, `.head()`, ...)
  returns a new object; nothing is done in-place, and the original column
  dtype is never reassigned — even for columns detected as datetime-like.
- Datetime detection is bounded: only a capped sample (first 100
  non-null values) of an object/string column is test-parsed to decide
  *whether* it's datetime-like, so this stays cheap even on large
  datasets. Only once a column is confirmed datetime-like is the full
  column parsed (into a temporary, discarded value) to compute min/max.
- All numeric/temporal values (min/max/mean, and every sample-row cell)
  are normalized to plain Python types via `_json_safe()` before being
  placed into the returned model, so `DatasetProfile` is always safely
  JSON-serializable — no numpy scalars, no NaN/NaT, no raw Timestamps.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd

from app.models.schemas import ColumnProfile, ColumnType, DatasetProfile
from app.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SAMPLE_ROWS = 5

# Datetime-like detection: only a bounded sample is test-parsed, and a
# column is only classified as datetime if this fraction (or more) of
# the sample parses successfully. This avoids false positives on text
# columns where a handful of values coincidentally look like dates.
_DATETIME_DETECTION_SAMPLE_SIZE = 100
_DATETIME_DETECTION_THRESHOLD = 0.9


def profile_dataframe(df: pd.DataFrame, sample_rows: int = DEFAULT_SAMPLE_ROWS) -> DatasetProfile:
    """
    Compute a structural profile of a DataFrame.

    Parameters
    ----------
    df:
        The DataFrame to profile (e.g. from `load_csv()`). Never mutated.
    sample_rows:
        Number of leading rows to include as a sample. Defaults to 5.
        If `df` has fewer rows than this, all rows are included.

    Raises
    ------
    ValueError
        If `df` is not a pandas DataFrame, or `sample_rows` is negative.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if sample_rows < 0:
        raise ValueError("sample_rows must be >= 0")

    row_count = len(df)
    column_count = len(df.columns)

    columns = [_profile_column(str(col), df[col]) for col in df.columns]

    sample_records = df.head(sample_rows).to_dict(orient="records")
    sample_records = [
        {str(key): _json_safe(value) for key, value in row.items()} for row in sample_records
    ]

    profile = DatasetProfile(
        row_count=row_count,
        column_count=column_count,
        columns=columns,
        sample_rows=sample_records,
    )

    logger.info(
        "Profiled dataset | rows=%d | columns=%d | sample_rows=%d",
        row_count,
        column_count,
        len(sample_records),
    )

    return profile


def _profile_column(name: str, series: pd.Series) -> ColumnProfile:
    """Compute the profile for a single column, without mutating `series`."""
    total = len(series)
    non_null = series.dropna()
    null_count = int(series.isna().sum())
    null_percentage = round((null_count / total) * 100, 2) if total > 0 else 0.0
    unique_count = int(non_null.nunique())

    inferred_type = _classify_column(series, non_null)

    min_val: Any = None
    max_val: Any = None
    mean_val: Any = None

    if inferred_type in ("integer", "float") and len(non_null) > 0:
        min_val = _json_safe(non_null.min())
        max_val = _json_safe(non_null.max())
        mean_raw = non_null.mean()
        mean_val = None if pd.isna(mean_raw) else round(float(mean_raw), 4)
    elif inferred_type == "datetime" and len(non_null) > 0:
        # Full-column parse, done only now that the column is confirmed
        # datetime-like. The parsed values are used solely to compute
        # min/max and are then discarded — `series`'s own dtype is never
        # touched.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(non_null, errors="coerce")
            parsed_valid = parsed.dropna()
            if len(parsed_valid) > 0:
                min_val = parsed_valid.min().isoformat()
                max_val = parsed_valid.max().isoformat()
        except (TypeError, ValueError, OverflowError):
            pass  # leave min/max as None if the full-column parse unexpectedly fails
        # mean is deliberately left as None for datetime columns.

    return ColumnProfile(
        name=name,
        pandas_dtype=str(series.dtype),
        inferred_type=inferred_type,
        null_count=null_count,
        null_percentage=null_percentage,
        unique_count=unique_count,
        min=min_val,
        max=max_val,
        mean=mean_val,
    )


def _classify_column(series: pd.Series, non_null: pd.Series) -> ColumnType:
    """Determine the semantic type of a column from its dtype and values."""
    if len(non_null) == 0:
        return "empty"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        if _looks_like_datetime(non_null):
            return "datetime"
        # A boolean column with missing values can't be held in a native
        # bool array (no NaN support), so pandas upcasts it to object —
        # detect that case explicitly rather than letting it fall to
        # "mixed".
        if all(isinstance(v, (bool, np.bool_)) for v in non_null):
            return "boolean"
        if all(isinstance(v, str) for v in non_null):
            return "string"
        return "mixed"

    # Any other/unusual dtype (category, timedelta, complex, ...) —
    # a safe, non-crashing fallback.
    return "mixed"


def _looks_like_datetime(non_null: pd.Series) -> bool:
    """Bounded, non-mutating heuristic: does this column look date-like?"""
    sample = non_null.head(_DATETIME_DETECTION_SAMPLE_SIZE)
    if len(sample) == 0:
        return False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
    except (TypeError, ValueError, OverflowError):
        # Genuinely mixed non-string types (e.g. int/str/float in the
        # same object column) can raise here rather than coercing
        # cleanly, depending on pandas version. Treat any such failure
        # as "not datetime-like" rather than letting it propagate.
        return False

    success_ratio = parsed.notna().sum() / len(sample)
    return success_ratio >= _DATETIME_DETECTION_THRESHOLD


def _json_safe(value: Any) -> Any:
    """Normalize a single cell/stat value into a plain, JSON-safe Python type."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass  # value doesn't support isna checks (e.g. an unusual object) — pass through below

    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    return value