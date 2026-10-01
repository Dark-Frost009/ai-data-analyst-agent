"""Prepare an isolated execution copy using conservative, lossless coercions.

Profiling heuristics are not evidence that conversion is safe. Each coercion
must inspect every non-null value and either convert the entire column or
decline. Add future safe coercions to _COERCIONS without changing the loader
or executor. Unsupported, mixed and ambiguous representations stay unchanged.
"""

import re
from typing import Callable

import pandas as pd


def _coerce_datetime(series: pd.Series) -> pd.Series | None:
    """Accept ISO calendar dates or consistently evidenced US slash dates.

    US ordering requires at least one day > 12; otherwise month/day ordering
    is ambiguous. No sampling, fuzzy parsing, invalid-to-null conversion,
    bare years, mixed formats, or implicit timezone changes are allowed.
    Already typed datetimes are retained by the copy operation.
    """
    if not (pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)):
        return None
    values = series.dropna()
    if values.empty or not all(isinstance(value, str) for value in values):
        return None

    if all(re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) for value in values):
        date_format = "%Y-%m-%d"
    elif all(re.fullmatch(r"[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}", value) for value in values):
        parts = [tuple(map(int, value.split("/"))) for value in values]
        if not all(1 <= month <= 12 for month, day, year in parts):
            return None
        if not any(day > 12 for month, day, year in parts):
            return None
        date_format = "%m/%d/%Y"
    else:
        return None

    try:
        parsed = pd.to_datetime(series, format=date_format, exact=True, errors="raise")
    except (ValueError, TypeError, OverflowError):
        return None
    if not parsed.isna().equals(series.isna()):
        return None
    return parsed


_COERCIONS: tuple[Callable[[pd.Series], pd.Series | None], ...] = (
    _coerce_datetime,
)


def prepare_execution_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a typed copy without mutating source values, dtypes or metadata."""
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe must be a pandas DataFrame")
    execution = dataframe.copy(deep=True)
    for position in range(len(execution.columns)):
        series = execution.iloc[:, position]
        for coercion in _COERCIONS:
            converted = coercion(series)
            if converted is not None:
                execution.isetitem(position, converted)
                break
    return execution
