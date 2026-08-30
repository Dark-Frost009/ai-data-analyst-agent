"""
Chart generation utilities.

Converts a QueryResult containing a pandas DataFrame into a chart-friendly
specification that the Streamlit UI can render.

This module intentionally does NOT render charts itself. It only inspects
query results and determines:

- whether the result is suitable for visualization,
- which columns should be used for the x-axis and y-axis,
- which chart type is most appropriate,
- and a JSON-safe representation of the result data.

Supported chart types:
    - bar
    - line
    - pie
    - scatter

Design goals:
    - Never mutate the input DataFrame.
    - Never make network or AWS calls.
    - Never execute SQL.
    - Keep chart-selection deterministic and testable.
    - Return JSON-safe chart specifications.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.models.schemas import QueryResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DEFAULT_MAX_CATEGORIES = 20
DEFAULT_MAX_ROWS = 1000

SUPPORTED_CHART_TYPES = {
    "bar",
    "line",
    "pie",
    "scatter",
}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def generate_chart(
    query_result: QueryResult,
    chart_type: Optional[str] = None,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> Optional[Dict[str, Any]]:
    """
    Generate a chart specification from a QueryResult.

    Parameters
    ----------
    query_result:
        QueryResult containing the pandas DataFrame to visualize.

    chart_type:
        Optional explicit chart type. Supported values are:
        "bar", "line", "pie", and "scatter".

        If omitted, the chart type is inferred automatically.

    max_categories:
        Maximum number of categorical values to include in a chart.

    max_rows:
        Maximum number of rows copied into the chart specification.

    Returns
    -------
    Optional[Dict[str, Any]]
        A JSON-safe chart specification, or None when the result cannot
        reasonably be visualized.

    Raises
    ------
    ValueError
        If chart_type or limits are invalid.
    """

    if not isinstance(query_result, QueryResult):
        raise ValueError("query_result must be a QueryResult")

    if max_categories <= 0:
        raise ValueError("max_categories must be greater than 0")

    if max_rows <= 0:
        raise ValueError("max_rows must be greater than 0")

    if chart_type is not None:
        chart_type = chart_type.strip().lower()

        if chart_type not in SUPPORTED_CHART_TYPES:
            raise ValueError(
                f"Unsupported chart_type '{chart_type}'. "
                f"Supported types: {sorted(SUPPORTED_CHART_TYPES)}"
            )

    df = query_result.dataframe

    if not isinstance(df, pd.DataFrame):
        raise ValueError("query_result.dataframe must be a pandas DataFrame")

    if df.empty or len(df.columns) == 0:
        logger.info("Skipping chart generation because query result is empty")
        return None

    numeric_columns = _numeric_columns(df)
    categorical_columns = _categorical_columns(df)
    datetime_columns = _datetime_columns(df)

    selected_type = chart_type or _infer_chart_type(
        df=df,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
    )

    if selected_type is None:
        logger.info(
            "No suitable chart type found | rows=%d | columns=%d",
            len(df),
            len(df.columns),
        )
        return None

    if selected_type == "scatter":
        spec = _build_scatter_chart(
            df,
            numeric_columns,
            max_rows=max_rows,
        )

    elif selected_type == "line":
        spec = _build_line_chart(
            df,
            numeric_columns,
            datetime_columns,
            max_rows=max_rows,
        )

    elif selected_type == "pie":
        spec = _build_pie_chart(
            df,
            categorical_columns,
            numeric_columns,
            max_categories=max_categories,
        )

    elif selected_type == "bar":
        spec = _build_bar_chart(
            df,
            categorical_columns,
            numeric_columns,
            max_categories=max_categories,
        )

    else:
        # Defensive guard. The value is already validated above.
        return None

    if spec is None:
        logger.info(
            "Chart generation produced no usable specification | type=%s",
            selected_type,
        )
        return None

    logger.info(
        "Generated chart | type=%s | rows=%d | columns=%d",
        selected_type,
        len(df),
        len(df.columns),
    )

    return spec


# --------------------------------------------------------------------------
# Chart type inference
# --------------------------------------------------------------------------


def _infer_chart_type(
    df: pd.DataFrame,
    numeric_columns: List[str],
    categorical_columns: List[str],
    datetime_columns: List[str],
) -> Optional[str]:
    """
    Infer the most useful chart type from DataFrame structure.

    Priority:
        1. scatter for two or more numeric columns with row-level data
        2. line for datetime + numeric
        3. pie for one categorical + one numeric with small cardinality
        4. bar for categorical + numeric
        5. line for multiple numeric columns
        6. None
    """

    if len(numeric_columns) >= 2 and len(df) >= 2:
        return "scatter"

    if datetime_columns and numeric_columns:
        return "line"

    if categorical_columns and numeric_columns:
        category = categorical_columns[0]

        if df[category].nunique(dropna=True) <= 6:
            return "pie"

        return "bar"

    if len(numeric_columns) >= 2:
        return "line"

    return None


# --------------------------------------------------------------------------
# Individual chart builders
# --------------------------------------------------------------------------


def _build_bar_chart(
    df: pd.DataFrame,
    categorical_columns: List[str],
    numeric_columns: List[str],
    max_categories: int,
) -> Optional[Dict[str, Any]]:
    """Build a bar chart from categorical and numeric columns."""

    if not categorical_columns or not numeric_columns:
        return None

    category_column = categorical_columns[0]
    value_column = numeric_columns[0]

    working = df[[category_column, value_column]].copy()

    working = working.dropna(subset=[category_column, value_column])

    if working.empty:
        return None

    # Aggregate duplicate categories so the chart remains meaningful.
    working = (
        working.groupby(category_column, as_index=False)[value_column]
        .sum()
        .sort_values(value_column, ascending=False)
        .head(max_categories)
    )

    if working.empty:
        return None

    data = [
        {
            "category": _json_safe(row[category_column]),
            "value": _json_safe(row[value_column]),
        }
        for _, row in working.iterrows()
    ]

    return {
        "chart_type": "bar",
        "title": f"{value_column} by {category_column}",
        "x_column": category_column,
        "y_column": value_column,
        "data": data,
    }


def _build_line_chart(
    df: pd.DataFrame,
    numeric_columns: List[str],
    datetime_columns: List[str],
    max_rows: int,
) -> Optional[Dict[str, Any]]:
    """Build a line chart using datetime/numeric data."""

    if not numeric_columns:
        return None

    if datetime_columns:
        x_column = datetime_columns[0]
    else:
        # For purely numeric data, use the DataFrame index as x-axis.
        x_column = "__index__"

    value_column = numeric_columns[0]

    if x_column == "__index__":
        working = df[[value_column]].copy()
        working.insert(0, x_column, range(len(working)))
    else:
        working = df[[x_column, value_column]].copy()
        working[x_column] = pd.to_datetime(
            working[x_column],
            errors="coerce",
        )
        working = working.dropna(subset=[x_column, value_column])

    if working.empty:
        return None

    working = working.sort_values(x_column).head(max_rows)

    data = [
        {
            "x": _json_safe(row[x_column]),
            "y": _json_safe(row[value_column]),
        }
        for _, row in working.iterrows()
    ]

    return {
        "chart_type": "line",
        "title": f"{value_column} over {x_column}",
        "x_column": x_column,
        "y_column": value_column,
        "data": data,
    }


def _build_pie_chart(
    df: pd.DataFrame,
    categorical_columns: List[str],
    numeric_columns: List[str],
    max_categories: int,
) -> Optional[Dict[str, Any]]:
    """Build a pie chart for a small categorical distribution."""

    if not categorical_columns or not numeric_columns:
        return None

    category_column = categorical_columns[0]
    value_column = numeric_columns[0]

    working = df[[category_column, value_column]].copy()
    working = working.dropna(subset=[category_column, value_column])

    if working.empty:
        return None

    working = (
        working.groupby(category_column, as_index=False)[value_column]
        .sum()
        .sort_values(value_column, ascending=False)
        .head(max_categories)
    )

    if working.empty:
        return None

    data = [
        {
            "label": _json_safe(row[category_column]),
            "value": _json_safe(row[value_column]),
        }
        for _, row in working.iterrows()
    ]

    return {
        "chart_type": "pie",
        "title": f"{value_column} by {category_column}",
        "label_column": category_column,
        "value_column": value_column,
        "data": data,
    }


def _build_scatter_chart(
    df: pd.DataFrame,
    numeric_columns: List[str],
    max_rows: int,
) -> Optional[Dict[str, Any]]:
    """Build a scatter chart using the first two numeric columns."""

    if len(numeric_columns) < 2:
        return None

    x_column = numeric_columns[0]
    y_column = numeric_columns[1]

    working = df[[x_column, y_column]].copy()
    working = working.dropna(subset=[x_column, y_column]).head(max_rows)

    if working.empty:
        return None

    data = [
        {
            "x": _json_safe(row[x_column]),
            "y": _json_safe(row[y_column]),
        }
        for _, row in working.iterrows()
    ]

    return {
        "chart_type": "scatter",
        "title": f"{y_column} vs {x_column}",
        "x_column": x_column,
        "y_column": y_column,
        "data": data,
    }


# --------------------------------------------------------------------------
# Column classification helpers
# --------------------------------------------------------------------------


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric columns excluding boolean columns."""

    return [
        str(column)
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
    ]


def _datetime_columns(df: pd.DataFrame) -> List[str]:
    """Return columns that are already datetime-like."""

    return [
        str(column)
        for column in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[column])
    ]


def _categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Return columns suitable as categorical chart dimensions.

    Numeric, datetime, and boolean columns are excluded.
    """

    columns: List[str] = []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            continue

        if pd.api.types.is_bool_dtype(series):
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            columns.append(str(column))

    return columns


# --------------------------------------------------------------------------
# JSON safety
# --------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """
    Convert pandas/numpy values into ordinary JSON-safe Python values.

    NaN and NaT become None.
    """

    try:
        missing = pd.isna(value)

        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None

    except (TypeError, ValueError):
        pass

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()

    if isinstance(value, pd.Timedelta):
        return str(value)

    # Convert numpy arrays or other obvious numpy objects conservatively.
    if isinstance(value, np.ndarray):
        return value.tolist()

    return value