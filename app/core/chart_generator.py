"""
Chart generation utilities — the "Visualization Intelligence" stage of the
pipeline.

Converts a QueryResult containing a pandas DataFrame into a chart-friendly
specification that the Streamlit UI can render.

This module intentionally does NOT render charts itself. It only inspects
query results (and, optionally, light context about the question/SQL that
produced them) and determines:

- whether the result is suitable for visualization at all,
- which columns should be used for the x-axis and y-axis,
- which chart type is most appropriate,
- rendering hints (e.g. horizontal vs. vertical bar orientation), and
- a JSON-safe representation of the result data.

Supported chart types:
    - bar         (vertical or horizontal, via the "orientation" hint)
    - grouped_bar (one categorical dimension + 2+ numeric metrics, e.g.
                   "revenue vs cost by region" — each metric becomes its
                   own series within each category's group of bars)
    - line        (single-series, or multi-series when a time axis
                   carries several numeric columns)
    - pie
    - scatter

Core product rule
------------------
"If the result can meaningfully communicate information visually,
automatically visualize it. Do NOT generate meaningless charts."

The DataFrame is always the primary signal for this decision. An optional
`question`/`sql` string may be supplied purely as *supporting* context (for
example, to prefer a ranked bar chart over a pie chart for a "top N"
question) — they never override what the DataFrame itself supports, and a
caller that omits them gets the same DataFrame-only behavior as before.

High-cardinality categories
----------------------------
Bar, grouped_bar, and pie all cap themselves to `max_categories` rows.
When the (already-aggregated) result has more distinct categories than
that, the top categories are kept and everything beyond them is summed
into a single synthetic "Other" row — never silently dropped. A result
that was already constrained by the query itself (e.g. a generated
`ORDER BY ... LIMIT 10` with the default max_categories=20) naturally
ends up with nothing left over, so no "Other" row is fabricated in that
case: whether "Other" appears is judged purely from how many distinct
categories the returned DataFrame actually contains relative to
max_categories, exactly the same DataFrame-first principle used
everywhere else in this module. Chart specs that gained an "Other" row
set `has_other: True` and `other_count: <n>` so the UI can say so.

Design goals:
    - Never mutate the input DataFrame.
    - Never make network or AWS calls.
    - Never execute SQL.
    - Keep chart-selection deterministic and testable.
    - Return JSON-safe chart specifications.
    - Never raise merely because a result isn't visualizable — return None.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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

# Cardinality thresholds for choosing pie/donut over bar. A slightly higher
# ceiling applies when the question explicitly signals a "proportion of
# total" framing (e.g. "percentage of orders by category").
PIE_MAX_CATEGORIES_DEFAULT = 6
PIE_MAX_CATEGORIES_PROPORTION = 8

# Cap on how many numeric columns are drawn as separate lines in a
# multi-series time chart, so the chart stays legible.
DEFAULT_MAX_LINE_SERIES = 4

# Same idea for a grouped bar chart's metric series.
DEFAULT_MAX_GROUPED_BAR_SERIES = 4

SUPPORTED_CHART_TYPES = {
    "bar",
    "grouped_bar",
    "line",
    "pie",
    "scatter",
}

# Lightweight keyword signals used only to *tie-break* between chart types
# that are otherwise equally valid for the same DataFrame shape (e.g.
# category+numeric could reasonably be a bar or a pie). These never decide
# *whether* to chart — only *which* chart — and the DataFrame shape always
# has final say.
_RANKING_KEYWORDS = (
    "top",
    "highest",
    "lowest",
    "best",
    "worst",
    "most",
    "least",
    "rank",
    "ranked",
    "ranking",
    "bottom",
    "largest",
    "smallest",
)

_PROPORTION_KEYWORDS = (
    "percentage",
    "percent",
    "%",
    "share",
    "proportion",
    "distribution",
    "breakdown",
    "split",
    "composition",
)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def generate_chart(
    query_result: QueryResult,
    chart_type: Optional[str] = None,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
    max_rows: int = DEFAULT_MAX_ROWS,
    question: Optional[str] = None,
    sql: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generate a chart specification from a QueryResult.

    Parameters
    ----------
    query_result:
        QueryResult containing the pandas DataFrame to visualize. This is
        the primary — and in "automatic" mode, the only — signal used to
        decide whether and how to visualize the result.

    chart_type:
        Optional explicit chart type. Supported values are:
        "bar", "grouped_bar", "line", "pie", and "scatter".

        If omitted (None / "Automatic" in the UI), the chart type — and
        whether to chart at all — is inferred automatically from the
        DataFrame's shape.

        If provided, that type is attempted against the DataFrame. When
        the data isn't compatible with the requested type (e.g. "scatter"
        with only one numeric column), this returns None rather than
        raising or fabricating data, so the caller can fail gracefully.

    max_categories:
        Maximum number of categorical values to include in a chart.

    max_rows:
        Maximum number of rows copied into the chart specification.

    question:
        Optional natural-language question that produced this result.
        Used only as a *tie-breaker* between otherwise-equally-valid chart
        types for the same DataFrame shape (e.g. preferring a ranked bar
        chart over a pie chart for "top 10 ..." questions). Never expands
        *whether* a chart is shown — that is decided from the DataFrame.

    sql:
        Optional generated SQL for the same purpose as `question` (e.g.
        detecting an `ORDER BY ... LIMIT` ranking query).

    Returns
    -------
    Optional[Dict[str, Any]]
        A JSON-safe chart specification, or None when the result cannot
        reasonably be visualized (or isn't useful to visualize).

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

    is_ranking = _looks_like_ranking(question, sql)
    is_proportion = _looks_like_proportion(question)

    selected_type = chart_type or _infer_chart_type(
        df=df,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        is_ranking=is_ranking,
        is_proportion=is_proportion,
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
            orientation="horizontal" if is_ranking else "vertical",
        )

    elif selected_type == "grouped_bar":
        spec = _build_grouped_bar_chart(
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
    is_ranking: bool = False,
    is_proportion: bool = False,
) -> Optional[str]:
    """
    Infer the most useful chart type from DataFrame structure.

    This is the core "should we chart, and with what" decision for
    Automatic mode. `is_ranking`/`is_proportion` only ever choose between
    chart types that the DataFrame shape already supports — they cannot
    conjure a chart out of data that doesn't support one.

    Priority:
        1. A single row (or no rows) carries nothing to visually compare —
           a metric or table communicates it better than a chart. No chart.
        2. line for datetime + numeric (single or multi-series).
        3. scatter for two or more numeric columns (no datetime axis).
        4. categorical + 2 or more numeric metrics: grouped_bar (e.g.
           "revenue vs cost by region") — a single metric could reasonably
           be squeezed into a pie, but two or more genuinely need their
           own series, not a two-point scatter that discards the category.
        5. categorical + exactly one numeric metric: pie for a small,
           proportion-flavored distribution; horizontal-bar-flagged bar
           for a ranking question; otherwise pie for small cardinality,
           bar for larger cardinality.
        6. None — no combination of columns says anything visually.
    """

    # A single row (e.g. "which customer had the highest purchase?") or an
    # empty frame has nothing to compare visually — a metric/table is the
    # right presentation, not a chart with one bar/slice/point.
    if len(df) <= 1:
        return None

    if datetime_columns and numeric_columns:
        return "line"

    if len(numeric_columns) >= 2 and not categorical_columns:
        return "scatter"

    if categorical_columns and numeric_columns:

        if len(numeric_columns) >= 2:
            return "grouped_bar"

        category = categorical_columns[0]
        cardinality = df[category].nunique(dropna=True)

        if is_proportion and cardinality <= PIE_MAX_CATEGORIES_PROPORTION:
            return "pie"

        if is_ranking:
            return "bar"

        if cardinality <= PIE_MAX_CATEGORIES_DEFAULT:
            return "pie"

        return "bar"

    return None


# --------------------------------------------------------------------------
# Question/SQL context signals (tie-breakers only, see module docstring)
# --------------------------------------------------------------------------


def _looks_like_ranking(
    question: Optional[str],
    sql: Optional[str],
) -> bool:
    """Detect a "top N" / ranking framing from the question or SQL."""

    combined = f"{question or ''} {sql or ''}".strip().lower()

    if not combined:
        return False

    if any(keyword in combined for keyword in _RANKING_KEYWORDS):
        return True

    # A generated `ORDER BY ... LIMIT ...` is a strong structural signal
    # of a ranking query even when the question's wording doesn't use an
    # obvious ranking word (e.g. "which products sell the most?").
    return "order by" in combined and "limit" in combined


def _looks_like_proportion(question: Optional[str]) -> bool:
    """Detect a "share of the whole" framing from the question."""

    if not question:
        return False

    lowered = question.lower()

    return any(keyword in lowered for keyword in _PROPORTION_KEYWORDS)


# --------------------------------------------------------------------------
# Individual chart builders
# --------------------------------------------------------------------------


def _apply_top_n_with_other(
    working: pd.DataFrame,
    category_column: str,
    value_columns: List[str],
    max_categories: int,
) -> Tuple[pd.DataFrame, int]:
    """
    Cap an already-aggregated, descending-sorted-by-primary-metric
    DataFrame to at most `max_categories` rows.

    - If the data already fits within `max_categories`, it is returned
      unchanged. This is what keeps an explicitly-limited ranking query
      (e.g. a generated `ORDER BY ... LIMIT 10` with the default
      max_categories=20) from ever gaining a fabricated "Other" row —
      there is nothing left over to aggregate, because the DataFrame
      itself doesn't contain any more categories than we're charting.

    - If there ARE more categories than `max_categories` allows, the
      top `max_categories - 1` categories are kept as-is and everything
      else is summed into one synthetic "Other" row, preserving the
      total. "Other" is always appended last — it's a catch-all bucket,
      not a ranked value, so it is never re-sorted into the list by its
      (possibly large) aggregate value.

    Returns the possibly-truncated DataFrame plus the count of original
    categories that were folded into "Other" (0 when nothing was).
    """

    total_categories = len(working)

    if total_categories <= max_categories:
        return working.reset_index(drop=True), 0

    keep = max(max_categories - 1, 0)
    top = working.head(keep)
    remainder = working.iloc[keep:]

    if remainder.empty:
        return top.reset_index(drop=True), 0

    other_row: Dict[str, Any] = {category_column: "Other"}

    for column in value_columns:
        other_row[column] = remainder[column].sum()

    combined = pd.concat(
        [top, pd.DataFrame([other_row])],
        ignore_index=True,
    )

    return combined, len(remainder)


def _build_bar_chart(
    df: pd.DataFrame,
    categorical_columns: List[str],
    numeric_columns: List[str],
    max_categories: int,
    orientation: str = "vertical",
) -> Optional[Dict[str, Any]]:
    """
    Build a bar chart from categorical and numeric columns.

    `orientation` is a rendering hint only ("horizontal" or "vertical") —
    it does not affect which rows are selected. Callers typically request
    "horizontal" for ranking/top-N results, where long category labels
    (e.g. restaurant names) read better on a horizontal axis.

    When there are more distinct categories than `max_categories`, the
    remainder is folded into a single "Other" row rather than dropped —
    see `_apply_top_n_with_other`.
    """

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
        .reset_index(drop=True)
    )

    if working.empty:
        return None

    working, other_count = _apply_top_n_with_other(
        working,
        category_column=category_column,
        value_columns=[value_column],
        max_categories=max_categories,
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
        "orientation": (
            "horizontal" if orientation == "horizontal" else "vertical"
        ),
        "data": data,
        "has_other": other_count > 0,
        "other_count": other_count,
    }


def _build_grouped_bar_chart(
    df: pd.DataFrame,
    categorical_columns: List[str],
    numeric_columns: List[str],
    max_categories: int,
    max_series: int = DEFAULT_MAX_GROUPED_BAR_SERIES,
) -> Optional[Dict[str, Any]]:
    """
    Build a grouped (multi-series) bar chart: one categorical dimension
    plus two or more numeric metrics, e.g. "revenue vs cost by region"
    or "actual vs budget by department". Every metric becomes its own
    series within each category's group of bars.

    High-cardinality categories are handled the same way as a plain bar
    chart — see `_apply_top_n_with_other`. Metrics beyond `max_series`
    are dropped (not aggregated) purely to keep the chart legible; the
    metrics kept are always the leading `numeric_columns`, in order.
    """

    if not categorical_columns or len(numeric_columns) < 2:
        return None

    category_column = categorical_columns[0]
    value_columns = numeric_columns[:max_series]

    working = df[[category_column, *value_columns]].copy()
    working = working.dropna(subset=[category_column, *value_columns])

    if working.empty:
        return None

    # Aggregate duplicate categories, ranking by the first metric.
    working = (
        working.groupby(category_column, as_index=False)[value_columns]
        .sum()
        .sort_values(value_columns[0], ascending=False)
        .reset_index(drop=True)
    )

    if working.empty:
        return None

    working, other_count = _apply_top_n_with_other(
        working,
        category_column=category_column,
        value_columns=value_columns,
        max_categories=max_categories,
    )

    if working.empty:
        return None

    data = [
        {
            "category": _json_safe(row[category_column]),
            **{
                column: _json_safe(row[column])
                for column in value_columns
            },
        }
        for _, row in working.iterrows()
    ]

    return {
        "chart_type": "grouped_bar",
        "title": f"{', '.join(value_columns)} by {category_column}",
        "x_column": category_column,
        "y_columns": value_columns,
        "data": data,
        "has_other": other_count > 0,
        "other_count": other_count,
    }


def _build_line_chart(
    df: pd.DataFrame,
    numeric_columns: List[str],
    datetime_columns: List[str],
    max_rows: int,
    max_series: int = DEFAULT_MAX_LINE_SERIES,
) -> Optional[Dict[str, Any]]:
    """
    Build a line chart using datetime/numeric data.

    When a time axis is available and the result carries more than one
    numeric column (e.g. "revenue" and "profit" over "month"), this
    produces a multi-series line chart: every data point is a single
    dict keyed by column name (plus "x"), and `y_columns` lists the
    series in order. Callers that only care about a single series can
    keep reading `y_column` (the first series) as before.

    With exactly one numeric column, the output is unchanged from before:
    each data point is `{"x": ..., "y": ...}` and only `y_column` is set.
    """

    if not numeric_columns:
        return None

    if datetime_columns:
        x_column = datetime_columns[0]
    else:
        # For purely numeric data, use the DataFrame index as x-axis.
        x_column = "__index__"

    use_multi_series = bool(datetime_columns) and len(numeric_columns) > 1
    value_columns = (
        numeric_columns[:max_series] if use_multi_series else numeric_columns[:1]
    )

    if x_column == "__index__":
        working = df[value_columns].copy()
        working.insert(0, x_column, range(len(working)))
    else:
        working = df[[x_column, *value_columns]].copy()
        working[x_column] = pd.to_datetime(
            working[x_column],
            errors="coerce",
        )
        working = working.dropna(subset=[x_column])

    if working.empty:
        return None

    working = working.sort_values(x_column).head(max_rows)

    if use_multi_series:
        data = [
            {
                "x": _json_safe(row[x_column]),
                **{
                    column: _json_safe(row[column])
                    for column in value_columns
                },
            }
            for _, row in working.iterrows()
        ]

        if not data:
            return None

        return {
            "chart_type": "line",
            "title": f"{', '.join(value_columns)} over {x_column}",
            "x_column": x_column,
            "y_column": value_columns[0],
            "y_columns": value_columns,
            "data": data,
        }

    value_column = value_columns[0]

    working = working.dropna(subset=[value_column])

    if working.empty:
        return None

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
    """
    Build a pie chart for a small categorical distribution.

    When there are more distinct categories than `max_categories`
    (relevant mainly for an explicit user override on a high-cardinality
    result, since automatic inference already keeps pie to small
    cardinalities), the remainder is folded into a single "Other" slice
    rather than dropped — see `_apply_top_n_with_other`.
    """

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
        .reset_index(drop=True)
    )

    if working.empty:
        return None

    working, other_count = _apply_top_n_with_other(
        working,
        category_column=category_column,
        value_columns=[value_column],
        max_categories=max_categories,
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
        "has_other": other_count > 0,
        "other_count": other_count,
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