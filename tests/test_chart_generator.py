"""
Tests for app.core.chart_generator.

These tests are completely local:
    - no AWS
    - no Bedrock
    - no Streamlit
    - no DuckDB
    - no network access

The tests verify chart selection, chart structure, JSON safety,
input immutability, and invalid-input handling.
"""

import json

import numpy as np
import pandas as pd
import pytest

from app.core.chart_generator import generate_chart
from app.models.schemas import QueryResult


def _query_result(df: pd.DataFrame) -> QueryResult:
    """Create a QueryResult for testing."""

    return QueryResult(
        dataframe=df,
        row_count=len(df),
        truncated=False,
    )


# --------------------------------------------------------------------------
# 1. Basic bar chart
# --------------------------------------------------------------------------


def test_generate_bar_chart_from_category_and_numeric_data():
    df = pd.DataFrame(
        {
            "city": ["Delhi", "Mumbai", "Delhi", "Kolkata"],
            "sales": [100, 200, 150, 50],
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="bar",
    )

    assert result is not None
    assert result["chart_type"] == "bar"
    assert result["x_column"] == "city"
    assert result["y_column"] == "sales"

    # Delhi should be aggregated to 250.
    assert result["data"][0]["category"] == "Delhi"
    assert result["data"][0]["value"] == 250


# --------------------------------------------------------------------------
# 2. Pie chart
# --------------------------------------------------------------------------


def test_generate_pie_chart():
    df = pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "revenue": [50, 30, 20],
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="pie",
    )

    assert result is not None
    assert result["chart_type"] == "pie"
    assert result["label_column"] == "category"
    assert result["value_column"] == "revenue"

    assert len(result["data"]) == 3
    assert result["data"][0]["label"] == "A"
    assert result["data"][0]["value"] == 50


# --------------------------------------------------------------------------
# 3. Scatter chart
# --------------------------------------------------------------------------


def test_generate_scatter_chart():
    df = pd.DataFrame(
        {
            "advertising": [10, 20, 30],
            "revenue": [100, 180, 260],
            "city": ["A", "B", "C"],
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="scatter",
    )

    assert result is not None
    assert result["chart_type"] == "scatter"
    assert result["x_column"] == "advertising"
    assert result["y_column"] == "revenue"

    assert result["data"][0] == {
        "x": 10,
        "y": 100,
    }


# --------------------------------------------------------------------------
# 4. Line chart with datetime
# --------------------------------------------------------------------------


def test_generate_line_chart_with_datetime():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-03-01",
                    "2024-01-01",
                    "2024-02-01",
                ]
            ),
            "sales": [300, 100, 200],
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="line",
    )

    assert result is not None
    assert result["chart_type"] == "line"
    assert result["x_column"] == "date"
    assert result["y_column"] == "sales"

    # The line chart should sort dates.
    assert result["data"][0]["x"] == "2024-01-01T00:00:00"
    assert result["data"][0]["y"] == 100


# --------------------------------------------------------------------------
# 5. Automatic chart selection: scatter
# --------------------------------------------------------------------------


def test_auto_selects_scatter_for_two_numeric_columns():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "income": [1000, 2000, 3000],
        }
    )

    result = generate_chart(_query_result(df))

    assert result is not None
    assert result["chart_type"] == "scatter"


# --------------------------------------------------------------------------
# 6. Automatic chart selection: line
# --------------------------------------------------------------------------


def test_auto_selects_line_for_datetime_and_numeric():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                ]
            ),
            "orders": [10, 20, 30],
        }
    )

    result = generate_chart(_query_result(df))

    assert result is not None
    assert result["chart_type"] == "line"


# --------------------------------------------------------------------------
# 7. Automatic chart selection: pie
# --------------------------------------------------------------------------


def test_auto_selects_pie_for_small_category_distribution():
    df = pd.DataFrame(
        {
            "department": ["Sales", "HR", "IT"],
            "employees": [50, 20, 30],
        }
    )

    result = generate_chart(_query_result(df))

    assert result is not None
    assert result["chart_type"] == "pie"


# --------------------------------------------------------------------------
# 8. Automatic chart selection: bar
# --------------------------------------------------------------------------


def test_auto_selects_bar_for_many_categories():
    df = pd.DataFrame(
        {
            "product": [f"Product {i}" for i in range(10)],
            "sales": list(range(10, 20)),
        }
    )

    result = generate_chart(_query_result(df))

    assert result is not None
    assert result["chart_type"] == "bar"


# --------------------------------------------------------------------------
# 9. Empty DataFrame
# --------------------------------------------------------------------------


def test_empty_dataframe_returns_none():
    df = pd.DataFrame(
        {
            "category": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
        }
    )

    result = generate_chart(_query_result(df))

    assert result is None


# --------------------------------------------------------------------------
# 10. DataFrame with no useful chart dimensions
# --------------------------------------------------------------------------


def test_non_numeric_single_string_column_returns_none():
    df = pd.DataFrame(
        {
            "description": [
                "hello",
                "world",
                "example",
            ]
        }
    )

    result = generate_chart(_query_result(df))

    assert result is None


# --------------------------------------------------------------------------
# 11. Explicit invalid chart type
# --------------------------------------------------------------------------


def test_invalid_chart_type_is_rejected():
    df = pd.DataFrame(
        {
            "category": ["A", "B"],
            "value": [10, 20],
        }
    )

    with pytest.raises(ValueError):
        generate_chart(
            _query_result(df),
            chart_type="histogram",
        )


# --------------------------------------------------------------------------
# 12. Invalid max_categories
# --------------------------------------------------------------------------


def test_invalid_max_categories_is_rejected():
    df = pd.DataFrame(
        {
            "category": ["A", "B"],
            "value": [10, 20],
        }
    )

    with pytest.raises(ValueError):
        generate_chart(
            _query_result(df),
            max_categories=0,
        )


# --------------------------------------------------------------------------
# 13. Invalid max_rows
# --------------------------------------------------------------------------


def test_invalid_max_rows_is_rejected():
    df = pd.DataFrame(
        {
            "x": [1, 2],
            "y": [10, 20],
        }
    )

    with pytest.raises(ValueError):
        generate_chart(
            _query_result(df),
            max_rows=0,
        )


# --------------------------------------------------------------------------
# 14. Invalid QueryResult
# --------------------------------------------------------------------------


def test_invalid_query_result_is_rejected():
    with pytest.raises(ValueError):
        generate_chart(None)


# --------------------------------------------------------------------------
# 15. JSON safety
# --------------------------------------------------------------------------


def test_chart_result_is_json_serializable():
    df = pd.DataFrame(
        {
            "category": np.array(["A", "B", "C"]),
            "value": np.array(
                [10, 20, 30],
                dtype=np.int64,
            ),
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="bar",
    )

    assert result is not None

    serialized = json.dumps(
        result,
        allow_nan=False,
    )

    assert "category" in serialized
    assert "value" in serialized


# --------------------------------------------------------------------------
# 16. NaN values become None
# --------------------------------------------------------------------------


def test_nan_values_are_removed_from_chart_data():
    df = pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "value": [10.0, np.nan, 30.0],
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="bar",
    )

    assert result is not None

    # NaN row is dropped because it cannot produce a meaningful chart point.
    assert len(result["data"]) == 2

    json.dumps(
        result,
        allow_nan=False,
    )


# --------------------------------------------------------------------------
# 17. Input DataFrame is not mutated
# --------------------------------------------------------------------------


def test_generate_chart_does_not_mutate_input_dataframe():
    df = pd.DataFrame(
        {
            "city": ["Delhi", "Mumbai", "Kolkata"],
            "sales": [100, 200, 150],
        }
    )

    snapshot = df.copy(deep=True)

    generate_chart(
        _query_result(df),
        chart_type="bar",
    )

    pd.testing.assert_frame_equal(
        df,
        snapshot,
    )


# --------------------------------------------------------------------------
# 18. max_categories is respected
# --------------------------------------------------------------------------


def test_bar_chart_respects_max_categories():
    """
    Exceeding max_categories now produces Top-(N-1) categories plus a
    single "Other" row summing the remainder, rather than silently
    dropping the tail (see the Top-N + "Other" requirement). "Other" is
    always appended last, not re-ranked by its aggregate value.
    """

    df = pd.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "value": [50, 40, 30, 20, 10],
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="bar",
        max_categories=3,
    )

    assert result is not None
    assert len(result["data"]) == 3

    categories = [
        row["category"]
        for row in result["data"]
    ]

    assert categories == ["A", "B", "Other"]

    other_row = result["data"][-1]

    assert other_row["value"] == 60  # 30 + 20 + 10
    assert result["has_other"] is True
    assert result["other_count"] == 3


# --------------------------------------------------------------------------
# 19. max_rows is respected
# --------------------------------------------------------------------------


def test_scatter_chart_respects_max_rows():
    df = pd.DataFrame(
        {
            "x": range(100),
            "y": range(1000, 1100),
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="scatter",
        max_rows=10,
    )

    assert result is not None
    assert len(result["data"]) == 10


# --------------------------------------------------------------------------
# 20. Boolean columns are not treated as numeric
# --------------------------------------------------------------------------


def test_boolean_columns_are_not_used_as_numeric_axes():
    df = pd.DataFrame(
        {
            "is_active": [True, False, True],
            "name": ["A", "B", "C"],
        }
    )

    result = generate_chart(_query_result(df))

    assert result is None


# --------------------------------------------------------------------------
# 21. Single-row results are not charted automatically
# --------------------------------------------------------------------------


def test_single_row_result_returns_no_chart_automatically():
    """
    "Which customer has the highest purchase?" style results: a single
    row with a category and a numeric value has nothing to visually
    compare. Automatic mode should present it as a metric/table, not a
    one-slice pie or one-bar chart.
    """

    df = pd.DataFrame(
        {
            "customer": ["Alice"],
            "amount": [999],
        }
    )

    result = generate_chart(
        _query_result(df),
        question="Which customer has the highest purchase?",
    )

    assert result is None


def test_single_row_single_numeric_value_returns_no_chart():
    df = pd.DataFrame({"total_revenue": [5000]})

    result = generate_chart(
        _query_result(df),
        question="What is the total revenue?",
    )

    assert result is None


def test_explicit_chart_type_on_single_row_is_still_honored():
    """
    Automatic suppression of single-row results should not apply when the
    user explicitly picks a chart type — a single bar is still rendered,
    per the "attempt to honor the requested type" override rule.
    """

    df = pd.DataFrame(
        {
            "customer": ["Alice"],
            "amount": [999],
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="bar",
    )

    assert result is not None
    assert result["chart_type"] == "bar"
    assert len(result["data"]) == 1


# --------------------------------------------------------------------------
# 22. Ranking questions prefer a horizontal bar over a pie
# --------------------------------------------------------------------------


def test_ranking_question_prefers_horizontal_bar_over_pie():
    """
    "What are the top 10 restaurants by revenue?" is a ranking question.
    Even though the category count is small enough to otherwise qualify
    for a pie chart, a ranking should produce a (horizontal) bar chart.
    """

    df = pd.DataFrame(
        {
            "restaurant": ["A", "B", "C"],
            "revenue": [100, 200, 150],
        }
    )

    result = generate_chart(
        _query_result(df),
        question="What are the top 10 restaurants by revenue?",
    )

    assert result is not None
    assert result["chart_type"] == "bar"
    assert result["orientation"] == "horizontal"


def test_ranking_signal_from_sql_order_by_limit():
    """
    Even without an obvious ranking word in the question, a generated
    `ORDER BY ... LIMIT` is itself a structural ranking signal.
    """

    df = pd.DataFrame(
        {
            "product": ["A", "B", "C"],
            "units_sold": [30, 20, 10],
        }
    )

    result = generate_chart(
        _query_result(df),
        question="Which products sell well?",
        sql="SELECT product, units_sold FROM dataset ORDER BY units_sold DESC LIMIT 3",
    )

    assert result is not None
    assert result["chart_type"] == "bar"
    assert result["orientation"] == "horizontal"


def test_non_ranking_bar_chart_defaults_to_vertical_orientation():
    df = pd.DataFrame(
        {
            "product": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "sales": [8, 7, 6, 5, 4, 3, 2, 1],
        }
    )

    result = generate_chart(_query_result(df))

    assert result is not None
    assert result["chart_type"] == "bar"
    assert result["orientation"] == "vertical"


# --------------------------------------------------------------------------
# 23. Proportion questions prefer pie/donut for small distributions
# --------------------------------------------------------------------------


def test_proportion_question_prefers_pie_chart():
    df = pd.DataFrame(
        {
            "category": ["Food", "Grocery", "Electronics"],
            "orders": [400, 250, 150],
        }
    )

    result = generate_chart(
        _query_result(df),
        question="What percentage of orders come from each category?",
    )

    assert result is not None
    assert result["chart_type"] == "pie"


# --------------------------------------------------------------------------
# 24. Time series with multiple numeric columns produces a multi-line chart
# --------------------------------------------------------------------------


def test_datetime_with_multiple_numeric_columns_produces_multi_series_line():
    df = pd.DataFrame(
        {
            "month": pd.date_range("2025-01-01", periods=4, freq="MS"),
            "revenue": [10, 20, 30, 40],
            "profit": [1, 2, 3, 4],
        }
    )

    result = generate_chart(_query_result(df))

    assert result is not None
    assert result["chart_type"] == "line"
    assert result["y_columns"] == ["revenue", "profit"]
    assert result["y_column"] == "revenue"

    first_point = result["data"][0]
    assert "x" in first_point
    assert "revenue" in first_point
    assert "profit" in first_point


def test_datetime_with_single_numeric_column_keeps_original_line_shape():
    """
    Backward compatibility: a single numeric series must keep producing
    the original {"x": ..., "y": ...} data shape and no `y_columns` key
    surprises for existing single-series consumers.
    """

    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5),
            "orders": [5, 8, 3, 9, 4],
        }
    )

    result = generate_chart(_query_result(df))

    assert result is not None
    assert result["chart_type"] == "line"
    assert result["data"][0] == {
        "x": result["data"][0]["x"],
        "y": result["data"][0]["y"],
    }
    assert "y_columns" not in result


def test_multi_series_line_caps_number_of_series():
    columns = {
        "month": pd.date_range("2025-01-01", periods=3, freq="MS"),
    }

    for i in range(8):
        columns[f"metric_{i}"] = [i, i + 1, i + 2]

    df = pd.DataFrame(columns)

    result = generate_chart(_query_result(df))

    assert result is not None
    assert result["chart_type"] == "line"
    assert len(result["y_columns"]) <= 4


# --------------------------------------------------------------------------
# 25. Category + datetime + numeric still prefers the time axis
# --------------------------------------------------------------------------


def test_datetime_takes_priority_over_scatter_when_two_numeric_columns_present():
    """
    Regression guard: previously, a result with a datetime column plus
    two numeric columns could be misclassified as a scatter plot,
    discarding the time axis entirely.
    """

    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5),
            "revenue": [10, 20, 30, 40, 50],
            "profit": [1, 2, 3, 4, 5],
        }
    )

    result = generate_chart(_query_result(df))

    assert result is not None
    assert result["chart_type"] == "line"


def test_category_takes_priority_over_scatter_when_two_numeric_columns_present():
    """
    Regression guard: a result with a category column plus two numeric
    columns should prefer the categorical breakdown over an
    uninformative scatter of the two numeric columns. With grouped_bar
    now implemented, that categorical breakdown is specifically a
    grouped bar (see the dedicated grouped_bar test section below).
    """

    df = pd.DataFrame(
        {
            "restaurant": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "revenue": [80, 70, 60, 50, 40, 30, 20, 10],
            "profit": [8, 7, 6, 5, 4, 3, 2, 1],
        }
    )

    result = generate_chart(_query_result(df))

    assert result is not None
    assert result["chart_type"] == "grouped_bar"


# --------------------------------------------------------------------------
# 26. Empty DataFrame with an explicit chart type still returns None
# --------------------------------------------------------------------------


def test_empty_dataframe_with_explicit_chart_type_returns_none():
    df = pd.DataFrame({"category": [], "value": []})

    result = generate_chart(
        _query_result(df),
        chart_type="bar",
    )

    assert result is None


# --------------------------------------------------------------------------
# 27. Incompatible explicit chart type fails gracefully (no crash)
# --------------------------------------------------------------------------


def test_incompatible_explicit_scatter_request_fails_gracefully():
    """
    Requesting a scatter plot against a result with only one numeric
    column (no second numeric axis) is incompatible. The generator must
    return None rather than raise or fabricate a second axis.
    """

    df = pd.DataFrame(
        {
            "city": ["Delhi", "Mumbai", "Kolkata"],
            "sales": [100, 200, 150],
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="scatter",
    )

    assert result is None


def test_incompatible_explicit_pie_request_without_category_fails_gracefully():
    df = pd.DataFrame(
        {
            "price": [10, 20, 30],
            "sales": [1, 2, 3],
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="pie",
    )

    assert result is None


# --------------------------------------------------------------------------
# 28. High-cardinality categorical results still produce a usable
#     top-N chart rather than an unreadable dump of every category
# --------------------------------------------------------------------------


def test_high_cardinality_categories_produce_limited_top_n_bar():
    df = pd.DataFrame(
        {
            "restaurant": [f"Restaurant {i}" for i in range(200)],
            "revenue": list(range(200, 0, -1)),
        }
    )

    result = generate_chart(
        _query_result(df),
        question="Show revenue by restaurant.",
        max_categories=15,
    )

    assert result is not None
    assert result["chart_type"] == "bar"
    assert len(result["data"]) == 15
    # Highest-revenue restaurant should be first.
    assert result["data"][0]["value"] == 200


# --------------------------------------------------------------------------
# 29. Null-heavy category/value columns are handled defensively
# --------------------------------------------------------------------------


def test_null_heavy_columns_drop_incomplete_rows_without_crashing():
    df = pd.DataFrame(
        {
            "category": ["A", None, "B", None, "C"],
            "value": [10, 20, None, None, 30],
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="bar",
    )

    assert result is not None
    # Only rows with both a category and a value survive (A and C keep
    # both fields; B's value is null and the two fully-null rows drop).
    categories = {row["category"] for row in result["data"]}
    assert categories == {"A", "C"}
    assert all(row["value"] is not None for row in result["data"])


def test_all_null_value_column_returns_no_chart():
    df = pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "value": [None, None, None],
        }
    )

    result = generate_chart(_query_result(df))

    assert result is None


# --------------------------------------------------------------------------
# 30. Grouped bar: categorical + 2 or more numeric metrics
# --------------------------------------------------------------------------


def test_categorical_with_two_numeric_metrics_auto_selects_grouped_bar():
    """
    "Revenue vs cost by region" style results: a categorical column plus
    two (or more) numeric metrics should automatically become a grouped
    bar chart, not a scatter (which would discard the category) and not
    a plain single-metric bar (which would discard the second metric).
    """

    df = pd.DataFrame(
        {
            "region": ["North", "South", "East"],
            "revenue": [50000, 42000, 38000],
            "cost": [32000, 28000, 25000],
        }
    )

    result = generate_chart(
        _query_result(df),
        question="Revenue vs cost by region",
    )

    assert result is not None
    assert result["chart_type"] == "grouped_bar"
    assert result["x_column"] == "region"
    assert result["y_columns"] == ["revenue", "cost"]


def test_grouped_bar_preserves_all_requested_metrics_per_category():
    """Every metric column must appear, correctly aggregated, per row."""

    df = pd.DataFrame(
        {
            "department": ["Sales", "Sales", "Engineering", "Marketing"],
            "actual": [100, 50, 200, 80],
            "budget": [90, 40, 210, 75],
        }
    )

    result = generate_chart(
        _query_result(df),
        question="Actual vs budget by department",
    )

    assert result is not None
    assert result["chart_type"] == "grouped_bar"

    by_department = {
        row["category"]: row
        for row in result["data"]
    }

    # Sales appears twice in the input (100+50 actual, 90+40 budget) and
    # must be summed, not overwritten or dropped.
    assert by_department["Sales"]["actual"] == 150
    assert by_department["Sales"]["budget"] == 130
    assert by_department["Engineering"]["actual"] == 200
    assert by_department["Engineering"]["budget"] == 210
    assert by_department["Marketing"]["actual"] == 80
    assert by_department["Marketing"]["budget"] == 75


def test_single_numeric_metric_still_produces_plain_bar_not_grouped():
    """"Revenue by region" (one metric) must stay a plain bar chart, not
    a grouped bar — grouped_bar only applies once there are 2+ metrics.
    Enough regions are used here to clear the small-cardinality pie
    threshold, so the result is unambiguously a bar chart.
    """

    df = pd.DataFrame(
        {
            "region": [f"Region {i}" for i in range(10)],
            "revenue": list(range(100, 0, -10)),
        }
    )

    result = generate_chart(
        _query_result(df),
        question="Revenue by region",
    )

    assert result is not None
    assert result["chart_type"] == "bar"


def test_two_numeric_columns_without_category_still_selects_scatter():
    """"Price vs sales" (no categorical column) must remain a scatter."""

    df = pd.DataFrame(
        {
            "price": [10, 20, 30, 40],
            "sales": [400, 300, 200, 100],
        }
    )

    result = generate_chart(
        _query_result(df),
        question="Price vs sales",
    )

    assert result is not None
    assert result["chart_type"] == "scatter"


def test_explicit_grouped_bar_request_is_honored():
    df = pd.DataFrame(
        {
            "region": ["North", "South"],
            "revenue": [100, 200],
            "cost": [50, 90],
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="grouped_bar",
    )

    assert result is not None
    assert result["chart_type"] == "grouped_bar"


def test_explicit_grouped_bar_without_second_metric_fails_gracefully():
    """Requesting grouped_bar with only one numeric column is incompatible."""

    df = pd.DataFrame(
        {
            "region": ["North", "South"],
            "revenue": [100, 200],
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="grouped_bar",
    )

    assert result is None


def test_explicit_grouped_bar_without_category_fails_gracefully():
    """Requesting grouped_bar with no categorical column is incompatible."""

    df = pd.DataFrame(
        {
            "revenue": [100, 200, 300],
            "cost": [50, 90, 120],
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="grouped_bar",
    )

    assert result is None


def test_grouped_bar_caps_number_of_metric_series():
    columns = {
        "region": ["North", "South", "East"],
    }

    for i in range(8):
        columns[f"metric_{i}"] = [i, i + 1, i + 2]

    df = pd.DataFrame(columns)

    result = generate_chart(
        _query_result(df),
        chart_type="grouped_bar",
    )

    assert result is not None
    assert len(result["y_columns"]) <= 4


# --------------------------------------------------------------------------
# 31. High-cardinality categories: Top-N + "Other"
# --------------------------------------------------------------------------


def test_high_cardinality_bar_produces_top_12_plus_other():
    """
    The canonical example from the visualization spec: 100 categories,
    only the top 12 (by revenue) are shown individually, and everything
    else is summed into a single trailing "Other" row.
    """

    df = pd.DataFrame(
        {
            "restaurant": [f"Restaurant {i}" for i in range(100)],
            "revenue": list(range(100, 0, -1)),
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="bar",
        max_categories=13,  # Top 12 + 1 "Other" row = 13 total
    )

    assert result is not None
    assert len(result["data"]) == 13

    categories = [row["category"] for row in result["data"]]

    assert categories[:12] == [
        f"Restaurant {i}" for i in range(12)
    ]
    assert categories[12] == "Other"
    assert result["has_other"] is True
    assert result["other_count"] == 88  # 100 - 12


def test_other_bucket_value_is_the_correct_aggregate_of_remaining_categories():
    df = pd.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E", "F"],
            "value": [100, 90, 10, 8, 6, 4],
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="bar",
        max_categories=3,  # Top 2 ("A","B") + Other
    )

    assert result is not None

    other_row = result["data"][-1]

    assert other_row["category"] == "Other"
    assert other_row["value"] == 28  # 10 + 8 + 6 + 4
    assert result["other_count"] == 4


def test_other_bucket_sums_correctly_for_grouped_bar():
    df = pd.DataFrame(
        {
            "region": [f"Region {i}" for i in range(6)],
            "revenue": [100, 90, 10, 8, 6, 4],
            "cost": [50, 45, 5, 4, 3, 2],
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="grouped_bar",
        max_categories=3,
    )

    assert result is not None

    other_row = result["data"][-1]

    assert other_row["category"] == "Other"
    assert other_row["revenue"] == 28  # 10 + 8 + 6 + 4
    assert other_row["cost"] == 14  # 5 + 4 + 3 + 2
    assert result["has_other"] is True
    assert result["other_count"] == 4


def test_no_other_row_when_result_fits_within_max_categories():
    df = pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "value": [30, 20, 10],
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="bar",
        max_categories=20,
    )

    assert result is not None
    assert result["has_other"] is False
    assert result["other_count"] == 0
    assert "Other" not in [row["category"] for row in result["data"]]


def test_explicit_top_n_sql_does_not_fabricate_other_when_already_limited():
    """
    A generated `ORDER BY ... LIMIT 10` already constrained the result to
    10 categories. With the default max_categories=20, the DataFrame
    doesn't actually contain any categories beyond what's shown, so no
    "Other" row should be fabricated even though this is a ranking query.
    """

    df = pd.DataFrame(
        {
            "product": [f"Product {i}" for i in range(10)],
            "units_sold": list(range(100, 90, -1)),
        }
    )

    result = generate_chart(
        _query_result(df),
        question="What are the top 10 best-selling products?",
        sql="SELECT product, units_sold FROM dataset ORDER BY units_sold DESC LIMIT 10",
    )

    assert result is not None
    assert result["chart_type"] == "bar"
    assert result["orientation"] == "horizontal"
    assert len(result["data"]) == 10
    assert result["has_other"] is False
    assert "Other" not in [row["category"] for row in result["data"]]


def test_ranking_query_still_gets_other_when_its_own_limit_exceeds_max_categories():
    """
    The flip side of the previous test: if the SQL's own LIMIT is larger
    than max_categories (e.g. `LIMIT 50` with max_categories=20), the
    DataFrame genuinely does contain more categories than we can chart,
    so "Other" is still added — being a ranking query doesn't exempt a
    result from the same max_categories safeguard everything else gets.
    """

    df = pd.DataFrame(
        {
            "product": [f"Product {i}" for i in range(50)],
            "units_sold": list(range(100, 50, -1)),
        }
    )

    result = generate_chart(
        _query_result(df),
        question="top selling products",
        sql="SELECT product, units_sold FROM dataset ORDER BY units_sold DESC LIMIT 50",
        max_categories=20,
    )

    assert result is not None
    assert len(result["data"]) == 20
    assert result["has_other"] is True
    assert result["other_count"] == 31  # 50 - 19 kept individually


def test_pie_chart_applies_top_n_with_other_on_explicit_high_cardinality_request():
    """
    Pie is never auto-selected for high-cardinality data, but an explicit
    user override on a high-cardinality result must still be safe rather
    than rendering an unreadable hundred-slice pie.
    """

    df = pd.DataFrame(
        {
            "category": [f"Category {i}" for i in range(30)],
            "value": list(range(30, 0, -1)),
        }
    )

    result = generate_chart(
        _query_result(df),
        chart_type="pie",
        max_categories=10,
    )

    assert result is not None
    assert len(result["data"]) == 10
    assert result["data"][-1]["label"] == "Other"
    assert result["has_other"] is True
    assert result["other_count"] == 21  # 30 - 9 kept individually

