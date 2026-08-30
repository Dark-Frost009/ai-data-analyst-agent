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

    assert categories == ["A", "B", "C"]


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