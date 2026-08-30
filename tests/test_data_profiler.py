"""
Tests for app.core.data_profiler.profile_dataframe.

No Streamlit, AWS, or DuckDB involved — these run purely against pandas
DataFrames constructed in-memory.
"""

import json

import numpy as np
import pandas as pd
import pytest

from app.core.data_profiler import profile_dataframe
from app.models.schemas import DatasetProfile


def _column(profile: DatasetProfile, name: str):
    for col in profile.columns:
        if col.name == name:
            return col
    raise AssertionError(f"column '{name}' not found in profile")


# --------------------------------------------------------------------------
# 1. Basic mixed-type DataFrame
# --------------------------------------------------------------------------
def test_profile_basic_mixed_dataframe():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "score": [1.5, 2.5, 3.5],
            "label": ["a", "b", "c"],
        }
    )
    profile = profile_dataframe(df)

    assert profile.row_count == 3
    assert profile.column_count == 3
    assert profile.column_names == ["id", "score", "label"]
    assert _column(profile, "id").inferred_type == "integer"
    assert _column(profile, "score").inferred_type == "float"
    assert _column(profile, "label").inferred_type == "string"


# --------------------------------------------------------------------------
# 2. Numeric column, no nulls
# --------------------------------------------------------------------------
def test_profile_numeric_column_no_nulls():
    df = pd.DataFrame({"value": [10, 20, 30, 40]})
    col = _column(profile_dataframe(df), "value")

    assert col.min == 10
    assert col.max == 40
    assert col.mean == pytest.approx(25.0)
    assert col.null_count == 0
    assert col.null_percentage == 0.0


# --------------------------------------------------------------------------
# 3. Numeric column with nulls
# --------------------------------------------------------------------------
def test_profile_numeric_column_with_nulls():
    df = pd.DataFrame({"value": [10.0, None, 30.0, None]})
    col = _column(profile_dataframe(df), "value")

    assert col.null_count == 2
    assert col.null_percentage == 50.0
    assert col.mean == pytest.approx(20.0)  # mean of [10, 30] only


# --------------------------------------------------------------------------
# 4. Entirely-null column
# --------------------------------------------------------------------------
def test_profile_entirely_null_column():
    df = pd.DataFrame({"a": [1, 2, 3], "empty_col": [None, None, None]})
    col = _column(profile_dataframe(df), "empty_col")

    assert col.inferred_type == "empty"
    assert col.null_count == 3
    assert col.null_percentage == 100.0
    assert col.unique_count == 0
    assert col.min is None
    assert col.max is None
    assert col.mean is None


# --------------------------------------------------------------------------
# 5. Zero-row DataFrame
# --------------------------------------------------------------------------
def test_profile_zero_row_dataframe():
    df = pd.DataFrame({"a": pd.Series([], dtype="object"), "b": pd.Series([], dtype="object")})
    profile = profile_dataframe(df)

    assert profile.row_count == 0
    assert profile.sample_rows == []
    for col in profile.columns:
        assert col.inferred_type == "empty"
        assert col.null_count == 0
        assert col.null_percentage == 0.0  # guarded, not a division-by-zero crash
        assert col.unique_count == 0


# --------------------------------------------------------------------------
# 6. Unique-value counts
# --------------------------------------------------------------------------
def test_profile_unique_value_counts():
    df = pd.DataFrame({"category": ["x", "y", "x", "x", "y", "z"]})
    col = _column(profile_dataframe(df), "category")

    assert col.unique_count == 3


# --------------------------------------------------------------------------
# 7. Datetime-like string column
# --------------------------------------------------------------------------
def test_profile_datetime_like_string_column():
    original_df = pd.DataFrame(
        {"signup_date": ["2023-01-01", "2023-06-15", "2023-12-31", "2024-02-10"]}
    )
    df = original_df.copy(deep=True)
    dtype_before = df["signup_date"].dtype

    col = _column(profile_dataframe(df), "signup_date")

    assert col.inferred_type == "datetime"
    assert col.min == "2023-01-01T00:00:00"
    assert col.max == "2024-02-10T00:00:00"
    assert col.mean is None  # deliberately not computed for datetime columns

    # The profiler must not have mutated the column's actual dtype. Compare
    # against the dtype captured before profiling rather than a hardcoded
    # dtype name — pandas 3.0+ may default string columns to StringDtype
    # instead of the legacy object dtype, and this check must hold either way.
    pd.testing.assert_frame_equal(df, original_df)
    assert df["signup_date"].dtype == dtype_before


# --------------------------------------------------------------------------
# 8. String column that only coincidentally resembles dates
# --------------------------------------------------------------------------
def test_profile_string_column_not_misclassified_as_datetime():
    df = pd.DataFrame(
        {"city": ["Chicago", "Denver", "May", "Boston", "Austin", "Seattle", "Miami"]}
    )
    col = _column(profile_dataframe(df), "city")

    # Only "May" coincidentally parses as a date — well under the 90%
    # threshold, so this must stay classified as plain string data.
    assert col.inferred_type == "string"


# --------------------------------------------------------------------------
# 9. Boolean column
# --------------------------------------------------------------------------
def test_profile_boolean_column():
    df = pd.DataFrame({"is_active": [True, False, True, True]})
    col = _column(profile_dataframe(df), "is_active")

    assert col.inferred_type == "boolean"
    assert col.mean is None
    assert col.min is None
    assert col.max is None


# --------------------------------------------------------------------------
# 10. Sample rows are fully JSON-safe
# --------------------------------------------------------------------------
def test_profile_sample_rows_are_json_serializable():
    df = pd.DataFrame(
        {
            "id": np.array([1, 2, 3], dtype="int64"),
            "score": [1.1, None, 3.3],
            "signed_up": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]),
        }
    )
    profile = profile_dataframe(df, sample_rows=3)

    # No numpy scalars should have leaked through.
    for row in profile.sample_rows:
        for value in row.values():
            assert not isinstance(value, (np.integer, np.floating, np.bool_))

    # The missing score must be None, not NaN.
    assert profile.sample_rows[1]["score"] is None

    # allow_nan=False makes json.dumps raise if any NaN/Infinity slipped
    # through — this is the strict proof of JSON-safety.
    serialized = json.dumps(profile.model_dump(), allow_nan=False)
    assert "signed_up" in serialized


# --------------------------------------------------------------------------
# 11. Invalid input
# --------------------------------------------------------------------------
def test_profile_rejects_none_and_non_dataframe():
    with pytest.raises(ValueError):
        profile_dataframe(None)
    with pytest.raises(ValueError):
        profile_dataframe([{"a": 1}])


# --------------------------------------------------------------------------
# 12. Mixed-type object column
# --------------------------------------------------------------------------
def test_profile_mixed_type_object_column():
    df = pd.DataFrame({"weird": pd.Series([1, "two", 3.0, None], dtype="object")})
    col = _column(profile_dataframe(df), "weird")

    assert col.inferred_type == "mixed"
    assert col.null_count == 1


# --------------------------------------------------------------------------
# 13. Immutability of the input DataFrame
# --------------------------------------------------------------------------
def test_profile_does_not_mutate_input_dataframe():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "note": ["a", None, "c"],
            "date_str": ["2023-01-01", "2023-01-02", "2023-01-03"],
        }
    )
    snapshot = df.copy(deep=True)

    profile_dataframe(df, sample_rows=2)

    pd.testing.assert_frame_equal(df, snapshot)