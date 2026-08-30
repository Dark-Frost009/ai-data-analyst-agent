"""
Tests for app.models.schemas.

These tests verify the Pydantic data contracts shared between the
different stages of the AI Data Analyst Agent pipeline.

No Streamlit, AWS, DuckDB, or network access is required.
"""

import json

import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from app.models.schemas import (
    ColumnProfile,
    DatasetProfile,
    QueryResult,
    ValidationError,
    ValidationResult,
)


# --------------------------------------------------------------------------
# ColumnProfile
# --------------------------------------------------------------------------


def test_column_profile_basic():
    """ColumnProfile should accept a valid column description."""
    profile = ColumnProfile(
        name="age",
        pandas_dtype="int64",
        inferred_type="integer",
        null_count=2,
        null_percentage=10.0,
        unique_count=18,
        min=18,
        max=65,
        mean=35.5,
    )

    assert profile.name == "age"
    assert profile.pandas_dtype == "int64"
    assert profile.inferred_type == "integer"
    assert profile.null_count == 2
    assert profile.null_percentage == 10.0
    assert profile.unique_count == 18
    assert profile.min == 18
    assert profile.max == 65
    assert profile.mean == pytest.approx(35.5)


def test_column_profile_optional_statistics_default_to_none():
    """min, max, and mean should default to None."""
    profile = ColumnProfile(
        name="category",
        pandas_dtype="object",
        inferred_type="string",
        null_count=0,
        null_percentage=0.0,
        unique_count=3,
    )

    assert profile.min is None
    assert profile.max is None
    assert profile.mean is None


@pytest.mark.parametrize(
    "inferred_type",
    [
        "integer",
        "float",
        "boolean",
        "datetime",
        "string",
        "mixed",
        "empty",
    ],
)
def test_column_profile_accepts_all_valid_column_types(inferred_type):
    """Every declared ColumnType literal should be accepted."""
    profile = ColumnProfile(
        name="test",
        pandas_dtype="object",
        inferred_type=inferred_type,
        null_count=0,
        null_percentage=0.0,
        unique_count=0,
    )

    assert profile.inferred_type == inferred_type


def test_column_profile_rejects_invalid_column_type():
    """An unsupported inferred_type should fail validation."""
    with pytest.raises(PydanticValidationError):
        ColumnProfile(
            name="test",
            pandas_dtype="object",
            inferred_type="currency",
            null_count=0,
            null_percentage=0.0,
            unique_count=0,
        )


# --------------------------------------------------------------------------
# DatasetProfile
# --------------------------------------------------------------------------


def test_dataset_profile_basic():
    """DatasetProfile should contain columns and sample rows."""
    columns = [
        ColumnProfile(
            name="id",
            pandas_dtype="int64",
            inferred_type="integer",
            null_count=0,
            null_percentage=0.0,
            unique_count=3,
            min=1,
            max=3,
            mean=2.0,
        ),
        ColumnProfile(
            name="name",
            pandas_dtype="object",
            inferred_type="string",
            null_count=0,
            null_percentage=0.0,
            unique_count=3,
        ),
    ]

    profile = DatasetProfile(
        row_count=3,
        column_count=2,
        columns=columns,
        sample_rows=[
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ],
    )

    assert profile.row_count == 3
    assert profile.column_count == 2
    assert len(profile.columns) == 2
    assert len(profile.sample_rows) == 2


def test_dataset_profile_column_names_property():
    """column_names should return column names in their original order."""
    columns = [
        ColumnProfile(
            name="id",
            pandas_dtype="int64",
            inferred_type="integer",
            null_count=0,
            null_percentage=0.0,
            unique_count=3,
        ),
        ColumnProfile(
            name="revenue",
            pandas_dtype="float64",
            inferred_type="float",
            null_count=0,
            null_percentage=0.0,
            unique_count=3,
        ),
        ColumnProfile(
            name="category",
            pandas_dtype="object",
            inferred_type="string",
            null_count=0,
            null_percentage=0.0,
            unique_count=2,
        ),
    ]

    profile = DatasetProfile(
        row_count=3,
        column_count=3,
        columns=columns,
        sample_rows=[],
    )

    assert profile.column_names == [
        "id",
        "revenue",
        "category",
    ]


def test_dataset_profile_allows_empty_dataset():
    """A zero-row dataset should still be a valid DatasetProfile."""
    profile = DatasetProfile(
        row_count=0,
        column_count=0,
        columns=[],
        sample_rows=[],
    )

    assert profile.row_count == 0
    assert profile.column_count == 0
    assert profile.columns == []
    assert profile.sample_rows == []
    assert profile.column_names == []


# --------------------------------------------------------------------------
# ValidationError
# --------------------------------------------------------------------------


def test_validation_error_basic():
    """ValidationError should store a security rejection reason."""
    error = ValidationError(
        code="DISALLOWED_STATEMENT",
        message="Only SELECT statements are allowed.",
    )

    assert error.code == "DISALLOWED_STATEMENT"
    assert error.message == "Only SELECT statements are allowed."
    assert error.detail is None


def test_validation_error_with_detail():
    """ValidationError should optionally preserve additional detail."""
    error = ValidationError(
        code="DISALLOWED_TABLE",
        message="Query references a table that is not allowed.",
        detail="secret_table",
    )

    assert error.code == "DISALLOWED_TABLE"
    assert error.message == "Query references a table that is not allowed."
    assert error.detail == "secret_table"


# --------------------------------------------------------------------------
# ValidationResult
# --------------------------------------------------------------------------


def test_validation_result_valid_query():
    """A successful SQL validation result should store cleaned SQL."""
    result = ValidationResult(
        is_valid=True,
        cleaned_sql="SELECT * FROM dataset",
    )

    assert result.is_valid is True
    assert result.cleaned_sql == "SELECT * FROM dataset"
    assert result.errors == []


def test_validation_result_invalid_query():
    """An invalid SQL validation result should contain errors."""
    error = ValidationError(
        code="DISALLOWED_STATEMENT",
        message="Only SELECT statements are allowed.",
    )

    result = ValidationResult(
        is_valid=False,
        cleaned_sql=None,
        errors=[error],
    )

    assert result.is_valid is False
    assert result.cleaned_sql is None
    assert len(result.errors) == 1
    assert result.errors[0].code == "DISALLOWED_STATEMENT"


def test_validation_result_errors_default_to_empty_list():
    """errors should default to a fresh empty list."""
    first = ValidationResult(is_valid=True)
    second = ValidationResult(is_valid=True)

    assert first.errors == []
    assert second.errors == []

    # Verify the lists are independent.
    first.errors.append(
        ValidationError(
            code="TEST",
            message="test",
        )
    )

    assert len(first.errors) == 1
    assert len(second.errors) == 0


# --------------------------------------------------------------------------
# QueryResult
# --------------------------------------------------------------------------


def test_query_result_accepts_pandas_dataframe():
    """QueryResult should accept a real pandas DataFrame."""
    df = pd.DataFrame(
        {
            "category": ["A", "B", "A"],
            "amount": [100.0, 200.0, 150.0],
        }
    )

    result = QueryResult(
        dataframe=df,
        row_count=3,
        truncated=False,
    )

    assert result.dataframe is df
    assert result.row_count == 3
    assert result.truncated is False


def test_query_result_preserves_dataframe_contents():
    """The DataFrame stored in QueryResult should remain usable."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        }
    )

    result = QueryResult(
        dataframe=df,
        row_count=3,
        truncated=False,
    )

    pd.testing.assert_frame_equal(result.dataframe, df)


def test_query_result_truncated_flag():
    """QueryResult should correctly represent result truncation."""
    df = pd.DataFrame({"value": range(10)})

    result = QueryResult(
        dataframe=df,
        row_count=10,
        truncated=True,
    )

    assert result.truncated is True


def test_query_result_allows_empty_dataframe():
    """An empty pandas DataFrame is still a valid query result."""
    df = pd.DataFrame(columns=["id", "value"])

    result = QueryResult(
        dataframe=df,
        row_count=0,
        truncated=False,
    )

    assert result.dataframe.empty
    assert result.row_count == 0
    assert result.truncated is False


# --------------------------------------------------------------------------
# JSON serialization
# --------------------------------------------------------------------------


def test_dataset_profile_is_json_serializable():
    """
    DatasetProfile contains only JSON-safe primitives and should serialize
    cleanly through Pydantic's model_dump/model_dump_json.
    """
    profile = DatasetProfile(
        row_count=2,
        column_count=2,
        columns=[
            ColumnProfile(
                name="id",
                pandas_dtype="int64",
                inferred_type="integer",
                null_count=0,
                null_percentage=0.0,
                unique_count=2,
                min=1,
                max=2,
                mean=1.5,
            ),
            ColumnProfile(
                name="name",
                pandas_dtype="object",
                inferred_type="string",
                null_count=0,
                null_percentage=0.0,
                unique_count=2,
            ),
        ],
        sample_rows=[
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ],
    )

    serialized = profile.model_dump_json()

    assert isinstance(serialized, str)
    assert "Alice" in serialized
    assert "Bob" in serialized

    # Strictly confirm the result is valid JSON.
    parsed = json.loads(serialized)

    assert parsed["row_count"] == 2
    assert parsed["column_count"] == 2


def test_validation_result_is_json_serializable():
    """ValidationResult should serialize cleanly to JSON."""
    result = ValidationResult(
        is_valid=False,
        cleaned_sql=None,
        errors=[
            ValidationError(
                code="DISALLOWED_TABLE",
                message="Table is not allowed.",
                detail="secret_table",
            )
        ],
    )

    serialized = result.model_dump_json()

    parsed = json.loads(serialized)

    assert parsed["is_valid"] is False
    assert parsed["cleaned_sql"] is None
    assert parsed["errors"][0]["code"] == "DISALLOWED_TABLE"
    assert parsed["errors"][0]["detail"] == "secret_table"


# --------------------------------------------------------------------------
# Validation / type safety
# --------------------------------------------------------------------------


def test_column_profile_requires_required_fields():
    """Required ColumnProfile fields should not be silently omitted."""
    with pytest.raises(PydanticValidationError):
        ColumnProfile(
            name="id",
            pandas_dtype="int64",
            inferred_type="integer",
        )


def test_dataset_profile_requires_required_fields():
    """Required DatasetProfile fields should be enforced."""
    with pytest.raises(PydanticValidationError):
        DatasetProfile(
            row_count=10,
            column_count=2,
        )


def test_query_result_requires_dataframe():
    """QueryResult must contain a pandas DataFrame."""
    with pytest.raises(PydanticValidationError):
        QueryResult(
            row_count=0,
            truncated=False,
        )


def test_query_result_rejects_non_dataframe():
    """QueryResult should reject arbitrary non-DataFrame values."""
    with pytest.raises(PydanticValidationError):
        QueryResult(
            dataframe=[{"id": 1}],
            row_count=1,
            truncated=False,
        )