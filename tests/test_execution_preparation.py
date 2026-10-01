"""Regression coverage for conservative preparation and real DuckDB execution."""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.core.agent import DataAnalystAgent
from app.core.data_profiler import profile_dataframe
from app.core.execution_preparation import prepare_execution_dataframe
from app.core.query_planner import QueryPlanner, DEFAULT_SYSTEM_PROMPT
from app.core.sql_executor import SQLExecutor
from app.utils.security import validate_sql


@pytest.mark.parametrize("values,expected", [
    (["12/7/2023", "1/15/2022", "2/29/2024"],
     ["2023-12-07", "2022-01-15", "2024-02-29"]),
    (["2023-12-07", "2022-01-15", "2024-02-29"],
     ["2023-12-07", "2022-01-15", "2024-02-29"]),
])
def test_confident_dates(values, expected):
    source = pd.DataFrame({"date": values})
    prepared = prepare_execution_dataframe(source)
    assert pd.api.types.is_datetime64_any_dtype(prepared["date"])
    assert prepared["date"].tolist() == list(pd.to_datetime(expected))


@pytest.mark.parametrize("values", [
    ["2020", "2021", None],
    [2020, 2021],
    ["12/7/2023", "1/2/2022"],
    ["13/7/2023", "1/15/2022"],
    ["2023-12-07", "invalid"],
    ["1/15/2022", "invalid"],
    ["2023-02-29", "2024-02-29"],
    ["2/30/2023", "1/15/2022"],
    ["2023-12-07", "1/15/2022"],
    ["2023-12-07", 2022],
    ["2023-12-07", ""],
    ["9999-12-31", "2023-12-07"],
    [None, pd.NA],
    [],
    ["2023-12-07"] * 100 + ["invalid"],
])
def test_unsafe_columns_are_unchanged(values):
    source = pd.DataFrame({"date": pd.Series(values, dtype=object)})
    assert_frame_equal(prepare_execution_dataframe(source), source)


@pytest.mark.parametrize("dtype", [object, "string"])
def test_nulls_structure_and_source_preserved(dtype):
    source = pd.DataFrame({
        "Join_Date": pd.Series(["1/15/2022", None, pd.NA, np.nan, "12/7/2023"], dtype=dtype),
        "salary": ["$1,200", "$2,000", "unknown", None, "$5,000"],
        "year": ["2020", "2021", "2022", None, "2023"],
    })
    source.index = pd.Index([9, 3, 3, 1, 8], name="employee")
    source.attrs["source"] = "upload"
    before = source.copy(deep=True)
    prepared = prepare_execution_dataframe(source)
    assert_frame_equal(source, before)
    assert prepared is not source
    assert prepared.index.equals(source.index)
    assert prepared.columns.equals(source.columns)
    assert prepared.attrs == source.attrs
    assert prepared["Join_Date"].isna().equals(source["Join_Date"].isna())
    assert_frame_equal(prepared[["salary", "year"]], source[["salary", "year"]])
    prepared.iloc[0, 1] = "changed"
    assert_frame_equal(source, before)


def test_existing_datetime_retains_timezone_and_nulls():
    source = pd.DataFrame({"date": pd.to_datetime(["2023-12-07", None], utc=True)})
    assert_frame_equal(prepare_execution_dataframe(source), source)


def test_execution_profile_and_duckdb_type():
    prepared = prepare_execution_dataframe(pd.DataFrame({"date": ["2023-12-07", None]}))
    column = profile_dataframe(prepared).columns[0]
    assert column.inferred_type == "datetime"
    assert column.pandas_dtype.startswith("datetime64")
    assert column.null_count == 1
    assert column.min == "2023-12-07T00:00:00"
    # A date-part operation on the actual executor table requires temporal typing.
    validation = validate_sql('SELECT EXTRACT(YEAR FROM "date") AS year FROM dataset', allowed_tables=["dataset"])
    assert validation.is_valid
    with SQLExecutor(prepared) as executor:
        result = executor.execute(validation)
    assert result.dataframe.iloc[0]["year"] == 2023
    assert pd.isna(result.dataframe.iloc[1]["year"])


def test_employee_after_2022_uses_execution_schema_and_real_executor():
    source = pd.DataFrame({
        "First_Name": ["Asha", "Ravi", "Mira", "Dev", "Nila"],
        "Last_Name": ["A", "B", "C", "D", "E"],
        "Join_Date": ["12/7/2023", "1/15/2022", "11/30/2021", "1/1/2023", None],
    })
    before = source.copy(deep=True)
    raw_profile = profile_dataframe(source)
    llm = MagicMock()
    llm.generate_text.return_value = (
        'SELECT "First_Name", "Last_Name" FROM dataset '
        'WHERE "Join_Date" > DATE \'2022-12-31\' ORDER BY "First_Name"'
    )
    agent = DataAnalystAgent(source, raw_profile, query_planner=QueryPlanner(llm_client=llm))
    result = agent.run("Which employees joined after 2022?", False, False)
    assert result.query_result.dataframe["First_Name"].tolist() == ["Asha", "Dev"]
    assert result.validation.is_valid
    assert agent.dataframe is source
    assert agent.dataset_profile is raw_profile
    assert agent.execution_profile.columns[2].pandas_dtype.startswith("datetime64")
    prompt = llm.generate_text.call_args.kwargs["prompt"]
    assert '"pandas_dtype": "datetime64' in prompt
    assert "2023-12-07T00:00:00" in prompt
    assert_frame_equal(source, before)


def test_planner_date_guidance_uses_physical_dtype_in_both_prompts():
    profile = profile_dataframe(pd.DataFrame({"date": ["2023-12-07"]}))
    for prompt in [DEFAULT_SYSTEM_PROMPT, QueryPlanner._build_prompt("Dates?", profile)]:
        assert "Use pandas_dtype as the physical execution type" in prompt
        assert "Do not wrap already-normalized datetime columns in TRY_CAST or TRY_STRPTIME" in prompt
        assert "For monthly analysis on these columns use DATE_TRUNC" in prompt
        assert "Do not guess a date format" in prompt


def test_invalid_input():
    with pytest.raises(ValueError, match="pandas DataFrame"):
        prepare_execution_dataframe("invalid")
