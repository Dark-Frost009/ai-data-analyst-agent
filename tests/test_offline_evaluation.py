"""Offline evaluation cases for the end-to-end analytics pipeline.

Each case supplies a fixed SQL response from a mocked planner, then runs the
same validation and DuckDB execution path used by the Streamlit application.
This provides a fast, repeatable regression suite without a live Bedrock
dependency.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.core.agent import (
    AgentExecutionError,
    AgentValidationError,
    DataAnalystAgent,
)
from app.core.data_profiler import profile_dataframe


@pytest.fixture
def sales_dataframe() -> pd.DataFrame:
    """A small, deliberately varied dataset for representative questions."""

    return pd.DataFrame(
        {
            "product": ["A", "B", "A", "C", "B"],
            "region": ["East", "West", "East", "North", "West"],
            "sales": [100, 200, 300, 150, 50],
        }
    )


def _agent_with_planned_sql(
    dataframe: pd.DataFrame,
    sql: str,
) -> tuple[DataAnalystAgent, MagicMock]:
    """Build an agent whose planner returns one deterministic SQL response."""

    planner = MagicMock()
    planner.plan.return_value = sql

    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=profile_dataframe(dataframe),
        query_planner=planner,
    )

    return agent, planner


@pytest.mark.parametrize(
    ("question", "sql", "expected_rows"),
    [
        (
            "How many records are in this dataset?",
            "SELECT COUNT(*) AS record_count FROM dataset",
            [{"record_count": 5}],
        ),
        (
            "What are total sales by product?",
            "SELECT product, SUM(sales) AS total_sales "
            "FROM dataset GROUP BY product ORDER BY product",
            [
                {"product": "A", "total_sales": 400},
                {"product": "B", "total_sales": 250},
                {"product": "C", "total_sales": 150},
            ],
        ),
        (
            "Which two products have the highest sales?",
            "SELECT product, SUM(sales) AS total_sales "
            "FROM dataset GROUP BY product "
            "ORDER BY total_sales DESC LIMIT 2",
            [
                {"product": "A", "total_sales": 400},
                {"product": "B", "total_sales": 250},
            ],
        ),
    ],
)
def test_representative_analysis_cases(
    sales_dataframe: pd.DataFrame,
    question: str,
    sql: str,
    expected_rows: list[dict],
):
    """Representative analytics questions produce the expected result data."""

    agent, planner = _agent_with_planned_sql(
        sales_dataframe,
        sql,
    )

    result = agent.run(
        question=question,
        generate_explanation=False,
        generate_chart_spec=False,
    )

    planner.plan.assert_called_once()
    assert result.validation.is_valid is True
    assert result.query_result.dataframe.to_dict("records") == expected_rows


def test_unsafe_planner_output_is_rejected_before_execution(
    sales_dataframe: pd.DataFrame,
):
    """An unsafe LLM response must not cross the validation boundary."""

    agent, _ = _agent_with_planned_sql(
        sales_dataframe,
        "DROP TABLE dataset",
    )

    with pytest.raises(AgentValidationError):
        agent.run(
            question="Delete the dataset",
            generate_explanation=False,
            generate_chart_spec=False,
        )


def test_unknown_column_becomes_a_safe_execution_error(
    sales_dataframe: pd.DataFrame,
):
    """A syntactically safe but invalid plan fails without leaking past the agent."""

    agent, _ = _agent_with_planned_sql(
        sales_dataframe,
        "SELECT missing_column FROM dataset",
    )

    with pytest.raises(AgentExecutionError):
        agent.run(
            question="Show a column that does not exist",
            generate_explanation=False,
            generate_chart_spec=False,
        )
