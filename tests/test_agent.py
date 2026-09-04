import pandas as pd
import pytest
from unittest.mock import MagicMock

from app.core.agent import (
    AgentChartError,
    AgentExecutionError,
    AgentExplanationError,
    AgentPlanningError,
    AgentResult,
    AgentValidationError,
    DataAnalystAgent,
    run_analysis,
)
from app.core.explainer import ExplainerLLMError
from app.core.query_planner import QueryPlanningError
from app.core.sql_executor import QueryExecutionError
from app.models.schemas import (
    ColumnProfile,
    DatasetProfile,
    QueryResult,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def dataframe():
    return pd.DataFrame(
        {
            "product": ["A", "B", "C", "A"],
            "sales": [100, 200, 300, 150],
            "quantity": [2, 4, 6, 3],
        }
    )


@pytest.fixture
def dataset_profile():
    return DatasetProfile(
        row_count=4,
        column_count=3,
        columns=[
            ColumnProfile(
                name="product",
                pandas_dtype="object",
                inferred_type="string",
                null_count=0,
                null_percentage=0.0,
                unique_count=3,
            ),
            ColumnProfile(
                name="sales",
                pandas_dtype="int64",
                inferred_type="integer",
                null_count=0,
                null_percentage=0.0,
                unique_count=4,
                min=100,
                max=300,
                mean=187.5,
            ),
            ColumnProfile(
                name="quantity",
                pandas_dtype="int64",
                inferred_type="integer",
                null_count=0,
                null_percentage=0.0,
                unique_count=4,
                min=2,
                max=6,
                mean=3.75,
            ),
        ],
        sample_rows=[
            {
                "product": "A",
                "sales": 100,
                "quantity": 2,
            },
            {
                "product": "B",
                "sales": 200,
                "quantity": 4,
            },
        ],
    )


@pytest.fixture
def mock_planner():
    planner = MagicMock()
    planner.plan.return_value = (
        "SELECT product, SUM(sales) AS total_sales "
        "FROM dataset "
        "GROUP BY product "
        "ORDER BY total_sales DESC"
    )
    return planner


@pytest.fixture
def mock_query_result():
    return QueryResult(
        dataframe=pd.DataFrame(
            {
                "product": ["C", "A", "B"],
                "total_sales": [300, 250, 200],
            }
        ),
        row_count=3,
        truncated=False,
    )


# --------------------------------------------------------------------------
# Constructor tests
# --------------------------------------------------------------------------


def test_agent_initializes(
    dataframe,
    dataset_profile,
    mock_planner,
):
    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=mock_planner,
    )

    assert agent.dataframe is dataframe
    assert agent.dataset_profile is dataset_profile


def test_agent_rejects_invalid_dataframe(
    dataset_profile,
):
    with pytest.raises(ValueError):
        DataAnalystAgent(
            dataframe="not a dataframe",
            dataset_profile=dataset_profile,
        )


def test_agent_rejects_invalid_dataset_profile(
    dataframe,
):
    with pytest.raises(ValueError):
        DataAnalystAgent(
            dataframe=dataframe,
            dataset_profile="not a profile",
        )


def test_agent_rejects_invalid_max_result_rows(
    dataframe,
    dataset_profile,
):
    with pytest.raises(ValueError):
        DataAnalystAgent(
            dataframe=dataframe,
            dataset_profile=dataset_profile,
            max_result_rows=0,
        )


def test_agent_rejects_invalid_max_explanation_rows(
    dataframe,
    dataset_profile,
):
    with pytest.raises(ValueError):
        DataAnalystAgent(
            dataframe=dataframe,
            dataset_profile=dataset_profile,
            max_explanation_rows=-1,
        )


def test_agent_rejects_invalid_max_chart_rows(
    dataframe,
    dataset_profile,
):
    with pytest.raises(ValueError):
        DataAnalystAgent(
            dataframe=dataframe,
            dataset_profile=dataset_profile,
            max_chart_rows=0,
        )


def test_agent_rejects_invalid_max_categories(
    dataframe,
    dataset_profile,
):
    with pytest.raises(ValueError):
        DataAnalystAgent(
            dataframe=dataframe,
            dataset_profile=dataset_profile,
            max_categories=0,
        )


# --------------------------------------------------------------------------
# Question validation
# --------------------------------------------------------------------------


def test_agent_rejects_empty_question(
    dataframe,
    dataset_profile,
    mock_planner,
):
    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=mock_planner,
    )

    with pytest.raises(ValueError):
        agent.run("")


def test_agent_rejects_whitespace_question(
    dataframe,
    dataset_profile,
    mock_planner,
):
    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=mock_planner,
    )

    with pytest.raises(ValueError):
        agent.run("   ")


def test_agent_rejects_non_string_question(
    dataframe,
    dataset_profile,
    mock_planner,
):
    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=mock_planner,
    )

    with pytest.raises(ValueError):
        agent.run(None)


# --------------------------------------------------------------------------
# Planning tests
# --------------------------------------------------------------------------


def test_agent_calls_query_planner(
    dataframe,
    dataset_profile,
    mock_planner,
    monkeypatch,
    mock_query_result,
):
    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=mock_planner,
    )

    monkeypatch.setattr(
        "app.core.agent.explain_query_result",
        lambda **kwargs: "Explanation.",
    )

    monkeypatch.setattr(
        "app.core.agent.generate_chart",
        lambda **kwargs: None,
    )

    # SQLExecutor is tested independently. Here we replace it so this
    # test focuses on orchestration.
    mock_executor = MagicMock()

    mock_executor.__enter__.return_value.execute.return_value = (
        mock_query_result
    )

    monkeypatch.setattr(
        "app.core.agent.SQLExecutor",
        lambda *args, **kwargs: mock_executor,
    )

    result = agent.run(
        "What are the top products by sales?"
    )

    mock_planner.plan.assert_called_once()

    assert result.sql.startswith("SELECT")
    assert result.query_result is mock_query_result


def test_agent_forwards_conversation_context_to_planner(
    dataframe,
    dataset_profile,
    mock_planner,
    monkeypatch,
    mock_query_result,
):
    """Follow-up context is passed to planning, not directly to execution."""

    mock_executor = MagicMock()
    mock_executor.__enter__.return_value.execute.return_value = (
        mock_query_result
    )

    monkeypatch.setattr(
        "app.core.agent.SQLExecutor",
        lambda *args, **kwargs: mock_executor,
    )

    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=mock_planner,
    )

    context = [
        {
            "question": "What are total sales by product?",
            "sql": (
                "SELECT product, SUM(sales) AS total_sales "
                "FROM dataset GROUP BY product"
            ),
        }
    ]

    agent.run(
        "Now show only the top two.",
        generate_explanation=False,
        generate_chart_spec=False,
        conversation_context=context,
    )

    mock_planner.plan.assert_called_once_with(
        question="Now show only the top two.",
        dataset_profile=dataset_profile,
        conversation_context=context,
    )


def test_agent_wraps_planning_error(
    dataframe,
    dataset_profile,
):
    planner = MagicMock()
    planner.plan.side_effect = QueryPlanningError(
        "Planner failed"
    )

    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=planner,
    )

    with pytest.raises(AgentPlanningError):
        agent.run("Top products")


def test_agent_rejects_empty_planner_response(
    dataframe,
    dataset_profile,
):
    planner = MagicMock()
    planner.plan.return_value = ""

    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=planner,
    )

    with pytest.raises(AgentPlanningError):
        agent.run("Top products")


# --------------------------------------------------------------------------
# Security validation tests
# --------------------------------------------------------------------------


def test_agent_rejects_unsafe_sql(
    dataframe,
    dataset_profile,
):
    planner = MagicMock()
    planner.plan.return_value = (
        "DROP TABLE dataset"
    )

    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=planner,
    )

    with pytest.raises(AgentValidationError):
        agent.run("Delete everything")


def test_agent_rejects_external_table(
    dataframe,
    dataset_profile,
):
    planner = MagicMock()
    planner.plan.return_value = (
        "SELECT * FROM secret_table"
    )

    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=planner,
    )

    with pytest.raises(AgentValidationError):
        agent.run("Show secret data")


# --------------------------------------------------------------------------
# Execution tests
# --------------------------------------------------------------------------


def test_agent_executes_only_validated_sql(
    dataframe,
    dataset_profile,
    mock_planner,
    monkeypatch,
    mock_query_result,
):
    mock_executor = MagicMock()

    mock_executor.__enter__.return_value.execute.return_value = (
        mock_query_result
    )

    monkeypatch.setattr(
        "app.core.agent.SQLExecutor",
        lambda *args, **kwargs: mock_executor,
    )

    monkeypatch.setattr(
        "app.core.agent.explain_query_result",
        lambda **kwargs: "Explanation.",
    )

    monkeypatch.setattr(
        "app.core.agent.generate_chart",
        lambda **kwargs: None,
    )

    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=mock_planner,
    )

    result = agent.run(
        "Top products by sales"
    )

    execute_call = (
        mock_executor.__enter__.return_value.execute.call_args
    )

    validation_result = execute_call.args[0]

    assert validation_result.is_valid is True
    assert validation_result.cleaned_sql is not None
    assert "LIMIT" in validation_result.cleaned_sql.upper()

    assert result.query_result is mock_query_result


def test_agent_wraps_query_execution_error(
    dataframe,
    dataset_profile,
    mock_planner,
    monkeypatch,
):
    mock_executor = MagicMock()

    mock_executor.__enter__.return_value.execute.side_effect = (
        QueryExecutionError("DuckDB failed")
    )

    monkeypatch.setattr(
        "app.core.agent.SQLExecutor",
        lambda *args, **kwargs: mock_executor,
    )

    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=mock_planner,
    )

    with pytest.raises(AgentExecutionError):
        agent.run("Top products")


# --------------------------------------------------------------------------
# Explanation tests
# --------------------------------------------------------------------------


def test_agent_generates_explanation(
    dataframe,
    dataset_profile,
    mock_planner,
    monkeypatch,
    mock_query_result,
):
    mock_executor = MagicMock()

    mock_executor.__enter__.return_value.execute.return_value = (
        mock_query_result
    )

    monkeypatch.setattr(
        "app.core.agent.SQLExecutor",
        lambda *args, **kwargs: mock_executor,
    )

    explainer = MagicMock()
    explainer.return_value = "The top product is C."

    monkeypatch.setattr(
        "app.core.agent.explain_query_result",
        explainer,
    )

    monkeypatch.setattr(
        "app.core.agent.generate_chart",
        lambda **kwargs: None,
    )

    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=mock_planner,
    )

    result = agent.run(
        "Top products",
        generate_explanation=True,
        generate_chart_spec=False,
    )

    assert result.explanation == (
        "The top product is C."
    )

    explainer.assert_called_once()


def test_agent_can_skip_explanation(
    dataframe,
    dataset_profile,
    mock_planner,
    monkeypatch,
    mock_query_result,
):
    mock_executor = MagicMock()

    mock_executor.__enter__.return_value.execute.return_value = (
        mock_query_result
    )

    monkeypatch.setattr(
        "app.core.agent.SQLExecutor",
        lambda *args, **kwargs: mock_executor,
    )

    explainer = MagicMock()

    monkeypatch.setattr(
        "app.core.agent.explain_query_result",
        explainer,
    )

    monkeypatch.setattr(
        "app.core.agent.generate_chart",
        lambda **kwargs: None,
    )

    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=mock_planner,
    )

    result = agent.run(
        "Top products",
        generate_explanation=False,
        generate_chart_spec=False,
    )

    assert result.explanation is None
    explainer.assert_not_called()


def test_agent_wraps_explanation_error(
    dataframe,
    dataset_profile,
    mock_planner,
    monkeypatch,
    mock_query_result,
):
    mock_executor = MagicMock()

    mock_executor.__enter__.return_value.execute.return_value = (
        mock_query_result
    )

    monkeypatch.setattr(
        "app.core.agent.SQLExecutor",
        lambda *args, **kwargs: mock_executor,
    )

    monkeypatch.setattr(
        "app.core.agent.explain_query_result",
        MagicMock(
            side_effect=ExplainerLLMError(
                "Bedrock failed"
            )
        ),
    )

    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=mock_planner,
    )

    result = agent.run(
        "Top products",
        generate_chart_spec=False,
    )

    assert result.query_result is mock_query_result
    assert result.explanation is None
    assert result.warnings == [
        "The query succeeded, but an explanation could not be generated."
    ]


# --------------------------------------------------------------------------
# Chart tests
# --------------------------------------------------------------------------


def test_agent_generates_chart(
    dataframe,
    dataset_profile,
    mock_planner,
    monkeypatch,
    mock_query_result,
):
    mock_executor = MagicMock()

    mock_executor.__enter__.return_value.execute.return_value = (
        mock_query_result
    )

    monkeypatch.setattr(
        "app.core.agent.SQLExecutor",
        lambda *args, **kwargs: mock_executor,
    )

    monkeypatch.setattr(
        "app.core.agent.explain_query_result",
        lambda **kwargs: "Explanation.",
    )

    chart = {
        "chart_type": "bar",
        "x_column": "product",
        "y_column": "total_sales",
        "data": [],
    }

    chart_generator = MagicMock(
        return_value=chart
    )

    monkeypatch.setattr(
        "app.core.agent.generate_chart",
        chart_generator,
    )

    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=mock_planner,
    )

    result = agent.run(
        "Top products",
        generate_explanation=False,
        generate_chart_spec=True,
    )

    assert result.chart == chart
    chart_generator.assert_called_once()


def test_agent_can_skip_chart(
    dataframe,
    dataset_profile,
    mock_planner,
    monkeypatch,
    mock_query_result,
):
    mock_executor = MagicMock()

    mock_executor.__enter__.return_value.execute.return_value = (
        mock_query_result
    )

    monkeypatch.setattr(
        "app.core.agent.SQLExecutor",
        lambda *args, **kwargs: mock_executor,
    )

    monkeypatch.setattr(
        "app.core.agent.explain_query_result",
        lambda **kwargs: "Explanation.",
    )

    chart_generator = MagicMock()

    monkeypatch.setattr(
        "app.core.agent.generate_chart",
        chart_generator,
    )

    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=mock_planner,
    )

    result = agent.run(
        "Top products",
        generate_explanation=False,
        generate_chart_spec=False,
    )

    assert result.chart is None
    chart_generator.assert_not_called()


def test_agent_wraps_chart_error(
    dataframe,
    dataset_profile,
    mock_planner,
    monkeypatch,
    mock_query_result,
):
    mock_executor = MagicMock()

    mock_executor.__enter__.return_value.execute.return_value = (
        mock_query_result
    )

    monkeypatch.setattr(
        "app.core.agent.SQLExecutor",
        lambda *args, **kwargs: mock_executor,
    )

    monkeypatch.setattr(
        "app.core.agent.generate_chart",
        MagicMock(
            side_effect=ValueError(
                "Invalid chart type"
            )
        ),
    )

    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=mock_planner,
    )

    result = agent.run(
        "Top products",
        generate_explanation=False,
    )

    assert result.query_result is mock_query_result
    assert result.chart is None
    assert result.warnings == [
        "The query succeeded, but a visualization could not be generated."
    ]


# --------------------------------------------------------------------------
# Result contract tests
# --------------------------------------------------------------------------


def test_agent_result_contains_all_pipeline_outputs(
    dataframe,
    dataset_profile,
    mock_planner,
    monkeypatch,
    mock_query_result,
):
    mock_executor = MagicMock()

    mock_executor.__enter__.return_value.execute.return_value = (
        mock_query_result
    )

    monkeypatch.setattr(
        "app.core.agent.SQLExecutor",
        lambda *args, **kwargs: mock_executor,
    )

    monkeypatch.setattr(
        "app.core.agent.explain_query_result",
        lambda **kwargs: "Explanation.",
    )

    chart = {
        "chart_type": "bar",
        "data": [],
    }

    monkeypatch.setattr(
        "app.core.agent.generate_chart",
        lambda **kwargs: chart,
    )

    agent = DataAnalystAgent(
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=mock_planner,
    )

    result = agent.run(
        "Top products"
    )

    assert isinstance(result, AgentResult)

    assert result.question == "Top products"
    assert result.sql.startswith("SELECT")
    assert result.validation.is_valid is True
    assert result.query_result is mock_query_result
    assert result.explanation == "Explanation."
    assert result.chart == chart


# --------------------------------------------------------------------------
# Convenience wrapper tests
# --------------------------------------------------------------------------


def test_run_analysis_wrapper(
    dataframe,
    dataset_profile,
    mock_planner,
    monkeypatch,
    mock_query_result,
):
    mock_executor = MagicMock()

    mock_executor.__enter__.return_value.execute.return_value = (
        mock_query_result
    )

    monkeypatch.setattr(
        "app.core.agent.SQLExecutor",
        lambda *args, **kwargs: mock_executor,
    )

    monkeypatch.setattr(
        "app.core.agent.explain_query_result",
        lambda **kwargs: "Explanation.",
    )

    monkeypatch.setattr(
        "app.core.agent.generate_chart",
        lambda **kwargs: None,
    )

    result = run_analysis(
        question="Top products",
        dataframe=dataframe,
        dataset_profile=dataset_profile,
        query_planner=mock_planner,
    )

    assert isinstance(result, AgentResult)
    assert result.question == "Top products"
    assert result.query_result is mock_query_result