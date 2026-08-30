"""
Tests for app.core.explainer.

All tests use a mocked BedrockClient, so they do not require AWS credentials,
network access, or a live Bedrock endpoint.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.core.explainer import (
    ExplainerLLMError,
    ExplainerResponseError,
    _build_prompt,
    _build_result_summary,
    _build_system_prompt,
    explain_query_result,
)
from app.core.llm_client import BedrockAPIError
from app.models.schemas import QueryResult


def _query_result(
    data=None,
    row_count=None,
    truncated=False,
):
    """Create a QueryResult for tests."""
    if data is None:
        data = {
            "product": ["A", "B", "C"],
            "sales": [100, 250, 175],
        }

    dataframe = pd.DataFrame(data)

    if row_count is None:
        row_count = len(dataframe)

    return QueryResult(
        dataframe=dataframe,
        row_count=row_count,
        truncated=truncated,
    )


# --------------------------------------------------------------------------
# 1. Successful explanation
# --------------------------------------------------------------------------


def test_explain_query_result_success():
    client = MagicMock()
    client.generate_text.return_value = (
        "Product B generated the highest sales at 250."
    )

    result = _query_result()

    explanation = explain_query_result(
        sql="SELECT product, sales FROM sales ORDER BY sales DESC",
        query_result=result,
        bedrock_client=client,
    )

    assert explanation == "Product B generated the highest sales at 250."
    client.generate_text.assert_called_once()


# --------------------------------------------------------------------------
# 2. Whitespace is stripped from SQL and response
# --------------------------------------------------------------------------


def test_explain_query_result_strips_response_whitespace():
    client = MagicMock()
    client.generate_text.return_value = "  The result is positive. \n"

    result = _query_result()

    explanation = explain_query_result(
        sql="   SELECT * FROM sales   ",
        query_result=result,
        bedrock_client=client,
    )

    assert explanation == "The result is positive."

    called_kwargs = client.generate_text.call_args.kwargs
    assert "SELECT * FROM sales" in called_kwargs["prompt"]


# --------------------------------------------------------------------------
# 3. System prompt is passed
# --------------------------------------------------------------------------


def test_explain_query_result_passes_system_prompt():
    client = MagicMock()
    client.generate_text.return_value = "Summary."

    result = _query_result()

    explain_query_result(
        sql="SELECT * FROM sales",
        query_result=result,
        bedrock_client=client,
    )

    called_kwargs = client.generate_text.call_args.kwargs

    assert called_kwargs["system_prompt"]
    assert "data analyst" in called_kwargs["system_prompt"].lower()


# --------------------------------------------------------------------------
# 4. max_tokens and temperature are forwarded
# --------------------------------------------------------------------------


def test_explain_query_result_passes_generation_parameters():
    client = MagicMock()
    client.generate_text.return_value = "Summary."

    result = _query_result()

    explain_query_result(
        sql="SELECT * FROM sales",
        query_result=result,
        bedrock_client=client,
        max_tokens=300,
        temperature=0.2,
    )

    called_kwargs = client.generate_text.call_args.kwargs

    assert called_kwargs["max_tokens"] == 300
    assert called_kwargs["temperature"] == 0.2


# --------------------------------------------------------------------------
# 5. SQL validation at the explainer boundary
# --------------------------------------------------------------------------


def test_explain_query_result_rejects_empty_sql():
    client = MagicMock()
    result = _query_result()

    with pytest.raises(ValueError):
        explain_query_result(
            sql="   ",
            query_result=result,
            bedrock_client=client,
        )

    client.generate_text.assert_not_called()


def test_explain_query_result_rejects_non_string_sql():
    client = MagicMock()
    result = _query_result()

    with pytest.raises(ValueError):
        explain_query_result(
            sql=None,
            query_result=result,
            bedrock_client=client,
        )

    client.generate_text.assert_not_called()


# --------------------------------------------------------------------------
# 6. QueryResult validation
# --------------------------------------------------------------------------


def test_explain_query_result_rejects_invalid_query_result():
    client = MagicMock()

    with pytest.raises(TypeError):
        explain_query_result(
            sql="SELECT 1",
            query_result=None,
            bedrock_client=client,
        )

    client.generate_text.assert_not_called()


# --------------------------------------------------------------------------
# 7. Empty query result
# --------------------------------------------------------------------------


def test_explain_empty_query_result():
    client = MagicMock()
    client.generate_text.return_value = "The query returned no matching rows."

    result = QueryResult(
        dataframe=pd.DataFrame(columns=["product", "sales"]),
        row_count=0,
        truncated=False,
    )

    explanation = explain_query_result(
        sql="SELECT product, sales FROM sales WHERE sales > 1000000",
        query_result=result,
        bedrock_client=client,
    )

    assert explanation == "The query returned no matching rows."

    called_kwargs = client.generate_text.call_args.kwargs
    assert "(The query returned no rows.)" in called_kwargs["prompt"]


# --------------------------------------------------------------------------
# 8. Truncated results are explicitly included in prompt
# --------------------------------------------------------------------------


def test_explain_truncated_result_mentions_truncation():
    client = MagicMock()
    client.generate_text.return_value = "The result contains the top rows."

    result = _query_result(
        data={
            "product": ["A", "B", "C"],
            "sales": [100, 250, 175],
        },
        row_count=100,
        truncated=True,
    )

    explain_query_result(
        sql="SELECT product, sales FROM sales",
        query_result=result,
        bedrock_client=client,
        max_result_rows=3,
    )

    called_kwargs = client.generate_text.call_args.kwargs

    assert "Rows returned: 100" in called_kwargs["prompt"]
    assert "Result truncated: True" in called_kwargs["prompt"]


# --------------------------------------------------------------------------
# 9. Result rows are bounded
# --------------------------------------------------------------------------


def test_explain_result_rows_are_bounded():
    client = MagicMock()
    client.generate_text.return_value = "Summary."

    dataframe = pd.DataFrame(
        {
            "id": list(range(1, 11)),
            "value": list(range(101, 111)),
        }
    )

    result = QueryResult(
        dataframe=dataframe,
        row_count=10,
        truncated=False,
    )

    explain_query_result(
        sql="SELECT id, value FROM data",
        query_result=result,
        bedrock_client=client,
        max_result_rows=3,
    )

    called_kwargs = client.generate_text.call_args.kwargs
    prompt = called_kwargs["prompt"]

    assert "1. {'id': 1" in prompt
    assert "2. {'id': 2" in prompt
    assert "3. {'id': 3" in prompt

    assert "4. {'id': 4" not in prompt
    assert "additional row(s) omitted" in prompt


# --------------------------------------------------------------------------
# 10. Zero result rows allowed in prompt
# --------------------------------------------------------------------------


def test_build_result_summary_zero_rows_limit():
    result = _query_result()

    summary = _build_result_summary(
        query_result=result,
        max_result_rows=0,
    )

    assert summary == "(Result rows intentionally omitted.)"


# --------------------------------------------------------------------------
# 11. Bedrock errors are wrapped
# --------------------------------------------------------------------------


def test_explain_query_result_wraps_bedrock_error():
    client = MagicMock()
    client.generate_text.side_effect = BedrockAPIError(
        "Bedrock failed",
        error_code="ValidationException",
    )

    result = _query_result()

    with pytest.raises(ExplainerLLMError) as excinfo:
        explain_query_result(
            sql="SELECT * FROM sales",
            query_result=result,
            bedrock_client=client,
        )

    assert "Could not generate explanation" in str(excinfo.value)
    assert "Bedrock failed" in str(excinfo.value)


# --------------------------------------------------------------------------
# 12. Empty LLM response is rejected
# --------------------------------------------------------------------------


def test_explain_query_result_rejects_empty_llm_response():
    client = MagicMock()
    client.generate_text.return_value = "   "

    result = _query_result()

    with pytest.raises(ExplainerResponseError):
        explain_query_result(
            sql="SELECT * FROM sales",
            query_result=result,
            bedrock_client=client,
        )


# --------------------------------------------------------------------------
# 13. None LLM response is rejected
# --------------------------------------------------------------------------


def test_explain_query_result_rejects_none_llm_response():
    client = MagicMock()
    client.generate_text.return_value = None

    result = _query_result()

    with pytest.raises(ExplainerResponseError):
        explain_query_result(
            sql="SELECT * FROM sales",
            query_result=result,
            bedrock_client=client,
        )


# --------------------------------------------------------------------------
# 14. Invalid generation parameters
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 0},
        {"max_tokens": -1},
        {"temperature": -0.1},
        {"max_result_rows": -1},
    ],
)
def test_explain_query_result_rejects_invalid_parameters(kwargs):
    client = MagicMock()
    result = _query_result()

    with pytest.raises(ValueError):
        explain_query_result(
            sql="SELECT * FROM sales",
            query_result=result,
            bedrock_client=client,
            **kwargs,
        )

    client.generate_text.assert_not_called()


# --------------------------------------------------------------------------
# 15. Result values are represented in the prompt
# --------------------------------------------------------------------------


def test_explain_prompt_contains_result_values():
    client = MagicMock()
    client.generate_text.return_value = "Product B performed best."

    result = _query_result(
        data={
            "product": ["A", "B"],
            "sales": [100, 250],
        }
    )

    explain_query_result(
        sql="SELECT product, sales FROM sales",
        query_result=result,
        bedrock_client=client,
    )

    prompt = client.generate_text.call_args.kwargs["prompt"]

    assert "A" in prompt
    assert "B" in prompt
    assert "100" in prompt
    assert "250" in prompt


# --------------------------------------------------------------------------
# 16. Helper: system prompt
# --------------------------------------------------------------------------


def test_build_system_prompt_contains_grounding_instruction():
    prompt = _build_system_prompt()

    assert isinstance(prompt, str)
    assert len(prompt.strip()) > 0
    assert "Do not invent" in prompt


# --------------------------------------------------------------------------
# 17. Helper: prompt contains SQL and metadata
# --------------------------------------------------------------------------


def test_build_prompt_contains_sql_and_metadata():
    result = _query_result(
        data={
            "category": ["Food", "Drinks"],
            "revenue": [500, 300],
        },
        row_count=2,
        truncated=False,
    )

    prompt = _build_prompt(
        sql="SELECT category, revenue FROM sales",
        query_result=result,
        result_summary="1. {'category': 'Food', 'revenue': 500}",
    )

    assert "SELECT category, revenue FROM sales" in prompt
    assert "Rows returned: 2" in prompt
    assert "Result truncated: False" in prompt
    assert "Food" in prompt
    assert "500" in prompt


# --------------------------------------------------------------------------
# 18. pandas missing values become null
# --------------------------------------------------------------------------


def test_explain_result_handles_missing_values():
    client = MagicMock()
    client.generate_text.return_value = "The result contains a missing value."

    result = QueryResult(
        dataframe=pd.DataFrame(
            {
                "name": ["Alice", "Bob"],
                "score": [100.0, float("nan")],
            }
        ),
        row_count=2,
        truncated=False,
    )

    explain_query_result(
        sql="SELECT name, score FROM results",
        query_result=result,
        bedrock_client=client,
    )

    prompt = client.generate_text.call_args.kwargs["prompt"]

    assert "null" in prompt
    assert "nan" not in prompt.lower()