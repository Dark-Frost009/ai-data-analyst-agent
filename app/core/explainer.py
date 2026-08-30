"""
LLM-powered explanation generation.

Takes a validated SQL query and its QueryResult, then uses the shared
BedrockClient to produce a concise, human-readable explanation of the
analysis.

This module intentionally does not:

- execute SQL,
- validate SQL,
- generate charts,
- access Streamlit,
- access AWS/Bedrock directly.

All Bedrock communication goes through BedrockClient.
"""

from typing import Any, Optional

import pandas as pd

from app.core.llm_client import (
    BedrockClient,
    BedrockClientError,
    get_bedrock_client,
)
from app.models.schemas import QueryResult
from app.utils.logger import get_logger


logger = get_logger(__name__)


DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RESULT_ROWS = 20


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class ExplainerError(Exception):
    """Base class for all errors raised by the explainer."""


class ExplainerLLMError(ExplainerError):
    """Raised when the underlying LLM/Bedrock request fails."""


class ExplainerResponseError(ExplainerError):
    """Raised when the LLM returns an unusable explanation."""


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def explain_query_result(
    sql: str,
    query_result: QueryResult,
    bedrock_client: Optional[BedrockClient] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_result_rows: int = DEFAULT_MAX_RESULT_ROWS,
) -> str:
    """
    Generate a human-readable explanation of a SQL query result.

    Parameters
    ----------
    sql:
        The SQL statement that produced the result.

    query_result:
        QueryResult containing the resulting pandas DataFrame.

    bedrock_client:
        Optional BedrockClient. If omitted, the shared singleton is used.

    max_tokens:
        Maximum number of tokens requested from the LLM.

    temperature:
        Sampling temperature passed to Bedrock.

    max_result_rows:
        Maximum number of result rows included in the LLM prompt.

    Returns
    -------
    str
        The generated explanation.

    Raises
    ------
    ValueError
        If the SQL is empty or numeric arguments are invalid.

    TypeError
        If query_result is not a QueryResult.

    ExplainerLLMError
        If Bedrock fails.

    ExplainerResponseError
        If Bedrock returns an empty or invalid explanation.
    """

    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("sql must be a non-empty string")

    if not isinstance(query_result, QueryResult):
        raise TypeError("query_result must be a QueryResult")

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    if temperature < 0:
        raise ValueError(
            "temperature must be greater than or equal to 0"
        )

    if max_result_rows < 0:
        raise ValueError(
            "max_result_rows must be greater than or equal to 0"
        )

    client = bedrock_client or get_bedrock_client()

    result_summary = _build_result_summary(
        query_result=query_result,
        max_result_rows=max_result_rows,
    )

    prompt = _build_prompt(
        sql=sql,
        query_result=query_result,
        result_summary=result_summary,
    )

    system_prompt = _build_system_prompt()

    logger.debug(
        "Generating explanation | query_rows=%d | truncated=%s | "
        "result_rows_in_prompt=%d",
        query_result.row_count,
        query_result.truncated,
        min(query_result.row_count, max_result_rows),
    )

    try:
        explanation = client.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except BedrockClientError as exc:
        logger.error(
            "Bedrock explanation request failed: %s",
            exc,
        )

        raise ExplainerLLMError(
            f"Could not generate explanation from the language model: {exc}"
        ) from exc

    if not isinstance(explanation, str) or not explanation.strip():
        raise ExplainerResponseError(
            "The language model returned an empty explanation."
        )

    explanation = explanation.strip()

    logger.info(
        "Generated explanation | characters=%d",
        len(explanation),
    )

    return explanation


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


def _build_system_prompt() -> str:
    """Return the system instructions used for explanation generation."""

    return (
        "You are an expert data analyst explaining SQL query results to a "
        "business user. Be accurate, concise, and grounded strictly in the "
        "provided SQL and result data. Do not invent facts, trends, causes, "
        "or business context that is not present in the data. Clearly mention "
        "important limitations such as result truncation when applicable. "
        "Use plain language. Do not output SQL unless it is necessary to "
        "clarify a point."
    )


def _build_prompt(
    sql: str,
    query_result: QueryResult,
    result_summary: str,
) -> str:
    """Build the user prompt sent to the LLM."""

    return f"""
Explain the following data analysis result.

SQL query:
{sql.strip()}

Query result metadata:

- Rows returned: {query_result.row_count}
- Result truncated: {query_result.truncated}

Result data:
{result_summary}

Please provide:

1. A concise summary of what the query found.
2. The most important values or patterns visible in the result.
3. Any useful comparison or ranking that is directly supported by the data.
4. A brief note about limitations if the result was truncated or otherwise
   requires caution.

Do not invent information that is not contained in the SQL or result data.
""".strip()


# --------------------------------------------------------------------------
# Result formatting
# --------------------------------------------------------------------------


def _build_result_summary(
    query_result: QueryResult,
    max_result_rows: int,
) -> str:
    """
    Convert the QueryResult DataFrame into a compact text representation.

    Only a bounded number of rows are included so a large SQL result cannot
    accidentally create an enormous LLM prompt.
    """

    dataframe = query_result.dataframe

    if dataframe.empty:
        return "(The query returned no rows.)"

    if max_result_rows == 0:
        return "(Result rows intentionally omitted.)"

    limited_df = dataframe.head(max_result_rows)

    try:
        records = limited_df.to_dict(orient="records")
    except Exception as exc:
        logger.warning(
            "Could not convert result DataFrame to records: %s",
            exc,
        )

        return (
            "(Result data could not be converted into a readable format.)"
        )

    lines = []

    for index, record in enumerate(records, start=1):
        safe_record = {
            str(key): _format_value(value)
            for key, value in record.items()
        }

        # Build the dictionary representation manually so that:
        #
        #   integers stay integers
        #   floats stay floats
        #   strings stay quoted
        #   booleans stay booleans
        #   missing values are represented as `null`
        #
        # This keeps compatibility with the existing prompt tests while
        # giving the LLM an explicit representation for missing values.

        formatted_parts = []

        for key, value in safe_record.items():
            if value is None:
                formatted_value = "null"
            else:
                formatted_value = repr(value)

            formatted_parts.append(
                f"{key!r}: {formatted_value}"
            )

        formatted_record = (
            "{"
            + ", ".join(formatted_parts)
            + "}"
        )

        lines.append(
            f"{index}. {formatted_record}"
        )

    if query_result.row_count > max_result_rows:
        lines.append(
            f"... {query_result.row_count - max_result_rows} "
            "additional row(s) omitted from the explanation prompt."
        )

    return "\n".join(lines)


def _format_value(value: Any) -> Any:
    """
    Normalize a result value while preserving useful Python scalar types.

    Numeric and boolean values remain numeric/boolean instead of being
    converted to strings.

    Missing values become None and are subsequently rendered as `null`
    in the result summary.

    Datetime values become ISO-formatted strings.
    """

    if value is None:
        return None

    try:
        missing = pd.isna(value)

        # pd.isna(some_list) can return an array rather than a bool.
        if isinstance(missing, bool) and missing:
            return None

    except (TypeError, ValueError):
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, pd.Timedelta):
        return str(value)

    # Convert numpy scalar values to their normal Python equivalents.
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value