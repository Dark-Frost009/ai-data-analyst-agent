"""
Natural-language query planner for the AI Data Analyst Agent.

Converts a user's natural-language analytical question into exactly one
DuckDB-compatible SQL SELECT statement using Amazon Bedrock.

Architectural boundary:

    User question
        ↓
    QueryPlanner
        ↓
    Generated SQL
        ↓
    Security validation
        ↓
    SQLExecutor
        ↓
    QueryResult

QueryPlanner NEVER executes SQL and NEVER talks directly to DuckDB.
"""

import re
from typing import Optional

from app.core.llm_client import BedrockClient, get_bedrock_client
from app.models.schemas import DatasetProfile


# ============================================================================
# Exceptions
# ============================================================================


class QueryPlannerError(Exception):
    """Base class for errors raised by the query planner."""


class QueryPlanningError(QueryPlannerError):
    """The LLM response could not be converted into usable SQL."""


class EmptyQuestionError(QueryPlannerError):
    """The user's analytical question was empty."""


# ============================================================================
# System prompt
# ============================================================================


DEFAULT_SYSTEM_PROMPT = """
You are an expert data analyst and DuckDB SQL query planner.

Your job is to convert a user's natural-language analytical question into
ONE safe, read-only DuckDB-compatible SQL SELECT statement.

The dataset is available as a table named `dataset`.

======================================================================
CORE RULES
======================================================================

1. Return exactly ONE SQL SELECT statement.
2. Never return explanations.
3. Never return Markdown.
4. Never return code fences.
5. Never return multiple SQL statements.
6. Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE,
   COPY, ATTACH, DETACH, INSTALL, LOAD, CALL, EXPORT, IMPORT, or any
   other data-modifying, data-definition, file-access, extension,
   network-access, or administrative operation.
7. Use ONLY columns that actually exist in the supplied dataset profile.
8. NEVER invent columns.
9. NEVER rename or normalize a column name.
10. Preserve the exact spelling of every column.
11. Preserve capitalization.
12. Preserve spaces.
13. Preserve underscores.
14. Preserve punctuation.
15. Preserve parentheses and brackets.
16. Preserve hyphens.
17. Preserve Unicode characters.
18. Preserve Unicode whitespace, including non-breaking spaces.
19. When referencing a column, surround its exact name with double quotes.
20. Query ONLY the table `dataset`.
21. Use DuckDB-compatible SQL.
22. Handle NULL values sensibly.
23. Return ONLY the SQL SELECT statement.

======================================================================
EXACT COLUMN NAME HANDLING
======================================================================

The dataset profile is the authoritative source of column names.

Column names must be copied EXACTLY from the profile.

This is extremely important because CSV files may contain invisible
Unicode characters such as non-breaking spaces.

Do NOT:

- replace spaces with underscores
- remove spaces
- change capitalization
- replace Unicode whitespace with ASCII whitespace
- remove punctuation
- simplify column names
- invent alternative spellings

Always prefer:

"Exact column name from profile"

over an unquoted or normalized identifier.

Always quote dataset column names using double quotes.

======================================================================
NUMERIC COLUMN HANDLING
======================================================================

First inspect the dataset profile.

If a column is already numeric, use it directly.

If a numeric-looking column is VARCHAR/text, use:

TRY_CAST("column" AS DOUBLE)

when appropriate.

For numeric text containing commas, use nested REPLACE calls:

TRY_CAST(
    REPLACE("column", ',', '')
    AS DOUBLE
)

For example:

"1,234,567.89"

becomes:

1234567.89

Do NOT use REGEXP_REPLACE.

======================================================================
CURRENCY HANDLING
======================================================================

Currency values may appear as text such as:

"$1,234.50"
"£1,234.50"
"€1,234.50"
"₹1,25,000"
"$ 1,234.50"

When the relevant column is VARCHAR/text, safely remove the currency
symbol and commas using nested REPLACE calls.

Example:

TRY_CAST(
    REPLACE(
        REPLACE(
            REPLACE(
                REPLACE(
                    REPLACE(
                        CAST("column" AS VARCHAR),
                        ',',
                        ''
                    ),
                    '$',
                    ''
                ),
                '₹',
                ''
            ),
            '£',
            ''
        ),
        '€',
        ''
    )
    AS DOUBLE
)

Use only the transformations actually required by the dataset.

Do NOT perform currency conversion between currencies unless the user
explicitly asks for it.

"$1,000" means 1000, not a converted foreign-currency amount.

======================================================================
PERCENTAGE HANDLING
======================================================================

Percentage values may appear as:

"25%"
"12.5%"
"100%"

When a VARCHAR percentage contains `%`, remove `%` before converting:

TRY_CAST(
    REPLACE(
        CAST("percentage_column" AS VARCHAR),
        '%',
        ''
    )
    AS DOUBLE
)

Only divide by 100 when the user explicitly asks for a proportion
between 0 and 1.

Example:

TRY_CAST(
    REPLACE(
        CAST("percentage_column" AS VARCHAR),
        '%',
        ''
    )
    AS DOUBLE
) / 100.0

Do NOT divide by 100 simply because a column represents a percentage.

If values are already stored as proportions such as 0.25, keep them
as 0.25 unless the user explicitly requests another representation.

======================================================================
NEGATIVE CURRENCY / NUMERIC VALUES
======================================================================

Some financial datasets represent negative values using parentheses:

(1,234.50)

If the dataset actually uses this representation and numeric calculations
are required, handle it with CASE.

Example:

CASE
    WHEN TRIM(CAST("column" AS VARCHAR)) LIKE '(%)'
    THEN
        -TRY_CAST(
            REPLACE(
                REPLACE(
                    TRIM(CAST("column" AS VARCHAR)),
                    '(',
                    ''
                ),
                ')',
                ''
            )
            AS DOUBLE
        )
    ELSE
        TRY_CAST(
            REPLACE(
                TRIM(CAST("column" AS VARCHAR)),
                ',',
                ''
            )
            AS DOUBLE
        )
END

Only use this complexity when necessary.

======================================================================
DATE AND TIMESTAMP HANDLING
======================================================================

Some CSV date/time columns may be loaded as VARCHAR/text.

Before applying:

DATE_TRUNC
DATE_PART
EXTRACT

to a VARCHAR date/time column, convert it safely.

For dates:

TRY_CAST("date_column" AS DATE)

For timestamps:

TRY_CAST("timestamp_column" AS TIMESTAMP)

For monthly analysis:

DATE_TRUNC(
    'MONTH',
    TRY_CAST("date_column" AS DATE)
)

Do NOT apply DATE_TRUNC, DATE_PART, or EXTRACT directly to a VARCHAR
date column.

Use TRY_CAST so invalid values become NULL instead of causing the
entire query to fail.

Do not unnecessarily cast columns that are already DATE or TIMESTAMP.

======================================================================
AGGREGATIONS
======================================================================

Use appropriate aggregation functions:

SUM
AVG
COUNT
MIN
MAX
ROUND

Use GROUP BY whenever aggregation requires it.

When the user asks for an average, use AVG.

When the user asks for total revenue/gross/sales, use SUM.

When the user asks how many records/items there are, use COUNT.

======================================================================
PERCENTAGE INCREASE / DECREASE
======================================================================

When calculating percentage increase:

((new_value - old_value) / old_value) * 100

When calculating percentage decrease:

((old_value - new_value) / old_value) * 100

Only use this formula when the user is asking for percentage increase
or decrease.

Protect against division by zero when appropriate.

For example:

CASE
    WHEN old_value IS NULL OR old_value = 0 THEN NULL
    ELSE ((new_value - old_value) / old_value) * 100
END

For VARCHAR numeric values, clean and convert them before performing
the calculation.

======================================================================
TOP / BOTTOM / RANKING QUESTIONS
======================================================================

For:

"top 5"
"highest 10"
"best 3"
"lowest 5"
"bottom 10"

use ORDER BY and LIMIT.

Top/highest:

ORDER BY value DESC
LIMIT N

Bottom/lowest:

ORDER BY value ASC
LIMIT N

For numeric VARCHAR columns, perform the safe numeric conversion before
ordering.

If the numeric expression can produce NULL, exclude NULL values when
that is necessary for a meaningful ranking.

Example:

SELECT
    "Tour title",
    TRY_CAST(
        REPLACE(
            CAST("Actual gross" AS VARCHAR),
            ',',
            ''
        )
        AS DOUBLE
    ) AS actual_gross
FROM dataset
WHERE TRY_CAST(
    REPLACE(
        CAST("Actual gross" AS VARCHAR),
        ',',
        ''
    )
    AS DOUBLE
) IS NOT NULL
ORDER BY actual_gross DESC
LIMIT 5

IMPORTANT:

The column name above is illustrative.

Always replace it with the EXACT column name from the actual dataset
profile.

======================================================================
NULL HANDLING
======================================================================

Do not automatically convert NULL to zero.

Use COALESCE only when zero is logically appropriate.

For rankings, it is often appropriate to exclude rows where the ranking
value is NULL.

======================================================================
AVAILABLE SQL FUNCTIONS
======================================================================

Prefer only these safe functions:

SUM
AVG
COUNT
MIN
MAX
ROUND
ABS
CEIL
CEILING
FLOOR
LOWER
UPPER
TRIM
LENGTH
CONCAT
REPLACE
COALESCE
NULLIF
DATE_TRUNC
DATETRUNC
TIMESTAMPTRUNC
DATE_PART
DATEPART
EXTRACT
CAST
TRY_CAST
CASE

Do NOT use:

REGEXP_REPLACE
REGEXP_MATCHES
REGEXP_EXTRACT
SUBSTRING
STRPOS
FORMAT
IF
ANY other function not explicitly allowed.

======================================================================
SQL SAFETY
======================================================================

The generated query must:

- Start with SELECT.
- Query only `dataset`.
- Be read-only.
- Contain exactly one SQL statement.
- Never modify data.
- Never access files.
- Never access URLs.
- Never load extensions.
- Never install extensions.
- Never attach databases.
- Never query external tables.
- Never execute procedures.
- Never contain multiple statements.

Return ONLY the SQL SELECT statement.
""".strip()


# ============================================================================
# QueryPlanner
# ============================================================================


class QueryPlanner:
    """
    Convert natural-language analytical questions into SQL.

    The planner is deliberately isolated from SQL execution.
    """

    def __init__(
        self,
        llm_client: Optional[BedrockClient] = None,
        system_prompt: Optional[str] = None,
    ):
        self._llm_client = llm_client or get_bedrock_client()
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def plan(
        self,
        question: str,
        dataset_profile: DatasetProfile,
    ) -> str:
        """
        Generate one SQL SELECT statement for the user's question.
        """

        if not isinstance(question, str) or not question.strip():
            raise EmptyQuestionError(
                "The analytical question must be a non-empty string."
            )

        if not isinstance(dataset_profile, DatasetProfile):
            raise ValueError(
                "dataset_profile must be an instance of DatasetProfile"
            )

        prompt = self._build_prompt(
            question=question.strip(),
            dataset_profile=dataset_profile,
        )

        response = self._llm_client.generate_text(
            prompt=prompt,
            system_prompt=self._system_prompt,
            max_tokens=1024,
            temperature=0.0,
        )

        return self._extract_sql(response)

    # ----------------------------------------------------------------------
    # Prompt construction
    # ----------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        question: str,
        dataset_profile: DatasetProfile,
    ) -> str:
        """
        Build the prompt containing the complete dataset profile.
        """

        profile_json = dataset_profile.model_dump_json(indent=2)

        return f"""
Dataset information:

{profile_json}

======================================================================
AUTHORITATIVE SCHEMA
======================================================================

The dataset information above is the source of truth.

Use ONLY columns present in that profile.

Column identifiers must be copied EXACTLY.

Preserve:

- capitalization
- spaces
- underscores
- punctuation
- parentheses
- brackets
- hyphens
- Unicode characters
- Unicode whitespace
- non-breaking spaces

Always quote column names with double quotes.

Do NOT normalize a column name.

Do NOT:

- replace spaces with underscores
- remove spaces
- change capitalization
- replace Unicode whitespace
- simplify punctuation
- invent columns
- rename columns

======================================================================
NUMERIC VALUES
======================================================================

If a column is numeric, use it directly.

If a numeric-looking column is VARCHAR/text, use TRY_CAST.

For commas:

TRY_CAST(
    REPLACE("column", ',', '')
    AS DOUBLE
)

For currency values, remove only the required currency symbols and
commas using nested REPLACE calls.

For percentages containing `%`:

TRY_CAST(
    REPLACE(
        CAST("column" AS VARCHAR),
        '%',
        ''
    )
    AS DOUBLE
)

Only divide by 100 when the user explicitly requests a 0-to-1
proportion.

Do NOT use REGEXP_REPLACE.

======================================================================
DATES
======================================================================

If a date/time column is VARCHAR/text:

TRY_CAST("date_column" AS DATE)

or:

TRY_CAST("timestamp_column" AS TIMESTAMP)

For monthly analysis:

DATE_TRUNC(
    'MONTH',
    TRY_CAST("date_column" AS DATE)
)

Do not apply DATE_TRUNC, DATE_PART, or EXTRACT directly to a VARCHAR
date column.

======================================================================
PERCENTAGE CALCULATIONS
======================================================================

For percentage increase:

((new_value - old_value) / old_value) * 100

For percentage decrease:

((old_value - new_value) / old_value) * 100

Protect against division by zero with CASE when appropriate.

======================================================================
RANKING
======================================================================

For top/highest questions:

ORDER BY value DESC
LIMIT N

For bottom/lowest questions:

ORDER BY value ASC
LIMIT N

If the ranking value is a formatted numeric VARCHAR, clean and safely
convert it before ordering.

Exclude NULL ranking values when appropriate.

======================================================================
SQL REQUIREMENTS
======================================================================

- Generate exactly ONE SELECT statement.
- Query only `dataset`.
- Use only columns from the dataset profile.
- Use exact column names.
- Quote column names with double quotes.
- Use DuckDB-compatible SQL.
- Use only safe/allowed functions.
- Do not use REGEXP_REPLACE.
- Do not include Markdown.
- Do not include code fences.
- Do not include explanations.
- Do not include multiple statements.
- Do not include a semicolon except an optional final semicolon.

======================================================================
USER'S ANALYTICAL QUESTION
======================================================================

{question}

Return ONLY the SQL SELECT statement.
""".strip()

    # ----------------------------------------------------------------------
    # SQL extraction
    # ----------------------------------------------------------------------

    @staticmethod
    def _extract_sql(response: str) -> str:
        """
        Extract and normalize exactly one SQL SELECT statement.

        This is intentionally lightweight.

        The authoritative SQL security validator performs the final
        AST-level validation.
        """

        if not isinstance(response, str) or not response.strip():
            raise QueryPlanningError(
                "The LLM returned an empty response while planning the query."
            )

        sql = response.strip()

        # --------------------------------------------------------------
        # Remove Markdown code fences if the model ignored instructions.
        # --------------------------------------------------------------

        fenced_match = re.fullmatch(
            r"```(?:sql|SQL)?\s*(.*?)\s*```",
            sql,
            flags=re.DOTALL,
        )

        if fenced_match:
            sql = fenced_match.group(1).strip()

        # --------------------------------------------------------------
        # Remove accidental leading/trailing whitespace.
        # --------------------------------------------------------------

        sql = sql.strip()

        # --------------------------------------------------------------
        # Remove a single trailing semicolon.
        # --------------------------------------------------------------

        if sql.endswith(";"):
            sql = sql[:-1].rstrip()

        if not sql:
            raise QueryPlanningError(
                "The LLM response did not contain a SQL query."
            )

        # --------------------------------------------------------------
        # Must begin with SELECT.
        # --------------------------------------------------------------

        if not re.match(
            r"^\s*SELECT\b",
            sql,
            flags=re.IGNORECASE,
        ):
            raise QueryPlanningError(
                "The LLM response does not contain a SELECT statement."
            )

        # --------------------------------------------------------------
        # Reject multiple statements.
        # --------------------------------------------------------------

        if ";" in sql:
            raise QueryPlanningError(
                "The LLM response contains multiple SQL statements."
            )

        # --------------------------------------------------------------
        # Additional planner-level dangerous keyword check.
        #
        # The security validator remains authoritative.
        # --------------------------------------------------------------

        dangerous_keywords = (
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "ALTER",
            "CREATE",
            "TRUNCATE",
            "COPY",
            "ATTACH",
            "DETACH",
            "INSTALL",
            "LOAD",
            "CALL",
            "EXPORT",
            "IMPORT",
        )

        for keyword in dangerous_keywords:
            if re.search(
                rf"\b{keyword}\b",
                sql,
                flags=re.IGNORECASE,
            ):
                raise QueryPlanningError(
                    "The generated SQL contains a forbidden SQL operation: "
                    f"{keyword}"
                )

        return sql