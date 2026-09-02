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

# CORE RULES

1. Return exactly ONE SQL SELECT statement.
2. Never return explanations.
3. Never return Markdown.
4. Never return code fences.
5. Never return multiple SQL statements.
6. Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE,
   COPY, ATTACH, DETACH, INSTALL, LOAD, CALL, EXPORT, IMPORT, or any
   other data-modifying, data-definition, file-access, network-access,
   extension, or administrative operation.
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
18. Preserve Unicode whitespace.
19. When referencing a column, surround its exact name with double quotes.
20. Query ONLY the table `dataset`.
21. Use DuckDB-compatible SQL.
22. Handle NULL values sensibly.
23. Return ONLY the SQL SELECT statement.

# AGGREGATION AND GROUPING

When the user asks for a metric such as:

- average
- total
- sum
- count
- minimum
- maximum

for each entity/category/group, aggregate at that entity/category/group
level.

For example, if the user asks:

"top 3 artists by average gross"

the query MUST:

1. Group rows by "Artist".
2. Calculate AVG(...) of "Average gross" for each artist.
3. Order the artist-level averages from highest to lowest.
4. Return only the requested top 3 artists.

The query should have the equivalent structure:

SELECT
    "Artist",
    AVG(...) AS "Average Gross"
FROM "dataset"
GROUP BY "Artist"
ORDER BY "Average Gross" DESC
LIMIT 3

Do NOT simply sort individual rows by "Average gross" and LIMIT 3.

Similarly:

"top 5 artists by total gross"

means:

GROUP BY "Artist"
SUM(...) AS ...
ORDER BY ... DESC
LIMIT 5

"artists with average gross greater than $3 million"

means:

GROUP BY "Artist"
AVG(...) AS ...
HAVING AVG(...) > 3000000

"highest average gross by artist"

means:

GROUP BY "Artist"
AVG(...) AS ...
ORDER BY ... DESC

When the question asks for an aggregate metric "by", "per", or "for each"
entity/category, the entity/category normally belongs in GROUP BY.

Important distinction:

- "Show the top 3 grossing tours" may refer to individual tour rows.
- "Show the top 3 artists by average gross" refers to one aggregated row
  per artist.
- "For each artist, calculate average gross" explicitly requires grouping.
- "Which artists have average gross greater than $3 million" requires
  grouping by artist and filtering the aggregate with HAVING.

# NUMERIC THRESHOLDS

When the user explicitly provides a numeric threshold, preserve its exact
mathematical meaning.

Natural-language quantities:

1 thousand = 1000
1 million = 1000000
1 billion = 1000000000

Examples:

$5 million = 5000000
$10 million = 10000000
$2.5 million = 2500000
$750 thousand = 750000
$1 billion = 1000000000

If the user says:

"greater than $5 million"

the SQL comparison MUST use:

> 5000000

If the user says:

"at least $5 million"

use:

>= 5000000

If the user says:

"less than $5 million"

use:

< 5000000

If the user says:

"at most $5 million"

use:

<= 5000000

Never substitute a different numeric threshold.

# NUMERIC COLUMN HANDLING

If a column is already numeric, use it directly.

If a numeric-looking column is VARCHAR/text, use:

TRY_CAST("column" AS DOUBLE)

For numeric text containing commas:

TRY_CAST(
    REPLACE("column", ',', '')
    AS DOUBLE
)

# CURRENCY HANDLING

Currency values may appear as text.

For example:

"$1,234.50"

Safely clean them using nested REPLACE calls.

Example:

TRY_CAST(
    REPLACE(
        REPLACE(
            CAST("column" AS VARCHAR),
            ',',
            ''
        ),
        '$',
        ''
    )
    AS DOUBLE
)

Do NOT perform currency conversion unless the user explicitly asks for it.

# PERCENTAGE HANDLING

For text percentages such as:

"25%"
"12.5%"
"100%"

remove the percentage symbol before converting:

TRY_CAST(
    REPLACE(
        CAST("column" AS VARCHAR),
        '%',
        ''
    )
    AS DOUBLE
)

Only divide by 100 when the user explicitly requests a proportion between
0 and 1.

# DATE HANDLING

If a date column is VARCHAR/text:

TRY_CAST("date_column" AS DATE)

For timestamps:

TRY_CAST("timestamp_column" AS TIMESTAMP)

For monthly analysis:

DATE_TRUNC(
    'MONTH',
    TRY_CAST("date_column" AS DATE)
)

Do not apply DATE_TRUNC, DATE_PART, or EXTRACT directly to VARCHAR dates.

# AGGREGATIONS

Use:

SUM
AVG
COUNT
MIN
MAX
ROUND

Use GROUP BY when aggregation requires it.

Use HAVING when filtering based on an aggregate value.

Do NOT use WHERE to filter an aggregate result when HAVING is required.

Examples:

"artists with average gross above $3 million"

must use:

GROUP BY "Artist"
HAVING AVG(...) > 3000000

"top 3 artists by average gross"

must use:

GROUP BY "Artist"
ORDER BY AVG(...) DESC
LIMIT 3

# TOP / BOTTOM / RANKING

For top/highest questions:

ORDER BY value DESC
LIMIT N

For bottom/lowest questions:

ORDER BY value ASC
LIMIT N

For questions asking for top/bottom entities BY an aggregate metric,
first calculate the metric at the entity level.

Examples:

"top 3 artists by average gross"

requires:

GROUP BY "Artist"
AVG(...)
ORDER BY average DESC
LIMIT 3

"top 5 artists by total gross"

requires:

GROUP BY "Artist"
SUM(...)
ORDER BY total DESC
LIMIT 5

For numeric VARCHAR columns, clean and safely convert the value before
ordering or aggregating.

# NULL HANDLING

Do not automatically convert NULL to zero.

Use COALESCE only when zero is logically appropriate.

# AVAILABLE SQL FUNCTIONS

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

or any other function not explicitly allowed.

# SQL SAFETY

The generated query must:

- Start with SELECT.
- Query only dataset.
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
        self._system_prompt = (
            system_prompt or DEFAULT_SYSTEM_PROMPT
        )

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

        question = question.strip()

        prompt = self._build_prompt(
            question=question,
            dataset_profile=dataset_profile,
        )

        response = self._llm_client.generate_text(
            prompt=prompt,
            system_prompt=self._system_prompt,
            max_tokens=1024,
            temperature=0.0,
        )

        sql = self._extract_sql(response)

        # Deterministic correction for explicit monetary thresholds.
        #
        # This protects against an LLM returning:
        #
        #     > 1000000
        #
        # when the user actually asked for:
        #
        #     > $5 million
        #
        sql = self._correct_explicit_monetary_threshold(
            question=question,
            sql=sql,
        )

        return sql

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

        profile_json = dataset_profile.model_dump_json(
            indent=2
        )

        return f"""
Dataset information:

{profile_json}

# AUTHORITATIVE SCHEMA

The dataset information above is the source of truth.

The dataset is available as a table named `dataset`.

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

Do NOT:

- replace spaces with underscores
- remove spaces
- change capitalization
- replace Unicode whitespace
- simplify punctuation
- invent columns
- rename columns

# NUMERIC VALUES

If a column is numeric, use it directly.

If a numeric-looking column is VARCHAR/text, use TRY_CAST.

For comma-separated numeric values:

TRY_CAST(
    REPLACE("column", ',', '')
    AS DOUBLE
)

# MONEY

Explicit monetary quantities must be converted mathematically.

Examples:

$5 million -> 5000000
$10 million -> 10000000
$2.5 million -> 2500000
$750 thousand -> 750000
$1 billion -> 1000000000

Never substitute a different threshold.

# AGGREGATION AND GROUPING

When a question asks for an aggregate metric for each entity,
category, or group, aggregate at that level.

For example:

"top 3 artists by average gross"

means:

1. GROUP BY "Artist"
2. Calculate AVG(...) of "Average gross"
3. ORDER BY the artist-level average DESC
4. LIMIT 3

Do NOT simply sort individual rows by "Average gross".

Correct structure:

SELECT
    "Artist",
    AVG(...) AS "Average Gross"
FROM "dataset"
GROUP BY "Artist"
ORDER BY "Average Gross" DESC
LIMIT 3

Similarly:

"artists with average gross greater than $3 million"

requires:

GROUP BY "Artist"
HAVING AVG(...) > 3000000

Use HAVING when filtering an aggregate result.

For "top", "highest", "lowest", or "bottom" questions involving an
aggregate metric, calculate the metric at the requested entity/category
level before sorting and applying LIMIT.

# DATES

If a date/time column is VARCHAR/text:

TRY_CAST("date_column" AS DATE)

or:

TRY_CAST("timestamp_column" AS TIMESTAMP)

For monthly analysis:

DATE_TRUNC(
    'MONTH',
    TRY_CAST("date_column" AS DATE)
)

# RANKING

For top/highest questions:

ORDER BY value DESC
LIMIT N

For bottom/lowest questions:

ORDER BY value ASC
LIMIT N

For top/bottom questions involving an aggregate metric, first GROUP BY the
entity/category and calculate the requested aggregate.

# SQL REQUIREMENTS

- Generate exactly ONE SELECT statement.
- Query only dataset.
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

# USER'S ANALYTICAL QUESTION

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
        """

        if not isinstance(response, str) or not response.strip():
            raise QueryPlanningError(
                "The LLM returned an empty response while planning the query."
            )

        sql = response.strip()

        # Remove Markdown code fences.
        fenced_match = re.fullmatch(
            r"```(?:sql)?\s*(.*?)\s*```",
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if fenced_match:
            sql = fenced_match.group(1).strip()

        # Remove a single trailing semicolon.
        if sql.endswith(";"):
            sql = sql[:-1].rstrip()

        if not sql:
            raise QueryPlanningError(
                "The LLM response did not contain a SQL query."
            )

        # Planner requires SELECT.
        if not re.match(
            r"^\s*SELECT\b",
            sql,
            flags=re.IGNORECASE,
        ):
            raise QueryPlanningError(
                "The LLM response does not contain a SELECT statement."
            )

        # Reject multiple statements.
        if ";" in sql:
            raise QueryPlanningError(
                "The LLM response contains multiple SQL statements."
            )

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
                    "The generated SQL contains a forbidden SQL "
                    f"operation: {keyword}"
                )

        return sql

    # ----------------------------------------------------------------------
    # Explicit monetary threshold correction
    # ----------------------------------------------------------------------

    @staticmethod
    def _correct_explicit_monetary_threshold(
        question: str,
        sql: str,
    ) -> str:
        """
        Correct an explicit monetary threshold from the user's question.

        Example:

            Question:
                Which tours had an average gross greater than $5 million?

            LLM:
                WHERE ... > 1000000

            Corrected:
                WHERE ... > 5000000
        """

        threshold = QueryPlanner._extract_monetary_threshold(
            question
        )

        if threshold is None:
            return sql

        # Find numeric comparisons, keeping enough of the left-hand
        # expression to identify the monetary comparison. A generated
        # query may contain unrelated numeric predicates (for example a
        # year filter) before the monetary predicate.
        comparison_pattern = re.compile(
            r"(?P<expression>[^<>=!\n]+?)"
            r"\s*(?P<operator>>=|<=|<>|!=|>|<|=)"
            r"\s*"
            r"(?P<number>\d+(?:\.\d+)?)"
            r"\b",
            flags=re.IGNORECASE,
        )

        matches = list(comparison_pattern.finditer(sql))

        if not matches:
            return sql

        # Prefer a comparison whose expression contains a monetary concept
        # explicitly mentioned in the question (e.g. ``gross`` in
        # ``average gross greater than $5 million``). This prevents an
        # unrelated predicate such as ``Year > 2015`` from being rewritten.
        monetary_terms = {
            "amount",
            "cost",
            "earnings",
            "fee",
            "gross",
            "income",
            "price",
            "profit",
            "revenue",
            "salary",
            "sales",
            "spend",
            "ticket",
            "value",
        }

        question_words = set(
            re.findall(r"[a-z]+", question.lower())
        )
        relevant_terms = monetary_terms & question_words

        match = None

        if relevant_terms:
            for candidate in matches:
                expression_words = set(
                    re.findall(
                        r"[a-z]+",
                        candidate.group("expression").lower(),
                    )
                )

                if relevant_terms & expression_words:
                    match = candidate
                    break

        # Preserve the previous behavior when the question does not contain
        # a recognizable monetary term: use the first numeric comparison.
        if match is None:
            match = matches[0]
            
        old_number = match.group("number")
        new_number = str(threshold)

        if old_number == new_number:
            return sql

        corrected_sql = (
            sql[:match.start("number")]
            + new_number
            + sql[match.end("number"):]
        )

        return corrected_sql

    # ----------------------------------------------------------------------
    # Monetary quantity parser
    # ----------------------------------------------------------------------

    @staticmethod
    def _extract_monetary_threshold(
        question: str,
    ) -> Optional[int]:
        """
        Extract an explicit monetary quantity from a natural-language question.

        Examples:
            "$5 million" -> 5000000
            "5 million dollars" -> 5000000
            "$2.5 million" -> 2500000
            "$750 thousand" -> 750000
            "$1 billion" -> 1000000000
            "$5M" -> 5000000
            "$750K" -> 750000
            "5 mn" -> 5000000
            "5bn" -> 5000000000
            "5k" -> 5000

        Returns None when no supported monetary quantity is found.
        """

        if not isinstance(question, str):
            return None

        text = question.lower().strip()

        pattern = re.compile(
            r"\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*"
            r"(billion|bn|b|million|mn|m|thousand|k)\b",
            re.IGNORECASE,
        )

        match = pattern.search(text)

        if not match:
            return None

        raw_number = match.group(1).replace(",", "")
        unit = match.group(2).lower()

        try:
            number = float(raw_number)
        except ValueError:
            return None

        multipliers = {
            "billion": 1_000_000_000,
            "bn": 1_000_000_000,
            "b": 1_000_000_000,
            "million": 1_000_000,
            "mn": 1_000_000,
            "m": 1_000_000,
            "thousand": 1_000,
            "k": 1_000,
        }

        multiplier = multipliers.get(unit)

        if multiplier is None:
            return None

        return int(round(number * multiplier))
